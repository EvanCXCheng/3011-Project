"""bot_043 — family: adaptive (style-pure, no simulation) — parent: bot_005

Change vs bot_005 (one idea): opponents classified by greedy-prediction hit rate (search bot_016's finding) instead
of move direction: >= 0.85 greedy, >= 0.4 strong, else erratic (after the static check). A 'strong' power is treated
worst-case: each of its units threatens every adjacent province (threat 1.0) and probably holds (0.85), so the rules
garrison against it and attack its units only with support. (Part of the style-pure sparring field; the family
champion bot_009 is a lookahead hybrid, so this line keeps the rule-based adaptive style.)

bot_005 notes (parent):

Hypothesis: modelling the opponents pays off. Each opponent is classified from its order history (static /
greedy / erratic / unknown) and its next moves are predicted from its class. A rule bot then
  1. garrisons own SCs that predicted enemy moves hit,
  2. makes 2-vs-1 supported attacks on SCs whose occupant is predicted to hold (the only way to grow against
     static powers once neutral SCs are gone),
  3. routes the other units to free targets while avoiding predicted holders,
  4. has any remaining unit support an adjacent attack of ours, or hold.

Technique tags: opponent-classification, opponent-prediction, prediction-accuracy-gating
"""
import time
from collections import deque

from agent_baselines import Agent

CONFIG = {
    'TIME_BUDGET': 0.5,
    'STATIC_HOLD_FRAC': 0.95,   # classify static when at least this fraction of unit-orders were holds
    'GREEDY_TOWARD_FRAC': 0.8,  # classify greedy when this fraction of its moves reduced its distance to a target
    'MIN_OBS': 2,               # unit-orders needed before classifying
    'ACC_GATE': True,           # False = bot_005 classifier
    'ACC_GREEDY': 0.85, 'ACC_STRONG': 0.4, 'STRONG_HOLD': 0.85,
    'DEF_THRESH': 0.5,          # predicted threat on an own SC that triggers a garrison
    'HOLDER_P': 0.6,            # hold probability above which an enemy unit counts as a holder (needs 2v1)
    'V_NEUTRAL': 10.0,
    'V_STATIC': 9.0,
    'V_ERRATIC': 8.0,
    'V_GREEDY': 6.0,
    'FALL_STAY_BONUS': 5.0,     # score for staying on an unowned SC in Fall
    'SC_STEP_BONUS': 2.0,       # score for stepping onto a free target SC
}

INF = 10 ** 6


# ----------------------------------------------------------------------------------------------------------------
# Map precomputation (shared helper; computed once per map and kept in memory)
# ----------------------------------------------------------------------------------------------------------------
_MAP_CACHE = {}


def _base(loc):
    return loc.split('/')[0]


def _bfs(adj, src):
    dist = {src: 0}
    q = deque([src])
    while q:
        u = q.popleft()
        for v in adj.get(u, ()):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def map_info(game):
    m = game.map
    key = m.name
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]
    locs = [l.upper() for l in m.locs]
    ltype = {l: m.loc_type.get(l, m.area_type(l)) for l in locs}
    army_nodes = [l for l in locs if '/' not in l and ltype[l] in ('LAND', 'COAST')]
    has_coasts = {_base(l) for l in locs if '/' in l}
    fleet_nodes = [l for l in locs if ltype[l] in ('WATER', 'COAST') and l not in has_coasts]
    adj = {'A': {}, 'F': {}}
    for t, nodes in (('A', army_nodes), ('F', fleet_nodes)):
        nodeset = set(nodes)
        for a in nodes:
            nbrs = []
            for b in m.loc_abut.get(a, []):
                b = b.upper()
                if b in nodeset and m.abuts(t, a, '-', b):
                    nbrs.append(b)
            adj[t][a] = nbrs
    scs = [s.upper() for s in m.scs]
    scdist = {'A': {}, 'F': {}}
    for t, nodes in (('A', army_nodes), ('F', fleet_nodes)):
        for a in nodes:
            d = _bfs(adj[t], a)
            row = {}
            for sc in scs:
                best = INF
                for l, dl in d.items():
                    if _base(l) == sc and dl < best:
                        best = dl
                row[sc] = best
            scdist[t][a] = row
    info = {'adj': adj, 'scs': scs, 'scset': set(scs), 'scdist': scdist}
    _MAP_CACHE[key] = info
    return info


def parse_order(order):
    tok = order.split()
    if len(tok) < 3:
        return None, None, tok[-1] if tok else None, None
    ut, loc, kind = tok[0], tok[1], tok[2]
    dest = None
    if kind in ('-', 'R') and len(tok) >= 4:
        dest = tok[3]
    return ut, loc, kind, dest


def _unit_split(u):
    u = u.lstrip('*')
    t, loc = u.split()[:2]
    return t, loc


# ----------------------------------------------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------------------------------------------
class StudentAgent(Agent):

    def __init__(self, agent_name='bot_043_adaptive_hitrate'):
        super().__init__(agent_name)

    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
        self.obs = {}            # power -> [n_unit_orders, n_holds, n_moves, n_toward]
        self.seen_phases = set()
        try:
            self.info = map_info(game)
        except Exception:
            self.info = None

    def update_game(self, all_power_orders):
        for power_name in all_power_orders.keys():
            self.game.set_orders(power_name, all_power_orders[power_name])
        self.game.process()

    # --------------------------------------------------------------------------------------------------------
    def get_actions(self):
        t0 = time.perf_counter()
        orders = []
        try:
            if self.info is None:
                self.info = map_info(self.game)
            try:
                self._observe()
            except Exception:
                pass
            possible = self.game.get_all_possible_orders()
            locs = self.game.get_orderable_locations(self.power_name)
            ptype = self.game.phase_type
            if ptype == 'M':
                orders = self._movement(possible, locs, t0)
            elif ptype == 'R':
                orders = self._retreats(possible, locs)
            elif ptype == 'A':
                orders = self._adjustments(possible, locs)
        except Exception:
            pass
        return orders

    # --------------------------------------------------------------------------------------------------------
    # opponent modelling
    def _near(self, utype, loc, targets):
        row = self.info['scdist'][utype].get(loc)
        if not row or not targets:
            return INF
        return min(row[sc] for sc in targets)

    def _observe(self):
        """Update per-power order statistics from movement phases not yet seen."""
        g = self.game
        for phase in list(g.order_history.keys()):
            ph = str(phase)
            if ph in self.seen_phases or not ph.endswith('M'):
                continue
            self.seen_phases.add(ph)
            st = g.state_history.get(phase)
            if st is None:
                continue
            orders_by_power = g.order_history[phase]
            for p, units in st['units'].items():
                if p == self.power_name:
                    continue
                own = set(st['centers'].get(p, []))
                targets = [sc for sc in self.info['scs'] if sc not in own]
                ordered = {}
                for o in orders_by_power.get(p, []) or []:
                    tok = o.split()
                    if len(tok) >= 3:
                        ordered[_base(tok[1])] = o
                ob = self.obs.setdefault(p, [0, 0, 0, 0, 0, 0])
                for u in units:
                    t, loc = _unit_split(u)
                    ob[0] += 1
                    o = ordered.get(_base(loc))
                    act = _base(loc)
                    if o is not None and o.split()[2] == '-':
                        act = _base(o.split()[3])
                    ob[4] += 1
                    if act in self._greedy_dests(t, loc, targets):
                        ob[5] += 1
                    if o is None or o.split()[2] == 'H':
                        ob[1] += 1
                        continue
                    tok = o.split()
                    if tok[2] == '-':
                        ob[2] += 1
                        dest = tok[3]
                        if self._near(t, dest, targets) < self._near(t, loc, targets):
                            ob[3] += 1

    def _greedy_dests(self, t, loc, targets):
        here = self._near(t, loc, targets)
        nbrs = self.info['adj'][t].get(loc, [])
        if here > 0 and nbrs:
            dists = [(self._near(t, n, targets), n) for n in nbrs]
            best = min(d for d, _ in dists)
            if best < here:
                return {_base(n) for d, n in dists if d == best}
        return {_base(loc)}

    def _classify(self, p):
        ob = self.obs.get(p)
        if not ob or ob[0] < CONFIG['MIN_OBS']:
            return 'unknown'
        if ob[1] >= CONFIG['STATIC_HOLD_FRAC'] * ob[0]:
            return 'static'
        if CONFIG['ACC_GATE'] and len(ob) > 4 and ob[4]:
            acc = ob[5] / ob[4]
            if acc >= CONFIG['ACC_GREEDY']:
                return 'greedy'
            if acc >= CONFIG['ACC_STRONG']:
                return 'strong'
            return 'erratic'
        if ob[2] > 0 and ob[3] >= CONFIG['GREEDY_TOWARD_FRAC'] * ob[2]:
            return 'greedy'
        return 'erratic'

    def _predict(self):
        """threat[prov] = expected enemy units moving in; holdp[prov] = P(enemy unit there stays)."""
        g = self.game
        adj = self.info['adj']
        threat, holdp, occ_owner = {}, {}, {}
        classes = {}
        for p, pw in g.powers.items():
            if p == self.power_name:
                continue
            cls = self._classify(p)
            classes[p] = cls
            own = set(pw.centers)
            targets = [sc for sc in self.info['scs'] if sc not in own]
            for u in pw.units:
                t, loc = _unit_split(u)
                prov = _base(loc)
                occ_owner[prov] = p
                nbrs = adj[t].get(loc, [])
                moves = {}
                if cls == 'static':
                    stay = 1.0
                elif cls == 'strong':
                    stay = CONFIG['STRONG_HOLD']
                    for n in nbrs:
                        moves[_base(n)] = 1.0
                else:
                    # greedy component
                    here = self._near(t, loc, targets)
                    gm = {}
                    if here > 0 and nbrs:
                        dists = [(self._near(t, n, targets), n) for n in nbrs]
                        best = min(d for d, _ in dists)
                        if best < here:
                            bests = [n for d, n in dists if d == best]
                            for n in bests:
                                gm[_base(n)] = gm.get(_base(n), 0.0) + 1.0 / len(bests)
                    g_stay = 1.0 - sum(gm.values())
                    # uniform component
                    um = {}
                    for n in nbrs:
                        um[_base(n)] = um.get(_base(n), 0.0) + 1.0 / (len(nbrs) + 1)
                    u_stay = 1.0 / (len(nbrs) + 1)
                    w = {'greedy': 1.0, 'unknown': 0.5, 'erratic': 0.0}[cls]
                    stay = w * g_stay + (1 - w) * u_stay
                    for k in set(gm) | set(um):
                        moves[k] = w * gm.get(k, 0.0) + (1 - w) * um.get(k, 0.0)
                holdp[prov] = stay
                for k, v in moves.items():
                    threat[k] = threat.get(k, 0.0) + v
        return threat, holdp, occ_owner, classes

    # --------------------------------------------------------------------------------------------------------
    def _movement(self, possible, locs, t0):
        me = self.power_name
        g = self.game
        fall = g.get_current_phase().startswith('F')
        own_scs = set(g.get_power(me).centers)
        threat, holdp, occ_owner, classes = self._predict()
        holder = {p for p, v in holdp.items() if v >= CONFIG['HOLDER_P']}

        # our units and their options
        U = []
        for b in locs:
            opts = possible.get(b) or []
            if not opts:
                continue
            ut, loc, _, _ = parse_order(opts[0])
            u = {'t': ut, 'loc': loc, 'prov': b, 'hold': None, 'moves': {}, 'sup': {}, 'suph': {}}
            for o in opts:
                tok = o.split()
                if len(tok) < 3:
                    continue
                if tok[2] == 'H':
                    u['hold'] = o
                elif tok[2] == '-' and tok[-1] != 'VIA':
                    u['moves'].setdefault(_base(tok[3]), []).append((tok[3], o))
                elif tok[2] == 'S':
                    if len(tok) >= 7 and tok[5] == '-':
                        u['sup'][(_base(tok[4]), _base(tok[6]))] = o
                    else:
                        u['suph'][_base(tok[4])] = o
            if u['hold'] is None:
                u['hold'] = f'{ut} {loc} H'
            U.append(u)
        if not U:
            return []
        my_provs = {u['prov'] for u in U}

        # target SC values
        def sc_value(sc):
            o = None
            for p, pw in g.powers.items():
                if sc in pw.centers:
                    o = p
                    break
            if o is None:
                return CONFIG['V_NEUTRAL']
            if o == me:
                return 0.0
            return {'static': CONFIG['V_STATIC'], 'erratic': CONFIG['V_ERRATIC']}.get(classes.get(o), CONFIG['V_GREEDY'])
        targets = {sc: sc_value(sc) for sc in self.info['scs'] if sc not in own_scs}
        free_targets = [sc for sc in targets if sc not in holder]
        all_targets = list(targets)

        orders = {}      # index -> order
        moving_to = {}   # index -> dest prov
        staying = set()  # provinces where our unit stays this turn

        def best_move_order(u, dprov):
            lst = u['moves'].get(dprov)
            if not lst:
                return None
            tg = free_targets or all_targets
            return min(lst, key=lambda x: (self._near(u['t'], x[0], tg), x[1]))[1]

        # 1. garrison threatened own SCs
        threatened = sorted((sc for sc in own_scs if threat.get(sc, 0.0) >= CONFIG['DEF_THRESH']),
                            key=lambda s: -threat.get(s, 0.0))
        for sc in threatened:
            idx = next((i for i, u in enumerate(U) if u['prov'] == sc), None)
            if idx is not None:
                if idx not in orders:
                    orders[idx] = U[idx]['hold']
                    staying.add(sc)
                continue
            cands = [i for i, u in enumerate(U) if i not in orders and sc in u['moves']
                     and not (u['prov'] in own_scs and threat.get(u['prov'], 0.0) >= CONFIG['DEF_THRESH'])]
            if cands:
                i = min(cands, key=lambda i: (U[i]['prov'] in targets, U[i]['prov']))
                orders[i] = best_move_order(U[i], sc)
                moving_to[i] = sc

        # 2. supported attacks on SCs held by predicted holders
        att_targets = sorted((sc for sc in targets if sc in holder), key=lambda s: -targets[s])
        for sc in att_targets:
            attackers = [i for i, u in enumerate(U) if i not in orders and sc in u['moves']]
            done = False
            for a in sorted(attackers, key=lambda i: U[i]['prov']):
                sups = [j for j, u in enumerate(U) if j != a and j not in orders
                        and (U[a]['prov'], sc) in u['sup']]
                need = 1 + (1 if threat.get(sc, 0.0) >= 1.0 else 0)
                if len(sups) >= need:
                    mo = best_move_order(U[a], sc)
                    if mo is None:
                        continue
                    orders[a] = mo
                    moving_to[a] = sc
                    for j in sups[:need]:
                        orders[j] = U[j]['sup'][(U[a]['prov'], sc)]
                        staying.add(U[j]['prov'])
                    done = True
                    break
            if done:
                continue

        # 3. remaining units: move toward targets (global greedy assignment, one unit per province)
        taken = set(moving_to.values()) | staying
        cand = []
        for i, u in enumerate(U):
            if i in orders:
                continue
            tg = free_targets
            here = self._near(u['t'], u['loc'], tg)
            if here >= INF:
                tg = all_targets
                here = self._near(u['t'], u['loc'], tg)
            stay_bonus = 0.0
            if u['prov'] in targets and u['prov'] not in holder:
                stay_bonus = CONFIG['FALL_STAY_BONUS'] if fall else CONFIG['SC_STEP_BONUS']
            cand.append((-min(here, 30) + stay_bonus, 1, i, None, u['prov']))
            for dprov, lst in u['moves'].items():
                if dprov in holder or dprov in my_provs:
                    continue
                for dloc, o in lst:
                    d = self._near(u['t'], dloc, tg)
                    s = -min(d, 30)
                    if dprov in targets and dprov not in holder:
                        s += CONFIG['SC_STEP_BONUS'] * targets[dprov] / CONFIG['V_NEUTRAL']
                    s -= 0.3 * threat.get(dprov, 0.0)
                    cand.append((s, 0, i, o, dprov))
        cand.sort(key=lambda x: (-x[0], x[1], x[2], x[3] or ''))
        for s, is_hold, i, o, dprov in cand:
            if i in orders or dprov in taken:
                continue
            if is_hold:
                orders[i] = U[i]['hold']
                staying.add(U[i]['prov'])
            else:
                orders[i] = o
                moving_to[i] = dprov
            taken.add(dprov)

        # 4. units left holding: support an adjacent attack of ours into a contested or occupied province
        for i, u in enumerate(U):
            if orders.get(i) != u['hold']:
                continue
            if u['prov'] in own_scs and threat.get(u['prov'], 0.0) >= CONFIG['DEF_THRESH']:
                continue
            if u['prov'] in targets and fall:
                continue
            best, best_s = None, 0.0
            for a, dprov in moving_to.items():
                key = (U[a]['prov'], dprov)
                if key in u['sup']:
                    s = threat.get(dprov, 0.0) + (1.0 if dprov in occ_owner else 0.0) + (0.5 if dprov in targets else 0.0)
                    if s > best_s:
                        best, best_s = u['sup'][key], s
            if best is not None:
                orders[i] = best
        return [orders[i] for i in sorted(orders) if orders[i]]

    # --------------------------------------------------------------------------------------------------------
    def _targets(self):
        own = set(self.game.get_power(self.power_name).centers)
        return [sc for sc in self.info['scs'] if sc not in own]

    def _retreats(self, possible, locs):
        targets = self._targets()
        orders = []
        for base in locs:
            best, best_key, disband = None, None, None
            for o in possible.get(base) or []:
                ut, loc, kind, dest = parse_order(o)
                if kind == 'R' and dest is not None:
                    key = (self._near(ut, dest, targets), o)
                    if best_key is None or key < best_key:
                        best, best_key = o, key
                elif kind == 'D':
                    disband = o
            if best is not None:
                orders.append(best)
            elif disband is not None:
                orders.append(disband)
        return orders

    def _adjustments(self, possible, locs):
        power = self.game.get_power(self.power_name)
        n = len(power.centers) - len(power.units)
        targets = self._targets()
        orders = []
        if n > 0:
            cands = []
            for base in locs:
                best, best_key = None, None
                for o in possible.get(base) or []:
                    ut, loc, kind, _ = parse_order(o)
                    if kind != 'B':
                        continue
                    key = (self._near(ut, loc, targets), 0 if ut == 'A' else 1, o)
                    if best_key is None or key < best_key:
                        best, best_key = o, key
                if best is not None:
                    cands.append((best_key, best))
            cands.sort()
            orders = [o for _, o in cands[:n]]
        elif n < 0:
            cands = []
            for base in locs:
                for o in possible.get(base) or []:
                    ut, loc, kind, _ = parse_order(o)
                    if kind == 'D':
                        cands.append((-self._near(ut, loc, targets), o))
            cands.sort()
            orders = [o for _, o in cands[:-n]]
        return orders
