"""bot_008 — family: search — parent: bot_003

Change vs bot_003 (one idea): the evaluation uses predicted enemy behaviour instead of raw adjacency. Each opponent is
classified from its order history (static / greedy / erratic / unknown) and each enemy unit gets a hold probability
and move probabilities. Contest counts become expected enemy entries (threat), moves into occupied provinces use
the occupant's hold probability, and cut-support and home-SC threat use the same threat map. Static units then no
longer block or threaten anything, which is what stalls bot_003 in Scenario 1.

Parent hypothesis: local search over our joint order set (moves, holds and supports to our own units) scored by a
heuristic evaluation finds coordinated orders (supported attacks, no self-bounces, covered home SCs) that per-unit
greedy rules miss. The evaluation estimates, for each move, a success probability from our attack strength
(1 + valid supports), whether the destination is occupied, and how many enemy units could contest it. It then
scores expected unit positions (unowned SC captured, distance to the nearest unowned SC) and penalises own SCs left
open to adjacent enemies. Search: coordinate-ascent hill climbing with random restarts until the time budget.

Technique tags: local-search, hill-climbing, heuristic-eval, opponent-aware-eval
"""
import random
import time
import zlib
from collections import deque

from agent_baselines import Agent

CONFIG = {
    'TIME_BUDGET': 0.40,     # s of search per movement phase (whole call stays well under 0.6 s)
    'W_SC': 10.0,            # value of ending a Fall move on an SC we do not own
    'SPRING_SC_FACTOR': 0.4, # the same in Spring (position only; ownership changes in Fall)
    'W_DIST': 1.0,           # per step to the nearest unowned SC
    'W_DEF': 6.0,            # own SC left open to an adjacent enemy (Fall; half in Spring)
    'P_OCC_1': 0.15,         # unsupported move into an enemy-occupied province
    'P_OCC_2': 0.85,         # supported (>=2) move into an enemy-occupied province
    'P_COMP_1': 0.65,        # per contesting enemy unit, unsupported move
    'P_COMP_2': 0.90,        # per contesting enemy unit, supported move
    'CUT_FACTOR': 0.6,       # support value when the supporter is adjacent to an enemy unit
    'OPP_AWARE': True,       # False = bot_003's adjacency-count evaluation
    'P_HOLDER_1': 0.0,       # unsupported move into a province whose occupant stays
    'P_HOLDER_2': 0.95,      # supported (>=2) move into a province whose occupant stays
    'THREAT_MIN': 0.1,       # own SC counts as threatened above this expected number of enemy entries
    'STATIC_HOLD_FRAC': 0.95,
    'GREEDY_TOWARD_FRAC': 0.8,
    'MIN_OBS': 2,
    'RESTARTS': True,
    'MAX_STALE_RESTARTS': 25,  # stop after this many restarts without improvement
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
    reach = {t: {a: {_base(b) for b in adj[t][a]} for a in adj[t]} for t in adj}
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
    info = {'adj': adj, 'reach': reach, 'scs': scs, 'scset': set(scs), 'scdist': scdist}
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

    def __init__(self, agent_name='bot_008_search_oppaware'):
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
    def _targets(self):
        own = set(self.game.get_power(self.power_name).centers)
        return [sc for sc in self.info['scs'] if sc not in own]

    def _near(self, utype, loc, targets):
        row = self.info['scdist'][utype].get(loc)
        if not row or not targets:
            return INF
        return min(row[sc] for sc in targets)

    # --------------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------
    # opponent model
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
                ob = self.obs.setdefault(p, [0, 0, 0, 0])
                for u in units:
                    t, loc = _unit_split(u)
                    ob[0] += 1
                    o = ordered.get(_base(loc))
                    if o is None or o.split()[2] == 'H':
                        ob[1] += 1
                        continue
                    tok = o.split()
                    if tok[2] == '-':
                        ob[2] += 1
                        if self._near(t, tok[3], targets) < self._near(t, loc, targets):
                            ob[3] += 1

    def _classify(self, p):
        ob = self.obs.get(p)
        if not ob or ob[0] < CONFIG['MIN_OBS']:
            return 'unknown'
        if ob[1] >= CONFIG['STATIC_HOLD_FRAC'] * ob[0]:
            return 'static'
        if ob[2] > 0 and ob[3] >= CONFIG['GREEDY_TOWARD_FRAC'] * ob[2]:
            return 'greedy'
        return 'erratic'

    def _predict(self):
        """threat[prov] = expected enemy units moving in; holdp[prov] = P(the enemy unit there stays)."""
        g = self.game
        adj = self.info['adj']
        threat, holdp = {}, {}
        for p, pw in g.powers.items():
            if p == self.power_name:
                continue
            cls = self._classify(p)
            own = set(pw.centers)
            targets = [sc for sc in self.info['scs'] if sc not in own]
            for u in pw.units:
                t, loc = _unit_split(u)
                prov = _base(loc)
                nbrs = adj[t].get(loc, [])
                moves = {}
                if cls == 'static':
                    stay = 1.0
                else:
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
        return threat, holdp

    def _movement(self, possible, locs, t0):
        info = self.info
        me = self.power_name
        game = self.game
        fall = game.get_current_phase().startswith('F')
        own_scs = set(game.get_power(me).centers)
        targets = [sc for sc in info['scs'] if sc not in own_scs]
        target_set = set(targets)

        enemy_reach, enemy_occ = {}, set()
        for p, power in game.powers.items():
            if p == me:
                continue
            for u in power.units:
                t, loc = _unit_split(u)
                enemy_occ.add(_base(loc))
                for prov in info['reach'][t].get(loc, ()):
                    enemy_reach[prov] = enemy_reach.get(prov, 0) + 1
        aware = CONFIG['OPP_AWARE']
        holdp = {}
        if aware:
            enemy_reach, holdp = self._predict()

        # units and their candidate orders
        units = []          # (utype, loc, prov)
        cands = []          # per unit: list of (order, kind, dest_loc, dest_prov, sup_src_prov, sup_dest_prov)
        my_provs = set()
        for base in locs:
            opts = possible.get(base) or []
            if not opts:
                continue
            ut, loc = None, None
            for o in opts:
                ut, loc, _, _ = parse_order(o)
                if ut:
                    break
            if ut is None:
                continue
            units.append((ut, loc, _base(loc)))
            my_provs.add(_base(loc))
        for (ut, loc, prov) in units:
            lst = []
            for o in possible.get(prov) or []:
                tok = o.split()
                if len(tok) < 3:
                    continue
                kind = tok[2]
                if kind == 'H':
                    lst.append((o, 'H', loc, prov, None, None))
                elif kind == '-':
                    if tok[-1] == 'VIA':
                        continue
                    lst.append((o, '-', tok[3], _base(tok[3]), None, None))
                elif kind == 'S':
                    sprov = _base(tok[4])
                    if sprov not in my_provs:
                        continue
                    if len(tok) >= 7 and tok[5] == '-':
                        lst.append((o, 'SM', loc, prov, sprov, _base(tok[6])))
                    else:
                        lst.append((o, 'SH', loc, prov, sprov, None))
            if not lst:
                lst.append((f'{ut} {loc} H', 'H', loc, prov, None, None))
            cands.append(lst)
        n = len(units)
        if n == 0:
            return []

        sc_w = CONFIG['W_SC'] * (1.0 if fall else CONFIG['SPRING_SC_FACTOR'])
        def_w = CONFIG['W_DEF'] * (1.0 if fall else 0.5)
        w_dist = CONFIG['W_DIST']
        near_cache = {}

        def value(ut, loc):
            key = (ut, loc)
            v = near_cache.get(key)
            if v is None:
                d = self._near(ut, loc, targets)
                v = (sc_w if _base(loc) in target_set else 0.0) - w_dist * min(d, 20)
                near_cache[key] = v
            return v

        if aware:
            cut = {prov: 1.0 - (1.0 - CONFIG['CUT_FACTOR']) * min(1.0, enemy_reach.get(prov, 0.0))
                   for (_, _, prov) in units}
            threatened_own = [sc for sc in own_scs if enemy_reach.get(sc, 0.0) > CONFIG['THREAT_MIN']]
        else:
            cut = {prov: (CONFIG['CUT_FACTOR'] if enemy_reach.get(prov, 0) > 0 else 1.0) for (_, _, prov) in units}
            threatened_own = [sc for sc in own_scs if enemy_reach.get(sc, 0) > 0]
        p_h1, p_h2 = CONFIG['P_HOLDER_1'], CONFIG['P_HOLDER_2']
        p_occ1, p_occ2 = CONFIG['P_OCC_1'], CONFIG['P_OCC_2']
        p_c1, p_c2 = CONFIG['P_COMP_1'], CONFIG['P_COMP_2']

        def evaluate(assign):
            chosen = [cands[i][assign[i]] for i in range(n)]
            moving = {}
            dest_count = {}
            sup_move, sup_hold = {}, {}
            for i, c in enumerate(chosen):
                kind = c[1]
                if kind == '-':
                    moving[units[i][2]] = c[3]
                    dest_count[c[3]] = dest_count.get(c[3], 0) + 1
            for i, c in enumerate(chosen):
                kind = c[1]
                if kind == 'SM':
                    if moving.get(c[4]) == c[5]:
                        k = (c[4], c[5])
                        sup_move[k] = sup_move.get(k, 0.0) + cut[units[i][2]]
                elif kind == 'SH':
                    if c[4] not in moving:
                        sup_hold[c[4]] = sup_hold.get(c[4], 0.0) + cut[units[i][2]]
            score = 0.0
            p_success = {}
            for i, c in enumerate(chosen):
                ut, loc, prov = units[i]
                if c[1] != '-':
                    score += value(ut, loc)
                    continue
                dprov = c[3]
                s = 1.0 + sup_move.get((prov, dprov), 0.0)
                if dest_count[dprov] > 1:
                    p = 0.0
                elif dprov in my_provs and (dprov not in moving or moving[dprov] == prov):
                    p = 0.0
                else:
                    strong = s >= 1.99
                    p = 1.0
                    if dprov in enemy_occ:
                        if aware:
                            hp = holdp.get(dprov, 1.0)
                            p = hp * (p_h2 if strong else p_h1) + (1.0 - hp)
                        else:
                            p = p_occ2 if strong else p_occ1
                    k = enemy_reach.get(dprov, 0)
                    if k:
                        p *= (p_c2 if strong else p_c1) ** k
                p_success[prov] = p
                score += p * value(ut, c[2]) + (1.0 - p) * value(ut, loc)
            # own SCs open to adjacent enemies
            for sc in threatened_own:
                q = 0.0
                if sc in my_provs:
                    q = 1.0 - p_success.get(sc, 0.0) if sc in moving else 1.0
                for src, dst in moving.items():
                    if dst == sc:
                        q = max(q, p_success.get(src, 0.0))
                if q < 1.0:
                    score -= def_w * (1.0 - q) * min(1.0, (1.0 if aware else 0.5) * enemy_reach[sc])
            return score

        # initial: every unit's best order in isolation (others hold)
        rng = random.Random(zlib.crc32((game.get_current_phase() + me).encode()))
        hold_idx = []
        for lst in cands:
            hi = 0
            for j, c in enumerate(lst):
                if c[1] == 'H':
                    hi = j
                    break
            hold_idx.append(hi)
        budget = CONFIG['TIME_BUDGET']

        def climb(assign, cur):
            improved = True
            while improved:
                improved = False
                order = list(range(n))
                rng.shuffle(order)
                for i in order:
                    if time.perf_counter() - t0 > budget:
                        return assign, cur
                    best_j, best_v = assign[i], cur
                    for j in range(len(cands[i])):
                        if j == assign[i]:
                            continue
                        assign[i] = j
                        v = evaluate(assign)
                        if v > best_v + 1e-9:
                            best_j, best_v = j, v
                    assign[i] = best_j
                    if best_v > cur + 1e-9:
                        cur = best_v
                        improved = True
            return assign, cur

        start = list(hold_idx)
        best_assign, best_val = climb(start, evaluate(start))
        best_assign = list(best_assign)
        stale = 0
        while CONFIG['RESTARTS'] and time.perf_counter() - t0 < budget and stale < CONFIG['MAX_STALE_RESTARTS']:
            stale += 1
            a = list(best_assign)
            for i in rng.sample(range(n), max(1, n // 3)):
                a[i] = rng.randrange(len(cands[i]))
            a, v = climb(a, evaluate(a))
            if v > best_val + 1e-9:
                best_assign, best_val = list(a), v
                stale = 0
        return [cands[i][best_assign[i]][0] for i in range(n)]

    # --------------------------------------------------------------------------------------------------------
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
