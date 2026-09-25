"""bot_015 — family: lookahead — parent: bot_007

Change vs bot_007 (one idea): defensive candidates. bot_007 only ever proposes attacks and supports of our own moves,
so the rollouts never get to test a defence. Here perturbations may also pick support-holds of our own units (the
supported unit is then set to hold), and one extra candidate garrisons every own SC that an enemy unit can reach:
the unit on it holds, and neighbours whose greedy move does not step onto an unowned SC support-hold it.
Aimed at S3, where bot_007 loses >=3 SC from its peak in 25% of games (mostly against the strong stand-in).

bot_007 change vs bot_004: opponent orders in the rollouts are sampled from a per-power model learned from the
order history (static / greedy / erratic / unknown, as in the adaptive family's classifier) instead of one fixed mix.
Static powers always hold; greedy powers mostly make their greedy move; erratic ones mostly random.

Parent hypothesis: simulating candidate order sets one move deep beats choosing orders by rules. Each movement phase:
  1. generate K candidate joint orders for us: a greedy candidate (each unit to the neighbour closest to an unowned
     SC, no two units to one province) plus random perturbations (other good moves, supports of our own moves);
  2. sample opponent orders from a simple model (hold / move toward that power's nearest unowned SC / random);
  3. simulate every candidate against the same opponent samples on cheap history-free copies of the position,
     score the outcome (SCs held and occupied, units kept, distance to unowned SCs) and keep the best mean.
Engine note: copying the real game costs ~8 ms late in the game (history), a light copy ~0.3 ms.

Technique tags: one-ply-simulation, opponent-sampling, light-game-copy, opponent-model-sampling, defensive-candidates
"""
import copy
import random
import time
import zlib
from collections import deque

from diplomacy import Game

from agent_baselines import Agent

CONFIG = {
    'TIME_BUDGET': 0.45,     # s per get_actions call (simulation stops here)
    'N_CAND': 10,            # candidate joint orders
    'P_PERTURB': 0.3,        # per-unit probability of deviating from the greedy order in a perturbed candidate
    'P_SUPPORT': 0.5,        # when deviating, chance to pick a support of an own move (if any)
    'OPP_HOLD': 0.4,         # opponent model: hold probability
    'OPP_RANDOM': 0.1,       # opponent model: random legal order probability (else greedy move)
    'OPP_MODEL': True,       # sample opponents by class (False = bot_004's single mix for everyone)
    'DEF_CANDS': True,       # False = bot_007 behaviour
    # (hold, random) probabilities per class; the rest is the greedy move
    'MIX_STATIC': (1.0, 0.0),
    'MIX_GREEDY': (0.1, 0.0),
    'MIX_ERRATIC': (0.3, 0.5),
    'STATIC_HOLD_FRAC': 0.95,
    'GREEDY_TOWARD_FRAC': 0.8,
    'MIN_OBS': 2,
    'W_SC': 1.0,             # per SC we own after the move (occupied unowned SCs count in Fall)
    'W_OCC_SPRING': 0.5,     # per unowned SC occupied after a Spring move
    'W_UNIT': 0.6,           # per unit kept (not dislodged)
    'W_DIST': 0.05,          # per step from each unit to its nearest unowned SC
    'W_LOST': 1.0,           # own SC occupied by an enemy after a Fall move
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


def light_game(game):
    """History-free copy of the current movement position (units, centres, phase)."""
    g = Game(map_name=game.map.name)
    g.set_current_phase(game.get_current_phase())
    for p, pw in game.powers.items():
        g.set_units(p, list(pw.units), reset=True)
        g.set_centers(p, list(pw.centers), reset=True)
    return g


# ----------------------------------------------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------------------------------------------
class StudentAgent(Agent):

    def __init__(self, agent_name='bot_015_lookahead_defcands'):
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
                        if self._near_set(t, tok[3], targets) < self._near_set(t, loc, targets):
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

    def _mix(self, p):
        """(hold, random) probabilities used to sample power p's orders."""
        default = (CONFIG['OPP_HOLD'], CONFIG['OPP_RANDOM'])
        if not CONFIG['OPP_MODEL']:
            return default
        return {'static': CONFIG['MIX_STATIC'], 'greedy': CONFIG['MIX_GREEDY'],
                'erratic': CONFIG['MIX_ERRATIC']}.get(self._classify(p), default)

    # --------------------------------------------------------------------------------------------------------
    def _near_set(self, utype, loc, targets):
        row = self.info['scdist'][utype].get(loc)
        if not row or not targets:
            return INF
        return min(row[sc] for sc in targets)

    def _targets(self, power_name=None):
        own = set(self.game.get_power(power_name or self.power_name).centers)
        return [sc for sc in self.info['scs'] if sc not in own]

    def _unit_options(self, possible, base, targets, target_set):
        """Scored move/hold options for one unit, best first: list of (score, order, dest_prov)."""
        out = []
        for o in possible.get(base) or []:
            ut, loc, kind, dest = parse_order(o)
            if kind == 'H':
                at = loc
            elif kind == '-' and dest is not None and not o.endswith('VIA'):
                at = dest
            else:
                continue
            d = self._near_set(ut, at, targets)
            s = -min(d, 20) + (3.0 if _base(at) in target_set else 0.0)
            out.append((s, o, _base(at)))
        out.sort(key=lambda x: (-x[0], x[1]))
        return out

    def _greedy_joint(self, units, opts):
        """Each unit its best option with no two units in one province (units in order of best score)."""
        order = sorted(range(len(units)), key=lambda i: -(opts[i][0][0] if opts[i] else -INF))
        taken, chosen = set(), {}
        for i in order:
            for s, o, prov in opts[i]:
                if prov not in taken:
                    taken.add(prov)
                    chosen[i] = o
                    break
        return chosen

    def _movement(self, possible, locs, t0):
        info = self.info
        me = self.power_name
        game = self.game
        budget = CONFIG['TIME_BUDGET']
        rng = random.Random(zlib.crc32((game.get_current_phase() + me).encode()))
        targets = self._targets()
        target_set = set(targets)

        units = [b for b in locs if possible.get(b)]
        if not units:
            return []
        opts = [self._unit_options(possible, b, targets, target_set) for b in units]
        greedy = self._greedy_joint(units, opts)
        fallback = [greedy[i] for i in sorted(greedy)]

        # support options per unit: supports of own units' moves, keyed by (supported unit prov, dest prov)
        my_provs = set(units)
        sup_opts = []
        for b in units:
            lst = []
            for o in possible.get(b) or []:
                tok = o.split()
                if len(tok) >= 7 and tok[2] == 'S' and tok[5] == '-' and _base(tok[4]) in my_provs:
                    lst.append((o, _base(tok[4]), tok[6]))
                elif CONFIG['DEF_CANDS'] and len(tok) == 5 and tok[2] == 'S' and _base(tok[4]) in my_provs:
                    lst.append((o, _base(tok[4]), None))
            sup_opts.append(lst)

        # candidates
        cands = [dict(greedy)]
        seen = {tuple(sorted(greedy.values()))}
        if CONFIG['DEF_CANDS']:
            enemy_adj = {}
            for p, pw in game.powers.items():
                if p == me:
                    continue
                for u in pw.units:
                    t, loc = _unit_split(u)
                    for n in info['adj'][t].get(loc, ()):
                        enemy_adj[_base(n)] = enemy_adj.get(_base(n), 0) + 1
            own_scs = set(game.get_power(me).centers)
            garrison = [i for i, b in enumerate(units) if b in own_scs and enemy_adj.get(b, 0) > 0]
            if garrison:
                d = dict(greedy)
                for i in garrison:
                    hold = next((o for o in possible.get(units[i]) or [] if o.endswith(' H')), None)
                    if hold:
                        d[i] = hold
                gset = {units[i] for i in garrison}
                for i in range(len(units)):
                    if i in garrison:
                        continue
                    g_o = greedy.get(i, '')
                    tok = g_o.split()
                    if len(tok) >= 4 and tok[2] == '-' and _base(tok[3]) in target_set:
                        continue
                    best = None
                    for o, sprov, sdest in sup_opts[i]:
                        if sdest is None and sprov in gset:
                            if best is None or enemy_adj.get(sprov, 0) > enemy_adj.get(best[1], 0):
                                best = (o, sprov)
                    if best:
                        d[i] = best[0]
                key = tuple(sorted(d.values()))
                if key not in seen:
                    seen.add(key)
                    cands.append(d)
        tries = 0
        while len(cands) < CONFIG['N_CAND'] and tries < CONFIG['N_CAND'] * 5:
            tries += 1
            c = dict(greedy)
            for i in range(len(units)):
                if rng.random() < CONFIG['P_PERTURB']:
                    if sup_opts[i] and rng.random() < CONFIG['P_SUPPORT']:
                        c[i] = rng.choice(sup_opts[i])[0]
                    elif len(opts[i]) > 1:
                        c[i] = opts[i][rng.randrange(min(4, len(opts[i])))][1]
            # make supports consistent: supported unit must be ordered to that move
            for i in range(len(units)):
                o = c.get(i)
                if o and ' S ' in o:
                    tok = o.split()
                    j = units.index(_base(tok[4]))
                    if len(tok) == 5:
                        hold = f'{tok[3]} {tok[4]} H'
                        if j in c and c[j] != hold and ' S ' not in c[j]:
                            if hold in (possible.get(units[j]) or []):
                                c[j] = hold
                            else:
                                c[i] = greedy.get(i, c[i])
                        continue
                    mv = f'{tok[3]} {tok[4]} - {tok[6]}'
                    if j in c and c[j] != mv:
                        if mv in (possible.get(units[j]) or []):
                            c[j] = mv
                        else:
                            c[i] = greedy.get(i, c[i])
            key = tuple(sorted(c.values()))
            if key in seen:
                continue
            seen.add(key)
            cands.append(c)
        if time.perf_counter() - t0 > budget * 0.5:
            return fallback

        base_game = light_game(game)
        opp_powers = [p for p in game.powers if p != me and game.get_power(p).units]
        opp_ctx = {}
        for p in opp_powers:
            ptargets = self._targets(p)
            pset = set(ptargets)
            plocs = [b for b in game.get_orderable_locations(p) if possible.get(b)]
            mix = self._mix(p)
            if mix[0] >= 1.0:
                continue            # always holds: no orders needed
            popts = [self._unit_options(possible, b, ptargets, pset) for b in plocs]
            opp_ctx[p] = (plocs, popts, mix)

        def sample_opponents():
            out = {}
            for p, (plocs, popts, mix) in opp_ctx.items():
                lst = []
                for b, po in zip(plocs, popts):
                    r = rng.random()
                    if r < mix[0] or not po:
                        continue
                    if r < mix[0] + mix[1]:
                        lst.append(rng.choice(possible[b]))
                    else:
                        best = po[0][0]
                        top = [x for x in po if x[0] >= best - 1e-9]
                        lst.append(rng.choice(top)[1])
                out[p] = lst
            return out

        fall = game.get_current_phase().startswith('F')
        own_before = set(game.get_power(me).centers)

        def score(g):
            pw = g.get_power(me)
            occ = {}
            for p, x in g.powers.items():
                for u in x.units:
                    occ[_base(_unit_split(u)[1])] = p
            my_units = [_unit_split(u) for u in pw.units]
            s = CONFIG['W_UNIT'] * len(my_units)
            tg = [sc for sc in info['scs'] if sc not in own_before]
            for t, loc in my_units:
                s -= CONFIG['W_DIST'] * min(self._near_set(t, loc, tg), 20)
            if fall:
                owned = set(own_before)
                for sc in info['scset']:
                    o = occ.get(sc)
                    if o == me:
                        owned.add(sc)
                    elif o is not None and sc in owned:
                        owned.discard(sc)
                s += CONFIG['W_SC'] * len(owned)
            else:
                s += CONFIG['W_OCC_SPRING'] * sum(1 for sc in tg if occ.get(sc) == me)
                s -= CONFIG['W_LOST'] * 0.5 * sum(1 for sc in own_before if occ.get(sc) not in (None, me))
            return s

        totals = [0.0] * len(cands)
        n_eval = 0
        cand_orders = [[c[i] for i in sorted(c)] for c in cands]
        while time.perf_counter() - t0 < budget:
            opp = sample_opponents()
            round_scores = []
            for co in cand_orders:
                if time.perf_counter() - t0 > budget:
                    break
                g = copy.deepcopy(base_game)
                for p, lst in opp.items():
                    g.set_orders(p, lst)
                g.set_orders(me, co)
                g.process()
                round_scores.append(score(g))
            if len(round_scores) < len(cand_orders):
                break   # partial round: discard so every candidate has the same samples
            for k, v in enumerate(round_scores):
                totals[k] += v
            n_eval += 1
        if n_eval == 0:
            return fallback
        best = max(range(len(cands)), key=lambda k: totals[k])
        return cand_orders[best]

    # --------------------------------------------------------------------------------------------------------
    def _retreats(self, possible, locs):
        targets = self._targets()
        orders = []
        for base in locs:
            best, best_key, disband = None, None, None
            for o in possible.get(base) or []:
                ut, loc, kind, dest = parse_order(o)
                if kind == 'R' and dest is not None:
                    key = (self._near_set(ut, dest, targets), o)
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
                    key = (self._near_set(ut, loc, targets), 0 if ut == 'A' else 1, o)
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
                        cands.append((-self._near_set(ut, loc, targets), o))
            cands.sort()
            orders = [o for _, o in cands[:-n]]
        return orders
