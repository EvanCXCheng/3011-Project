"""bot_059 — family: evolution (new family) — parent: none

Hypothesis: a genetic algorithm over whole joint-order plans, with fitness from engine rollouts, is a distinct
search method from our candidate race (hill climbing + fixed candidate pool) and from per-unit bandits. Each unit's
genes are its candidate orders (hold, top greedy moves, supports of own units). The population starts from the greedy
plan and mutations of it; every generation all plans are played against the same fresh opponent samples (class-based
sampling, shared-opponent copies), the better half survives (running mean fitness) and the rest is refilled by uniform
crossover of survivors plus mutation, until the time budget. The best-mean plan is played.
Shared helpers (map precomputation, light copies, opponent classification, rollout score, retreats/builds) are copied
from earlier bots; the search is new.

Technique tags: genetic-algorithm
"""
import copy
import math
import random
import time
import zlib
from collections import deque

from diplomacy import Game

from agent_baselines import Agent

CONFIG = {
    'TIME_BUDGET': 0.45,     # s per movement phase
    'POP': 12,               # population size
    'SAMPLES': 2,            # opponent samples per generation (shared by the whole population)
    'P_MUT': 0.15,           # per-unit mutation probability
    'P_INIT_MUT': 0.3,       # per-unit mutation probability for the initial population
    'MAX_ARMS': 10,          # candidate orders per unit (best by greedy score; supports added on top)
    'MIX': {'greedy': (0.1, 0.0), 'strong': (0.3, 0.2), 'erratic': (0.3, 0.5), 'unknown': (0.4, 0.1)},
    'R_W_SC': 1.0, 'R_W_OCC_SPRING': 0.5, 'R_W_UNIT': 0.6, 'R_W_DIST': 0.05, 'R_W_LOST': 1.0,
    'STATIC_HOLD_FRAC': 0.95, 'GREEDY_TOWARD_FRAC': 0.8, 'MIN_OBS': 2,
    'ACC_GATE': True, 'ACC_GREEDY': 0.85, 'ACC_STRONG': 0.4,
}

INF = 10 ** 6


# ----------------------------------------------------------------------------------------------------------------
# Map precomputation and small helpers (shared helpers, copied from earlier bots)
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


def light_game(game):
    """History-free copy of the current movement position (units, centres, phase)."""
    g = Game(map_name=game.map.name)
    g.set_current_phase(game.get_current_phase())
    for p, pw in game.powers.items():
        g.set_units(p, list(pw.units), reset=True)
        g.set_centers(p, list(pw.centers), reset=True)
    return g


def _unit_split(u):
    u = u.lstrip('*')
    t, loc = u.split()[:2]
    return t, loc


# ----------------------------------------------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------------------------------------------


class StudentAgent(Agent):

    def __init__(self, agent_name='bot_059_evolution_base'):
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
                ob = self.obs.setdefault(p, [0, 0, 0, 0, 0, 0])
                for u in units:
                    t, loc = _unit_split(u)
                    ob[0] += 1
                    o = ordered.get(_base(loc))
                    # greedy-prediction hit
                    pred = self._greedy_dests(t, loc, targets)
                    act = _base(loc)
                    if o is not None and o.split()[2] == '-':
                        act = _base(o.split()[3])
                    ob[4] += 1
                    if act in pred:
                        ob[5] += 1
                    if o is None or o.split()[2] == 'H':
                        ob[1] += 1
                        continue
                    tok = o.split()
                    if tok[2] == '-':
                        ob[2] += 1
                        if self._near(t, tok[3], targets) < self._near(t, loc, targets):
                            ob[3] += 1

    def _greedy_dests(self, t, loc, targets):
        """Provinces our greedy model predicts for a unit: best-distance neighbours, or its own province."""
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
        if CONFIG['ACC_GATE'] and ob[4]:
            acc = ob[5] / ob[4]
            if acc >= CONFIG['ACC_GREEDY']:
                return 'greedy'
            if acc >= CONFIG['ACC_STRONG']:
                return 'strong'
            return 'erratic'
        if ob[2] > 0 and ob[3] >= CONFIG['GREEDY_TOWARD_FRAC'] * ob[2]:
            return 'greedy'
        return 'erratic'


    # --------------------------------------------------------------------------------------------------------
    # genetic algorithm over joint plans
    def _movement(self, possible, locs, t0):
        me = self.power_name
        g0 = self.game
        info = self.info
        rng = random.Random(zlib.crc32((g0.get_current_phase() + me).encode()))
        own_scs = set(g0.get_power(me).centers)
        targets = [sc for sc in info['scs'] if sc not in own_scs]
        tset = set(targets)
        units = [b for b in locs if possible.get(b)]
        if not units:
            return []
        my_provs = set(units)
        arms = []                         # per unit: list of orders
        prior = []                        # per unit: greedy score per arm
        for b in units:
            scored, sups = [], []
            for o in possible.get(b) or []:
                tok = o.split()
                if len(tok) < 3:
                    continue
                if tok[2] == 'H':
                    at = tok[1]
                elif tok[2] == '-' and tok[-1] != 'VIA':
                    at = tok[3]
                elif tok[2] == 'S' and _base(tok[4]) in my_provs:
                    sups.append(o)
                    continue
                else:
                    continue
                sc_ = -min(self._near(tok[0], at, targets), 20) + (3.0 if _base(at) in tset else 0.0)
                scored.append((sc_, o))
            scored.sort(key=lambda x: (-x[0], x[1]))
            scored = scored[:CONFIG['MAX_ARMS']]
            base_sc = scored[-1][0] - 1.0 if scored else 0.0
            lst = [o for _, o in scored] + sups
            arms.append(lst)
            prior.append([s for s, _ in scored] + [base_sc] * len(sups))
        n_units = len(units)
        greedy_idx = [max(range(len(p)), key=lambda i: p[i]) for p in prior]

        # opponent model
        base_game = light_game(g0)
        ctx = {}
        for p, pw in g0.powers.items():
            if p == me or not pw.units:
                continue
            cls = self._classify(p)
            if cls == 'static':
                continue
            ptg = [sc for sc in info['scs'] if sc not in set(pw.centers)]
            us = []
            for b in g0.get_orderable_locations(p):
                opts = possible.get(b) or []
                if not opts:
                    continue
                t, loc = opts[0].split()[:2]
                gd = self._greedy_dests(t, loc, ptg)
                greedy = [o for o in opts if len(o.split()) >= 4 and o.split()[2] == '-' and o.split()[-1] != 'VIA'
                          and _base(o.split()[3]) in gd]
                us.append((opts, greedy))
            ctx[p] = (us, CONFIG['MIX'].get(cls, CONFIG['MIX']['unknown']))

        def sample():
            out = {}
            for p, (us, mix) in ctx.items():
                lst = []
                for opts, greedy in us:
                    r = rng.random()
                    if r < mix[0]:
                        continue
                    if r < mix[0] + mix[1] or not greedy:
                        lst.append(rng.choice(opts))
                    else:
                        lst.append(rng.choice(greedy))
                out[p] = lst
            return out

        fall = g0.get_current_phase().startswith('F')

        def score(g):
            occ = {}
            for p, x in g.powers.items():
                for u in x.units:
                    occ[_base(_unit_split(u)[1])] = p
            my_units = [_unit_split(u) for u in g.get_power(me).units]
            s = CONFIG['R_W_UNIT'] * len(my_units)
            for t, loc in my_units:
                s -= CONFIG['R_W_DIST'] * min(self._near(t, loc, targets), 20)
            if fall:
                owned = set(own_scs)
                for c in info['scs']:
                    o = occ.get(c)
                    if o == me:
                        owned.add(c)
                    elif o is not None:
                        owned.discard(c)
                s += CONFIG['R_W_SC'] * len(owned)
            else:
                s += CONFIG['R_W_OCC_SPRING'] * sum(1 for c in targets if occ.get(c) == me)
                s -= CONFIG['R_W_LOST'] * 0.5 * sum(1 for c in own_scs if occ.get(c) not in (None, me))
            return s

        def rollout(round_base, choice):
            g = copy.deepcopy(round_base)
            g.set_orders(me, [arms[i][choice[i]] for i in range(n_units)])
            g.process()
            return score(g)

        greedy_g = tuple(greedy_idx)
        n_arms = [len(a) for a in arms]

        def mutate(gen, pm):
            return tuple(rng.randrange(n_arms[i]) if rng.random() < pm else gen[i] for i in range(n_units))

        pop = {greedy_g: [0.0, 0]}
        tries = 0
        while len(pop) < CONFIG['POP'] and tries < CONFIG['POP'] * 5:
            tries += 1
            pop.setdefault(mutate(greedy_g, CONFIG['P_INIT_MUT']), [0.0, 0])
        budget = CONFIG['TIME_BUDGET']
        gens = 0
        while time.perf_counter() - t0 < budget:
            done = True
            for _ in range(CONFIG['SAMPLES']):
                opp = sample()
                rb = copy.deepcopy(base_game)
                for p, lst in opp.items():
                    rb.set_orders(p, lst)
                rb.clear_cache()
                for gen, st in pop.items():
                    if time.perf_counter() - t0 > budget:
                        done = False
                        break
                    st[0] += rollout(rb, gen)
                    st[1] += 1
                if not done:
                    break
            if not done:
                break
            gens += 1
            ranked = sorted(pop.items(), key=lambda kv: -(kv[1][0] / kv[1][1]))
            keep = dict(ranked[:max(2, CONFIG['POP'] // 2)])
            parents = list(keep)
            tries = 0
            while len(keep) < CONFIG['POP'] and tries < CONFIG['POP'] * 5:
                tries += 1
                a, b = rng.sample(parents, 2)
                child = tuple(a[i] if rng.random() < 0.5 else b[i] for i in range(n_units))
                keep.setdefault(mutate(child, CONFIG['P_MUT']), [0.0, 0])
            pop = keep
        scored = [(st[0] / st[1], g) for g, st in pop.items() if st[1] > 0]
        best = max(scored)[1] if scored else greedy_g
        return [arms[i][best[i]] for i in range(n_units)]

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

