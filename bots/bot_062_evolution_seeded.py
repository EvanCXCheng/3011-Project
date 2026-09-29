"""bot_062 — family: evolution — parent: bot_059

Change vs bot_059 (one idea): seeded population. Besides the greedy plan and its mutations, the initial population
contains the plans of three rule planners (aggressive archetype bot_040, valuemap bot_014-style, greedy no-bounce),
so evolution starts from strong, diverse parents; orders a seed needs that are not in a unit's gene list are added.
Planner code copied from our earlier bots.

bot_059 description:

Hypothesis: a genetic algorithm over whole joint-order plans, with fitness from engine rollouts, is a distinct
search method from our candidate race (hill climbing + fixed candidate pool) and from per-unit bandits. Each unit's
genes are its candidate orders (hold, top greedy moves, supports of own units). The population starts from the greedy
plan and mutations of it; every generation all plans are played against the same fresh opponent samples (class-based
sampling, shared-opponent copies), the better half survives (running mean fitness) and the rest is refilled by uniform
crossover of survivors plus mutation, until the time budget. The best-mean plan is played.
Shared helpers (map precomputation, light copies, opponent classification, rollout score, retreats/builds) are copied
from earlier bots; the search is new.

Technique tags: genetic-algorithm, seeded-population
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
    'SEEDED': True,          # False = bot_059
    'AG_STYLE': 'aggressive', 'AG_TURTLE_RANGE': 2,
    'W_NEUTRAL': 10.0, 'W_ENEMY': 7.0, 'W_DEFEND': 5.0, 'DIFF_ITERS': 6, 'DIFF_MAX': 0.6, 'DIFF_SUM': 0.05,
    'SUPPORTS': True, 'STRENGTH': True, 'OCC_FACTOR': 0.15, 'COMP': 0.0,
    'P_PERTURB': 0.3, 'P_SUPPORT': 0.5, 'N_LA': 1,
    'CONVOYS': False, 'P_CONVOY': 0.0, 'N_CONVOY_CANDS': 0,   # needed by the copied greedy planner
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

    def __init__(self, agent_name='bot_062_evolution_seeded'):
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
        if CONFIG['SEEDED']:
            seeds = []
            for gen_f in (lambda: self._ag_movement(possible, locs),
                          lambda: self._vm_movement(possible, locs, self._vm_value_map(), t0),
                          lambda: self._la_candidates(possible, locs, targets, rng)[0]):
                try:
                    seeds.append(gen_f() or [])
                except Exception:
                    pass
            for plan in seeds:
                by_unit = {' '.join(o.split()[:2]): o for o in plan}
                gene = []
                for i in range(n_units):
                    unit = ' '.join(arms[i][0].split()[:2])
                    o = by_unit.get(unit)
                    if o is None:
                        gene.append(greedy_idx[i])
                        continue
                    if o not in arms[i]:
                        arms[i].append(o)
                        n_arms[i] += 1
                    gene.append(arms[i].index(o))
                pop.setdefault(tuple(gene), [0.0, 0])
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

    # valuemap candidate generator (ported from bot_014)
    def _vm_value_map(self):
        """values[t][loc] for t in 'A','F'."""
        info = self.info
        me = self.power_name
        owner = {}
        enemy_units = []
        for p, power in self.game.powers.items():
            for c in power.centers:
                owner[c] = p
            if p != me:
                for u in power.units:
                    enemy_units.append(_unit_split(u))
        threat = {}
        for t, loc in enemy_units:
            for prov in info['reach'][t].get(loc, ()):
                threat[prov] = threat.get(prov, 0) + 1
        base = {}
        for prov in sorted({_base(l.upper()) for l in self.game.map.locs}):
            v = 0.0
            if prov in info['scset']:
                o = owner.get(prov)
                if o is None:
                    v = CONFIG['W_NEUTRAL']
                elif o != me:
                    v = CONFIG['W_ENEMY']
                else:
                    v = CONFIG['W_DEFEND'] * threat.get(prov, 0)
            base[prov] = v
        values = {}
        for t in ('A', 'F'):
            adj = info['adj'][t]
            b = {l: base[_base(l)] for l in adj}
            v = dict(b)
            for _ in range(CONFIG['DIFF_ITERS']):
                nv = {}
                for l, nbrs in adj.items():
                    if nbrs:
                        vals = [v[n] for n in nbrs]
                        nv[l] = b[l] + CONFIG['DIFF_MAX'] * max(vals) + CONFIG['DIFF_SUM'] * sum(vals)
                    else:
                        nv[l] = b[l]
                v = nv
            values[t] = v
        return values

    def _vm_movement(self, possible, locs, values, t0):
        me = self.power_name
        g = self.game
        enemy_occ, enemy_reach = set(), {}
        for p, pw in g.powers.items():
            if p == me:
                continue
            for u in pw.units:
                t, loc = _unit_split(u)
                enemy_occ.add(_base(loc))
                for prov in self.info['reach'][t].get(loc, ()):
                    enemy_reach[prov] = enemy_reach.get(prov, 0) + 1
        own_scs = set(g.get_power(me).centers)
        # parse our options once
        mv, sup = {}, {}          # base -> {dest_prov: order}; base -> {(src_prov, dest_prov): order}
        for base in locs:
            mv[base], sup[base] = {}, {}
            for o in possible.get(base) or []:
                tok = o.split()
                if len(tok) >= 4 and tok[2] == '-' and tok[-1] != 'VIA':
                    ut = tok[0]
                    d = _base(tok[3])
                    if d not in mv[base] or values[ut].get(tok[3], 0.0) > values[ut].get(mv[base][d].split()[3], 0.0):
                        mv[base][d] = o
                elif len(tok) >= 7 and tok[2] == 'S' and tok[5] == '-':
                    sup[base][(_base(tok[4]), _base(tok[6]))] = o
        fixed, taken, orders = set(), set(), []
        moving_to = {}
        if CONFIG['SUPPORTS']:
            scs = self.info['scset']
            targets = [prov for prov in enemy_occ if prov in scs and prov not in own_scs]

            def tval(prov):
                return max(values['A'].get(prov, 0.0), values['F'].get(prov, 0.0))
            for prov in sorted(targets, key=lambda x: (-tval(x), x)):
                for a in sorted(b for b in locs if b not in fixed and prov in mv[b]):
                    sups = [b for b in locs if b != a and b not in fixed and (a, prov) in sup[b]]
                    if sups:
                        s_ = sups[0]
                        orders.append(mv[a][prov])
                        orders.append(sup[s_][(a, prov)])
                        fixed.update((a, s_))
                        taken.update((prov, s_))
                        moving_to[a] = prov
                        break
        cands = []
        for base in locs:
            if base in fixed:
                continue
            for o in possible.get(base) or []:
                ut, loc, kind, dest = parse_order(o)
                if kind == 'H':
                    target = loc
                elif kind == '-' and dest is not None and not o.endswith('VIA'):
                    target = dest
                else:
                    continue
                val = values[ut].get(target, 0.0)
                if CONFIG['STRENGTH'] and kind == '-':
                    tp = _base(target)
                    if tp in enemy_occ:
                        val *= CONFIG['OCC_FACTOR']
                    else:
                        val /= 1.0 + CONFIG['COMP'] * enemy_reach.get(tp, 0)
                # small preference for holding on ties, deterministic order otherwise
                cands.append((-val, 0 if kind == 'H' else 1, o, base, _base(target)))
        cands.sort()
        done = set(fixed)
        chosen = {}
        for _, _, o, base, prov in cands:
            if base in done or prov in taken:
                continue
            done.add(base)
            taken.add(prov)
            chosen[base] = o
            if o.split()[2] == '-':
                moving_to[base] = prov
            if time.perf_counter() - t0 > CONFIG['TIME_BUDGET']:
                break
        if CONFIG['SUPPORTS']:
            for base, o in list(chosen.items()):
                if o.split()[2] != 'H' or base in own_scs:
                    continue
                best, best_s = None, 0
                for a, prov in moving_to.items():
                    if (a, prov) in sup[base]:
                        s_ = enemy_reach.get(prov, 0) + (2 if prov in enemy_occ else 0)
                        if s_ > best_s:
                            best, best_s = sup[base][(a, prov)], s_
                if best is not None:
                    chosen[base] = best
        return orders + list(chosen.values())


    # aggressive-archetype candidate generator (ported from bot_040)
    def _ag_active_powers(self):
        g = self.game
        act = set()
        for ph in g.order_history.keys():
            for p, olist in g.order_history[ph].items():
                if p != self.power_name and any(' - ' in o for o in olist or []):
                    act.add(p)
        return act

    def _ag_target_priority(self):
        """{sc: priority}; higher first."""
        g = self.game
        me = self.power_name
        style = CONFIG['AG_STYLE']
        owner = {c: p for p, pw in g.powers.items() for c in pw.centers}
        my_units = [u.lstrip('*').split()[:2] for u in g.get_power(me).units]
        pri = {}
        victim = None
        if style == 'opportunist':
            best = None
            for p, pw in g.powers.items():
                if p == me or not pw.centers:
                    continue
                ds = [min((self.info['scdist'][t].get(l, {}).get(c, INF) for t, l in my_units), default=INF)
                      for c in pw.centers]
                sc_ = len(pw.units) + 0.5 * sum(min(d, 15) for d in ds) / len(ds)
                if best is None or sc_ < best:
                    best, victim = sc_, p
        for c in self.info['scs']:
            o = owner.get(c)
            if o == me:
                continue
            if style == 'aggressive':
                pri[c] = 2 if o is not None else 1
            elif style == 'turtle':
                if o is None and min((self._near(t, l, [c]) for t, l in my_units), default=INF) <= CONFIG['AG_TURTLE_RANGE']:
                    pri[c] = 1
            else:
                pri[c] = 2 if o == victim else (1 if o is None else 0)
        return {c: v for c, v in pri.items() if v > 0}

    def _ag_movement(self, possible, locs):
        g = self.game
        me = self.power_name
        style = CONFIG['AG_STYLE']
        adj = self.info['adj']
        own = set(g.get_power(me).centers)
        pri = self._ag_target_priority()
        targets = list(pri)
        active = self._ag_active_powers()
        enemy_occ, enemy_adj = {}, {}
        for p, pw in g.powers.items():
            if p == me:
                continue
            for u in pw.units:
                t, loc = u.lstrip('*').split()[:2]
                enemy_occ[loc.split('/')[0]] = p
                if p in active:
                    for n in adj[t].get(loc, ()):
                        b = n.split('/')[0]
                        enemy_adj[b] = enemy_adj.get(b, 0) + 1
        U = {}
        for b in locs:
            opts = possible.get(b) or []
            if not opts:
                continue
            u = {'hold': None, 'moves': {}, 'sup': {}, 'suph': {}, 't': opts[0].split()[0], 'loc': opts[0].split()[1]}
            for o in opts:
                tok = o.split()
                if len(tok) < 3:
                    continue
                if tok[2] == 'H':
                    u['hold'] = o
                elif tok[2] == '-' and tok[-1] != 'VIA':
                    u['moves'].setdefault(tok[3].split('/')[0], o)
                elif tok[2] == 'S' and len(tok) >= 7 and tok[5] == '-':
                    u['sup'][(tok[4].split('/')[0], tok[6].split('/')[0])] = o
                elif tok[2] == 'S':
                    u['suph'][tok[4].split('/')[0]] = o
            U[b] = u
        orders, taken, moving = {}, set(), {}
        # 1. defence
        if style != 'aggressive':
            thr = sorted((c for c in own if enemy_adj.get(c, 0) > 0), key=lambda c: -enemy_adj[c])
            for c in thr:
                if c in U and c not in orders:
                    orders[c] = U[c]['hold']
                    taken.add(c)
                    if style == 'turtle' and enemy_adj[c] >= 2:
                        for b, u in U.items():
                            if b not in orders and c in u['suph']:
                                orders[b] = u['suph'][c]
                                taken.add(b)
                                break
                elif c not in U and style == 'turtle':
                    for b, u in U.items():
                        if b not in orders and c in u['moves'] and not (b in own and enemy_adj.get(b, 0) > 0):
                            orders[b] = u['moves'][c]
                            taken |= {b, c}
                            moving[b] = c
                            break
        # 2. supported attacks on enemy-occupied targets
        need = {'aggressive': 1, 'opportunist': 1, 'turtle': 2}[style]
        for c in sorted((c for c in pri if c in enemy_occ), key=lambda c: -pri[c]):
            if c in taken:
                continue
            for a in sorted(b for b in U if b not in orders and c in U[b]['moves']):
                sups = [b for b in U if b != a and b not in orders and (a, c) in U[b]['sup']]
                if len(sups) >= need:
                    use = sups if style == 'aggressive' else sups[:need]
                    orders[a] = U[a]['moves'][c]
                    moving[a] = c
                    taken |= {a, c}
                    for b in use:
                        orders[b] = U[b]['sup'][(a, c)]
                        taken.add(b)
                    break
        # 3. step toward targets by priority, then distance; one unit per province
        cand = []
        for b, u in U.items():
            if b in orders:
                continue
            here = self._near(u['t'], u['loc'], targets)
            stay = 0.0
            if b in pri and b not in enemy_occ:
                stay = 3.0 * pri[b]
            cand.append((-(stay - min(here, 30)), 1, b, None, b))
            for dprov, o in u['moves'].items():
                if dprov in U or (dprov in enemy_occ):
                    continue
                d = self._near(u['t'], o.split()[3], targets)
                s = -min(d, 30) + (2.0 * pri[dprov] if dprov in pri else 0.0)
                cand.append((-s, 0, b, o, dprov))
        cand.sort(key=lambda x: (x[0], x[1], x[2], x[3] or ''))
        for _, is_hold, b, o, dprov in cand:
            if b in orders or dprov in taken:
                continue
            if is_hold:
                orders[b] = U[b]['hold']
            else:
                orders[b] = o
                moving[b] = dprov
            taken.add(dprov)
        # 4. idle units support our moves
        for b, u in U.items():
            if orders.get(b) != u['hold'] or (b in own and enemy_adj.get(b, 0) > 0 and style != 'aggressive'):
                continue
            for a, dprov in moving.items():
                if (a, dprov) in u['sup']:
                    orders[b] = u['sup'][(a, dprov)]
                    break
        return [o for o in orders.values() if o]


    # --------------------------------------------------------------------------------------------------------
    def _la_candidates(self, possible, locs, targets, rng):
        """Greedy distance-based joint order plus random perturbations (incl. supports of our own moves)."""
        tset = set(targets)
        units = [b for b in locs if possible.get(b)]
        opts = []
        for b in units:
            out = []
            for o in possible.get(b) or []:
                tok = o.split()
                if len(tok) < 3:
                    continue
                if tok[2] == 'H':
                    at = tok[1]
                elif tok[2] == '-' and tok[-1] != 'VIA':
                    at = tok[3]
                else:
                    continue
                sc_ = -min(self._near(tok[0], at, targets), 20) + (3.0 if _base(at) in tset else 0.0)
                out.append((sc_, o, _base(at)))
            out.sort(key=lambda x: (-x[0], x[1]))
            opts.append(out)
        order = sorted(range(len(units)), key=lambda i: -(opts[i][0][0] if opts[i] else -INF))
        taken, greedy = set(), {}
        for i in order:
            for sc_, o, prov in opts[i]:
                if prov not in taken:
                    taken.add(prov)
                    greedy[i] = o
                    break
        my_provs = set(units)
        sup_opts = []
        for b in units:
            lst = []
            for o in possible.get(b) or []:
                tok = o.split()
                if len(tok) >= 7 and tok[2] == 'S' and tok[5] == '-' and _base(tok[4]) in my_provs:
                    lst.append(o)
            sup_opts.append(lst)
        # convoy options: (army index, VIA order, [(fleet index, convoy order)], score)
        conv = []
        if CONFIG['CONVOYS']:
            idx = {b: i for i, b in enumerate(units)}
            for i, b in enumerate(units):
                for o in possible.get(b) or []:
                    tok = o.split()
                    if len(tok) != 5 or tok[2] != '-' or tok[-1] != 'VIA' or tok[0] != 'A':
                        continue
                    src, dst = tok[1], tok[3]
                    here = self._near('A', src, targets)
                    there = self._near('A', dst, targets)
                    if not (_base(dst) in tset or there < here):
                        continue
                    want = f'A {src} - {dst}'
                    fl = []
                    for j, fb in enumerate(units):
                        if j == i:
                            continue
                        for fo in possible.get(fb) or []:
                            if ' C ' in fo and fo.endswith(want):
                                fl.append((j, fo))
                                break
                    if fl:
                        conv.append((i, o, fl, (3.0 if _base(dst) in tset else 0.0) - min(there, 20)))
            conv.sort(key=lambda x: (-x[3], x[1]))

        def with_convoy(c, cv):
            i, o, fl, _ = cv
            c[i] = o
            for j, fo in fl:
                c[j] = fo

        out = [[greedy[i] for i in sorted(greedy)]]
        seen = {tuple(sorted(out[0]))}
        used_armies = set()
        for cv in conv:
            if len(used_armies) >= CONFIG['N_CONVOY_CANDS']:
                break
            if cv[0] in used_armies:
                continue
            used_armies.add(cv[0])
            c = dict(greedy)
            with_convoy(c, cv)
            lst = [c[i] for i in sorted(c)]
            key = tuple(sorted(lst))
            if key not in seen:
                seen.add(key)
                out.append(lst)
        tries = 0
        while len(out) < CONFIG['N_LA'] and tries < CONFIG['N_LA'] * 5:
            tries += 1
            c = dict(greedy)
            for i in range(len(units)):
                if rng.random() < CONFIG['P_PERTURB']:
                    if sup_opts[i] and rng.random() < CONFIG['P_SUPPORT']:
                        c[i] = rng.choice(sup_opts[i])
                    elif len(opts[i]) > 1:
                        c[i] = opts[i][rng.randrange(min(4, len(opts[i])))][1]
            for i in range(len(units)):
                o = c.get(i)
                if o and ' S ' in o:
                    tok = o.split()
                    j = units.index(_base(tok[4]))
                    mv = f'{tok[3]} {tok[4]} - {tok[6]}'
                    if j in c and c[j] != mv:
                        if mv in (possible.get(units[j]) or []):
                            c[j] = mv
                        else:
                            c[i] = greedy.get(i, c[i])
            if conv and rng.random() < CONFIG['P_CONVOY']:
                with_convoy(c, rng.choice(conv))
            lst = [c[i] for i in sorted(c)]
            key = tuple(sorted(lst))
            if key not in seen:
                seen.add(key)
                out.append(lst)
        return out

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

