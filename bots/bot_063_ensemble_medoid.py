"""bot_063 — family: ensemble — parent: bot_061

Change vs bot_061 (one idea): plan-level consensus. bot_061 mixed orders unit by unit, which on 2–2 ties breaks the
coordination inside each plan (−1.22 vs bot_060). Now the whole proposal that agrees most with the other proposals
(the medoid: most identical unit orders summed over the others) is played unchanged; ties by planner order
(aggressive > valuemap > greedy-matching > greedy). Each played plan stays internally coherent.

bot_061 notes:

Change vs bot_060 (one idea): a fourth voter, the greedy family champion's plan (bot_012: no self-bounces and 2v1
attacks assigned by the Hungarian method), which coordinates supports explicitly. Ties: aggressive > valuemap >
greedy-matching > greedy.

bot_060 description:

Hypothesis: 'wisdom of crowds' over our rule planners. Three fast rule planners each propose a full plan — the
aggressive archetype (bot_040), the valuemap generator (bot_014 style) and the greedy distance plan (no self-bounces)
— and every unit plays the order most planners chose (ties: aggressive > valuemap > greedy); supports whose supported
move lost the vote fall back to that unit's first non-support proposal, else hold. No simulation: a few ms per move.
The planners' code is copied from our earlier bots (the rollout machinery in the file is unused).

Technique tags: ensemble-voting, supported-attack-matching, plan-level-consensus
"""
import copy

import numpy as np
from scipy.optimize import linear_sum_assignment
import random
import time
import zlib
from collections import deque

from diplomacy import Game

from agent_baselines import Agent

CONFIG = {
    'TIME_BUDGET': 0.45,     # s per movement phase: search + rollouts (whole call stays well under 0.6 s)
    'ROLLOUT': True,         # False = bot_016 behaviour (search only, 0.40 s)
    'SEARCH_BUDGET': 0.15,   # s of hill climbing when ROLLOUT is on (bot_020: 0.20)
    'TOP_K': 8,              # local optima entered into the race (bot_020: 6)
    'LA_CANDS': True,        # add lookahead-style candidates (False + HALVING False ~ bot_020)
    'N_LA': 12,              # lookahead-style candidates (greedy + perturbations)
    'P_PERTURB': 0.3, 'P_SUPPORT': 0.5,
    'HALVING': True, 'HALVE_EVERY': 3, 'MIN_ALIVE': 3,
    'TWO_PLY': True,         # False = no two-ply Spring rollouts
    'CONVOYS': True,         # False = no convoy candidates
    'VM_CANDS': True,        # False = bot_029 (no valuemap candidate)
    'GM_ATTACK_MATCH': True,
    'AG_CANDS': True,        # False = bot_045
    'AG_STYLE': 'aggressive', 'AG_TURTLE_RANGE': 2,
    'W_NEUTRAL': 10.0, 'W_ENEMY': 7.0, 'W_DEFEND': 5.0, 'DIFF_ITERS': 6, 'DIFF_MAX': 0.6, 'DIFF_SUM': 0.05,
    'SUPPORTS': True, 'STRENGTH': True, 'OCC_FACTOR': 0.15, 'COMP': 0.0,
    'P_CONVOY': 0.4,         # chance a perturbed candidate also carries one random convoy
    'N_CONVOY_CANDS': 3,     # dedicated candidates: greedy + the best convoy of one army
    # opponent sampling mixes (hold, random); the rest is the greedy move
    'MIX': {'greedy': (0.1, 0.0), 'strong': (0.3, 0.2), 'erratic': (0.3, 0.5), 'unknown': (0.4, 0.1)},
    'R_W_SC': 1.0, 'R_W_OCC_SPRING': 0.5, 'R_W_UNIT': 0.6, 'R_W_DIST': 0.05, 'R_W_LOST': 1.0,
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
    'ACC_GATE': True,        # False = bot_008 behaviour
    'ACC_GREEDY': 0.85,      # greedy-prediction hit rate at or above which a power is 'greedy'
    'ACC_STRONG': 0.4,       # hit rate at or above which (and below ACC_GREEDY) a power is 'strong'
    'STRONG_HOLD': 0.85,     # hold probability assumed for units of a 'strong' power
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

    def __init__(self, agent_name='bot_063_ensemble_medoid'):
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
                elif cls == 'strong':
                    stay = CONFIG['STRONG_HOLD']
                    for n in nbrs:
                        moves[_base(n)] = 1.0
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
        """Per-unit majority vote of three rule planners (aggressive, valuemap, greedy); no simulation."""
        me = self.power_name
        info = self.info
        own_scs = set(self.game.get_power(me).centers)
        targets = [sc for sc in info['scs'] if sc not in own_scs]
        rng = random.Random(zlib.crc32((self.game.get_current_phase() + me).encode()))
        plans = []
        for gen in (lambda: self._ag_movement(possible, locs),
                    lambda: self._vm_movement(possible, locs, self._vm_value_map(), t0),
                    lambda: self._gm_movement(possible, locs, t0),
                    lambda: self._la_candidates(possible, locs, targets, rng)[0]):
            try:
                plans.append(gen() or [])
            except Exception:
                plans.append([])
        by_unit = []
        for pl in plans:
            by_unit.append({' '.join(o.split()[:2]): o for o in pl})
        live = [k for k, d in enumerate(by_unit) if d]
        if live:
            def agree(k):
                return sum(sum(1 for u, o in by_unit[k].items() if by_unit[j].get(u) == o) for j in live if j != k)
            best = max(live, key=lambda k: (agree(k), -k))
            return list(by_unit[best].values())
        units = sorted({u for d in by_unit for u in d})
        chosen = {}
        for u in units:
            votes = {}
            for rank, d in enumerate(by_unit):
                o = d.get(u)
                if o:
                    v = votes.setdefault(o, [0, rank])
                    v[0] += 1
            if votes:
                chosen[u] = max(votes.items(), key=lambda kv: (kv[1][0], -kv[1][1]))[0]
        for u, o in list(chosen.items()):
            tok = o.split()
            if len(tok) >= 7 and tok[2] == 'S' and tok[5] == '-':
                mt = chosen.get(' '.join(tok[3:5]), '').split()
                if not (len(mt) >= 4 and mt[2] == '-' and _base(mt[3]) == _base(tok[6])):
                    alt = next((d.get(u) for d in by_unit if d.get(u) and ' S ' not in d.get(u)), None)
                    chosen[u] = alt or f'{u} H'
        return list(chosen.values())

    # --------------------------------------------------------------------------------------------------------
    def _rollout_select(self, possible, cand_orders, t0, targets):
        """Index of the candidate with the best mean one-move simulated outcome (0 if nothing was simulated)."""
        g0 = self.game
        me = self.power_name
        info = self.info
        rng = random.Random(zlib.crc32((g0.get_current_phase() + me + 'R').encode()))
        base_game = light_game(g0)
        ctx = {}
        for p, pw in g0.powers.items():
            if p == me or not pw.units:
                continue
            cls = self._classify(p)
            if cls == 'static':
                continue
            own = set(pw.centers)
            ptg = [sc for sc in info['scs'] if sc not in own]
            units = []
            for b in g0.get_orderable_locations(p):
                opts = possible.get(b) or []
                if not opts:
                    continue
                t, loc = opts[0].split()[:2]
                gd = self._greedy_dests(t, loc, ptg)
                greedy = [o for o in opts if len(o.split()) >= 4 and o.split()[2] == '-' and o.split()[-1] != 'VIA'
                          and _base(o.split()[3]) in gd]
                units.append((opts, greedy))
            ctx[p] = (units, CONFIG['MIX'].get(cls, CONFIG['MIX']['unknown']))
        fall = g0.get_current_phase().startswith('F')
        own_before = set(g0.get_power(me).centers)
        tg = targets
        two = CONFIG['TWO_PLY'] and not fall
        ply2 = {}
        if two:
            for p, pw in g0.powers.items():
                if p != me and self._classify(p) == 'static':
                    continue
                own = set(pw.centers)
                ply2[p] = [sc for sc in info['scs'] if sc not in own]
        adj = info['adj']

        def fall_reply(g):
            """Advance a simulated Spring result through a greedy Fall move of every non-static power."""
            if g.phase_type == 'R':
                g.process()                       # no retreat orders: dislodged units disband
            if g.phase_type != 'M':
                return
            for p, ptg in ply2.items():
                pw = g.powers.get(p)
                if pw is None or not pw.units:
                    continue
                lst = []
                for u in pw.units:
                    t, loc = _unit_split(u)
                    here = self._near(t, loc, ptg)
                    best, bd = None, here
                    for n in adj[t].get(loc, ()):
                        d = self._near(t, n, ptg)
                        if d < bd:
                            best, bd = n, d
                    if best is not None:
                        lst.append(f'{t} {loc} - {best}')
                if lst:
                    g.set_orders(p, lst)
            g.process()

        def sample():
            out = {}
            for p, (units, mix) in ctx.items():
                lst = []
                for opts, greedy in units:
                    r = rng.random()
                    if r < mix[0]:
                        continue
                    if r < mix[0] + mix[1] or not greedy:
                        lst.append(rng.choice(opts))
                    else:
                        lst.append(rng.choice(greedy))
                out[p] = lst
            return out

        def score(g):
            occ = {}
            for p, x in g.powers.items():
                for u in x.units:
                    occ[_base(_unit_split(u)[1])] = p
            my_units = [_unit_split(u) for u in g.get_power(me).units]
            sc_ = CONFIG['R_W_UNIT'] * len(my_units)
            for t, loc in my_units:
                sc_ -= CONFIG['R_W_DIST'] * min(self._near(t, loc, tg), 20)
            if fall or two:
                owned = set(own_before)
                for c in info['scs']:
                    o = occ.get(c)
                    if o == me:
                        owned.add(c)
                    elif o is not None:
                        owned.discard(c)
                sc_ += CONFIG['R_W_SC'] * len(owned)
            else:
                sc_ += CONFIG['R_W_OCC_SPRING'] * sum(1 for c in tg if occ.get(c) == me)
                sc_ -= CONFIG['R_W_LOST'] * 0.5 * sum(1 for c in own_before if occ.get(c) not in (None, me))
            return sc_

        totals = [0.0] * len(cand_orders)
        n_eval = 0
        alive = list(range(len(cand_orders)))
        since = 0
        while time.perf_counter() - t0 < CONFIG['TIME_BUDGET']:
            opp = sample()
            rs = []
            for k in alive:
                if time.perf_counter() - t0 > CONFIG['TIME_BUDGET']:
                    break
                g = copy.deepcopy(base_game)
                for p, lst in opp.items():
                    g.set_orders(p, lst)
                g.set_orders(me, cand_orders[k])
                g.process()
                if two:
                    fall_reply(g)
                rs.append(score(g))
            if len(rs) < len(alive):
                break
            for k, v in zip(alive, rs):
                totals[k] += v
            n_eval += 1
            since += 1
            if CONFIG['HALVING'] and since >= CONFIG['HALVE_EVERY'] and len(alive) > CONFIG['MIN_ALIVE']:
                alive.sort(key=lambda k: (-totals[k], k))
                alive = alive[:max(CONFIG['MIN_ALIVE'], (len(alive) + 1) // 2)]
                since = 0
        if n_eval == 0:
            return 0
        return max(alive, key=lambda k: (totals[k], -k))

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
    # greedy attack-matching planner (ported from bot_012)
    def _gm_attack_pairs(self, possible, locs):
        """Match units to attacker/supporter roles on enemy-occupied SCs. Returns ({base: order}, blocked provinces)."""
        me = self.power_name
        g = self.game
        own = set(g.get_power(me).centers)
        occ = set()
        for p, pw in g.powers.items():
            if p != me:
                for u in pw.units:
                    occ.add(u.lstrip('*').split()[1].split('/')[0])
        scs = set(self.info['scs'])
        targets = sorted(t for t in occ if t in scs and t not in own)
        if not targets:
            return {}, set()
        mv, sup = {}, {}
        for b in locs:
            mv[b], sup[b] = {}, {}
            for o in possible.get(b) or []:
                tok = o.split()
                if len(tok) >= 4 and tok[2] == '-' and tok[-1] != 'VIA':
                    mv[b].setdefault(tok[3].split('/')[0], o)
                elif len(tok) >= 7 and tok[2] == 'S' and tok[5] == '-':
                    sup[b].setdefault(tok[6].split('/')[0], []).append((tok[4].split('/')[0], o))
        units = [b for b in locs if mv[b] or sup[b]]
        if not units:
            return {}, set()
        slots = [(t, r) for t in targets for r in ('att', 'sup')]
        big = 1e6
        cost = np.full((len(units), len(slots)), big)
        for i, b in enumerate(units):
            for j, (t, r) in enumerate(slots):
                if r == 'att' and t in mv[b]:
                    cost[i, j] = -10.0
                elif r == 'sup' and t in sup[b]:
                    cost[i, j] = -10.0
        rows, cols = linear_sum_assignment(cost)
        role = {}
        for i, j in zip(rows, cols):
            if cost[i, j] < big:
                role.setdefault(slots[j][0], {})[slots[j][1]] = units[i]
        fixed, blocked = {}, set()
        for t, r in role.items():
            a, s_ = r.get('att'), r.get('sup')
            if a is None or s_ is None:
                continue
            so = next((o for src, o in sup[s_][t] if src == a), None)
            if so is None:
                continue                 # supporter cannot support this particular attacker
            fixed[a] = mv[a][t]
            fixed[s_] = so
            blocked.update((t, a, s_))
        return fixed, blocked

    def _gm_movement(self, possible, locs, t0):
        targets = self._targets()
        fixed, blocked = {}, set()
        if CONFIG['GM_ATTACK_MATCH']:
            try:
                fixed, blocked = self._gm_attack_pairs(possible, locs)
            except Exception:
                fixed, blocked = {}, set()
        opts = {}                                   # base -> sorted [(key, order, dest_prov)]
        for base in locs:
            if base in fixed:
                continue
            lst = []
            for o in possible.get(base) or []:
                ut, loc, kind, dest = parse_order(o)
                if kind == 'H':
                    d, prov = self._near(ut, loc, targets), base
                elif kind == '-' and dest is not None and not o.endswith('VIA'):
                    d, prov = self._near(ut, dest, targets), dest.split('/')[0]
                else:
                    continue
                lst.append(((d, 0 if kind == '-' else 1, o), o, prov))
            lst.sort()
            if lst:
                opts[base] = lst
        choice = {b: 0 for b in opts}               # index into opts[b]
        for _ in range(4 * len(opts) + 4):
            if time.perf_counter() - t0 > CONFIG['TIME_BUDGET']:
                break
            # resolve duplicate destinations: the unit with the better key keeps it
            claim = {p: None for p in blocked}
            changed = False
            for b in sorted(opts, key=lambda b: opts[b][choice[b]][0]):
                prov = opts[b][choice[b]][2]
                if prov in claim and choice[b] + 1 < len(opts[b]):
                    choice[b] += 1
                    changed = True
                else:
                    claim.setdefault(prov, b)
            # entering a province held by our unit: only if that unit moves out (and not a swap)
            for b in opts:
                prov = opts[b][choice[b]][2]
                if prov == b or prov not in opts:
                    continue
                occ_dest = opts[prov][choice[prov]][2]
                if (occ_dest == prov or occ_dest == b) and choice[b] + 1 < len(opts[b]):
                    choice[b] += 1
                    changed = True
            if not changed:
                break
        return list(fixed.values()) + [opts[b][choice[b]][1] for b in opts]


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
