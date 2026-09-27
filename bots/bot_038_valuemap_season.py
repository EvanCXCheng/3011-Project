"""bot_038 — family: valuemap — parent: bot_014

Change vs bot_014 (one idea, rule-based, no simulation): season-dependent value maps. SCs change hands only after
Fall moves, so in Fall the SC base values are multiplied by FALL_SC_MULT and diffused over only FALL_DIFF_ITERS
passes (grab or keep SCs now), while Spring keeps the full diffusion (position for the next Fall).
(Valuemap backlog item 4; part of the style-pure sparring field.)

bot_014 notes (parent):

Change vs bot_011 (one idea): strength-aware destination values. In the value assignment (after the 2v1 pre-pass),
a move into an enemy-occupied province keeps only OCC_FACTOR of its value (alone it only bounces off a holder).
(A second term, dividing empty-province values by 1 + COMP * adjacent enemy units, is off by default: COMP=0.25
cut S1 Austria/Italy/Turkey to 7/8/10 SC in sanity games, versus 18/15/18 without it.) Units then gather next to targets,
where they can support each other, instead of jamming against holders (bot_011 S1 Austria: 9 units deadlocked for 15
years, each attacking a static holder or an own unit that could not leave).

bot_011 change vs bot_002: supported attacks. Before the value assignment, each enemy-occupied SC (highest value
first) that one of our units can enter and another can support gets an attacker + supporter pair (2 vs 1). After
the assignment, units left holding off our own SCs support an adjacent move of ours into a contested province.

Parent hypothesis: a province value map beats nearest-SC greedy. Every province gets a base value (neutral SC, enemy SC,
or own SC weighted by the enemy units next to it). The values are diffused over the per-unit-type movement graph
so that units feel distant targets, and units are then greedily assigned their highest-value reachable province
with no two of our units sent to the same province.
Idea credit: value-map strategy in the style of DumbBot (D. Norman); the code here is written from scratch.

Technique tags: value-map, value-diffusion, supported-attacks, strength-aware-values, season-weights
"""
import time
from collections import deque

from agent_baselines import Agent

CONFIG = {
    'TIME_BUDGET': 0.5,      # s per get_actions call
    'W_NEUTRAL': 10.0,       # base value of an SC nobody owns
    'W_ENEMY': 7.0,          # base value of an SC another power owns
    'W_DEFEND': 5.0,         # per adjacent enemy unit, base value of an SC we own
    'DIFF_ITERS': 6,         # diffusion passes
    'DIFF_MAX': 0.6,         # weight of the best neighbour in a pass
    'DIFF_SUM': 0.05,        # weight of the neighbour sum in a pass
    'SUPPORTS': True,        # False = bot_002 behaviour
    'STRENGTH': True,        # False = bot_011 behaviour
    'SEASON': True,          # False = bot_014 behaviour
    'FALL_SC_MULT': 1.5,     # Fall: SC base values x this
    'FALL_DIFF_ITERS': 2,    # Fall: diffusion passes (Spring keeps DIFF_ITERS)
    'OCC_FACTOR': 0.15,      # value kept for an unsupported move into an enemy-occupied province
    'COMP': 0.0,             # per adjacent enemy unit, divisor for moves into empty provinces (0.25 hurt S1: see JOURNAL)
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
    """Adjacency per unit type, one-move reach (as base provinces) and SC distances."""
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
    info = {'adj': adj, 'reach': reach, 'scs': scs, 'scset': set(scs),
            'provinces': sorted({_base(l) for l in locs})}
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

    def __init__(self, agent_name='bot_038_valuemap_season'):
        super().__init__(agent_name)

    def new_game(self, game, power_name):
        self.game = game
        self.power_name = power_name
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
            possible = self.game.get_all_possible_orders()
            locs = self.game.get_orderable_locations(self.power_name)
            values = self._value_map()
            ptype = self.game.phase_type
            if ptype == 'M':
                orders = self._movement(possible, locs, values, t0)
            elif ptype == 'R':
                orders = self._retreats(possible, locs, values)
            elif ptype == 'A':
                orders = self._adjustments(possible, locs, values)
        except Exception:
            pass
        return orders

    # --------------------------------------------------------------------------------------------------------
    def _value_map(self):
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
        for prov in info['provinces']:
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
        fall = CONFIG['SEASON'] and self.game.get_current_phase().startswith('F')
        if fall:
            for prov in base:
                if prov in info['scset']:
                    base[prov] *= CONFIG['FALL_SC_MULT']
        iters = CONFIG['FALL_DIFF_ITERS'] if fall else CONFIG['DIFF_ITERS']
        values = {}
        for t in ('A', 'F'):
            adj = info['adj'][t]
            b = {l: base[_base(l)] for l in adj}
            v = dict(b)
            for _ in range(iters):
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

    def _movement(self, possible, locs, values, t0):
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

    def _retreats(self, possible, locs, values):
        orders, taken = [], set()
        for base in locs:
            best, best_key, disband = None, None, None
            for o in possible.get(base) or []:
                ut, loc, kind, dest = parse_order(o)
                if kind == 'R' and dest is not None and _base(dest) not in taken:
                    key = (-values[ut].get(dest, 0.0), o)
                    if best_key is None or key < best_key:
                        best, best_key = o, key
                elif kind == 'D':
                    disband = o
            if best is not None:
                orders.append(best)
                taken.add(_base(parse_order(best)[3]))
            elif disband is not None:
                orders.append(disband)
        return orders

    def _adjustments(self, possible, locs, values):
        power = self.game.get_power(self.power_name)
        n = len(power.centers) - len(power.units)
        orders = []
        if n > 0:
            cands = []
            for base in locs:
                best, best_key = None, None
                for o in possible.get(base) or []:
                    ut, loc, kind, _ = parse_order(o)
                    if kind != 'B':
                        continue
                    key = (-values[ut].get(loc, 0.0), 0 if ut == 'A' else 1, o)
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
                        cands.append((values[ut].get(loc, 0.0), o))
            cands.sort()
            orders = [o for _, o in cands[:-n]]
        return orders
