"""bot_002 — family: valuemap — parent: none

Hypothesis: a province value map beats nearest-SC greedy. Every province gets a base value (neutral SC, enemy SC,
or own SC weighted by the enemy units next to it). The values are diffused over the per-unit-type movement graph
so that units feel distant targets, and units are then greedily assigned their highest-value reachable province
with no two of our units sent to the same province.
Idea credit: value-map strategy in the style of DumbBot (D. Norman); the code here is written from scratch.

Technique tags: value-map, value-diffusion
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

    def __init__(self, agent_name='bot_002_valuemap_base'):
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

    def _movement(self, possible, locs, values, t0):
        cands = []
        for base in locs:
            for o in possible.get(base) or []:
                ut, loc, kind, dest = parse_order(o)
                if kind == 'H':
                    target = loc
                elif kind == '-' and dest is not None and not o.endswith('VIA'):
                    target = dest
                else:
                    continue
                val = values[ut].get(target, 0.0)
                # small preference for holding on ties, deterministic order otherwise
                cands.append((-val, 0 if kind == 'H' else 1, o, base, _base(target)))
        cands.sort()
        done, taken, orders = set(), set(), []
        for _, _, o, base, prov in cands:
            if base in done or prov in taken:
                continue
            done.add(base)
            taken.add(prov)
            orders.append(o)
            if time.perf_counter() - t0 > CONFIG['TIME_BUDGET']:
                break
        return orders

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
