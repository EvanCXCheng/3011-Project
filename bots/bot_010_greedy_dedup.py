"""bot_010 — family: greedy — parent: bot_001

Change vs bot_001 (one idea): no self-bounces. Units are assigned destinations jointly, best (unit, destination)
pair first, so no two of our units target the same province. A unit may enter a province one of our units occupies
only if that unit is moving out (and not into the entering unit's province, which would be a swap); otherwise the
entering unit falls back to its next-best option.

Parent hypothesis: moving every unit one step toward its nearest supply centre (SC) that we do not own, using
BFS distances computed separately for armies and fleets (coast-aware), is a working basic agent.
Deliberately minimal: no bounce avoidance, no supports, no convoys (those are later greedy iterations).

Technique tags: bfs-greedy, no-self-bounce
"""
import time
from collections import deque

from agent_baselines import Agent

CONFIG = {
    'TIME_BUDGET': 0.5,      # s per get_actions call
    'DEDUP': True,           # False = bot_001 behaviour
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
    """Adjacency and SC distances per unit type. scdist[t][loc][sc] = moves for a unit of type t at loc to reach sc."""
    m = game.map
    key = m.name
    if key in _MAP_CACHE:
        return _MAP_CACHE[key]
    locs = [l.upper() for l in m.locs]
    ltype = {l: m.loc_type.get(l, m.area_type(l)) for l in locs}
    army_nodes = [l for l in locs if '/' not in l and ltype[l] in ('LAND', 'COAST')]
    # fleets occupy water, coasts without split coasts, and the split coasts themselves
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
    info = {'adj': adj, 'scs': scs, 'scdist': scdist, 'army_nodes': set(army_nodes), 'fleet_nodes': set(fleet_nodes)}
    _MAP_CACHE[key] = info
    return info


def parse_order(order):
    """Return (unit_type, loc, kind, dest) for simple order strings; dest is None where not applicable."""
    tok = order.split()
    if len(tok) < 3:
        return None, None, tok[-1] if tok else None, None
    ut, loc, kind = tok[0], tok[1], tok[2]
    dest = None
    if kind in ('-', 'R') and len(tok) >= 4:
        dest = tok[3]
    return ut, loc, kind, dest


# ----------------------------------------------------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------------------------------------------------
class StudentAgent(Agent):

    def __init__(self, agent_name='bot_010_greedy_dedup'):
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

    def _movement(self, possible, locs, t0):
        if CONFIG['DEDUP']:
            return self._movement_dedup(possible, locs, t0)
        return self._movement_single(possible, locs, t0)

    def _movement_dedup(self, possible, locs, t0):
        targets = self._targets()
        opts = {}                                   # base -> sorted [(key, order, dest_prov)]
        for base in locs:
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
            claim = {}
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
        return [opts[b][choice[b]][1] for b in opts]

    def _movement_single(self, possible, locs, t0):
        targets = self._targets()
        orders = []
        for base in locs:
            opts = possible.get(base) or []
            if not opts:
                continue
            best, best_key = None, None
            for o in opts:
                ut, loc, kind, dest = parse_order(o)
                if kind == 'H':
                    d = self._near(ut, loc, targets)
                elif kind == '-' and dest is not None and not o.endswith('VIA'):
                    d = self._near(ut, dest, targets)
                else:
                    continue
                key = (d, 0 if kind == '-' else 1, o)
                if best_key is None or key < best_key:
                    best, best_key = o, key
            if best is not None:
                orders.append(best)
            if time.perf_counter() - t0 > CONFIG['TIME_BUDGET']:
                break
        return orders

    def _retreats(self, possible, locs):
        targets = self._targets()
        orders = []
        for base in locs:
            opts = possible.get(base) or []
            best, best_key, disband = None, None, None
            for o in opts:
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
