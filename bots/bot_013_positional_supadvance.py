"""bot_013 — family: positional — parent: bot_006

Change vs bot_006 (one idea): supported advance. bot_006 lets a move pass the strength gate by counting supporters
that could help, but never orders them, so gated moves into contested provinces often bounce. Here, a move that needs
k supporters to reach the gate (k = adjacent enemy units − 1) is taken only if k free units can be committed to
support it at the same time; their support orders are issued with the move.

Parent hypothesis: slower, safer expansion loses fewer SCs against aggressive neighbours.
  1. Garrison: an own SC with an enemy unit next to it is never left empty (our unit there holds, or a free
     neighbour moves in).
  2. Target power: one power is chosen by weakness (its unit count) plus reachability (mean distance from our units
     to its SCs). Targets are the neutral SCs plus that power's SCs.
  3. Occupied target SCs are attacked only with enough support to beat the occupant plus any enemy unit that could
     support it.
  4. Other moves are strength-gated: a unit only enters a province where our strength (mover + our units able to
     support it) is at least the number of enemy units able to enter it.
  5. Leftover units support contested moves of ours.

Technique tags: target-power-selection, strength-gated-moves, home-garrison, supported-advance
"""
import time
from collections import deque

from agent_baselines import Agent

CONFIG = {
    'TIME_BUDGET': 0.5,
    'W_REACH': 0.5,          # target-power score = units + W_REACH * mean distance to its SCs (lower is better)
    'SC_BONUS': 2.0,         # score for moving onto a target SC
    'FALL_STAY_BONUS': 5.0,  # score for staying on an unowned SC in Fall
    'GATE': True,            # strength gating of moves
    'SUP_ADVANCE': True,     # False = bot_006 behaviour (count potential supporters, do not commit them)
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

    def __init__(self, agent_name='bot_013_positional_supadvance'):
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

    def _near(self, utype, loc, targets):
        row = self.info['scdist'][utype].get(loc)
        if not row or not targets:
            return INF
        return min(row[sc] for sc in targets)

    # --------------------------------------------------------------------------------------------------------
    def _target_set(self):
        """Neutral SCs plus the SCs of the chosen target power; falls back to every SC we do not own."""
        g = self.game
        me = self.power_name
        my_units = [_unit_split(u) for u in g.get_power(me).units]
        owned = {}
        for p, pw in g.powers.items():
            for c in pw.centers:
                owned[c] = p
        neutral = [sc for sc in self.info['scs'] if sc not in owned]
        best_p, best_s = None, None
        for p, pw in g.powers.items():
            if p == me or not pw.centers:
                continue
            ds = []
            for sc in pw.centers:
                d = min((self.info['scdist'][t].get(loc, {}).get(sc, INF) for t, loc in my_units), default=INF)
                ds.append(min(d, 15))
            s = len(pw.units) + CONFIG['W_REACH'] * (sum(ds) / len(ds))
            if best_s is None or s < best_s:
                best_p, best_s = p, s
        targets = set(neutral)
        if best_p is not None:
            targets |= set(g.get_power(best_p).centers)
        if not targets:
            targets = {sc for sc in self.info['scs'] if owned.get(sc) != me}
        return targets, best_p

    def _movement(self, possible, locs, t0):
        me = self.power_name
        g = self.game
        info = self.info
        fall = g.get_current_phase().startswith('F')
        own_scs = set(g.get_power(me).centers)
        all_targets = [sc for sc in info['scs'] if sc not in own_scs]
        targets, _ = self._target_set()
        tlist = list(targets)

        enemy_adj, enemy_occ = {}, {}
        for p, pw in g.powers.items():
            if p == me:
                continue
            for u in pw.units:
                t, loc = _unit_split(u)
                enemy_occ[_base(loc)] = p
                for prov in info['reach'][t].get(loc, ()):
                    enemy_adj[prov] = enemy_adj.get(prov, 0) + 1

        U = []
        for b in locs:
            opts = possible.get(b) or []
            if not opts:
                continue
            ut, loc, _, _ = parse_order(opts[0])
            u = {'t': ut, 'loc': loc, 'prov': b, 'hold': f'{ut} {loc} H', 'moves': {}, 'sup': {}}
            for o in opts:
                tok = o.split()
                if len(tok) < 3:
                    continue
                if tok[2] == 'H':
                    u['hold'] = o
                elif tok[2] == '-' and tok[-1] != 'VIA':
                    u['moves'].setdefault(_base(tok[3]), []).append((tok[3], o))
                elif tok[2] == 'S' and len(tok) >= 7 and tok[5] == '-':
                    u['sup'][(_base(tok[4]), _base(tok[6]))] = o
            U.append(u)
        if not U:
            return []
        my_provs = {u['prov'] for u in U}

        def move_order(u, dprov, tg):
            lst = u['moves'].get(dprov)
            if not lst:
                return None
            return min(lst, key=lambda x: (self._near(u['t'], x[0], tg), x[1]))[1]

        orders, moving_to, staying = {}, {}, set()

        # 1. garrison own SCs next to enemy units
        for sc in sorted(own_scs, key=lambda s: -enemy_adj.get(s, 0)):
            if enemy_adj.get(sc, 0) == 0:
                continue
            idx = next((i for i, u in enumerate(U) if u['prov'] == sc), None)
            if idx is not None:
                if idx not in orders:
                    orders[idx] = U[idx]['hold']
                    staying.add(sc)
                continue
            cands = [i for i, u in enumerate(U) if i not in orders and sc in u['moves']
                     and not (u['prov'] in own_scs and enemy_adj.get(u['prov'], 0) > 0)]
            if cands:
                i = min(cands, key=lambda i: U[i]['prov'])
                orders[i] = move_order(U[i], sc, tlist)
                moving_to[i] = sc

        # 2. supported attacks on occupied target SCs
        for sc in sorted(t for t in targets if t in enemy_occ):
            need = 1 + (1 if enemy_adj.get(sc, 0) > 0 else 0)
            for a in sorted((i for i, u in enumerate(U) if i not in orders and sc in u['moves']),
                            key=lambda i: U[i]['prov']):
                sups = [j for j, u in enumerate(U) if j != a and j not in orders and (U[a]['prov'], sc) in u['sup']]
                if len(sups) >= need:
                    mo = move_order(U[a], sc, tlist)
                    if mo is None:
                        continue
                    orders[a] = mo
                    moving_to[a] = sc
                    for j in sups[:need]:
                        orders[j] = U[j]['sup'][(U[a]['prov'], sc)]
                        staying.add(U[j]['prov'])
                    break

        # 3. strength-gated moves toward targets
        free = [i for i in range(len(U)) if i not in orders]
        support_avail = {}
        for i in free:
            for (sp, dp) in U[i]['sup']:
                support_avail[(sp, dp)] = support_avail.get((sp, dp), 0) + 1
        taken = set(moving_to.values()) | staying
        cand = []
        for i in free:
            u = U[i]
            tg = tlist if self._near(u['t'], u['loc'], tlist) < INF else all_targets
            here = self._near(u['t'], u['loc'], tg)
            bonus = 0.0
            if u['prov'] not in own_scs and u['prov'] in info['scset']:
                bonus = CONFIG['FALL_STAY_BONUS'] if fall else CONFIG['SC_BONUS']
            cand.append((-min(here, 30) + bonus, 1, i, None, u['prov']))
            for dprov, lst in u['moves'].items():
                if dprov in my_provs or dprov in enemy_occ:
                    continue
                strength = 1 + support_avail.get((u['prov'], dprov), 0)
                if CONFIG['GATE'] and strength < enemy_adj.get(dprov, 0):
                    continue
                for dloc, o in lst:
                    s = -min(self._near(u['t'], dloc, tg), 30)
                    if dprov in info['scset'] and dprov not in own_scs:
                        s += CONFIG['SC_BONUS']
                    cand.append((s, 0, i, o, dprov))
        cand.sort(key=lambda x: (-x[0], x[1], x[2], x[3] or ''))
        for s, is_hold, i, o, dprov in cand:
            if i in orders or dprov in taken:
                continue
            if is_hold:
                orders[i] = U[i]['hold']
                staying.add(U[i]['prov'])
            else:
                if CONFIG['SUP_ADVANCE']:
                    need = max(0, enemy_adj.get(dprov, 0) - 1)
                    if need:
                        key = (U[i]['prov'], dprov)
                        sups = [j for j in free if j != i and j not in orders and key in U[j]['sup']
                                and U[j]['prov'] not in taken]
                        if len(sups) < need:
                            continue
                        for j in sups[:need]:
                            orders[j] = U[j]['sup'][key]
                            staying.add(U[j]['prov'])
                            taken.add(U[j]['prov'])
                orders[i] = o
                moving_to[i] = dprov
            taken.add(dprov)

        # 4. holding units support contested moves of ours
        for i, u in enumerate(U):
            if orders.get(i) != u['hold']:
                continue
            if u['prov'] in own_scs and enemy_adj.get(u['prov'], 0) > 0:
                continue
            if fall and u['prov'] in info['scset'] and u['prov'] not in own_scs:
                continue
            best, best_s = None, 0
            for a, dprov in moving_to.items():
                key = (U[a]['prov'], dprov)
                if key in u['sup']:
                    s = enemy_adj.get(dprov, 0) + (2 if dprov in enemy_occ else 0)
                    if s > best_s:
                        best, best_s = u['sup'][key], s
            if best is not None:
                orders[i] = best
        return [orders[i] for i in sorted(orders) if orders[i]]

    # --------------------------------------------------------------------------------------------------------
    def _retreats(self, possible, locs):
        own = set(self.game.get_power(self.power_name).centers)
        targets = [sc for sc in self.info['scs'] if sc not in own]
        orders = []
        for base in locs:
            best, best_key, disband = None, None, None
            for o in possible.get(base) or []:
                ut, loc, kind, dest = parse_order(o)
                if kind == 'R' and dest is not None:
                    key = (0 if _base(dest) in own else 1, self._near(ut, dest, targets), o)
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
        targets, _ = self._target_set()
        tlist = list(targets)
        orders = []
        if n > 0:
            cands = []
            for base in locs:
                best, best_key = None, None
                for o in possible.get(base) or []:
                    ut, loc, kind, _ = parse_order(o)
                    if kind != 'B':
                        continue
                    key = (self._near(ut, loc, tlist), 0 if ut == 'A' else 1, o)
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
                        cands.append((-self._near(ut, loc, tlist), o))
            cands.sort()
            orders = [o for _, o in cands[:-n]]
        return orders
