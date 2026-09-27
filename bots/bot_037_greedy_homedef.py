"""bot_037 — family: greedy — parent: bot_012

Change vs bot_012 (one idea, rule-based, no simulation): home defence. Before the attack matching, every own SC that
an enemy unit of an active power (one that has ever ordered a move) can move into is protected: the unit standing on it holds (and, with 2+ adjacent enemy units, a free
neighbour support-holds it); an empty threatened SC gets a free adjacent unit moved in. Those units and provinces
are then excluded from attacks and the no-bounce greedy. (Greedy backlog item 6; part of the style-pure sparring
field.)

bot_012 notes (parent):

Change vs bot_010 (one idea): supported attacks on occupied SCs, assigned as a matching problem. Every enemy-occupied
SC we do not own offers two roles, attacker (a unit that can move in) and supporter (a unit that can support that
move). Units are matched to roles with scipy's linear_sum_assignment (Hungarian method) to maximise total value;
only targets whose attacker and supporter roles are both filled are kept. The other units run bot_010's
no-self-bounce greedy and avoid the provinces these pairs use.

bot_010 change vs bot_001: no self-bounces. Units are assigned destinations jointly, best (unit, destination)
pair first, so no two of our units target the same province. A unit may enter a province one of our units occupies
only if that unit is moving out (and not into the entering unit's province, which would be a swap); otherwise the
entering unit falls back to its next-best option.

Parent hypothesis: moving every unit one step toward its nearest supply centre (SC) that we do not own, using
BFS distances computed separately for armies and fleets (coast-aware), is a working basic agent.
Deliberately minimal: no bounce avoidance, no supports, no convoys (those are later greedy iterations).

Technique tags: bfs-greedy, no-self-bounce, supported-attack-matching, home-defence
"""
import time
from collections import deque

import numpy as np
from scipy.optimize import linear_sum_assignment

from agent_baselines import Agent

CONFIG = {
    'TIME_BUDGET': 0.5,      # s per get_actions call
    'DEDUP': True,           # False = bot_001 behaviour
    'ATTACK_MATCH': True,    # False = bot_010 behaviour
    'HOME_DEF': True,        # False = bot_012 behaviour
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

    def __init__(self, agent_name='bot_037_greedy_homedef'):
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

    def _attack_pairs(self, possible, locs):
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

    def _home_defence(self, possible, locs):
        """Hold / support-hold / garrison own SCs that an enemy unit can enter."""
        me = self.power_name
        g = self.game
        adj = self.info['adj']
        active = set()                   # powers that have ever ordered a move (static powers never threaten)
        for ph in g.order_history.keys():
            for p, olist in g.order_history[ph].items():
                if p != me and any(' - ' in o for o in olist or []):
                    active.add(p)
        enemy_adj = {}
        for p, pw in g.powers.items():
            if p == me or p not in active:
                continue
            for u in pw.units:
                t, loc = u.lstrip('*').split()[:2]
                for n in adj[t].get(loc, ()):
                    b = n.split('/')[0]
                    enemy_adj[b] = enemy_adj.get(b, 0) + 1
        own = set(g.get_power(me).centers)
        mine = set(locs)
        fixed, blocked = {}, set()
        for sc in sorted((c for c in own if enemy_adj.get(c, 0) > 0), key=lambda c: -enemy_adj[c]):
            if sc in mine:
                if sc in fixed:
                    continue
                hold = next((o for o in possible.get(sc) or [] if o.split()[2:3] == ['H']), None)
                if hold is None:
                    continue
                fixed[sc] = hold
                blocked.add(sc)
                if enemy_adj[sc] >= 2:
                    unit = ' '.join(hold.split()[:2])
                    for b in locs:
                        if b in fixed or b == sc:
                            continue
                        sup = next((o for o in possible.get(b) or [] if o.endswith(f' S {unit}')), None)
                        if sup:
                            fixed[b] = sup
                            blocked.add(b)
                            break
            else:
                for b in locs:
                    if b in fixed or (b in own and enemy_adj.get(b, 0) > 0):
                        continue
                    mv = next((o for o in possible.get(b) or []
                               if len(o.split()) >= 4 and o.split()[2] == '-' and o.split()[3].split('/')[0] == sc
                               and not o.endswith('VIA')), None)
                    if mv:
                        fixed[b] = mv
                        blocked |= {sc, b}
                        break
        return fixed, blocked

    def _movement_dedup(self, possible, locs, t0):
        targets = self._targets()
        fixed, blocked = {}, set()
        if CONFIG['HOME_DEF']:
            try:
                fixed, blocked = self._home_defence(possible, locs)
            except Exception:
                fixed, blocked = {}, set()
        if CONFIG['ATTACK_MATCH']:
            try:
                f2, b2 = self._attack_pairs(possible, [b for b in locs if b not in fixed])
                fixed.update(f2)
                blocked |= b2
            except Exception:
                pass
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
