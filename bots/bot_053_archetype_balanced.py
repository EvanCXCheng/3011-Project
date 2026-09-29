"""bot_053 — family: archetype — parent: bot_040 (new STYLE 'balanced')

Change vs bot_040 (one idea): a 'balanced' style — the aggressive targeting and all-in supported attacks of bot_040,
plus defence only against *active* threats: an own SC that an active enemy (one that has ever moved) can enter is held,
with a support-hold when 2+ such enemies are adjacent. bot_040 tied the champion in S2 but lost 1.9 SC in S3 against a
strong Hidden Agent; the question is whether minimal defence closes that gap. (Sparring field / S4 stand-in.)

bot_040 description:

A rule-based archetype opponent for the Scenario 4 sparring field (human request, 27 Sep): a plausible style another
group might write, deliberately different from our simulation bots. One rule engine, three styles (STYLE):
  aggressive   targets enemy-held SCs first (neutral second), piles every available supporter into attacks,
               never garrisons;
  turtle       garrisons every threatened own SC first (hold + support-hold), expands only to neutral SCs within
               TURTLE_RANGE of its units, attacks an enemy-held SC only with 3v1;
  opportunist  each turn picks the weakest reachable power (units + 0.5 x mean distance) and converges on its SCs
               (neutral SCs second), light defence.
Movement: (1) defence per style, (2) supported attacks on enemy-occupied targets, (3) remaining units step toward
the best target by priority and BFS distance, no two units to one province, (4) idle units support our attacks.
Builds/retreats/disbands: nearest-target rules. Helpers (map precomputation) copied from earlier bots.

Technique tags: archetype-aggressive, archetype-balanced
"""
import time
from collections import deque

from agent_baselines import Agent

CONFIG = {
    'STYLE': 'balanced',     # aggressive | turtle | opportunist | balanced
    'TIME_BUDGET': 0.3,
    'TURTLE_RANGE': 2,
}

INF = 10 ** 6


# ----------------------------------------------------------------------------------------------------------------
# Map precomputation (shared helper copied from earlier bots)
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

    def __init__(self, agent_name='bot_053_archetype_balanced'):
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

    def get_actions(self):
        orders = []
        try:
            if self.info is None:
                self.info = map_info(self.game)
            possible = self.game.get_all_possible_orders()
            locs = self.game.get_orderable_locations(self.power_name)
            ptype = self.game.phase_type
            if ptype == 'M':
                orders = self._movement(possible, locs)
            elif ptype == 'R':
                orders = self._retreats(possible, locs)
            elif ptype == 'A':
                orders = self._adjustments(possible, locs)
        except Exception:
            pass
        return orders

    # --------------------------------------------------------------------------------------------------------
    def _near(self, utype, loc, targets):
        row = self.info['scdist'][utype].get(loc)
        if not row or not targets:
            return INF
        return min(row[sc] for sc in targets)

    def _active_powers(self):
        g = self.game
        act = set()
        for ph in g.order_history.keys():
            for p, olist in g.order_history[ph].items():
                if p != self.power_name and any(' - ' in o for o in olist or []):
                    act.add(p)
        return act

    def _target_priority(self):
        """{sc: priority}; higher first."""
        g = self.game
        me = self.power_name
        style = CONFIG['STYLE']
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
            if style in ('aggressive', 'balanced'):
                pri[c] = 2 if o is not None else 1
            elif style == 'turtle':
                if o is None and min((self._near(t, l, [c]) for t, l in my_units), default=INF) <= CONFIG['TURTLE_RANGE']:
                    pri[c] = 1
            else:
                pri[c] = 2 if o == victim else (1 if o is None else 0)
        return {c: v for c, v in pri.items() if v > 0}

    def _movement(self, possible, locs):
        g = self.game
        me = self.power_name
        style = CONFIG['STYLE']
        adj = self.info['adj']
        own = set(g.get_power(me).centers)
        pri = self._target_priority()
        targets = list(pri)
        active = self._active_powers()
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
                    if style in ('turtle', 'balanced') and enemy_adj[c] >= 2:
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
        need = {'aggressive': 1, 'opportunist': 1, 'turtle': 2, 'balanced': 1}[style]
        for c in sorted((c for c in pri if c in enemy_occ), key=lambda c: -pri[c]):
            if c in taken:
                continue
            for a in sorted(b for b in U if b not in orders and c in U[b]['moves']):
                sups = [b for b in U if b != a and b not in orders and (a, c) in U[b]['sup']]
                if len(sups) >= need:
                    use = sups if style in ('aggressive', 'balanced') else sups[:need]
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
    def _retreats(self, possible, locs):
        targets = list(self._target_priority()) or [c for c in self.info['scs']]
        orders = []
        for b in locs:
            best, bk, dis = None, None, None
            for o in possible.get(b) or []:
                tok = o.split()
                if len(tok) >= 4 and tok[2] == 'R':
                    k = (self._near(tok[0], tok[3], targets), o)
                    if bk is None or k < bk:
                        best, bk = o, k
                elif len(tok) >= 3 and tok[2] == 'D':
                    dis = o
            if best or dis:
                orders.append(best or dis)
        return orders

    def _adjustments(self, possible, locs):
        pw = self.game.get_power(self.power_name)
        n = len(pw.centers) - len(pw.units)
        targets = list(self._target_priority()) or [c for c in self.info['scs'] if c not in pw.centers]
        orders = []
        if n > 0:
            cands = []
            for b in locs:
                best = None
                for o in possible.get(b) or []:
                    tok = o.split()
                    if len(tok) >= 3 and tok[2] == 'B':
                        k = (self._near(tok[0], tok[1], targets), 0 if tok[0] == 'A' else 1, o)
                        if best is None or k < best:
                            best = k
                if best:
                    cands.append(best)
            cands.sort()
            orders = [k[2] for k in cands[:n]]
        elif n < 0:
            cands = []
            for b in locs:
                for o in possible.get(b) or []:
                    tok = o.split()
                    if len(tok) >= 3 and tok[2] == 'D':
                        cands.append((-self._near(tok[0], tok[1], targets), o))
            cands.sort()
            orders = [o for _, o in cands[:-n]]
        return orders
