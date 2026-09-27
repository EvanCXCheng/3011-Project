"""bot_046 — family: bandit — parent: bot_036

Change vs bot_036 (one idea): holder-aware priors. Units of powers classified static never move, so an unsupported
move into a province one of them holds always bounces, and a supported one always wins. The arm priors now say so:
unsupported moves into a static holder get a large prior penalty, pair arms against a static holder a larger bonus,
and moves that end next to a static-held target SC (where the unit can later support or attack) a small bonus.
bot_036 still stalled at ~12 SC in S1 (15% wins) because its units did not gather next to holders.

bot_036 change vs bot_034: pair arms for coordinated attacks. Independent per-unit bandits rarely draw a move and
its support together (bot_034: S1 8 SC in a sanity game). For every enemy-occupied SC we do not own, each unit that
can move in gets extra arms "move in, supported by unit Y" (up to MAX_PAIRS per unit); drawing one also sets Y's
order to that support for the rollout (Y's own draw is not credited in that rollout).

bot_034 hypothesis: a combinatorial-bandit search over our joint orders (decoupled UCB, as in decoupled UCT for simultaneous
moves) can find coordinated orders without any hand-made candidate generator. Each unit keeps its own UCB1 bandit
over its candidate orders (hold, moves, supports of neighbouring units' moves or holds). Every simulation draws one
order per unit by UCB, plays the joint orders one move deep against opponent orders sampled from each power's class
(static / greedy / strong / erratic, by greedy-prediction hit rate), and credits the outcome score to every unit's
chosen order. At the time limit each unit plays its best-mean order. Arms start from a greedy distance prior.
Opponent orders are shared by a small batch of draws (set once, clear_cache, deep-copy per draw).
Shared helpers (map precomputation, light game copy, opponent classification, retreats/builds) are copied from
earlier bots; the search itself is new.

Technique tags: decoupled-ucb, combinatorial-bandit, pair-arms, holder-aware-priors
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
    'UCB_C': 1.0,            # exploration constant, in units of the reward standard deviation
    'PRIOR_N': 2.0,          # virtual visits of the greedy prior per arm
    'PRIOR_BONUS': 0.5,      # prior advantage (in reward sd) of each unit's greedy-best arm
    'MAX_ARMS': 10,          # candidate orders per unit (best by greedy score; supports added on top)
    'BATCH': 4,              # joint draws per opponent sample
    'PAIRS': True,           # False = bot_034 behaviour
    'MAX_PAIRS': 4,          # pair arms per unit
    'PAIR_PRIOR': 2.0,       # prior score bonus of a pair arm over the plain move (greedy-score units)
    'HOLDER_PRIORS': True,   # False = bot_036 behaviour
    'STATIC_BOUNCE_PEN': 6.0,  # prior penalty: unsupported move into a static holder
    'STATIC_PAIR_BONUS': 4.0,  # extra prior bonus: pair arm against a static holder
    'STAGE_BONUS': 1.0,        # prior bonus: move ending next to a static-held target SC
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

    def __init__(self, agent_name='bot_046_bandit_staticprior'):
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
    # decoupled-UCB search
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
        # pair arms: forced[i][a] = (j, support order) or None
        forced = [[None] * len(arms[i]) for i in range(n_units)]
        if CONFIG['PAIRS']:
            enemy_occ = set()
            for p, pw in g0.powers.items():
                if p != me:
                    for u in pw.units:
                        enemy_occ.add(_base(_unit_split(u)[1]))
            hot = {c for c in enemy_occ if c in tset}
            for i, b in enumerate(units):
                added = 0
                for a0 in range(len(arms[i])):
                    tok = arms[i][a0].split()
                    if len(tok) < 4 or tok[2] != '-' or _base(tok[3]) not in hot:
                        continue
                    want = f'{tok[0]} {tok[1]} - '
                    for j, bj in enumerate(units):
                        if j == i or added >= CONFIG['MAX_PAIRS']:
                            continue
                        so = next((o for o in possible.get(bj) or []
                                   if ' S ' in o and o.split(' S ', 1)[1].startswith(want)
                                   and _base(o.split()[-1]) == _base(tok[3])), None)
                        if so:
                            arms[i].append(arms[i][a0])
                            prior[i].append(prior[i][a0] + CONFIG['PAIR_PRIOR'])
                            forced[i].append((j, so))
                            added += 1
        if CONFIG['HOLDER_PRIORS']:
            static_held = set()
            for p, pw in g0.powers.items():
                if p != me and self._classify(p) == 'static':
                    for u in pw.units:
                        static_held.add(_base(_unit_split(u)[1]))
            hot_static = {c for c in static_held if c in tset}
            if static_held:
                near_hot = set()
                for c in hot_static:
                    for t in ('A', 'F'):
                        for l, nb in info['adj'][t].items():
                            if any(_base(x) == c for x in nb):
                                near_hot.add(_base(l))
                for i in range(n_units):
                    for a in range(len(arms[i])):
                        tok = arms[i][a].split()
                        if len(tok) < 4 or tok[2] != '-':
                            continue
                        d = _base(tok[3])
                        if d in static_held:
                            if forced[i][a] is None:
                                prior[i][a] -= CONFIG['STATIC_BOUNCE_PEN']
                            else:
                                prior[i][a] += CONFIG['STATIC_PAIR_BONUS']
                        elif d in near_hot:
                            prior[i][a] += CONFIG['STAGE_BONUS']
        greedy_idx = [max(range(len(p)), key=lambda i: p[i]) for p in prior]

        def joint(choice):
            """Orders for a choice vector, applying pair overrides; returns (orders, set of overridden units)."""
            orders = [arms[i][choice[i]] for i in range(n_units)]
            over = set()
            for i in range(n_units):
                f = forced[i][choice[i]]
                if f and f[0] not in over and i not in over:
                    orders[f[0]] = f[1]
                    over.add(f[0])
            return orders, over

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
            g.set_orders(me, joint(choice)[0])
            g.process()
            return score(g)

        # calibrate reward scale with the greedy joint order
        opp = sample()
        rb = copy.deepcopy(base_game)
        for p, lst in opp.items():
            rb.set_orders(p, lst)
        rb.clear_cache()
        r0 = rollout(rb, greedy_idx)
        rewards = [r0]
        # arm statistics initialised from the prior: value = r0 + scaled prior advantage
        sums, cnts = [], []
        for i in range(n_units):
            pr = prior[i]
            top = max(pr)
            s_i, c_i = [], []
            for a in range(len(arms[i])):
                bonus = CONFIG['PRIOR_BONUS'] if a == greedy_idx[i] else 0.0
                v = r0 + 0.1 * (pr[a] - top) + bonus
                s_i.append(v * CONFIG['PRIOR_N'])
                c_i.append(CONFIG['PRIOR_N'])
            sums.append(s_i)
            cnts.append(c_i)
        total = 1
        budget = CONFIG['TIME_BUDGET']
        while time.perf_counter() - t0 < budget:
            opp = sample()
            rb = copy.deepcopy(base_game)
            for p, lst in opp.items():
                rb.set_orders(p, lst)
            rb.clear_cache()
            m = sum(rewards) / len(rewards)
            sd = math.sqrt(max(1e-6, sum((x - m) ** 2 for x in rewards) / len(rewards))) if len(rewards) > 1 else 1.0
            for _ in range(CONFIG['BATCH']):
                if time.perf_counter() - t0 > budget:
                    break
                total += 1
                lt = math.log(total + 1)
                choice = []
                for i in range(n_units):
                    best, bv = 0, None
                    for a in range(len(arms[i])):
                        v = sums[i][a] / cnts[i][a] + CONFIG['UCB_C'] * sd * math.sqrt(lt / cnts[i][a])
                        if bv is None or v > bv:
                            best, bv = a, v
                    choice.append(best)
                r = rollout(rb, choice)
                rewards.append(r)
                over = joint(choice)[1]
                for i, a in enumerate(choice):
                    if i in over:
                        continue
                    sums[i][a] += r
                    cnts[i][a] += 1
        final = [max(range(len(arms[i])), key=lambda a: (sums[i][a] / cnts[i][a], cnts[i][a])) for i in range(n_units)]
        return joint(final)[0]

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

