"""bot_072 — family: evolution — parent: bot_065

Change vs bot_065 (one infrastructure change, from bot_070/071): the GA's fitness rollouts run on the own movement
resolver instead of engine copies (FAST_RES; ~x34 rollouts per move), with the split-coast army adjacency fix the
resolver needs ([090]). bot_065 managed only a few generations of 2 opponent samples each in 0.45 s; the hypothesis is
that the GA gains more from cheap rollouts than the halving race (bot_071), because it can breed and re-test many
more generations. FAST_RES False = bot_065 + map fix.

bot_065 notes:
bot_065 — family: evolution — parent: bot_045 (code base) / bot_062 (idea)

Change vs bot_045 (one idea): the halving race is replaced by a genetic algorithm over bot_045's candidate pool.
bot_062 showed a GA is only competitive with good seeds; bot_045's pool (hill-climbing optima, lookahead candidates,
valuemap plan, convoys) is the best seed set we have. Genes = each unit's orders appearing in any pool plan (+ hold);
each generation all plans meet the same fresh opponent samples (two-ply in Spring, as bot_045), the better half
survives, the rest is refilled by uniform crossover + mutation with support/convoy repair; best mean is played.

bot_045 notes:

Change vs bot_029 (one idea, stacking two measured near-misses against the champion bot_022): adds bot_035's valuemap
candidate (bot_014's joint order) to bot_029's race (two-ply Spring rollouts + convoy candidates). Each was about
+0.3 SC vs bot_022 at 210 games (029 +0.33±0.20, 035 +0.30±0.17) and they target different weaknesses (Spring
evaluation / England / candidate diversity); this tests whether the gains add up.

bot_029 notes (parent):

Change vs bot_028: adds bot_026's convoy candidates (both ideas measured separately against bot_022: two-ply
+0.30±0.19, convoys +0.26±0.21; they target different weaknesses, so this tests whether they add up).

bot_028 change vs bot_022: two-ply rollouts in Spring. After the simulated Spring move, every power plays a cheap
Fall reply (each unit steps to the neighbour closest to its power's nearest unowned SC; static powers hold;
dislodged units disband), and the rollout is scored on SC ownership after Fall, when ownership actually changes.
bot_022 scored Spring positions with a heuristic (occupied SCs x 0.5). Fall phases stay one-ply. Costs roughly
2x per rollout, so fewer samples per candidate.

bot_022 notes (parent):

Change vs bot_020 (one idea, a hybrid): a mixed candidate pool raced by successive halving. The pool is bot_020's
top local optima from heuristic hill climbing (SEARCH_BUDGET 0.15 s) plus bot_021-style lookahead candidates (a
greedy distance-based joint order and random perturbations with supports of our own moves). The pool is raced
with common opponent samples, dropping the worse half every HALVE_EVERY rounds (bot_021's allocation).
Motivation: bot_020 (S3 +1.71 vs bot_007) and bot_021 (S2 69% wins) each improve a different scenario.

bot_020 notes (parent):

Change vs bot_016 (one idea): rollout selection. The hill climbing (now given SEARCH_BUDGET of the time) keeps every
distinct local optimum it reaches; the TOP_K best by heuristic value are then compared by one-move simulations on a
history-free copy of the position against opponent orders sampled from each power's class (static hold, greedy
greedy move, strong/erratic/unknown mixes), and the best mean simulated outcome is played. The heuristic evaluation
proposes, the engine decides.

bot_016 notes (parent):

Change vs bot_008 (one idea): prediction-accuracy gating. For every opponent we also score how often its units did
what our greedy prediction said (move to one of the predicted best neighbours, or hold when none is better). After the
static check, the hit rate alone classifies: >= ACC_GREEDY greedy (the Greedy baseline scores 1.0), >= ACC_STRONG
'strong', else erratic (Random/Attitude score 0.1-0.24). In a sample S3 game the lookahead stand-in scored 0.64 with
79% target-seeking moves, so bot_008 called it erratic and assumed its units rarely stay put, making unsupported
attacks on it look good. A 'strong' power is treated worst-case like bot_003: every province it can reach counts as a full
threat and its units are assumed to hold (hold prob STRONG_HOLD). Static/greedy/erratic powers keep bot_008's model.
Motivation: bot_008 gained +6.2 SC in S1 but lost 1.2 SC in S3 against the lookahead stand-in.

bot_008 change vs bot_003: the evaluation uses predicted enemy behaviour instead of raw adjacency. Each opponent is
classified from its order history (static / greedy / erratic / unknown) and each enemy unit gets a hold probability
and move probabilities. Contest counts become expected enemy entries (threat), moves into occupied provinces use
the occupant's hold probability, and cut-support and home-SC threat use the same threat map. Static units then no
longer block or threaten anything, which is what stalls bot_003 in Scenario 1.

Parent hypothesis: local search over our joint order set (moves, holds and supports to our own units) scored by a
heuristic evaluation finds coordinated orders (supported attacks, no self-bounces, covered home SCs) that per-unit
greedy rules miss. The evaluation estimates, for each move, a success probability from our attack strength
(1 + valid supports), whether the destination is occupied, and how many enemy units could contest it. It then
scores expected unit positions (unowned SC captured, distance to the nearest unowned SC) and penalises own SCs left
open to adjacent enemies. Search: coordinate-ascent hill climbing with random restarts until the time budget.

Technique tags: local-search, hill-climbing, heuristic-eval, opponent-aware-eval, prediction-accuracy-gating, rollout-selection, hybrid-candidate-race, two-ply-spring, convoy-candidates, multi-source-candidates, genetic-algorithm
"""
import copy
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
    'GA_POP': 16, 'GA_SAMPLES': 2, 'GA_MUT': 0.1,
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
    'FAST_RES': True,        # False = bot_065 + map fix (engine copies for rollouts)
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
            for b in m.loc_abut.get(a) or m.loc_abut.get(a.lower(), []):
                b = b.upper()
                if t == 'A':
                    b = _base(b)
                if b in nodeset and b not in nbrs and m.abuts(t, a, '-', b):
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


# own movement resolver (validated against the engine, see docstring)
_PARSE_CACHE = {}


def parse_mo(order, reach_a):
    tok = order.split()
    loc = tok[1]
    b = _base(loc)
    if len(tok) < 4:
        return b, ('H',)
    k = tok[2]
    if k == '-':
        dest = tok[3]
        db = _base(dest)
        adj = tok[0] == 'F' or db in reach_a.get(loc, ())
        return b, ('M', dest, db, tok[-1] == 'VIA' or not adj, adj)
    if k == 'S':
        tb = _base(tok[4])
        if len(tok) >= 7 and tok[5] == '-':
            return b, ('S', tb, _base(tok[6]), tok[6])
        return b, ('S', tb, None)
    if k == 'C' and len(tok) >= 7:
        return b, ('C', _base(tok[4]), _base(tok[6]))
    return b, ('H',)


def _convoy_path(units, orders, reach_f, a, d, gone):
    """True if fleets ordered to convoy a -> d (minus those in gone) link a to d."""
    live = [f for f, o in orders.items() if o[0] == 'C' and o[1] == a and o[2] == d and f in units
            and units[f][1] == 'F' and f not in gone]
    if not live:
        return False
    rf = {f: reach_f.get(units[f][2], ()) for f in live}
    frontier = [f for f in live if a in rf[f]]
    seen = set(frontier)
    while frontier:
        f = frontier.pop()
        if d in rf[f]:
            return True
        for g in live:
            if g not in seen and g in rf[f]:
                seen.add(g)
                frontier.append(g)
    return False


def resolve_moves(units, orders, reach_f):
    moves = {}
    via = {}
    into = {}
    convoyed = {(o[1], o[2]) for o in orders.values() if o[0] == 'C'}
    for b, o in orders.items():
        if o[0] == 'M' and b in units and o[2] != b:
            moves[b] = o
            via[b] = o[3] and not (o[4] and (b, o[2]) not in convoyed)   # VIA with no convoy ordered: overland
            into.setdefault(o[2], []).append(b)
    if not moves:
        return dict(units), set()
    own_conv = {(o[1], o[2]) for f, o in orders.items() if o[0] == 'C' and f in units and o[1] in units
                and units[f][0] == units[o[1]][0]}
    for b in list(via):
        o = moves[b]
        if o[4] and units[b][1] == 'A' and (via[b] or (b, o[2]) in own_conv):
            # engine: an adjacent army move is convoyed if VIA or its own power orders a convoy, and a convoy route
            # exists; otherwise it moves overland
            via[b] = _convoy_path(units, orders, reach_f, b, o[2], ())
    sup = {}
    for b, o in orders.items():
        if o[0] != 'S' or b not in units:
            continue
        tb, db = o[1], o[2]
        if tb not in units:
            continue
        if db is None:
            if tb in moves:
                continue
        else:
            m = moves.get(tb)
            if m is None or m[2] != db or ('/' in o[3] and o[3] != m[1]):
                continue               # a support naming another coast does not match the move
        sup.setdefault((tb, db), []).append(b)

    res = {}
    state = {}              # 1 = guessing, 2 = resolved
    dep = []
    conv_cache = {}

    def dislodged(f):
        for x in into.get(f, ()):
            if resolve(x):
                return True
        return False

    def convoy_ok(a):
        fl = conv_cache.get(a)
        if fl is None:
            fl = conv_cache[a] = [f for f, o in orders.items() if o[0] == 'C' and o[1] == a]
        return _convoy_path(units, orders, reach_f, a, moves[a][2], [f for f in fl if dislodged(f)])

    def valid(a):
        return not via[a] or convoy_ok(a)

    def cut(s):
        o = orders[s]
        into_prov = o[2] if o[2] is not None else o[1]
        pw = units[s][0]
        for a in into.get(s, ()):
            if units[a][0] == pw or not valid(a):
                continue
            if a != into_prov:
                return True
            if resolve(a):          # attack from the supported-into province cuts only by dislodging
                return True
        return False

    def nsup(key, not_power=None):
        n = 0
        for s in sup.get(key, ()):
            if not_power is not None and units[s][0] == not_power:
                continue
            if not cut(s):
                n += 1
        return n

    def h2h(a, occ):
        mo = moves.get(occ)
        return mo is not None and mo[2] == a and not via[a] and not via[occ]

    def adjudicate(a):
        d = moves[a][2]
        if not valid(a):
            return False
        pw = units[a][0]
        occ = d if d in units else None
        hh = occ is not None and h2h(a, occ)
        if occ is not None and (hh or occ not in moves or not resolve(occ)):
            opw = units[occ][0]
            # own-power supports cannot dislodge their own unit (engine: not applied to explicit VIA / non-adjacent orders)
            attack = 0 if opw == pw else 1 + nsup((a, d), None if via[a] and moves[a][3] else opw)
        else:
            attack = 1 + nsup((a, d))
        if hh:
            if attack <= 1 + nsup((occ, a)):
                return False
        elif occ is not None:
            if occ in moves:
                hold = 0 if resolve(occ) else 1
            else:
                hold = 1 + nsup((occ, None))
            if attack <= hold:
                return False
        for b in into[d]:
            if b == a or not valid(b):
                continue
            if d in units and h2h(b, d) and resolve(d):
                continue            # b lost a head-to-head battle: no prevent strength
            if attack <= 1 + nsup((b, d)):
                return False
        return True

    def resolve(a):
        st = state.get(a)
        if st == 2:
            return res[a]
        if st == 1:
            if a not in dep:
                dep.append(a)
            return res[a]
        n0 = len(dep)
        res[a] = False
        state[a] = 1
        r1 = adjudicate(a)
        if len(dep) == n0:
            if state.get(a) != 2:
                res[a] = r1
                state[a] = 2
            return res[a]
        if dep[n0] != a:
            dep.append(a)
            res[a] = r1
            return r1
        for x in dep[n0:]:
            state.pop(x, None)
        del dep[n0:]
        res[a] = True
        state[a] = 1
        r2 = adjudicate(a)
        if r1 == r2:
            for x in dep[n0:]:
                state.pop(x, None)
            del dep[n0:]
            res[a] = r1
            state[a] = 2
            return r1
        cyc = dep[n0:] if a in dep[n0:] else dep[n0:] + [a]
        del dep[n0:]
        if (not r1) and r2:        # two consistent outcomes: circular movement, all succeed
            for x in cyc:
                res[x] = True
                state[x] = 2
        else:                      # no consistent outcome: convoy paradox, the convoyed moves fail
            anyvia = any(via.get(x) for x in cyc)
            for x in cyc:
                if via.get(x) or not anyvia:
                    res[x] = False
                    state[x] = 2
                else:
                    state.pop(x, None)
        return resolve(a)

    for a in moves:
        resolve(a)
    out = {}
    moved_in = set()
    for a, o in moves.items():
        if res[a]:
            moved_in.add(o[2])
    lost = set()
    for b, u in units.items():
        m = moves.get(b)
        if m is not None and res[b]:
            continue
        if b in moved_in:
            lost.add(b)
            continue
        out[b] = u
    for a, o in moves.items():
        if res[a]:
            pw, t, _ = units[a]
            out[o[2]] = (pw, t, o[1])
    return out, lost


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

    def __init__(self, agent_name='bot_072_evolution_fastga'):
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
        info = self.info
        me = self.power_name
        game = self.game
        fall = game.get_current_phase().startswith('F')
        own_scs = set(game.get_power(me).centers)
        targets = [sc for sc in info['scs'] if sc not in own_scs]
        target_set = set(targets)

        enemy_reach, enemy_occ = {}, set()
        for p, power in game.powers.items():
            if p == me:
                continue
            for u in power.units:
                t, loc = _unit_split(u)
                enemy_occ.add(_base(loc))
                for prov in info['reach'][t].get(loc, ()):
                    enemy_reach[prov] = enemy_reach.get(prov, 0) + 1
        aware = CONFIG['OPP_AWARE']
        holdp = {}
        if aware:
            enemy_reach, holdp = self._predict()

        # units and their candidate orders
        units = []          # (utype, loc, prov)
        cands = []          # per unit: list of (order, kind, dest_loc, dest_prov, sup_src_prov, sup_dest_prov)
        my_provs = set()
        for base in locs:
            opts = possible.get(base) or []
            if not opts:
                continue
            ut, loc = None, None
            for o in opts:
                ut, loc, _, _ = parse_order(o)
                if ut:
                    break
            if ut is None:
                continue
            units.append((ut, loc, _base(loc)))
            my_provs.add(_base(loc))
        for (ut, loc, prov) in units:
            lst = []
            for o in possible.get(prov) or []:
                tok = o.split()
                if len(tok) < 3:
                    continue
                kind = tok[2]
                if kind == 'H':
                    lst.append((o, 'H', loc, prov, None, None))
                elif kind == '-':
                    if tok[-1] == 'VIA':
                        continue
                    lst.append((o, '-', tok[3], _base(tok[3]), None, None))
                elif kind == 'S':
                    sprov = _base(tok[4])
                    if sprov not in my_provs:
                        continue
                    if len(tok) >= 7 and tok[5] == '-':
                        lst.append((o, 'SM', loc, prov, sprov, _base(tok[6])))
                    else:
                        lst.append((o, 'SH', loc, prov, sprov, None))
            if not lst:
                lst.append((f'{ut} {loc} H', 'H', loc, prov, None, None))
            cands.append(lst)
        n = len(units)
        if n == 0:
            return []

        sc_w = CONFIG['W_SC'] * (1.0 if fall else CONFIG['SPRING_SC_FACTOR'])
        def_w = CONFIG['W_DEF'] * (1.0 if fall else 0.5)
        w_dist = CONFIG['W_DIST']
        near_cache = {}

        def value(ut, loc):
            key = (ut, loc)
            v = near_cache.get(key)
            if v is None:
                d = self._near(ut, loc, targets)
                v = (sc_w if _base(loc) in target_set else 0.0) - w_dist * min(d, 20)
                near_cache[key] = v
            return v

        if aware:
            cut = {prov: 1.0 - (1.0 - CONFIG['CUT_FACTOR']) * min(1.0, enemy_reach.get(prov, 0.0))
                   for (_, _, prov) in units}
            threatened_own = [sc for sc in own_scs if enemy_reach.get(sc, 0.0) > CONFIG['THREAT_MIN']]
        else:
            cut = {prov: (CONFIG['CUT_FACTOR'] if enemy_reach.get(prov, 0) > 0 else 1.0) for (_, _, prov) in units}
            threatened_own = [sc for sc in own_scs if enemy_reach.get(sc, 0) > 0]
        p_h1, p_h2 = CONFIG['P_HOLDER_1'], CONFIG['P_HOLDER_2']
        p_occ1, p_occ2 = CONFIG['P_OCC_1'], CONFIG['P_OCC_2']
        p_c1, p_c2 = CONFIG['P_COMP_1'], CONFIG['P_COMP_2']

        def evaluate(assign):
            chosen = [cands[i][assign[i]] for i in range(n)]
            moving = {}
            dest_count = {}
            sup_move, sup_hold = {}, {}
            for i, c in enumerate(chosen):
                kind = c[1]
                if kind == '-':
                    moving[units[i][2]] = c[3]
                    dest_count[c[3]] = dest_count.get(c[3], 0) + 1
            for i, c in enumerate(chosen):
                kind = c[1]
                if kind == 'SM':
                    if moving.get(c[4]) == c[5]:
                        k = (c[4], c[5])
                        sup_move[k] = sup_move.get(k, 0.0) + cut[units[i][2]]
                elif kind == 'SH':
                    if c[4] not in moving:
                        sup_hold[c[4]] = sup_hold.get(c[4], 0.0) + cut[units[i][2]]
            score = 0.0
            p_success = {}
            for i, c in enumerate(chosen):
                ut, loc, prov = units[i]
                if c[1] != '-':
                    score += value(ut, loc)
                    continue
                dprov = c[3]
                s = 1.0 + sup_move.get((prov, dprov), 0.0)
                if dest_count[dprov] > 1:
                    p = 0.0
                elif dprov in my_provs and (dprov not in moving or moving[dprov] == prov):
                    p = 0.0
                else:
                    strong = s >= 1.99
                    p = 1.0
                    if dprov in enemy_occ:
                        if aware:
                            hp = holdp.get(dprov, 1.0)
                            p = hp * (p_h2 if strong else p_h1) + (1.0 - hp)
                        else:
                            p = p_occ2 if strong else p_occ1
                    k = enemy_reach.get(dprov, 0)
                    if k:
                        p *= (p_c2 if strong else p_c1) ** k
                p_success[prov] = p
                score += p * value(ut, c[2]) + (1.0 - p) * value(ut, loc)
            # own SCs open to adjacent enemies
            for sc in threatened_own:
                q = 0.0
                if sc in my_provs:
                    q = 1.0 - p_success.get(sc, 0.0) if sc in moving else 1.0
                for src, dst in moving.items():
                    if dst == sc:
                        q = max(q, p_success.get(src, 0.0))
                if q < 1.0:
                    score -= def_w * (1.0 - q) * min(1.0, (1.0 if aware else 0.5) * enemy_reach[sc])
            return score

        # initial: every unit's best order in isolation (others hold)
        rng = random.Random(zlib.crc32((game.get_current_phase() + me).encode()))
        hold_idx = []
        for lst in cands:
            hi = 0
            for j, c in enumerate(lst):
                if c[1] == 'H':
                    hi = j
                    break
            hold_idx.append(hi)
        budget = CONFIG['SEARCH_BUDGET'] if CONFIG['ROLLOUT'] else 0.40
        pool = {}

        def climb(assign, cur):
            improved = True
            while improved:
                improved = False
                order = list(range(n))
                rng.shuffle(order)
                for i in order:
                    if time.perf_counter() - t0 > budget:
                        return assign, cur
                    best_j, best_v = assign[i], cur
                    for j in range(len(cands[i])):
                        if j == assign[i]:
                            continue
                        assign[i] = j
                        v = evaluate(assign)
                        if v > best_v + 1e-9:
                            best_j, best_v = j, v
                    assign[i] = best_j
                    if best_v > cur + 1e-9:
                        cur = best_v
                        improved = True
            return assign, cur

        start = list(hold_idx)
        best_assign, best_val = climb(start, evaluate(start))
        best_assign = list(best_assign)
        pool[tuple(best_assign)] = best_val
        stale = 0
        while CONFIG['RESTARTS'] and time.perf_counter() - t0 < budget and stale < CONFIG['MAX_STALE_RESTARTS']:
            stale += 1
            a = list(best_assign)
            for i in rng.sample(range(n), max(1, n // 3)):
                a[i] = rng.randrange(len(cands[i]))
            a, v = climb(a, evaluate(a))
            pool[tuple(a)] = v
            if v > best_val + 1e-9:
                best_assign, best_val = list(a), v
                stale = 0
        best_orders = [cands[i][best_assign[i]][0] for i in range(n)]
        if not CONFIG['ROLLOUT'] or (len(pool) < 2 and not CONFIG['LA_CANDS']):
            return best_orders
        top = sorted(pool.items(), key=lambda kv: -kv[1])[:CONFIG['TOP_K']]
        cand_orders = [[cands[i][a[i]][0] for i in range(n)] for a, _ in top]
        if CONFIG['LA_CANDS']:
            try:
                seen = {tuple(sorted(c)) for c in cand_orders}
                for c in self._la_candidates(possible, locs, targets, rng):
                    key = tuple(sorted(c))
                    if key not in seen:
                        seen.add(key)
                        cand_orders.append(c)
            except Exception:
                pass
        if CONFIG['VM_CANDS']:
            try:
                vm = self._vm_movement(possible, locs, self._vm_value_map(), t0)
                if vm and tuple(sorted(vm)) not in {tuple(sorted(c)) for c in cand_orders}:
                    cand_orders.append(vm)
            except Exception:
                pass
        try:
            k = self._rollout_select(possible, cand_orders, t0, targets)
            return cand_orders[k]
        except Exception:
            return best_orders

    # --------------------------------------------------------------------------------------------------------
    def _rollout_select(self, possible, cand_orders, t0, targets):
        """Index of the candidate with the best mean one-move simulated outcome (0 if nothing was simulated)."""
        g0 = self.game
        me = self.power_name
        info = self.info
        rng = random.Random(zlib.crc32((g0.get_current_phase() + me + 'R').encode()))
        fast = CONFIG['FAST_RES']
        base_game = None if fast else light_game(g0)
        reach_a, reach_f = info['reach']['A'], info['reach']['F']
        units0 = {}
        for p, pw in g0.powers.items():
            for u in pw.units:
                t, loc = _unit_split(u)
                units0[_base(loc)] = (p, t, loc)
        pc = _PARSE_CACHE

        def parse_into(lst, out):
            for o in lst:
                r = pc.get(o)
                if r is None:
                    r = pc[o] = parse_mo(o, reach_a)
                out[r[0]] = r[1]
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

        def fall_reply_fast(units):
            """fall_reply on a resolver position (dislodged units are already gone)."""
            orders = {}
            for b, (p, t, loc) in units.items():
                ptg = ply2.get(p)
                if ptg is None:
                    continue
                here = self._near(t, loc, ptg)
                best, bd = None, here
                for n in adj[t].get(loc, ()):
                    d = self._near(t, n, ptg)
                    if d < bd:
                        best, bd = n, d
                if best is not None:
                    orders[b] = ('M', best, _base(best), False, True)
            return resolve_moves(units, orders, reach_f)[0] if orders else units

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
            return score_occ(occ, [_unit_split(u) for u in g.get_power(me).units])

        def score_units(units):
            return score_occ({b: u[0] for b, u in units.items()}, [(u[1], u[2]) for u in units.values() if u[0] == me])

        def score_occ(occ, my_units):
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

        # genetic algorithm over the candidate pool (replaces the halving race)
        units = sorted({' '.join(o.split()[:2]) for c in cand_orders for o in c})
        genes = {u: [] for u in units}
        for c in cand_orders:
            for o in c:
                u = ' '.join(o.split()[:2])
                if o not in genes[u]:
                    genes[u].append(o)
        for u in units:
            if f'{u} H' not in genes[u]:
                genes[u].append(f'{u} H')

        def repair(plan):
            d = {' '.join(o.split()[:2]): o for o in plan}
            for u, o in list(d.items()):
                tok = o.split()
                if len(tok) >= 7 and tok[2] == 'S' and tok[5] == '-':
                    mt = d.get(' '.join(tok[3:5]), '').split()
                    if not (len(mt) >= 4 and mt[2] == '-' and _base(mt[3]) == _base(tok[6])):
                        d[u] = f'{u} H'
                elif len(tok) >= 5 and tok[2] == 'C':
                    if d.get(' '.join(tok[3:5]), '').split()[-1:] != ['VIA']:
                        d[u] = f'{u} H'
            return [d[u] for u in sorted(d)]

        pop = {}
        for c in cand_orders:
            pop.setdefault(tuple(sorted(c)), [list(c), 0.0, 0])
        budget = CONFIG['TIME_BUDGET']
        done = True
        while time.perf_counter() - t0 < budget:
            for _ in range(CONFIG['GA_SAMPLES']):
                opp = sample()
                if fast:
                    opp_orders = {}
                    for p, lst in opp.items():
                        parse_into(lst, opp_orders)
                for key, st in pop.items():
                    if time.perf_counter() - t0 > budget:
                        done = False
                        break
                    if fast:
                        if len(st) < 4:
                            d = {}
                            parse_into(st[0], d)
                            st.append(d)
                        orders = dict(opp_orders)
                        orders.update(st[3])
                        u1 = resolve_moves(units0, orders, reach_f)[0]
                        if two:
                            u1 = fall_reply_fast(u1)
                        st[1] += score_units(u1)
                        st[2] += 1
                        continue
                    g = copy.deepcopy(base_game)
                    for p, lst in opp.items():
                        g.set_orders(p, lst)
                    g.set_orders(me, st[0])
                    g.process()
                    if two:
                        fall_reply(g)
                    st[1] += score(g)
                    st[2] += 1
                if not done:
                    break
            if not done:
                break
            ranked = sorted(pop.items(), key=lambda kv: -(kv[1][1] / max(1, kv[1][2])))
            keep = dict(ranked[:max(2, CONFIG['GA_POP'] // 2)])
            parents = [v[0] for v in keep.values()]
            tries = 0
            while len(keep) < CONFIG['GA_POP'] and tries < CONFIG['GA_POP'] * 5:
                tries += 1
                a, b = rng.sample(parents, 2)
                da = {' '.join(o.split()[:2]): o for o in a}
                db = {' '.join(o.split()[:2]): o for o in b}
                child = []
                for u in units:
                    o = db.get(u) if (u in db and rng.random() < 0.5) else da.get(u, db.get(u))
                    if rng.random() < CONFIG['GA_MUT']:
                        o = rng.choice(genes[u])
                    if o:
                        child.append(o)
                child = repair(child)
                keep.setdefault(tuple(sorted(child)), [child, 0.0, 0])
            pop = keep
        scored = [(st[1] / st[2], key) for key, st in pop.items() if st[2] > 0]
        if not scored:
            return 0
        best = pop[max(scored)[1]][0]
        cand_orders.append(best)
        return len(cand_orders) - 1

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
