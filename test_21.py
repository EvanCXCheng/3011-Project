"""test_21.py - experiments for group 21's Diplomacy agent (CITS3011, 2026).

A self-contained copy of the evaluation method used to develop agent_21.py. Games are played with the
unmodified game.run_one_game; our seat is wrapped in an instrumented proxy that enforces the 1 s limit with
timeout_decorator (as in marking), times every call, checks every order against the true game's legal orders,
and checks that the agent's internal game stays in sync with the real one.

Scenarios (our power is POWERS[seed % 7]; opponents are drawn deterministically from the seed, so two agents
evaluated on the same seeds face identical seats and line-ups -> paired comparisons):
  1  all Static          2  each opponent from [Random, Attitude, Attitude, Greedy, Greedy] (as test.py)
  3  scenario 2 with one opponent seat replaced by a Hidden Agent stand-in (--standin)

Commands (run from the project folder that holds game.py, agent_baselines.py and agent_21.py):
  python test_21.py eval   --agent agent_21.py --scenarios 1 2 3 --n 42 [--seed-base 400000] [--standin greedy]
                           [--set KEY=VAL ...] [--workers 3] [--out results_21.jsonl]
  python test_21.py compare --a LABEL_A --b LABEL_B [--out results_21.jsonl]      paired mean-SC differences
  python test_21.py ablate --agent agent_21.py --keys ROLLOUT LA_CANDS VM_CANDS TWO_PLY CONVOYS FAST_RES CONFIRM ACC_GATE OPP_AWARE BASIC_SEARCH=true --n 42
                           (evaluates the agent with each technique switched off, then compares with the full agent)
  python test_21.py stress --agent agent_21.py [--procs 4]                          slow-machine timing check
  python test_21.py summary [--out results_21.jsonl]

--set overrides entries of the agent module's CONFIG dict (technique switches / parameters) for ablations.
Every game is one JSON line in --out; re-running skips games already recorded for the same agent label.
Memory: each worker process caps its address space at 512 MB (RLIMIT_AS), matching the marking limit.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
END_YEAR = 1920
WIN_SC = 18
TIME_LIMIT = 1.0
MEM_LIMIT_MB = 512
S2_POOL = ['random', 'attitude', 'attitude', 'greedy', 'greedy']
BASELINES = ('static', 'random', 'attitude', 'greedy')
AGENT_CLASS = 'StudentAgent'
# rubric: scenario -> [(points, win-rate threshold, mean-SC threshold)], strict '>'
RUBRIC = {1: [(1, 0.02, 7), (3, 0.20, 12), (5, 0.90, 16)],
          2: [(1, 0.02, 7), (3, 0.25, 10), (5, 0.50, 13)],
          3: [(1, 0.02, 7), (3, 0.20, 9), (5, 0.40, 12)]}


# ------------------------------------------------------------------------------------------ agents

def label_of(spec, overrides=None):
    lab = spec if spec in BASELINES else os.path.splitext(os.path.basename(spec))[0]
    if overrides:
        lab += '[' + ','.join(f'{k}={v}' for k, v in sorted(overrides.items())) + ']'
    return lab


def file_hash(spec):
    path = os.path.join(ROOT, 'agent_baselines.py') if spec in BASELINES else os.path.join(ROOT, spec)
    with open(path, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()[:10]


def parse_overrides(items):
    out = {}
    for item in items or []:
        k, v = item.split('=', 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


_CLASSES = {}


def load_agent_class(spec, overrides=None):
    key = (spec, json.dumps(overrides or {}, sort_keys=True))
    if key in _CLASSES:
        return _CLASSES[key]
    if spec in BASELINES:
        import agent_baselines as B
        cls = {'static': B.StaticAgent, 'random': B.RandomAgent,
               'attitude': B.AttitudeAgent, 'greedy': B.GreedyAgent}[spec]
    else:
        path = os.path.join(ROOT, spec)
        name = 'agent21mod_' + hashlib.sha1((path + key[1]).encode()).hexdigest()[:12]
        ms = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(ms)
        sys.modules[name] = mod
        ms.loader.exec_module(mod)
        if overrides:
            cfg = getattr(mod, 'CONFIG', None)
            if not isinstance(cfg, dict):
                raise ValueError(f'{spec} has no CONFIG dict')
            for k, v in overrides.items():
                if k not in cfg:
                    raise KeyError(f'{spec}: unknown CONFIG key {k}')
                cfg[k] = v
        cls = getattr(mod, AGENT_CLASS)
    _CLASSES[key] = cls
    return cls


class InstrumentedAgent:
    """Proxy for our seat: 1 s timeout per call (timeout_decorator), timing, legality and desync checks."""

    def __init__(self, cls, true_game):
        import timeout_decorator
        self._timeout = timeout_decorator.timeout(TIME_LIMIT)
        self.true_game = true_game
        self.power_name = None
        self.s = {'times': [], 'n_timeouts': 0, 'n_exceptions': 0, 'n_illegal': 0, 'n_desync': 0,
                  'exc': [], 'illegal': [], 't_init': 0.0, 't_new_game': 0.0, 't_update_max': 0.0}
        t = time.perf_counter()
        try:
            self.agent = self._timeout(cls)()
        except Exception as e:  # noqa: BLE001
            self._exc('init', e)
            self.agent = None
        self.s['t_init'] = time.perf_counter() - t

    def _exc(self, where, e):
        self.s['n_exceptions'] += 1
        if len(self.s['exc']) < 3:
            self.s['exc'].append(f'{where}: {type(e).__name__}: {str(e)[:150]}')

    def new_game(self, game, power_name):
        self.power_name = power_name
        if self.agent is None:
            return
        t = time.perf_counter()
        try:
            self._timeout(self.agent.new_game)(game, power_name)
        except Exception as e:  # noqa: BLE001
            self._exc('new_game', e)
        self.s['t_new_game'] = time.perf_counter() - t

    def get_actions(self):
        if self.agent is None:
            return []
        phase = self.true_game.get_current_phase()
        t = time.perf_counter()
        try:
            orders = self._timeout(self.agent.get_actions)()
        except Exception as e:  # noqa: BLE001
            self._exc('get_actions ' + phase, e)
            orders = []
        dt = time.perf_counter() - t
        self.s['times'].append(dt)
        if dt > TIME_LIMIT:
            self.s['n_timeouts'] += 1
            orders = []
        if not isinstance(orders, list):
            orders = []
        self._check(orders, phase)
        return orders

    def _check(self, orders, phase):
        g = self.true_game
        possible = g.get_all_possible_orders()
        legal = set()
        for loc in g.get_orderable_locations(self.power_name):
            legal.update(possible.get(loc, []))
        seen = set()
        for o in orders:
            bad = not isinstance(o, str) or o not in legal
            if not bad and g.phase_type != 'A':
                unit = ' '.join(o.split()[:2])
                bad = unit in seen
                seen.add(unit)
            if bad:
                self.s['n_illegal'] += 1
                if len(self.s['illegal']) < 5:
                    self.s['illegal'].append(f'{phase}: {o}')
        try:
            ag = self.agent.game
            if ag.get_current_phase() != phase or ag.get_units() != g.get_units():
                self.s['n_desync'] += 1
        except Exception:  # noqa: BLE001
            self.s['n_desync'] += 1

    def update_game(self, all_power_orders):
        if self.agent is None:
            return
        t = time.perf_counter()
        try:
            self._timeout(self.agent.update_game)(all_power_orders)
        except Exception as e:  # noqa: BLE001
            self._exc('update_game', e)
        self.s['t_update_max'] = max(self.s['t_update_max'], time.perf_counter() - t)

    def summary(self):
        ts = sorted(self.s['times'])
        return {'t_max': round(ts[-1], 4) if ts else 0.0, 't_mean': round(sum(ts) / len(ts), 4) if ts else 0.0,
                'n_moves': len(ts), 'n_over_0.6': sum(1 for x in ts if x > 0.6),
                'n_timeouts': self.s['n_timeouts'], 'n_exceptions': self.s['n_exceptions'],
                'n_illegal': self.s['n_illegal'], 'n_desync': self.s['n_desync'], 'exc': self.s['exc'],
                'illegal': self.s['illegal'], 't_init': round(self.s['t_init'], 4),
                't_new_game': round(self.s['t_new_game'], 4), 't_update_max': round(self.s['t_update_max'], 4)}


# ------------------------------------------------------------------------------------------ one game

def worker_init():
    for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[k] = '1'
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    lim = MEM_LIMIT_MB * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    except (ValueError, OSError):
        pass


def seat_plan(scenario, seed, standin=None):
    power = POWERS[seed % 7]
    rng = random.Random(seed * 10 + scenario)
    others = [p for p in POWERS if p != power]
    if scenario == 1:
        return power, {p: 'static' for p in others}
    opp = {p: rng.choice(S2_POOL) for p in others}
    if scenario == 3:
        opp[rng.choice(others)] = standin or 'greedy'
    return power, opp


def play(seats, seed, instrument, overrides=None):
    """seats: {power: spec}; instrument: powers to wrap. Returns a record dict."""
    from diplomacy import Game
    from game import run_one_game
    import numpy as np
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    true_game = Game()
    agents, wrapped = {}, {}
    for p in POWERS:
        cls = load_agent_class(seats[p], overrides if p in instrument else None)
        if p in instrument:
            wrapped[p] = InstrumentedAgent(cls, true_game)
            agents[p] = wrapped[p]
        else:
            agents[p] = cls()
    t0 = time.perf_counter()
    results, year = run_one_game(agents, game=true_game, end_year=END_YEAR)
    return {'final_sc': results, 'year': int(year) if year else END_YEAR,
            'wall': round(time.perf_counter() - t0, 2), 'agents': {p: w.summary() for p, w in wrapped.items()},
            'maxrss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)}


def run_scenario_game(task):
    agent, scenario, seed, standin, overrides = task
    power, opp = seat_plan(scenario, seed, standin)
    seats = dict(opp)
    seats[power] = agent
    rec = {'label': label_of(agent, overrides), 'hash': file_hash(agent), 'scenario': scenario, 'seed': seed,
           'power': power, 'standin': label_of(standin) if (scenario == 3 and standin) else None}
    try:
        out = play(seats, seed, [power], overrides)
        a = out['agents'][power]
        rec.update(status='ok', sc=min(out['final_sc'][power], WIN_SC), win=out['final_sc'][power] >= WIN_SC,
                   year=out['year'], wall=out['wall'], maxrss_mb=out['maxrss_mb'], **a)
    except BaseException as e:  # noqa: BLE001
        rec.update(status='error', error=f'{type(e).__name__}: {e}'[:300])
    return rec


# ------------------------------------------------------------------------------------------ records / stats

def load_records(path):
    recs = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        recs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return recs


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return float('nan'), float('nan')
    m = sum(xs) / n
    if n < 2:
        return m, float('nan')
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n)


def mark_of(scenario, recs):
    if not recs:
        return 0
    sc = sum(r['sc'] for r in recs) / len(recs)
    win = sum(1 for r in recs if r['win']) / len(recs)
    pts = 0
    for p, wt, st in RUBRIC[scenario]:
        if win > wt or sc > st:
            pts = p
    return pts


def summarize(recs, label):
    lines = [f'{label}:']
    total = 0
    for sc in (1, 2, 3):
        rs = [r for r in recs if r['label'] == label and r['scenario'] == sc and r['status'] == 'ok']
        if not rs:
            continue
        m, se = mean_se([r['sc'] for r in rs])
        w = sum(1 for r in rs if r['win']) / len(rs)
        pts = mark_of(sc, rs)
        total += pts
        tmax = max(r['t_max'] for r in rs)
        issues = sum(r['n_timeouts'] + r['n_exceptions'] + r['n_illegal'] + r['n_desync'] for r in rs)
        mem = max(r.get('maxrss_mb', 0) for r in rs)
        lines.append(f'  S{sc}: n={len(rs)} mean SC {m:.2f}±{se:.2f} win {w * 100:.1f}% -> {pts} pts | '
                     f'slowest move {tmax:.3f}s | issues {issues} | peak mem {mem:.0f} MB')
    lines.append(f'  estimated rubric points: {total}')
    return '\n'.join(lines)


def compare(recs, la, lb):
    """Paired mean-SC difference a - b per scenario and pooled (games matched on scenario, seed, stand-in)."""
    ka = {(r['scenario'], r['seed'], r['standin']): r for r in recs if r['label'] == la and r['status'] == 'ok'}
    kb = {(r['scenario'], r['seed'], r['standin']): r for r in recs if r['label'] == lb and r['status'] == 'ok'}
    shared = sorted(set(ka) & set(kb))
    lines = [f'{la} vs {lb}: {len(shared)} paired games']
    pooled = []
    for sc in (1, 2, 3):
        d = [ka[k]['sc'] - kb[k]['sc'] for k in shared if k[0] == sc]
        if d:
            m, se = mean_se(d)
            pooled += d
            lines.append(f'  S{sc}: n={len(d)} diff {m:+.2f}±{se:.2f}')
    if pooled:
        m, se = mean_se(pooled)
        lines.append(f'  pooled diff {m:+.2f}±{se:.2f} ({"significant" if se == se and m > 2 * se else "not significant"} at 2 SE)')
    return '\n'.join(lines)


# ------------------------------------------------------------------------------------------ commands

def run_eval(agent, scenarios, n, seed_base, standin, overrides, workers, out):
    have = {(r['label'], r['scenario'], r['seed'], r['standin']) for r in load_records(out) if r['status'] == 'ok'}
    label = label_of(agent, overrides)
    sl = label_of(standin) if standin else None
    tasks = []
    for sc in scenarios:
        for i in range(n):
            seed = seed_base + i
            if (label, sc, seed, sl if sc == 3 else None) not in have:
                tasks.append((agent, sc, seed, standin if sc == 3 else None, overrides))
    t0 = time.perf_counter()
    done = 0
    if tasks:
        with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as ex, open(out, 'a') as f:
            for rec in ex.map(run_scenario_game, tasks):
                f.write(json.dumps(rec) + '\n')
                f.flush()
                done += 1
                if done % 21 == 0:
                    print(f'  {label}: {done}/{len(tasks)} games ({time.perf_counter() - t0:.0f}s)', flush=True)
    return label


def cmd_eval(a):
    ovr = parse_overrides(a.set)
    label = run_eval(a.agent, a.scenarios, a.n, a.seed_base, a.standin, ovr, a.workers, a.out)
    print(summarize(load_records(a.out), label))


def cmd_compare(a):
    print(compare(load_records(a.out), a.a, a.b))


def cmd_ablate(a):
    full = run_eval(a.agent, a.scenarios, a.n, a.seed_base, a.standin, {}, a.workers, a.out)
    recs = load_records(a.out)
    print(summarize(recs, full))
    for key in a.keys:
        # KEY switches a technique off; KEY=VAL sets it (e.g. BASIC_SEARCH=true runs the basic technique (hill-climbing search) alone)
        if '=' in key:
            k, v = key.split('=', 1)
            v = {'true': True, 'false': False}.get(v.lower(), v)
            over = {k: v}
        else:
            over = {key: False}
        lab = run_eval(a.agent, a.scenarios, a.n, a.seed_base, a.standin, over, a.workers, a.out)
        recs = load_records(a.out)
        print(summarize(recs, lab))
        print(compare(recs, full, lab))


def _calib(_=None):
    t = time.perf_counter()
    s = 0
    for i in range(3_000_000):
        s += i * i
    return time.perf_counter() - t


def cmd_stress(a):
    """Several games pinned to one core (about procs x slower), as a pessimistic model of a slower marking CPU."""
    os.sched_setaffinity(0, {a.core})
    solo = min(_calib() for _ in range(3))
    tasks = [(a.agent, sc, 900050 + i, 'greedy' if sc == 3 else None, {}) for sc in (1, 2, 3) for i in range(a.n)]
    with ProcessPoolExecutor(max_workers=a.procs, initializer=worker_init) as ex:
        cont = sorted(ex.map(_calib, range(a.procs)))
        recs = list(ex.map(run_scenario_game, tasks))
    ok = [r for r in recs if r['status'] == 'ok']
    tmax = max((r['t_max'] for r in ok), default=float('nan'))
    bad = sum(r['n_timeouts'] + r['n_exceptions'] for r in ok) + len(recs) - len(ok)
    verdict = 'PASS' if ok and tmax < a.gate and bad == 0 else 'FAIL'
    print(f'stress: {a.procs} games on core {a.core}, slowdown x{cont[len(cont) // 2] / solo:.2f}, '
          f'slowest move {tmax:.3f}s, timeouts/exceptions {bad} -> {verdict} (gate {a.gate}s)')


def cmd_summary(a):
    recs = load_records(a.out)
    for lab in sorted({r['label'] for r in recs}):
        print(summarize(recs, lab))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def common(p, agent=True):
        if agent:
            p.add_argument('--agent', default='agent_21.py')
        p.add_argument('--out', default='results_21.jsonl')
        p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))

    p = sub.add_parser('eval')
    common(p)
    p.add_argument('--scenarios', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--n', type=int, default=42)
    p.add_argument('--seed-base', type=int, default=400000)
    p.add_argument('--standin', default='greedy')
    p.add_argument('--set', nargs='*', default=[])
    p.set_defaults(func=cmd_eval)
    p = sub.add_parser('compare')
    common(p, agent=False)
    p.add_argument('--a', required=True)
    p.add_argument('--b', required=True)
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser('ablate')
    common(p)
    p.add_argument('--keys', nargs='+', default=['ROLLOUT', 'LA_CANDS', 'VM_CANDS', 'TWO_PLY', 'CONVOYS', 'FAST_RES', 'CONFIRM', 'ACC_GATE', 'OPP_AWARE', 'BASIC_SEARCH=true'])
    p.add_argument('--scenarios', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--n', type=int, default=42)
    p.add_argument('--seed-base', type=int, default=400000)
    p.add_argument('--standin', default='greedy')
    p.set_defaults(func=cmd_ablate)
    p = sub.add_parser('stress')
    common(p)
    p.add_argument('--procs', type=int, default=4)
    p.add_argument('--n', type=int, default=4)
    p.add_argument('--core', type=int, default=0)
    p.add_argument('--gate', type=float, default=0.8)
    p.set_defaults(func=cmd_stress)
    p = sub.add_parser('summary')
    common(p, agent=False)
    p.set_defaults(func=cmd_summary)
    a = ap.parse_args()
    a.func(a)


if __name__ == '__main__':
    main()
