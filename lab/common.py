"""Shared lab helpers: bot loading, instrumentation, raw-record access, statistics and rubric marks."""
import glob
import hashlib
import importlib.util
import json
import math
import os
import sys
import time

import config as C

if C.ROOT not in sys.path:
    sys.path.insert(0, C.ROOT)

BASELINES = ('static', 'random', 'attitude', 'greedy')


# ---------------------------------------------------------------- bot specs

def spec_path(spec):
    """Absolute file path for a bot spec, or None for a baseline name."""
    if spec in BASELINES:
        return None
    path = spec if os.path.isabs(spec) else os.path.join(C.ROOT, spec)
    if not os.path.exists(path):
        # allow a bare bot id such as bot_003 or bot_003_greedy_x
        hits = sorted(glob.glob(os.path.join(C.BOTS_DIR, os.path.basename(spec) + '*.py')))
        if len(hits) == 1:
            return hits[0]
        raise FileNotFoundError(f'bot spec not found: {spec}')
    return os.path.abspath(path)


def spec_label(spec, overrides=None):
    """Stable display name: bot file stem, 'baseline:<name>', plus '[K=V,..]' when overridden."""
    if spec in BASELINES:
        label = 'baseline:' + spec
    else:
        label = os.path.splitext(os.path.basename(spec_path(spec)))[0]
    if overrides:
        label += '[' + ','.join(f'{k}={v}' for k, v in sorted(overrides.items())) + ']'
    return label


def spec_hash(spec):
    path = spec_path(spec) or os.path.join(C.ROOT, 'agent_baselines.py')
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


_CLASS_CACHE = {}


def load_agent_class(spec, overrides=None):
    """Return an Agent class for a spec. Bot files are imported under a unique module name."""
    key = (spec, json.dumps(overrides or {}, sort_keys=True))
    if key in _CLASS_CACHE:
        return _CLASS_CACHE[key]
    if spec in BASELINES:
        import agent_baselines as B
        cls = {'static': B.StaticAgent, 'random': B.RandomAgent,
               'attitude': B.AttitudeAgent, 'greedy': B.GreedyAgent}[spec]
    else:
        path = spec_path(spec)
        name = 'labbot_' + hashlib.sha1((path + key[1]).encode()).hexdigest()[:12]
        mod_spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(mod_spec)
        sys.modules[name] = mod
        mod_spec.loader.exec_module(mod)
        if overrides:
            cfg = getattr(mod, 'CONFIG', None)
            if not isinstance(cfg, dict):
                raise ValueError(f'{spec} has no CONFIG dict; cannot apply overrides')
            for k, v in overrides.items():
                if k not in cfg:
                    raise KeyError(f'{spec}: unknown CONFIG key {k}')
                cfg[k] = v
        cls = getattr(mod, C.AGENT_CLASS)
    _CLASS_CACHE[key] = cls
    return cls


# ---------------------------------------------------------------- instrumentation

class InstrumentedAgent:
    """Proxy around a real agent. Times every call, enforces the 1 s limit the way marking does
    (timeout_decorator), records exceptions, and checks orders against the true game.
    The game loop itself is the unmodified game.run_one_game."""

    def __init__(self, cls, true_game):
        import timeout_decorator
        self._timeout = timeout_decorator.timeout(C.TIME_LIMIT)
        self.true_game = true_game
        self.power_name = None
        self.stats = {'t_init': 0.0, 't_new_game': 0.0, 'move_times': [], 't_update_max': 0.0,
                      'n_timeouts': 0, 'n_exceptions': 0, 'exc_samples': [], 'n_illegal': 0,
                      'illegal_samples': [], 'n_desync': 0, 'phase_times': {}}
        t = time.perf_counter()
        try:
            self.agent = self._timeout(cls)()
        except Exception as e:  # noqa: BLE001
            self._exc('init', e)
            self.agent = None
        self.stats['t_init'] = time.perf_counter() - t
        if self.stats['t_init'] > C.TIME_LIMIT:
            self.stats['n_timeouts'] += 1

    def _exc(self, where, e):
        self.stats['n_exceptions'] += 1
        if len(self.stats['exc_samples']) < 3:
            self.stats['exc_samples'].append(f'{where}: {type(e).__name__}: {str(e)[:200]}')

    def new_game(self, game, power_name):
        self.power_name = power_name
        if self.agent is None:
            return
        t = time.perf_counter()
        try:
            self._timeout(self.agent.new_game)(game, power_name)
        except Exception as e:  # noqa: BLE001
            self._exc('new_game', e)
        self.stats['t_new_game'] = time.perf_counter() - t
        if self.stats['t_new_game'] > C.TIME_LIMIT:
            self.stats['n_timeouts'] += 1

    def get_actions(self):
        if self.agent is None:
            return []
        phase = self.true_game.get_current_phase()
        t = time.perf_counter()
        try:
            orders = self._timeout(self.agent.get_actions)()
        except Exception as e:  # noqa: BLE001  (timeout_decorator.TimeoutError is an Exception)
            self._exc('get_actions ' + phase, e)
            orders = []
        dt = time.perf_counter() - t
        self.stats['move_times'].append(dt)
        pt = phase[-1]
        self.stats['phase_times'][pt] = max(self.stats['phase_times'].get(pt, 0.0), dt)
        if dt > C.TIME_LIMIT:
            self.stats['n_timeouts'] += 1
            orders = []
        if not isinstance(orders, list):
            self._exc('get_actions ' + phase, TypeError(f'returned {type(orders).__name__}'))
            orders = []
        self._check(orders, phase)
        return orders

    def _check(self, orders, phase):
        """Legality check against the true game (outside the timed region)."""
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
                self.stats['n_illegal'] += 1
                if len(self.stats['illegal_samples']) < 5:
                    self.stats['illegal_samples'].append(f'{phase}: {o}')
        # internal state must mirror the real game
        try:
            ag = self.agent.game
            if ag.get_current_phase() != phase or ag.get_units() != g.get_units():
                self.stats['n_desync'] += 1
        except Exception:  # noqa: BLE001
            self.stats['n_desync'] += 1

    def update_game(self, all_power_orders):
        if self.agent is None:
            return
        t = time.perf_counter()
        try:
            self._timeout(self.agent.update_game)(all_power_orders)
        except Exception as e:  # noqa: BLE001
            self._exc('update_game', e)
        dt = time.perf_counter() - t
        self.stats['t_update_max'] = max(self.stats['t_update_max'], dt)
        if dt > C.TIME_LIMIT:
            self.stats['n_timeouts'] += 1

    def summary(self):
        s = self.stats
        mt = sorted(s['move_times'])
        out = {k: v for k, v in s.items() if k != 'move_times'}
        out['n_moves'] = len(mt)
        out['t_max'] = round(mt[-1], 4) if mt else 0.0
        out['t_mean'] = round(sum(mt) / len(mt), 4) if mt else 0.0
        out['n_over_target'] = sum(1 for x in mt if x > C.TIME_TARGET)
        out['n_over_p99gate'] = sum(1 for x in mt if x > C.TIME_P99_GATE)
        for k in ('t_init', 't_new_game', 't_update_max'):
            out[k] = round(out[k], 4)
        out['phase_times'] = {k: round(v, 4) for k, v in s['phase_times'].items()}
        return out


# ---------------------------------------------------------------- raw records

def iter_raw(kind='game'):
    for path in sorted(glob.glob(os.path.join(C.RAW_DIR, '*.jsonl'))):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get('kind', 'game') == kind:
                    yield rec


def current_hash_for_label(label):
    """Hash of the file behind a label now (None if unknown). Used to ignore results of edited bots."""
    base = label.split('[', 1)[0]
    if base.startswith('baseline:'):
        return spec_hash(base.split(':', 1)[1])
    for cand in (os.path.join(C.BOTS_DIR, base + '.py'), os.path.join(C.ROOT, base + '.py')):
        if os.path.exists(cand):
            return spec_hash(cand)
    return None


def records_by_label(scenarios=(1, 2, 3), seedset=None, current_only=True):
    out = {}
    hashes = {}
    for r in iter_raw('game'):
        if r.get('status') != 'ok' or r['scenario'] not in scenarios:
            continue
        if seedset and r['seedset'] != seedset:
            continue
        lab = r['bot']
        if current_only:
            if lab not in hashes:
                hashes[lab] = current_hash_for_label(lab)
            if hashes[lab] and r['bot_hash'] != hashes[lab]:
                continue
        out.setdefault(lab, []).append(r)
    return out


# ---------------------------------------------------------------- statistics

def mean_se(xs):
    n = len(xs)
    if n == 0:
        return float('nan'), float('nan')
    m = sum(xs) / n
    if n < 2:
        return m, float('nan')
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n)


def scenario_stats(recs):
    scs = [r['sc'] for r in recs]
    wins = [1.0 if r['win'] else 0.0 for r in recs]
    m, se = mean_se(scs)
    w, wse = mean_se(wins)
    return {'n': len(recs), 'mean_sc': m, 'se_sc': se, 'win': w, 'se_win': wse}


def rubric_mark(scenario, st):
    """(points, borderline flag) for one scenario from win rate or mean SC (whichever is higher)."""
    pts = 0
    border = False
    if st['n'] == 0:
        return 0, False
    for p, wt, st_thr in C.RUBRIC[scenario]:
        if st['win'] > wt or st['mean_sc'] > st_thr:
            pts = p
        se_sc = st['se_sc'] if st['se_sc'] == st['se_sc'] else 0.0
        se_w = st['se_win'] if st['se_win'] == st['se_win'] else 0.0
        if abs(st['mean_sc'] - st_thr) <= se_sc or (abs(st['win'] - wt) <= se_w and se_w > 0):
            border = True
    return pts, border


def fmt(x, nd=2):
    if x is None or x != x:
        return '-'
    return f'{x:.{nd}f}'
