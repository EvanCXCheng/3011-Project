"""Play one instrumented game through the unmodified game.run_one_game.

CLI:  python lab/run_game.py --bot bots/bot_001_greedy_base.py --scenario 2 --seed 7 [--standin greedy]
"""
import argparse
import json
import os
import random
import resource
import signal
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import common  # noqa: E402


class GameCpuTimeout(BaseException):
    """Raised by the SIGPROF watchdog; BaseException so bots' `except Exception` cannot swallow it."""


def _on_sigprof(signum, frame):
    raise GameCpuTimeout()


def worker_init():
    """Per-process setup for pool workers: memory cap, single-threaded BLAS, repo on sys.path."""
    for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[k] = '1'
    if C.ROOT not in sys.path:
        sys.path.insert(0, C.ROOT)
    lim = C.MEM_LIMIT_AS_MB * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    except (ValueError, OSError):
        pass


def seat_plan(scenario, seed, standin=None):
    """Deterministic seat and opponents for (scenario, seed). Returns (our_power, {power: spec})."""
    power = C.POWERS[seed % 7]
    rng = random.Random(seed * 10 + scenario)
    others = [p for p in C.POWERS if p != power]
    opp = {}
    if scenario == 1:
        opp = {p: 'static' for p in others}
    else:
        opp = {p: rng.choice(C.S2_POOL) for p in others}
        if scenario == 3:
            hidden_seat = rng.choice(others)
            opp[hidden_seat] = standin or 'greedy'
    return power, opp


def play(seats, seed, instrument, overrides_by_power=None, save_file=None):
    """Play one game. seats: {power: spec}. instrument: powers to wrap. Returns a record dict."""
    from diplomacy import Game
    from game import run_one_game
    import numpy as np

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    overrides_by_power = overrides_by_power or {}
    true_game = Game()
    agents = {}
    wrapped = {}
    for p in C.POWERS:
        cls = common.load_agent_class(seats[p], overrides_by_power.get(p))
        if p in instrument:
            wrapped[p] = common.InstrumentedAgent(cls, true_game)
            agents[p] = wrapped[p]
        else:
            agents[p] = cls()

    traj = []                # SC counts of all powers, sampled at each spring movement phase
    last_year = [None]
    if wrapped:
        first = next(iter(wrapped.values()))
        real_get = first.get_actions

        def get_actions_with_traj():
            ph = true_game.get_current_phase()
            if ph.startswith('S') and ph.endswith('M') and ph[1:5] != last_year[0]:
                last_year[0] = ph[1:5]
                traj.append([int(ph[1:5]), [len(true_game.powers[q].centers) for q in C.POWERS]])
            return real_get()
        first.get_actions = get_actions_with_traj

    t0 = time.perf_counter()
    results, year = run_one_game(agents, game=true_game, end_year=C.END_YEAR, save_file=save_file)
    wall = time.perf_counter() - t0
    traj.append([int(year) if year else C.END_YEAR, [results[q] for q in C.POWERS]])
    rec = {
        'final_sc': results,
        'year': int(year) if year else C.END_YEAR,
        'traj': traj,
        'wall': round(wall, 2),
        'agents': {p: w.summary() for p, w in wrapped.items()},
    }
    return rec


def run_task(task):
    """Pool entry point. task keys: kind, seats, seed, instrument, overrides, meta."""
    old = signal.signal(signal.SIGPROF, _on_sigprof)
    signal.setitimer(signal.ITIMER_PROF, C.GAME_CPU_LIMIT)
    out = dict(task['meta'])
    try:
        ovr = {p: task.get('overrides') for p in task['instrument']} if task.get('overrides') else None
        rec = play(task['seats'], task['seed'], task['instrument'], ovr, task.get('save_file'))
        out.update(rec)
        out['status'] = 'ok'
    except GameCpuTimeout:
        out['status'] = 'error'
        out['error'] = 'game cpu timeout'
    except BaseException as e:  # noqa: BLE001
        out['status'] = 'error'
        out['error'] = f'{type(e).__name__}: {e}'[:300]
        out['trace'] = traceback.format_exc()[-1500:]
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGPROF, old)
    out['maxrss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    return out


def scenario_task(bot, scenario, seed, seedset, standin=None, overrides=None, tag='', run_id=''):
    power, opp = seat_plan(scenario, seed, standin)
    seats = dict(opp)
    seats[power] = bot
    meta = {
        'kind': 'game', 'run_id': run_id, 'tag': tag, 'bot': common.spec_label(bot, overrides),
        'bot_hash': common.spec_hash(bot), 'overrides': overrides or {}, 'scenario': scenario,
        'seedset': seedset, 'seed': seed, 'power': power,
        'opponents': {p: (common.spec_label(s) if s not in common.BASELINES else s) for p, s in opp.items()},
        'standin': common.spec_label(standin) if (scenario == 3 and standin) else None,
        'standin_hash': common.spec_hash(standin) if (scenario == 3 and standin) else None,
    }
    return {'kind': 'game', 'seats': seats, 'seed': seed, 'instrument': [power],
            'overrides': overrides, 'meta': meta}


def finish_game_record(rec):
    """Add our-seat convenience fields (sc capped at 18, win) to a finished scenario record."""
    if rec.get('status') == 'ok':
        p = rec['power']
        rec['sc'] = min(rec['final_sc'][p], C.WIN_SC)
        rec['win'] = rec['final_sc'][p] >= C.WIN_SC
        a = rec['agents'].get(p, {})
        for k in ('t_max', 't_mean', 'n_moves', 'n_over_target', 'n_over_p99gate', 'n_timeouts',
                  'n_exceptions', 'n_illegal', 'n_desync', 't_new_game', 't_init', 't_update_max'):
            rec[k] = a.get(k)
        rec['exc_samples'] = a.get('exc_samples', [])
        rec['illegal_samples'] = a.get('illegal_samples', [])
        rec['phase_times'] = a.get('phase_times', {})
        del rec['agents']
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bot', required=True)
    ap.add_argument('--scenario', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--standin', default=None)
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--save', default=None, help='save game JSON for the web visualiser (lab output only)')
    a = ap.parse_args()
    worker_init()
    ovr = common.parse_overrides(a.set)
    task = scenario_task(a.bot, a.scenario, a.seed, 'X', a.standin, ovr)
    if a.save:
        task['save_file'] = a.save
    rec = finish_game_record(run_task(task))
    rec.pop('traj', None)
    print(json.dumps(rec, indent=1)[:3000])


if __name__ == '__main__':
    main()
