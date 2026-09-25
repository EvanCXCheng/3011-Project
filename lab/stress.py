"""Slow-machine timing stress test: several games pinned to ONE core so every game runs ~K times slower.

  python lab/stress.py --bot bots/bot_004_lookahead_base.py [--procs 4] [--n 4] [--core 0] [--gate 0.8]

Models a marking machine with a slower CPU (or one running several games at once). Bots stop searching on
wall-clock time, so a slow CPU mostly lowers search quality. The risk is the work done outside the budget check
(order generation, the simulation in flight when the budget expires, final selection), which scales with CPU
speed. The script reports:
  slowdown   measured by a fixed CPU loop run K-at-once on the core vs alone (≈ K)
  tmax       slowest move over all games (verdict: PASS iff tmax < gate and no timeouts or exceptions)
  overshoot  tmax(movement) - the bot's CONFIG['TIME_BUDGET']
Games use seed set X and are NOT written to results/raw (so they never enter evaluation statistics).
Summary goes to results/logs/stress_<bot>.json. Run it only when no evaluation is running (it shares the CPU).
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import common  # noqa: E402
import run_game  # noqa: E402


def _calib(_=None):
    t = time.perf_counter()
    s = 0
    for i in range(3_000_000):
        s += i * i
    return time.perf_counter() - t


def _game(args):
    bot, scenario, seed, standin = args
    rec = run_game.finish_game_record(run_game.run_task(
        run_game.scenario_task(bot, scenario, seed, 'X', standin, None)))
    keep = ('status', 'error', 'scenario', 'seed', 'power', 'sc', 't_max', 't_mean', 'n_moves', 'n_timeouts',
            'n_exceptions', 'n_illegal', 'n_over_p99gate', 'phase_times', 't_new_game', 'wall')
    return {k: rec.get(k) for k in keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bot', required=True)
    ap.add_argument('--procs', type=int, default=4, help='concurrent games on the one core (≈ slowdown factor)')
    ap.add_argument('--n', type=int, default=4, help='games per scenario')
    ap.add_argument('--core', type=int, default=0)
    ap.add_argument('--gate', type=float, default=0.8, help='PASS iff slowest move < gate (s)')
    ap.add_argument('--standin', default='greedy')
    a = ap.parse_args()

    os.sched_setaffinity(0, {a.core})          # children inherit the single-core affinity
    solo = min(_calib() for _ in range(3))
    with ProcessPoolExecutor(max_workers=a.procs, initializer=run_game.worker_init) as ex:
        contended = sorted(ex.map(_calib, range(a.procs)))
        slowdown = contended[len(contended) // 2] / solo
        tasks = [(a.bot, sc, C.SEED_SETS['X'] + 50 + i, a.standin) for sc in (1, 2, 3) for i in range(a.n)]
        t0 = time.perf_counter()
        recs = list(ex.map(_game, tasks))
        wall = time.perf_counter() - t0

    budget = None
    try:
        spec = importlib.util.spec_from_file_location('stress_bot', common.spec_path(a.bot))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        budget = getattr(mod, 'CONFIG', {}).get('TIME_BUDGET')
    except Exception:  # noqa: BLE001
        pass

    ok = [r for r in recs if r['status'] == 'ok']
    tmax = max((r['t_max'] or 0.0) for r in ok) if ok else None
    tmax_m = max((r['phase_times'] or {}).get('M', 0.0) for r in ok) if ok else None
    bad = sum((r['n_timeouts'] or 0) + (r['n_exceptions'] or 0) for r in ok) + (len(recs) - len(ok))
    verdict = 'PASS' if ok and tmax < a.gate and bad == 0 else 'FAIL'
    out = {
        'bot': common.spec_label(a.bot), 'bot_hash': common.spec_hash(a.bot), 'procs': a.procs,
        'slowdown': round(slowdown, 2), 'gate': a.gate, 'budget': budget, 'games': len(recs),
        'tmax': tmax, 'tmax_movement': tmax_m,
        'overshoot': round(tmax_m - budget, 4) if (budget is not None and tmax_m is not None) else None,
        'mean_move': round(sum(r['t_mean'] or 0 for r in ok) / max(1, len(ok)), 4),
        'n_over_0.9': sum(r['n_over_p99gate'] or 0 for r in ok), 'errors': bad, 'verdict': verdict,
        'wall_s': round(wall, 1), 'games_detail': recs,
    }
    os.makedirs(C.LOG_DIR, exist_ok=True)
    path = os.path.join(C.LOG_DIR, f"stress_{out['bot']}.json")
    with open(path, 'w') as f:
        json.dump(out, f, indent=1)
    print(f"{out['bot']} stress: {a.procs} games on core {a.core}, slowdown x{out['slowdown']} | "
          f"tmax {tmax} s (movement {tmax_m}, budget {budget}, overshoot {out['overshoot']}) | "
          f"moves>0.9s {out['n_over_0.9']} | errors {bad} | {verdict} (gate {a.gate} s) | {wall:.0f}s")


if __name__ == '__main__':
    main()
