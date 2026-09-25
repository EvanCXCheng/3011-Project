"""Evaluate a bot (or baseline) on scenarios 1-3 over a seed set, in parallel.

  python lab/evaluate.py --bot bots/bot_001_greedy_base.py --scenarios 1 2 3 --seedset A --n 42 --tag tier1
  options: --standin auto|greedy|<bot path>   (S3 hidden-agent stand-in; default auto = CLAUDE.md rule)
           --set KEY=VAL ...                   (override the bot's CONFIG dict, for ablations)
           --workers N  --offset I  --no-reuse

Writes results/raw/<run_id>.jsonl (new games only), appends rows to results/results.csv,
logs progress to results/logs/<run_id>.log, and prints a short summary.
Games already recorded for the same (bot, hash, overrides, scenario, seedset, seed, stand-in) are reused.
"""
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
import multiprocessing as mp

for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import common  # noqa: E402
import run_game  # noqa: E402

CSV_FIELDS = ['run_id', 'date', 'bot', 'bot_hash', 'tag', 'seedset', 'scenario', 'standin', 'n', 'n_errors',
              'win_rate', 'mean_sc', 'se_sc', 'mark', 'borderline', 't_max', 'n_over_p99gate', 'maxrss_mb',
              'n_exceptions', 'n_illegal', 'n_timeouts', 'n_desync']


def reuse_key(r):
    return (r['bot'], r['bot_hash'], r['scenario'], r['seedset'], r['seed'],
            r.get('standin') if r['scenario'] == 3 else None,
            r.get('standin_hash') if r['scenario'] == 3 else None)


def run_tasks(tasks, workers, log, label):
    """Run tasks in a spawn pool; recreate the pool if a worker dies. Yields finished records."""
    pending = list(tasks)
    crashes = 0
    done_n = 0
    t0 = time.time()
    while pending:
        if workers <= 1:
            run_game.worker_init()
            for t in pending:
                yield run_game.run_task(t)
                done_n += 1
                log.write(f'{done_n}/{len(tasks)} {time.time()-t0:.0f}s\n')
                log.flush()
            return
        ctx = mp.get_context('spawn')
        finished = set()
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=run_game.worker_init,
                                     max_tasks_per_child=C.MAX_TASKS_PER_CHILD) as ex:
                futs = {ex.submit(run_game.run_task, t): i for i, t in enumerate(pending)}
                for fu in as_completed(futs):
                    rec = fu.result()
                    finished.add(futs[fu])
                    done_n += 1
                    if done_n % 10 == 0 or done_n == len(tasks):
                        log.write(f'{label} {done_n}/{len(tasks)} {time.time()-t0:.0f}s\n')
                        log.flush()
                    yield rec
                pending = []
        except BrokenProcessPool:
            crashes += 1
            pending = [t for i, t in enumerate(pending) if i not in finished]
            log.write(f'worker crash #{crashes}; {len(pending)} tasks left\n')
            log.flush()
            if crashes >= 3:
                for t in pending:
                    out = dict(t['meta'])
                    out.update(status='error', error='worker process crashed (memory?)')
                    yield out
                return


def summarize(recs, scenario):
    ok = [r for r in recs if r.get('status') == 'ok']
    st = common.scenario_stats(ok)
    mark, border = common.rubric_mark(scenario, st)
    agg = lambda k, f=sum: f([r.get(k) or 0 for r in ok]) if ok else 0  # noqa: E731
    return {
        'n': len(ok), 'n_errors': len(recs) - len(ok), 'win_rate': st['win'], 'mean_sc': st['mean_sc'],
        'se_sc': st['se_sc'], 'mark': mark, 'borderline': border,
        't_max': agg('t_max', max), 'n_over_p99gate': agg('n_over_p99gate'),
        'maxrss_mb': max([r.get('maxrss_mb') or 0 for r in recs] or [0]),
        'n_exceptions': agg('n_exceptions'), 'n_illegal': agg('n_illegal'),
        'n_timeouts': agg('n_timeouts'), 'n_desync': agg('n_desync'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bot', required=True)
    ap.add_argument('--scenarios', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--seedset', default='A')
    ap.add_argument('--n', type=int, default=C.TIER1_N)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--standin', default='auto')
    ap.add_argument('--tag', default='')
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--workers', type=int, default=C.WORKERS)
    ap.add_argument('--no-reuse', action='store_true')
    a = ap.parse_args()

    overrides = common.parse_overrides(a.set)
    label = common.spec_label(a.bot, overrides)
    bhash = common.spec_hash(a.bot)
    standin = None
    if 3 in a.scenarios:
        if a.standin == 'auto':
            import state as S
            standin = S.choose_standin(S.load(), label)
        else:
            standin = a.standin
        if standin not in common.BASELINES:
            standin = os.path.relpath(common.spec_path(standin), C.ROOT)

    run_id = datetime.now().strftime('%Y%m%d-%H%M%S') + f'_{label.split("[")[0].replace(":", "-")}_{a.tag or "eval"}'
    os.makedirs(C.RAW_DIR, exist_ok=True)
    os.makedirs(C.LOG_DIR, exist_ok=True)
    base = C.SEED_SETS[a.seedset]
    seeds = [base + a.offset + i for i in range(a.n)]

    existing = {}
    if not a.no_reuse:
        for r in common.iter_raw('game'):
            if r.get('status') == 'ok' and r['bot'] == label and r['bot_hash'] == bhash:
                existing[reuse_key(r)] = r

    tasks, reused = [], {s: [] for s in a.scenarios}
    for sc in a.scenarios:
        for seed in seeds:
            t = run_game.scenario_task(a.bot, sc, seed, a.seedset, standin if sc == 3 else None,
                                       overrides, a.tag, run_id)
            k = reuse_key(t['meta'])
            if k in existing:
                reused[sc].append(existing[k])
            else:
                tasks.append(t)

    raw_path = os.path.join(C.RAW_DIR, run_id + '.jsonl')
    log_path = os.path.join(C.LOG_DIR, run_id + '.log')
    marker = os.path.join(C.LOG_DIR, run_id + '.running')
    with open(marker, 'w') as f:
        f.write(f'{os.getpid()}\n{" ".join(sys.argv)}\n')
    new = {s: [] for s in a.scenarios}
    t0 = time.time()
    try:
        with open(log_path, 'w') as log:
            log.write(f'{label} {len(tasks)} new games, {sum(map(len, reused.values()))} reused\n')
            if tasks:
                with open(raw_path, 'a') as raw:
                    for rec in run_tasks(tasks, a.workers, log, label):
                        rec = run_game.finish_game_record(rec)
                        raw.write(json.dumps(rec) + '\n')
                        raw.flush()
                        new[rec['scenario']].append(rec)
    finally:
        if os.path.exists(marker):
            os.remove(marker)

    write_header = not os.path.exists(C.RESULTS_CSV)
    lines = [f'{label} seedset={a.seedset} n={a.n} new={len(tasks)} ({time.time()-t0:.0f}s) '
             f'standin={common.spec_label(standin) if standin else "-"}']
    with open(C.RESULTS_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            w.writeheader()
        for sc in a.scenarios:
            recs = reused[sc] + new[sc]
            s = summarize(recs, sc)
            row = {'run_id': run_id, 'date': datetime.now().isoformat(timespec='minutes'), 'bot': label,
                   'bot_hash': bhash, 'tag': a.tag, 'seedset': a.seedset, 'scenario': sc,
                   'standin': common.spec_label(standin) if (sc == 3 and standin) else ''}
            row.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
            w.writerow(row)
            flags = []
            if s['n_errors']: flags.append(f"ERRORS={s['n_errors']}")
            if s['n_exceptions']: flags.append(f"exc={s['n_exceptions']}")
            if s['n_illegal']: flags.append(f"illegal={s['n_illegal']}")
            if s['n_timeouts']: flags.append(f"TIMEOUTS={s['n_timeouts']}")
            if s['n_desync']: flags.append(f"desync={s['n_desync']}")
            if s['n_over_p99gate']: flags.append(f"slow>{C.TIME_P99_GATE}s={s['n_over_p99gate']}")
            lines.append(f"  S{sc}: n={s['n']} win={s['win_rate']*100:.1f}% sc={common.fmt(s['mean_sc'])}"
                         f"±{common.fmt(s['se_sc'])} mark={s['mark']}{'(borderline)' if s['borderline'] else ''}"
                         f" tmax={s['t_max']:.3f}s mem={s['maxrss_mb']:.0f}MB {' '.join(flags)}")
    errs = [r for sc in a.scenarios for r in new[sc] if r.get('status') != 'ok']
    if errs:
        lines.append(f"  first error: {errs[0].get('error')}")
    samples = [x for sc in a.scenarios for r in new[sc] for x in (r.get('exc_samples') or [])][:2]
    samples += [x for sc in a.scenarios for r in new[sc] for x in (r.get('illegal_samples') or [])][:2]
    for x in samples:
        lines.append('  sample: ' + x[:160])
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
