"""Tournament (Scenario 4 stand-in): 7-seat games of bots in rotating seats, ranked by mean final SCs.

  python lab/tournament.py --bots hof --games 28          # hall of fame, one bot per family preferred
  python lab/tournament.py --bots bots/a.py bots/b.py greedy --games 28
With fewer than 7 bots, seats are filled by cycling the list (duplicates play each other).
Game g puts bot (i + g) mod k in seat i, so every bot visits every seat.
"""
import argparse
import json
import os
import sys
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import common  # noqa: E402
import run_game  # noqa: E402
import state as S  # noqa: E402


def hof_specs():
    st = S.load()
    specs, fams = [], set()
    for bid in S.hall_of_fame(st):
        fam = st['bots'][bid]['family']
        if fam in fams:
            continue
        fams.add(fam)
        specs.append(st['bots'][bid]['file'])
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bots', nargs='+', required=True)
    ap.add_argument('--games', type=int, default=28)
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--workers', type=int, default=C.WORKERS)
    a = ap.parse_args()
    specs = hof_specs() if a.bots == ['hof'] else a.bots
    if not specs:
        sys.exit('no bots')
    labels = [common.spec_label(s) for s in specs]
    k = len(specs)
    run_id = datetime.now().strftime('%Y%m%d-%H%M%S') + '_tournament'
    tasks = []
    for g in range(a.games):
        seed = C.SEED_SETS['T'] + a.offset + g
        seats = {p: specs[(i + g) % k] for i, p in enumerate(C.POWERS)}
        meta = {'kind': 'tournament', 'run_id': run_id, 'seed': seed,
                'seat_labels': {p: common.spec_label(s) for p, s in seats.items()},
                'seat_hashes': {p: common.spec_hash(s) for p, s in seats.items()}}
        instrument = [p for p, s in seats.items() if s not in common.BASELINES]
        tasks.append({'kind': 'tournament', 'seats': seats, 'seed': seed, 'instrument': instrument, 'meta': meta})
    raw_path = os.path.join(C.RAW_DIR, run_id + '.jsonl')
    results = []
    with open(raw_path, 'w') as raw, ProcessPoolExecutor(
            max_workers=a.workers, mp_context=mp.get_context('spawn'), initializer=run_game.worker_init,
            max_tasks_per_child=C.MAX_TASKS_PER_CHILD) as ex:
        for fu in as_completed([ex.submit(run_game.run_task, t) for t in tasks]):
            rec = fu.result()
            if rec.get('status') == 'ok':
                rec['agents'] = {p: {kk: v for kk, v in s.items() if kk in ('t_max', 'n_exceptions', 'n_illegal',
                                                                           'n_timeouts')}
                                 for p, s in rec['agents'].items()}
            raw.write(json.dumps(rec) + '\n')
            results.append(rec)
    agg = {lab: [] for lab in labels}
    problems = {lab: 0 for lab in labels}
    for r in results:
        if r.get('status') != 'ok':
            continue
        for p, lab in r['seat_labels'].items():
            agg[lab].append(min(r['final_sc'][p], C.WIN_SC))
            s = r['agents'].get(p, {})
            problems[lab] += (s.get('n_exceptions') or 0) + (s.get('n_illegal') or 0) + (s.get('n_timeouts') or 0)
    ranked = sorted(((common.mean_se(v), lab) for lab, v in agg.items() if v), key=lambda x: -x[0][0])
    print(f'tournament {run_id}: {len(results)} games, {k} bots')
    for i, ((m, se), lab) in enumerate(ranked, 1):
        print(f'  {i}. {lab}: {m:.2f}±{common.fmt(se)} SC over {len(agg[lab])} seats, issues={problems[lab]}')


if __name__ == '__main__':
    main()
