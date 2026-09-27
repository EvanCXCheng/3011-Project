"""Failure analysis for one bot: where does it lose SCs?

  python lab/analyze.py --bot bot_007 [--seedset B] [--scenarios 2 3]
Prints: mean SC by power per scenario; mean SC trajectory by year; results by number of Greedy
opponents; who ends strongest in our worst games; worst seeds (for replay with run_game.py --save).
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import common  # noqa: E402
import compare  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bot', required=True)
    ap.add_argument('--seedset', default=None)
    ap.add_argument('--scenarios', type=int, nargs='+', default=[1, 2, 3])
    a = ap.parse_args()
    label = compare.label_of(a.bot)
    recs = common.records_by_label(scenarios=tuple(a.scenarios), seedset=a.seedset).get(label, [])
    recs = [r for r in recs if r['seedset'] not in ('X', 'T')] if not a.seedset else recs
    if not recs:
        sys.exit(f'no records for {label}')
    print(f'{label}: {len(recs)} games')
    for sc in a.scenarios:
        rs = [r for r in recs if r['scenario'] == sc]
        if not rs:
            continue
        print(f'S{sc} (n={len(rs)})')
        cells = []
        for p in C.POWERS:
            st = common.scenario_stats([r for r in rs if r['power'] == p])
            cells.append(f"{p[:3]} {common.fmt(st['mean_sc'], 1)}/{st['win']*100:.0f}%")
        print('  by power (sc/win): ' + '  '.join(cells))
        years = {}
        for r in rs:
            idx = C.POWERS.index(r['power'])
            tr = dict((y, c[idx]) for y, c in r.get('traj', []))
            last = None
            for y in range(1901, C.END_YEAR + 1):   # carry the final count past an early game end
                last = tr.get(y, last)
                if last is not None:
                    years.setdefault(y, []).append(last)
        traj = ' '.join(f'{y % 100:02d}:{sum(v)/len(v):.1f}' for y, v in sorted(years.items()))
        print(f'  mean SC by year: {traj}')
        if sc > 1:
            by_g = {}
            for r in rs:
                g = sum(1 for v in r['opponents'].values() if v == 'greedy')
                by_g.setdefault(g, []).append(r['sc'])
            print('  by #greedy opponents: ' + '  '.join(
                f'{g}:{sum(v)/len(v):.1f}(n={len(v)})' for g, v in sorted(by_g.items())))
            lost = [r for r in rs if r.get('traj') and max(c[C.POWERS.index(r['power'])] for _, c in r['traj'])
                    - r['sc'] >= 3]
            print(f'  games losing >=3 SC from peak: {len(lost)}/{len(rs)}')
            top = Counter()
            for r in sorted(rs, key=lambda r: r['sc'])[: max(1, len(rs) // 5)]:
                best = max((p for p in C.POWERS if p != r['power']), key=lambda p: r['final_sc'][p])
                top[r['opponents'][best]] += 1
            print(f'  strongest opponent type in our worst 20%: {dict(top)}')
        worst = sorted(rs, key=lambda r: r['sc'])[:3]
        print('  worst seeds: ' + ', '.join(f"{r['seed']}({r['power'][:3]} {r['sc']}sc)" for r in worst))


if __name__ == '__main__':
    main()
