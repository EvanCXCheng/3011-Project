"""Paired comparison of a candidate against a reference bot on shared seeds.

  python lab/compare.py --a bot_007 --b bot_004 --seedset B [--scenarios 1 2 3]

Pairs games by (scenario, seed, and stand-in for S3). Prints per-scenario mean-SC difference ± SE,
the pooled paired difference, estimated marks, and the promotion verdict:
PROMOTE iff mark(a) >= mark(b) and pooled mean-SC gain > 2 SE.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def label_of(spec):
    if spec in common.BASELINES or spec.startswith('baseline:'):
        return spec if spec.startswith('baseline:') else 'baseline:' + spec
    if '[' in spec:
        return spec
    try:
        return common.spec_label(spec)
    except FileNotFoundError:
        return spec


def compare(a, b, seedset, scenarios=(1, 2, 3), quiet=False):
    la, lb = label_of(a), label_of(b)
    recs = common.records_by_label(scenarios=tuple(scenarios), seedset=seedset)
    ka = {(r['scenario'], r['seed'], r.get('standin')): r for r in recs.get(la, [])}
    kb = {(r['scenario'], r['seed'], r.get('standin')): r for r in recs.get(lb, [])}
    shared = sorted(set(ka) & set(kb))
    lines = [f'{la} vs {lb} on seedset {seedset}: {len(shared)} paired games']
    pooled = []
    marks = {la: 0, lb: 0}
    per = {}
    for sc in scenarios:
        keys = [k for k in shared if k[0] == sc]
        if not keys:
            lines.append(f'  S{sc}: no paired games')
            continue
        diffs = [ka[k]['sc'] - kb[k]['sc'] for k in keys]
        pooled += diffs
        m, se = common.mean_se(diffs)
        sa = common.scenario_stats([ka[k] for k in keys])
        sb = common.scenario_stats([kb[k] for k in keys])
        ma, ba = common.rubric_mark(sc, sa)
        mb, bb = common.rubric_mark(sc, sb)
        marks[la] += ma
        marks[lb] += mb
        per[sc] = {'diff': m, 'se': se, 'a': sa, 'b': sb, 'mark_a': ma, 'mark_b': mb}
        lines.append(f"  S{sc} n={len(keys)}: a sc={common.fmt(sa['mean_sc'])} win={sa['win']*100:.1f}% mark={ma}"
                     f"{'*' if ba else ''} | b sc={common.fmt(sb['mean_sc'])} win={sb['win']*100:.1f}% mark={mb}"
                     f"{'*' if bb else ''} | diff={common.fmt(m)}±{common.fmt(se)}")
    m, se = common.mean_se(pooled)
    promote = bool(pooled) and marks[la] >= marks[lb] and se == se and m > 2 * se
    lines.append(f'  pooled diff={common.fmt(m)}±{common.fmt(se)}  marks a={marks[la]} b={marks[lb]}  '
                 f'verdict: {"PROMOTE" if promote else "REJECT"}   (* = borderline)')
    if not quiet:
        print('\n'.join(lines))
    return {'promote': promote, 'diff': m, 'se': se, 'marks': marks, 'per': per, 'n': len(shared)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--seedset', default='B')
    ap.add_argument('--scenarios', type=int, nargs='+', default=[1, 2, 3])
    x = ap.parse_args()
    compare(x.a, x.b, x.seedset, x.scenarios)


if __name__ == '__main__':
    main()
