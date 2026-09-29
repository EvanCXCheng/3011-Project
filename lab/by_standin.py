"""Paired S3 differences split by Hidden Agent stand-in (lab tool).

  .venv/bin/python lab/by_standin.py --a bot_069 --b bot_045 [--seedset B]
"""
import argparse
import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True)
    ap.add_argument('--b', required=True)
    ap.add_argument('--seedset', default='B')
    ap.add_argument('--scenario', type=int, default=3)
    a = ap.parse_args()
    recs = {}
    for path in glob.glob(os.path.join(ROOT, 'results/raw/*.jsonl')):
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if d.get('kind') != 'game' or d.get('status') != 'ok' or d.get('seedset') != a.seedset:
                    continue
                if d.get('scenario') != a.scenario:
                    continue
                for tag in (a.a, a.b):
                    if (d.get('bot') or '').startswith(tag):
                        recs[(tag, d.get('standin'), d['seed'])] = d['sc']
    standins = sorted({k[1] for k in recs if k[0] == a.a}, key=str)
    for si in standins:
        diffs = [recs[(a.a, si, s)] - recs[(a.b, si, s)] for (t, x, s) in recs
                 if t == a.a and x == si and (a.b, si, s) in recs]
        n = len(diffs)
        if n < 2:
            continue
        m = sum(diffs) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in diffs) / (n - 1))
        ma = sum(recs[(a.a, si, s)] for (t, x, s) in recs if t == a.a and x == si and (a.b, si, s) in recs) / n
        print(f'S{a.scenario} stand-in {si}: n={n} {a.a} {ma:.2f} diff={m:+.2f}±{sd / math.sqrt(n):.2f}')


if __name__ == '__main__':
    main()
