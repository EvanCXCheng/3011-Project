"""Regenerate LEADERBOARD.md from results/raw and results/state.json. Never hand-edit LEADERBOARD.md.

  python lab/leaderboard.py
Stats pool all seed sets except X (smoke/debug) and T (tournament). Only results for the current
file hash of each bot are used.
"""
import hashlib
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import common  # noqa: E402
import state as S  # noqa: E402

EXCLUDE_SEEDSETS = {'X', 'T'}


def file_sha(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.sha1(f.read()).hexdigest()[:10]
    except OSError:
        return None


def bot_rows(st):
    recs = common.records_by_label()
    rows = []
    for label, rs in recs.items():
        rs = [r for r in rs if r['seedset'] not in EXCLUDE_SEEDSETS]
        if not rs:
            continue
        row = {'label': label, 'per': {}, 'mark': 0, 'border': [], 'sc_sum': 0.0}
        for sc in (1, 2, 3):
            sub = [r for r in rs if r['scenario'] == sc]
            stt = common.scenario_stats(sub)
            m, b = common.rubric_mark(sc, stt)
            row['per'][sc] = (stt, m, b)
            row['mark'] += m
            if b:
                row['border'].append(sc)
            if stt['n']:
                row['sc_sum'] += stt['mean_sc']
        row['tmax'] = max((r.get('t_max') or 0) for r in rs)
        row['mem'] = max((r.get('maxrss_mb') or 0) for r in rs)
        row['problems'] = sum((r.get('n_exceptions') or 0) + (r.get('n_illegal') or 0) +
                              (r.get('n_timeouts') or 0) for r in rs)
        row['s3_standins'] = sorted({r.get('standin') or '-' for r in rs if r['scenario'] == 3})
        bid = S.bot_id(label)
        info = st['bots'].get(bid, {})
        row['family'] = info.get('family', '-')
        row['status'] = info.get('status', '-')
        rows.append(row)
    rows.sort(key=lambda r: (-r['mark'], -r['sc_sum'], r['tmax']))
    return rows


def tournament_table():
    agg = {}
    for r in common.iter_raw('tournament'):
        if r.get('status') != 'ok':
            continue
        for p, lab in r['seat_labels'].items():
            agg.setdefault(lab, []).append(min(r['final_sc'][p], C.WIN_SC))
    ranked = sorted(((sum(v) / len(v), len(v), lab) for lab, v in agg.items()), reverse=True)
    return ranked


def main():
    st = S.load()
    rows = bot_rows(st)
    tour = tournament_table()
    trank = {lab: i + 1 for i, (_, _, lab) in enumerate(tour)}
    now = datetime.now(C.AWST)
    ch = S.overall_champion(st)
    ch_file = os.path.join(C.ROOT, st['bots'][ch]['file']) if ch else None
    match = (file_sha(ch_file) == file_sha(C.AGENT_FILE)) if ch else None
    out = ['# Leaderboard', '',
           f'_Generated {now:%Y-%m-%d %H:%M} AWST by lab/leaderboard.py — do not hand-edit._  ',
           f'Freeze in {(C.FREEZE - now).total_seconds() / 3600:.1f} h, deadline in '
           f'{(C.DEADLINE - now).total_seconds() / 3600:.1f} h.', '',
           f'**Overall champion:** {ch or "none"}  ' +
           (f'(agent_{C.GROUP}.py {"matches" if match else "DOES NOT MATCH"})' if ch else ''), '',
           '## Bots', '',
           'Mark = estimated rubric points /15 (per scenario: highest tier reached by win rate or mean SC). '
           '`*` = within 1 SE of a threshold. Pooled over seed sets A/B/C/FINAL, current file hash only.', '',
           '| # | bot | family | status | S1 sc (win%) n | S2 sc (win%) n | S3 sc (win%) n | mark | tmax s | mem MB '
           '| issues | tourn |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for i, r in enumerate(rows, 1):
        cells = []
        for sc in (1, 2, 3):
            stt, m, b = r['per'][sc]
            if stt['n']:
                cells.append(f"{stt['mean_sc']:.2f}±{common.fmt(stt['se_sc'])} ({stt['win']*100:.0f}%) "
                             f"{stt['n']} → {m}{'*' if b else ''}")
            else:
                cells.append('-')
        out.append(f"| {i} | {r['label']} | {r['family']} | {r['status']} | {' | '.join(cells)} | **{r['mark']}** "
                   f"| {r['tmax']:.3f} | {r['mem']:.0f} | {r['problems']} | {trank.get(r['label'], '-')} |")

    out += ['', '## Families', '', '| family | status | iterations (share) | champion | S1 | S2 | S3 | mark | reason |',
            '|---|---|---|---|---|---|---|---|---|']
    total_it = sum(f['iterations'] for f in st['families'].values()) or 1
    byid = {S.bot_id(r['label']): r for r in rows if '[' not in r['label']}
    for fam, f in st['families'].items():
        c = f.get('champion')
        r = byid.get(c) if c else None
        marks = [str(r['per'][sc][1]) if r else '-' for sc in (1, 2, 3)]
        out.append(f"| {fam} | {f['status']} | {f['iterations']} ({f['iterations']/total_it*100:.0f}%) | {c or '-'} "
                   f"| {' | '.join(marks)} | {r['mark'] if r else '-'} | {f.get('reason', '')} |")

    out += ['', f"**Hall of fame:** {', '.join(st.get('hall_of_fame', [])) or 'empty'}  ",
            f"**Overall champion history:** {' → '.join(e['bot'] for e in st['overall_champions']) or 'none'}", '',
            '## Report coverage: techniques', '',
            f"Distinct techniques measured: {len(st['techniques'])} (need basic + 3 new).", '',
            '| technique tag | first bot | measured effect vs parent |', '|---|---|---|']
    for tag, t in st['techniques'].items():
        out.append(f"| {tag} | {t['bot']} | {t['effect']} |")

    if ch and ch in byid:
        crecs = [x for x in common.records_by_label().get(os.path.splitext(os.path.basename(ch_file))[0], [])
                 if x['seedset'] not in EXCLUDE_SEEDSETS]
        out += ['', f'## Champion {ch} by power (mean SC / win%)', '',
                '| scenario | ' + ' | '.join(C.POWERS) + ' |', '|---|' + '---|' * 7]
        for sc in (1, 2, 3):
            cells = []
            for p in C.POWERS:
                sub = [x for x in crecs if x['scenario'] == sc and x['power'] == p]
                stt = common.scenario_stats(sub)
                cells.append(f"{stt['mean_sc']:.1f} / {stt['win']*100:.0f}%" if stt['n'] else '-')
            out.append(f'| S{sc} | ' + ' | '.join(cells) + ' |')

    if tour:
        out += ['', '## Tournament (stand-in for Scenario 4)', '', '| rank | bot | mean SC | seats |', '|---|---|---|---|']
        for i, (m, n, lab) in enumerate(tour, 1):
            out.append(f'| {i} | {lab} | {m:.2f} | {n} |')

    with open(C.LEADERBOARD, 'w') as f:
        f.write('\n'.join(out) + '\n')
    print(f'LEADERBOARD.md: {len(rows)} bots, champion={ch}, agent file match={match}')


if __name__ == '__main__':
    main()
