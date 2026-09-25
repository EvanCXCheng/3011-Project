"""results/state.json: the single source of truth for bots, families, champions and hall of fame.

CLI:
  state.py init                                    -> create state.json if missing
  state.py show
  state.py new --family greedy --name base --parent none --tags "bfs-greedy"   -> allocates bots/bot_NNN_<family>_<name>.py
  state.py set-status bot_003 REJECTED|PROMOTED|BROKEN|RUNNING|PASSED_T1 ["note"]
  state.py family greedy ACTIVE|DORMANT "reason"
  state.py technique TAG bot_003 "effect vs parent (numbers)"
  state.py standin --for bot_003 [--rotate K]      -> prints the S3 stand-in spec for a candidate
"""
import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402


def empty_state():
    return {
        'next_bot_id': 1,
        'bots': {},                 # id -> {file, family, parent, tags, status, note, created}
        'families': {f: {'status': 'ACTIVE', 'iterations': 0, 'champion': None, 'reason': ''}
                     for f in C.FAMILIES},
        'overall_champions': [],    # history, last = current; entries {bot, date}
        'hall_of_fame': [],         # derived: family champions + last 3 overall champions
        'techniques': {},           # tag -> {bot, effect}
        'standin_rotation': 0,
    }


def load():
    if not os.path.exists(C.STATE_FILE):
        return empty_state()
    with open(C.STATE_FILE) as f:
        return json.load(f)


def save(st):
    st['hall_of_fame'] = hall_of_fame(st)
    tmp = C.STATE_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(st, f, indent=1, sort_keys=True)
    os.replace(tmp, C.STATE_FILE)


def hall_of_fame(st):
    hof = [fam['champion'] for fam in st['families'].values() if fam.get('champion')]
    for e in st['overall_champions'][-3:]:
        if e['bot'] not in hof:
            hof.append(e['bot'])
    return hof


def bot_id(label):
    """'bot_003_greedy_x[K=V]' -> 'bot_003'."""
    base = label.split('[', 1)[0]
    parts = base.split('_')
    return '_'.join(parts[:2]) if base.startswith('bot_') else base


def bot_file(st, bid):
    return st['bots'][bid]['file']


def family_of(st, label):
    b = st['bots'].get(bot_id(label))
    return b['family'] if b else None


def overall_champion(st):
    return st['overall_champions'][-1]['bot'] if st['overall_champions'] else None


def choose_standin(st, candidate_label, rotate=None):
    """Hidden-agent stand-in for S3 (see CLAUDE.md): strongest hall-of-fame bot from another family
    by S2 win rate; else the overall champion if from another family; else the Greedy baseline."""
    import common
    fam = family_of(st, candidate_label)
    others = [b for b in hall_of_fame(st) if st['bots'][b]['family'] != fam]
    if others:
        recs = common.records_by_label(scenarios=(2,))
        def s2(b):
            stem = os.path.splitext(os.path.basename(bot_file(st, b)))[0]
            rs = recs.get(stem, [])
            return (sum(r['win'] for r in rs) / len(rs) if rs else 0.0,
                    sum(r['sc'] for r in rs) / len(rs) if rs else 0.0)
        others.sort(key=s2, reverse=True)
        k = (st.get('standin_rotation', 0) if rotate is None else rotate) % len(others)
        return bot_file(st, others[k])
    ch = overall_champion(st)
    if ch and st['bots'][ch]['family'] != fam:
        return bot_file(st, ch)
    return 'greedy'


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('init')
    sub.add_parser('show')
    n = sub.add_parser('new')
    n.add_argument('--family', required=True)
    n.add_argument('--name', required=True)
    n.add_argument('--parent', default='none')
    n.add_argument('--tags', default='')
    s = sub.add_parser('set-status')
    s.add_argument('bot')
    s.add_argument('status')
    s.add_argument('note', nargs='?', default='')
    f = sub.add_parser('family')
    f.add_argument('family')
    f.add_argument('status')
    f.add_argument('reason', nargs='?', default='')
    t = sub.add_parser('technique')
    t.add_argument('tag')
    t.add_argument('bot')
    t.add_argument('effect')
    sd = sub.add_parser('standin')
    sd.add_argument('--for', dest='cand', required=True)
    sd.add_argument('--rotate', type=int, default=None)
    sub.add_parser('rotate-standin')
    a = ap.parse_args()

    st = load()
    if a.cmd == 'init':
        if not os.path.exists(C.STATE_FILE):
            save(st)
        print('ok')
        return
    if a.cmd == 'show':
        print(json.dumps({k: st[k] for k in ('families', 'overall_champions', 'hall_of_fame', 'next_bot_id')},
                         indent=1))
        return
    if a.cmd == 'new':
        if a.family not in st['families']:
            st['families'][a.family] = {'status': 'ACTIVE', 'iterations': 0, 'champion': None, 'reason': 'new'}
        bid = f"bot_{st['next_bot_id']:03d}"
        rel = f"bots/{bid}_{a.family}_{a.name}.py"
        st['next_bot_id'] += 1
        st['families'][a.family]['iterations'] += 1
        st['bots'][bid] = {'file': rel, 'family': a.family, 'parent': a.parent, 'tags': a.tags,
                           'status': 'RUNNING', 'note': '', 'created': datetime.now().isoformat(timespec='minutes')}
        save(st)
        print(rel)
        return
    if a.cmd == 'set-status':
        b = st['bots'][bot_id(a.bot)]
        b['status'] = a.status
        if a.note:
            b['note'] = a.note
    elif a.cmd == 'family':
        st['families'][a.family]['status'] = a.status
        st['families'][a.family]['reason'] = a.reason
    elif a.cmd == 'technique':
        st['techniques'][a.tag] = {'bot': bot_id(a.bot), 'effect': a.effect}
    elif a.cmd == 'standin':
        print(choose_standin(st, a.cand, a.rotate))
        return
    elif a.cmd == 'rotate-standin':
        st['standin_rotation'] = st.get('standin_rotation', 0) + 1
    save(st)
    print('ok')


if __name__ == '__main__':
    main()
