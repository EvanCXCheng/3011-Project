"""Resume helper: what state is the lab in right now?

  python lab/status.py
"""
import glob
import hashlib
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import state as S  # noqa: E402


def sha(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.sha1(f.read()).hexdigest()
    except OSError:
        return None


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def main():
    now = datetime.now(C.AWST)
    print(f'now {now:%a %d %b %H:%M} AWST | freeze in {(C.FREEZE-now).total_seconds()/3600:.1f} h'
          f'{"  ** FROZEN: final evals/ablations/trim only **" if now >= C.FREEZE else ""}')
    st = S.load()
    ch = S.overall_champion(st)
    if ch:
        same = sha(os.path.join(C.ROOT, st['bots'][ch]['file'])) == sha(C.AGENT_FILE)
        print(f'overall champion {ch}; agent_{C.GROUP}.py {"matches" if same else "DOES NOT MATCH -> fix"}')
    else:
        print('overall champion: none yet (agent file is the stub)')
    total = sum(f['iterations'] for f in st['families'].values()) or 1
    print('families: ' + '; '.join(f"{k}:{v['status'][0]} it={v['iterations']}({v['iterations']/total*100:.0f}%)"
                                   f" ch={v['champion'] or '-'}" for k, v in st['families'].items()))
    print(f"techniques measured: {len(st['techniques'])} | next bot id: {st['next_bot_id']}")
    running = [b for b, v in st['bots'].items() if v['status'] == 'RUNNING']
    if running:
        print('bots in RUNNING state: ' + ', '.join(running))
    if os.path.exists(C.JOURNAL):
        txt = open(C.JOURNAL).read()
        heads = re.findall(r'^## .*RUNNING.*$', txt, flags=re.M)
        for h in heads:
            print('JOURNAL RUNNING: ' + h[3:])
    for m in glob.glob(os.path.join(C.LOG_DIR, '*.running')):
        lines = open(m).read().splitlines()
        pid = int(lines[0]) if lines and lines[0].isdigit() else -1
        log = m[:-8] + '.log'
        lines_log = open(log).read().splitlines() if os.path.exists(log) else []
        tail = lines_log[-1] if lines_log else ''
        print(f"eval {'RUNNING' if pid_alive(pid) else 'DEAD (stale marker)'}: {os.path.basename(m)[:-8]} | {tail}")


if __name__ == '__main__':
    main()
