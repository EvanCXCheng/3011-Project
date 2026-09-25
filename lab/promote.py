"""Promote a bot to family champion and/or overall champion.

  python lab/promote.py bot_007 --family          # family champion only
  python lab/promote.py bot_007 --family --overall  # also overall champion -> copied to agent_21.py

Overall promotion checks: file <= 100 KB, the module imports, it defines the agent class,
and it uses only allowed imports. agent_21.py is then a byte-for-byte copy of the bot.
"""
import argparse
import ast
import os
import shutil
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import common  # noqa: E402
import state as S  # noqa: E402

ALLOWED_TOP = {'diplomacy', 'tqdm', 'random', 'networkx', 'numpy', 'scipy', 'sklearn', 'timeout_decorator',
               'simpleai', 'agent_baselines'}


def check_imports(path):
    tree = ast.parse(open(path).read())
    bad = []
    stdlib = sys.stdlib_module_names
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for n in names:
            top = n.split('.')[0]
            if top not in ALLOWED_TOP and top not in stdlib:
                bad.append(n)
    forbidden = {'socket', 'urllib', 'http', 'requests', 'subprocess', 'shutil', 'pickle', 'multiprocessing'}
    bad += [n for n in _imported(tree) if n.split('.')[0] in forbidden]
    return bad


def _imported(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def check_bot(path):
    problems = []
    if os.path.getsize(path) > C.FILE_SIZE_LIMIT:
        problems.append(f'size {os.path.getsize(path)} > 100KB')
    bad = check_imports(path)
    if bad:
        problems.append(f'disallowed imports: {bad}')
    src = open(path).read()
    if 'open(' in src:
        problems.append('uses open( — bots must not write files; check manually')
    try:
        common.load_agent_class(path)
    except Exception as e:  # noqa: BLE001
        problems.append(f'import failed: {e}')
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bot')
    ap.add_argument('--family', action='store_true')
    ap.add_argument('--overall', action='store_true')
    ap.add_argument('--force', action='store_true', help='promote despite check warnings')
    a = ap.parse_args()
    st = S.load()
    bid = S.bot_id(a.bot)
    if bid not in st['bots']:
        sys.exit(f'{bid} not registered in state.json (use state.py new)')
    info = st['bots'][bid]
    path = os.path.join(C.ROOT, info['file'])
    problems = check_bot(path)
    if problems and not a.force:
        sys.exit('refusing to promote: ' + '; '.join(problems))
    if a.family:
        st['families'][info['family']]['champion'] = bid
        print(f"{bid} is now {info['family']} family champion")
    if a.overall:
        st['overall_champions'].append({'bot': bid, 'date': datetime.now().isoformat(timespec='minutes')})
        shutil.copyfile(path, C.AGENT_FILE)
        print(f'{bid} is now overall champion; copied to {os.path.basename(C.AGENT_FILE)}')
    info['status'] = 'PROMOTED'
    S.save(st)


if __name__ == '__main__':
    main()
