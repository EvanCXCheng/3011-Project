"""Check a bot's own movement resolver against the diplomacy engine on random positions (lab tool, not a bot).

  .venv/bin/python lab/validate_resolver.py --module bots/bot_070_x.py [--games 40] [--seed 1] [--show 3]

The module must define parse_mo(order, reach_a) and resolve_moves(units, orders, reach_f); map reach comes from its
map_info(game) (or bot_045's when the module has none). Orders are drawn with a structured random policy (moves,
supports of chosen moves, holds, convoys of chosen VIA moves) so supports, cuts, bounces, swaps and convoys all occur.
Prints the agreement rate over movement phases and the time per resolution vs the engine's process().
"""
import argparse
import importlib.util
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from diplomacy import Game  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def random_orders(game, rng):
    possible = game.get_all_possible_orders()
    chosen = {}
    plan = {}
    for p, pw in game.powers.items():
        for b in game.get_orderable_locations(p):
            opts = possible.get(b) or []
            if not opts:
                continue
            r = rng.random()
            plan[(p, b)] = 'H' if r < 0.15 else 'M' if r < 0.6 else 'S' if r < 0.93 else 'C'
    for (p, b), k in plan.items():
        opts = possible[b]
        if k == 'M':
            mv = [o for o in opts if ' - ' in o]
            if mv:
                chosen[(p, b)] = rng.choice(mv)
    moved = {o.split(' - ')[0][2:].split()[0]: o for o in chosen.values()}
    move_strs = {o.split(' VIA')[0] for o in chosen.values()}
    for (p, b), k in plan.items():
        if (p, b) in chosen:
            continue
        opts = possible[b]
        if k == 'S':
            good = [o for o in opts if ' S ' in o and (o.split(' S ', 1)[1] in move_strs
                    or (' - ' not in o and o.split(' S ', 1)[1].split()[1].split('/')[0] not in moved))]
            sp = [o for o in opts if ' S ' in o]
            if good and rng.random() < 0.8:
                chosen[(p, b)] = rng.choice(good)
            elif sp:
                chosen[(p, b)] = rng.choice(sp)
        elif k == 'C':
            good = [o for o in opts if ' C ' in o and o.split(' C ', 1)[1] in move_strs]
            cv = [o for o in opts if ' C ' in o]
            if good:
                chosen[(p, b)] = rng.choice(good)
            elif cv and rng.random() < 0.3:
                chosen[(p, b)] = rng.choice(cv)
        if (p, b) not in chosen:
            hold = [o for o in opts if o.endswith(' H')]
            if hold:
                chosen[(p, b)] = hold[0]
    out = {p: [] for p in game.powers}
    for (p, b), o in chosen.items():
        out[p].append(o)
    return out


def snapshot(game):
    units = {}
    for p, pw in game.powers.items():
        for u in pw.units:
            if u.startswith('*'):
                continue
            t, loc = u.split()[:2]
            units[loc.split('/')[0]] = (p, t, loc)
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--module', required=True)
    ap.add_argument('--games', type=int, default=40)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--show', type=int, default=3)
    a = ap.parse_args()
    mod = load(a.module, 'resmod')
    mi = mod.map_info if hasattr(mod, 'map_info') else load(os.path.join(ROOT, 'bots/bot_045_search_stack.py'),
                                                             'b045').map_info
    rng = random.Random(a.seed)
    n = bad = 0
    t_ours = t_eng = 0.0
    shown = 0
    for gi in range(a.games):
        game = Game()
        info = mi(game)
        reach_a, reach_f = info['reach']['A'], info['reach']['F']
        while not game.is_game_done and int(game.get_current_phase()[1:5]) < 1912:
            if game.phase_type != 'M':
                game.process()
                continue
            orders = random_orders(game, rng)
            units = snapshot(game)
            t0 = time.perf_counter()
            parsed = {}
            for p, lst in orders.items():
                for o in lst:
                    b, po = mod.parse_mo(o, reach_a)
                    parsed[b] = po
            ours, lost = mod.resolve_moves(units, parsed, reach_f)
            t_ours += time.perf_counter() - t0
            for p, lst in orders.items():
                game.set_orders(p, lst)
            t0 = time.perf_counter()
            game.process()
            t_eng += time.perf_counter() - t0
            eng = snapshot(game)
            eng_lost = {u.split()[1].split('/')[0] for pw in game.powers.values() for u in pw.units if u.startswith('*')}
            eng_lost |= {u.split()[1].split('/')[0] for pw in game.powers.values() for u in pw.retreats}
            n += 1
            if eng != ours or eng_lost != lost:
                bad += 1
                if shown < a.show:
                    shown += 1
                    print(f'--- mismatch game {gi} phase before {game.get_current_phase()}')
                    for b in sorted(set(eng) | set(ours)):
                        if eng.get(b) != ours.get(b):
                            print(f'  {b}: engine {eng.get(b)} ours {ours.get(b)}')
                    print(f'  dislodged engine {sorted(eng_lost)} ours {sorted(lost)}')
                    rel = set(eng) ^ set(ours) | {b for b in eng if eng.get(b) != ours.get(b)}
                    for p, lst in orders.items():
                        for o in lst:
                            if any(x.split('/')[0] in rel for x in o.split()):
                                print('   ', p, o)
    print(f'movement phases {n}: agree {n - bad} ({100.0 * (n - bad) / max(n, 1):.1f}%) | '
          f'ours {1e6 * t_ours / max(n, 1):.0f} us/phase vs engine process {1e6 * t_eng / max(n, 1):.0f} us/phase')


if __name__ == '__main__':
    main()
