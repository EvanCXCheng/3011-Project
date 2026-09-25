"""Benchmark engine primitives and full baseline games (sizes the tiers; informs the lookahead family).

  python lab/bench_engine.py
"""
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402
import common  # noqa: F401,E402  (puts repo root on sys.path)
from diplomacy import Game  # noqa: E402
from game import copy_game, run_one_game  # noqa: E402
import agent_baselines as B  # noqa: E402


def timeit(fn, n=30):
    t = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) / n * 1000


def main():
    random.seed(0)
    g = Game()
    # advance a few phases with greedy play so the position is non-trivial
    agents = {p: B.GreedyAgent() for p in C.POWERS}
    for p in C.POWERS:
        agents[p].new_game(copy_game(g), p)
    for _ in range(8):
        orders = {p: agents[p].get_actions() for p in C.POWERS}
        for p in C.POWERS:
            g.set_orders(p, orders[p])
        g.process()
        for p in C.POWERS:
            agents[p].update_game(orders)
    print(f'position {g.get_current_phase()}')
    print(f'copy_game               {timeit(lambda: copy_game(g)):.2f} ms')
    print(f'get_all_possible_orders {timeit(g.get_all_possible_orders):.2f} ms')

    def cp():
        c = copy_game(g)
        c.process()
    print(f'copy + process          {timeit(cp):.2f} ms')
    for name, pool in (('7 static', [B.StaticAgent] * 7), ('7 greedy', [B.GreedyAgent] * 7),
                       ('S2 mix', None)):
        ts = []
        for s in range(3):
            random.seed(s)
            if pool is None:
                ag = {p: random.choice([B.RandomAgent, B.AttitudeAgent, B.AttitudeAgent, B.GreedyAgent,
                                        B.GreedyAgent])() for p in C.POWERS}
            else:
                ag = {p: cls() for p, cls in zip(C.POWERS, pool)}
            t = time.perf_counter()
            run_one_game(ag)
            ts.append(time.perf_counter() - t)
        print(f'full game ({name:8s})   {sum(ts)/len(ts):.2f} s (mean of 3)')


if __name__ == '__main__':
    main()
