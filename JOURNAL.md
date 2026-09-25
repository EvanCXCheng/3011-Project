# JOURNAL

Append-only. One entry per iteration. Terse notes and numbers, not report prose.
Entry header format: `## [NNN] <bot or task> — <STATUS>` with STATUS one of RUNNING / PROMOTED / REJECTED / BROKEN / DONE.
Fields: family, parent, hypothesis, test plan, results (key numbers), takeaway.

## [000] Lab setup — DONE

- Env: `.venv` (Python 3.12.3) with `requirements.txt`. 4 cores → WORKERS=3. 7 GB RAM.
- Memory: 512 MB `RLIMIT_AS` per worker works with numpy/scipy/sklearn imported (`OPENBLAS_NUM_THREADS=1`).
  A 7×Greedy game peaks at ~150 MB RSS for the whole process.
- Engine timings (`lab/bench_engine.py`, F1903M position): copy_game 2.5 ms, get_all_possible_orders 0.9 ms,
  copy+process 2.6 ms. Full games: 7 static 0.1 s, 7 greedy 1.1 s, S2 mix 1.0 s.
  → Simulation is cheap: roughly 100+ copy+process rollouts fit in 0.5 s, so lookahead is viable with the engine itself.
- Game cost is dominated by our bot's think time: 60–90 of our calls per game × t_mean.
- API notes (engine 1.1.2 source):
  - `get_orderable_locations(p)`: sorted 3-letter base locs (`STP` even for `F STP/SC`). M: unit locs; R: `power.retreats`
    units; A: build sites if builds > 0, unit locs if disbands needed, [] otherwise.
  - `get_all_possible_orders()`: dict keyed by base loc and by coast loc; orders use the full coast (`F STP/SC - BOT`).
    Adjustments include `WAIVE`. The call costs ~1 ms, so call it once per phase.
  - Game rules include `IGNORE_ERRORS`: invalid orders are silently voided, which is why the lab checks legality itself.
  - `is_game_done` fires at 18 SCs (victory = 34//2+1). Seen: an S2 game ended in 1909 with Greedy on 19 SCs.
  - Baseline GreedyAgent emits `F SPA H` for split-coast fleets (illegal/void), so the checker flags it. It's their bug, not ours.
  - TODO (iteration 0): confirm how the engine resolves missing disbands for Static powers in S1.
- Harness verified with dummy bots (in the scratchpad, not committed): exceptions, timeouts (including a bot that swallows
  the timeout exception), illegal/duplicate orders and MemoryError under the cap are all flagged. Reuse skips games
  already recorded. Smoke-test raw data deleted before the first commit.
- Lab design: `game.run_one_game` is used unmodified. Our seat is wrapped in `InstrumentedAgent` (timing via
  `timeout_decorator(1)`, legality vs the true game, desync check of the bot's internal game).

## [001] Baseline profiling — DONE

- Seed set A, 49 games per scenario (7 per power), S3 stand-in = Greedy baseline. Mean SC ± SE / win%:

| player | S1 | S2 | S3 (greedy stand-in) |
|---|---|---|---|
| Static | 3.14±0.05 / 0% | 2.45±0.16 / 0% | 2.41±0.16 / 0% |
| Random | 6.55±0.25 / 0% | 1.55±0.30 / 0% | 1.37±0.29 / 0% |
| Attitude | 6.76±0.24 / 0% | 1.55±0.26 / 0% | 1.37±0.23 / 0% |
| Greedy | 7.39±0.40 / 4.1% | 9.92±0.98 / 30.6% | 7.65±0.85 / 14.3% |

- Greedy baseline ≈ 5/15 estimated (S1 1, S2 3, S3 1), with everything borderline. In S1 it plateaus at ~7.7 SC from
  1908: it never dislodges holders, which confirms the key fact for S1.
- Greedy is by far the strongest baseline; Random and Attitude are fodder (~1.5 SC in S2). The S2 threat is Greedy neighbours.
- The Hidden Agent (~50% S2 win) is much stronger than the Greedy baseline, so the S3 numbers with a Greedy stand-in are optimistic.
- Targets for our bots: S1 needs supported attacks on holders (>12 SC for 3 pts, >16 for 5); S2 needs >50% wins, well above Greedy's 31%.
- Lab fix: Python 3.12 `ProcessPoolExecutor(max_tasks_per_child=...)` deadlocked after ~75 games (all workers
  exited, parent waiting). `evaluate.py` now recycles workers by running batches in fresh pools. The 75 games already
  recorded were kept and reused.
