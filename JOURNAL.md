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
