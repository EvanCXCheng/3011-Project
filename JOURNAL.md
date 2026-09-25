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

## [002] bot_001 greedy/base — PROMOTED

- family: greedy | parent: none | tags: bfs-greedy
- hypothesis: each unit moving to the neighbour closest (BFS, per unit type, coast-aware) to the nearest unowned SC
  gives a working baseline near the Greedy baseline (S1 ~7 SC, S2 ~10 SC). No dedup, supports or convoys yet.
- plan: Tier 0 (X, n=3, workers 1) → Tier 1 (A, n=42). Bootstrap: first bot to pass Tier 1 becomes overall champion.
- results: Tier 0 clean (tmax 2 ms). Tier 1 A42: S1 6.14±0.40 / S2 8.31±0.82 (11.9% win) / S3 6.10±0.64 (2.4%).
  Tier 2 B210: S1 6.14±0.18 (0%) / S2 7.43±0.33 (6.7%) / S3 6.14±0.28 (3.3%) → est. mark 2 (S2 1, S3 1). tmax 0.07 s.
- takeaway: below the Greedy baseline (S1 7.4, S2 9.9). Weaknesses: self-bounces, no convoys (England armies stuck),
  S1 stalls after neutrals (no supported attacks). Bootstrap → greedy champion + overall champion (agent_21.py).

## [003] bot_002 valuemap/base — RUNNING

- family: valuemap | parent: none | tags: value-map, value-diffusion (DumbBot-style idea, cite; own code)
- hypothesis: a province value map (neutral SC 10, enemy SC 7, own SC 5×adjacent enemy units, diffused over the
  per-unit-type graph: v = base + 0.6·max(nbr) + 0.05·sum(nbr), 6 passes), with units greedily assigned the highest-value
  reachable province (no two units on one province), beats plain nearest-SC greedy in S2/S3 through defence and dedup.
- plan: Tier 0 → Tier 1 (A, 42) → Tier 2 (B, 210) vs bot_001.
- Tier 0 clean (tmax 4 ms). Tier 1 A42 (S3 stand-in bot_001): S1 12.29±0.44 (0%) / S2 9.17±0.84 (14.3%) / S3 6.40±0.66 (4.8%)
  → est. 5. Bootstrap → valuemap family champion. Tier 2 queued (chain1).

## [004] bot_003 search/base — RUNNING

- family: search | parent: none | tags: local-search, hill-climbing, heuristic-eval
- hypothesis: coordinate-ascent hill climbing with random restarts (≤0.4 s, stop after 25 stale restarts) over our joint
  orders (hold/move/support own units), scored by a heuristic (move success prob from strength vs occupancy and
  contesting enemies; expected SC capture + distance to nearest unowned SC; penalty for open threatened own SCs),
  finds supported attacks and avoids self-bounces → better than bot_001 in S1 (needs 2v1) and S2.
- plan: Tier 0 → Tier 1 (A, 42) → Tier 2 (B, 210, or 126 if too slow).
- Tier 0 clean (tmax 0.40 s). Tier 1 A42 (stand-in bot_001): S1 11.50±0.65 (23.8%) / S2 13.24±0.90 (45.2%) /
  S3 12.86±0.89 (47.6%) → est. 13 (all borderline). Bootstrap → search family champion. Tier 2 queued (chain1).

## [005] bot_004 lookahead/base — RUNNING

- family: lookahead | parent: none | tags: one-ply-simulation, opponent-sampling, light-game-copy
- engine note: copy_game (saved-format round trip) costs ~8 ms by S1912 because it carries history; a history-free
  "light" Game (Game() + set_units/set_centers/set_current_phase) costs 0.6 ms to build once, then copy.deepcopy 0.3 ms
  and deepcopy+process ~0.6 ms. → rollouts on light copies.
- hypothesis: generate K candidate joint orders (greedy distance/SC scorer + random perturbations incl. supports of
  own moves), simulate each 1 move deep against sampled opponent orders (hold / move to own-nearest SC / random),
  common random numbers across candidates, pick best mean score (SCs held+occupied, units kept, distance). Beats bot_001.
- plan: Tier 0 → Tier 1 (A, 42) → Tier 2 (B, 210 or 126).

## [006] bot_005 adaptive/base — RUNNING

- family: adaptive | parent: none | tags: opponent-classification, opponent-prediction
- hypothesis: classifying each opponent from its order history (static: ≥95% holds; greedy: ≥80% of moves reduce its
  distance to its nearest unowned SC; erratic otherwise; unknown before history → treated as greedy) and predicting
  its next moves lets a rule bot (1) garrison own SCs that predicted moves hit, (2) make 2v1 supported attacks on SCs
  whose occupant is predicted to hold, (3) route other units to free targets, avoiding predicted holders.
  Expect strong S1 (all static detected after 1 phase), competitive S2.
- plan: Tier 0 → Tier 1 (A, 42) → Tier 2.

## [007] bot_006 positional/base — RUNNING

- family: positional | parent: none | tags: target-power-selection, strength-gated-moves, home-garrison
- hypothesis: slow, safe expansion: (1) never leave an own SC empty when an enemy unit is adjacent (garrison / move in);
  (2) choose one target power by weakness (units) + reachability (mean distance from our units to its SCs), targets =
  neutral SCs + that power's SCs; (3) only move into a province where our adjacent strength (mover + possible
  supporters) ≥ adjacent enemy strength, and add a support when it is contested or occupied. Loses fewer SCs in S2/S3.
- plan: Tier 0 → Tier 1 (A, 42) → Tier 2.

## [008] Lab: slow-machine timing gate — DONE

- Concern: the marking CPU may be slower than ours. Bots stop on wall-clock budgets (0.40–0.50 s), so a slower CPU
  mainly lowers search quality; the risk is the unbudgeted work (possible-orders, candidate generation, rollout in
  flight, final selection), which scales with CPU speed. bot_004 already reached 0.52 s vs a 0.45 s budget under mild load.
- Added `lab/stress.py`: K games (default 4) pinned to one core ≈ K× slowdown, measured with a calibration loop; reports
  tmax, movement-phase overshoot past CONFIG TIME_BUDGET, timeouts. Not written to results/raw. Gate: tmax < 0.8 s.
- CLAUDE.md: the gate is required before an overall-champion promotion when tmax > 0.1 s; run only between evaluations.
- Validated on bot_001 (1 proc, 3 games): PASS, tmax 0.006 s. First real use: bot_003/bot_004 once chains 1–2 finish.
