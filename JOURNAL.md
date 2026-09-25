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

## [003] bot_002 valuemap/base — PROMOTED

- family: valuemap | parent: none | tags: value-map, value-diffusion (DumbBot-style idea, cite; own code)
- hypothesis: a province value map (neutral SC 10, enemy SC 7, own SC 5×adjacent enemy units, diffused over the
  per-unit-type graph: v = base + 0.6·max(nbr) + 0.05·sum(nbr), 6 passes), with units greedily assigned the highest-value
  reachable province (no two units on one province), beats plain nearest-SC greedy in S2/S3 through defence and dedup.
- plan: Tier 0 → Tier 1 (A, 42) → Tier 2 (B, 210) vs bot_001.
- Tier 0 clean (tmax 4 ms). Tier 1 A42 (S3 stand-in bot_001): S1 12.29±0.44 (0%) / S2 9.17±0.84 (14.3%) / S3 6.40±0.66 (4.8%)
  → est. 5. Bootstrap → valuemap family champion.
- Tier 2 B210 (stand-in bot_001): S1 12.29±0.19 (0%) / S2 8.12±0.37 (11.4%) / S3 6.53±0.33 (7.1%) → est. 5.
  vs bot_001 paired: S1 +6.14±0.20, S2 +0.69±0.36, S3 −0.22±0.31, pooled +2.20±0.20 (PROMOTE verdict, but bot_003 better).
- takeaway: value map + dedup solves S1 neutral grab (12.3 SC, 0 wins: never 2v1) but adds little in S2/S3.

## [004] bot_003 search/base — PROMOTED

- family: search | parent: none | tags: local-search, hill-climbing, heuristic-eval
- hypothesis: coordinate-ascent hill climbing with random restarts (≤0.4 s, stop after 25 stale restarts) over our joint
  orders (hold/move/support own units), scored by a heuristic (move success prob from strength vs occupancy and
  contesting enemies; expected SC capture + distance to nearest unowned SC; penalty for open threatened own SCs),
  finds supported attacks and avoids self-bounces → better than bot_001 in S1 (needs 2v1) and S2.
- plan: Tier 0 → Tier 1 (A, 42) → Tier 2 (B, 210, or 126 if too slow).
- Tier 0 clean (tmax 0.40 s). Tier 1 A42 (stand-in bot_001): S1 11.50±0.65 (23.8%) / S2 13.24±0.90 (45.2%) /
  S3 12.86±0.89 (47.6%) → est. 13 (all borderline). Bootstrap → search family champion.
- Tier 2 B210 (stand-in bot_001): S1 11.76±0.30 (22.4%) / S2 13.57±0.37 (49.5%) / S3 12.72±0.39 (43.8%) → est. 13.
  vs bot_001 paired: +5.61±0.31 / +6.14±0.37 / +5.97±0.39, pooled +5.91±0.21 → PROMOTE.
- stress (lab/stress.py, ×4.18 slowdown): tmax 0.412 s, overshoot 0.012 s → PASS. → OVERALL CHAMPION (agent_21.py).
- takeaway: coordinated joint orders (supports, dedup, garrisons) are worth ~+6 SC over per-unit greedy. S2 win 49.5%
  sits right on the 50% 5-pt line; S1 11.8 just under the 12 line.

## [005] bot_004 lookahead/base — PROMOTED

- family: lookahead | parent: none | tags: one-ply-simulation, opponent-sampling, light-game-copy
- engine note: copy_game (saved-format round trip) costs ~8 ms by S1912 because it carries history; a history-free
  "light" Game (Game() + set_units/set_centers/set_current_phase) costs 0.6 ms to build once, then copy.deepcopy 0.3 ms
  and deepcopy+process ~0.6 ms. → rollouts on light copies.
- hypothesis: generate K candidate joint orders (greedy distance/SC scorer + random perturbations incl. supports of
  own moves), simulate each 1 move deep against sampled opponent orders (hold / move to own-nearest SC / random),
  common random numbers across candidates, pick best mean score (SCs held+occupied, units kept, distance). Beats bot_001.
- plan: Tier 0 → Tier 1 (A, 42) → Tier 2 (B, 210 or 126).
- Tier 0 clean (tmax 0.52 s under contention). ~200 rollouts/phase (20 rounds × 10 candidates).
- Tier 1 A42 (stand-in auto = bot_003): S1 16.52±0.49 (76.2%) / S2 15.07±0.69 (59.5%) / S3 10.14±1.01 (31.0%) → est. 13.
  Bootstrap → lookahead family champion. stress ×4.21: tmax 0.543 s, overshoot 0.093 s → PASS.
- Tier 2 at the 126 fallback (210 would take ~63 min > 45 min limit): chain3, stand-in bot_003.
- Tier 2 B126: S1 16.10±0.29 (63.5%) / S2 14.79±0.44 (61.9%) / S3 11.40±0.54 (34.1%) → est. 13. tmax 0.533 s.
  vs bot_003 paired (S3 both with stand-in bot_003): S1 +4.31±0.44, S2 +1.35±0.47, S3 +0.83±0.63, pooled +2.16±0.31,
  marks 13 vs 11 → PROMOTE → OVERALL CHAMPION (stress already PASS).
- takeaway: 1-ply engine rollouts on light copies beat the heuristic-eval search everywhere. S1 still 36% non-wins
  (adaptive gets 100%): the fixed 40% opponent-hold mix underrates static holders → bot_007 tests class-based sampling.

## [006] bot_005 adaptive/base — PROMOTED

- family: adaptive | parent: none | tags: opponent-classification, opponent-prediction
- hypothesis: classifying each opponent from its order history (static: ≥95% holds; greedy: ≥80% of moves reduce its
  distance to its nearest unowned SC; erratic otherwise; unknown before history → treated as greedy) and predicting
  its next moves lets a rule bot (1) garrison own SCs that predicted moves hit, (2) make 2v1 supported attacks on SCs
  whose occupant is predicted to hold, (3) route other units to free targets, avoiding predicted holders.
  Expect strong S1 (all static detected after 1 phase), competitive S2.
- plan: Tier 0 → Tier 1 (A, 42) → Tier 2.
- Tier 0 clean (tmax 3 ms): S1 13.0 / S2 7.3 / S3 5.7 (n=3). Tier 1 in chain3 (stand-in bot_004).
- Tier 1 A42 (stand-in bot_004): S1 12.29±0.61 (14.3%) / S2 8.60±0.82 (11.9%) / S3 5.57±0.59 (0%) → est. 4.
  Bootstrap → positional family champion; new family → Tier 2 queued (chain6).
- takeaway: safety gating is too passive in S2/S3 (8.6 / 5.6 SC); S1 fine thanks to 2v1 attacks.
- Tier 0 clean (tmax 6 ms). Tier 1 A42 (stand-in bot_003): S1 18.00±0.00 (100%) / S2 14.02±0.77 (45.2%) /
  S3 9.79±0.96 (26.2%) → est. 13. Bootstrap → adaptive family champion. Tier 2 in chain3 (stand-in bot_004).
- Tier 2 B210 (stand-in bot_004): S1 18.00±0.00 (100%) / S2 12.75±0.42 (48.1%) / S3 9.74±0.44 (26.2%) → est. 11.
  vs bot_003 paired (S3 stand-in bot_004 both): S1 +6.24±0.30, S2 −0.82±0.41, S3 −1.44±0.45, pooled +1.33±0.27, marks
  11 vs 11 → PROMOTE verdict vs bot_003, but bot_004 is stronger (13) → stays adaptive family champion only.
- takeaway: static detection + 2v1 supported attacks solves S1 completely (210/210 wins). S2/S3 weaker than search:
  rule-based moves lack coordination. Hybrid candidate: adaptive S1 plan + lookahead elsewhere.

## [007] bot_006 positional/base — PROMOTED

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

## [009] bot_007 lookahead/oppmodel — RUNNING

- Tier 0 clean. Tier 1 A42 (stand-in bot_003): S1 18.00 (100%) / S2 14.69±0.72 (57.1%) / S3 10.26±1.06 (33.3%);
  vs bot_004 paired: S1 +1.48±0.49, S2 −0.38±0.72, S3 +0.12±1.04, pooled +0.40±0.46 → passes screen. Tier 2 running.

- family: lookahead | parent: bot_004 | tags: opponent-model-sampling (hybrid: adaptive's classifier inside lookahead)
- hypothesis: sampling opponent orders from a per-power class (static → always hold; greedy → 90% greedy move;
  erratic → 30% hold / 50% random / 20% greedy; unknown → bot_004's 40/10/50 mix) makes rollouts match reality,
  so S1 moves stop fearing holders (→ S1 wins like bot_005's 100%) and S2 defence/attack gets sharper.
- plan: Tier 0 → Tier 1 (A, 42, same stand-in as bot_004: bot_003) → Tier 2 (B, 126) vs bot_004 and bot_003.

## [010] bot_008 search/oppaware — RUNNING

- family: search | parent: bot_003 | tags: opponent-aware-eval
- hypothesis: bot_003's eval counts every adjacent enemy unit as a contester/threat, so static units freeze moves in S1
  (S1 11.8 SC vs adaptive's 18.0). Replacing counts with class-predicted expected entries (threat) and occupant hold
  probability (unsupported into a holder: 0; supported: 0.95; mover-out: 1) should fix S1 and sharpen S2 defence.
- sanity: S1 seed 4 → 18 SC, S2 seed 4 → 18 SC, tmax 0.28 s.
- plan: Tier 0 → Tier 1 (A, 42, stand-in bot_001 = bot_003's Tier 1) → Tier 2 (B, 210, stand-in bot_004 = chain3 ref).

## [011] bot_009 adaptive/scenswitch — RUNNING

- family: adaptive | parent: bot_005 | tags: scenario-detection, hybrid-switch (lookahead code copied from bot_004)
- analysis (lab/analyze.py bot_005): S1 100% for every power; S2/S3 SC falls with #greedy opponents
  (S2: 0 greedy 18.0 → 4 greedy 9.3), worst 20% of S3 games dominated by the lookahead stand-in.
- hypothesis: all opponents classified static → bot_005 rule plan (S1 100%); otherwise bot_004 lookahead (S2 14.8,
  S3 11.4). Expect ≈ S1 18 / S2 ≈ bot_004 / S3 ≈ bot_004 → beats both parents.
- sanity: S1 seed 11 → 18 SC (won 1907, t_mean 18 ms: switch fires after S1901M); S2 seed 11 → 6 SC; tmax 0.46 s.
- plan: Tier 0 → Tier 1 (A, 42, stand-in bot_003 = bot_005's Tier 1) → Tier 2 (B, 126 fallback, stand-in bot_004;
  refs bot_005 B210 and bot_004 B126 S3-with-bot_004 from chain6).

## [012] bot_010 greedy/dedup — PROMOTED

- family: greedy | parent: bot_001 | tags: no-self-bounce
- hypothesis: joint destination assignment (best unit–destination pair first, no shared destinations, enter an
  own-occupied province only if the occupant leaves and it is not a swap) removes self-bounces → S1/S2 gain over bot_001.
- plan: Tier 0 → Tier 1 (A, 42, --workers 1 on the spare core; stand-in = greedy baseline, as bot_001's Tier 1) →
  Tier 2 (B, 210, stand-in bot_001, pairs with bot_001's B210 stand-in-bot_001 run). Cheap bot → run outside the chains.
- Tier 0 clean (tmax 4 ms): S1 9.67 / S2 8.67 / S3 4.00 (n=3).
- Tier 1 A42: S1 8.57±0.32 / S2 10.45±0.95 (28.6%) / S3 9.62±0.76 (14.3%); vs bot_001 pooled +2.70±0.36.
- Tier 2 B210 (stand-in bot_001): S1 8.57±0.14 (0%) / S2 10.92±0.39 (25.7%) / S3 9.38±0.35 (13.3%) → est. 7.
  vs bot_001 paired: +2.43±0.13 / +3.49±0.35 / +2.62±0.29, pooled +2.85±0.16 → PROMOTE → greedy family champion.
- takeaway: self-bounces cost ~3 SC; greedy with dedup now beats the Greedy baseline (S2 10.9 vs 9.9). Next: supports.

## [013] bot_011 valuemap/supports — PROMOTED

- family: valuemap | parent: bot_002 | tags: supported-attacks (backlog item 3 moved ahead of item 2: bot_002's S1
  stalls at 12.3 SC with 0 wins because it never attacks holders 2v1)
- hypothesis: attacker+supporter pairs on enemy-occupied SCs (highest value first) before the value assignment, and
  leftover holders supporting contested own moves → S1 wins, some S2 gain.
- sanity: S1 seed 11 → 15 SC, S2 seed 11 → 8 SC, tmax 9 ms.
- plan: spare-core queue (--workers 1): Tier 0 → Tier 1 (A, stand-in bot_001 = bot_002's) → Tier 2 (B210, stand-in bot_001).
- Tier 1 A42: S1 16.00±0.48 (57.1%) / S2 12.55±0.90 (42.9%) / S3 9.95±0.87 (16.7%); vs bot_002 pooled +3.55±0.39.
- Tier 2 B210 (stand-in bot_001): S1 16.00±0.21 (57.1%) / S2 12.36±0.39 (39.5%) / S3 10.50±0.41 (27.1%) → est. 9.
  vs bot_002 paired: +3.71±0.14 / +4.24±0.35 / +3.97±0.36, pooled +3.97±0.17 → PROMOTE → valuemap family champion.
- takeaway: 2v1 supports are the single biggest rule-bot gain so far (+4 SC in every scenario). S1 16.0 but only 57%
  wins: some powers stall below 18 (S1 is deterministic per power).

## [014] bot_012 greedy/attackmatch — RUNNING

- family: greedy | parent: bot_010 | tags: supported-attack-matching
- hypothesis: matching units to attacker/supporter roles on enemy-occupied SCs with linear_sum_assignment (Hungarian),
  keeping only fully staffed targets, turns bot_010's S1 plateau (8.6 SC, 0 wins) into steady 2v1 growth.
- sanity: S1 seed 11 → 18 SC; S2 seed 11 → 6 SC; tmax 6 ms; new_game 29 ms (scipy import at module load).
- plan: spare-core queue after bot_011: Tier 0 → Tier 1 (A, stand-in greedy baseline = bot_010's) → Tier 2 (B210, stand-in bot_001).

## [015] bot_013 positional/supadvance — RUNNING

- family: positional | parent: bot_006 | tags: supported-advance
- analysis (bot_006 Tier 1): S1 deterministic per power (AUS 7, ITA 8 stall); S2 growth stops after 1905 (8.6 SC)
  while losses stay low (6/42 games lose ≥3 from peak) → defence ok, offence too weak.
- hypothesis: bot_006's gate counts potential supporters but never orders them, so gated moves into contested
  provinces bounce. Committing the k = enemy_adj − 1 needed supporters with the move makes the advance real → S2/S3 gain.
- plan: spare-core queue after bot_012: Tier 0 → Tier 1 (A, stand-in bot_004 = bot_006's) → Tier 2 (B210, stand-in bot_004).
