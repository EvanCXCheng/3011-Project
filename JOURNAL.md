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
- Tier 2 B210 (stand-in bot_004): S1 12.29±0.27 (14.3%) / S2 8.69±0.35 (8.1%) / S3 6.30±0.29 (3.3%) → est. 5.
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

## [009] bot_007 lookahead/oppmodel — PROMOTED

- Tier 0 clean. Tier 1 A42 (stand-in bot_003): S1 18.00 (100%) / S2 14.69±0.72 (57.1%) / S3 10.26±1.06 (33.3%);
  vs bot_004 paired: S1 +1.48±0.49, S2 −0.38±0.72, S3 +0.12±1.04, pooled +0.40±0.46 → passes screen.
- Tier 2 B126 (stand-in bot_003): S1 18.00 (100%) / S2 15.27±0.39 (57.9%) / S3 11.29±0.54 (32.5%) → est. 13.
  vs bot_004 paired: S1 +1.90±0.29, S2 +0.48±0.47, S3 −0.12±0.53, pooled +0.75±0.26, marks 13 vs 13 → PROMOTE
  (lookahead family champion now; overall promotion waits for lab/stress.py on an idle CPU).
- stress (core 0 alone; bot_013's evaluation pinned to core 3): ×3.94, tmax 0.542 s, overshoot 0.092 s → PASS →
  OVERALL CHAMPION (agent_21.py).
- takeaway: class-based opponent sampling fixes S1 completely (126/126 wins); no measurable effect in S2/S3.

- family: lookahead | parent: bot_004 | tags: opponent-model-sampling (hybrid: adaptive's classifier inside lookahead)
- hypothesis: sampling opponent orders from a per-power class (static → always hold; greedy → 90% greedy move;
  erratic → 30% hold / 50% random / 20% greedy; unknown → bot_004's 40/10/50 mix) makes rollouts match reality,
  so S1 moves stop fearing holders (→ S1 wins like bot_005's 100%) and S2 defence/attack gets sharper.
- plan: Tier 0 → Tier 1 (A, 42, same stand-in as bot_004: bot_003) → Tier 2 (B, 126) vs bot_004 and bot_003.

## [010] bot_008 search/oppaware — PROMOTED

- family: search | parent: bot_003 | tags: opponent-aware-eval
- hypothesis: bot_003's eval counts every adjacent enemy unit as a contester/threat, so static units freeze moves in S1
  (S1 11.8 SC vs adaptive's 18.0). Replacing counts with class-predicted expected entries (threat) and occupant hold
  probability (unsupported into a holder: 0; supported: 0.95; mover-out: 1) should fix S1 and sharpen S2 defence.
- sanity: S1 seed 4 → 18 SC, S2 seed 4 → 18 SC, tmax 0.28 s.
- plan: Tier 0 → Tier 1 (A, 42, stand-in bot_001 = bot_003's Tier 1) → Tier 2 (B, 210, stand-in bot_004 = chain3 ref).
- Tier 0 clean (tmax 0.29 s). Tier 1 A42: S1 18.00 (100%) / S2 13.55±0.90 (50%) / S3 12.55±0.93 (35.7%), vs bot_003
  pooled +2.17±0.55.
- Tier 2 B210 (stand-in bot_004): S1 18.00 (100%) / S2 13.04±0.38 (44.8%) / S3 9.95±0.42 (26.7%) → est. 13.
  vs bot_003 paired: S1 +6.24±0.30, S2 −0.53±0.35, S3 −1.23±0.39, pooled +1.49±0.24, marks 13 vs 11 → PROMOTE →
  search family champion (not overall: below bot_007 in S2/S3).
- takeaway: class-predicted threats fix S1 (100%) but hurt vs the strong lookahead stand-in in S3: a strong unknown
  opponent gets classified greedy/erratic and its moves are mispredicted, so the eval under-defends. Idea: fall back to
  raw adjacency for opponents that are neither static nor well predicted (per-power prediction accuracy).

## [011] bot_009 adaptive/scenswitch — PROMOTED

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

## [014] bot_012 greedy/attackmatch — PROMOTED

- family: greedy | parent: bot_010 | tags: supported-attack-matching
- hypothesis: matching units to attacker/supporter roles on enemy-occupied SCs with linear_sum_assignment (Hungarian),
  keeping only fully staffed targets, turns bot_010's S1 plateau (8.6 SC, 0 wins) into steady 2v1 growth.
- sanity: S1 seed 11 → 18 SC; S2 seed 11 → 6 SC; tmax 6 ms; new_game 29 ms (scipy import at module load).
- plan: spare-core queue after bot_011: Tier 0 → Tier 1 (A, stand-in greedy baseline = bot_010's) → Tier 2 (B210, stand-in bot_001).
- Tier 1 A42: S1 12.29±0.74 (28.6%) / S2 13.88±0.87 (54.8%) / S3 13.38±0.81 (47.6%); vs bot_010 pooled +3.63±0.45.
- Tier 2 B210 (stand-in bot_001): S1 12.29±0.33 (28.6%) / S2 13.49±0.36 (47.6%) / S3 12.08±0.40 (35.7%) → est. 13.
  vs bot_010: +3.71±0.30 / +2.57±0.35 / +2.70±0.34, pooled +3.00±0.19 → PROMOTE → greedy family champion.
  Peak memory 141 MB (numpy/scipy import), fine.
- takeaway: Hungarian role matching for 2v1 attacks gives +3 SC; greedy family now at est. 13 (S3 with weak stand-in bot_001).

## [015] bot_013 positional/supadvance — REJECTED

- family: positional | parent: bot_006 | tags: supported-advance
- analysis (bot_006 Tier 1): S1 deterministic per power (AUS 7, ITA 8 stall); S2 growth stops after 1905 (8.6 SC)
  while losses stay low (6/42 games lose ≥3 from peak) → defence ok, offence too weak.
- hypothesis: bot_006's gate counts potential supporters but never orders them, so gated moves into contested
  provinces bounce. Committing the k = enemy_adj − 1 needed supporters with the move makes the advance real → S2/S3 gain.
- plan: spare-core queue after bot_012: Tier 0 → Tier 1 (A, stand-in bot_004 = bot_006's) → Tier 2 (B210, stand-in bot_004).
- Tier 1 A42: S1 11.86 / S2 8.33 / S3 5.19; vs bot_006 pooled −0.36±0.23 (not < −2 SE → Tier 2).
- Tier 2 B210 (stand-in bot_004): S1 11.86±0.24 (0%) / S2 8.70±0.35 (11.9%) / S3 6.35±0.30 (2.9%) → est. 3.
  vs bot_006: S1 −0.43±0.07, S2 +0.01±0.29, S3 +0.05±0.28, pooled −0.12±0.14, marks 3 vs 5 → REJECT.
- takeaway: the bounce hypothesis was wrong: committing supporters just removes units from expansion. Positional
  weakness is target selection/passivity, not failed gated moves.

## [016] bot_014 valuemap/strength — PROMOTED

- family: valuemap | parent: bot_011 | tags: strength-aware-values (backlog item 2)
- analysis (bot_011): S1 by power AUS 9 / ITA 15 / TUR 16 (others 18). Trace S1 Austria: from 1905 to 1920 the same 9
  orders every turn: each unit attacks a static holder unsupported or an own unit that cannot leave → permanent jam.
- hypothesis: discount unsupported moves into enemy-occupied provinces (keep 15% of value) so units gather next to
  targets and the 2v1 pre-pass can fire.
- sanity: with the planned second term COMP=0.25 (competition divisor) S1 AUS/ITA/TUR = 7/8/10 (worse); OCC only
  (COMP=0) → 18/15/18. Default set to COMP=0 (knob kept). S2 seeds 11/12 → 3/18 SC, no errors.
- plan: spare-core queue after bot_013: Tier 0 → Tier 1 (A, stand-in bot_001 = bot_011's) → Tier 2 (B210, stand-in bot_001).
- Tier 1 A42: S1 17.57±0.16 (85.7%) / S2 14.07±0.73 (42.9%) / S3 10.86±0.90 (21.4%); vs bot_011 pooled +1.33±0.38.
- Tier 2 B210 (stand-in bot_001): S1 17.57±0.07 (85.7%) / S2 12.14±0.40 (39.5%) / S3 11.37±0.42 (33.3%) → est. 11.
  vs bot_011: S1 +1.57±0.22, S2 −0.22±0.33, S3 +0.87±0.34, pooled +0.74±0.18, marks 11 vs 9 → PROMOTE → valuemap champion.
- takeaway: the jam fix works in S1 (one power, ITA, still short of 18); S2 unchanged.

## [017] bot_015 lookahead/defcands — REJECTED

- family: lookahead | parent: bot_007 | tags: defensive-candidates
- analysis (bot_007 S3, n=168): 42/168 games lose ≥3 SC from peak; worst 20% mostly vs the bot_003 stand-in; growth
  flattens after 1905 (7.9 → 11.2 SC). bot_007's candidates never contain support-holds or garrisons.
- hypothesis: adding support-hold perturbations (supported unit set to hold) and one garrison candidate (own SCs in
  enemy reach held, neighbours support-hold them) lets the rollouts choose defence when it pays → S3 gain.
- sanity S3 (stand-in bot_003) seeds 11/12/13 → 14/18/14 SC, tmax 0.47 s, no errors.
- plan: main2 queue: Tier 0 → Tier 1 (A, stand-in bot_003 = bot_007's) → Tier 2 (B126, stand-in bot_003).
- Tier 1 A42: S1 18.00 / S2 14.00±0.85 / S3 9.00±0.94; vs bot_007 pooled −0.65±0.41 (not < −2 SE → Tier 2).
- Tier 2 B126 (stand-in bot_003): S1 17.99 / S2 14.39±0.46 (57.9%) / S3 11.27±0.50 (28.6%). vs bot_007: S1 −0.01,
  S2 −0.88±0.43, S3 −0.02±0.51, pooled −0.30±0.22, marks 13 vs 13 → REJECT.
- takeaway: extra defensive candidates don't help S3 and cost S2 (fewer attacking candidates in a fixed 10-candidate
  budget; the 1-ply score with sampled opponents rarely rewards defence). The S3 gap is about the opponent model,
  not the candidate set → bot_018.

## [018] bot_016 search/predacc — PROMOTED

- family: search | parent: bot_008 | tags: prediction-accuracy-gating
- analysis: classifier check in an S3 game (seed 12, after 1905): Greedy baseline greedy-prediction hit rate 0.91–1.0;
  Random/Attitude 0.08–0.30; the lookahead stand-in 0.64–0.65 with 74–79% target-seeking moves → bot_008 classed it
  'erratic' (uniform moves, low hold prob) so unsupported attacks on it looked good → S3 −1.23 vs bot_003.
- hypothesis: classify by hit rate (≥0.85 greedy, ≥0.4 'strong', else erratic); 'strong' powers get bot_003's worst
  case (every reachable province a full threat, hold prob 0.85) → recover S3 while keeping S1 100%.
- sanity: classes now correct (stand-in → strong); S3 seeds 11/13 (stand-in bot_004) → 2/10 SC, no errors.
- plan: Tier 0 → Tier 1 (A, stand-in bot_001 = bot_008's) → Tier 2 (B210, stand-in bot_004 = bot_008's).
- Tier 0 clean. Tier 1 A42: S1 18.00 / S2 13.36±0.82 (47.6%) / S3 12.29±0.90 (35.7%); vs bot_008 pooled −0.15±0.38 → Tier 2.
- Tier 2 B210 (stand-in bot_004): S1 18.00 (100%) / S2 13.83±0.37 (53.3%) / S3 11.42±0.41 (32.4%) → est. 13.
  vs bot_008: S1 0, S2 +0.79±0.39, S3 +1.47±0.41, pooled +0.75±0.19 → PROMOTE → search champion.
  vs bot_007 (overall): S2 −1.45±0.46, S3 −0.49±0.47, pooled −0.62±0.25 → not overall.
- takeaway: hit-rate classes recover bot_008's S3 loss (+1.47) and keep S1 at 100%. Search still trails lookahead in S2.

## [019] bot_017 positional/threatw — PROMOTED

- family: positional | parent: bot_006 | tags: threat-weighted-safety (uses bot_016's hit-rate classifier)
- hypothesis: bot_006 freezes because any adjacent enemy unit (incl. static/random/attitude) triggers garrisons and
  blocks moves. Weighting enemy units by class (static 0, erratic 0.35, unknown 0.7, greedy/strong 1; garrison at
  expected threat ≥ 0.5) frees units for expansion without dropping defence against real attackers → S2/S3 gain.
- sanity: S1/S2/S3 seed 11 → 18/2/10 SC, no errors, tmax 6 ms.
- plan: spare queue after bot_014: Tier 0 → Tier 1 (A, stand-in bot_004 = bot_006's) → Tier 2 (B210, stand-in bot_004).

## [020] Every-5 review (after 17 iterations) — DONE

- bot_009 (adaptive scenswitch) Tier 2 B126 (stand-in bot_004): S1 18.00 (100%) / S2 15.16±0.39 (62.7%) / S3 11.52±0.52
  (32.5%) → est. 13. vs bot_005: S2 +2.03±0.50, S3 +1.80±0.48, pooled +1.28±0.24 → PROMOTE → adaptive champion.
  vs bot_007 (overall): S2 −0.11±0.45, S3 −0.34±0.51, pooled −0.15±0.23 → tie, not promoted. Tier 1 A42: S1 18.00 /
  S2 14.81±0.79 / S3 11.31±0.95.
- bot_017 (positional threatw) Tier 1 A42: S1 17.71 / S2 10.64 / S3 6.62, pooled +2.84±0.39 vs bot_006. Tier 2 B210
  (stand-in bot_004): S1 17.71±0.05 (85.7%) / S2 10.41±0.38 (23.3%) / S3 7.40±0.35 (5.7%) → est. 9. vs bot_006: +5.43±0.26
  / +1.72±0.31 / +1.11±0.30, pooled +2.75±0.19 → PROMOTE → positional champion. Takeaway: raw-adjacency caution was
  the positional weakness; class weights free the units.
- Tournament (28 games, HoF, 6 bots): 005 5.58±0.78, 008 5.03±0.71, 007 4.91±0.93, 011 4.69±0.66, 006 4.61±0.49,
  012 4.27±0.75 SC. All within ~1 SE: no separation at 28 games (7-seat all-bot games are low-scoring and noisy).
- Held-out C (bot_007, n=42, stand-in bot_003): S1 18.00 (100%) / S2 14.67±0.78 (59.5%) / S3 11.38±0.90 (28.6%),
  in line with B (18.00 / 15.27 / 11.29) → no sign of overfitting seed sets A/B.
- Family review: positional DORMANT (3 iters, est 9 vs 13), valuemap DORMANT (3 iters, est 11 vs 13; S3 only vs weak
  stand-in). greedy/search/lookahead/adaptive ACTIVE (champions at est. 13; greedy's S3 uses weak stand-in bot_001 →
  re-measure S3 with the bot_007 stand-in before trusting it).
- Overall picture: S1 solved (100%) by 005/007/008/009; S2 ≈ 15 SC / ~60% wins (5 pts); S3 ≈ 11.5 SC / ~35% (3 pts,
  5 pts needs >12 SC or >40% wins). S3 is where the remaining marks are.

## [021] bot_018 lookahead/strongopp — REJECTED

- family: lookahead | parent: bot_007 | tags: adversarial-sampling (+ bot_016's hit-rate classes)
- hypothesis: bot_007 labels a strong opponent 'erratic' (move-direction rule) and samples it as mostly random, so
  rollouts underrate attacks by the Hidden Agent. Hit-rate classes + adversarial sampling for 'strong' powers (each
  unit attacks one of our provinces w.p. 0.5, with a supporting unit w.p. 0.7 when possible) → S3 gain.
- sanity S3 (stand-in bot_004) seeds 11–14 → 18/12/8/18 SC, tmax 0.47 s, no errors.
- stand-in: rule gives bot_009, but bot_009 runs bot_004's lookahead code outside S1 (same family in effect) → use
  bot_008 (search, S2 44.8%) for Tier 2 and a matching bot_007 S3 reference. Deviation noted.
- plan: Tier 0 + Tier 1 (A, stand-in bot_003 = bot_007's) on the spare core; Tier 2 (B126, stand-in bot_008) + ref in main4.
- Tier 0 clean. Tier 1 A42: S1 18.00 / S2 14.83±0.79 (61.9%) / S3 11.45±0.93 (28.6%); vs bot_007 pooled +0.44±0.40 → Tier 2.
- Tier 2 B126 (stand-in bot_008): S1 18.00 / S2 14.77±0.43 (58.7%) / S3 12.02±0.54 (38.9%). Ref bot_007 S3 with
  stand-in bot_008: 12.22±0.50 (34.1%). vs bot_007: S2 −0.50±0.45, S3 −0.21±0.54, pooled −0.24±0.23 → REJECT.
- takeaway: modelling strong opponents as attackers in 1-ply rollouts makes play more cautious without saving SCs.
  Two lookahead opponent-model tweaks (015, 018) failed; 1-ply rollouts seem saturated → lookahead needs a different
  lever (deeper search / better score) rather than sampling tweaks.

## [022] bot_019 adaptive/peaceful — REJECTED

- family: adaptive | parent: bot_009 | tags: provocation-avoidance (adaptive backlog item 3)
- baseline reading (agent_baselines.py AttitudeAgent): starts FRIENDLY to all; never moves into provinces (units/SCs)
  of powers it is friendly to; being attacked → neutral/hostile; being supported → back toward friendly.
- hypothesis: a non-static, non-greedy power that has never moved into our provinces is likely a friendly Attitude
  agent. Charging 0.3 score per move into its provinces (in bot_009's lookahead choice) keeps it passive toward us
  while other targets exist → fewer losses in S2/S3.
- sanity S2 seeds 11/12/13 → 5/18/12 SC, tmax 0.48 s, no errors.
- plan: main5 after main4: Tier 0 → Tier 1 (A, stand-in bot_003 = bot_009's) → Tier 2 (B126, stand-in bot_004 = bot_009's).
- Tier 0 clean. Tier 1 A42: S1 18.00 / S2 15.60±0.64 (64.3%) / S3 9.95±0.99 (28.6%); vs bot_009 pooled −0.19±0.44 → Tier 2.
- Tier 2 B126 (stand-in bot_004): S1 18.00 / S2 15.02±0.43 (61.9%) / S3 11.46±0.51 (31.7%). vs bot_009: S2 −0.14±0.49,
  S3 −0.06±0.52, pooled −0.07±0.24 → REJECT (no effect).
- takeaway: Attitude/Random agents are too weak (≈1.5 SC in S2) for their goodwill to matter; not provoking them
  neither helps nor hurts. Their SCs are the growth source, so there is nothing to gain from sparing them.

## [023] S3 re-measure of greedy/valuemap champions — DONE

- Their S3 numbers were against the weak stand-in bot_001. Re-run S3 B126 with stand-in bot_008 (search, strong):
  bot_012 greedy 9.99±0.55 (26.2%), bot_014 valuemap 9.88±0.54 (24.6%) → both est. 11 (was 13 / 11).
- Family review: greedy → DORMANT (3 iters, 11 vs 13); valuemap stays DORMANT. ACTIVE: search, lookahead, adaptive.

## [024] bot_020 search/rolloutsel — PROMOTED

- family: search | parent: bot_016 | tags: rollout-selection (hybrid of search and lookahead ideas)
- hypothesis: bot_016's heuristic eval picks well-coordinated orders but misjudges outcomes (S2 −1.45 vs bot_007).
  Hill climbing for 0.20 s, keeping the top-6 distinct local optima, then choosing among them by 1-ply engine rollouts
  (class-based opponent sampling, bot_004-style outcome score) for the rest of the 0.45 s → closes the S2 gap.
- sanity: S2/S3 seeds 11–12 → 14/1/18/18 SC, tmax 0.455 s; 139–294 rollouts per phase in a test game.
- plan: main6 after main5: Tier 0 → Tier 1 (A, stand-in bot_001 = bot_016's) → Tier 2 (B126 fallback, stand-in bot_004).
- Tier 0 clean (tmax 0.454 s). Tier 1 A42 (spare core): S1 18.00 / S2 15.02±0.73 (54.8%) / S3 13.05±0.91 (45.2%);
  vs bot_016 S2 +1.67±0.76, S3 +0.76±0.73, pooled +0.81±0.35 → Tier 2 (S1 part on the spare core, rest in main6).
- Tier 2 B126 (stand-in bot_004): S1 18.00 (100%) / S2 14.72±0.43 (57.1%) / S3 13.57±0.49 (49.2%) → est. 15.
  vs bot_016: S2 +0.90±0.44, S3 +2.41±0.53, pooled +1.11±0.23 → PROMOTE → search champion.
  vs bot_007 (overall): S2 −0.55±0.51, S3 +1.71±0.53, pooled +0.39±0.25 (< 2 SE) → not promoted overall by the rule,
  although its est. mark is 15 vs 13 (S3 crosses both 5-pt lines). Revisit after bot_021's Tier 2.
- takeaway: heuristic local search as the candidate generator + engine rollouts to choose is the best S3 bot so far.

## [025] bot_021 lookahead/halving — PROMOTED

- family: lookahead | parent: bot_007 | tags: successive-halving
- hypothesis: after two failed opponent-model tweaks (015, 018), change how the rollout budget is spent: 24 candidates
  (bot_007: 10) raced with common opponent samples, dropping the worse half every 3 rounds (min 3 alive) → better
  final choice at the same ~200–330 rollouts per phase.
- sanity: 202–334 rollouts/phase; S2 seeds 11/12 → 18/18 SC, tmax 0.463 s.
- plan: Tier 0 + Tier 1 (A, stand-in bot_003 = bot_007's) on the spare core; Tier 2 (B126, stand-in bot_003) in main7.
- Tier 0 clean (tmax 0.481 s). Tier 1 A42: S1 18.00 / S2 15.19±0.73 (64.3%) / S3 12.88±0.94 (47.6%); vs bot_007
  pooled +1.04±0.42 → Tier 2.
- Tier 2 B126 (stand-in bot_003): S1 18.00 / S2 15.54±0.42 (68.3%) / S3 12.37±0.49 (34.9%) → est. 15.
  vs bot_007: S2 +0.27±0.43, S3 +1.08±0.52, pooled +0.45±0.23 (z=1.96, just under 2 SE), marks 15 vs 13 → borderline.
- decision: both bot_020 (+0.39±0.25) and bot_021 (+0.45±0.23) are est. 15 vs bot_007's 13 but sit just under the 2-SE
  bar on the 126-game fallback. Extend bot_007/020/021 to the standard 210 on S2/S3 (matching stand-ins; S1 saturated
  at 18 for all three) and re-compare (main8 + spare11).
- 210-game results (B; S1 126 all 18.00): bot_007 S2 14.96±0.32 (57.1%), S3 (stand-in 003) 11.49±0.41 (31.4%);
  bot_021 S2 15.70±0.31 (70.0%), S3 (003) 12.66±0.38 (39.5%); bot_020 S2 14.69±0.33 (57.6%), S3 (004) 13.18±0.38 (46.2%).
- bot_021 vs bot_007 (546 paired): S2 +0.75±0.34, S3 +1.17±0.41, pooled +0.74±0.21, marks 15 vs 13 → PROMOTE.
  stress (core 0, bot_022 eval pinned to cores 1–3): ×3.5, tmax 0.526 s, overshoot 0.076 s → PASS →
  lookahead champion + OVERALL CHAMPION (agent_21.py).
- bot_020 vs bot_007 (546 paired): S2 −0.27±0.39, S3 +1.27±0.43, pooled +0.39±0.22 → stays search champion only.
- takeaway: racing more candidates beats spreading rollouts evenly; S3 now ≥12 SC (5 pts on SC) for 021 and 020.

## [026] bot_022 search/hybridrace — PROMOTED

- family: search | parent: bot_020 | tags: hybrid-candidate-race (combines bot_020 and bot_021)
- hypothesis: bot_020 gains in S3 (+1.71 vs bot_007) and bot_021 in S2 (69% wins). One pool = top-8 hill-climbing
  optima (0.15 s) + 12 lookahead candidates (greedy + perturbations with supports), raced by successive halving with
  common opponent samples → keeps both gains.
- sanity: pool 13–20 candidates per phase; S2/S3 seeds 11–12 → 18/18/18/18 SC, tmax 0.46 s.
- plan: main9 after main8: Tier 0 → Tier 1 (A, stand-in bot_001 = bot_020's) → Tier 2 (B126, stand-in bot_004 = bot_020's).
- Tier 0 clean. Tier 1 A42: S1 18.00 / S2 15.90±0.62 (71.4%) / S3 15.21±0.71 (64.3%); vs bot_020 pooled +1.02±0.35.
- Tier 2 B126 (stand-in bot_004): S1 18.00 / S2 16.03±0.39 (77.8%) / S3 14.08±0.50 (57.1%) → est. 15. tmax 0.489 s.
  vs bot_020: S2 +1.31±0.37, S3 +0.51±0.50, pooled +0.61±0.21 → PROMOTE → search champion.
  vs bot_007: S2 +0.76±0.45, S3 +2.22±0.58, pooled +0.99±0.25. vs bot_021 (overall): pending 021 S3 ref with stand-in 004.
- ref bot_021 S3 (stand-in 004, B126): 13.21±0.51 (46.0%). bot_022 vs bot_021: S2 +0.49±0.49, S3 +0.87±0.51,
  pooled +0.46±0.24 (z=1.9, just under 2 SE), marks 15/15 → extend both to 210 on S2/S3 (spare12, core 0).
- 210-game results (stand-in 004): bot_022 S2 16.06±0.28 (75.7%), S3 14.35±0.37 (58.1%); bot_021 S3 13.29±0.38 (45.2%).
  bot_022 vs bot_021 (546 paired): S2 +0.36±0.33, S3 +1.06±0.37, pooled +0.54±0.19 → PROMOTE.
  stress (core 0; main13 pinned to 1–3; redundant spare12 copy stopped): ×4.14, tmax 0.539 s, overshoot 0.089 s → PASS
  → OVERALL CHAMPION (agent_21.py).
- takeaway: the two candidate sources are complementary: best bot in every scenario so far.

## [027] bot_023 search/springthreat — REJECTED

- family: search | parent: bot_022 | tags: spring-threat-score
- hypothesis: bot_022's Spring rollout score counts occupied SCs but not exposure; Fall losses come from own SCs left
  empty next to enemy units. Charging 0.3 per own SC left empty and reachable by a non-static enemy unit after a
  simulated Spring move → fewer Fall losses, S2/S3 gain.
- sanity: S3 (stand-in bot_004) seeds 11–13 → 18/18/18 SC, tmax 0.478 s; rollout path verified (no silent fallback).
- plan: main12 after main11: Tier 0 → Tier 1 (A, stand-in bot_001) → Tier 2 (B126, stand-in bot_004); vs bot_022 and bot_021.
- Tier 1 A42: S1 18.00 / S2 15.71±0.69 / S3 14.93±0.79; vs bot_022 pooled −0.16±0.31.
- Tier 2 B126 (stand-in 004): S1 18.00 / S2 15.85±0.36 (70.6%) / S3 13.79±0.49 (52.4%). vs bot_022: S2 −0.18±0.37,
  S3 −0.29±0.52, pooled −0.16±0.21 → REJECT.
- takeaway: the rollouts already see Spring exposure through the sampled opponent moves; an extra heuristic term only
  biases toward passive garrisons.

## [028] Every-5 review (after 23 iterations) — DONE

- Held-out C (bot_021, n=42, offset 42, stand-in bot_003): S1 18.00 (100%) / S2 16.05±0.61 (73.8%) / S3 13.57±0.87 (50.0%)
  → consistent with B (15.70 / 12.66 at 210): no overfitting.
- Tournament (56 games, 6 family champions, rotating seats): 014 valuemap 7.20±0.68, 022 search-hybrid 6.77±0.76,
  021 lookahead 5.43±0.69, 017 positional 4.80±0.43, 012 greedy 2.83±0.51, 009 adaptive 2.12±0.42 SC.
  → in all-bot tables the cheap valuemap bot does best; the rollout bots' opponent models (static/greedy/erratic mixes)
  fit the baseline scenarios, not tables of strong bots. Relevant for Scenario 4 (bonus) only.
- Family review: adaptive → DORMANT (champion bot_009 est. 13 vs 15; last in tournament). ACTIVE: search (champion
  bot_022), lookahead (bot_021). DORMANT: greedy, valuemap, positional, adaptive.
- S1 solved everywhere; S2 ~16 SC / 70–78% wins; S3 13–14 SC / 46–57% wins (vs lookahead stand-in) → all 5-pt lines
  cleared by 021/022. Remaining work: robustness (timing, unknown Hidden Agent), Scenario 4 behaviour.

## [029] Engine finding: deepcopy and the unit-owner cache — DONE

- `copy.deepcopy(Game)` (engine `Game.__deepcopy__`) deep-copies `_unit_owner_cache` separately from `powers`, so a copy
  whose source had the cache built maps units to Power objects that are not the copy's own. `set_orders` then fails
  `owner is not power` → 'UNORDERABLE UNIT', silently dropped under IGNORE_ERRORS, and `process` treats all units as
  holding. The cache is built by set_orders/process/_unit_owner and by a fresh Game(); `clear_cache()` resets it.
- Our rollout bots (004 onward) are NOT affected: they deep-copy a light game (Game() + set_units/set_centers/
  set_current_phase, which clear the cache) and never order the base itself; verified: copies process orders correctly.
- Speed-up this enables: set the opponents' sampled orders once per round on a copy, `clear_cache()`, then deep-copy
  it per candidate and set only our orders. Measured (S1901, under load): 2.46 → 1.52 ms per rollout (×1.6).

## [030] bot_024 lookahead/fastroll — PROMOTED

- family: lookahead | parent: bot_021 | tags: shared-opponent-rollouts (engine finding [029])
- hypothesis: set sampled opponent orders once per round (+ clear_cache), deep-copy per candidate → ~1.5x rollouts
  in the same 0.45 s (test game: ~230 vs ~160 per phase under load) → sharper choices, small S2/S3 gain.
- checks: all 941 rollouts in a test game moved units (orders applied); S3 seeds 11/12 (stand-in 004) → 18/18, tmax 0.48 s.
- plan: main13 after the bot_022 extension: Tier 0 → Tier 1 (A, stand-in 003 = bot_021's) → Tier 2 (B126, stand-in 003).
- Tier 0 clean. Tier 1 A42: S1 18.00 / S2 15.83±0.59 (71.4%) / S3 11.79±0.94 (33.3%); vs bot_021 pooled −0.15±0.37 → Tier 2.
- Tier 2 B126 (stand-in 003): S1 18.00 / S2 16.17±0.35 (73.0%) / S3 13.21±0.50 (50.8%) → est. 15. vs bot_021:
  S2 +0.63±0.36, S3 +0.85±0.52, pooled +0.49±0.21 → PROMOTE → lookahead champion (overall comparison with bot_022
  needs matching S3 stand-ins; the same change on bot_022 is bot_025).
- takeaway: more rollouts per move still pay off, so rollout count is a binding constraint.
- cross-check vs overall champion (S3 stand-in 004, B126): bot_024 S3 13.09±0.50 (45.2%). vs bot_022: S2 +0.14±0.43,
  S3 −0.99±0.52, pooled −0.28±0.23 → bot_022 stays overall champion (hybrid candidates matter more than rollout count).

## [031] bot_025 search/fastroll — REJECTED

- family: search | parent: bot_022 (overall champion) | tags: shared-opponent-rollouts (same change as bot_024)
- hypothesis: ~1.5x rollouts in the race → small S2/S3 gain for the champion line.
- checks: 1543/1543 test rollouts moved units; S3 seeds 11/12 (stand-in 004) → 18/16 SC, tmax 0.453 s.
- plan: Tier 0 + Tier 1 (A, stand-in bot_001 = bot_022's) on core 0 (spare13); Tier 2 (B126, stand-in 004) in main14.
- Tier 0 clean. Tier 1 A42: S1 18.00 / S2 14.90±0.87 (73.8%) / S3 14.81±0.77 (64.3%); vs bot_022 pooled −0.47±0.39 → Tier 2.
- Tier 2 B126 (stand-in 004): S1 18.00 / S2 16.25±0.34 (74.6%) / S3 13.93±0.52 (57.9%). vs bot_022: S2 +0.21±0.42,
  S3 −0.15±0.52, pooled +0.02±0.22 → REJECT (no effect).
- takeaway: the speed-up helps the pure lookahead line (bot_024 +0.49) but not the hybrid: bot_022 races a smaller,
  stronger pool (hill-climb optima) after 0.15 s of search, so extra rollouts no longer change its choice.

## [032] Champion robustness in S3 (bot_022) — DONE

- S3 B126 with different Hidden Agent stand-ins: Greedy baseline 15.60±0.40 (69.8%); valuemap bot_014 14.34±0.49
  (61.1%); positional bot_017 15.13±0.44 (69.0%); lookahead bot_004 (main runs, B210) 14.35±0.37 (58.1%).
  → S3 5-pt lines (>12 SC, >40% wins) cleared against every opponent style tried; no stand-in-specific overfit.

## [033] test_21.py draft — DONE

- Self-contained experiments file (20 KB): eval (scenarios 1–3, seed-paired, instrumented seat: timeout_decorator 1 s,
  timing, legality, desync; 512 MB RLIMIT_AS per worker), compare (paired diffs), ablate (CONFIG switches off one at a
  time vs the full agent), stress (games pinned to one core), tournament, summary. Uses only game.py,
  agent_baselines.py, diplomacy, numpy, timeout_decorator + stdlib; writes results_21.jsonl.
- Smoke-tested: eval n=1 (S1/S2/S3 18 SC, slowest 0.457 s), ablate ROLLOUT on S2, tournament 1 game.
- Freeze: re-check the ablate default keys against the final champion's CONFIG toggles, run FINAL eval/ablations.

## [034] bot_026 search/convoy — REJECTED

- family: search | parent: bot_022 | tags: convoy-candidates (greedy backlog item 9, never tried)
- analysis (lab/analyze.py bot_022): weakest seats S3 AUS 11.4 (46%), ENG 13.1 (35%), TUR 14.9 (49%); S2 ENG 42% wins
  (lowest). No bot ever issued VIA/convoy orders, so English armies could not leave the island.
- hypothesis: convoy candidates (army VIA move toward an unowned SC + convoy orders for all our fleets that can take
  part; 3 dedicated candidates + 40% of perturbations) let the rollouts pick convoys when they pay → ENG/TUR gain.
- checks: England S2 seed 8 test game: convoys played (e.g. A LON - BEL VIA + F ENG C A LON - BEL); S2 England seeds
  1/8/15/22 → 13/18/18/18 (bot_022: 18/9/14/18); no illegal orders, tmax 0.474 s.
- plan: Tier 0 + Tier 1 (A, stand-in bot_001) on core 0; Tier 2 (B126, stand-in 004) on cores 1–3 after main16.
- Tier 0 clean. Tier 1 A42: S1 18.00 / S2 17.00±0.44 (85.7%) / S3 15.02±0.77 (64.3%); vs bot_022 pooled +0.30±0.29 → Tier 2.
- Tier 2 B126 (stand-in 004): S1 18.00 / S2 16.26±0.35 (77.8%) / S3 14.63±0.50 (66.7%). vs bot_022: S2 +0.23±0.42,
  S3 +0.56±0.47, pooled +0.26±0.21 (z≈1.2) → REJECT. Per power (S2+S3, 36 games each): ENG +0.86±0.78 (wins 17 vs 13),
  AUS +1.11±1.12, ITA −0.25±1.15, TUR −0.75±0.83.
- decision: not extended to 210; extensions were used only at z≈1.9–2.0, and extending only promising-looking bots
  would bias verdicts (optional stopping). Convoys stay available as a measured technique (neutral/positive trend).

## [035] Held-out + 7-bot tournament (after 26 iterations) — DONE

- Held-out C (bot_022, n=42, offset 84, stand-in 003): S1 18.00 / S2 16.95±0.40 (81.0%) / S3 13.19±0.89 (47.6%) → consistent.
- Tournament (56 games; 024, 022, 014, 020, 017, 012, greedy baseline; rotating seats): bot_024 11.36±0.87,
  bot_022 6.89±0.85, bot_014 4.21±0.68, bot_020 3.98±0.64, bot_017 3.71±0.44, bot_012 2.07±0.42, greedy 1.54±0.27 SC.
  → in all-bot tables the pure lookahead line (bot_024) is far stronger than the hybrid champion, the opposite of S3
  (bot_024 −0.99 vs bot_022). Scenario 4 (+3 bonus) is bot-vs-bot, so this matters. Next: a second tournament to
  separate the fast-rollout effect (024 vs 025) from the hill-climb candidates (022 vs 021).

## [036] Tournament 2 (field: 024, 025, 022, 021, 014, 020, greedy; 56 games, offset 100) — DONE

- bot_022 7.52±1.00, bot_024 6.21±0.81, bot_020 5.50±0.88, bot_021 5.41±0.73, bot_025 5.27±0.79, bot_014 2.52±0.42,
  greedy 1.38±0.37 SC.
- takeaway: bot_024's lead in tournament 1 (11.4 vs 6.9) did not replicate; rankings depend heavily on the field.
  Tournaments give no stable reason to prefer the lookahead line for S4 → keep bot_022 (best in S1–S3).

## [037] Early ablations of the champion (bot_022) — DONE

- purpose: measure each technique's contribution inside the champion before the freeze (FINAL seeds stay reserved)
  and find any component that hurts. Seed set A, n=42, stand-in bot_001 (= bot_022's Tier 1), --set one switch off:
  OPP_AWARE, ACC_GATE, LA_CANDS, HALVING, ROLLOUT. Paired vs bot_022 on A (main19).
- results (A42 each, paired vs bot_022 A42; S1 18.00/100% in every variant):
  ROLLOUT=off: S2 13.60 (47.6%), S3 12.90 (42.9%), pooled −1.54±0.36 | LA_CANDS=off: S2 13.45 (42.9%), S3 13.76 (47.6%),
  pooled −1.30±0.38 | HALVING=off: S2 15.60, S3 15.36, pooled −0.06±0.30 | OPP_AWARE=off: S2 16.57, S3 15.14, pooled
  +0.20±0.31 | ACC_GATE=off: S2 16.45, S3 15.36, pooled +0.23±0.30.
- takeaway: inside the hybrid, rollout selection and the lookahead candidates carry the gains; halving and the
  opponent-model terms of the hill-climb heuristic no longer matter (the rollouts decide). Check OPP_AWARE/ACC_GATE off
  on B126 (paired with bot_022's B runs) before considering a simpler champion.
- B126 check (stand-in 004, paired with bot_022): OPP_AWARE=off S2 +0.01±0.40, S3 −0.54±0.43, pooled −0.18±0.20;
  ACC_GATE=off S2 +0.05±0.40, S3 −0.20±0.48, pooled −0.05±0.21. With A (+0.20, +0.23): both neutral inside the hybrid
  → keep bot_022 unchanged (no simpler champion).

## [038] bot_027 search/crossover — REJECTED

- family: search | parent: bot_022 | tags: candidate-crossover (genetic-algorithm-style recombination in the race)
- hypothesis: ablations show candidate quality drives bot_022 (LA_CANDS off −1.30, ROLLOUT off −1.54). After the first
  halving, recombining the top-6 survivors unit by unit (support consistency repaired) adds up to 6 children that can
  beat both parents → S2/S3 gain.
- checks: 2–6 children per race, occasionally picked; no illegal orders; S3 seeds 11/12 (stand-in 004) → 18/18 SC,
  tmax 0.468 s.
- plan: main21: Tier 0 → Tier 1 (A, stand-in bot_001) → Tier 2 (B126, stand-in 004) vs bot_022.
- Tier 1 A42: S1 18.00 / S2 15.95±0.59 (71.4%) / S3 16.50±0.47 (76.2%); vs bot_022 pooled +0.44±0.33.
- Tier 2 B126 (stand-in 004): S1 18.00 / S2 16.14±0.34 (73.8%) / S3 14.49±0.46 (57.9%). vs bot_022: S2 +0.11±0.34,
  S3 +0.41±0.54, pooled +0.17±0.21 → REJECT (n.s.).
- takeaway: recombined children are rarely better than their parents under 1-ply rollouts; the race is near its
  ceiling for this evaluation. Further gains need a different evaluation (deeper lookahead) rather than more candidates.

## [039] bot_028 lookahead/twoply — REJECTED

- family: lookahead | parent: bot_022 (hybrid lineage; the change is a lookahead idea) | tags: two-ply-spring
- hypothesis: three candidate-side changes (025 speed, 026 convoys, 027 crossover) were n.s. → the 1-ply evaluation
  is the ceiling. In Spring, extend each rollout by a greedy Fall reply of every non-static power (dislodged units
  disband) and score SC ownership after Fall instead of the Spring occupancy heuristic → better Spring moves.
- checks: every simulated process moved units; rollouts per phase: Spring 83–137 (half), Fall 179–259 (unchanged);
  S3 seeds 11/12 (stand-in 004) → 18/18, tmax 0.455 s.
- plan: main22: Tier 0 → Tier 1 (A, stand-in bot_001) → Tier 2 (B126, stand-in 004) vs bot_022.
- Tier 1 A42: S1 18.00 / S2 15.86±0.61 / S3 14.95±0.84; vs bot_022 −0.10±0.34.
- Tier 2 B126 (stand-in 004): S1 18.00 / S2 16.07±0.36 (73.8%) / S3 14.94±0.44 (65.9%). vs bot_022: S2 +0.04±0.37,
  S3 +0.86±0.45, pooled +0.30±0.19 (z=1.6) → REJECT (n.s.).
- note: three bot_022 variants lean positive but n.s. (026 convoys +0.26, 027 crossover +0.17, 028 two-ply +0.30).
  Convoys and two-ply target different weaknesses (England; Spring evaluation) → try the combination (bot_029).

## [040] bot_029 lookahead/twoplyconvoy — REJECTED

- family: lookahead | parent: bot_028 | tags: two-ply-spring + convoy-candidates (combination of two measured ideas)
- hypothesis: two-ply (+0.30±0.19, mainly S3) and convoys (+0.26±0.21, mainly England) target different weaknesses;
  combined they should reach a significant gain over bot_022.
- checks: England test game plays convoys (A WAL - BRE VIA, A CLY - NWY VIA with fleet convoys); rollout path runs;
  S3 seeds 11/12 (stand-in 004) → 18/18, tmax 0.456 s.
- plan: main23: Tier 0 → Tier 1 (A, stand-in bot_001) → Tier 2 (B126, stand-in 004) vs bot_022.
- Tier 1 A42: S1 18.00 / S2 16.57±0.53 (83.3%) / S3 15.90±0.65 (73.8%); vs bot_022 +0.45±0.34.
- Tier 2 B126 (stand-in 004): S1 18.00 / S2 16.59±0.31 (81.0%) / S3 14.67±0.43 (61.9%). vs bot_022: S2 +0.56±0.44,
  S3 +0.60±0.48, pooled +0.38±0.22 (z=1.7).
- extension rule (fixed now, applied to all candidates): a Tier 2 (126) result vs the champion with z ≥ 1.5 is extended
  to the standard Tier 2 size of 210 on S2/S3 (S1 saturated). Qualify: bot_028 (z=1.6), bot_029 (z=1.7); not bot_026
  (1.2) or bot_027 (0.8). Caveat: interim look + extension slightly inflates the false-positive rate; any promotion
  from it will also be checked on held-out seed set C.
- 210-game extension (S2/S3, stand-in 004; bot_022 at 210): bot_029 S2 16.62±0.24 (81.0%), S3 14.64±0.36 (65.7%);
  vs bot_022 S2 +0.56±0.31, S3 +0.29±0.41, pooled +0.33±0.20 (z=1.65) → REJECT. bot_028 S2 16.14±0.26 (73.8%),
  S3 14.84±0.36 (66.2%); vs bot_022 S2 +0.08±0.26, S3 +0.50±0.36, pooled +0.22±0.17 (z=1.3) → REJECT (stays rejected).
- takeaway: variants of the champion now land at +0.2–0.4 SC, below what 210 paired games can confirm. The champion
  (est. 15/15) is at diminishing returns; effort shifts to submission checks and freeze preparation.

## [041] Submission check: agent_21.py under the course test.py — DONE

- Ran test.experiment() unmodified (imports StudentAgent from agent_21, StudentAgent() with no args), repeat_nums=2:
  S1 14/14 wins (18.0 SC every power), S2 14/14 wins. No errors. visualize.py imports the agent the same way.

## [042] Tournament 3 (112 games, fresh seeds offset 200) — DONE

- bot_022 8.78±0.69, bot_021 6.56±0.62, bot_024 5.77±0.53, bot_020 4.41±0.46, bot_014 3.38±0.38, bot_017 2.71±0.34,
  greedy baseline 2.15±0.31 SC.
- takeaway: with twice the games, the champion is clearly best in all-bot tables too; tournament 1's bot_024 lead was
  field noise. No S4-specific mode needed.

## [043] Tuning check: bot_022 with TIME_BUDGET 0.5 — DONE

- CLAUDE.md allows stopping the search at 0.5 s; bot_022 uses 0.45. Test --set TIME_BUDGET=0.5 on B126 (stand-in 004),
  paired with bot_022. Parameter tuning only (not a technique). If it helps: new bot file + stress gate (tmax must stay
  < 0.6 s serial, < 0.8 s at ×4).
- result B126: S1 18.00 / S2 15.74±0.40 (70.6%) / S3 13.06±0.53 (51.6%), tmax 0.535 s. vs bot_022: S2 −0.29±0.40,
  S3 −1.02±0.46, pooled −0.44±0.20 (z≈−2.2) → keep 0.45 s. Possible cause: more rounds → more halvings, so the race
  narrows to MIN_ALIVE candidates on few samples; longer budgets need a matching HALVE_EVERY. Not pursued (tuning).

## [044] Every-5 review (after bots 024–029) — DONE

- Held-out: bot_022 on C (offset 84) done in [035]; tournaments 2 and 3 done ([036], [042]); failure analysis of
  bot_022 in [034] (weak seats AUS/ENG/TUR; convoys tried in bot_026).
- Family review: search ACTIVE (champion bot_022, overall), lookahead ACTIVE (champion bot_024; lost to bot_022 in S3
  and in tournament 3). greedy/valuemap/positional/adaptive stay DORMANT (≥3 iters, est. 9–13 vs 15).
- State: champion est. 15/15 (S1 100%, S2 16.1 SC / 76%, S3 14.4 SC / 58% vs lookahead stand-in, 14.3–15.6 SC vs 3
  other stand-ins), stress PASS, test.py check PASS. Last 6 variants all n.s. (+0.2–0.4 SC) → low-intensity mode.

## [045] Tuning check: bot_022 with HALVE_EVERY 6 (was 3) — DONE

- B126 S2/S3 (stand-in 004): S2 16.22±0.36 (79.4%), S3 15.05±0.45 (65.1%). vs bot_022: S2 +0.19±0.27, S3 +0.97±0.54,
  pooled (S2+S3) +0.58±0.30 (z≈1.9) → extension rule [040] applies: extend to 210 on S2/S3. If significant: new bot
  file (HALVE_EVERY=6 default), S1 check, stress gate, held-out C confirmation, then promotion. Tuning, not a technique.
- 210 (S2/S3): S2 16.05±0.28 (76.2%), S3 14.72±0.36 (61.9%). vs bot_022: S2 −0.01±0.22, S3 +0.38±0.42, pooled +0.18±0.24
  → n.s.; the 126-game signal regressed toward zero (as the extension rule is meant to catch). Keep HALVE_EVERY=3.

## [046] bot_030 search/buildroll — REJECTED

- family: search | parent: bot_022 | tags: rollout-builds (adjustment phase was never optimised)
- hypothesis: Winter builds chosen by simulating up to 8 build sets (rule choice, type flips, next-best site, all
  armies, all fleets) through the next Spring+Fall (greedy steps; opponents by class; common random numbers) and
  scoring SC count after Fall beat the distance rule → better army/fleet mix, S2/S3 gain.
- checks: light Winter game accepts builds and processes to Spring; test game: rollout builds 0.40 s, legal.
- plan: main29: Tier 0 → Tier 1 (A, stand-in bot_001) → Tier 2 (B126, stand-in 004) vs bot_022.
- Tier 1 A42: S1 18.00 / S2 16.29±0.56 / S3 15.33±0.72; vs bot_022 +0.17±0.32.
- Tier 2 B126 (stand-in 004): S1 18.00 / S2 15.93±0.37 (74.6%) / S3 14.24±0.49 (58.7%). vs bot_022: S2 −0.10±0.41,
  S3 +0.16±0.49, pooled +0.02±0.21 → REJECT (no effect): the distance rule already picks sensible builds.

## [047] Plateau — maintenance mode until the freeze — DONE

- Last five variants of bot_022 (026 convoys, 027 crossover, 028 two-ply, 029 two-ply+convoys, 030 rollout builds)
  and two tuning checks were all n.s. or worse. Champion bot_022 clears every 5-pt line with margin, passes stress,
  test.py, robustness (4 stand-ins) and the 112-game tournament.
- Mode: no new bots unless a genuinely new idea appears; hourly heartbeats. Freeze tasks at Thu 1 Oct 12:00 AWST:
  FINAL eval (504/scenario), ablations on FINAL seeds (210/scenario, 5 switches), finalise test_21.py, regenerate
  LLM_PROMPTS.md (tools/export_prompts.py), final size/import checks of agent_21.py and test_21.py.

## [048] Back to active mode (human request) — DONE

- 27 Sep 04:20: the human asked to keep generating bots rather than idle. CLAUDE.md updated (no maintenance mode).
- Idea queue (all from champion bot_022, one idea each): 031 risk-averse selection (lower-confidence score instead of
  mean), 032 competitive rollout score (our SCs relative to the strongest rival), 033 UCB bandit allocation instead of
  halving, 034 per-power empirical opponent sampling, 035 value-map term in the rollout score.

## [049] bot_031 search/riskaverse — REJECTED

- family: search | parent: bot_022 | tags: risk-averse-selection
- hypothesis: ranking race candidates by mean − 0.3·sd over sampled opponent replies prefers robust orders over
  gambles → fewer SC losses (esp. S3 vs a strong Hidden Agent).
- plan: queue q1 (T0 → T1 A42 stand-in 001 → T2 B126 stand-in 004) vs bot_022.

## [050] bot_032 search/rivalscore — REJECTED

- family: search | parent: bot_022 | tags: competitive-score
- hypothesis: subtracting 0.3 × the strongest rival's SC count after the simulated move rewards taking SCs from the
  leader (win condition, strong Hidden Agent) → S3 wins.
- plan: queue q1 after bot_031.
- results q1 (T1 A42 stand-in 001 / T2 B126 stand-in 004, paired vs bot_022):
  bot_031: T1 S2 16.81, S3 16.40 (pooled +0.70±0.29); T2 S2 15.93±0.39 (76.2%), S3 13.96±0.49 (57.1%), pooled −0.07±0.21
  → REJECT. bot_032: T1 pooled +0.39±0.33; T2 S2 16.01±0.37 (76.2%), S3 14.47±0.46 (61.1%), pooled +0.12±0.21, tmax
  0.552 s (S1) → REJECT.
- takeaway: Tier 1 gains keep vanishing at Tier 2. Tier 1's S3 uses the weak bot_001 stand-in; against the lookahead
  stand-in the differences disappear. Neither the selection rule nor the score's rival term changes outcomes.

## [051] bot_033 search/cbrace — REJECTED

- family: search | parent: bot_022 | tags: confidence-bound-racing
- hypothesis: dropping a candidate only when its upper bound (mean + 1.0·sd/√n) is below the best lower bound, from
  round 3 on, keeps close contenders sampled and drops clear losers early → better final choice than fixed halving.
- plan: queue q2 (after q1).
- Tier 2 B126 (stand-in 004): S1 18.00 / S2 15.77±0.40 (74.6%) / S3 13.90±0.50 (55.6%). vs bot_022: S2 −0.26±0.48,
  S3 −0.18±0.53, pooled −0.15±0.24 → REJECT. The allocation rule (halving vs confidence bounds) does not matter,
  consistent with the HALVING ablation (neutral).
- note: human (27 Sep) also invited completely new bots → new family 'bandit' planned (decoupled per-unit UCB over
  orders, credited from joint rollouts).

## [052] bot_034 bandit/base — PROMOTED (bandit family champion, bootstrap)

- family: bandit (new; human invited completely new bots) | parent: none | tags: decoupled-ucb, combinatorial-bandit
- hypothesis: decoupled per-unit UCB1 bandits over each unit's orders (hold, top-10 greedy moves, supports of own
  units), credited from joint 1-ply rollouts vs class-sampled opponents (~300–400 rollouts/phase, opponent sample shared
  by batches of 4), find coordinated orders without a hand-made candidate generator.
- sanity: S1/S2/S3 seed 11 → 8/1/18 SC, no errors, tmax 0.48 s. Expected weakness: 2v1 needs a move and a support
  chosen jointly, which independent per-unit bandits find slowly (S1 8 SC in the sanity game).
- plan: queue q3 (T0 → T1 A42 → T2 B126) vs bot_022; new family → Tier 2 regardless.
- Tier 1 A42: S1 8.07±0.27 (0%) / S2 13.10±0.85 (45.2%) / S3 12.24±0.88 (40.5%, weak stand-in 001).
- Tier 2 B126 (stand-in 004): S1 7.73±0.15 (0%) / S2 13.29±0.47 (43.7%) / S3 9.50±0.50 (19.0%) → est. 9.
  vs bot_022: −10.27 / −2.75 / −4.58, pooled −5.87±0.28. Bootstrap → bandit family champion.
- takeaway: as predicted, independent per-unit bandits cannot coordinate 2v1 (S1 0 wins, stalls at ~8 SC like the
  plain greedy bots); S2 is decent (13.3). bot_036 (pair arms) targets exactly this.

## [053] bot_035 valuemap/poolsource — REJECTED (dormant family revisited)

- family: valuemap (revisit) | parent: bot_022 | tags: multi-source-candidates
- hypothesis: candidate diversity drives bot_022 (LA_CANDS off −1.30); adding the valuemap champion's (bot_014) joint
  order as a third source lets the rollouts use its 2v1/strength-aware plans when they are better.
- checks: pool 14–21 per phase incl. the VM candidate; S2/S3 seed 12 sanity below; no errors.
- plan: queue q4 (after q3).
- Tier 1 A42: pooled +0.15±0.33 vs bot_022. Tier 2 B126 (stand-in 004): S1 18.00 / S2 16.48±0.30 (77.0%) /
  S3 14.63±0.45 (62.7%); vs bot_022 S2 +0.44±0.33, S3 +0.55±0.47, pooled +0.33±0.19 (z≈1.7) → extension rule [040]:
  extend to 210 on S2/S3 (core 0, 1 worker).

## [054] bot_036 bandit/pairs — PROMOTED

- family: bandit | parent: bot_034 | tags: pair-arms
- hypothesis: independent per-unit bandits rarely draw a move and its support together. Pair arms ("move into an
  enemy-held target SC, supported by unit Y"; Y's order is overridden and not credited) make 2v1 attacks one draw.
- sanity: S1 seeds 11/12 → 8/18 SC, S2 11/12 → 4/18 SC, no errors, ~500 rollouts/phase.
- plan: queue q5 (after q4).
- Tier 2 B126 (stand-in 004): S1 11.75±0.40 (15.1%) / S2 15.34±0.36 (60.3%) / S3 12.33±0.48 (34.9%) → est. 11.
  vs bot_034: S1 +4.02±0.35, S2 +2.06±0.40, S3 +2.83±0.56, pooled +2.97±0.26 → PROMOTE → bandit champion.
  vs bot_022: pooled −2.89±0.28. S1 still stalls (~12 SC): units do not gather next to static holders.

## [055] Sparring field, round 1 (human request) — DONE (results in [060], [061])

- bot_037 greedy/homedef (parent bot_012, rule-based): hold / support-hold / garrison own SCs that an *active* enemy
  (one that has ever ordered a move) can enter, before attack matching. First version garrisoned against static units
  (S1 seed 12: 11 SC) → restricted to active powers (then 18). Sanity S2 seeds 12/13 → 8/7 SC.
- bot_038 valuemap/season (parent bot_014, rule-based): Fall SC values ×1.5 with 2 diffusion passes, Spring full 6.
  Sanity S1 18/18, S2 18/3, S3 13/18.
- plan: queue q6 (queue_family.sh): 037 vs bot_012 (T1 greedy, T2 bot_001 B210); 038 vs bot_014 (T1/T2 bot_001, B210).
- bot_039 positional/frontline (parent bot_017, rule-based): idle rear units outside enemy reach step toward the
  nearest enemy-owned SC. Sanity S1 18/16, S2 8/6, S3 12/1.
- new family 'archetype' (human permission): one fresh rule engine, three styles as separate bots —
  bot_040 aggressive (enemy SCs first, all supporters into attacks, no defence), bot_041 turtle (garrison +
  support-hold first, neutral SCs within 2 only, 3v1 attacks), bot_042 opportunist (weakest reachable power's SCs).
  Sanity S1/S2 seed 12: 040 18/18, 041 16/9, 042 18/5; tmax ≤ 3 ms, no errors.
- plan: queue q7: 039 vs bot_017 (T1/T2 bot_004, B126); 040–042 vs bot_012 for reference (T1 greedy, T2 bot_001, B210).
- bot_043 adaptive/hitrate (parent bot_005, style-pure rules): hit-rate opponent classes (≥0.85 greedy, ≥0.4 strong,
  else erratic); 'strong' units threaten every adjacent province and probably hold (0.85). Sanity S1 18/18, S2 16/9,
  S3 18/2 (stand-in 004). plan: queue q8 vs bot_005 (T1 bot_003, T2 bot_004, B126).

## [056] Classifier check vs differently styled bots — DONE

- bot_022 in AUSTRIA vs six of our styles (to S1908): all → 'strong': bot_012 hit 0.71 hold 0.00, bot_014 0.47/0.30,
  bot_040 0.62/0.06, bot_041 0.40/0.07, bot_042 0.49/0.44, bot_024 0.68/0.00. With baselines in the field: Greedy →
  greedy (0.86), Attitude → erratic (0.18), Random → erratic (0.10). → vs other groups' agents bot_022 will treat almost
  every competent bot with one cautious model (30/20/50 mix, worst-case threat): robust but not style-exploiting.

## [057] bot_044 search/empmix — REJECTED

- family: search | parent: bot_022 | tags: empirical-opponent-mix
- hypothesis: blending each opponent's class mix with its own observed hold / greedy-hit / other rates (weight
  n/(n+10)) fits the varied 'strong' styles ([056]) better → gains vs varied opponents (S3, S4).
- checks: rollout path runs; S3 seeds 12/13 (stand-in 004) → 18/7, tmax 0.454 s.
- plan: queue q9 (queue_bots.sh vs bot_022); also include it in the final style tournament.
- bot_035 210 extension (S2/S3, stand-in 004): S2 16.48±0.23 (76.2%), S3 14.70±0.35 (63.3%); vs bot_022 S2 +0.41±0.27,
  S3 +0.36±0.37, pooled +0.30±0.17 (z=1.76) → REJECT (n.s.).

## [058] bot_045 search/stack — PROMOTED (overall champion)

- family: search | parent: bot_029 (+ bot_035's valuemap candidate) | tags: stacked-near-misses
- motivation: four independent changes each land at ≈ +0.2–0.35 SC vs bot_022 at 210 games (026, 028, 029, 035),
  none significant alone. Stacking the two best (029: two-ply + convoys; 035: valuemap candidate) tests additivity.
- checks: pool 16–21, VM candidate picked in a test game; S3 seeds 12/13 → 18/6 SC, tmax 0.472 s.
- plan: queue q10 (queue_bots.sh vs bot_022; extension rule applies).

## [059] bot_046 bandit/staticprior — REJECTED

- family: bandit | parent: bot_036 | tags: holder-aware-priors
- hypothesis: priors that encode static holders (unsupported move into one −6, pair arm against one +4, move ending
  next to a static-held target SC +1) let the bandits set up 2v1 in S1 (bot_036: 11.75 SC, 15% wins).
- sanity S1 seeds 11–14 → 18/18/9/7 SC, no errors.
- plan: queue q11 vs bot_036 (T1 bot_001, T2 bot_004 B126).

## [060] Sparring field round 1 results: bot_037, bot_038 — DONE

- bot_037 greedy/homedef: Tier 2 B210 (stand-in 001): S1 12.29±0.33 (28.6%) / S2 9.80±0.38 (17.1%) / S3 8.54±0.33
  (9.5%) → est. 5. vs bot_012 pooled −2.41±0.19 → REJECTED: garrisoning every threatened SC ties down too many units.
- bot_038 valuemap/season: Tier 1 +0.60±0.33 vs bot_014. Tier 2 B210 (stand-in 001): S1 18.00 (100%) / S2 13.83±0.38
  (54.8%) / S3 12.32±0.41 (40.0%) → est. 15 (S3 vs weak stand-in). vs bot_014 pooled +1.02±0.17 → PROMOTED → valuemap
  champion. A 35–55 ms rule bot at est. 15: strong sparring partner and a fast fallback design. S3 to be re-measured
  with a strong stand-in for the final field.

## [061] Sparring field round 1 results: bots 039–042 — DONE

- bot_039 positional/frontline: Tier 2 B126 (stand-in 004): S1 17.71 (85.7%) / S2 10.48±0.48 (24.6%) / S3 7.82±0.44 (9.5%);
  vs bot_017 pooled +0.05±0.20 → REJECTED (no effect).
- bot_040 archetype aggressive (B210, stand-in 001): S1 18.00 (100%) / S2 15.91±0.31 (76.7%) / S3 15.94±0.30 (77.1%);
  vs bot_012 pooled +4.00±0.23. **Paired vs bot_022 (S1+S2): S1 0, S2 −0.15±0.39 → statistically tied with the
  champion against the baseline mix**, at 40 ms per move. → archetype family champion. S3 vs a strong stand-in
  (bot_004, B210) running on core 0 for a paired S3 comparison.
- bot_041 turtle: S1 14.29 (0%) / S2 6.53 / S3 5.76 — passive by design (sparring only).
- bot_042 opportunist: S1 17.59 (81.4%) / S2 8.08 / S3 7.29 — focusing one victim leaves it exposed.
- takeaway: against the provided baselines, relentless supported aggression is nearly as good as the simulation
  champion. Ideas: bot_040's plan as a candidate in bot_022's race; bot_040 is a key S4 sparring partner.

## [062] bot_047 search/aggcand — REJECTED

- family: search | parent: bot_022 | tags: multi-source-candidates (aggressive archetype plan)
- hypothesis: bot_040's all-out aggressive plan ties bot_022 in S2; adding it to bot_022's race lets the rollouts use
  aggression when it pays → S2/S3 gain.
- checks: generator returns legal plans (e.g. S1901 France: A MAR - SPA, F BRE - ENG, A PAR - BUR); S2 seeds 12/13 →
  18/10 SC, no errors.
- plan: queue q12 (queue_bots.sh vs bot_022).
- bot_043 adaptive/hitrate (B126, stand-in 004): S1 18.00 (100%) / S2 12.87±0.49 (42.9%) / S3 9.87±0.52 (19.8%);
  vs bot_005 pooled −0.03±0.23 → REJECTED (no effect): the rule plan does not use the finer threat model much.
- bot_040 S3 vs strong stand-in (bot_004, B210): 12.41±0.45 (48.6%). Paired vs bot_022: S1 0, S2 −0.15±0.39,
  S3 −1.93±0.41, pooled −0.80±0.22. → aggression ties the champion against the baselines but loses clearly against a
  strong Hidden Agent; simulation pays where it matters (S3). bot_040 still est. 15 (S3 just > 12): fast strong sparring bot.
- bot_044 Tier 2 B126 (stand-in 004): S1 18.00 / S2 15.90±0.37 (73.0%) / S3 14.44±0.47 (57.1%). vs bot_022: S2 −0.13±0.39,
  S3 +0.36±0.46, pooled +0.07±0.20 → REJECT: per-power sampling rates do not beat the class mixes.
- bot_045 Tier 2 B126 (stand-in 004): S1 18.00 / S2 16.73±0.31 (85.7%) / S3 15.33±0.44 (69.8%) → est. 15.
  vs bot_022: S2 +0.70±0.41, S3 +1.25±0.48, pooled +0.65±0.21 (z≈3.1) → PROMOTE. The separate ≈+0.3 gains add up.
  stress (core 0; queue pinned to 1–3): ×4.16, tmax 0.543 s, overshoot 0.093 → PASS → OVERALL CHAMPION (agent_21.py).
- caveat: combining earlier near-misses is a garden-of-forking-paths risk → confirmation queued: 210-game extension
  (S2/S3) and held-out C (n=126, fresh offset) vs bot_022, plus a test.py check of the new agent_21.py.

## [063] Shared with the group — DONE

- 27 Sep: human (away, remote control) asked for a PR so the group can see the work. `gh` is not installed (and
  CLAUDE.md forbids installing extra tools), so the local main was pushed to a new remote branch `lab/progress` and a
  pre-filled GitHub compare link was given to open the PR. Remote `main` was not touched. Pushing is otherwise still
  local-only per CLAUDE.md; this push was an explicit human request.
- bot_046 Tier 2 B126 (stand-in 004): S1 14.24±0.42 (54.8%) / S2 14.45±0.43 (54.8%) / S3 11.41±0.56 (35.7%).
  vs bot_036: S1 +2.48±0.43, S2 −0.89±0.39, S3 −0.92±0.59, pooled +0.22±0.29 → REJECT (trade-off).
  Family review: bandit → DORMANT (3 iters, est. 11 vs 15).

## [064] bot_048 bandit/s1switch — REJECTED (dormant family revisited: clear fix)

- parent bot_046 | tags: scenario-detection. Holder-aware priors only when every opponent with units is static.
- plan: queue q14 vs bot_036 (T1 bot_001, T2 bot_004 B126).
- bot_047 Tier 2 B126 (stand-in 004): S1 18.00 / S2 16.19±0.35 (77.0%) / S3 14.44±0.47 (61.9%). vs bot_022: S2 +0.16±0.38,
  S3 +0.36±0.46, pooled +0.17±0.20 → REJECT (n.s.; the race already covers most of what the aggressive plan offers).

## [065] Refinements on the new champion: bot_049, bot_050 — REJECTED

- bot_049 search/stackseason (parent bot_045): valuemap candidate uses bot_038's season weights (bot_038 beat bot_014
  by +1.02 stand-alone).
- bot_050 search/stackagg (parent bot_045): + bot_040's aggressive plan as a race candidate (+0.17 on bot_022 in 047).
- checks: both generators return legal plans; S3 seed 12 (stand-in 004) → 18/18, tmax ≤ 0.466 s.
- plan: queue q15 (queue_family.sh) vs bot_045 (T1 bot_001, T2 bot_004 B126).

## [066] bot_045 confirmation — DONE

- B 210 (S2/S3, stand-in 004): S2 16.60±0.25 (82.4%), S3 15.15±0.35 (69.0%); vs bot_022 S2 +0.53±0.33, S3 +0.80±0.41,
  pooled +0.51±0.20.
- Held-out C (n=126, offset 200, stand-in 004; neither bot selected on these seeds): bot_045 S2 16.79±0.26 (81.0%),
  S3 14.86±0.48 (69.0%); bot_022 S2 15.59±0.39 (69.8%), S3 13.12±0.54 (52.4%); paired diff S2 +1.21±0.41,
  S3 +1.74±0.60, pooled +0.98±0.24 (z≈4) → promotion confirmed on unseen seeds.
- test.py (course script, S2, repeat 1): runs cleanly, 71% wins.
- takeaway: several individually non-significant ideas targeting different weaknesses combined into a clear gain.
- bot_048 Tier 2 B126 (stand-in 004): S1 14.13±0.42 (54.0%) / S2 14.91±0.39 (57.1%) / S3 11.75±0.49 (32.5%). vs bot_036:
  S1 +2.37±0.43, S2 −0.43±0.36, S3 −0.58±0.53, pooled +0.46±0.26 → REJECT (n.s.). The S1 gain is real; S2/S3 within
  noise. Not extended: the extension rule targets champion challengers, and bandit is a dormant sparring family.
- bot_049 Tier 2 B126: S2 16.92±0.29 (88.1%) / S3 15.21±0.43 (68.3%); vs bot_045 S2 +0.19, S3 −0.12, pooled +0.02±0.20.
- bot_050 Tier 2 B126: S2 16.99±0.27 (88.1%) / S3 14.98±0.45 (68.3%); vs bot_045 S2 +0.26, S3 −0.34, pooled −0.03±0.20.
  → both neutral: the champion's pool already covers these plans. (tmax 0.523–0.529 s in S2, possibly host load.)

## [067] Scenario 4 sparring evaluation (champion bot_045 vs the style field) — DONE

- field (one champion per family = seven styles): 045 champion, 024 lookahead, 038 valuemap, 012 greedy, 017 positional,
  036 bandit, 040 aggressive archetype.
- (1) tournament, 112 games, rotating seats, offset 300; (2) bot_045 S3 B126 with each of the six other styles as the
  Hidden Agent stand-in. Queue q16. Re-run at the freeze if the champion changes.
- Tournament (112 games, 7 styles, rotating seats): bot_045 10.19±0.69, bot_024 5.14±0.51, bot_036 4.88±0.56,
  bot_040 4.16±0.52, bot_038 3.29±0.39, bot_017 3.22±0.34, bot_012 2.65±0.44 SC → champion scores 2x the next style.
- bot_045 S3 (B126) with each style as the Hidden Agent: vs 017 positional 15.91±0.40 (76.2%), vs 012 greedy 15.68±0.38
  (71.4%), vs 036 bandit 15.05±0.45 (69.0%), vs 038 valuemap 14.89±0.46 (67.5%), vs 024 lookahead 13.97±0.50 (57.9%),
  vs 040 aggressive 13.75±0.52 (59.5%). All above the S3 5-pt lines; hardest: aggressive and lookahead opponents.

## [068] Tuning sweep on bot_045 — DONE

- The rollout score weights and pool sizes were set in bot_004/020/022 and never tuned. --set variants on B126 S2/S3
  (stand-in 004), paired with bot_045: R_W_UNIT=1.0 (0.6), R_W_DIST=0.1 (0.05), N_LA=20 (12), SEARCH_BUDGET=0.10 (0.15).
  Extension rule applies (z ≥ 1.5 → 210); a winner becomes a new bot file (tuning, not a technique). Queue q17.

## [069] bot_051 search/policyopp — REJECTED

- family: search | parent: bot_045 | tags: policy-opponent-model
- motivation: [067] hardest Hidden-Agent styles are aggressive (13.75) and lookahead (13.97). 'strong' powers are
  sampled from a fixed mix; now, per rollout round, a strong power plays with p=0.5 the orders our aggressive rule
  engine (bot_040) would give as that power (computed once per turn from its point of view).
- check: in a 1901–1904 test game, 24/24 policy predictions computed for the 3 opponents classed strong.
- plan: queue q18 vs bot_045 (T1 bot_001, T2 bot_004 B126); also re-run its S3 vs bot_040 and bot_024 if promising.
- results (B126 S2/S3 vs bot_045): R_W_UNIT=1.0 S2 +0.51 S3 −0.54 pooled −0.02±0.27; R_W_DIST=0.1 +0.48/−0.68, −0.10±0.31;
  N_LA=20 +0.38/−0.38, 0.00±0.28; SEARCH_BUDGET=0.1 S2 17.37 (92.1%) +0.63, S3 15.68 (72.2%) +0.36, pooled +0.50±0.26
  (z≈1.9) → extension rule: SEARCH_BUDGET=0.1 to 210 on S2/S3 (queued after bot_051).
- note: all four variants are +0.4–0.6 in S2 → bot_045's own B126 S2 (16.73) was probably on the low side; the 210
  comparison uses its 210-game S2 (16.60).
- bot_051 Tier 2 B126 (stand-in 004): S1 18.00 / S2 17.11±0.23 (85.7%) / S3 14.78±0.44 (61.9%). vs bot_045: S2 +0.38±0.24,
  S3 −0.55±0.48, pooled −0.06±0.18 → REJECT: predicting a strong (lookahead) opponent with an aggressive rule policy
  misleads the rollouts as much as it helps.
- SEARCH_BUDGET=0.1 at 210 (S2/S3, stand-in 004): S2 17.06±0.21 (88.6%), S3 15.26±0.33 (69.0%); vs bot_045 S2 +0.46±0.26,
  S3 +0.11±0.32, pooled +0.29±0.21 (z≈1.4) → n.s.; keep 0.15.

## [070] bot_052 search/stack2 — REJECTED

- family: search | parent: bot_045 | tags: stacked-near-misses (round 2)
- stack: SEARCH_BUDGET 0.10 (+0.29±0.21 on 045 at 210) + candidate crossover (027: +0.17 on 022) + competitive rival
  score term (032: +0.12 on 022). Crossover also keeps convoy orders consistent (fleet convoy → hold if its army
  is not convoying in the child).
- checks: pool 15–27 with children, a child picked in a test game; S3 seeds 12/13 → 18/18, tmax 0.454 s.
- plan: queue q20 vs bot_045 (T1 bot_001, T2 bot_004 B126); extension rule applies.
- bot_052 Tier 2 B126 (stand-in 004): S2 16.90±0.28 (84.1%) / S3 14.87±0.44 (63.5%); vs bot_045 S2 +0.17±0.36,
  S3 −0.45±0.52, pooled −0.09±0.21 → REJECT. Stacking round 2 fails: these smaller effects were mostly noise.
- search family at 38% of iterations (cap 40%) → next bots from other families.

## [071] bot_053 archetype/balanced, bot_054 lookahead/twoply — REJECTED

- bot_053 (parent bot_040): new STYLE 'balanced' = aggressive targeting and all-in supported attacks + hold threatened
  own SCs against active enemies (support-hold with 2+ adjacent). Question: does minimal defence close bot_040's S3 gap
  (−1.93 vs the champion against a strong stand-in)? Sanity S1/S2/S3 seed 12 → 18/18/18, 1–3 ms per move.
- bot_054 (parent bot_024): bot_028's two-ply Spring rollouts on the pure lookahead line. All 1152 test rollouts moved
  units; S3 seeds 12/13 (stand-in 003) → 18/2.
- plan: queue q21: 053 vs bot_040 (T1 greedy, T2 bot_004 B210, the stand-in that exposed bot_040), 054 vs bot_024
  (T1/T2 bot_003, B126).
- bot_053 Tier 2 B210 (stand-in 004): S1 18.00 / S2 10.83±0.42 (29.5%) / S3 8.40±0.39 (13.8%); vs bot_040 S2 −5.09, S3 −4.02,
  pooled −3.03±0.23 → REJECT: tying units to defence cripples the aggressive style (same lesson as bot_037).
- bot_054 Tier 2 B126 (stand-in 003): S2 16.67±0.30 (81.0%) / S3 12.70±0.50 (42.1%); vs bot_024 S2 +0.49, S3 −0.52,
  pooled −0.01±0.24 → REJECT: two-ply only paid off inside the hybrid stack.

## [072] Failure analysis of bot_045 in S3 — DONE

- S3 (all stand-ins, n=1137): 777 wins; non-wins ≥14 SC 65, 8–13 SC 141, <8 SC 154 → losses are mostly early collapses,
  not near-misses. In collapses the Hidden-Agent stand-in ends strongest in 104/154 (Greedy 50). By power: Austria
  92/162 wins, Turkey 87/162, England 96/162, Germany 107/163 vs France 142/162, Russia 145/163.
- defence-based fixes have failed repeatedly (015, 023, 037, 053) → try a different lever: opening quality (bot_055).

## [073] bot_055 search/openings — REJECTED

- family: search | parent: bot_045 | tags: opening-book-candidates (search 21/56 iterations = 37.5%, under the cap)
- three standard S1901 openings per power (published opening theory; our transcription, all legal) added to the first
  race. Check: they enter the pool (Austria picked a book opening; Germany/England preferred generated plans).
- plan: queue q22 vs bot_045 (T1 bot_001, T2 bot_004 B126).
- bot_055 Tier 2 B126 (stand-in 004): S2 16.96±0.28 (86.5%) / S3 15.29±0.43 (68.3%); vs bot_045 S2 +0.23, S3 −0.03,
  pooled +0.07±0.21 → REJECT: the race already finds sound openings.

## [074] bot_056 greedy/convoy — REJECTED (dormant family, style-pure)

- family: greedy | parent: bot_012 | tags: convoys (greedy backlog item 9)
- armies that cannot approach any target over land take the VIA move whose landing is closest to a target, with every
  fleet of ours that can convoy it ordered to convoy (multi-fleet chains work: A CLY - DEN VIA with F NTH + F NWG).
- sanity England S2 seeds 1/8 → 14/10 SC, no errors. plan: queue q23 vs bot_012 (T1 greedy, T2 bot_001 B210).
- bot_056 Tier 2 B210 (stand-in 001): S1 12.29 (28.6%) / S2 14.09±0.36 (53.8%) / S3 12.15±0.39 (38.1%); vs bot_012 S1 0,
  S2 +0.60±0.33, S3 +0.07±0.36, pooled +0.22±0.16 → REJECT (n.s.; England +0.21±0.36 over 90 games).

## [075] bot_057 adaptive/vulture — REJECTED (adaptive family revisited)

- parent bot_045 | tags: vulture-candidate. A power is 'weakened' if a third power stands on one of its SCs, ≥2 other
  (not ours) units are adjacent to its SCs, or it owns fewer SCs than at the start of the year. One extra race
  candidate = greedy plan aimed only at weakened powers' SCs.
- checks: 4–12 vulture target SCs per phase in a random-opponent test game; S3 seeds 12/13 (stand-in 004) → 18/12 SC.
- plan: queue q24 vs bot_045 (T1 bot_001, T2 bot_004 B126).
- bot_057 Tier 2 B126 (stand-in 004): S2 17.19±0.26 (89.7%) / S3 15.35±0.42 (70.6%); vs bot_045 S2 +0.46±0.38,
  S3 +0.02±0.46, pooled +0.16±0.20 → REJECT (n.s.).

## [076] Leave-one-out ablation of the champion's stack (bot_045) — DONE

- The last eight bot_045 variants all landed within ±0.2 SC (noise band). Report-relevant question instead: how much
  does each stacked part contribute inside bot_045? --set TWO_PLY=false / CONVOYS=false / VM_CANDS=false on B126 S2/S3
  (stand-in 004), paired with bot_045. Queue q25.
- results (B126 S2/S3, stand-in 004, paired with bot_045):
  TWO_PLY off: S2 16.25 (80.2%) −0.48±0.39, S3 14.83 (62.7%) −0.49±0.45, pooled −0.48±0.30;
  CONVOYS off: S2 16.72 −0.01±0.35, S3 14.98 −0.35±0.52, pooled −0.18±0.31;
  VM_CANDS off: S2 16.56 −0.17±0.36, S3 14.33 (62.7%) −0.99±0.53, pooled −0.58±0.32.
- takeaway: all three parts help inside the stack (consistent with additivity); valuemap candidate matters most (S3),
  then two-ply Spring, convoys least. Report: leave-one-out table next to the one-at-a-time results (026/028/035).

## [077] bot_058 search/stackfast — REJECTED

- parent bot_045 | tags: shared-opponent-rollouts. Two-ply halves Spring rollouts, so the ×1.5 speed-up (neutral on
  bot_022 in 025) may matter here. Check: 1583/1586 processes moved units (the rest are retreat steps in the two-ply
  reply); S3 seeds 12/13 → 18/7. plan: queue q26 vs bot_045 (T1 bot_001, T2 bot_004 B126).
- bot_058 Tier 2 B126: S2 17.13±0.24 (88.1%) / S3 14.98±0.44 (68.3%); vs bot_045 S2 +0.40, S3 −0.35, pooled +0.02±0.21
  → REJECT: rollout count is not the bottleneck for bot_045 either.

## [078] New families: evolution (bot_059), ensemble (bot_060) — PROMOTED (family champions, bootstrap)

- bot_059 evolution/base (new family, genetic-algorithm): population of 12 joint plans (greedy plan + mutations),
  per generation all plans vs the same 2 fresh opponent samples (shared-opponent copies), keep the better half by
  running mean fitness, refill by uniform crossover + 15% per-unit mutation, until 0.45 s; play the best mean.
  Sanity S1/S2/S3 seed 12 → 18/18/18, tmax 0.453 s.
- bot_060 ensemble/vote (new family, ensemble-voting): per-unit majority vote of three rule planners (aggressive
  bot_040, valuemap bot_014-style, greedy no-bounce); ties aggressive > valuemap > greedy; support repair. No
  simulation, ~2 ms/move. All three voters verified. Sanity S1/S2/S3 seed 12 → 18/18/18.
- plan: queue q27, both vs bot_045 for reference (T1 bot_001, T2 bot_004 B126); new families → Tier 2 regardless.
- bot_059 Tier 2 B126 (stand-in 004): S1 18.00 / S2 15.56±0.40 (68.3%) / S3 11.98±0.53 (38.9%) → est. 13. vs bot_045
  S2 −1.17, S3 −3.34, pooled −1.51±0.24 → evolution family champion (bootstrap). GA over whole plans is clearly weaker
  than hill-climb + candidate race within 0.45 s.
- bot_060 Tier 2 B126 (stand-in 004): S1 18.00 / S2 16.33±0.38 (80.2%) / S3 12.94±0.53 (47.6%) → est. 15, ~40 ms/move,
  no simulation. vs bot_045 S2 −0.40, S3 −2.39, pooled −0.93±0.25; vs bot_040 (same seeds) S2 +0.14, S3 +0.83, pooled
  +0.33±0.23 → ensemble family champion; the strongest pure rule bot so far (a fast fallback design and a strong
  S4 sparring partner).

## [079] bot_061 ensemble/vote4 — REJECTED

- parent bot_060 | tags: ensemble-voting + a 4th voter (bot_012's greedy attack-matching plan). All voters verified.
- sanity: S1 seeds 13/14/15 → 6/18/18 (bot_060: 18/18/18): with four voters 2–2 ties become common and the tie-break
  can pick a bad mix (Turkey, seed 13). Evaluated as is (fast rule bot) to measure it.
- plan: queue q28 vs bot_060 (T1 bot_001, T2 bot_004 B126).

## [080] bot_062 evolution/seeded — PROMOTED (evolution family champion)

- parent bot_059 | tags: genetic-algorithm, seeded-population. Initial population also holds the plans of three rule
  planners (aggressive, valuemap, greedy no-bounce), genes extended where needed.
- bug caught before evaluation: the copied greedy planner read CONFIG['CONVOYS'] (missing) → that seed was silently
  skipped. Fixed (keys added, convoys off). A check over all 62 bot files found no other missing CONFIG keys.
- sanity S1/S2/S3 seed 13 → 18/18/8 SC, tmax 0.465 s. plan: queue q29 vs bot_059 (T1 bot_001, T2 bot_004 B126).
- bot_061 Tier 2 B126 (stand-in 004): S1 16.29±0.38 (85.7%) / S2 14.61±0.48 (65.1%) / S3 12.71±0.57 (53.2%); vs bot_060
  S1 −1.71, S2 −1.71, S3 −0.23, pooled −1.22±0.30 → REJECT: per-unit mixing on 2–2 ties breaks plan coordination.

## [081] bot_063 ensemble/medoid — REJECTED

- parent bot_061 | tags: plan-level-consensus. Play the whole proposal that agrees most with the others (medoid) —
  keeps each plan coherent. Sanity S1 12/13 → 18/8, S2 → 18/18, S3 → 5/7. plan: queue q30 vs bot_060 (T1 001, T2 004).
- bot_062 Tier 2 B126 (stand-in 004): S1 18.00 / S2 16.69±0.29 (82.5%) / S3 14.39±0.47 (57.9%) → est. 15. vs bot_059:
  S2 +1.13±0.33, S3 +2.40±0.55, pooled +1.18±0.22 → PROMOTE → evolution champion. Seeding the GA with rule planners'
  plans is what makes evolution competitive.
- bot_062 vs champion bot_045 (same seeds): S2 −0.04±0.36, S3 −0.94±0.49, pooled −0.33±0.20 → not overall.

## [082] bot_064 evolution/twoply — REJECTED

- parent bot_062 | tags: two-ply-spring (GA fitness in Spring = SC ownership after a greedy Fall reply; LOO showed two-ply
  worth ≈0.5 SC inside bot_045). Sanity S2/S3 seed 13 → 18/7. plan: queue q31 vs bot_062 (T1 bot_001, T2 bot_004 B126).
- bot_063 Tier 2 B126 (stand-in 004): S1 10.43±0.40 (14.3%) / S2 14.06±0.47 (54.8%) / S3 12.01±0.55 (39.7%); vs bot_060
  S1 −7.57, S2 −2.26, S3 −0.93, pooled −3.59±0.32 → REJECT: the medoid is the most 'typical' plan, usually a cautious one
  without supported attacks (fatal in S1). Ensemble family: 3 iterations (060/061/063), champion bot_060 (est. 15) → ACTIVE.
- bot_064 Tier 2 B126: S2 16.28±0.34 (77.8%) / S3 13.90±0.50 (56.3%); vs bot_062 S2 −0.41, S3 −0.49, pooled −0.30±0.23
  (vs bot_045 −0.63±0.23) → REJECT: the GA needs many fitness evaluations; two-ply halves them. Evolution family:
  3 iterations (059/062/064), champion bot_062 (est. 15) → ACTIVE.

## [083] Tournament with the new families (112 games) — DONE

- field: bot_045 (champion), bot_062 (evolution), bot_060 (ensemble), bot_024 (lookahead), bot_040 (aggressive),
  bot_038 (valuemap), bot_036 (bandit); rotating seats, offset 400. Queue q32.
- bot_045 8.46±0.69, bot_062 6.54±0.66, bot_024 5.33±0.51, bot_040 4.44±0.58, bot_036 3.78±0.43, bot_038 2.56±0.27,
  bot_060 2.29±0.41 SC. Champion first again; seeded GA second. The ensemble (est. 15 vs the baseline mix) finishes last
  among strong bots: pure rule planners do well against weak opponents only.
- family review pending one measurement: bot_038 S3 vs a strong stand-in (only measured vs bot_001 so far) → q33.
- test_21.py: default ablation keys updated for the new champion's switches (ROLLOUT, LA_CANDS, VM_CANDS, TWO_PLY,
  CONVOYS, HALVING, ACC_GATE, OPP_AWARE); all present in agent_21.py; smoke-tested (ablate TWO_PLY, 1 game). 20 KB.

## [084] Family review (after bots 045–064) — DONE

- bot_038 S3 vs strong stand-in (bot_004, B126): 11.00±0.55 (34.9%) → est. 13 (S3 3 pts) → valuemap DORMANT.
- ACTIVE: search (champion bot_045), lookahead (bot_024, est 15), evolution (bot_062, est 15), ensemble (bot_060, est 15
  vs baselines but last in the strong-bot tournament), archetype (bot_040, sparring). DORMANT: greedy, valuemap,
  positional, adaptive, bandit.

## [085] bot_065 evolution/poolga — PROMOTED (evolution family champion)

- family: evolution | parent: bot_045 (code) / bot_062 (idea) | tags: genetic-algorithm, seeded-population
- GA replaces bot_045's halving race: population = bot_045's candidate pool (hill-climb optima, lookahead candidates,
  valuemap plan, convoys); genes = each unit's orders seen in the pool + hold; per generation 2 shared opponent samples
  (two-ply in Spring), keep the better half, refill by uniform crossover + 10% mutation with support/convoy repair.
- checks: rollout path verified (evolved plan appended and returned each move); S3 seeds 12/13 → 18/18, tmax 0.455 s.
- plan: queue q34 vs bot_045 (T1 bot_001, T2 bot_004 B126).
- bot_065 Tier 2 B126 (stand-in 004): S1 18.00 / S2 17.17±0.23 (88.1%) / S3 15.60±0.39 (70.6%) — best S2/S3 on these
  seeds so far. vs bot_045 S2 +0.44±0.29, S3 +0.27±0.50, pooled +0.24±0.19 (z≈1.3 < 1.5: no extension) → not overall.
  vs bot_062 S2 +0.48, S3 +1.21, pooled +0.56±0.19 → PROMOTE → evolution family champion.

## [086] bot_066 evolution/racega — REJECTED

- parent bot_065 | tags: race-then-evolve. First half of the rollout time: halving race over the pool; then the GA
  breeds only from the survivors (race totals carried over). Checks: rollout path ok; S3 seeds 12/13 → 18/10.
- plan: queue q35 vs bot_065 (T1 bot_001, T2 bot_004 B126).
- bot_066 Tier 2 B126: S2 17.04±0.28 (88.9%) / S3 14.36±0.49 (62.7%); vs bot_065 S2 −0.13, S3 −1.24, pooled −0.46±0.19
  → REJECT: racing first discards plans the GA could recombine usefully.

## [087] bot_067 adaptive/seataware — REJECTED

- parent bot_045 | tags: seat-aware-score. Central seats (AUS/GER/ITA): +0.5 extra cost per own SC lost in the rollout
  outcome, unit weight 1.0 (vs 0.6). Found while writing it: with two-ply Spring bot_045 never uses R_W_LOST (Spring is
  always scored on post-Fall ownership), so a lost own SC cost exactly as much as a gained one.
- checks: rollout path ok (Austria seat); S3 Austria seeds 7/14 → 6/18. plan: queue q36 vs bot_045 (T1 001, T2 004).
- bot_067 Tier 2 B126: S2 16.81±0.27 (82.5%) / S3 15.17±0.44 (69.8%); vs bot_045 pooled −0.03±0.21 → REJECT.
  By seat (S2+S3): central (AUS/GER/ITA, changed code) −0.60±0.58; other seats (identical code to bot_045) +0.38±0.33 —
  i.e. run-to-run noise alone produces ≈ ±0.35 SC differences on 126 games (useful calibration for the report).

## [088] bot_068 lookahead/riskaverse — REJECTED

- parent bot_045 | tags: risk-averse-selection. Race ranks candidates (halving + final pick) by mean − 0.5·sd of their
  rollout scores instead of the mean (RISK toggle). Target: S3 early collapses next to a strong opponent.
- sanity S3 seed 7 (Austria): 12 SC, tmax 0.46 s, no errors. plan: queue q37 T0 X3, T1 A42 (stand-in 001),
  T2 B126 (stand-in 004), compare vs bot_045.
- side work: tools/export_prompts.py now drops <task-notification>/compaction/interrupt turns and keeps /loop prompts
  (dry run: 40 entries, 0 notifications).
- Tier 0 X3 clean (tmax 0.473 s). Tier 1 A42 (stand-in 001): S1 18.00 / S2 16.14±0.67 (78.6%) / S3 16.76±0.53 (85.7%);
  vs bot_045 S2 −1.48±0.67, S3 −0.52±0.58, pooled −0.67±0.30 (< −2 SE) → REJECT at Tier 1; Tier 2 stopped early
  (partial raw games kept). Takeaway: penalising outcome spread favours passive plans; bot_045's mean is the better
  criterion. Risk aversion, if any, belongs in the S3/S4 opponent model, not the selection rule.
- human direction (28 Sep): next priorities are (a) a fast own move resolver for rollouts, validated against the
  engine, and (b) S4 fixes vs the styles bot_045 is weakest against (aggressive bot_040, lookahead bot_024).

## [089] bot_069 adaptive/oppsupport — REJECTED

- parent bot_045 | tags: coordinated-opponent-model (S4 priority, human direction 28 Sep). Rollout sampling: for
  opponents classified 'strong', a sampled holder supports one of its power's sampled moves when legal (p 0.8).
  bot_045 samples units independently, so sampled attacks were almost never supported; the aggressive and lookahead
  styles (hardest Hidden Agents in [067]) do support attacks.
- checks: support string format matches the engine; instrumented 10-phase game vs bot_040 field: 290 coordinated
  rounds, 39 supports added. S3 seeds 7/8/9 vs stand-in 040: 4/8/18 (bot_045: 1/18/4) — noise, no errors.
- plan: queue q38: T0 X3; T1 A42 (001) vs bot_045; S3 B126 with stand-ins 040 and 024 (bot_045 refs from [067]);
  T2 B126 (004) vs bot_045.

## [090] Found bug: army adjacency at split-coast provinces (all bots) — bot_070 search/mapfix PROMOTED (overall champion)

- Found while validating an own resolver: map_info (copied into every bot since bot_001) built army neighbours from
  map.loc_abut, which lists coast-qualified names ('SPA/SC', 'BUL/EC', 'STP/NC') and keys SPA/BUL/STP in lower case.
  20 army links were missing vs map.abuts (MAR/GAS/POR-SPA, CON/GRE/RUM/SER-BUL, FIN/LVN/MOS/NWY-STP, both ways);
  SPA/BUL/STP had no army neighbours at all. Fleet adjacency was correct. Effect: army BFS distances treated three
  SCs as unreachable (or far), armies there as stuck, and greedy predictions of opponents' armies were wrong nearby.
  Engine-side results (rollouts, legality) were never affected: orders always come from the engine's possible orders.
- bot_070 = bot_045 + fixed map_info (base names for army neighbours, lower-case key fallback); verified 0 missing /
  0 extra links for armies and fleets vs map.abuts. Sanity S2 seeds 1/2/13: 18/18/11, tmax 0.458 s, no errors.
- plan: queue q39 (after q38): same protocol as bot_069 (T0, T1 A42, S3 vs 040/024 B126, T2 B126 vs bot_045).

## [091] bot_071 lookahead/fastres — PROMOTED (lookahead family champion; not overall)

- parent bot_070 (map fix needed: the resolver's convoy/adjacency test uses army reach) | tags: own-fast-resolver
  (human priority, 28 Sep). Own movement resolver replaces engine copies in the rollouts: move decisions resolved
  recursively with guess-and-check for cycles (circular movement succeeds; convoy paradox: convoyed moves fail);
  supports/cuts incl. the "attack from the supported-into province cuts only by dislodging" rule; head-to-head,
  prevent strength, own-unit dislodgement ban; convoy routes with dislodged fleets removed.
- validation (lab/validate_resolver.py, new): random games to 1912 with structured random orders (moves; supports of
  chosen moves 80%; convoys of chosen VIA moves). Engine conventions found and matched: (1) VIA with no convoy route
  to an adjacent province moves overland; (2) an adjacent army move is convoyed when its own power orders a convoy
  route; (3) own-power supports DO count against an own unit for explicit VIA / non-adjacent attacks (land attacks:
  not); (4) coast-specific supports must name the move's coast. Result: identical unit positions in 14,080/14,080
  movement phases (8 seeds) + 2,640/2,640 with the code as pasted in the bot; ~42 us vs ~600 us per engine.process.
  Dislodged lists differ in ~2/1760 phases only in reporting (units with no retreat removed at once), positions same.
- rollouts per movement phase (France vs six bot_040s, loaded machine): 131 (engine) → 4,461 (resolver), ×34.
  Sanity S1/S2/S3 seeds 4/12: 18/18, 18/18, 13/18; tmax 0.456 s; no errors.
- plan: queue q40: T0, T1 A42 vs bot_045, S3 B126 vs stand-ins 040/024, T2 B126 (004) vs bot_045 and bot_070.
  FAST_RES False = bot_070 (ablation switch for the report).

## [092] bot_072 evolution/fastga — REJECTED (not significant)

- parent bot_065 | tags: own-fast-resolver, genetic-algorithm (human asked to keep the evolution line going). bot_065's
  GA fitness rollouts moved onto bot_071's resolver (+ the [090] map fix it needs). Resolver copy validated 660/660.
- generations per movement phase (Turkey vs six bot_040s, loaded machine): 3.4 → 138; rollouts 125 → 4,447.
  Sanity S2/S3 seeds 5/13: 18/18/18/18, tmax 0.463 s, no errors.
- plan: queue q41: T0, T1 A42 vs bot_045, S3 B126 vs stand-ins 040/024, T2 B126 (004) vs bot_045 and bot_065.

- bot_069 results (appended after [092]): Tier 0 clean. T1 A42 vs bot_045 pooled −0.04±0.17. T2 B126 (004): S1 18.00 /
  S2 17.36±0.19 (89.7%) / S3 14.37±0.46 (57.9%). Paired vs bot_045 by S3 stand-in (lab/by_standin.py, new):
  vs 040 aggressive 14.70, +0.94±0.47; vs 024 lookahead 14.21, +0.25±0.56; vs 004 14.37, −0.61±0.47. S2 +0.63±0.34.
  compare (S3 pooled over the three stand-ins): pooled +0.17±0.19 → REJECT (below 2 SE).
- takeaway: modelling supported opponent attacks helps against the aggressive style (the S4 weak spot, z≈2.0) and in
  S2, but not against bot_004. Retest on the resolver line, where rollout noise is far lower (backlog).

- bot_070 results (appended after [092]): Tier 0 clean. T1 A42 vs bot_045 pooled −0.21±0.20 (pass). T2 B126 (004): S1 18.00
  / S2 16.98±0.27 (87.3%) / S3 15.71±0.43 (75.4%). vs bot_045: S2 +0.25±0.32, S3 (3 stand-ins, 378) +0.75±0.31,
  pooled +0.50±0.20 → PROMOTE. By stand-in: 004 +0.73±0.49, 024 lookahead +0.98±0.58 (14.94), 040 aggressive
  +0.88±0.55 (14.63) — consistent gains, incl. both S4 weak-spot styles.
- stress (evals pinned to cores 1–3, stress on core 0): slowdown ×3.66, tmax 0.518 s, 0 errors → PASS. Promoted:
  search family champion + overall champion; agent_21.py = bot_070.
- test.py (course script via its experiment(), repeat 1, S1 + S2): 7/7 wins each, 18.0 SC, no errors.
- takeaway: a correct army map is worth ≈0.5 SC; the found bug cost every earlier bot. bot_071/072 (resolver) already
  build on the fixed map. Style sparring bots still carry the bug (backlog).

- bot_071 results (appended after [092]): Tier 0 clean; T1 A42 vs bot_045 −0.13±0.20. T2 B126 (004): S1 18.00 / S2
  16.89±0.28 (85.7%) / S3 15.55±0.43 (74.6%); S3 vs 040 15.00, vs 024 14.71.
  vs parent bot_070 (isolates the resolver): S2 −0.10±0.23, S3 −0.01±0.28, pooled −0.02±0.17 → no gain.
  By stand-in vs 070: 004 −0.16, 024 −0.23, 040 +0.37 (all n.s.). vs bot_045 +0.47±0.20 (= the map fix).
  vs lookahead champion bot_024: pooled +1.06±0.23 → lookahead family champion.
- takeaway: ×34 rollouts buy nothing once the race has ~130 rollouts per phase: selection is no longer sampling-limited;
  what limits it now is the opponent model / score and the candidate set. Resolver still useful where evaluations are
  scarce (GA, bot_072) and for bigger candidate pools or richer opponent models at no time cost.

## [093] bot_073 lookahead/widepool (lookahead family champion; not overall), bot_074 adaptive/oppsupres (REJECTED)

- after bot_071's null result (rollouts no longer binding), two ways to use the spare rollouts, both on bot_071:
- bot_073 (tags: wide-candidate-pool): TOP_K 8 → 16, N_LA 12 → 36; measured pool 15–21 → 30–51 candidates per phase.
- bot_074 (tags: coordinated-opponent-model): bot_069's supported-attack sampling for 'strong' powers on the resolver
  line (lower rollout noise; bot_069 was +0.94±0.47 vs the aggressive stand-in).
- sanity S2/S3 seed 11 vs stand-in 040: both 18/18, tmax ≤ 0.451 s, no errors.
- plan: queues q43/q44 (after q41 bot_072 and q42 held-out): T0, T1 A42 vs bot_071, S3 B126 vs 040/024, T2 B126 (004)
  vs bot_070 (overall) and bot_071 (parent).

- bot_072 results (appended after [093]): Tier 0 clean; T1 A42 S2 17.45 / S3 16.36. T2 B126 (004): S1 18.00 / S2 17.32±0.25
  (92.1%, best S2 of any bot) / S3 15.67±0.42 (74.6%); S3 vs 040 15.28, vs 024 14.97.
  vs bot_065 (parent): pooled +0.07±0.18; vs bot_070 (overall): S2 +0.33±0.29, S3 +0.21±0.28, pooled +0.19±0.18
  (z 1.06 < 1.5: no extension); vs bot_045 +0.69±0.19. By stand-in vs 070: 004 −0.04, 024 +0.02, 040 +0.64±0.52.
- takeaway: ×40 GA generations and the map fix together add only +0.07 over bot_065, so the GA was not evaluation-limited
  either; bot_065 ≈ bot_070 ≈ bot_071 ≈ bot_072 on these seeds (all within ±0.2). Evolution line stays level with
  the champion (best S2), not ahead.

## [094] bot_075 evolution/confirm (PROMOTED: overall champion), bot_076 lookahead/vmscore (REJECTED)

- bots 065/070/071/072 tie within ±0.2 and neither rollout count (071) nor GA generations (072) was binding, so the
  next two target selection bias and the score function:
- bot_075 (parent bot_072; tags evolve-then-confirm): GA stops 0.07 s early; its 5 best plans race on fresh opponent
  samples (~143 rounds each) and the best fresh mean is played. Instrumented: changed the GA's pick in 2/6 phases.
- bot_076 (parent bot_071; tags valuemap-rollout-score): rollout score adds 0.02 × valuemap value of each of our units
  after the rollout (values 20–57 per unit at the start → ≈0.4–1.1 each; neighbouring provinces differ ≈0.1–0.3).
- sanity S2/S3 seed 16 vs stand-in 040: 075 18/18, 076 18/9; tmax ≤ 0.455 s; no errors.
- plan: queues q45/q46 (after q42–q44): T0, T1 A42 vs parent, S3 B126 vs 040/024, T2 B126 (004) vs bot_070 and parent.

## [095] Held-out C check of bot_070 vs bot_045 (offset 400, n=126, S2/S3, stand-in 004) — DONE

- bot_070 S2 17.25±0.19 (86.5%) / S3 16.20±0.37 (78.6%); bot_045 S2 17.10±0.22 (84.1%) / S3 15.54±0.40 (69.0%).
  Paired: S2 +0.14±0.25, S3 +0.66±0.44, pooled +0.40±0.25 — same direction and size as seed set B (+0.50±0.20);
  fresh seeds neither bot was selected on → the map-fix gain is not a seed-set artefact. (compare's "REJECT" and
  marks 10 only reflect the missing S1 and a single-set test.)

- bot_073 results (appended after [095]): T1 A42 vs bot_071 S2 −0.10, S3 −0.19 (pass). T2 B126 (004): S1 18.00 / S2
  17.43±0.22 (92.9%) / S3 15.77±0.38 (72.2%); S3 vs 040 15.16, vs 024 15.52.
  vs bot_070: S2 +0.44±0.29, S3 (378) +0.39±0.25, pooled +0.32±0.16 → PROMOTE (z≈2.0, borderline);
  vs parent bot_071: pooled +0.35±0.17 → REJECT (just under 2 SE). By stand-in vs 070: 004 +0.06, 024 +0.58, 040 +0.52.
- borderline on both → [040] extension rule: q47 extends 073/070/071 to B210 (S2/S3, stand-in 004) and adds held-out C
  (offset 400; bot_070 already has it) before any promotion.

- bot_074 results (appended after [095]): T1 A42 vs bot_071 S3 −0.95±0.59 (pass). T2 B126 (004): S1 18.00 / S2 16.73±0.31
  (84.9%) / S3 15.65±0.40 (71.4%); S3 vs 040 15.14, vs 024 14.88. vs parent bot_071 pooled +0.05±0.18; vs bot_070
  +0.03±0.18; by stand-in vs 071: 004 +0.10, 024 +0.17, 040 +0.14 → REJECTED.
- takeaway: bot_069's +0.94 vs the aggressive stand-in did not replicate on the low-noise resolver line → most likely
  noise (one of several stand-in splits). Coordinated-opponent sampling closed as no measurable effect.

## [096] bot_077 lookahead/agcand — REJECTED

- parent bot_073 (still under extension) | tags: archetype-plan-candidate. The aggressive archetype plan (bot_040's
  engine as ported into bot_062) joins the race as one more candidate. Rationale: candidate diversity is the only
  lever that has paid lately (073 +0.35 vs 071; 062 seeding +1.18; 035 valuemap candidate +0.30).
- checks: archetype plan produced in 8/8 movement phases (0 exceptions); sanity S1/S2/S3 seed 20 → 18/18/18, tmax
  0.452 s. File 66 KB (< 100 KB).
- plan: queue q48 (after q45–q47): T0, T1 A42 vs bot_073, S3 B126 vs 040/024, T2 B126 (004) vs bot_070 and bot_073.

- bot_075 results (appended after [096]): T1 A42 vs bot_072 pooled +0.06±0.25. T2 B126 (004): S1 18.00 / S2 17.12±0.29
  (91.3%) / S3 16.51±0.34 (83.3%, best S3 so far); S3 vs 040 15.40, vs 024 15.63.
  vs bot_070 (overall): S2 +0.13±0.32, S3 (378) +0.75±0.26, pooled +0.48±0.17 → PROMOTE (z 2.8).
  vs bot_065 (evolution champion; S3 paired on stand-in 004 only): S2 −0.05, S3 +0.91±0.43, pooled +0.29±0.18 (z 1.6);
  vs parent bot_072 +0.29±0.17. By stand-in vs 072: 004 +0.84, 024 +0.67, 040 +0.12.
- family comparison borderline → [040] extension: q49 extends 075/065/070 to B210 (S2/S3) + held-out C for 075.
- takeaway so far: the confirmation race (fresh samples for the GA's finalists) helps mainly in S3 → the GA's
  best-by-mean was biased toward plans lucky on few samples (winner's curse), which matters most vs a strong opponent.

- bot_076 results (appended after [096]): T1 A42 vs bot_071 −0.17±0.23. T2 B126 (004): S1 18.00 / S2 17.10±0.26 (87.3%) /
  S3 16.72±0.31 (84.9%); S3 vs 040 14.81, vs 024 14.75. vs parent bot_071 pooled +0.24±0.17 (z 1.4 < 1.5, no
  extension); vs bot_070 +0.22±0.17. By stand-in vs 071: 004 +1.17±0.37, 024 +0.03±0.50, 040 −0.19±0.48 → REJECTED.
- takeaway: the valuemap score helps a lot against bot_004 but not against the stronger lookahead/aggressive styles, so
  it is not a general gain. Possible follow-up: stack on bot_075 if that promotes (different mechanism: score vs
  selection); low priority.

## [097] bot_078 evolution/widepool — REJECTED (extension + held-out: no gain over bot_075)

- parent bot_075 (under extension) | tags: wide-candidate-pool. Stacks the two positive signals: bot_073's wider pool
  (TOP_K 16, N_LA 36) seeds bot_075's GA (with confirmation race). Sanity S2/S3 seed 22 vs stand-in 024: 18/18,
  tmax 0.452 s, no errors.
- plan: queue q50: T0, T1 A42 vs bot_075, S3 B126 vs 040/024, T2 B126 (004) vs bot_070 and bot_075.

- bot_073 extension (q47, B210 S2/S3, stand-in 004): 073 S2 17.27±0.19 (91.9%) / S3 16.04±0.27 (75.2%); 070 S2 17.01 /
  S3 15.50; 071 S2 16.93 / S3 15.65. vs bot_070 pooled +0.38±0.15 (PROMOTE on B); vs bot_071 +0.34±0.15 (PROMOTE).
- held-out C (offset 400) vs bot_070: S2 −0.09±0.27, S3 −0.59±0.45, pooled −0.34±0.26 → does NOT confirm.
  Inverse-variance B+C: ≈ +0.20±0.13 (z 1.5). Decision: lookahead family champion (beats bot_071 on B210), NOT overall
  champion (borderline B gain not confirmed on fresh seeds; bot_070 stays).
- takeaway: seed set B has now picked many winners; a +0.3–0.4 B gain for a borderline bot can be a selection
  effect. Held-out C is the deciding test for every further overall promotion (bot_075 pending in q49).

- bot_077 results (appended after [097]): T1 A42 vs bot_073 −0.28±0.31. T2 B126 (004): S1 18.00 / S2 17.27±0.23 (91.3%) /
  S3 15.66±0.41 (73.8%); S3 vs 040 15.44, vs 024 15.85. vs parent bot_073 pooled +0.07±0.15 → REJECTED (vs bot_070
  +0.39±0.16 on B only inherits bot_073's unconfirmed B gain). By stand-in vs 073: 004 −0.11, 024 +0.33, 040 +0.29.
- takeaway: the archetype plan rarely wins the race once the pool is wide; candidate diversity saturates.

## [098] Failure analysis of champion bot_070 (after bots 068–077) — DONE

- 1176 games. S1 100% every power. S2 (378): weakest ENG 15.7/72% and AUS 16.1/76% (FRA 98%, RUS 100%, TUR 93%);
  45/378 games lose ≥3 SC from peak; worst 20% mostly vs greedy opponents (70/75); S2 falls from 18.0 (0 greedy
  opponents) to 16.6 (4–5 greedy). Worst seeds: ENG collapses to 1–3 SC.
- S3 (630, three stand-ins): AUS 13.6/60%, GER 14.1/63%, ITA 14.5/63%, ENG 15.4/60% vs FRA 89%, RUS 92%, TUR 79%;
  72/630 lose ≥3 SC from peak; worst-20% strongest opponent spread over 004 (40), 040 (31), 024 (26), greedy (29).
- unchanged picture since [072]: central powers and England lose early, mostly to greedy/strong neighbours.

## [099] bot_079 search/buildmix — REJECTED (n.s.)

- parent bot_070 | tags: build-choice-k-nearest. From [098]: replay of S2 seed 100010 (England) showed W1901 'A EDI B'
  (LON lost in 1901; army and fleet tie at distance 2 to LON, ties go to armies), then A EDI and A LON held on the
  island for the rest of the game (2 of 5–7 units idle). Build type now by the mean of the 3 smallest target distances
  (capped at 12): England EDI army 8.67 vs fleet 2.0 → fleet; continental armies unaffected when several land SCs
  are near.
- sanity: S2 100010 ENG 13 (bot_070 replay 12; recorded 1), S2 8 ENG 18, S3 15 ENG 16, S1 1 ENG 18; tmax 0.476 s.
- plan: queue q51: T0, T1 A42 vs bot_070, S3 B126 vs 040/024, T2 B126 (004) vs bot_070; check the England seat split.

## [100] bot_075 promoted: overall champion (evolution family)

- extension (q49, B210 S2/S3, stand-in 004): 075 S2 17.17±0.21 (91.4%) / S3 16.70±0.24 (84.8%); 065 S2 16.70 / S3 15.27;
  070 S2 17.01 / S3 15.50. vs bot_070 pooled +0.59±0.15 (S3 over 3 stand-ins +0.94±0.24); vs bot_065 +0.73±0.18.
- held-out C (offset 400) vs bot_070: S2 −0.06±0.25, S3 +0.25±0.37, pooled +0.10±0.22 (same direction; unlike 073).
  Inverse-variance B+C ≈ +0.43±0.12. Win rates on C: S2 90.5% vs 86.5%, S3 84.1% vs 78.6%.
- stress (evals pinned to cores 1–3): slowdown ×4.26, tmax 0.495 s, 0 errors → PASS.
- promoted: evolution family champion + overall champion; agent_21.py = bot_075 (61 KB). Champion history now
  … → bot_045 → bot_070 → bot_075. First overall champion from a non-search family since bot_021.
- test.py (course script via experiment(), repeat 1, core 0): S1 7/7 wins 18.0 SC; S2 5/7 wins, 15.0 SC (7 games; runs clean).
- test_21.py ablation defaults updated: ROLLOUT LA_CANDS VM_CANDS TWO_PLY CONVOYS FAST_RES CONFIRM ACC_GATE OPP_AWARE
  (HALVING is dead code in the GA line). No GA on/off switch exists (the GA replaced the race in bot_065); an
  ablation for the GA itself = compare with bot_070/bot_071 (same machinery, race instead of GA).

- bot_078 results (appended after [100]): T1 A42 vs bot_075 +0.24±0.27. T2 B126 (004): S1 18.00 / S2 17.44±0.21 (92.9%) /
  S3 16.61±0.34 (83.3%); S3 vs 040 15.84, vs 024 16.01. vs bot_075 (now champion): S2 +0.33±0.28, S3 +0.31±0.23,
  pooled +0.25±0.15 (z 1.67 ≥ 1.5 → [040] extension); vs bot_070 +0.73±0.16. By stand-in vs 075: 004 +0.10,
  024 +0.37, 040 +0.44. q53: B210 extension + held-out C (offset 400) vs bot_075.

- bot_079 results (appended after [100]): T1 A42 vs bot_070 −0.04±0.25. T2 B126 (004): S1 18.00 / S2 17.58±0.18 (92.9%) /
  S3 15.36±0.43 (71.4%). vs bot_070: S2 +0.60±0.27, S3 (378) +0.01±0.25, pooled +0.13±0.16 → REJECT. By power
  (S2+S3, 72 each): ENG +0.43±0.64, ITA +0.68±0.67, TUR +0.11, FRA +0.17, GER +0.15, AUS −0.69±0.65, RUS +0.25.
- takeaway: the stuck-army builds are real but rare; effect below what 630 games resolve. Candidate for a later stack
  onto the champion together with other small fixes (backlog).

## [101] Tournament (7 family champions, 112 games, offset 500) + family review — DONE

- field: 075 evolution (champion), 070 search, 073 lookahead, 036 bandit, 040 archetype, 038 valuemap, 060 ensemble.
- result: bot_075 8.96±0.73, bot_070 7.66±0.66, bot_073 5.82±0.69, bot_036 4.34±0.43, bot_040 2.53±0.41, bot_038 2.13±0.35,
  bot_060 1.91±0.33 SC; 0 issues. New champion first in the all-bot table as well as vs baselines.
- family review (iterations incl. 068–079): evolution ACTIVE (champion line; 078 extending); search ACTIVE (070, 2nd in
  tournament); lookahead ACTIVE (073 family champion); adaptive DORMANT (069/074 opponent-support idea closed as noise);
  archetype ACTIVE for sparring only; ensemble ACTIVE-sparring (est. 15 vs baselines but last among strong bots; no
  further iterations planned); bandit, greedy, valuemap, positional stay DORMANT.

- bot_078 extension (q53, appended after [101]): B210 S2 17.48±0.16 (92.9%) / S3 16.66±0.26 (84.8%); vs bot_075 S2
  +0.31±0.21, S3 (462) +0.20±0.21, pooled +0.20±0.13 → REJECT. Held-out C (offset 400): S2 17.06 (87.3%) / S3 16.57
  (84.1%); vs bot_075 pooled −0.00±0.26. → REJECTED; bot_075 stays champion.
- takeaway: the wider pool adds nothing once the GA with confirmation is in place (candidate breadth saturated, as in
  077). Remaining gaps between the top bots are below what 210 + 126 games resolve (≈ ±0.13–0.25).

## [102] bot_080 evolution/stack — REJECTED (n.s.)

- parent bot_075 | tags: build-choice-k-nearest, valuemap-rollout-score. Stack of two near-misses measured on other
  parents: bot_079's build rule (+0.13±0.16 vs 070; S2 +0.60) and bot_076's valuemap rollout score (+0.24±0.17 vs
  071). Different mechanisms (builds vs move scoring). Sanity: S2 100010 ENG 18, S3 15 ENG 18, S3 16 FRA 18; tmax 0.450 s.
- plan: queue q54: T0, T1 A42 vs bot_075, T2 S1 B126 + S2/S3 B210 (004) vs bot_075, held-out C (offset 400) vs bot_075.
- results: T0 clean (tmax 0.451 s, 181 MB serial). T1 A42 vs 075 +0.34±0.24. T2: S1 18.00 (100%); B210 S2 17.39±0.17
  (92.4%) / S3 16.86±0.21 (84.8%); vs bot_075 S2 +0.22±0.21, S3 +0.16±0.29, pooled +0.14±0.14 → REJECT.
  Held-out C: S2 17.49 (92.9%) / S3 16.45 (81.0%); vs 075 pooled +0.15±0.19. Combined ≈ +0.14±0.11 (n.s.).
- timing note: one move of 0.715 s in held-out C S3 (all other runs ≤ 0.48 s) — likely the valuemap computation in
  _rollout_select plus machine load; another reason not to promote without a stress re-check.
- takeaway: the two small fixes do not add up to a measurable gain on top of the GA champion; bot_075 stays.

## [103] bot_081 evolution/basicswitch (freeze prep: basic technique in the submitted file) — PROMOTED (identity)

- parent bot_075 | tags: basic-greedy-switch. Spec rubric note [3]: the basic technique must be implemented in the
  submitted code. Adds CONFIG BASIC_GREEDY (default False): movement phases play _la_candidates(...)[0], the pure
  BFS-greedy no-self-bounce plan. Diff vs bot_075: the CONFIG key, the agent name and a 2-line early return — default
  play is identical code. Sanity basic mode S1/S2/S3 seed 12 (Russia): 11/18/9 SC, tmax 2 ms; default S3 seed 12: 18.
- plan: q55: T0 (default); basic mode B126 all scenarios (--set BASIC_GREEDY=true, tag basic) for the report's
  basic-technique numbers from the submitted file; then promote bot_081 → agent_21.py by identity (no paired test:
  default behaviour unchanged).

## [104] Failure analysis of champion bot_075 → bot_082 evolution/antilead — REJECTED

- bot_075 (1638 games): S2 AUS 14.9/74% is the one weak seat (ENG 81%, ITA 87%, others 96–100%); S3 AUS 13.1/62%,
  ENG 71%, TUR 74%, ITA 79% (FRA 92%, RUS 97%). Austria S2 is bimodal: 40/54 wins, 14 games at 0–9 SC. Their SC
  trajectories: mostly not early collapses but games that END when a greedy France (sometimes Germany/Russia)
  snowballs to 18 while Austria grows slowly in the centre (e.g. C 300433: FRA 18 in 1907, AUS 4).
- bot_082 (parent bot_081, tags anti-leader-score): once any opponent owns ≥ 10 SCs, rollout outcomes pay 0.5 per SC
  the strongest opponent would own above 12 (ownership after Fall: occupant else previous owner) and 3.0 if it would
  reach 18. Checks: term active in 14 phases from 1907 in S2 C 300433 (no activation in a 7-greedy test game to
  1912); sanity S2 300433 AUS 5 (France still won: too far away), S2 200151 AUS 18; tmax 0.456 s; no errors.
- plan: q56: T0, T1 A42, T2 S1 B126 + S2/S3 B210 (004), held-out C (offset 400); all vs bot_075 (same play as 081).

- bot_081 results (appended after [104]): T0 default X3 clean (tmax 0.467 s, 184 MB serial). Basic mode
  (BASIC_GREEDY=true) B126: S1 7.71±0.12 (0%), S2 10.80±0.48 (21.4%), S3 7.34±0.46 (8.7%, stand-in 004) → est. 5
  (S1 1, S2 3, S3 1); tmax 0.042 s. (bot_001, the original greedy base without no-self-bounce: S1 6.14 / S2 7.43 /
  S3 6.75, est. 2.)
- promoted by identity: default play is bot_075's code path unchanged (diff = CONFIG key + name + inert early return),
  so no paired test or new stress run (bot_075 stress PASS applies). agent_21.py = bot_081; evolution family and
  overall champion records point at bot_081; bot_075's results remain the performance evidence.
- test_21.py ablate accepts KEY=VAL (BASIC_GREEDY=true added to the default keys); smoke-tested n=1 S2: full 18 SC,
  basic 6 SC, 0 issues. 20.7 KB.

- bot_082 results (appended after [103]): T0 clean. T1 A42 vs 075 +0.18±0.27. T2: S1 18.00; B210 S2 17.39±0.17 (91.9%) /
  S3 16.60±0.26 (81.4%); vs bot_075 S2 +0.21±0.21, S3 −0.11±0.31, pooled +0.04±0.14. Held-out C: S2 +0.17±0.21,
  S3 −0.39±0.37, pooled −0.11±0.22 → REJECTED.
- takeaway: penalising a runaway leader does not change results vs the baseline mix: when a greedy France snowballs,
  Austria is usually too far away to act on it. Not tested in all-bot tables; left as an S4 idea only.

## [105] Final S4 sparring evaluation of the champion (human request, 27 Sep) — DONE

- champion bot_081 (plays as bot_075). (1) Style tournament, 112 games, rotating seats, offset 600: 081 evolution,
  070 search, 073 lookahead, 038 valuemap, 012 greedy, 040 aggressive archetype, 036 bandit. (2) S3 B126 with each
  other style as the Hidden Agent stand-in (run as bot_075, same play, so its 004/024/040 runs are reused): 070, 073,
  038, 012, 036, 017, 060. Queue q57 (~4 h). Style bots keep their historical code (incl. the [090] map bug) so the
  field matches [067]/[101]. Re-run only if the champion changes before the freeze.

- [105] results. Style tournament (112 games, offset 600): bot_081 8.11±0.72, bot_070 7.78±0.68, bot_073 7.77±0.63,
  bot_036 2.82±0.38, bot_012 2.39±0.38, bot_038 2.22±0.29, bot_040 2.16±0.35 SC. Champion first but within 1 SE of the
  two other search-machinery bots; the rule-based styles are far behind (≈2–3 SC each) in all-bot tables.
- issues: 1 timeout each for bot_073 (seed 500675) and bot_070 (seed 500674), both the Italy seat, t_max 1.89 / 1.78 s,
  in consecutive seeds that ran concurrently on different workers → a machine-wide stall (WSL/host), not a bot bug;
  champion 0 issues in 112 games. (Neither bot is submitted; noted as the one timing anomaly of the lab.)
- champion S3 (B126, bot_075 play) with each style as the Hidden Agent: vs 017 positional 17.04 (87.3%), vs 004 16.70*
  (84.8%, B210), vs 012 greedy 16.13 (80.2%), vs 038 valuemap 15.67 (75.4%), vs 036 bandit 15.44 (74.6%), vs 040
  aggressive 15.40 (B126 from [094]), vs 024 lookahead 15.63 ([094]), vs 060 ensemble 14.92 (67.5%, tmax 0.534 s),
  vs 070 search 14.16 (55.6%), vs 073 lookahead-widepool 13.46 (51.6%). All above the S3 5-pt lines (>12 SC / >40%).
  Hardest Hidden Agents are our own strongest search bots — the real Hidden Agent (~50% S2 wins) is weaker than
  070/073 (85–92% S2 wins), so the S3 mark has a wide margin.

## [106] Tuning sweep on the champion's GA/confirm parameters — DONE (no change)

- GA and confirmation parameters were set by hand in bot_059/065/075 and never tuned. --set variants of bot_075 (same
  play as agent_21/bot_081) on B210 S2/S3 (stand-in 004), paired with bot_075: GA_SAMPLES=4 (2), CONFIRM_K=8 (5),
  CONFIRM_T=0.12 (0.07), GA_POP=24 (16). Tuning, not a technique; a winner (z ≥ 2, confirmed on held-out C) becomes
  a new bot file. Queue q58 (~3 h).

- [106] results (B210 S2/S3 paired with bot_075): GA_SAMPLES=4 −0.07±0.18; CONFIRM_K=8 −0.02±0.20; CONFIRM_T=0.12
  −0.03±0.18; GA_POP=24 −0.20±0.20 (S3 −0.55±0.32). None better → keep bot_075's hand-set values.
- one game error in GA_POP=24 (S2 B 200175, our seat GERMANY): KeyError 'A MOS' raised in the BASELINE
  AttitudeAgent.update_attitude (agent_baselines.py:125, `order_status[unit]`). Engine get_order_status(power) is
  keyed by that power's units at the start of the phase, so the key is missing when some power's order list holds a
  move for a unit it did not own at phase start (or the attitude agent's internal game differs). First and only such
  error in 65,502 recorded games; our seat has 0 illegal orders over all of them; replay of the seed (timing-
  dependent) did not reproduce. Not attributable to our bot from the evidence; noted as a baseline robustness issue.
- lab/run_game.py: on any game error, the true game is now dumped (to_saved_game_format) to
  results/logs/crash_<run>_<seed>.json for post-mortem (self-tested with a throwaway patched process). Harness-only
  change; engine and baselines untouched.
- HOLD (human request 29 Sep 03:54): no new evaluations until the human says they are done gaming.
- HOLD lifted 2026-09-29 04:19:59 (human sent /loop).

## [107] bot_083 evolution/oppplans — REJECTED

- parent bot_081 | tags: opponent-plan-model. For 'strong' opponents, rollout samples play (p 0.6) one of the first 4
  plans of our own _la_candidates generator built from that power's view (BFS-greedy no-self-bounce + perturbations
  with supports), instead of independent per-unit sampling. Target: strong opponents (S3 vs 070/073 stand-ins
  14.2/13.5 SC; tournament tie with 070/073). Check: vs 3× bot_070 + 3× bot_012, all six classified 'strong'; plan
  sets built for ~4.6 opponents per phase; tmax 0.45 s.
- plan: q59: T0; T1 A42; S3 B126 vs 070 and 073 (075 refs from [105]); T2 S1 B126 + S2/S3 B210; held-out C; all vs
  bot_075; then the [105] tournament rerun with 083 in 081's seat (same seeds/seats → paired S4 comparison).

- bot_083 results (appended): T0 clean; T1 A42 vs 075 +0.31±0.26 (S2 tmax 0.742 s). S3 B126 vs 070 13.89 (−0.27±0.45),
  vs 073 13.45 (−0.01±0.52; tmax 0.988 s, 1 move > 0.9 s). T2: S1 18.00; B210 S2 17.30 (92.4%) / S3 16.73 (85.7%);
  vs bot_075 pooled −0.00±0.14. Held-out C +0.25±0.23. Tournament rerun (offset 600, same seats): bot_083 6.74 SC,
  3rd, vs bot_081's 8.11 in the same seats → paired −1.51±0.77.
- takeaway: modelling strong opponents as playing our own greedy-plan generator does not help against them (they are
  not greedy planners) and costs 20–40 ms per phase per strong opponent — the slowest moves of the lab (0.99 s)
  come from that precompute in all-strong tables. REJECTED; the per-unit sampling mix is kept.

## [108] Report evidence: GA vs race and the confirmation race on identical machinery — DONE

- No switch turns the GA off inside agent_21, so its effect is measured between bots that differ only in selection:
  bot_072 (GA) vs bot_071 (halving race) — same resolver, map fix, candidates, opponent model; and bot_075 (GA +
  confirm) vs bot_072 (GA). q60: extend 072 to B210 (S2/S3), held-out C (offset 400) for 072 and 071, then paired
  compares on B and C.

- [108] results. bot_072 B210: S2 16.98±0.22 (87.6%) / S3 15.51±0.33 (72.9%); C: S2 17.21 / S3 16.09. bot_071 C: S2 17.50
  (92.1%) / S3 16.01.
  GA vs race (072 − 071): B S2/S3 +0.07±0.19 (672 paired), C −0.11±0.26 → no difference: the GA alone does not beat
  the halving race on identical machinery.
  Confirmation race (075 − 072): B +0.58±0.18 (S3 +0.76±0.24, wins 77.9% vs 70.8%), C +0.17±0.25 (S3 +0.37) → the
  champion's gain over the race line comes from re-testing the GA's finalists on fresh samples (removing the
  winner's-curse bias of newly bred plans), not from evolution itself.
- implication: a confirmation step only helps a selector whose final pick has uneven, small sample counts (GA
  children); the halving race's survivors already share many samples, so race+confirm would be a no-op (not built).
- Loop stopped by the human 2026-09-29 07:32:01; no bots in progress. Freeze runs (results/logs/freeze_final.sh) to be launched later when the human is away from the PC.
- Loop resumed by the human (downtime, not enough for the freeze runs).

## [109] bot_084 evolution/crn — REJECTED (n.s.)

- parent bot_081 | tags: common-random-numbers-ga. From [108] (the gain is the confirmation race = removing unequal-
  sample ranking bias): every GA plan is scored on the same bank of 12 opponent samples drawn once per move before it
  can be ranked; fresh-sample confirmation kept. Instrumented (France vs six bot_040s): 40.6 generations/phase, 11.4
  fully scored plans at the end, 147 confirmation rounds/phase, tmax 0.45 s. Sanity S2/S3 seed 19 → 18/18.
- plan: q61: T0, T1 A42, T2 S1 B126 + S2/S3 B210 (004), held-out C (offset 400); all vs bot_075 (= agent_21 play).

- bot_084 results: T0 clean; T1 A42 vs 075 −0.06±0.23. T2: S1 18.00; B210 S2 17.30 (91.9%) / S3 16.49 (83.3%); vs 075
  pooled −0.04±0.14. Held-out C: S2 17.45 / S3 16.58; vs 075 +0.19±0.22. Combined ≈ +0.03±0.12 → REJECTED.
- takeaway: once the fresh-sample confirmation is in place, equalising the GA's sample counts adds nothing — the
  confirmation already removes the ranking bias. Selection is saturated; bot_075/081 stays.

## [110] bot_085 evolution/fallply — REJECTED (n.s.)

- parent bot_081 | tags: two-ply-fall. Fall rollouts continue with a greedy Spring reply (all non-static powers, no
  builds) and add 0.3 × SCs we would newly occupy − 0.3 × owned SCs an enemy would occupy after that Spring. The
  Spring counterpart (TWO_PLY) is a measured gain; Fall had no view of the next year. Checks: term active (5,698
  calls, 60% non-zero, Germany vs six bot_012s to 1905); sanity S1/S2/S3 seed 20 → 18/18/18, tmax 0.451 s.
- plan: q62: T0, T1 A42, T2 S1 B126 + S2/S3 B210 (004), held-out C (offset 400); all vs bot_075.

- bot_085 results: T2 S1 18.00; B210 S2 17.23 (90.0%) / S3 16.81 (83.8%); vs 075 pooled +0.06±0.13. Held-out C: S2 16.90 /
  S3 16.56; vs 075 −0.09±0.23 → REJECTED.
- takeaway: the Fall look-ahead does not add to the Spring one; after Fall, the next Spring's value is already
  mostly captured by SC ownership. Ninth consecutive challenger since bot_075 within ±0.2 SC (076–085, tuning sweep).
