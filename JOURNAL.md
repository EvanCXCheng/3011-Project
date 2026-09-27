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

## [052] bot_034 bandit/base — RUNNING (new family 'bandit')

- family: bandit (new; human invited completely new bots) | parent: none | tags: decoupled-ucb, combinatorial-bandit
- hypothesis: decoupled per-unit UCB1 bandits over each unit's orders (hold, top-10 greedy moves, supports of own
  units), credited from joint 1-ply rollouts vs class-sampled opponents (~300–400 rollouts/phase, opponent sample shared
  by batches of 4), find coordinated orders without a hand-made candidate generator.
- sanity: S1/S2/S3 seed 11 → 8/1/18 SC, no errors, tmax 0.48 s. Expected weakness: 2v1 needs a move and a support
  chosen jointly, which independent per-unit bandits find slowly (S1 8 SC in the sanity game).
- plan: queue q3 (T0 → T1 A42 → T2 B126) vs bot_022; new family → Tier 2 regardless.

## [053] bot_035 valuemap/poolsource — RUNNING (dormant family revisited)

- family: valuemap (revisit) | parent: bot_022 | tags: multi-source-candidates
- hypothesis: candidate diversity drives bot_022 (LA_CANDS off −1.30); adding the valuemap champion's (bot_014) joint
  order as a third source lets the rollouts use its 2v1/strength-aware plans when they are better.
- checks: pool 14–21 per phase incl. the VM candidate; S2/S3 seed 12 sanity below; no errors.
- plan: queue q4 (after q3).

## [054] bot_036 bandit/pairs — RUNNING

- family: bandit | parent: bot_034 | tags: pair-arms
- hypothesis: independent per-unit bandits rarely draw a move and its support together. Pair arms ("move into an
  enemy-held target SC, supported by unit Y"; Y's order is overridden and not credited) make 2v1 attacks one draw.
- sanity: S1 seeds 11/12 → 8/18 SC, S2 11/12 → 4/18 SC, no errors, ~500 rollouts/phase.
- plan: queue q5 (after q4).

## [055] Sparring field, round 1 (human request) — RUNNING

- bot_037 greedy/homedef (parent bot_012, rule-based): hold / support-hold / garrison own SCs that an *active* enemy
  (one that has ever ordered a move) can enter, before attack matching. First version garrisoned against static units
  (S1 seed 12: 11 SC) → restricted to active powers (then 18). Sanity S2 seeds 12/13 → 8/7 SC.
- bot_038 valuemap/season (parent bot_014, rule-based): Fall SC values ×1.5 with 2 diffusion passes, Spring full 6.
  Sanity S1 18/18, S2 18/3, S3 13/18.
- plan: queue q6 (queue_family.sh): 037 vs bot_012 (T1 greedy, T2 bot_001 B210); 038 vs bot_014 (T1/T2 bot_001, B210).
