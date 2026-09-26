# BACKLOG

Ideas queued per family, top = next. Mark items `[done bot_NNN]` or `[dropped: reason]`; don't delete them.

## lab / cross-cutting
1. [done: JOURNAL 001] Iteration 0: profile baselines. Static, Random, Attitude and Greedy as the player in S1 and S2, 49 games each
   (seed set A), plus the S3 (Greedy stand-in) number for Greedy. Record in JOURNAL.
2. [done: engine civil-disorder auto-disbands units farthest from home when no disband orders; checked 26 Sep] Confirm engine behaviour for missing disbands (Static powers in S1) and for retreat defaults.

## greedy (basic technique)
1. Base: each unit moves toward the nearest SC we don't own (BFS distance per unit type, coast-aware).
2. No self-bounces: unique targets per province; resolve chains where one unit vacates a province another enters.
3. Supports: units with no useful move support a neighbouring unit's attack.
4. Supported attacks on occupied SCs (2 vs 1), assigned as a matching problem (`scipy.optimize.linear_sum_assignment`).
5. Fall priority: in Fall, occupy or stay on unowned SCs before ownership updates.
6. Home defence: hold or support-hold SCs threatened by an adjacent enemy.
7. Build choice: army vs fleet by which remaining targets need sea access; build nearest the front.
8. Retreats: prefer SCs, then provinces nearer to targets; disband otherwise.
9. Convoys for England and Turkey.

## valuemap (DumbBot-style idea: cite it; own code)
1. Base: province value = SC value (unowned/enemy weighted) spread over the map by iterated neighbour blur;
   separate attack and defence weights; each unit moves to its highest-value reachable province, avoiding duplicates.
2. Competition and strength estimates per province (adjacent enemy units) to discount contested targets.
3. Supports assigned toward the highest-value contested destinations.
4. Season-dependent weights (Fall: occupy SCs; Spring: position).

## search (local search over our joint orders)
1. Base: hill climbing with random restarts over the joint order set, scored by a heuristic evaluation
   (expected SC gain, supported-attack bonus, threatened-own-SC penalty, distance to targets), within the time budget.
2. Simulated annealing instead of hill climbing.
3. Alternative evaluation functions (unit safety, threatened SCs).

## lookahead (simulate, then choose)
1. Base: generate top-N candidate joint orders (from a greedy/value generator), simulate each against sampled
   opponent orders on engine copies (1 move deep), and pick the best mean resulting SC/position score. Stop at the budget.
2. Opponent model for sampling: static in S1-like games, greedy-like elsewhere.
3. Own lightweight resolver if engine copies become the bottleneck.
4. Lightweight MCTS over our candidate set.

## adaptive (opponent modelling)
1. Base: classify each opponent (static / greedy / random / attitude) from order history; predict greedy moves exactly;
   defend where predicted attacks land; attack powers that are weak or static first.
2. Scenario detection: all holding → S1 plan (supported attacks on held SCs); otherwise S2/S3 plan.
3. Avoid provoking Attitude powers that are friendly (don't attack their SCs while other targets exist).

## positional (slower, safer expansion)
1. Base: compact front, never leave a home SC open to an adjacent enemy, expand into the weakest reachable neighbour power.
2. Target-power selection by weakness plus reachability.

## added 26 Sep (after bots 007–012)
- search: per-power prediction accuracy → use class-predicted threat only for powers we predict well (static / greedy
  with high hit rate); raw adjacency for the rest (bot_008 S3 −1.23 vs bot_003 against the lookahead stand-in).
- lookahead: same idea for rollout sampling (bot_007 S2/S3 flat vs bot_004).
- adaptive/lookahead: hybrid of adaptive S1 plan and lookahead is bot_009 (queued).
- greedy: fall priority / home defence on top of bot_012 (S3 still with weak stand-in; re-run S3 with bot_007 stand-in).

## added 26 Sep (review after 23 iterations)
- S4 / robustness: in the 56-game HoF tournament valuemap bot_014 (7.2 SC) beat bot_022 (6.8) and bot_021 (5.4). Check
  what the rollout bots assume about strong opponents in all-bot games (classes, sampling mixes); try a 'strong'
  default when every opponent is unpredictable. Measure with more tournament games before acting (SEs ~0.7).
- adaptive bot_009 last in the tournament (2.1 SC): check whether its static-scenario switch misfires in bot tables.

## freeze preparation (noted 26 Sep)
- Spec note [7]: the new techniques must be implemented in the submitted code (may be switched off). The champion
  line (bot_022 → agent_21.py) carries CONFIG toggles for: OPP_AWARE (opponent-aware eval), ACC_GATE (prediction-
  accuracy gating), ROLLOUT (rollout selection), LA_CANDS + HALVING (hybrid candidate race); basic technique =
  BFS-greedy distance scoring (lookahead candidate generator / hill-climb start). Freeze ablations: toggle each on
  FINAL seeds. Keep these toggles in any later champion.
- test_21.py: lab is 66 KB now (+ nothing else needed); trim to a self-contained runner (evaluate / compare / ablation
  / stress / tournament) ≤ 100 KB at the freeze.
