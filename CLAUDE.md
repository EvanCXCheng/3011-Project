# CLAUDE.md — CITS3011 Diplomacy Agent Project

## Project

Build an agent that plays **No-Press Diplomacy** (standard map, game ends in 1920) using the `diplomacy` Python engine.
UWA CITS3011 group project, worth 30% of the unit. Due **Fri 2 Oct 2026, 11:59 pm AWST**.

Our group number is **21**. The agent file is `agent_21.py`, which `test.py` and `visualize.py` already import, and the
experiments file is `test_21.py`. Wherever the course materials say `groupnumber`, read `21`.

## How this repo is worked on

Claude runs **unattended in auto mode for hours**, continuously generating bot variants, testing them against the
baselines and against each other, and promoting the best one. The human is not watching.

- **Never stop to ask questions.** Make the most reasonable choice, write it down in `JOURNAL.md`, and keep going.
- If something is blocked (a crash you can't fix, a missing dependency), log it in `JOURNAL.md`, skip it, and move to the next idea.
- The session can end at any moment when usage runs out. Work in small steps so that little is lost when it does (see _Resume protocol_).
- **Keeping it going:** the human starts the session with `/loop` (self-paced) and the prompt
  `Continue the lab loop per CLAUDE.md`. Each wake-up advances the loop by one or more steps. Long evaluations run in
  the background (Bash `run_in_background`, wrapped in `timeout`); when one exits, Claude is woken to record it.
  While one is running, do useful work that doesn't compete for CPU, such as writing the next bot, analysis or backlog
  grooming. Don't poll in a tight loop.
- **Schedule** (AWST): all 6 family base bots through Tier 1 by **Mon 28 Sep**; at least 3 distinct measured new
  techniques by **Tue 29 Sep**; freeze **Thu 1 Oct 12:00**. `lab/status.py` prints the hours left.

## Hard rules (breaking these can mean zero marks)

- Every agent action must finish in **under 1 second**. Aim for ≤ 0.6 s wall time and stop searching when the budget runs out.
- The agent may use at most **512 MB** of memory.
- A bot must **not** save files, use the network, call LLMs or APIs, or use a GPU. (The lab harness may write files; bots may not.)
- Allowed imports in bots only: Python standard library, `diplomacy`, `tqdm`, `random`, `networkx`, `numpy`, `scipy`,
  `scikit-learn`, `timeout-decorator`, `simpleai`. aima-python may be used as a _reference_ only.
- **No reuse of existing Diplomacy code or solutions** (DumbBot, Albert, Cicero, open-source bots, etc.).
  Existing _ideas_ are fine if the report cites them; all code must be written from scratch.
- **Do not copy code from `agent_baselines.py`**, and that includes its Greedy Agent. Reading it to understand
  baseline behaviour is fine, but our bots are written independently.
- Never tamper with, monkey-patch, or work around the game engine or the test harness.
- `agent_21.py` and `test_21.py` must each be **≤ 100 KB**.
- Keep the agent interface from `agent_baselines.py`: subclass `Agent`, override its methods, and don't change their signatures.

## Rules for Claude

- **Do not write report prose.** `JOURNAL.md` holds terse experiment notes and numbers, not report text.
- Before relying on any engine API, check the docs (https://diplomacy.readthedocs.io/en/stable/) or the installed engine source.
- Never modify `agent_baselines.py`, the engine, or files in `results/raw/`.
- Never delete a bot, a result, or a journal entry. Rejected bots are experimental evidence for the report.
- Install nothing beyond `requirements.txt`. Use git only locally: commit, but never push, force, or rewrite history.
- Keep terminal output short: lab scripts print summaries, and full data goes to files.
  Never `cat` large CSVs or logs; query them with small scripts.
- **Commit with `git add -A && git commit -m "..."`**. A plain `git commit -am` would miss new bots and raw results.

## Environment

- Always use the venv interpreter: `.venv/bin/python` (Python 3.12, packages from `requirements.txt`). Run lab scripts
  from the repo root, e.g. `.venv/bin/python lab/evaluate.py ...`.
- The machine has 4 cores and 7 GB RAM, so `WORKERS = 3`. Measured costs: copy_game ≈ 2.5 ms, copy+process ≈ 2.6 ms,
  get_all_possible_orders ≈ 1 ms, and a baseline-only game ≈ 1 s. Game time is dominated by our bot's think time
  (60–90 of our calls per game).
- Workers set `OPENBLAS_NUM_THREADS=1` and a 512 MB `RLIMIT_AS`, which numpy, scipy and sklearn tolerate
  (see JOURNAL [000]). If a future bot needs more than that to import, don't raise the cap. Fix the bot.

## Scenarios and marking targets

Our agent plays a random power in every scenario. A win means reaching 18 supply centres (SCs).

| Scenario | Opponents                                                               | 1 pt             | 3 pts          | 5 pts          |
| -------- | ----------------------------------------------------------------------- | ---------------- | -------------- | -------------- |
| 1        | All Static (always hold)                                                | >2% win or >7 SC | >20% or >12 SC | >90% or >16 SC |
| 2        | Random / Attitude / Greedy mix (Random rarer)                           | >2% or >7 SC     | >25% or >10 SC | >50% or >13 SC |
| 3        | Scenario 2 mix + exactly one Hidden Agent (~50% win rate in Scenario 2) | >2% or >7 SC     | >20% or >9 SC  | >40% or >12 SC |
| 4        | Tournament against other groups' agents                                 | top 3 = +3 bonus |                |                |

Key fact for Scenario 1: all 22 starting units sit on home SCs, so once the 12 neutral SCs are taken, further growth
**requires dislodging holding units**. That needs a supported attack (strength 2 vs 1).

## Strategy families: don't assume greedy wins

Last semester's project was similar and greedy-style agents performed best. This semester's baselines, Hidden Agent or
setup may differ, so treat greedy as a **prior to test, not a conclusion**.

- **Profile the baselines first.** Before building any bot, run the provided baselines against each other
  (≈50 games per scenario) and record in `JOURNAL.md` how strong each one is and how it behaves this semester.
  Don't rely on last semester's behaviour.
- **Develop several strategy families in parallel** (listed under _Families and backlog_). Each family has its own
  line of bots and its own **family champion**.
- The **overall champion**, which is copied to `agent_groupnumber.py`, is whichever family champion scores best.
  No family is favoured when deciding promotions.
- **Budget:** no single family may take more than 40% of iterations. Greedy is the reference point to beat, not the default.
- **Give each family a fair try:** it gets at least 3 iterations (its base bot plus two improvements) before it can be
  set aside. First versions of complex strategies are usually weak.
- A family may be marked `DORMANT` once it is clearly behind (see _Every 5 iterations_). A dormant family is revisited
  whenever failure analysis suggests it could fix a weakness of the overall champion.
- **Hybrids are allowed.** When two families are strong in different scenarios, try a bot that combines them
  (e.g. detects the likely scenario from opponents' early behaviour and switches strategy).
- Timing risk grows with complexity. A slightly weaker bot that is always fast beats a stronger bot that sometimes times out.

## Repo layout

```
bots/                  one self-contained file per bot: bot_NNN_<family>_<short_name>.py
lab/config.py          all constants: seed sets, tier sizes, limits, rubric, S2 pool, freeze time
lab/common.py          bot loading, InstrumentedAgent (timing/legality/desync), raw-record access, stats, rubric marks
lab/run_game.py        one instrumented game via the unmodified game.run_one_game (CLI: one game, --save for the visualiser)
lab/evaluate.py        parallel evaluation of one bot on scenarios × seed set; reuses games already recorded
lab/compare.py         paired comparison + PROMOTE/REJECT verdict
lab/promote.py         checks a bot (size, imports, loads) and promotes it; overall -> copies to agent_21.py
lab/state.py           results/state.json CLI: new bot ids, statuses, families, techniques, S3 stand-in choice
lab/leaderboard.py     regenerates LEADERBOARD.md
lab/tournament.py      Scenario 4 stand-in
lab/analyze.py         failure analysis
lab/bench_engine.py    engine timing benchmark
lab/stress.py          slow-machine timing gate: games pinned to one core (≈4× slower), reports tmax and budget overshoot
lab/status.py          resume helper (running evals, RUNNING entries, champion/agent file match, hours to freeze)
tools/export_prompts.py  writes LLM_PROMPTS.md from Claude Code transcripts, for llm_usage_21.pdf (not part of any bot)
results/raw/           one JSONL file per evaluation run (append-only); one line per game
results/results.csv    one summary row per (run, bot, scenario)
results/state.json     single source of truth: bots, families, champions, hall of fame, techniques
results/logs/          progress logs and .running markers (gitignored)
LEADERBOARD.md         generated by lab/leaderboard.py. Never hand-edit it
JOURNAL.md             append-only log, one entry per iteration
BACKLOG.md             queue of ideas, grouped by family
agent_21.py            always a copy of the overall champion (so a submittable best bot exists at all times)
test_21.py             submitted experiments file: a trimmed copy of the lab (≤ 100 KB), built at the freeze
```

## Lab commands

```bash
P=.venv/bin/python
$P lab/status.py                                              # start of every session / wake-up
$P lab/state.py new --family greedy --name base --parent none --tags "bfs-greedy"   # allocates bot id + path
$P lab/evaluate.py --bot bots/bot_001_greedy_base.py --seedset X --n 3 --workers 1 --tag tier0   # Tier 0 (serial timing)
$P lab/evaluate.py --bot bots/bot_001_greedy_base.py --seedset A --n 42 --tag tier1              # Tier 1
$P lab/evaluate.py --bot bots/bot_001_greedy_base.py --seedset B --n 210 --tag tier2             # Tier 2
$P lab/evaluate.py --bot bots/bot_004_greedy_x.py --seedset B --n 210 --standin <same as candidate> --tag tier2   # reference re-run (reused if already done)
$P lab/compare.py --a bot_007 --b bot_004 --seedset B                                          # verdict
$P lab/promote.py bot_007 --family [--overall]
$P lab/state.py set-status bot_007 REJECTED "one-line takeaway"
$P lab/state.py technique supported-attack-matching bot_007 "S1 +2.1 SC vs bot_004 (210 games, p<.01)"
$P lab/state.py standin --for bot_007                                                           # S3 stand-in rule
$P lab/tournament.py --bots hof --games 28
$P lab/analyze.py --bot bot_007
$P lab/stress.py --bot bots/bot_007_x.py                                                     # slow-CPU timing gate (no eval running)
$P lab/evaluate.py --bot bots/bot_007_x.py --seedset A --n 42 --set USE_SUPPORTS=false --tag ablation   # ablation
$P lab/leaderboard.py
```

Long runs go in the background, e.g. `timeout 90m .venv/bin/python lab/evaluate.py ... > results/logs/last_eval.txt 2>&1`.
The `--standin` default is `auto`, which applies the stand-in rule. For a paired comparison, pass the **same explicit**
`--standin` to both the candidate and the reference run, because S3 games are only paired when stand-ins match.
```

## Bot conventions

- Each bot is a **single self-contained file** that could be renamed to `agent_groupnumber.py` and submitted as is.
  No imports from other bots or from `lab/`.
- Use the class name that `test.py` imports, `StudentAgent`, so a promoted bot can be dropped in without edits
  (`lab/config.py: AGENT_CLASS`). The only repo import allowed is `from agent_baselines import Agent`.
- Allocate every new bot with `lab/state.py new ...`, which assigns the id and filename and counts the family iteration.
- Start each file with a docstring giving: bot id, family, parent bot id, hypothesis, and technique tags.
- Put every technique toggle and tunable parameter in one module-level `CONFIG = {...}` dict and read it at call
  time. Ablations then just run `evaluate.py --set KEY=VAL` on the same file, with no extra bot files.
- Time budget: measure from the start of each `get_actions` call and stop searching at **0.5 s**. `__init__` and
  `new_game` also count against the 1 s limit, so keep precomputation well under that. `update_game` must stay the
  scaffold's two-liner plus cheap bookkeeping.
- Call `get_all_possible_orders()` at most once per phase and cache it.
- Never call `set_orders`/`process` on `self.game`; only on a `copy_game` copy (the lab flags a desync if you do).
- A new bot starts as a copy of its parent. Change **one idea per bot** so the results can be attributed to that idea.
  A family's first bot is written fresh, not copied from another family.
- Shared helpers (map precomputation, safe fallback orders) may be copied between families, since bots must stay self-contained.
- Every decision is wrapped in `try/except`. On any error or when the time budget runs out, the bot returns the best
  orders found so far, otherwise safe defaults (hold or the phase default). A bot must never crash or stall.
- Only issue orders taken from the engine's list of possible orders for our orderable locations.
- Handle all three phase types (movement, retreats, adjustments/builds) and split coasts
  (`STP/NC`, `STP/SC`, `SPA/NC`, `SPA/SC`, `BUL/EC`, `BUL/SC`).
- Precompute map data (BFS distances per unit type, adjacency) once, lazily, and keep it in memory. Bots can't save files.

## The lab loop

Repeat until usage runs out:

1. **Resume.** Read `LEADERBOARD.md`, the last ~5 entries of `JOURNAL.md`, and `BACKLOG.md`.
2. **Pick a family, then an idea.**
   - Until every family has a base bot that passes Tier 1, build the next missing base bot.
   - After that, pick the `ACTIVE` family with the fewest iterations relative to its fair share, staying within the 40% cap.
   - Take that family's top backlog item.
3. **Log intent.** Append a JOURNAL entry with status `RUNNING`: bot id, family, parent, hypothesis, and the planned test.
4. **Build.** Create `bots/bot_NNN_<family>_<name>.py` with the single change, or write it fresh for a new family.
5. **Tier 0: smoke test.** Run 3 games per scenario on seed set X with `--workers 1`, so timing isn't distorted by
   contention. Pass requires no exceptions, only legal orders, no desync, slowest move < 0.6 s, and peak memory < 400 MB.
   On failure, fix it at most twice; after that, mark it `BROKEN` and move on.
   Parallel tiers also gate on zero timeouts and zero moves > 0.9 s.
6. **Tier 1: screen.** Run 42 games per scenario (1–3) on seed set A (6 per power).
   Reject if it is clearly below its **own family champion**, i.e. its paired pooled diff is < −2 SE on seed set A.
   New families are always passed on to Tier 2.
7. **Tier 2: confirm.** Run 210 games per scenario on seed set B (fallback 126), on the same seeds and with the same S3
   stand-in as the family champion and the overall champion. Reference results are reused automatically when they exist.
8. **Promote** when the estimated mark is at least the champion's and the mean-SC gain across scenarios is > 2 standard errors (paired):
   - **Family champion:** the bot beats its own family champion.
   - **Overall champion:** the bot also beats the overall champion. Copy it to `agent_groupnumber.py`.
   - **Slow-machine gate (overall champion only):** the marking machine's CPU may be slower than ours. If the bot's
     tmax is above 0.1 s, it must also pass `lab/stress.py` (4 games pinned to one core, ≈4× slowdown; slowest
     move < 0.8 s, no timeouts or exceptions) before it is copied to `agent_21.py`. Run the stress test only when no
     evaluation is running. On FAIL, lower `TIME_BUDGET` or cut the unbudgeted work, then re-run the stress test.
   - **Hall of fame:** every family champion, plus the 3 most recent overall champions.
   - **Bootstrap:** the first bot to pass Tier 1 becomes overall champion immediately, so `agent_21.py` is never the stub
     for long. Likewise, a family's first bot to pass Tier 1 becomes that family's champion.
9. **Record.** Regenerate `LEADERBOARD.md`, set the JOURNAL entry to `PROMOTED`, `REJECTED` or `BROKEN` with the key
   numbers and a one-line takeaway, and add follow-up ideas to `BACKLOG.md` under the right family.
10. **Commit**, e.g. `git add -A && git commit -m "bot_NNN (<family>): <result>"`.

**Every 5 iterations**

- Run a **tournament** (a stand-in for Scenario 4): 7-seat games filled with hall-of-fame bots from **different families**
  in rotating seats, ranked by mean final SCs.
- Run a **held-out check**: evaluate the overall champion on seed set C (use `--offset` to take fresh seeds each time),
  so seeds A and B aren't being overfit.
- Run a **failure analysis**: find the powers, phases and opponent types where the champion loses SCs.
  Add ideas for them to the backlog, including ideas for other families that might handle those cases better.
- Run a **family review**: set each family to `ACTIVE` or `DORMANT` with a one-line reason.
  A family can only become `DORMANT` once it has had its minimum 3 iterations and its champion is more than
  1 estimated mark behind the overall champion.

**Opponents in evaluation**

- Our seat is `POWERS[seed % 7]`, so seed counts that are multiples of 7 cover every power equally. Opponents are drawn
  deterministically from the seed, so two bots on the same seeds face the same seat and opponent line-up (paired design).
- Scenario 1: all Static. Scenario 2: each opponent drawn from `[Random, Attitude, Attitude, Greedy, Greedy]`, exactly as in `test.py`.
- Scenario 3: the Scenario 2 mix with one random opponent seat replaced by the **Hidden Agent stand-in**. The stand-in
  is the strongest hall-of-fame bot, by S2 win rate, from a **different family** than the candidate, so bots don't only
  learn to beat copies of themselves. If no such bot exists yet, use the overall champion if it's from another family;
  otherwise use the Greedy baseline, and treat those S3 numbers as provisional (the real Hidden Agent is far stronger).
  Every few evaluations, rotate which family provides the stand-in (`lab/state.py rotate-standin`).

**Sizing**

- Measured: a baseline-only game takes about 1 s; our bot adds (number of calls ≈ 60–90) × its mean move time.
  Use 3 workers.
- Size the tiers so one iteration takes at most ~45 minutes. If it takes longer, cut Tier 2 to 126 games and note that in the JOURNAL.
- Run long evaluations in the background with a hard timeout, and check on them instead of blocking.

## Scoring

`lab/leaderboard.py` computes for each bot:

- Win rate and mean SCs (± standard error) for each scenario, broken down by power.
- **Estimated rubric mark**: for each scenario, the highest tier reached by _either_ win rate or mean SCs (see the table above),
  summed out of 15. Flag any scenario within 1 standard error of a threshold as `borderline`.
- Slowest move time and peak memory seen.
- Tournament rank when available.

Ranking uses estimated mark first, then mean SCs, then the lower slowest-move time.

`LEADERBOARD.md` also shows a **family table** with, for each family: status, iterations used (and share of the total),
family champion, and its estimated mark per scenario. This makes it obvious if one family is winning some scenarios but not others.

## Families and backlog (initial `BACKLOG.md`)

Each family below gets a base bot, then improvements. Improvement ideas that fit several families may be tried in each.

**greedy**: move toward nearby SCs we don't own (the likely basic technique for the report)

1. Base: each unit moves toward the nearest SC we don't own, using BFS distance per unit type.
2. Avoid self-bounces: no two of our units target the same province, and resolve chains where one unit vacates a province another enters.
3. Supports: units with no useful move support a neighbouring unit's attack.
4. Supported attacks on occupied SCs (2 vs 1), assigned as a matching problem (`scipy.optimize.linear_sum_assignment`).
5. Fall priority: in Fall, occupy or stay on SCs we don't yet own before ownership updates.
6. Home defence: hold or support-hold SCs threatened by an adjacent enemy.
7. Build choice: army or fleet depending on which remaining target SCs need sea access; build nearest the front.
8. Retreats: prefer SCs, then provinces nearer to targets; disband otherwise.
9. Convoys for England and Turkey.

**valuemap**: rule-based strategy driven by a province value map (DumbBot-style idea; cite it, own code)

- Score each province by nearby SC value spread across the map, with separate weights for attack and defence.
- Choose destinations by value, with competition and strength estimates for each province.

**search**: local search over our joint orders

- Hill climbing or simulated annealing over the full set of our orders, scored by a heuristic evaluation function.
- Try different evaluation functions (SC count, threatened SCs, distance to targets, unit safety).

**lookahead**: simulate, then choose

- Evaluate the top-N candidate order sets against sampled opponent orders on a copy of the game (one move deep), stopping at the time budget.
- Lightweight Monte Carlo tree search, if simulation is fast enough (benchmark first).
- If the engine is too slow, write our own simplified strength-counting resolver for simulations.

**adaptive**: model the opponents and react to them

- Classify each opponent's type from its order history (always holds = static; heads for the nearest SC = greedy;
  no pattern = random; never attacks certain powers = attitude).
- Use predicted opponent moves to decide what to defend and whom to attack. Avoid provoking powers that aren't attacking us.
- Detect the scenario (e.g. every opponent holding means Scenario 1) and switch to the best plan for it.

**positional**: slower, safer expansion

- Keep a compact front, prioritise not losing SCs, then expand into the weakest neighbour.
- Pick the target power based on how weak it is and how easily we can reach it, not just the nearest SC.

Claude may add new families when failure analysis suggests a different approach. Each new family follows the same rules
(base bot, minimum 3 iterations, 40% cap).

## Report coverage (tracked in LEADERBOARD.md)

The report needs **one basic technique plus three distinct new techniques**, each implemented and measured with numbers.
Keep a table of technique tag → first bot that implemented it → measured effect compared with its parent.
The families naturally supply distinct techniques. If fewer than three distinct techniques have been measured,
prioritise them over further improvements to any single family. Parameter tuning alone does not count as a new technique.

## Reproducing marking conditions (WSL2 or Linux)

The lab already does all of this: each worker sets a 512 MB `RLIMIT_AS`, every call of our seat is wrapped in
`timeout_decorator.timeout(1)` and timed with `perf_counter`, and peak memory comes from `ru_maxrss`.
For a pessimistic single-core timing check before promoting an expensive bot:

```bash
taskset -c 0 .venv/bin/python lab/run_game.py --bot bots/bot_NNN_x.py --scenario 2 --seed 5
```

Slower marking CPU: the bots' search stops on wall-clock time, so a slower CPU mainly costs search quality. The risk
is the work outside the budget check (order generation, the rollout in flight, final selection), which scales with CPU
speed. `lab/stress.py` pins several games to one core to simulate a ≈4× slower machine and reports the slowest move
and the overshoot past the bot's `TIME_BUDGET` (summary in `results/logs/stress_<bot>.json`, never in `results/raw/`):

```bash
.venv/bin/python lab/stress.py --bot bots/bot_NNN_x.py            # --procs 4 --n 4 --gate 0.8 by default
```

## Engine API pointers (verify before use)

- `game.get_all_possible_orders()`: dict of location → list of legal orders
- `game.get_orderable_locations(power_name)`: locations where we can give orders
- `game.get_current_phase()`, `game.phase_type`: e.g. `S1901M`, and `M` / `R` / `A`
- `game.get_power(name).units` / `.centers`: current units and SCs
- `game.map.scs`, `game.map.abuts(...)`, `game.map.loc_abut`: SC list and adjacency
- `game.order_history`: past orders by phase and power
- `game.set_orders(...)`, `game.process()`: in a bot, only ever on a **copy** of the game
- Verified notes (JOURNAL [000]): orderable locations are 3-letter base names (`STP` for `F STP/SC`), but the orders
  use the full coast. Adjustments offer `WAIVE`. The rules include `IGNORE_ERRORS`, so illegal orders are silently
  voided, which is why the lab checks them. The game ends as soon as someone has 18 SCs.
- `game.copy_game(game)` (in `game.py`) is the cheap deep copy used by the harness; bots may use the same
  `to_saved_game_format` / `from_saved_game_format` approach themselves.

## Resume protocol

Every session, including one started after usage ran out:

1. Run `.venv/bin/python lab/status.py`. It lists evaluations still running (don't restart those), stale markers,
   JOURNAL entries in `RUNNING` status, and whether `agent_21.py` matches the champion.
2. For each `RUNNING` entry: if its results are in `results/raw/`, finish recording it. Otherwise re-run the missing tier.
   Re-running is cheap because `evaluate.py` reuses games already recorded.
3. If `agent_21.py` doesn't match the champion, re-run `lab/promote.py <champion> --overall`.
4. Continue the lab loop.

## Freeze

From **Thu 1 Oct 2026, 12:00 pm AWST**, stop generating new bots. Only do these:

- Run the final evaluation of the overall champion: `--seedset FINAL --n 504` (72 per power).
- Run ablations for the report's techniques: the champion with each technique toggled off via `--set`, on the same seeds.
- Trim `lab/` into `test_21.py`, which must be ≤ 100 KB, self-contained and runnable.
- Regenerate `LLM_PROMPTS.md` with `tools/export_prompts.py`. The group must submit `llm_usage_21.pdf`.
- Check that `agent_21.py` is ≤ 100 KB, imports only allowed packages (`lab/promote.py` checks this), and runs
  under `test.py`.
