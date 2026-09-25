# Leaderboard

_Generated 2026-09-26 02:22 AWST by lab/leaderboard.py — do not hand-edit._  
Freeze in 129.6 h, deadline in 165.6 h.

**Overall champion:** bot_003  (agent_21.py matches)

## Bots

Mark = estimated rubric points /15 (per scenario: highest tier reached by win rate or mean SC). `*` = within 1 SE of a threshold. Pooled over seed sets A/B/C/FINAL, current file hash only.

| # | bot | family | status | S1 sc (win%) n | S2 sc (win%) n | S3 sc (win%) n | mark | tmax s | mem MB | issues | tourn |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | bot_005_adaptive_base | adaptive | PROMOTED | 18.00±0.00 (100%) 42 → 5 | 14.02±0.77 (45%) 42 → 5* | 9.79±0.96 (26%) 42 → 3* | **13** | 0.037 | 102 | 0 | - |
| 2 | bot_004_lookahead_base | lookahead | PROMOTED | 16.52±0.49 (76%) 42 → 5 | 15.07±0.69 (60%) 42 → 5 | 10.14±1.01 (31%) 42 → 3 | **13** | 0.513 | 97 | 0 | - |
| 3 | bot_003_search_base | search | PROMOTED | 11.71±0.27 (23%) 252 → 3* | 13.52±0.34 (49%) 252 → 5* | 12.74±0.36 (44%) 252 → 5 | **13** | 0.403 | 103 | 0 | - |
| 4 | bot_002_valuemap_base | valuemap | PROMOTED | 12.29±0.18 (0%) 252 → 3 | 8.29±0.34 (12%) 252 → 1 | 6.51±0.30 (7%) 252 → 1 | **5** | 0.056 | 102 | 0 | - |
| 5 | baseline:greedy | - | - | 7.39±0.40 (4%) 49 → 1* | 9.92±0.98 (31%) 49 → 3* | 7.65±0.85 (14%) 49 → 1* | **5** | 0.051 | 101 | 12 | - |
| 6 | bot_006_positional_base | positional | RUNNING | 12.29±0.61 (14%) 42 → 3* | 8.60±0.82 (12%) 42 → 1 | - | **4** | 0.045 | 100 | 0 | - |
| 7 | bot_001_greedy_base | greedy | PROMOTED | 6.14±0.16 (0%) 252 → 0 | 7.58±0.31 (8%) 252 → 1 | 6.41±0.19 (3%) 462 → 1 | **2** | 0.074 | 102 | 0 | - |
| 8 | baseline:attitude | - | - | 6.76±0.24 (0%) 49 → 0 | 1.55±0.26 (0%) 49 → 0 | 1.37±0.23 (0%) 49 → 0 | **0** | 0.079 | 101 | 0 | - |
| 9 | baseline:random | - | - | 6.55±0.25 (0%) 49 → 0 | 1.55±0.30 (0%) 49 → 0 | 1.37±0.29 (0%) 49 → 0 | **0** | 0.074 | 100 | 0 | - |
| 10 | baseline:static | - | - | 3.14±0.05 (0%) 49 → 0 | 2.45±0.16 (0%) 49 → 0 | 2.41±0.16 (0%) 49 → 0 | **0** | 0.009 | 100 | 0 | - |

## Families

| family | status | iterations (share) | champion | S1 | S2 | S3 | mark | reason |
|---|---|---|---|---|---|---|---|---|
| adaptive | ACTIVE | 1 (17%) | bot_005 | 5 | 5 | 3 | 13 |  |
| greedy | ACTIVE | 1 (17%) | bot_001 | 0 | 1 | 1 | 2 |  |
| lookahead | ACTIVE | 1 (17%) | bot_004 | 5 | 5 | 3 | 13 |  |
| positional | ACTIVE | 1 (17%) | - | - | - | - | - |  |
| search | ACTIVE | 1 (17%) | bot_003 | 3 | 5 | 5 | 13 |  |
| valuemap | ACTIVE | 1 (17%) | bot_002 | 3 | 1 | 1 | 5 |  |

**Hall of fame:** bot_005, bot_001, bot_004, bot_003, bot_002  
**Overall champion history:** bot_001 → bot_003

## Report coverage: techniques

Distinct techniques measured: 3 (need basic + 3 new).

| technique tag | first bot | measured effect vs parent |
|---|---|---|
| bfs-greedy | bot_001 | basic technique: est. mark 2 (B210: S1 6.14, S2 7.43, S3 6.75) |
| local-search | bot_003 | pooled +5.91±0.21 SC vs bot_001 (630 paired, B); mark 13 vs 2 |
| value-map | bot_002 | pooled +2.20±0.20 SC vs bot_001 (630 paired, B); S1 +6.14±0.20 |

## Champion bot_003 by power (mean SC / win%)

| scenario | AUSTRIA | ENGLAND | FRANCE | GERMANY | ITALY | RUSSIA | TURKEY |
|---|---|---|---|---|---|---|---|
| S1 | 7.0 / 0% | 15.9 / 28% | 8.2 / 0% | 14.5 / 47% | 8.0 / 0% | 17.4 / 83% | 11.0 / 0% |
| S2 | 10.8 / 31% | 12.8 / 25% | 16.8 / 78% | 14.8 / 56% | 11.5 / 39% | 15.8 / 75% | 12.1 / 39% |
| S3 | 9.5 / 31% | 12.8 / 28% | 16.7 / 75% | 13.5 / 53% | 9.6 / 22% | 16.1 / 81% | 11.1 / 22% |
