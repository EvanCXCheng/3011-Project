# Leaderboard

_Generated 2026-09-26 03:35 AWST by lab/leaderboard.py — do not hand-edit._  
Freeze in 128.4 h, deadline in 164.4 h.

**Overall champion:** bot_004  (agent_21.py matches)

## Bots

Mark = estimated rubric points /15 (per scenario: highest tier reached by win rate or mean SC). `*` = within 1 SE of a threshold. Pooled over seed sets A/B/C/FINAL, current file hash only.

| # | bot | family | status | S1 sc (win%) n | S2 sc (win%) n | S3 sc (win%) n | mark | tmax s | mem MB | issues | tourn |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | bot_004_lookahead_base | lookahead | PROMOTED | 16.21±0.25 (67%) 168 → 5* | 14.86±0.37 (61%) 168 → 5 | 11.09±0.48 (33%) 168 → 3 | **13** | 0.533 | 97 | 0 | - |
| 2 | bot_005_adaptive_base | adaptive | PROMOTED | 18.00±0.00 (100%) 252 → 5 | 12.96±0.38 (48%) 252 → 3* | 9.75±0.40 (26%) 252 → 3 | **11** | 0.046 | 102 | 0 | - |
| 3 | bot_003_search_base | search | PROMOTED | 11.71±0.27 (23%) 252 → 3* | 13.52±0.34 (49%) 252 → 5* | 11.72±0.24 (37%) 588 → 3 | **11** | 0.403 | 103 | 0 | - |
| 4 | bot_002_valuemap_base | valuemap | PROMOTED | 12.29±0.18 (0%) 252 → 3 | 8.29±0.34 (12%) 252 → 1 | 6.51±0.30 (7%) 252 → 1 | **5** | 0.056 | 102 | 0 | - |
| 5 | baseline:greedy | - | - | 7.39±0.40 (4%) 49 → 1* | 9.92±0.98 (31%) 49 → 3* | 7.65±0.85 (14%) 49 → 1* | **5** | 0.051 | 101 | 12 | - |
| 6 | bot_006_positional_base | positional | PROMOTED | 12.29±0.61 (14%) 42 → 3* | 8.60±0.82 (12%) 42 → 1 | 5.57±0.59 (0%) 42 → 0 | **4** | 0.045 | 100 | 0 | - |
| 7 | bot_001_greedy_base | greedy | PROMOTED | 6.14±0.16 (0%) 252 → 0 | 7.58±0.31 (8%) 252 → 1 | 6.41±0.19 (3%) 462 → 1 | **2** | 0.074 | 102 | 0 | - |
| 8 | baseline:attitude | - | - | 6.76±0.24 (0%) 49 → 0 | 1.55±0.26 (0%) 49 → 0 | 1.37±0.23 (0%) 49 → 0 | **0** | 0.079 | 101 | 0 | - |
| 9 | baseline:random | - | - | 6.55±0.25 (0%) 49 → 0 | 1.55±0.30 (0%) 49 → 0 | 1.37±0.29 (0%) 49 → 0 | **0** | 0.074 | 100 | 0 | - |
| 10 | baseline:static | - | - | 3.14±0.05 (0%) 49 → 0 | 2.45±0.16 (0%) 49 → 0 | 2.41±0.16 (0%) 49 → 0 | **0** | 0.009 | 100 | 0 | - |

## Families

| family | status | iterations (share) | champion | S1 | S2 | S3 | mark | reason |
|---|---|---|---|---|---|---|---|---|
| adaptive | ACTIVE | 1 (12%) | bot_005 | 5 | 3 | 3 | 11 |  |
| greedy | ACTIVE | 1 (12%) | bot_001 | 0 | 1 | 1 | 2 |  |
| lookahead | ACTIVE | 2 (25%) | bot_004 | 5 | 5 | 3 | 13 |  |
| positional | ACTIVE | 1 (12%) | bot_006 | 3 | 1 | 0 | 4 |  |
| search | ACTIVE | 2 (25%) | bot_003 | 3 | 5 | 3 | 11 |  |
| valuemap | ACTIVE | 1 (12%) | bot_002 | 3 | 1 | 1 | 5 |  |

**Hall of fame:** bot_005, bot_001, bot_004, bot_006, bot_003, bot_002  
**Overall champion history:** bot_001 → bot_003 → bot_004

## Report coverage: techniques

Distinct techniques measured: 5 (need basic + 3 new).

| technique tag | first bot | measured effect vs parent |
|---|---|---|
| bfs-greedy | bot_001 | basic technique: est. mark 2 (B210: S1 6.14, S2 7.43, S3 6.75) |
| local-search | bot_003 | pooled +5.91±0.21 SC vs bot_001 (630 paired, B); mark 13 vs 2 |
| one-ply-simulation | bot_004 | pooled +2.16±0.31 SC vs bot_003 (378 paired, B); mark 13 vs 11; S1 +4.31±0.44 |
| opponent-classification | bot_005 | S1 100% win (210/210); pooled +1.33±0.27 SC vs bot_003 but S2 −0.82, S3 −1.44 |
| value-map | bot_002 | pooled +2.20±0.20 SC vs bot_001 (630 paired, B); S1 +6.14±0.20 |

## Champion bot_004 by power (mean SC / win%)

| scenario | AUSTRIA | ENGLAND | FRANCE | GERMANY | ITALY | RUSSIA | TURKEY |
|---|---|---|---|---|---|---|---|
| S1 | 14.8 / 46% | 17.6 / 62% | 13.3 / 25% | 14.2 / 54% | 18.0 / 100% | 18.0 / 100% | 17.5 / 79% |
| S2 | 13.8 / 58% | 13.4 / 29% | 17.1 / 88% | 15.0 / 67% | 13.9 / 62% | 16.6 / 88% | 14.2 / 38% |
| S3 | 9.7 / 29% | 9.5 / 17% | 11.8 / 46% | 11.4 / 33% | 11.3 / 33% | 14.3 / 67% | 9.5 / 8% |
