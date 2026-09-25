# Leaderboard

_Generated 2026-09-26 01:38 AWST by lab/leaderboard.py — do not hand-edit._  
Freeze in 130.4 h, deadline in 166.3 h.

**Overall champion:** bot_001  (agent_21.py matches)

## Bots

Mark = estimated rubric points /15 (per scenario: highest tier reached by win rate or mean SC). `*` = within 1 SE of a threshold. Pooled over seed sets A/B/C/FINAL, current file hash only.

| # | bot | family | status | S1 sc (win%) n | S2 sc (win%) n | S3 sc (win%) n | mark | tmax s | mem MB | issues | tourn |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | baseline:greedy | - | - | 7.39±0.40 (4%) 49 → 1* | 9.92±0.98 (31%) 49 → 3* | 7.65±0.85 (14%) 49 → 1* | **5** | 0.051 | 101 | 12 | - |
| 2 | bot_002_valuemap_base | valuemap | RUNNING | 12.43±0.60 (0%) 23 → 3* | - | - | **3** | 0.030 | 98 | 0 | - |
| 3 | bot_001_greedy_base | greedy | PROMOTED | 6.14±0.16 (0%) 252 → 0 | 7.58±0.31 (8%) 252 → 1 | 6.13±0.25 (3%) 252 → 1 | **2** | 0.074 | 102 | 0 | - |
| 4 | baseline:attitude | - | - | 6.76±0.24 (0%) 49 → 0 | 1.55±0.26 (0%) 49 → 0 | 1.37±0.23 (0%) 49 → 0 | **0** | 0.079 | 101 | 0 | - |
| 5 | baseline:random | - | - | 6.55±0.25 (0%) 49 → 0 | 1.55±0.30 (0%) 49 → 0 | 1.37±0.29 (0%) 49 → 0 | **0** | 0.074 | 100 | 0 | - |
| 6 | baseline:static | - | - | 3.14±0.05 (0%) 49 → 0 | 2.45±0.16 (0%) 49 → 0 | 2.41±0.16 (0%) 49 → 0 | **0** | 0.009 | 100 | 0 | - |

## Families

| family | status | iterations (share) | champion | S1 | S2 | S3 | mark | reason |
|---|---|---|---|---|---|---|---|---|
| adaptive | ACTIVE | 0 (0%) | - | - | - | - | - |  |
| greedy | ACTIVE | 1 (33%) | bot_001 | 0 | 1 | 1 | 2 |  |
| lookahead | ACTIVE | 0 (0%) | - | - | - | - | - |  |
| positional | ACTIVE | 0 (0%) | - | - | - | - | - |  |
| search | ACTIVE | 1 (33%) | - | - | - | - | - |  |
| valuemap | ACTIVE | 1 (33%) | - | - | - | - | - |  |

**Hall of fame:** bot_001  
**Overall champion history:** bot_001

## Report coverage: techniques

Distinct techniques measured: 0 (need basic + 3 new).

| technique tag | first bot | measured effect vs parent |
|---|---|---|

## Champion bot_001 by power (mean SC / win%)

| scenario | AUSTRIA | ENGLAND | FRANCE | GERMANY | ITALY | RUSSIA | TURKEY |
|---|---|---|---|---|---|---|---|
| S1 | 7.0 / 0% | 3.0 / 0% | 4.0 / 0% | 7.0 / 0% | 7.0 / 0% | 11.0 / 0% | 4.0 / 0% |
| S2 | 7.6 / 11% | 6.6 / 3% | 6.6 / 0% | 7.9 / 8% | 8.1 / 14% | 10.5 / 17% | 5.6 / 0% |
| S3 | 5.8 / 8% | 5.4 / 0% | 6.2 / 0% | 6.0 / 3% | 5.6 / 3% | 9.2 / 8% | 4.8 / 0% |
