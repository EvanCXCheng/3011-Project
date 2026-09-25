"""Lab configuration: the single place for constants used by the lab scripts."""
import os
from datetime import datetime, timezone, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYTHON = os.path.join(ROOT, '.venv', 'bin', 'python')

GROUP = 21
AGENT_FILE = os.path.join(ROOT, f'agent_{GROUP}.py')
AGENT_CLASS = 'StudentAgent'          # class name that test.py imports; every bot must define it
BOTS_DIR = os.path.join(ROOT, 'bots')
RESULTS_DIR = os.path.join(ROOT, 'results')
RAW_DIR = os.path.join(RESULTS_DIR, 'raw')
LOG_DIR = os.path.join(RESULTS_DIR, 'logs')
RESULTS_CSV = os.path.join(RESULTS_DIR, 'results.csv')
STATE_FILE = os.path.join(RESULTS_DIR, 'state.json')
JOURNAL = os.path.join(ROOT, 'JOURNAL.md')
LEADERBOARD = os.path.join(ROOT, 'LEADERBOARD.md')

POWERS = ['AUSTRIA', 'ENGLAND', 'FRANCE', 'GERMANY', 'ITALY', 'RUSSIA', 'TURKEY']
END_YEAR = 1920
WIN_SC = 18

# Parallelism: every core but one
WORKERS = max(1, (os.cpu_count() or 2) - 1)
MAX_TASKS_PER_CHILD = 25

# Hard rules
TIME_LIMIT = 1.0          # s, enforced per agent call with timeout_decorator
TIME_TARGET = 0.6         # s, Tier 0 gate on slowest get_actions (serial run)
TIME_P99_GATE = 0.9       # s, parallel runs: any move above this is flagged
MEM_LIMIT_AS_MB = 512     # RLIMIT_AS per worker process (whole game incl. opponents)
MEM_GATE_MB = 400         # Tier 0 gate on ru_maxrss of the worker
GAME_CPU_LIMIT = 900      # s of CPU per game before it is aborted (SIGPROF watchdog)
FILE_SIZE_LIMIT = 100 * 1024

# Scenario 2 opponent pool, exactly as test.py (Random is rarer)
S2_POOL = ['random', 'attitude', 'attitude', 'greedy', 'greedy']

# Seed sets: seed = base + i. Seat = POWERS[seed % 7], so n divisible by 7 balances powers.
SEED_SETS = {'A': 100000, 'B': 200000, 'C': 300000, 'FINAL': 400000, 'T': 500000, 'X': 900000}

# Tier sizes (games per scenario). Multiples of 7.
TIER0_N = 3
TIER1_N = 42
TIER2_N = 210
TIER2_FALLBACK_N = 126
FINAL_N = 504

# Rubric: scenario -> list of (points, win_rate_threshold, mean_sc_threshold); strict '>' comparisons.
RUBRIC = {
    1: [(1, 0.02, 7), (3, 0.20, 12), (5, 0.90, 16)],
    2: [(1, 0.02, 7), (3, 0.25, 10), (5, 0.50, 13)],
    3: [(1, 0.02, 7), (3, 0.20, 9), (5, 0.40, 12)],
}

FAMILIES = ['greedy', 'valuemap', 'search', 'lookahead', 'adaptive', 'positional']
FAMILY_CAP = 0.40
FAMILY_MIN_ITERS = 3

AWST = timezone(timedelta(hours=8))
FREEZE = datetime(2026, 10, 1, 12, 0, tzinfo=AWST)
DEADLINE = datetime(2026, 10, 2, 23, 59, tzinfo=AWST)
