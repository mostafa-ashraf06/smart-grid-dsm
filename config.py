"""All tunable constants and parameters of the project in ONE place.

Change any number here to experiment with different scenarios
(prices, solar size, appliance penalties, GA settings, ...).

Units used everywhere in this project:
  - power        in kW (kilowatt)
  - energy       in kWh (kilowatt-hour)
  - money        in EGP (Egyptian pound)
  - time         in hours (0 .. 23, one hour per slot)
"""

import os

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# A fixed seed makes every random choice (initial population, selection,
# crossover, mutation) repeatable, so results can be checked and compared.
SEED = 42

# ---------------------------------------------------------------------------
# Time grid
# ---------------------------------------------------------------------------
N_HOURS = 24          # one full day, hour 0 = midnight, hour 23 = 23:00.

# ---------------------------------------------------------------------------
# Time-of-use (TOU) electricity prices, in EGP per kWh
# ---------------------------------------------------------------------------
# Three tiers + a cheap "late night" return tier, chosen to look like real
# Egyptian residential tariffs (peak hours are the most expensive).
PRICE_OFF_PEAK = 1.15   # 00:00 - 05:59  (low demand, cheapest)
PRICE_MID = 2.30        # 06:00 - 16:59  (working hours)
PRICE_PEAK = 4.80       # 17:00 - 21:59  (evening peak, most expensive)
PRICE_LATE = 1.60       # 22:00 - 23:59  (demand collapses again)
OFF_PEAK_END = 6        # first tier covers hours 0 .. OFF_PEAK_END - 1
MID_END = 17            # second tier covers hours OFF_PEAK_END .. MID_END - 1
PEAK_END = 22           # third tier covers hours MID_END .. PEAK_END - 1

# ---------------------------------------------------------------------------
# Rooftop solar panel curve (kW)
# ---------------------------------------------------------------------------
# A Gaussian ("bell") curve centred a little after noon. Zero at night,
# about 4 kW at the best hour. In a real project you would replace this
# with measured data (e.g. NREL PVWatts or PVGIS export for your city);
# the rest of the code does not care where the 24 numbers come from.
SOLAR_PEAK_KW = 4.0       # highest output of the panels, at the centre hour
SOLAR_CENTER_HOUR = 12.5  # centre of the bell (between 12:00 and 13:00)
SOLAR_SIGMA_HOURS = 2.2   # width of the bell; bigger = sunshine lasts longer

# ---------------------------------------------------------------------------
# Fitness weights
# (fitness = cost + W_PEAK*peak + W_VIOLATION*violations + W_LIMIT*over_limit)
# ---------------------------------------------------------------------------
# Why W_PEAK = 3.0 (tuned on purpose)?  Daily costs in this scenario sit in
# the 200-280 EGP range, so a kilowatt of peak is NOT noise: peak shaping is
# a FIRST-CLASS goal next to the bill. With the naive baseline the
# cost-greedy schedule would pile almost every evening appliance into the
# cheap 22-24 zone - it saves a few extra EGP but leaves the grid peak at
# the baseline level (bad for the grid). Measuring the real tradeoff (see
# DEFENSE_GUIDE.md): a few EGP more buys a much flatter import profile, and
# with W_PEAK = 3.0 the flatter solution wins, so the optimiser reduces BOTH
# bill and peak - the classic DSM double win - while cost still dominates.
# Why W_VIOLATION = 1000?  A whole optimised day costs at most ~280 EGP, and
# even a deliberately wasteful schedule stays far below ~400 EGP, so ONE
# appliance-hour outside its window (1000 EGP) ALWAYS makes a violating
# schedule worse than every valid one. This is what forces the algorithms
# to respect user comfort first, then optimise.
W_PEAK = 3.0
W_VIOLATION = 1000.0

# ---------------------------------------------------------------------------
# Home power limit (smart meter / main breaker) - an extra, documented term
# ---------------------------------------------------------------------------
# A real home can not import more power than its grid connection allows
# (main breaker, typically 3-phase ~10-12 kW in Egypt). Buying a 1 kW
# appliance-hour ABOVE the limit usually means tripping the breaker, so it
# behaves like a second kind of constraint violation:
#     violation_amount = window-violation hours  (weight W_VIOLATION)
#                      + over-limit hours        (weight W_LIMIT)
# This is what makes the problem a real PACKING problem: the cheap 22:00-24:00
# zone can not simply swallow every evening appliance, and the optimizers
# must coordinate who runs where - per-appliance greedy logic fails, which
# is exactly the difficulty a memetic algorithm should pay off on.
GRID_LIMIT_KW = 9.0       # maximum grid import the home may draw, in kW
W_LIMIT = 25.0            # EGP-scale penalty for each over-limit hour

# ---------------------------------------------------------------------------
# Genetic Algorithm (GA) parameters
# ---------------------------------------------------------------------------
# A deliberately MODEST budget: each GA generation evaluates pop_size
# schedules, and with pop 30 x 40 generations the plain GA needs ~1.2k
# fitness evaluations and about 0.1 s per run. At this budget the plain GA
# often stops in a near-optimum (5 of 15 seeds in our studies), which is
# exactly where the memetic algorithm's local search pays off - see the
# comparison in experiments.py / DEFENSE_GUIDE.md.
GA_POP_SIZE = 30            # number of schedules (individuals) per generation
GA_GENERATIONS = 40         # how many generations to evolve
GA_CROSSOVER_RATE = 0.9     # probability that two parents swap genes
GA_MUTATION_RATE = 0.05     # per-gene probability of a small random change
GA_TOURNAMENT_K = 3         # tournament selection size (k = 3)
GA_ELITE_COUNT = 2          # best individuals copied to the next generation
GA_INIT_VALID_FRACTION = 0.9    # 90% of initial starts are inside the window
GA_MUTATION_MAX_SHIFT = 2   # mutation may shift a start by +-1 .. +-2 hours

# ---------------------------------------------------------------------------
# Memetic Algorithm (MA) parameters (GA + local search)
# ---------------------------------------------------------------------------
# Local search (hill climbing) is applied to the best LS_FRACTION of the
# population every LS_EVERY generations (every 2 generations leaves one
# "pure GA" generation in between, which keeps some diversity while the
# stabilizing LS still runs 20 of the 40 generations). One FINAL polish is a
# MULTI-START hill climbing that starts from the LS_FINAL_TOP best DISTINCT
# individuals of the final population - not only from the single best - so
# the MA gets several independent "last chances" to leave a mediocre basin.
# LS_MAX_SHIFT limits how far a start can move per trial, LS_MAX_ROUNDS
# limits how many improvement sweeps are done.
LS_FRACTION = 0.3           # apply local search to the best 30% each round
LS_MAX_SHIFT = 2            # try shifts of +-1 and +-2 hours
LS_MAX_ROUNDS = 8           # max sweeps of "try every appliance" per call
LS_EVERY = 2                # run local search every 2nd generation
LS_FINAL_TOP = 5            # final multi-start polish over 5 best individuals

# ---------------------------------------------------------------------------
# Chromosome layout (IMPORTANT - documented so it can be explained)
# ---------------------------------------------------------------------------
# The chromosome is a plain list of minutes... no, of HOURS (integers):
#
#   genes[0 .. N-1]                 -> ONE gene per appliance = its START hour
#   genes[N .. N + N_interruptible) -> one "pause" gene for every INTERRUPTIBLE
#                                      appliance (extra genes, see below)
#
# The main gene of each appliance is exactly its start hour, as the
# assignment requires ("a vector where each gene represents the start time
# of an appliance").
#
# EXTRA GENES - documented choice for interruptible appliances:
# An interruptible appliance may switch off for ONE hour somewhere inside
# its run (one pause allowed, e.g. the EV charger pauses mid-charge).
# Its extra gene p is an integer in {0, 1, ..., D-1} where D = duration:
#     p = 0        -> no pause, the appliance runs one continuous block
#                     [start, start + D - 1]
#     p = 1..D-1   -> it runs and switches off at hour (start + p):
#                     block 1 = [start, start + p - 1],
#                     pause   = (start + p)  (off for this hour),
#                     block 2 = [start + p + 1, start + D]
#     So exactly D hours of power are consumed in a span of D+1 hours.
# Because one pause hour is possible, the allowed start range of an
# interruptible appliance is shrunk by one hour so EVERY start/pause pair
# that mutation can produce is valid; window violations are then impossible
# except for the few "diversity" genes we inject on purpose (they are killed
# by the huge W_VIOLATION penalty). This keeps the GA fast and simple.

# ---------------------------------------------------------------------------
# File locations
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def ensure_results_dir():
    """Create the results/ folder if it does not exist yet.

    No inputs. Returns the path to the results folder as a string.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR
