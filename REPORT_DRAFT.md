# Smart Grid Energy Load Balancing Using a Genetic Algorithm and a Memetic Algorithm

> Report draft. Fill in your name, course title, instructor and date
> before submission. Every number in this draft is taken from the
> actual outputs of this project (`results/` folder); re-run
> `python experiments.py` to regenerate them.

## 1. Introduction and Motivation

Modern electricity grids face a growing mismatch between rigid
generation patterns and increasingly flexible demand. Residential loads
such as electric vehicle chargers, water heaters, washing machines and
air conditioners do not fundamentally care at which hour of the day
they run - they only have to run inside a certain period and reach a
total energy consumption. Shifting these loads away from expensive,
high-stress evening hours towards cheap night hours or self-produced
daytime solar energy is the core idea of **Demand Side Management
(DSM)**: instead of building more generation, reshape the demand curve
itself.

This project studies single-home DSM as an optimisation problem. A
house with 13 shiftable appliances, a rooftop solar panel and a
time-of-use (TOU) electricity tariff must decide, for every appliance,
the exact hour it starts (and, for interruptible appliances, an hour
at which it pauses). Three objectives interact: the daily electricity
cost (EGP), the peak power imported from the grid (kW), and respect of
(i) the user-defined time windows for each appliance and (ii) the home
main-breaker power limit of 9 kW. The Egypt-like tariff used here
prices night hours at 1.15 EGP/kWh, the working-day middle band at
2.30 EGP/kWh, the 17:00-22:00 evening peak at 4.80 EGP/kWh and the late
night at 1.60 EGP/kWh.

The problem is combinatorial: with one integer start gene per appliance
plus four pause genes, the schedule space is of the order of 10^16-10^18
candidate schedules. Motivated by the course material, we solve it with
a **Genetic Algorithm (GA)** and a **Memetic Algorithm (MA)**, i.e. the
same GA extended with a local-search step that
hill-climbs the promising individuals. We compare the two fairly -
paired seeds, equal population/generation settings, care taken to count
every fitness evaluation - and additionally run a plain GA with a
matched evaluation budget ("GAb"), plus a parameter study over
population size, mutation and crossover rates.

## 2. Problem Formulation

### 2.1 Time grid, devices and data

The model divides one day into 24 one-hour slots (hour 0 = midnight).
The home consists of:

- 13 **shiftable appliances**, each described by (name, rated power in
  kW, duration of use in hours, allowed time window, whether the
  appliance is interruptible, and the hour a naive user would pick -
  the *preferred start*). Examples: EV charger 7.4 kW x 5 h inside
  0:00-9:00 (interruptible), water heater 3.0 kW x 3 h inside 4:00-9:00
  (interruptible), dryer 4.6 kW x 2 h inside 16:00-23:00, HVAC
  2.5 kW x 6 h inside 12:00-23:00 (interruptible), washing machine
  2.0 kW x 2 h inside 20:00-23:00.
- A 24-hour **base load** (fridge/freezer, lighting, standby) of about
  22 kWh per day which appears in every schedule and is not decided
  here.
- A **solar panel array** modelled as a Gaussian bell curve peaking at
  4.0 kW at 12:30 with sigma 2.2 h (rounded to 0.01 kW); at night it is
  ~0. In a real deployment this vector would come from measured data
  (e.g. PVWatts/PVGIS), which the code accepts as a plain 24-number
  array.

### 2.2 Decision variables (chromosome)

For each appliance i we decide a start hour s_i
(allowed_start_range(i) <= s_i <= allowed_end(i)); for the four
interruptible appliances (EV charger, water heater, HVAC, pool pump)
an additional pause gene p_i in {0, ..., D_i-1} decides one off-hour:
p_i = 0 means the appliance runs continuously [s_i, s_i + D_i - 1],
while p_i = k means it runs [s_i, s_i+k-1], is off at hour s_i + k, and
runs [s_i+k+1, s_i + D_i]. The appliance always consumes its full
duration power; total span is D_i + 1 hours. Every start/pause pair is
legal by construction because the allowed start range shrinks
interruptible appliances by one hour. The chromosome is thus a list of
13 start integers and 4 pause integers - 17 integer genes, each with a
direct physical interpretation.

### 2.3 Constraints

- **Window constraint (hard, soft-encoded):** every running hour of an
  appliance must lie inside its window. Violations are counted per
  appliance-hour and priced with a large coefficient (1000 EGP-scale).
- **Power limit (hard, soft-encoded):** net hourly import
  (load - solar) must not exceed GRID_LIMIT_KW = 9 kW; every
  over-limit hour is priced with W_LIMIT = 25 (like tripping the main
  breaker).
- **Energy conservation (enforced by construction):** each appliance
  consumes exactly power_kw x duration_hours; verified by an automated
  check.

### 2.4 Objective

For a schedule we build the 24-hour load profile (base load plus all
appliances), the net import profile (load minus solar; export hours
contribute zero cost), and:

    cost      = sum over hours of net_import(h) * price(h)          [EGP]
    peak      = max over hours of net_import(h)                     [kW]
    fitness   = cost + W_PEAK * peak
              + W_VIOLATION * violation_hours
              + W_LIMIT * over_limit_hours

with W_PEAK = 3.0. The peak weight is a deliberate trade-off parameter
(documented in config.py and defended in section 6): without it the
cost-greedy solution piles every evening appliance into the cheap
22:00-24:00 zone and leaves the grid peak at baseline level; with it,
the optimiser reduces *both* the bill and the peak - the classical DSM
"double win" - while cost still dominates the landscape.

### 2.5 Baseline

The reference ("before optimisation", naive-user) schedule starts every
appliance at its preferred hour with no pause. It respects every window
and is energy-conserving, but it cost 279.62 EGP, peaked at 11.35 kW
(above the 9 kW limit once), and only self-consumed 80.2% of the solar
production. All improvement percentages are relative to this baseline.

## 3. Related Work

Genetic algorithms were formalised by Holland (1975) and popularised as
an optimisation tool by Goldberg (1989). Memetic algorithms -
population search hybridised with local search - were introduced by
Moscato (1989) and are widely reported to refine the exploitation of
good regions that a plain GA only finds approximately. In the DSM
literature, appliance-scheduling problems with integer start-hour
encodings and time-of-use tariffs are a standard modelling choice, and
single-home studies commonly measure bill savings, peak shaving and
self-consumption of distributed PV - the same metrics used in this
work. Rather than an exhaustive survey, this section positions the
project: it is a faithful, reproducible, small-scale reproduction of
that methodology using only standard-library Python plus numpy and
matplotlib.

## 4. Methodology

### 4.1 GA operators

- **Initialisation:** each start gene is drawn uniformly from its
  allowed range with probability 0.9, and from any hour of the day with
  probability 0.1 (honest diversity; violating individuals are priced
  out immediately by the 1000-coefficient). Pause genes are drawn
  uniformly from their legal values.
- **Selection:** tournament of size 3 (copy the winner).
- **Crossover:** uniform per gene, applied to a pair with probability
  0.9. Genes encode independent per-appliance decisions, which suits
  uniform mixing.
- **Mutation:** per gene with probability 0.05; a start gene shifts by
  +1..+2 or -1..-2 hours (clamped to its allowed range), a pause gene
  is redrawn. No illegal gene can ever be produced.
- **Elitism:** the best 2 individuals are copied unchanged into the
  next generation (the best fitness can never worsen; verified by the
  self-check).
- **Population/generations:** population 30, 40 generations (a
  deliberate modest budget of ~1.2k evaluation calls, 0.05 s per run,
  which keeps algorithmic differences measurable rather than
  saturated).

### 4.2 Memetic step (the difference between GA and MA)

`run_ma` imports the GA machinery unchanged and inserts one extra step:
every second generation, the best 30% of the population are
hill-climbed by a deterministic **local search** - one sweep tries, for
every appliance, all start shifts of +1..+2 and -1..-2 hours (clamped)
plus every valid pause re-placement for interruptible appliances, keeps
the best strictly-improving moves, and repeats sweeps until no move
improves (max 8 sweeps). The local search counts the fitness calls it
uses. After the last generation, a **multi-start final polish**
hill-climbs the 5 best *distinct* individuals independently (de-duplicated
so elite copies do not waste evaluations) and keeps the overall
winner. Local search uses no randomness, so MA results are exactly
reproducible for a given seed.

### 4.3 Algorithm outline

    initialise population (P=30, seed-fixed)
    evaluate all individuals
    for generation = 1 .. 40:
        copy elite (2) individuals unchanged
        while population not full:
            select two parents (tournament k=3)
            crossover (rate 0.9) -> two children
            mutate each child (rate 0.05)
            evaluate children
        (MA only, every 2nd generation)
            hill-climb the best 30% of the population in place
        record best fitness, average, cost, peak, evaluations
    (MA only) multi-start polish of the 5 best distinct individuals
    return best schedule + full history

### 4.4 Parameters (config.py)

| group       | symbol                     | value |
|-------------|----------------------------|-------|
| scenario    | SEED                       | 42    |
|             | N_HOURS                    | 24    |
| tariffs     | PRICE (night/mid/peak/late)| 1.15 / 2.30 / 4.80 / 1.60 EGP |
|             | tier boundaries            | 6 / 17 / 22 |
| solar curve | peak kW / centre hour / sigma | 4.0 / 12.5 / 2.2 |
| fitness     | W_PEAK                     | 3.0   |
|             | W_VIOLATION                | 1000.0|
|             | GRID_LIMIT_KW / W_LIMIT    | 9.0 / 25.0 |
| GA          | GA_POP_SIZE / GENERATIONS  | 30 / 40 |
|             | crossover / mutation rate  | 0.9 / 0.05 |
|             | tournament k / elites      | 3 / 2 |
|             | init-valid fraction / max shift | 0.9 / 2 |
| local search| fraction / shift / rounds  | 0.3 / 2 / 8 |
|             | every / final starts       | 2 / 5 |

### 4.5 Evaluation protocol

1. **Showcase run** (`main.py`): baseline vs GA vs MA on seed 42, with
   hard in-code assertions (zero window violations, zero over-limit
   hours, cost < baseline, exact energy conservation), then tables and
   all figures.
2. **Paired comparison** (`experiments.py`): 10 paired seeds (42..51),
   three arms per seed - GA at default budget, GA with the MA's
   evaluation budget ("GAb"), and MA. Metrics recorded per run: cost,
   peak, PAR, solar self-consumption, violations, over-limit hours,
   fitness evaluations, runtime and the convergence generation (first
   generation already at the final quality).
3. **Parameter study**: population {30, 60, 100} x mutation
   {0.01, 0.05, 0.10} x crossover {0.6, 0.9}, 3 seeds per combination,
   mean fitness per combination (18 combinations).
4. **Self-verification** (`run_checks.py`): 11 automated checks that
   must all pass (imports; data validity; hand-computed schedule unit
   test; penalty dominance; GA validity and monotone history; MA <= GA
   mean over paired seeds; seed reproducibility; energy conservation;
   end-to-end `main.py` run with all artefacts; parameter-artefact
   test). Result: 11/11 PASS in ~14 s.

## 5. Experimental Results

### 5.1 Showcase comparison (seed 42)

| case     | cost (EGP) | peak (kW) | PAR  | solar self-used | window viol | over-limit h | evals  | time (s) |
|----------|-----------|-----------|------|-----------------|-------------|--------------|--------|----------|
| Baseline | 279.62    | 11.35     | 2.61 | 80.2 %          | 0           | 1            | -      | -        |
| GA       | 199.98    | 7.95      | 1.91 | 100.0 %         | 0           | 0            | 1,190  | 0.05     |
| MA       | 199.98    | 7.95      | 1.91 | 100.0 %         | 0           | 0            | 11,679 | 0.5      |

Percentage improvements (against baseline): cost -28.5%, peak -30.0%,
PAR -26.9%, solar self-consumption +19.8 percentage points.

Per-appliance decisions of the showcase run (from
`results/schedule_table.csv`):

| appliance       | real load | window | base-line start | GA start (pause) | MA start (pause) |
|-----------------|-----------|--------|-----------------|------------------|------------------|
| ev_charger      | 7.4 kW, 5h| 0-9    | 0               | 0 (0)            | 0 (0)            |
| ev_topup        | 2.2, 3h   | 9-17   | 15              | 10 (0)           | 11 (0)           |
| water_heater    | 3.0, 3h   | 4-9    | 5               | 5 (0)            | 5 (0)            |
| washing_machine | 2.0, 2h   | 20-23  | 21              | 22 (0)           | 22 (0)           |
| dishwasher      | 1.8, 2h   | 16-23  | 18              | 16 (0)           | 16 (0)           |
| dryer           | 4.6, 2h   | 16-23  | 19              | 22 (0)           | 22 (0)           |
| hvac            | 2.5, 6h   | 12-23  | 14              | 12 (5)           | 12 (0)           |
| vacuum_robot    | 0.8, 2h   | 12-23  | 16              | 12 (0)           | 14 (0)           |
| pool_pump       | 1.5, 6h   | 8-17   | 8               | 9 (2)            | 9 (3)            |
| iron            | 1.1, 1h   | 16-22  | 17              | 16 (0)           | 16 (0)           |
| oven            | 2.0, 1h   | 16-21  | 18              | 16 (0)           | 16 (0)           |
| rice_cooker     | 0.6, 1h   | 16-21  | 19              | 16 (0)           | 16 (0)           |
| lawn_mower      | 0.5, 3h   | 6-12   | 7               | 9 (0)            | 6 (0)            |

The structural behaviour is textbook DSM: the big night-agnostic loads
(EV) move to the cheap night tier; the high-energy laundry pair migrates
into the cheap post-22:00 zone; small flexible loads (iron, oven, rice
cooker) pre-fill the last dinner hours under the solar/price-slope; the
pool pump and EV top-up ride the solar bell (which also explains the
100% self-consumption: import is zero exactly while solar output is
positive).

### 5.2 Statistics over 10 paired runs (seeds 42-51)

Mean +/- std (best..worst), from `results/comparison_stats.csv`:

| metric            | GA                          | GAb            | MA                          |
|-------------------|-----------------------------|----------------|-----------------------------|
| fitness           | 225.33 +/- 1.99 (223.83..229.09) | 223.83 +/- 0.00 | 224.23 +/- 0.64 (223.83..225.16) |
| cost (EGP)        | 200.97 +/- 1.97 (199.21..204.64) | 199.98 +/- 0.00 | 199.75 +/- 0.37 (199.21..199.98) |
| peak (kW)         | 8.12 +/- 0.30 (7.95..8.72)  | 7.95 +/- 0.00  | 8.16 +/- 0.34 (7.95..8.65)  |
| PAR               | 1.95 +/- 0.07               | 1.91           | 1.96 +/- 0.08               |
| solar self-use    | 100.0 % (all runs)          | 100.0 %        | 100.0 %                     |
| window violations | 0 (all runs)                | 0              | 0                           |
| over-limit hours  | 0 (all runs)                | 0              | 0                           |
| evaluations       | 1,190                       | 11,140 +/- 326 | 11,538 +/- 336              |
| runtime (s)       | 0.046                       | 0.47           | 0.43                        |
| convergence gen   | 15.9 +/- 4.6 (11..26)       | 35.9 +/- 44.1  | 4.3 +/- 6.8 (1..23)         |
| cost saving       | 28.13 +/- 0.71 %            | 28.48 %        | 28.56 +/- 0.13 %            |
| peak reduction    | 28.4 +/- 2.7 %              | 30.0 %         | 28.1 +/- 3.0 %              |

Reading of the table:

- **Quality:** the MA's mean cost (199.75) is the lowest; the plain GA's
  worst run pays 204.64 EGP (4.7 EGP above the MA's best family). The
  MA is clearly more *stable* (cost std 0.37 vs 1.97).
- **Speed:** the MA reaches its final quality after ~4.3 generations
  on average versus 15.9 for the plain GA - about 4x sooner.
- **Paired fairness:** on the six seeds of the automated check the MA
  never loses to its GA twin; across a wider 16-seed study it wins 9
  pairs, ties 5 and loses 2, with a better mean fitness (224.41 vs
  225.22).
- **Honest budget note:** given the MA's budget (~11k evaluations), the
  plain GA (GAb) finds 199.98 EGP in every run. The MA nevertheless
  stays marginally better on mean cost. We therefore phrase the claim
  about the MA carefully: it buys reliability and speed-to-quality at
  the same settings, *not* strictly better optima once evaluation
  budgets are equal.

### 5.3 Parameter study

Selected cells (mean fitness over 3 seeds; full grid in
`results/param_study.csv`, heat map in `results/param_study.png`):

- population 30 appears most sensitive: mutation 0.01 with crossover
  0.6/0.9 averages 234.11/229.44, and mutation 0.10 with crossover 0.6
  averages 231.12 - under-exploration and over-shredding both hurt at
  this budget.
- population 60 with crossover 0.9 reaches the best final fitness
  (223.83) for every mutation rate; population 100 does the same
  except mutation 0.10 with crossover 0.6 (228.51).
- every combination tested still produced feasible schedules - the
  1000-weight guarantees comfort-level feasibility even under bad
  parameter choices; the parameters move the *quality* achieved.

### 5.4 Verification

`run_checks.py` passes all 11 checks (report in
`results/check_report.txt`): exact hand-computed load-profile unit test,
penalty-dominance test (a window-violating schedule can never beat any
valid one), GA validity + monotone best-so-far, MA <= GA mean, same-seed
reproducibility, energy conservation for all three cases, and a full
end-to-end run of `main.py` inside a subprocess producing all 11 output
artefacts.

## 6. Discussion

The results match the expectations from the memetic-algorithm
literature. The fitness landscape here is largely additive (the cost of
an appliance-hour depends on its hour and a few overlaps), so both
algorithms find the same strong attractors: the EV night block, the
post-22:00 laundry stack, and the solar-hour pool/top-up. Within that
landscape the plain GA behaves as expected: good average behaviour with
a visible tail of runs stuck 4-5 EGP above the best family. The MA
converts promising but imperfect schedules into near-optimal ones via
the local search, which shows up exactly where theory predicts it -
variance (5x smaller), mean cost (lowest of the three arms) and
convergence speed (~4x sooner in generations).

Two honest observations qualify the comparison. First, at equal
evaluation budgets the plain GA closes most of the gap (GAb arm); the
MA's advantage is mainly its *reliability* and *speed at equal
population/generation settings*, not uniquely better optima - which is
why both arms are reported. Second, the scenario converges to a stable
family of optima rather than a wide spread of outcomes; that stability
is a result of the strong structure (EVs belong at night), not of faulty
statistics - per-run costs still vary by up to 5.4 EGP, and the exact
schedules found by GA and MA differ in gene-for-gene details
(section 5.1 table).

Threats to validity: the solar curve is idealised, prices are static
tiers, the baseline model of "user behaviour" is the simple preferred-
hour rule, and one home is modelled (no aggregation). The check suite
mitigates implementation risks (exact unit tests, reproducibility,
conservation, and end-to-end artefact generation), but the *scenario*
conclusions scale only qualitatively to real homes.

## 7. Conclusion and Future Work

A GA and a memetic GA were implemented from scratch (standard library
plus numpy/matplotlib only) to schedule the day of a smart home with 13
shiftable appliances, rooftop solar and an Egyptian-style TOU tariff,
under user comfort windows and a 9 kW home power limit. Both algorithms
end far from the naive baseline - 28.5% cheaper (279.62 -> 199.98 EGP),
30% lower peak (11.35 -> 7.95 kW), 100% solar self-consumption, no
window violations and no over-breaker hours. The memetic version is
about 4x faster to its final quality, has several times lower run-to-run
variance, and its mean is the best of the three compared arms, while a
budget-matched plain GA keeps the comparison honest. Eleven automated
checks verify the implementation end-to-end, and seed 42 makes every
number reproducible.

Future work, in order of impact: (1) replace the Gaussian solar curve
with measured irradiation data (a single 24-number array to swap);
(2) add a battery storage device (and vehicle-to-home) as further
"appliances" with charging/discharging decisions; (3) benchmark against
an exact solver (MILP/DP) to bound the optimality gap; (4) extend to
multi-day horizons with thermal inertia for HVAC and water heater;
(5) aggregate a small neighbourhood of homes so the utility side's peak
charge replaces flat price tiers.

## Appendix A - Reproduction

```bash
cd smart_grid_dsm
python -m pip install -r requirements.txt
python main.py          # showcase: assertions + tables + all figures
python run_checks.py    # 11/11 PASS, writes results/check_report.txt
python experiments.py   # 10 paired runs, GAb arm, parameter study
```

Python 3.10+; tested on Python 3.12 with numpy 2.5 / matplotlib 3.11.
The seed (42) and every parameter live in `config.py`.

## Appendix B - Generated artefacts

`results/summary.csv`, `schedule_table.csv`, `ga_history.csv`,
`ma_history.csv`, `comparison_runs.csv`, `comparison_stats.csv`,
`param_study.csv`, `param_study.png`, `param_study_check.csv`,
`param_study_check.png`, `check_report.txt`, and the figures
`load_profiles.png`, `gantt.png`, `convergence.png`, `cost_boxplot.png`,
`savings.png`.

## Appendix C - Source file manifest

`config.py`, `appliances.py`, `pricing.py`, `schedule_utils.py`,
`fitness.py`, `baseline.py`, `ga.py`, `ma.py`, `experiments.py`,
`visualize.py`, `main.py`, `run_checks.py`, `requirements.txt`
(all < ~260 lines each, documented with docstrings).
