# Smart Grid Energy Load Balancing - GA and Memetic Algorithm

A complete course project: optimise the daily electricity schedule of ONE
smart home with 13 shiftable appliances, a rooftop solar panel and a
time-of-use (TOU) tariff, using a Genetic Algorithm (GA) and a Memetic
Algorithm (MA = GA + local search). Pure Python 3.10+, only `numpy`,
`matplotlib` and the standard library - no GA libraries, no web services.

## The problem (one sentence)

Every hour of the day, the home must choose WHEN to run each appliance
(start hour + optional pause hour) so that the electricity bill, the
imported peak power and the home power limit are minimised, while every
appliance stays inside the time window the user allows (comfort) and
every appliance still runs its full duration.

## What the optimiser achieves (own scenario, seed 42 showcase)

| case     | cost (EGP) | peak import (kW) | PAR  | solar self-used | window viol | over-limit h |
|----------|-----------|------------------|------|-----------------|-------------|--------------|
| Baseline | 279.62    | 11.35            | 2.61 | 80.2 %          | 0           | 1            |
| GA       | 199.98    | 7.95             | 1.91 | 100.0 %         | 0           | 0            |
| MA       | 199.98    | 7.95             | 1.91 | 100.0 %         | 0           | 0            |

Over 10 paired runs: GA vs MA - MA mean fitness 224.23 (std 0.64) versus
GA 225.33 (std 1.99); the MA never loses to its GA twin on the seed family
used for the automated checks and converges in ~4 generations vs ~16.

## Quick start

```bash
cd smart_grid_dsm
python -m pip install -r requirements.txt   # numpy + matplotlib
python main.py                # showcase run: tables + plots in results/
python run_checks.py          # 11 self-checks, writes results/check_report.txt
python experiments.py         # 10 paired runs + GA-matched-budget arm
                              # + parameter study (heatmap), ~1-2 minutes
```

IMPORTANT: always run from inside the `smart_grid_dsm` folder - the modules
import each other with flat names (`from config import ...`).

## Project structure (all at the project root)

| file              | role                                                                 |
|-------------------|----------------------------------------------------------------------|
| `config.py`       | ALL tunable constants: seed, prices, solar, weights, GA/LS params    |
| `appliances.py`   | the 13 appliances + the 24-hour base (non-shiftable) load            |
| `pricing.py`      | TOU price array + solar bell curve                                   |
| `schedule_utils.py`| chromosome -> schedule -> hourly load; violation counting; checks   |
| `fitness.py`      | cost, peak, power-limit counting, solar self-consumption, `evaluate` |
| `baseline.py`     | the naive "user preferred hours" schedule (no optimisation)          |
| `ga.py`           | chromosome creation, tournament selection, crossover, mutation, `run_ga` |
| `ma.py`           | `local_search` (hill climbing) + `run_ma` (GA loop + memetic step)   |
| `experiments.py`  | paired GA/MA/GAb comparison, statistics, parameter study             |
| `visualize.py`    | all plots (load profiles, Gantt, convergence, box plot, heatmap)     |
| `main.py`         | end-to-end pipeline: baseline, GA, MA, checks, tables, plots         |
| `run_checks.py`   | self-verification of the whole project (imports, data validity,      |
|                   |   unit tests, GA/MA behaviour, reproducibility, end-to-end run)      |
| `results/`        | generated CSVs and PNGs (see below)                                  |

## Results folder contents

```
summary.csv             - baseline vs GA vs MA headline numbers (showcase run)
schedule_table.csv      - per-appliance: window, baseline start vs GA/MA start+pause
ga_history.csv          - GA convergence history (fitness per generation)
ma_history.csv          - MA convergence history
comparison_runs.csv     - raw numbers of the 10 paired runs (GA / GAb / MA)
comparison_stats.csv    - mean and std of every metric per algorithm
param_study.csv         - GA parameter grid (population x mutation x crossover)
param_study.png         - parameter heat map
load_profiles.png       - hourly load baseline vs optimised + price + solar
gantt.png               - before/after appliance schedule (Gantt chart)
convergence.png         - GA vs MA fitness over generations
cost_boxplot.png        - per-run cost distribution GA/GAb/MA
savings.png             - percent improvements per algorithm
check_report.txt        - full output of run_checks.py (11/11 PASS)
param_study_check.png / .csv - small parameter-study artefact of the checks
```

## The design in 60 seconds

- **Chromosome**: one gene per appliance = its START hour, plus one extra
  "pause" gene for each interruptible appliance (EV charger, water heater,
  HVAC, pool pump can switch off for one mid-run hour). Fixed length, one
  integer per gene - easy to explain, easy to mutate.
- **Fitness** (one number, EGP-scale):
  `cost + 3.0*peak_import + 1000*window_violation_hours + 25*over_limit_hours`.
  Cost and peak shape the tradeoff; 1000 and 25 kill every infeasible
  schedule comfortably below all valid ones.
- **GA**: tournament selection (k=3), uniform crossover (rate 0.9),
  bounded mutation (+-1..2 h, rate 0.05), 2 elites, population 30,
  40 generations (modest on purpose: about 1.2k evaluations, 0.05 s).
- **MA**: the same GA loop, but every 2nd generation the best 30% are
  hill-climbed (per-appliance best-improving start shifts + pause moves),
  plus a multi-start final polish over the 5 best distinct individuals.
- **Fairness**: every run records how many fitness evaluations it used;
  the comparison also contains a GA arm with MATCHED evaluation budget.

## Reproducibility

`config.SEED = 42` fixes every random choice; `run_checks.py` check 8
proves two runs with the same seed produce the identical result. All
experiments use paired seeds (run i uses the same seed for GA and MA),
which is why the per-run table in `comparison_runs.csv` is directly
comparable.

## Requirements

Python 3.10+ with `numpy>=1.24` and `matplotlib>=3.7`. Tested on Python
3.12 with numpy 2.5 / matplotlib 3.11. Windows console is fine - all
output is ASCII and diagrams are written to PNG files (no interactive
windows).

## Documentation

- `DEFENSE_GUIDE.md` - plain-language file-by-file explanation, worked
  example, design rationale, glossary, and 28 realistic professor Q&As.
- `REPORT_DRAFT.md` - ready-to-edit academic draft (7 sections) with the
  real produced numbers and figures.
