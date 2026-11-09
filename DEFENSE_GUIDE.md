# DEFENSE GUIDE - Smart Grid Energy Load Balancing (GA & MA)

This file is your "speaking map" for the project defense. It contains:
1. A plain-language tour of every file.
2. A worked step-by-step example (chromosome -> schedule -> fitness).
3. The design rationale (WHY every choice).
4. A glossary of every term you must be able to define.
5. 28 questions a professor is likely to ask, with answers that match the
   actual code and the actual produced numbers.

All numbers quoted below are real outputs produced by this project
(results/ folder), not made-up values.

========================================================================
1. THE STORY YOU CAN TELL IN 30 SECONDS
========================================================================

"One smart home has 13 shiftable appliances (EV charger, water heater,
washing machine, dryer, HVAC, pool pump, ...), a rooftop solar panel and
an Egyptian-style TOU tariff (1.15 EGP at night, 2.30 mid-day, 4.80 at
the 17:00-22:00 peak, 1.60 late). A naive user runs each appliance at its
preferred hour - that costs 279.62 EGP, peaks at 11.35 kW and exceeds the
home's 9 kW grid limit once. I represent a schedule by its start hours
(plus a pause hour for interruptible appliances) and optimise it with a
Genetic Algorithm; the Memetic Algorithm adds hill climbing on the best
individuals. The optimised day costs 199.98 EGP (-28.5%), peaks at
7.95 kW (-30.0%), never violates a comfort window, never trips the 9 kW
limit and self-consumes 100% of the solar production. The MA is more
stable than the plain GA (std 0.37 vs 1.97 EGP over 10 runs) and
converges in ~4 generations vs ~16, while I also report a plain GA given
the same evaluation budget as the MA."

========================================================================
2. FILE-BY-FILE TOUR (what it does, in words)
========================================================================

config.py
- The ONE place with every tunable number: SEED=42 (reproducibility),
  24-hour time grid, the four TOU price tiers, the solar bell curve
  (peak 4 kW at 12:30, sigma 2.2 h), fitness weights W_PEAK=3.0,
  W_VIOLATION=1000, the home power limit GRID_LIMIT_KW=9.0 with
  W_LIMIT=25, GA parameters (population 30, generations 40, crossover
  0.9, mutation 0.05, tournament 3, elite 2, init-valid 0.9, shift 2),
  local-search parameters (30% fraction, shift 2, 8 rounds, every 2
  generations, 5 final starts) and the results folder helper.
- Professor says "where is this number?" -> it is HERE, always.

appliances.py
- Thirteen appliances, each a small dictionary: name, power_kw,
  duration_hours, window_start/window_end (when the user ALLOWS it),
  interruptible (may pause one hour mid-run or not) and preferred_start
  (the hour the naive user picks: this defines the BASELINE).
- Also BASE_LOAD_KW: the 24-hour non-shiftable background load
  (fridge/freezer, lighting, standby, ...), ~22 kWh/day - this load
  appears in EVERY schedule, so the optimiser only decides about the 13.
- Examples to quote: ev_charger 7.4 kW x 5 h inside 0-9 (interruptible);
  dryer 4.6 kW x 2 h inside 16-23; hvac 2.5 kW x 6 h inside 12-23.

pricing.py
- get_price_array(): returns the 24 EGP/kWh values from the price tiers.
- get_solar_array(): a Gaussian solar bell: P = peak *
  exp(-((h-centre)^2)/(2*sigma^2)), rounded to 2 decimals; at night ~0.
  The docstring makes the honest point: in a real project you would take
  this vector from measured data (e.g. PVWatts/PVGIS export); the rest
  of the code does not care where the 24 numbers come from.

schedule_utils.py
- The chromosome interpreter. allowed_start_range() clamps each appliance
  to valid start hours (interruptible appliances lose one hour so any
  pause gene stays legal). get_running_hours() converts (start, pause)
  -> the exact list of on-hours (the pause gene is a documented design:
  p=0 runs continuously; p=k means "switch off at hour start+k").
  compute_load_profile() builds the 24-hour total load vector
  (BASE_LOAD + every appliance's on-hours). count_violation_hours()
  counts appliance-hours outside the window; schedule_is_valid() asks
  "is this schedule feasible?"; get_run_blocks() makes Gantt intervals;
  total_appliance_energy() proves energy conservation (power * hours).

fitness.py
- evaluate(schedule, ...) -> ONE number and a report:
  net_import = load - solar; cost = sum(import * price); peak = max
  import; PAR = peak / mean import; zero or negative import (export)
  hours contribute 0 EGP. The fitness is
      cost + 3.0*peak + 1000*window_violation_hours + 25*over_limit_hours.
  solar_self_consumption = sum(min(load, solar)) / sum(solar) over the
  day. The fitness dictionary also carries cost, peak, PAR, violations,
  limit_violations and the two arrays for the plots - "one call answers
  everything".

baseline.py
- get_baseline_schedule(): each appliance starts at its preferred hour,
  pause gene 0. This is (i) the "before" case, (ii) the way we measure
  improvement pct, and (iii) one valid individual. baseline_already_valid()
  proves the baseline respects every window (run_checks re-checks it).

ga.py
- Encoding + operators + the driver.
  create_individual(): 90% of start genes uniformly inside the allowed
  range, 10% anywhere (honest diversity); pause genes random valid.
  tournament_select(): pick k=3 individuals, return the best (copy).
  crossover(): per-gene uniform swap with probability rate 0.9.
  mutate(): each start gene with probability 0.05 shifts by +-1..2 h
  (clamped, so no illegal gene can appear); pause genes get randomised.
  run_ga(): init population -> 40 generations of elitism (best 2 copied)
  + selection + crossover + mutation -> records best/avg fitness and
  eval count each generation; returns everything in one dictionary.

ma.py
- local_search(): deterministic hill climbing. One sweep visits EVERY
  appliance and keeps its best-improving start shift (+-1, +-2 h), plus
  - for interruptible appliances - every valid pause placement. Sweeps
  repeat until nothing improves (max 8 sweeps). It counts evaluations so
  runtime comparisons stay honest, and uses NO randomness (same seed ->
  same MA behaviour).
- run_ma(): imports the GA machinery unchanged and inserts ONE extra
  step: every 2nd generation the best 30% of the population are
  hill-climbed (polished in place), and at the end a MULTI-START final
  polish hill-climbs the 5 best distinct individuals and keeps the
  overall winner. Everything else (selection, crossover, mutation,
  elitism, history) is literally the same code -> clean comparison.

experiments.py
- run_paired_comparison(): for n_runs seeds (default 10: 42..51), run GA,
  MA, and also "GAb" - a plain GA given the SAME evaluation budget as the
  MA (matched generations) - always with the SAME seed per run, writing
  comparison_runs.csv (raw) and comparison_stats.csv (mean/std of every
  metric, plus convergence generation and runtime).
- run_parameter_study(): grid over population x mutation x crossover
  (default 30/60/100 x 0.01/0.05/0.10 x 0.6/0.9), param_seeds each, mean
  fitness per combination -> param_study.csv + heat map PNG.

visualize.py
- Five figure builders, all matplotlib-only, PNG files, Agg backend
  (works headless, never opens windows):
  load_profiles.png (hourly load baseline vs optimised + price line +
  solar area), gantt.png (BEFORE baseline / AFTER optimised schedule),
  convergence.png (GA vs MA best fitness per generation, same seed),
  cost_boxplot.png (per-run cost distribution of GA / GAb / MA),
  savings.png, param_study.png.

main.py
- The 10-line script you can run for a quick demo: baseline ->
  run_ga(verbose) -> run_ma(verbose) -> hard assertions (zero window
  violations, zero over-limit hours for BOTH algorithms, cost < baseline,
  energy conservation: every appliance uses exactly power_kW *
  duration_hours) -> 6-seed paired comparison for the box plot ->
  summary.csv + schedule_table.csv + histories + all plots.

run_checks.py
- The self-verification harness (11 checks, PASS/FAIL, exit code 0/1,
  writes results/check_report.txt):
   1. imports + requirements.txt contents
   2. appliance data structure (windows fit, >=2 narrow + wide windows,
      >=2 interruptible + continuous, chromosome length, baseline valid)
   3. price tiers + solar curve behave as configured
   4. hand-computed unit test with tiny appliances (exact on-hours,
      blocks, allowed ranges, energy conservation)
   5. a schedule violating a window MUST lose against every valid one
      (the W_VIOLATION penalty really dominates)
   6. GA result: valid, under the limit, cheaper than baseline, history
      never worsens (elitism), beats generation-0 average
   7. MA mean fitness <= GA mean over 6 paired seeds (prints per seed)
   8. reproducibility: same seed -> identical GA/MA results, different
      seed -> different result
   9. energy conservation for baseline/GA/MA (x_many exact equalities)
   10. runs main.py as a real subprocess, asserts exit 0 and that all 11
      result artefacts appear
   11. small parameter study writes its CSV + PNG

========================================================================
3. WORKED EXAMPLE (say this when asked "walk me through it")
========================================================================

Take a chromosome for the real appliances. Its first 13 genes are start
hours; the next 4 are pause genes (interruptible appliances are indexes
0, 2, 6, 8: EV, water heater, HVAC, pool pump). Example (this is the
MA showcase result, results/schedule_table.csv):

  starts: EV=0, ev_topup=11, water_heater=5, washing=22, dishwasher=16,
          dryer=22, hvac=12, vacuum=14, pool=9, iron=16, oven=16,
          rice=16, lawn=6
  pauses: EV=0, water_heater=0, hvac=0, pool=3
  -> chromosome = [0, 11, 5, 22, 16, 22, 12, 14, 9, 16, 16, 16, 6,
                   0, 0, 0, 3]

Step A - decode: for each appliance expand start/pause into on-hours:
  EV at 0, pause 0 -> hours 0,1,2,3,4 (7.4 kW each -> 37 kWh)
  pool at 9, pause 3 -> hours 9,10,11 (pause 12), 13,14,15
  (pause gene 3 = "run first 3 hours, off at hour start+3, run next 3").

Step B - hourly load: add 7.4 kW at hour 0 to BASE_LOAD_KW[0] and so on;
  same for water heater at 5..7, HVAC 12..17, ... A window violation
  would appear as an on-hour outside the window; here there is none.

Step C - net import = load - solar (solar ~4 kW around noon). Import is
  never 0 EGP - but negative (surplus) hours contribute 0 EGP, we do not
  export for a profit.

Step D - cost: multiply import by the hour price and sum:
  e.g. hour 0: import 7.95 kW x 1.15 = 9.14 EGP; EV overnight charging is
  THE cheap choice (7.4 kW at 1.15 EGP/kWh instead of 4.80 during the
  evening).

Step E - fitness term: peak = max net import = 7.95 kW (hour 0), below
  the 9 kW limit -> no limit term. violations 0 -> no 1000-terms.
  fitness = 199.98 + 3.0*7.95 = 223.83.

Step F - evolution: selection tests that schedule against random
  opponents (tournament of 3) and uniform crossover mixes its start
  hours with those of another good schedule; mutation perturbs a gene by +-1..2 h occasionally. The
  memetic step then exact-polishes the best individuals: e.g. discovering
  "moving vacuum hour 16->14 avoids the 17:00 peak tariff" by evaluating
  that single move.

The FINAL showcase comparison: the baseline (preferred hours) leaves
  the dishwasher+oven+dryer+HVAC stacked in the expensive evening and
  peaking at 11.35 kW, paying 279.62 EGP for the day - and it exceeds
  the 9 kW home limit once. Optimised: EV overnight, dryer and
  washing-machine in the cheap 22-24 zone, dishwasher/iron/oven/rice in
  the last pre-peak hours at 12-16, pool+top-up under the solar bell:
  cost 199.98 EGP, peak 7.95 kW - the "double-win" schedule both GA and
  MA find.

========================================================================
4. DESIGN RATIONALE (the WHY, not only the WHAT)
========================================================================

Encoding: start-hour integers + pause genes
- The assignment asks for "a vector where each gene represents the start
  time of an appliance" - ours does EXACTLY that, plus one extra
  documented gene per interruptible appliance (the pause hour). Fixed length (17 integers), every gene has a physical meaning,
  no repair operator ever needed because mutation clamps to the allowed
  range. Binary encoding ("binstring = appliance <OR> hour") would be
  far harder to interpret and to clamp legally.

Baseline = preferred hours
- Fair "before" case: the user has a preference for every appliance; the
  naive day (each appliance at its preferred start, pause 0) is feasible
  in window terms but wastes money/peak/power - this makes improvement
  percentages meaningful, and run_checks verifies it stays valid.

Penalty weights
- W_VIOLATION=1000: a whole day costs 200-300 EGP, so ONE violating
  appliance-hour is priced above ANY valid day; check 5 proves a
  violating schedule can never beat any valid schedule. That turns
  "hard constraint" into a soft one the GA can rank without a repair
  operator. Early generations may still inject violating "diversity"
  individuals (10%) deliberately - they die immediately, showing the GA
  respects comfort first, then saves money.
- W_PEAK=3.0 (tuned): peak shaping is a second objective next to the
  bill. A purely cost-greedy schedule piles everything into 22-24; with
  3.0 EGP/kW the flatter solution (down ~30% peak) wins, because each
  peak kW "costs" 3 EGP-scale - the classic DSM double objective.
- W_LIMIT=25 + GRID_LIMIT_KW=9.0: main-breaker rule - each hour above
  the limit is priced like tripping the breaker; it converts the cheap
  zone into a real packing constraint that forces coordinated moves.

GA parameters
- Population 30 x 40 generations: honest modest budget (~1.2k evals,
  0.05 s/run), so differences between algorithms are measurable instead
  of everything just saturating; the parameter study shows larger
  populations/mutation pay off mostly at this scale.
- Tournament k=3: small tournaments preserve diversity, big ones
  collapse it; k=3 is the usual sweet spot for pop 30-100.
- Elitism 2: the best 2 schedules are never lost. run_checks proves the
  best fitness NEVER worsens during a run.
- Mutation 0.05 (per gene): with 17 genes that is ~0.85 random changes
  per child - keeps exploring without shredding good schedules.
- 90% valid initial starts: start mostly inside windows (fast start),
  keep 10% random for honest diversity.

Memetic design
- Local search = assignment's requirement ("apply a hill climbing that
  tries shifting each appliance's start forward/backward") plus pause
  placement trials; best-improvement, deterministic, max 8 sweeps.
- Applied to the best 30% every 2nd generation: a pure GA generation in
  between keeps diversity; polishing the top 3rd converts their good
  genes to good SCHEDULES.
- Multi-start final polish over the 5 best DISTINCT individuals: one
  polish call can strand in a mediocre basin; 5 starts give several
  independent last chances (this is what makes the MA never lose to its
  GA twin on our seed families).
- Evaluations are counted everywhere (even inside the local search), so
  the extra "GAb" arm shows a plain GA with the MA's budget - honest
  fairness rather than advantage-stacking.

Statistics
- Pairs share seeds, so "MA vs GA twin" is a directly paired comparison
  (same initial population); mean/std over 10 runs; "first generation
  that reached the final quality" reported as conv_gen; violations and
  over-limit hours reported as 0 for all optimised runs.

========================================================================
5. GLOSSARY (definitional ammunition)
========================================================================

DSM (Demand Side Management): reshaping WHEN demand runs instead of
  adding generation; energy costs, peaks and renewables support improve.
TOU tariff: time-of-use pricing - different EGP/kWh price bands per
  hour of the day.
Peak import: maximum power drawn from the grid over the day (kW).
PAR: peak-to-average ratio of net import - flatness measure.
Self-consumption: share of produced solar that the home uses itself.
Baseline: non-optimised reference schedule (preferred hours).
Gene / chromosome / individual: one start hour / the full start+pause
  vector / one candidate schedule.
Population: set of individuals alive in one generation.
Generation: one evolution round: selection + crossover + mutation
  (+ memetic step) + elitism.
Fitness: scalar to minimise (cost + peak + penalties, EGP scale).
Selection (tournament): sample k individuals, keep the fittest.
Crossover (uniform): per-gene, child gene comes from either parent
  with probability 1/2 (rate 0.9 gates whether swap happens at all).
Mutation: small random change per gene with low probability (0.05),
  bounded to legal values (clamped to the allowed range).
Elitism: best elites copied unchanged into the next generation.
Evaluation: one call of the fitness function - the true cost unit.
Local search / hill climbing: only neighbouring schedules are tried;
  move if strictly better; stops at a local optimum.
Neighbour: swap = shift one start by +-1..2 h (or one pause re-place).
Memetic algorithm: population-based search + local search on promising
  individuals (also called "Lamarckian" - learned changes are inherited
  by the population).
Local optimum / basin: neighbourhood where no single move improves.
Paired seeds: comparing algorithms running with the SAME seed so both
  start from the same initial population.
Convergence generation: first generation whose best fitness already
  equals the final best (how FAST quality is found).
Exact/soft constraint: exact = never break (windows); soft = priced
  (peak), strongly priced (limit), 1000-priced (window as soft too).
Parameter study: grid experiment over algorithm parameters, measuring
  mean fitness per combination.

========================================================================
6. PROFESSOR Q&A (28 questions and model answers)
========================================================================

Q1 Why use a GA here instead of solving exactly?
A The search space is 17 genes with ~10-20 legal values each (~ order
   10^16-10^18 combinations when start hours vary per appliance); exact
   (MILP/DP) solvers exist but would be a different course topic. The GA
   gives high-quality schedules in 0.05-0.5 s on a laptop and shows the
   course concepts live. We also verify feasibility and conservation
   exactly in code.

Q2 Where exactly is the optimisation problem formalised?
A config.py (scenario data and weights), schedule_utils.py (exact
   schedule semantics), fitness.py (the scalar objective). REPORT
   section 2 lists the formal notation.

Q3 How large is the search space?
A Ev charger 0..4-5 starts, others ~7-12 each, pause genes up to
   duration values; order 10^16-10^18 distinct feasible-ish schedules -
   enumerating at 1M evals/s would take years. GA: ~1.2k-11k evals.

Q4 Why integer start-hour genes (and not binary)?
A It is the requested encoding; it makes every gene interpretable,
   mutation is a physically meaningful time shift, and window legality
   can be enforced by clamping - no decoding step, no repair.

Q5 What is the pause gene exactly?
A For the 4 interruptible appliances: p=0 -> continuous run; p>0 ->
   switch off at hour start+p (that hour moved later). One pause = 1
   flexible hour per appliance. run_checks check 4 unit-tests both
   cases against hand-computed hour lists.

Q6 Is the pause "free" cost-wise?
A No - the appliance still runs its full duration, just shifted; the
   pause GENE simply lets evolution split the run into two blocks to
   dodge one expensive hour (e.g. water heater skipping the 17:00 hour).

Q7 Walk me through your fitness function.
A evaluate(): load profile -> net import -> cost (import*price),
   peak = max import, fitness = cost + 3*peak + 1000*violations +
   25*over-limit. One number, documented weights (see also section 4),
   full report dict returned.

Q8 Why multiply peak by 3 instead of minimising cost only?
A With cost only, everything piles into the cheap late zone and the
   evening peak stays huge (bad for the grid, and the home
   breaker). 3.0 makes the flatter schedule win the trade-off: the
   optimised schedule costs ~28% less money AND has a ~30% lower peak -
   the double win rather than a cost-only win.

Q9 Why 1000 for a window violation / why not repair or reject?
A One violating hour prices above any valid day, so the population
   naturally seeds them out; repair/reject logic would hide infeasible
   individuals and complicate the code. Big penalties remain the
   standard soft-constraint method; check 5 PROVES the dominance.

Q10 Your initial population: 90% valid individuals. Isn't that cheating?
A No - a GA is allowed to inject domain knowledge; we still keep 10%
   random genes (including possibly violating ones) as honest diversity
   injection; the algorithms must learn to seed them out. The baseline
   itself is a valid schedule, proving feasibility exists.

Q11 Why is the main power limit (9 kW) in a home project?
A Real homes have a main breaker (3-phase ~10-12 kW in Egypt). Energy
   you cannot draw cannot be bought; training the optimiser against a
   hard-ish limit produces realistic schedules AND creates the packing
   tension that separates GA from MA behaviour.

Q12 What exactly does tournament selection do?
A For every child slot, draw 3 random competitors, take the best. Rate
   explainable O(k), no sorting, tunable selection pressure (k=3 = mild).

Q13 Why uniform crossover, not 1-point/2-point?
A Genes are INDEPENDENT start hours (no adjacency semantics on the
   genome), so block crossover has no natural cut points; uniform mixes
   per-appliance decisions. Rate 0.9 controls whether the swap happens.

Q14 Why mutation AFTER crossover on children?
A Crossover recombines existing material; mutation injects NEW hour
   values so the population can never fully lose parts of the hour
   range. Rate 0.05 per gene (~0.85 changed genes per child).

Q15 What does elitism give you?
A Monotone best-so-far - proven in run_checks (GA history never
   worsens). Without elites, a lucky population can regress.

Q16 Why is your MA's local search deterministic?
A Same seed -> identical MA behaviour; comparisons pair-run fairly and
   results reproduce exactly (check 8). Randomness lives ONLY in the GA
   operators.

Q17 Describe your local search precisely.
A Best-improvement hill climbing: one sweep tries, per appliance, all
   shifts within +-2 h (legal-clamped) keeping the best strictly-
   improving move, plus all pause placements for interruptible ones;
   sweeps repeat until no improvement or 8 sweeps. Evaluations counted.

Q18 Why hill climb the TOP 30% and not everyone?
A Hill climbing is expensive (hundreds of evals per individual); on the
   top third it converts the population's best genes into full good
   schedules while leaving the rest to explore. Also, every 2nd gen
   keeps a pure GA generation in between.

Q19 Why the multi-start final polish (5 starts)?
A A single polish call can end in a mediocre basin; polishing the 5
   best DISTINCT final individuals independently gives several cheap
   restarts (each from a different structure) - this removed the
   MA-loses-to-GA-twin cases in our tests.

Q20 How do you make comparisons FAIR?
A Paired seeds (identical initial populations), evaluation counters
   everywhere, three arms: GA (default), MA, and GAb = GA given the MA's
   evaluation budget. We report eval counts, runtime AND quality.

Q21 What are your headline numbers?
A Showcase: 279.62 -> 199.98 EGP (-28.5%), peak 11.35 -> 7.95 kW
   (-30.0%), PAR 2.61 -> 1.91, solar self-consumption 80.2% -> 100%,
   zero window violations, zero over-limit hours. Over 10 runs: GA mean
   cost 200.97 (std 1.97), MA 199.75 (std 0.37), GAb 199.98 (std 0).

Q22 If MA and plain-GA reach similar final quality, why bother with MA?
A At OUR default budget: MA's mean is better (199.75 vs 200.97) and its
   variance is 5x smaller (0.37 vs 1.97 EGP); it converges ~4x sooner
   (4.3 vs 15.9 generations). Reliability and speed-to-quality are worth
   as much as one EGP in an embedded scheduler. With the matched budget,
   plain GA catches up - which is exactly WHY we built the GAb arm.

Q23 Why do your results look "too stable"?
A The landscape is mostly additive and the scenario has a strong
   attractor (EV must sit at night, wash/dry inside 22-24). Several
   seeds reach the same good family of optima - that is stability, not
   leakage: seeds still differ by EGP cents and by the exact schedule
   structure (see comparison_runs.csv; GA cost varies 199.21-204.64).

Q24 Why do GA and MA sometimes find different schedules with the same cost?
A Flat near-optimum: swapping which appliances stack WHERE in the cheap
   zone costs the same total. The table in schedule_table.csv shows both
   arms' exact choices for the showcase.

Q25 What does the parameter study conclude?
A At pop 30 - our budget - mutation 0.01 under-explores (mean 229-234
   with crossover 0.9/0.6), and pop>=60 x mutation>=0.01 x crossover 0.9
   all reach the same best fitness (223.83). Best combinations are
   insensitive to the combination - larger population buys robustness;
   should stay >= 0.05 at pop 30.

Q26 What are the limitations?
A 24-hour horizon (no multi-day storage/thermal inertia), static TOU
   prices, Gaussian approximated solar (no cloud/weather model), no
   battery, no appliance state machine, one home (no aggregation), and
   the baseline assumes a simple "preferred hour" behaviour.

Q27 What would you do next with more time?
A Swap the solar array for real PVGIS/NREL data (one function), add a
   battery as a 14th appliance-like agent, benchmark against a MILP
   solver for optimality gaps, add RL (DQN) as a learning baseline, try
   adaptive penalty weights, and aggregate 50 homes for a grid view.

Q28 How can I (professor) verify everything myself?
A From the project folder: `python run_checks.py` (all 11 checks,
   exit 0, report in results/check_report.txt), `python main.py`
   (showcase + plots), `python experiments.py` (paired stats +
   parameter study, ~1-2 min). Fixed seed 42 makes every number
   reproducible.

========================================================================
7. LIMITATIONS AND FUTURE WORK (one paragraph you can memorise)
========================================================================

The model treats a single home with 13 appliances over one day, prices
as fixed TOU tiers, and solar as an idealised Gaussian curve; there is
no battery, no thermal dynamics, no multi-day coupling and no grid
interaction beyond import-priced energy. Future work: measured PV data
(direct array swap), battery/EV vehicle-to-home modelling, MILP or DP
exact benchmark (the scenario is small enough to bound the gap), RL
baselines, adaptive penalty/weight tuning, and multi-home aggregation
where the utility instead of the home owns the optimisation objective.
