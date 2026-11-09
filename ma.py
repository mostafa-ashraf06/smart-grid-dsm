"""A Memetic Algorithm (MA) = Genetic Algorithm + LOCAL SEARCH.

Idea: a GA is good at exploring widely (global search), but it is slow at
polishing the small details. A local search (hill climbing) is the opposite:
it looks only at small changes around the current schedule, but it is very
precise. Combining them - a "memetic algorithm" - gives both:

  GA part : recombine and mutate whole schedules       (exploration)
  MA part : hill-climb the most promising schedules    (exploitation)

The local search here tries, for EVERY appliance, shifting its start hour
by -max_shift..-1 and +1..+max_shift hours; it keeps the best improving
shift and repeats these sweeps until nothing improves anymore (or a max
number of sweeps is reached). Consequence: appliances end up aligned with
cheap price hours and the solar peak - exactly the big wins a plain GA
often misses by 1-2 hours.

Important: the local search uses NO randomness, so for the same seed the
MA consumes the same random stream as the GA. This makes GA-vs-MA runs
directly comparable and MA can never end up worse early on.
"""

import random
import time

from config import (GA_POP_SIZE, GA_GENERATIONS, GA_CROSSOVER_RATE,
                    GA_MUTATION_RATE, GA_TOURNAMENT_K, GA_ELITE_COUNT,
                    GA_INIT_VALID_FRACTION, GA_MUTATION_MAX_SHIFT,
                    LS_FRACTION, LS_MAX_SHIFT, LS_MAX_ROUNDS, LS_EVERY,
                    LS_FINAL_TOP)
from schedule_utils import allowed_start_range
from fitness import evaluate
from ga import (create_individual, tournament_select,
                crossover, mutate)


def local_search(schedule, appliances, price_array, solar_array,
                 max_shift=LS_MAX_SHIFT, max_rounds=LS_MAX_ROUNDS):
    """Hill climbing on the genes of an individual schedule.

    Inputs:
      schedule    : the chromosome to improve (copied, input untouched)
      appliances  : appliance list
      price_array : 24 EGP/kWh values
      solar_array : 24 kW values
      max_shift   : try shifts of -max_shift .. -1 and +1 .. +max_shift
      max_rounds  : max number of full sweeps over all appliances
    Returns (improved_schedule, its_fitness, evaluations_used).

    One "round" visits every appliance once. For each appliance all its
    candidate shifts are evaluated and the BEST improving shift is kept
    (best-improvement hill climbing). For interruptible appliances the
    pause gene is also tried (every valid pause value), because pausing
    one different hour can save real money. Rounds repeat until a sweep
    makes no improvement. Evaluations are counted so GA vs MA comparisons
    stay fair.
    """
    current = list(schedule)
    current_fitness = evaluate(current, appliances, price_array, solar_array)["fitness"]
    evaluations = 1
    for sweep in range(max_rounds):
        made_improvement = False
        # ---- move start hours of all appliances ------------------------
        for gene_index, app in enumerate(appliances):
            min_start, max_start = allowed_start_range(app)
            # collect the candidate hours we could move this start to
            candidate_starts = []
            for distance in range(1, max_shift + 1):
                left = current[gene_index] - distance
                right = current[gene_index] + distance
                if left >= min_start and left != current[gene_index]:
                    candidate_starts.append(left)
                if right <= max_start:
                    candidate_starts.append(right)
            if not candidate_starts:
                continue
            best_start = current[gene_index]
            best_fitness_here = current_fitness
            for candidate in candidate_starts:
                trial = list(current)
                trial[gene_index] = candidate
                trial_fitness = evaluate(trial, appliances, price_array, solar_array)["fitness"]
                evaluations += 1
                if trial_fitness < best_fitness_here:
                    best_start = candidate
                    best_fitness_here = trial_fitness
            if best_start != current[gene_index]:
                current[gene_index] = best_start
                current_fitness = best_fitness_here
                made_improvement = True   # keep sweeping until no gain
        # ---- re-place the pause hour of interruptible appliances --------
        extra_gene_index = len(appliances)
        for gene_index, app in enumerate(appliances):
            if not app["interruptible"]:
                continue
            best_pause = current[extra_gene_index]
            best_fitness_here = current_fitness
            for pause_value in range(app["duration_hours"]):
                if pause_value == current[extra_gene_index]:
                    continue
                trial = list(current)
                trial[extra_gene_index] = pause_value
                trial_fitness = evaluate(trial, appliances, price_array, solar_array)["fitness"]
                evaluations += 1
                if trial_fitness < best_fitness_here:
                    best_pause = pause_value
                    best_fitness_here = trial_fitness
            if best_pause != current[extra_gene_index]:
                current[extra_gene_index] = best_pause
                current_fitness = best_fitness_here
                made_improvement = True
            extra_gene_index += 1
        if not made_improvement:
            break   # local optimum reached
    return current, current_fitness, evaluations


def run_ma(appliances, price_array, solar_array,
           pop_size=GA_POP_SIZE,
           generations=GA_GENERATIONS,
           crossover_rate=GA_CROSSOVER_RATE,
           mutation_rate=GA_MUTATION_RATE,
           tournament_k=GA_TOURNAMENT_K,
           elite_count=GA_ELITE_COUNT,
           init_valid_fraction=GA_INIT_VALID_FRACTION,
           mutation_max_shift=GA_MUTATION_MAX_SHIFT,
            ls_fraction=LS_FRACTION,
            ls_max_shift=LS_MAX_SHIFT,
            ls_max_rounds=LS_MAX_ROUNDS,
            ls_every=LS_EVERY,
            ls_final_top=LS_FINAL_TOP,
            seed=42,
           verbose=False,
           verbose_every=20):
    """Run the Memetic Algorithm and return the best schedule.

    Same inputs and same return dictionary as run_ga (see ga.py), with four
    extra local-search knobs:
    ls_fraction   : fraction of the BEST individuals that get hill-climbed
    ls_max_shift  : hill climbing shift range in hours
    ls_max_rounds : max sweeps per hill climbing call
    ls_every      : run the hill climbing every `ls_every` generations
    ls_final_top  : final multi-start polish starts from this many best
                    DISTINCT individuals (not just the single best)

    The GA machinery is imported unchanged from ga.py, so the ONLY
    difference to run_ga is the local search step - a clean comparison.
    """
    start_time = time.perf_counter()
    random.seed(seed)   # same seed as a GA run -> same starting population

    population = []
    for _ in range(pop_size):
        population.append(create_individual(appliances, init_valid_fraction))
    fitnesses = []
    for individual in population:
        fitnesses.append(evaluate(individual, appliances,
                                  price_array, solar_array)["fitness"])
    evaluations = pop_size

    history = []

    for generation in range(generations):
        order = sorted(range(pop_size), key=lambda i: fitnesses[i])
        new_population = [list(population[i]) for i in order[:elite_count]]
        new_fitnesses = [fitnesses[i] for i in order[:elite_count]]

        while len(new_population) < pop_size:
            parent1 = tournament_select(population, fitnesses, tournament_k)
            parent2 = tournament_select(population, fitnesses, tournament_k)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            child1 = mutate(child1, appliances, mutation_rate, mutation_max_shift)
            child2 = mutate(child2, appliances, mutation_rate, mutation_max_shift)
            for child in (child1, child2):
                if len(new_population) < pop_size:
                    new_population.append(child)
                    new_fitnesses.append(
                        evaluate(child, appliances, price_array, solar_array)["fitness"])
                    evaluations += 1

        population = new_population
        fitnesses = new_fitnesses

        # -------------------------------------------------------------
        # THE memetic step: hill-climb the best ~20% of the population.
        # Deterministic (no randomness) so fitness comparisons are exact.
        # -------------------------------------------------------------
        if generation % ls_every == 0:
            order = sorted(range(pop_size), key=lambda i: fitnesses[i])
            for i in order[:max(1, int(round(ls_fraction * pop_size)))]:
                polished, polished_fitness, ls_evals = local_search(
                    population[i], appliances, price_array, solar_array,
                    ls_max_shift, ls_max_rounds)
                evaluations += ls_evals
                if polished_fitness < fitnesses[i]:
                    population[i] = polished
                    fitnesses[i] = polished_fitness

        best_index = 0
        for i in range(pop_size):
            if fitnesses[i] < fitnesses[best_index]:
                best_index = i
        best_metrics = evaluate(population[best_index], appliances,
                                price_array, solar_array)
        evaluations += 1
        average_fitness = sum(fitnesses) / float(pop_size)
        history.append({
            "generation": generation,
            "best_fitness": fitnesses[best_index],
            "avg_fitness": average_fitness,
            "best_cost": best_metrics["cost"],
            "best_peak": best_metrics["peak_import_kw"],
            "evaluations": evaluations,
        })

        if verbose and (generation + 1) % verbose_every == 0:
            print("  gen %3d | best fitness %8.2f | avg %8.2f "
                  "| cost %6.2f EGP | peak %4.1f kW"
                  % (generation + 1, fitnesses[best_index], average_fitness,
                     best_metrics["cost"], best_metrics["peak_import_kw"]))

    # Final polish (mentioned in the assignment): a MULTI-START hill
    # climbing. Take the ls_final_top best DISTINCT individuals and
    # hill-climb each of them independently (duplicates-removed so we do
    # not spend evaluations on copies of the same elite); keep the overall
    # best of these starts. Several starts give the MA several independent
    # "last chances" to escape a mediocre basin before reporting.
    best_index = 0
    for i in range(pop_size):
        if fitnesses[i] < fitnesses[best_index]:
            best_index = i
    order = sorted(range(pop_size), key=lambda i: fitnesses[i])
    final_top = max(1, int(round(ls_final_top)))
    polished_schedule = population[best_index]
    polished_fitness = fitnesses[best_index]
    seen_chromosomes = {tuple(polished_schedule)}
    for i in order:
        if len(seen_chromosomes) >= final_top:
            break
        if tuple(population[i]) in seen_chromosomes:
            continue
        seen_chromosomes.add(tuple(population[i]))
        candidate, candidate_fitness, ls_evals = local_search(
            population[i], appliances, price_array, solar_array,
            ls_max_shift, ls_max_rounds)
        evaluations += ls_evals
        if candidate_fitness < polished_fitness:
            polished_schedule = candidate
            polished_fitness = candidate_fitness
    if polished_fitness < fitnesses[best_index]:
        population[best_index] = polished_schedule
        fitnesses[best_index] = polished_fitness
        best_metrics = evaluate(population[best_index], appliances,
                                price_array, solar_array)
        evaluations += 1
        history.append({
            "generation": generations,   # one row after the last generation
            "best_fitness": fitnesses[best_index],
            "avg_fitness": average_fitness,
            "best_cost": best_metrics["cost"],
            "best_peak": best_metrics["peak_import_kw"],
            "evaluations": evaluations,
        })

    return {
        "algorithm": "MA",
        "best_schedule": population[best_index],
        "best_fitness": fitnesses[best_index],
        "best_metrics": best_metrics,
        "history": history,
        "evaluations": evaluations,
        "runtime_s": time.perf_counter() - start_time,
        "pop_size": pop_size,
        "generations": generations,
    }
