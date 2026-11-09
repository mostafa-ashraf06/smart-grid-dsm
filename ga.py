"""A Genetic Algorithm written from scratch (no GA library).

Genetic Algorithm in one paragraph:
  A "population" of random schedules (individuals) is improved step by step.
  Every schedule is scored by the fitness function (lower = better).
  The best individuals are kept unchanged (elitism), while the rest of the
  next generation is built by SELECTING good parents, CROSSING OVER their
  genes (swapping start hours), and MUTATING a few genes randomly.
  Repeating this for many generations makes good scheduling patterns
  survive and spread, so the best schedule keeps improving.

Chromosome see config.py: N start-hour genes + one pause gene per
interruptible appliance.
"""

import random
import time

from config import (N_HOURS, GA_POP_SIZE, GA_GENERATIONS, GA_CROSSOVER_RATE,
                    GA_MUTATION_RATE, GA_TOURNAMENT_K, GA_ELITE_COUNT,
                    GA_INIT_VALID_FRACTION, GA_MUTATION_MAX_SHIFT)
from schedule_utils import allowed_start_range, chromosome_length
from fitness import evaluate


def create_individual(appliances, init_valid_fraction):
    """Create one random schedule (a list of genes).

    Inputs:
      appliances           : the appliance list
      init_valid_fraction  : probability that a start gene is chosen VALID.
    Returns a list of ints of length chromosome_length(appliances).

    Most genes (90%) start inside the allowed window so the population is
    mostly useful from generation 0; the remaining 10% are completely
    random for DIVERSITY - they violate windows and are killed by the
    violation penalty, but they let the GA explore even odd schedules.
    Pause genes are always chosen valid (0 = no pause, or 1..D-1).
    """
    individual = []
    for app in appliances:
        min_start, max_start = allowed_start_range(app)
        if random.random() < init_valid_fraction:
            start_hour = random.randint(min_start, max_start)   # valid gene
        else:
            start_hour = random.randint(0, N_HOURS - 1)         # wild gene
        individual.append(start_hour)
    for app in appliances:
        if app["interruptible"]:
            duration = app["duration_hours"]
            if random.random() < 0.5:
                individual.append(0)                    # no pause
            else:
                individual.append(random.randint(1, duration - 1))  # pause
    return individual


def tournament_select(population, fitnesses, tournament_k):
    """Tournament selection: look at k random individuals, return the best.

    Inputs:
      population : list of individuals
      fitnesses  : parallel list of fitness values (lower = better)
      tournament_k : how many individuals compete each round
    Returns a COPY of the winning individual (so we can mutate it safely).
    """
    winner_index = random.randrange(len(population))
    for _ in range(tournament_k - 1):
        challenger_index = random.randrange(len(population))
        if fitnesses[challenger_index] < fitnesses[winner_index]:
            winner_index = challenger_index
    return list(population[winner_index])


def crossover(parent1, parent2, crossover_rate):
    """Uniform crossover: each gene is swapped between the two parents
    with probability 0.5.

    Inputs: two parents (lists of ints) and the crossover probability.
    Returns two NEW child individuals.
    With probability (1 - rate) no crossover happens and copies of the
    parents are returned unchanged.
    """
    if random.random() >= crossover_rate:
        return list(parent1), list(parent2)
    child1 = []
    child2 = []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(gene1)   # take from parent 1
            child2.append(gene2)
        else:
            child1.append(gene2)   # take from parent 2 instead
            child2.append(gene1)
    return child1, child2


def mutate(individual, appliances, mutation_rate, max_shift):
    """Mutation: change some genes slightly to keep exploring new schedules.

    Inputs:
      individual    : list of int genes (it is copied, never edited in place)
      appliances    : appliance list
      mutation_rate : per-gENE probability that this gene changes
      max_shift     : biggest local shift allowed (we use +-1 .. +-max_shift)
    Returns the mutated copy.

    Start genes: with 50% chance jump to a completely random VALID hour
    (keeps reach across the whole window), otherwise shift by +-1..2 hours
    (a small local move, clamped into the valid range so the gene can
    never leave the window because of mutation).
    Pause genes: replaced by a random valid pause value (0..D-1).
    """
    mutant = list(individual)
    n_apps = len(appliances)
    # Collect the interruptible appliances in the same order used at
    # chromosome creation, so pause gene j belongs to extra_genes[j].
    interruptible_apps = []
    for app in appliances:
        if app["interruptible"]:
            interruptible_apps.append(app)
    # mutate the start genes
    for gene_index in range(n_apps):
        if random.random() < mutation_rate:
            min_start, max_start = allowed_start_range(appliances[gene_index])
            if random.random() < 0.5:
                mutant[gene_index] = random.randint(min_start, max_start)
            else:
                small_shifts = []
                for distance in range(1, max_shift + 1):
                    small_shifts.append(-distance)
                    small_shifts.append(distance)
                shift = random.choice(small_shifts)
                shifted = mutant[gene_index] + shift
                mutant[gene_index] = min(max(shifted, min_start), max_start)
    # mutate the pause genes
    for gene_index in range(n_apps, len(mutant)):
        if random.random() < mutation_rate:
            app = interruptible_apps[gene_index - n_apps]
            mutant[gene_index] = random.randint(0, app["duration_hours"] - 1)
    return mutant


def run_ga(appliances, price_array, solar_array,
           pop_size=GA_POP_SIZE,
           generations=GA_GENERATIONS,
           crossover_rate=GA_CROSSOVER_RATE,
           mutation_rate=GA_MUTATION_RATE,
           tournament_k=GA_TOURNAMENT_K,
           elite_count=GA_ELITE_COUNT,
           init_valid_fraction=GA_INIT_VALID_FRACTION,
           mutation_max_shift=GA_MUTATION_MAX_SHIFT,
           seed=42,
           verbose=False,
           verbose_every=20):
    """Run the Genetic Algorithm from scratch and return the best schedule.

    Inputs:
      appliances, price_array, solar_array : the problem data
      pop_size .. mutation_max_shift       : GA control knobs (see config.py)
      seed          : random seed; same seed -> same result (reproducible)
      verbose       : print a short progress line every verbose_every gens
    Returns a dictionary:
      algorithm, best_schedule, best_fitness, best_metrics (full evaluate()
      dict), history (list of per-generation dicts), evaluations (number of
      fitness function calls), runtime_s, pop_size, generations.

    The main loop is the classic GA cycle:
      evaluate -> selection -> crossover -> mutation -> elitism -> repeat.
    """
    start_time = time.perf_counter()
    random.seed(seed)   # reproducibility: same seed -> same evolution

    # ---- generation 0: one random population ------------------------------
    population = []
    for _ in range(pop_size):
        population.append(create_individual(appliances, init_valid_fraction))
    fitnesses = []
    for individual in population:
        fitnesses.append(evaluate(individual, appliances,
                                  price_array, solar_array)["fitness"])
    evaluations = pop_size

    history = []

    # ---- main evolution loop ----------------------------------------------
    for generation in range(generations):
        # sort by fitness (lower is better) - only used for elitism
        order = sorted(range(pop_size), key=lambda i: fitnesses[i])

        # 1) elitism: copy the best few individuals unchanged (with their
        #    known fitness, so they are NOT re-evaluated)
        new_population = [list(population[i]) for i in order[:elite_count]]
        new_fitnesses = [fitnesses[i] for i in order[:elite_count]]

        # 2) fill the rest with children: select -> crossover -> mutate
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

        # 3) bookkeeping for the convergence plots
        best_index = 0
        for i in range(pop_size):
            if fitnesses[i] < fitnesses[best_index]:
                best_index = i
        best_metrics = evaluate(population[best_index], appliances,
                                price_array, solar_array)
        evaluations += 1   # this re-evaluation is counted, to be fair
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

    return {
        "algorithm": "GA",
        "best_schedule": population[best_index],
        "best_fitness": fitnesses[best_index],
        "best_metrics": best_metrics,
        "history": history,
        "evaluations": evaluations,
        "runtime_s": time.perf_counter() - start_time,
        "pop_size": pop_size,
        "generations": generations,
    }
