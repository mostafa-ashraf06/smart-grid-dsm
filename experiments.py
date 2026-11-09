"""Experiments: statistics over many independent runs + a GA parameter study.

Two things happen here:
  1. run_paired_comparison: GA and MA are run n times with the same seeds
     (run i of MA uses the same seed as run i of GA, so both start from the
     SAME initial population). For every run we record cost, peak, PAR,
     solar self-consumption, fitness, runtime, fitness evaluations and the
     generation at which the run got within 2% of its final best fitness
     (convergence speed). Averages +- std across runs give stability.
  2. run_parameter_study: a grid of GA population sizes, mutation rates and
     crossover rates, each tested with several seeds, to find which
     combination gives the best (lowest) mean final fitness.
"""

import argparse
import csv
import os
import statistics

from config import (SEED, RESULTS_DIR, GA_GENERATIONS, ensure_results_dir)
from appliances import get_appliances
from pricing import get_price_array, get_solar_array
from baseline import get_baseline_schedule
from fitness import evaluate
from ga import run_ga
from ma import run_ma
from visualize import plot_param_study


def write_csv(path, header, rows):
    """Save a list of dictionaries as a CSV file.

    Inputs: path (output file name), header (list of column names),
    rows (list of dicts matching the header keys).
    No return value; the file is written into path.
    """
    with open(path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def round_row(row, digits=4):
    """Return a copy of a dict with every float rounded (nicer CSVs).

    Inputs: row (dict), digits (decimal places). Returns the rounded dict.
    """
    clean = {}
    for key, value in row.items():
        if isinstance(value, float):
            clean[key] = round(value, digits)
        else:
            clean[key] = value
    return clean


def system_metrics(schedule, appliances, price_array, solar_array):
    """Summarise one schedule with the reporting metrics of the project.

    Inputs: a chromosome + problem data.
    Returns a dict with cost, peak_import_kw, par, solar_self_consumption_pct
    and violations. This is what goes into summary tables.
    """
    metrics = evaluate(schedule, appliances, price_array, solar_array)
    return {
        "cost": metrics["cost"],
        "peak_import_kw": metrics["peak_import_kw"],
        "par": metrics["par"],
        "solar_self_consumption_pct": metrics["solar_self_consumption_pct"],
        "violations": metrics["violations"],
        "limit_violations": metrics["limit_violations"],
    }


def pct_reduction(before, after):
    """Percent improvement going from `before` to `after` (positive = better).

    Inputs: two numbers. Returns a float: 100 * (before-after)/before.
    If before is ~0 the improvement is defined as 0 to avoid dividing by 0.
    """
    if before <= 1e-9:
        return 0.0
    return 100.0 * (before - after) / before


def savings_table(baseline_metrics, ga_metrics, ma_metrics):
    """Compare optimised results against the naive baseline in percent.

    Inputs: the three metric dictionaries from system_metrics/evaluate.
    Returns a dict with one sub-dict per algorithm containing
    cost_saving_pct, peak_reduction_pct, par_reduction_pct and the absolute
    solar_self_consumption_pct for reference.
    """
    def reductions(metrics):
        return {
            "cost_saving_pct": pct_reduction(baseline_metrics["cost"],
                                             metrics["cost"]),
            "peak_reduction_pct": pct_reduction(baseline_metrics["peak_import_kw"],
                                                metrics["peak_import_kw"]),
            "par_reduction_pct": pct_reduction(baseline_metrics["par"],
                                               metrics["par"]),
            "solar_self_consumption_pct": metrics["solar_self_consumption_pct"],
        }
    return {"ga": reductions(ga_metrics), "ma": reductions(ma_metrics)}


def convergence_generation(history, final_best, tolerance=0.02):
    """How many generations until a run reached 'near-best' quality?

    Inputs: history (list of per-generation dicts), final_best (the run's
    final best fitness), tolerance (relative gap, default 2%).
    Returns the (1-based) number of generations of the FIRST generation
    whose best fitness is within tolerance of the final best fitness.
    """
    threshold = final_best * (1.0 + tolerance)
    for row in history:
        if row["best_fitness"] <= threshold:
            return row["generation"] + 1
    return len(history)


def stats_of(values):
    """Mean, std, best, worst of a list of numbers (stability summary).

    Input: values (list of floats). Returns a dict.
    """
    if len(values) == 1:
        std = 0.0
    else:
        std = statistics.stdev(values)
    return {"mean": statistics.mean(values), "std": std,
            "best": min(values), "worst": max(values)}


def run_paired_comparison(n_runs=10, seed0=SEED, tag=""):
    """Run GA and MA n times with identical seeds and store the statistics.

    Inputs: n_runs (how many independent runs), seed0 (first seed; run i
    uses seed0 + i), tag (suffix for the CSV file names).
    Returns (rows, stats) where each `rows` item is one run's numbers for
    both algorithms, and `stats` contains mean/std/best/worst per metric.
    Also writes results/comparison_runs<tag>.csv and
    results/comparison_stats<tag>.csv.
    """
    appliances = get_appliances()
    price_array = get_price_array()
    solar_array = get_solar_array()
    baseline_metrics = evaluate(get_baseline_schedule(appliances),
                                appliances, price_array, solar_array)

    rows = []
    print("Running %d paired GA/MA runs (seed %d .. %d)..."
          % (n_runs, seed0, seed0 + n_runs - 1))
    for run_index in range(n_runs):
        seed = seed0 + run_index
        ga_result = run_ga(appliances, price_array, solar_array, seed=seed)
        ma_result = run_ma(appliances, price_array, solar_array, seed=seed)
        # THIRD arm for a fair comparison: give a plain GA approximately the
        # SAME number of fitness evaluations the MA needed (the spec allows
        # "same evaluation budget" - so we report both small and matched GA).
        ga_pop = ga_result["pop_size"]
        matched_generations = max(1, (ma_result["evaluations"] - ga_pop) // ga_pop)
        ga_big = run_ga(appliances, price_array, solar_array,
                        generations=matched_generations, seed=seed)
        row = {
            "run": run_index,
            "seed": seed,
            "ga_cost": ga_result["best_metrics"]["cost"],
            "ga_peak_kw": ga_result["best_metrics"]["peak_import_kw"],
            "ga_par": ga_result["best_metrics"]["par"],
            "ga_solar_pct": ga_result["best_metrics"]["solar_self_consumption_pct"],
            "ga_violations": ga_result["best_metrics"]["violations"],
            "ga_over_limit": ga_result["best_metrics"]["limit_violations"],
            "ga_fitness": ga_result["best_fitness"],
            "ga_evals": ga_result["evaluations"],
            "ga_runtime_s": ga_result["runtime_s"],
            "ga_conv_gen": convergence_generation(ga_result["history"],
                                                  ga_result["best_fitness"]),
            "gab_cost": ga_big["best_metrics"]["cost"],
            "gab_peak_kw": ga_big["best_metrics"]["peak_import_kw"],
            "gab_par": ga_big["best_metrics"]["par"],
            "gab_solar_pct": ga_big["best_metrics"]["solar_self_consumption_pct"],
            "gab_violations": ga_big["best_metrics"]["violations"],
            "gab_over_limit": ga_big["best_metrics"]["limit_violations"],
            "gab_fitness": ga_big["best_fitness"],
            "gab_evals": ga_big["evaluations"],
            "gab_runtime_s": ga_big["runtime_s"],
            "gab_conv_gen": convergence_generation(ga_big["history"],
                                                   ga_big["best_fitness"]),
            "ma_cost": ma_result["best_metrics"]["cost"],
            "ma_peak_kw": ma_result["best_metrics"]["peak_import_kw"],
            "ma_par": ma_result["best_metrics"]["par"],
            "ma_solar_pct": ma_result["best_metrics"]["solar_self_consumption_pct"],
            "ma_violations": ma_result["best_metrics"]["violations"],
            "ma_over_limit": ma_result["best_metrics"]["limit_violations"],
            "ma_fitness": ma_result["best_fitness"],
            "ma_evals": ma_result["evaluations"],
            "ma_runtime_s": ma_result["runtime_s"],
            "ma_conv_gen": convergence_generation(ma_result["history"],
                                                  ma_result["best_fitness"]),
            "ga_cost_saving_pct": pct_reduction(baseline_metrics["cost"],
                                                ga_result["best_metrics"]["cost"]),
            "ga_peak_reduction_pct": pct_reduction(baseline_metrics["peak_import_kw"],
                                                   ga_result["best_metrics"]["peak_import_kw"]),
            "ga_par_reduction_pct": pct_reduction(baseline_metrics["par"],
                                                  ga_result["best_metrics"]["par"]),
            "gab_cost_saving_pct": pct_reduction(baseline_metrics["cost"],
                                                 ga_big["best_metrics"]["cost"]),
            "gab_peak_reduction_pct": pct_reduction(baseline_metrics["peak_import_kw"],
                                                    ga_big["best_metrics"]["peak_import_kw"]),
            "gab_par_reduction_pct": pct_reduction(baseline_metrics["par"],
                                                   ga_big["best_metrics"]["par"]),
            "ma_cost_saving_pct": pct_reduction(baseline_metrics["cost"],
                                                ma_result["best_metrics"]["cost"]),
            "ma_peak_reduction_pct": pct_reduction(baseline_metrics["peak_import_kw"],
                                                   ma_result["best_metrics"]["peak_import_kw"]),
            "ma_par_reduction_pct": pct_reduction(baseline_metrics["par"],
                                                  ma_result["best_metrics"]["par"]),
        }
        rows.append(row)
        print("  run %d (seed %d): GA cost %7.2f EGP (%5.1f%% saved) | "
              "GA-matched %7.2f (%5.1f%%) | MA cost %7.2f EGP (%5.1f%% saved)"
              % (run_index, seed, row["ga_cost"], row["ga_cost_saving_pct"],
                 row["gab_cost"], row["gab_cost_saving_pct"],
                 row["ma_cost"], row["ma_cost_saving_pct"]))

    # save raw per-run numbers
    write_csv(os.path.join(RESULTS_DIR, "comparison_runs%s.csv" % tag),
              list(rows[0].keys()), [round_row(row) for row in rows])

    # aggregate mean/std/best/worst per metric, one column per algorithm
    metric_columns = ["fitness", "cost", "peak_kw", "par", "solar_pct",
                      "violations", "over_limit", "evals", "runtime_s",
                      "conv_gen", "cost_saving_pct", "peak_reduction_pct",
                      "par_reduction_pct"]
    algorithm_prefixes = ["ga", "gab", "ma"]
    stats = []
    for metric in metric_columns:
        row = {"metric": metric}
        for prefix in algorithm_prefixes:
            values = [row_data[prefix + "_" + metric] for row_data in rows]
            stats_of_values = stats_of(values)
            row[prefix + "_mean"] = stats_of_values["mean"]
            row[prefix + "_std"] = stats_of_values["std"]
            row[prefix + "_best"] = stats_of_values["best"]
            row[prefix + "_worst"] = stats_of_values["worst"]
        stats.append(row)
    write_csv(os.path.join(RESULTS_DIR, "comparison_stats%s.csv" % tag),
              list(stats[0].keys()), [round_row(row) for row in stats])
    return rows, stats


def run_parameter_study(param_seeds=3,
                        pop_sizes=None,
                        mutation_rates=None,
                        crossover_rates=None,
                        tag=""):
    """Grid search over GA control parameters, recorded to CSV + heatmap.

    Inputs:
      param_seeds     : how many seeds per combination (1 seed = fast check)
      pop_sizes, mutation_rates, crossover_rates : the parameter grids;
        None means the default study grid (3 populations, 3 mutation rates,
        2 crossover rates = 18 combinations).
      tag             : suffix for output file names.
    Returns the list of result rows (one per parameter combination).
    """
    if pop_sizes is None:
        pop_sizes = [30, 60, 100]
    if mutation_rates is None:
        mutation_rates = [0.01, 0.05, 0.10]
    if crossover_rates is None:
        crossover_rates = [0.6, 0.9]

    appliances = get_appliances()
    price_array = get_price_array()
    solar_array = get_solar_array()

    rows = []
    print("GA parameter study: %d combinations x %d seeds..."
          % (len(pop_sizes) * len(mutation_rates) * len(crossover_rates),
             param_seeds))
    for pop_size in pop_sizes:
        for mutation_rate in mutation_rates:
            for crossover_rate in crossover_rates:
                best_fitnesses = []
                costs = []
                for seed_offset in range(param_seeds):
                    result = run_ga(appliances, price_array, solar_array,
                                    pop_size=pop_size,
                                    mutation_rate=mutation_rate,
                                    crossover_rate=crossover_rate,
                                    seed=SEED + 1000 + seed_offset)
                    best_fitnesses.append(result["best_fitness"])
                    costs.append(result["best_metrics"]["cost"])
                fitness_stats = stats_of(best_fitnesses)
                row = {
                    "pop_size": pop_size,
                    "mutation_rate": mutation_rate,
                    "crossover_rate": crossover_rate,
                    "n_seeds": param_seeds,
                    "mean_fitness": fitness_stats["mean"],
                    "std_fitness": fitness_stats["std"],
                    "best_fitness": fitness_stats["best"],
                    "worst_fitness": fitness_stats["worst"],
                    "mean_cost": statistics.mean(costs),
                }
                rows.append(row)
                print("  pop=%3d mut=%.2f cross=%.2f -> mean fitness %8.2f "
                      "(best %8.2f)"
                      % (pop_size, mutation_rate, crossover_rate,
                         row["mean_fitness"], row["best_fitness"]))

    csv_path = os.path.join(RESULTS_DIR, "param_study%s.csv" % tag)
    write_csv(csv_path, list(rows[0].keys()), [round_row(row) for row in rows])

    # rank the combinations and print the best ones
    ranked = sorted(rows, key=lambda row: row["mean_fitness"])
    print("Best parameter combinations (lowest mean fitness first):")
    for row in ranked[:5]:
        print("  pop=%3d mut=%.2f cross=%.2f -> %8.2f"
              % (row["pop_size"], row["mutation_rate"],
                 row["crossover_rate"], row["mean_fitness"]))

    plot_param_study(rows, tag)
    return rows


def main(n_runs=10, do_param_study=True, param_seeds=3):
    """Run the full experiment suite (used by the command line).

    Inputs: n_runs for the GA vs MA comparison, do_param_study on/off,
    param_seeds per combination. Writes all CSVs/plots into results/.
    """
    ensure_results_dir()
    appliances = get_appliances()
    price_array = get_price_array()
    solar_array = get_solar_array()
    baseline_metrics = system_metrics(get_baseline_schedule(appliances),
                                      appliances, price_array, solar_array)
    print("Baseline (no optimisation): cost %.2f EGP | peak %.2f kW | "
          "PAR %.2f | solar used %.1f%%"
          % (baseline_metrics["cost"], baseline_metrics["peak_import_kw"],
             baseline_metrics["par"],
             baseline_metrics["solar_self_consumption_pct"]))

    rows, stats = run_paired_comparison(n_runs=n_runs)

    print("\nMean results over %d runs:" % n_runs)
    print("  (GA = default budget, GAb = GA with the MA evaluation budget, MA = memetic)")
    for stat in stats:
        print("  %-18s GA %10.2f (std %8.2f) | GAb %10.2f (std %8.2f) | MA %10.2f (std %8.2f)"
              % (stat["metric"], stat["ga_mean"], stat["ga_std"],
                 stat["gab_mean"], stat["gab_std"],
                 stat["ma_mean"], stat["ma_std"]))

    if do_param_study:
        run_parameter_study(param_seeds=param_seeds)
    print("Experiment files are in: %s" % RESULTS_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart grid DSM experiments")
    parser.add_argument("--runs", type=int, default=10,
                        help="number of GA/MA runs in the comparison")
    parser.add_argument("--no-param-study", action="store_true",
                        help="skip the GA parameter study")
    parser.add_argument("--param-seeds", type=int, default=3,
                        help="seeds per parameter combination")
    arguments = parser.parse_args()
    main(n_runs=arguments.runs,
         do_param_study=not arguments.no_param_study,
         param_seeds=arguments.param_seeds)
