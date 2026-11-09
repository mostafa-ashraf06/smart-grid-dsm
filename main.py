"""main.py - run the whole project pipeline end to end.

Steps:
  1. show the scenario (appliances, prices, solar)
  2. build the naive baseline schedule (the "BEFORE" case)
  3. run the Genetic Algorithm
  4. run the Memetic Algorithm (GA + local search)
  5. run a small paired GA/MA comparison for the statistics and box plot
  6. verify the results (no window violations, really cheaper, energy
     conserved) - the project would refuse to print wrong numbers
  7. write CSV tables and all plots into results/

Run it from INSIDE this folder:   python main.py
"""

import csv
import os

from config import (SEED, RESULTS_DIR, ensure_results_dir)
from appliances import get_appliances, describe_appliances
from pricing import get_price_array, get_solar_array, describe_pricing
from baseline import get_baseline_schedule
from fitness import evaluate
from ga import run_ga
from ma import run_ma
from schedule_utils import (interpret_schedule, schedule_is_valid,
                            total_appliance_energy, chromosome_length)
from experiments import (system_metrics, savings_table, run_paired_comparison,
                         write_csv)
from visualize import make_all_plots


def print_case_line(name, metrics, extra=""):
    """One tidy line of numbers for a schedule (cost/peak/PAR/solar).

    Inputs: name (label), metrics (evaluate() dict), optional extra text.
    """
    print("  %-9s cost %8.2f EGP | peak import %5.2f kW | PAR %5.2f | "
          "solar used %5.1f%% %s"
          % (name, metrics["cost"], metrics["peak_import_kw"],
             metrics["par"], metrics["solar_self_consumption_pct"], extra))


def write_schedule_table(baseline_schedule, ga_result, ma_result,
                         appliances, path):
    """Save results/schedule_table.csv: when each appliance runs.

    Inputs: the three schedules, appliance list, output path.
    One row per appliance with start/pause genes and the on-hours.
    """
    rows = []
    parsed_baseline = interpret_schedule(baseline_schedule, appliances)
    parsed_ga = interpret_schedule(ga_result["best_schedule"], appliances)
    parsed_ma = interpret_schedule(ma_result["best_schedule"], appliances)
    for i, app in enumerate(appliances):
        base_start, base_pause = parsed_baseline[i]
        ga_start, ga_pause = parsed_ga[i]
        ma_start, ma_pause = parsed_ma[i]
        rows.append({
            "appliance": app["name"],
            "power_kw": app["power_kw"],
            "duration_hours": app["duration_hours"],
            "window": "%d-%d" % (app["window_start"], app["window_end"]),
            "interruptible": app["interruptible"],
            "baseline_start": base_start,
            "ga_start": ga_start,
            "ga_pause": ga_pause,
            "ma_start": ma_start,
            "ma_pause": ma_pause,
        })
    write_csv(path, list(rows[0].keys()), rows)


def write_history_csv(history, path):
    """Save one per-generation history dict list as results/xx_history.csv."""
    write_csv(path, list(history[0].keys()), history)


def check_energy_conservation(schedule, appliances, label):
    """Verify every scheduled appliance consumed exactly power*duration.

    Inputs: a chromosome, the appliance list, a label for messages.
    Prints one OK line; raises AssertionError on any mismatch.
    """
    energy = total_appliance_energy(schedule, appliances)
    for app in appliances:
        expected = app["power_kw"] * app["duration_hours"]
        got = energy[app["name"]]
        assert abs(got - expected) < 1e-6, (
            "%s: %s drew %.3f kWh but must draw %.3f kWh"
            % (label, app["name"], got, expected))
    print("  energy conservation OK for %s (every appliance = power * duration)"
          % label)


def main():
    """The full pipeline; also the entry point for run_checks.py."""
    ensure_results_dir()
    print("=" * 70)
    print("SMART GRID ENERGY LOAD BALANCING  (seed = %d)" % SEED)
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1) the scenario
    # ------------------------------------------------------------------
    appliances = get_appliances()
    price_array = get_price_array()
    solar_array = get_solar_array()
    describe_appliances(appliances)
    describe_pricing()
    print("Chromosome length: %d genes (%d starts + %d pause genes)"
          % (chromosome_length(appliances), len(appliances),
             chromosome_length(appliances) - len(appliances)))

    # ------------------------------------------------------------------
    # 2) baseline
    # ------------------------------------------------------------------
    baseline_schedule = get_baseline_schedule(appliances)
    baseline_metrics = evaluate(baseline_schedule, appliances,
                                price_array, solar_array)
    print("\nBEFORE optimisation (naive preferred-hour schedule):")
    print_case_line("baseline", baseline_metrics)

    # ------------------------------------------------------------------
    # 3) the Genetic Algorithm
    # ------------------------------------------------------------------
    ga_result = run_ga(appliances, price_array, solar_array, seed=SEED,
                       verbose=True)
    print("\nGA result:")
    print_case_line("GA", ga_result["best_metrics"],
                    "evals=%d, %.1f s" % (ga_result["evaluations"],
                                          ga_result["runtime_s"]))

    # ------------------------------------------------------------------
    # 4) the Memetic Algorithm
    # ------------------------------------------------------------------
    ma_result = run_ma(appliances, price_array, solar_array, seed=SEED,
                       verbose=True)
    print("\nMA result:")
    print_case_line("MA", ma_result["best_metrics"],
                    "evals=%d, %.1f s" % (ma_result["evaluations"],
                                          ma_result["runtime_s"]))

    # ------------------------------------------------------------------
    # 5) verification of the optimised results
    # ------------------------------------------------------------------
    print("\nChecking the optimised schedules...")
    valid_ga, violations_ga = schedule_is_valid(ga_result["best_schedule"],
                                                appliances)
    valid_ma, violations_ma = schedule_is_valid(ma_result["best_schedule"],
                                                appliances)
    assert valid_ga, "GA best schedule has %d violation hours!" % violations_ga
    assert valid_ma, "MA best schedule has %d violation hours!" % violations_ma
    assert ga_result["best_metrics"]["limit_violations"] == 0, \
        "GA trips the home power limit %d times!" \
        % ga_result["best_metrics"]["limit_violations"]
    assert ma_result["best_metrics"]["limit_violations"] == 0, \
        "MA trips the home power limit %d times!" \
        % ma_result["best_metrics"]["limit_violations"]
    print("  zero window violations: GA OK, MA OK")
    print("  home power limit never exceeded: GA OK, MA OK")
    assert ga_result["best_metrics"]["cost"] < baseline_metrics["cost"], \
        "GA did not beat the baseline cost!"
    assert ma_result["best_metrics"]["cost"] < baseline_metrics["cost"], \
        "MA did not beat the baseline cost!"
    print("  optimised cost is lower than baseline: GA OK, MA OK")
    check_energy_conservation(baseline_schedule, appliances, "baseline")
    check_energy_conservation(ga_result["best_schedule"], appliances, "GA")
    check_energy_conservation(ma_result["best_schedule"], appliances, "MA")

    # ------------------------------------------------------------------
    # 6) small paired comparison (statistics + box plot data)
    # ------------------------------------------------------------------
    print("\nRunning a paired GA/MA comparison over 6 seeds...")
    rows, _stats = run_paired_comparison(n_runs=6, seed0=SEED + 100)
    ga_costs = [row["ga_cost"] for row in rows]
    ma_costs = [row["ma_cost"] for row in rows]

    # ------------------------------------------------------------------
    # 7) percent improvements and tables
    # ------------------------------------------------------------------
    savings = savings_table(baseline_metrics, ga_result["best_metrics"],
                            ma_result["best_metrics"])
    print("\nImprovement vs baseline (single showcase run):")
    for algorithm in ("ga", "ma"):
        print("  %s: cost -%.1f%% | peak -%.1f%% | PAR -%.1f%% | "
              "solar self-consumption %.1f%%"
              % (algorithm, savings[algorithm]["cost_saving_pct"],
                 savings[algorithm]["peak_reduction_pct"],
                 savings[algorithm]["par_reduction_pct"],
                 savings[algorithm]["solar_self_consumption_pct"]))

    # summary CSV: one row per case
    summary_rows = []
    for case, metrics, note in (
            ("Baseline", baseline_metrics, "no optimisation"),
            ("GA", ga_result["best_metrics"],
             "evals %d, %.1f s" % (ga_result["evaluations"],
                                   ga_result["runtime_s"])),
            ("MA", ma_result["best_metrics"],
             "evals %d, %.1f s" % (ma_result["evaluations"],
                                   ma_result["runtime_s"]))):
        summary_rows.append({
            "case": case,
            "cost_egp": round(metrics["cost"], 2),
            "peak_import_kw": round(metrics["peak_import_kw"], 2),
            "par": round(metrics["par"], 3),
            "solar_self_consumption_pct": round(
                metrics["solar_self_consumption_pct"], 1),
            "violations": metrics["violations"],
            "over_limit_hours": metrics["limit_violations"],
            "note": note,
        })
    write_csv(os.path.join(RESULTS_DIR, "summary.csv"),
              list(summary_rows[0].keys()), summary_rows)
    write_schedule_table(baseline_schedule, ga_result, ma_result,
                         appliances, os.path.join(RESULTS_DIR,
                                                  "schedule_table.csv"))
    write_history_csv(ga_result["history"],
                      os.path.join(RESULTS_DIR, "ga_history.csv"))
    write_history_csv(ma_result["history"],
                      os.path.join(RESULTS_DIR, "ma_history.csv"))

    # ------------------------------------------------------------------
    # 8) plots
    # ------------------------------------------------------------------
    print("\nCreating plots...")
    make_all_plots(baseline_schedule, baseline_metrics, ga_result, ma_result,
                   price_array, solar_array, ga_costs, ma_costs, savings)

    print("\nDone. All tables and plots are in: %s" % RESULTS_DIR)


if __name__ == "__main__":
    main()
