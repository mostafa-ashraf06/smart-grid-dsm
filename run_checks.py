"""run_checks.py - automatic self-verification of the WHOLE project.

Run from inside this folder:   python run_checks.py
It prints one PASS/FAIL line per check, saves results/check_report.txt
and exits with code 0 only if EVERYTHING passes.

Checks:
  1. imports + requirements.txt contents
  2. appliance and array data validity
  3. price tiers and solar curve sanity
  4. schedule_utils hand-computed unit test (exact numbers)
  5. violating schedule ALWAYS loses against every valid schedule
  6. GA result: zero violations, beats the baseline, history improves
  7. MA result: zero violations, MA <= GA in mean over paired seeds
  8. reproducibility: same seed -> identical result
  9. energy conservation for baseline / GA / MA schedules
 10. main.py runs end to end (subprocess) and creates all artifacts
 11. a quick parameter-study run writes its CSV + heatmap
"""

import os
import random
import subprocess
import sys
import time

import config as cfg
from config import (SEED, RESULTS_DIR, N_HOURS, ensure_results_dir)
from appliances import get_appliances, get_base_load
from pricing import get_price_array, get_solar_array
from baseline import get_baseline_schedule, baseline_already_valid
from schedule_utils import (chromosome_length, allowed_start_range,
                            get_running_hours, interpret_schedule,
                            count_violation_hours, compute_load_profile,
                            schedule_is_valid, total_appliance_energy,
                            get_run_blocks)
from fitness import evaluate
from ga import run_ga, create_individual
from ma import run_ma

CACHE = {}          # lets later checks reuse results of earlier heavy checks
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def make_tiny_appliances():
    """Two tiny appliances for the hand-computed unit test in check 4.

    No inputs. Returns (appliance_list, expected cases description).
    """
    continuous = {"name": "mini_pump", "power_kw": 2.0, "duration_hours": 3,
                  "window_start": 0, "window_end": 6,
                  "interruptible": False, "preferred_start": 0}
    pausable = {"name": "mini_pausable", "power_kw": 2.0, "duration_hours": 3,
                "window_start": 0, "window_end": 6,
                "interruptible": True, "preferred_start": 0}
    return [continuous, pausable]


# ---------------------------------------------------------------------------
# The checks themselves - each returns True or raises an AssertionError.
# ---------------------------------------------------------------------------
def check_1_imports():
    """All project modules import and requirements.txt lists them."""
    try:
        import numpy
        import matplotlib
    except Exception as error:  # pragma: no cover - would fail the check
        raise AssertionError("numpy/matplotlib import failed: %s" % error)
    requirements = open(os.path.join(PROJECT_DIR, "requirements.txt"),
                        encoding="utf-8").read()
    assert "numpy" in requirements and "matplotlib" in requirements, \
        "requirements.txt must list numpy and matplotlib"
    return True


def check_2_appliance_data():
    """Appliance list and arrays are structurally valid."""
    appliances = get_appliances()
    assert len(appliances) >= 8, "need at least 8 appliances"
    narrow = 0
    wide = 0
    interruptible_count = 0
    continuous_count = 0
    for app in appliances:
        duration = app["duration_hours"]
        assert isinstance(duration, int) and duration > 0, \
            "%s: duration must be a positive integer" % app["name"]
        assert 0 <= app["window_start"] <= app["window_end"] <= 23, \
            "%s: window must fit into hour 0..23" % app["name"]
        assert app["window_start"] <= app["preferred_start"] <= app["window_end"], \
            "%s: preferred start must be inside the window" % app["name"]
        allowed_start_range(app)   # raises ValueError if window too short
        span = app["window_end"] - app["window_start"] + 1
        if span <= 6:      # narrow: 3 - 6 possible start hours
            narrow += 1
        if span >= 8:      # wide: 9+ possible start hours
            wide += 1
        if app["interruptible"]:
            assert span >= duration + 1, \
                "%s: window must fit duration + 1 pause hour" % app["name"]
            interruptible_count += 1
        else:
            assert span >= duration
            continuous_count += 1
    assert narrow >= 2, "need at least 2 narrow windows"
    assert wide >= 2, "need at least 2 wide windows"
    assert interruptible_count >= 2 and continuous_count >= 2, \
        "need at least 2 interruptible and 2 continuous appliances"
    assert chromosome_length(appliances) == len(appliances) + interruptible_count
    assert baseline_already_valid(appliances), \
        "the naive preferred-hour baseline must respect every window"
    base_load = get_base_load()
    assert len(base_load) == N_HOURS
    assert all(0.0 <= value < 5.0 for value in base_load)
    assert get_price_array() and len(get_price_array()) == N_HOURS
    assert len(get_solar_array()) == N_HOURS
    return True


def check_3_pricing_solar():
    """Price tiers and the solar bell curve behave as configured."""
    price = get_price_array()
    solar = get_solar_array()
    assert price[0] == cfg.PRICE_OFF_PEAK and price[5] == cfg.PRICE_OFF_PEAK
    assert price[6] == cfg.PRICE_MID and price[16] == cfg.PRICE_MID
    assert price[17] == cfg.PRICE_PEAK and price[21] == cfg.PRICE_PEAK
    assert price[22] == cfg.PRICE_LATE and price[23] == cfg.PRICE_LATE
    for hour in (0, 1, 2, 3, 22, 23):
        assert solar[hour] <= 0.05, "solar must be ~0 at night (hour %d)" % hour
    noon_peak = max(solar[11] , solar[12], solar[13])
    assert noon_peak >= 0.9 * cfg.SOLAR_PEAK_KW, "solar peak too low at noon"
    for hour in range(N_HOURS):
        assert solar[hour] >= 0.0
    return True


def check_4_schedule_utils_unit():
    """Hand-computed profiles match exactly (unit test with real numbers)."""
    tiny = make_tiny_appliances()
    base_load = get_base_load()

    # Case A: continuous appliance, start hour 2 -> power at 2,3,4.
    # NOTE: pass ONE appliance at a time, because the chromosome length must
    # always match the appliance list length it is interpreted against.
    tiny_cont_only = [tiny[0]]
    schedule_a = [2]
    expected_a = list(base_load)
    for hour in (2, 3, 4):
        expected_a[hour] += 2.0
    assert compute_load_profile(schedule_a, tiny_cont_only) == expected_a
    on_hours = get_running_hours(tiny[0], 2, 0)
    assert on_hours == [2, 3, 4]
    assert get_run_blocks(tiny[0], 2, 0) == [(2, 4)]
    assert allowed_start_range(tiny[0]) == (0, 4)   # 7 window vs 3 hours
    # energy conservation for the tiny case
    energy = total_appliance_energy(schedule_a, tiny_cont_only)
    assert energy["mini_pump"] == 6.0

    # Case B: pausable appliance, start 1, pause gene 2 ->
    # runs 2 hours (1,2), pauses hour 3, runs hour 4 (still 3 h total,
    # spread over a 4-hour span).
    tiny_pausable_only = [tiny[1]]
    schedule_b = [1, 2]
    expected_b = list(base_load)
    for hour in (1, 2, 4):
        expected_b[hour] += 2.0
    assert compute_load_profile(schedule_b, tiny_pausable_only) == expected_b
    assert get_running_hours(tiny[1], 1, 2) == [1, 2, 4]
    assert get_run_blocks(tiny[1], 1, 2) == [(1, 2), (4, 4)]
    assert allowed_start_range(tiny[1]) == (0, 3)   # span 4 must fit 0..6
    assert schedule_is_valid(schedule_b, tiny_pausable_only) == (True, 0)
    return True


def check_5_violation_dominates():
    """A violating schedule must lose against EVERY valid schedule tried."""
    appliances = get_appliances()
    price = get_price_array()
    solar = get_solar_array()
    random.seed(1234)
    valid_fitnesses = []
    valid_candidates = [get_baseline_schedule(appliances)]
    for _ in range(30):
        valid_candidates.append(create_individual(appliances, 1.0))
    for candidate in valid_candidates:
        is_valid, _violations = schedule_is_valid(candidate, appliances)
        if is_valid:
            valid_fitnesses.append(
                evaluate(candidate, appliances, price, solar)["fitness"])
    assert valid_fitnesses, "expected at least one valid schedule"

    # take a valid schedule and force the washing machine outside its window
    # (index found BY NAME so changing the appliance order can not break it)
    washing_index = [app["name"] == "washing_machine" for app in appliances].index(True)
    broken = list(get_baseline_schedule(appliances))
    broken[washing_index] = 12   # washing machine window is 20-23 -> runs 12,13
    broken_fitness = evaluate(broken, appliances, price, solar)["fitness"]
    violations = count_violation_hours(broken, appliances)
    assert violations == 2, "washing machine should sit 2 hours at window 20-23"
    assert broken_fitness > max(valid_fitnesses), \
        "violating schedule (%.1f) must beat no valid one (max valid %.1f)" \
        % (broken_fitness, max(valid_fitnesses))
    return True


def check_6_ga_result():
    """GA: zero violations, cheaper than baseline, history never worsens."""
    appliances = get_appliances()
    price = get_price_array()
    solar = get_solar_array()
    baseline_cost = evaluate(get_baseline_schedule(appliances), appliances,
                             price, solar)["cost"]
    result = run_ga(appliances, price, solar, seed=SEED)
    CACHE["ga_showcase"] = result
    is_valid, violations = schedule_is_valid(result["best_schedule"], appliances)
    assert is_valid, "GA best schedule still has %d violations" % violations
    assert result["best_metrics"]["limit_violations"] == 0, \
        "GA trips the home power limit %d times" \
        % result["best_metrics"]["limit_violations"]
    assert result["best_metrics"]["cost"] < baseline_cost, \
        "GA cost %.2f must be below baseline %.2f" \
        % (result["best_metrics"]["cost"], baseline_cost)
    first_best = result["history"][0]["best_fitness"]
    for row in result["history"]:
        assert row["best_fitness"] <= first_best + 1e-9, \
            "GA best fitness must never worsen (elitism broken)"
        first_best = row["best_fitness"]
    final_best = result["history"][-1]["best_fitness"]
    assert final_best < result["history"][0]["avg_fitness"], \
        "GA must do better than its average generation-0 schedule"
    return True


def check_7_ma_vs_ga_paired():
    """MA: zero violations every run, MA fitness <= GA fitness in the mean.

    Runs several seeds; each MA run uses the SAME seed as its GA twin, so
    both start from the identical random population (fair pairing).
    """
    appliances = get_appliances()
    price = get_price_array()
    solar = get_solar_array()
    ga_fits = []
    ma_fits = []
    lines = []
    for offset in range(6):
        seed = SEED + 300 + offset
        ga_result = run_ga(appliances, price, solar, seed=seed)
        ma_result = run_ma(appliances, price, solar, seed=seed)
        is_valid, violations = schedule_is_valid(ma_result["best_schedule"],
                                                 appliances)
        assert is_valid, "MA run seed %d has %d violations" % (seed, violations)
        assert ma_result["best_metrics"]["limit_violations"] == 0, \
            "MA run seed %d trips the limit %d times" \
            % (seed, ma_result["best_metrics"]["limit_violations"])
        ga_fits.append(ga_result["best_fitness"])
        ma_fits.append(ma_result["best_fitness"])
        lines.append("  seed %d: GA %.2f | MA %.2f" % (seed,
                                                       ga_result["best_fitness"],
                                                       ma_result["best_fitness"]))
        if offset == 0:
            CACHE["ma_showcase"] = ma_result
    print("\n".join(lines))
    mean_ga = sum(ga_fits) / len(ga_fits)
    mean_ma = sum(ma_fits) / len(ma_fits)
    assert mean_ma <= mean_ga + 1e-6, \
        "MA mean %.3f must not exceed GA mean %.3f (paired seeds)" \
        % (mean_ma, mean_ga)
    return True


def check_8_reproducibility():
    """Same seed -> the exact same result twice; different seed -> differs."""
    appliances = get_appliances()
    price = get_price_array()
    solar = get_solar_array()
    first = run_ga(appliances, price, solar, pop_size=30, generations=40,
                   seed=777)
    second = run_ga(appliances, price, solar, pop_size=30, generations=40,
                    seed=777)
    assert first["best_schedule"] == second["best_schedule"]
    assert first["best_fitness"] == second["best_fitness"]
    ma_first = run_ma(appliances, price, solar, pop_size=30, generations=40,
                      seed=777)
    ma_second = run_ma(appliances, price, solar, pop_size=30, generations=40,
                       seed=777)
    assert ma_first["best_schedule"] == ma_second["best_schedule"]
    third = run_ga(appliances, price, solar, pop_size=30, generations=40,
                   seed=778)
    assert third["best_schedule"] != first["best_schedule"] or \
        third["best_fitness"] != first["best_fitness"]
    return True


def check_9_energy_conservation():
    """Every appliance draws exactly power * duration, in all schedules."""
    appliances = get_appliances()
    for label, schedule in (
            ("baseline", get_baseline_schedule(appliances)),
            ("GA", CACHE.get("ga_showcase", {}).get("best_schedule")),
            ("MA", CACHE.get("ma_showcase", {}).get("best_schedule"))):
        if schedule is None:
            continue
        energy = total_appliance_energy(schedule, appliances)
        for app in appliances:
            expected = app["power_kw"] * app["duration_hours"]
            assert abs(energy[app["name"]] - expected) < 1e-6, \
                "%s / %s: %.3f kWh != %.3f kWh" % (
                    label, app["name"], energy[app["name"]], expected)
    return True


def check_10_main_end_to_end():
    """main.py runs from a clean state and creates every artifact."""
    started = time.perf_counter()
    process = subprocess.run([sys.executable, "main.py"],
                             cwd=PROJECT_DIR, capture_output=True,
                             text=True, timeout=300)
    elapsed = time.perf_counter() - started
    assert process.returncode == 0, \
        "main.py failed:\n%s" % (process.stdout[-2500:] + process.stderr[-2500:])
    assert elapsed < 300, "main.py took %.1f s (limit 300 s)" % elapsed
    expected_files = ["summary.csv", "schedule_table.csv", "ga_history.csv",
                      "ma_history.csv", "load_profiles.png", "gantt.png",
                      "convergence.png", "cost_boxplot.png", "savings.png",
                      "comparison_runs.csv", "comparison_stats.csv"]
    missing = [name for name in expected_files
               if not os.path.exists(os.path.join(RESULTS_DIR, name))]
    assert not missing, "main.py did not create: %s" % missing
    print("  main.py finished in %.1f s and created all %d artifacts"
          % (elapsed, len(expected_files)))
    return True


def check_11_parameter_study_artifacts():
    """A quick parameter-study run writes its CSV table and heatmap PNG."""
    from experiments import run_parameter_study
    run_parameter_study(param_seeds=1,
                        pop_sizes=[30, 60],
                        mutation_rates=[0.01, 0.05],
                        crossover_rates=[0.9],
                        tag="_check")
    for name in ("param_study_check.csv", "param_study_check.png"):
        assert os.path.exists(os.path.join(RESULTS_DIR, name)), \
            "missing %s" % name
    return True


CHECKS = [
    ("imports/requirements", check_1_imports),
    ("appliance data validity", check_2_appliance_data),
    ("pricing + solar sanity", check_3_pricing_solar),
    ("schedule_utils unit test", check_4_schedule_utils_unit),
    ("violation always loses", check_5_violation_dominates),
    ("GA beats baseline, no violations", check_6_ga_result),
    ("MA <= GA (mean over paired seeds)", check_7_ma_vs_ga_paired),
    ("reproducibility (same seed)", check_8_reproducibility),
    ("energy conservation", check_9_energy_conservation),
    ("main.py end-to-end", check_10_main_end_to_end),
    ("parameter study artifacts", check_11_parameter_study_artifacts),
]


def main():
    """Run every check, print PASS/FAIL lines, write the report, exit 0/1."""
    started = time.perf_counter()
    ensure_results_dir()
    report_lines = []
    print("SELF-CHECK of the smart grid DSM project")
    print("=" * 60)
    passed = 0
    failed = 0
    for index, (name, check_function) in enumerate(CHECKS, start=1):
        line = "CHECK %2d/%d  %-38s" % (index, len(CHECKS), name)
        print("%s ... " % line.strip(), end="", flush=True)
        check_started = time.perf_counter()
        try:
            ok = check_function() is not False
            note = ""
        except Exception as error:
            ok = False
            note = str(error)
        seconds = time.perf_counter() - check_started
        if ok:
            passed += 1
            result_line = "%s PASS (%.1f s)" % (line, seconds)
        else:
            failed += 1
            result_line = "%s FAIL (%.1f s)  ->  %s" % (line, seconds, note)
            print(result_line)
            report_lines.append(result_line)
            continue
        print("PASS (%.1f s)" % seconds)
        report_lines.append(result_line)
    elapsed = time.perf_counter() - started
    summary = "SUMMARY: %d/%d checks passed in %.1f s" % (
        passed, len(CHECKS), elapsed)
    print(summary)
    if failed:
        summary += "  !! FAILED CHECKS: %d" % failed
    report_lines.insert(0, summary)
    with open(os.path.join(RESULTS_DIR, "check_report.txt"), "w",
              encoding="utf-8") as report_file:
        report_file.write("\n".join(report_lines) + "\n")
    print("Full report saved to results/check_report.txt")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
