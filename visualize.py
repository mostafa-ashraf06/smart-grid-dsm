"""All plots of the project, saved as PNG files into results/.

No plot is ever shown on screen: matplotlib uses the headless "Agg"
backend, so everything also works over SSH / in CI. Every figure has a
title, axis labels with units, and a legend.
"""

import os

import matplotlib
matplotlib.use("Agg")            # save to files, never open a window
import matplotlib.pyplot as plt
import numpy as np

from config import (RESULTS_DIR, N_HOURS)
from appliances import get_appliances
from schedule_utils import interpret_schedule, get_run_blocks

CASE_COLORS = {"baseline": "dimgray", "ga": "tab:blue", "ma": "tab:green"}


def _save(fig, filename):
    """Save a matplotlib figure into results/ and close it (no windows).

    Inputs: fig (the figure object), filename (PNG name inside results/).
    """
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  plot saved: results/%s" % filename)


def plot_load_profiles(baseline_load, ga_load, ma_load, solar_array,
                       price_array, filename="load_profiles.png"):
    """Plot (a): 24h load BEFORE (baseline) vs AFTER GA vs AFTER MA.

    Inputs: three 24-value load curves (kW), the solar curve (kW) and the
    24-value price array (EGP/kWh).
    Top panel: the three load curves + the solar curve filled orange
    (everything the panels can cover). Bottom panel: the price curve.
    """
    hours = list(range(N_HOURS))
    fig, (ax_load, ax_price) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.3]})

    ax_load.plot(hours, baseline_load, linestyle="--", linewidth=1.6,
                 label="Load BEFORE (baseline)", color=CASE_COLORS["baseline"])
    ax_load.plot(hours, ga_load, "-", linewidth=1.6, color=CASE_COLORS["ga"],
                 label="Load AFTER GA")
    ax_load.plot(hours, ma_load, "-", linewidth=1.6, color=CASE_COLORS["ma"],
                 label="Load AFTER MA")
    ax_load.fill_between(hours, 0, solar_array, color="orange", alpha=0.30,
                         label="Solar generation (free energy)")
    ax_load.set_ylabel("Load (kW)")
    ax_load.set_title("Home load profile over 24 h: baseline vs optimised")
    ax_load.legend(loc="upper left", fontsize=8)
    ax_load.grid(alpha=0.3)

    ax_price.step(hours, [0.0] + price_array[:N_HOURS - 1], where="post",
                  color="darkred")
    ax_price.fill_between(hours, 0, price_array, step="pre", color="darkred",
                          alpha=0.15)
    ax_price.set_xlabel("Hour of day")
    ax_price.set_ylabel("Price (EGP/kWh)")
    ax_price.set_title("Time-of-use price", fontsize=10)
    ax_price.grid(alpha=0.3)
    ax_price.set_xlim(0, N_HOURS - 1)
    _save(fig, filename)


def plot_gantt(baseline_schedule, best_schedule, appliances,
               filename="gantt.png", best_name="AFTER MA"):
    """Plot (b): Gantt-style appliance schedule BEFORE vs AFTER optimisation.

    Inputs: baseline chromosome, optimised chromosome, appliance list.
    Two side-by-side panels: grey bars = allowed window of each appliance,
    green bars = hours actually chosen to run (split blocks are visible).
    """
    fig, (ax_before, ax_after) = plt.subplots(
        1, 2, figsize=(13, 5.5), sharey=True)

    for ax, schedule, title in ((ax_before, baseline_schedule, "BEFORE"),
                                (ax_after, best_schedule, best_name)):
        parsed = interpret_schedule(schedule, appliances)
        for row_index, app in enumerate(appliances):
            start_hour, pause_index = parsed[row_index]
            # allowed window as a light grey background bar
            ax.barh(row_index, app["window_end"] - app["window_start"] + 1,
                    left=app["window_start"], height=0.9,
                    color="#e0e0e0", edgecolor="gray",
                    linewidth=0.5,
                    label="allowed window" if row_index == 0 else None)
            for block_start, block_end in get_run_blocks(app, start_hour,
                                                         pause_index):
                ax.barh(row_index, block_end - block_start + 1,
                        left=block_start, height=0.55, color="#2e7d32",
                        label="scheduled run" if (row_index == 0 and
                                                  ax is ax_before)
                        else None)
        ax.set_xlim(0, N_HOURS)
        ax.set_xticks(range(0, N_HOURS + 1, 2))
        ax.set_xlabel("Hour of day")
        ax.set_title("%s baseline" % title if ax is ax_before
                     else "%s %s" % (title, "(optimised)"),
                     fontsize=11)
        ax.grid(axis="x", alpha=0.3)

    ax_after.set_yticks(range(len(appliances)))
    ax_after.set_yticklabels([app["name"] for app in appliances], fontsize=8)
    ax_after.invert_yaxis()
    ax_before.legend(loc="upper left", fontsize=8)
    fig.suptitle("Appliance schedules: chosen run times inside allowed windows")
    _save(fig, filename)


def plot_convergence(history_ga, history_ma, filename="convergence.png"):
    """Plot (c): convergence curves, best + average fitness per generation.

    Inputs: the `history` lists returned by run_ga and run_ma.
    Solid lines = best individual of the population, dashed = average.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    x_ga = [row["generation"] for row in history_ga]
    x_ma = [row["generation"] for row in history_ma]
    ax.plot(x_ga, [row["best_fitness"] for row in history_ga], "-",
            color=CASE_COLORS["ga"], label="GA best")
    ax.plot(x_ga, [row["avg_fitness"] for row in history_ga], "--",
            color=CASE_COLORS["ga"], alpha=0.6, label="GA average")
    ax.plot(x_ma, [row["best_fitness"] for row in history_ma], "-",
            color=CASE_COLORS["ma"], label="MA best")
    ax.plot(x_ma, [row["avg_fitness"] for row in history_ma], "--",
            color=CASE_COLORS["ma"], alpha=0.6, label="MA average")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (cost + peak/limit penalties, EGP scale)\nlower is better")
    ax.set_title("Convergence: GA vs Memetic Algorithm (same seed family)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, filename)


def plot_cost_boxes(ga_costs, ma_costs, filename="cost_boxplot.png"):
    """Plot (d): box plot of the final daily cost over many independent runs.

    Inputs: lists of final costs (EGP), one number per run, for GA and MA.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot([ga_costs, ma_costs], patch_artist=True, showmeans=True,
               boxprops={"facecolor": "#cfe3f7"},
               medianprops={"color": "black"})
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["GA", "MA"])
    ax.set_ylabel("Final daily electricity cost (EGP)")
    ax.set_title("Spread of final cost over independent runs\n(lower and tighter = better)")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, filename)


def plot_savings(savings, filename="savings.png"):
    """Plot (e): bar chart of cost saving % and peak reduction % vs baseline.

    Inputs: the dict returned by experiments.savings_table
    (keys 'ga' and 'ma', each holding *_reduction_pct / cost_saving_pct).
    """
    categories = ["Cost saving %", "Peak reduction %", "PAR reduction %"]
    keys = ["cost_saving_pct", "peak_reduction_pct", "par_reduction_pct"]
    ga_values = [savings["ga"][key] for key in keys]
    ma_values = [savings["ma"][key] for key in keys]
    x_positions = range(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_ga = ax.bar([x - width / 2 for x in x_positions], ga_values,
                     width, color=CASE_COLORS["ga"], label="GA")
    bars_ma = ax.bar([x + width / 2 for x in x_positions], ma_values,
                     width, color=CASE_COLORS["ma"], label="MA")
    for bars in (bars_ga, bars_ma):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.4,
                    "%.1f" % height, ha="center", fontsize=9)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(categories)
    ax.set_ylabel("Improvement vs naive baseline (%)")
    ax.set_title("How much did the optimisation help?")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, filename)


def plot_param_study(rows, tag=""):
    """Plot (f): heatmap of the GA parameter study.

    Input: rows from experiments.run_parameter_study (dicts with pop_size,
    mutation_rate, crossover_rate, mean_fitness). One heatmap per crossover
    rate: x = mutation rate, y = population size, cell = mean final fitness
    (lower = darker = better).
    """
    crossover_rates = sorted(set(row["crossover_rate"] for row in rows))
    pop_sizes = sorted(set(row["pop_size"] for row in rows))
    mutation_rates = sorted(set(row["mutation_rate"] for row in rows))

    fig, axes = plt.subplots(len(crossover_rates), 1,
                             figsize=(8, 4.2 * len(crossover_rates)))
    if len(crossover_rates) == 1:
        axes = [axes]
    for ax, rate in zip(axes, crossover_rates):
        matrix = np.full((len(pop_sizes), len(mutation_rates)), np.nan)
        for row in rows:
            if row["crossover_rate"] != rate:
                continue
            row_i = pop_sizes.index(row["pop_size"])
            col_i = mutation_rates.index(row["mutation_rate"])
            matrix[row_i][col_i] = row["mean_fitness"]
        image = ax.imshow(matrix, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(mutation_rates)))
        ax.set_xticklabels(["%.2f" % value for value in mutation_rates])
        ax.set_yticks(range(len(pop_sizes)))
        ax.set_yticklabels([str(value) for value in pop_sizes])
        ax.set_xlabel("Mutation rate")
        ax.set_ylabel("Population size")
        ax.set_title("Crossover rate = %.2f (cell = mean fitness, lower better)"
                     % rate)
        for row_i in range(len(pop_sizes)):
            for col_i in range(len(mutation_rates)):
                if not np.isnan(matrix[row_i][col_i]):
                    ax.text(col_i, row_i, "%.0f" % matrix[row_i][col_i],
                            ha="center", va="center", color="white",
                            fontsize=9)
        fig.colorbar(image, ax=ax, label="mean best fitness")
    fig.suptitle("GA parameter study")
    _save(fig, "param_study%s.png" % tag)


def make_all_plots(baseline_schedule, baseline_metrics, ga_result, ma_result,
                   price_array, solar_array, ga_costs, ma_costs, savings):
    """Create the five plots that only need the single-pipeline results.

    Inputs: baseline chromosome + result dictionaries (with load profiles),
    price + solar arrays, lists of per-run costs (for the box plot) and the
    savings dict. Saves: load_profiles.png, gantt.png, convergence.png,
    cost_boxplot.png, savings.png.
    """
    plot_load_profiles(baseline_metrics["load_profile"],
                       ga_result["best_metrics"]["load_profile"],
                       ma_result["best_metrics"]["load_profile"],
                       solar_array, price_array)
    plot_gantt(baseline_schedule, ma_result["best_schedule"],
               get_appliances(), best_name="AFTER MA")
    plot_convergence(ga_result["history"], ma_result["history"])
    plot_cost_boxes(ga_costs, ma_costs)
    plot_savings(savings)
