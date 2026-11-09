"""The fitness function: how good is one schedule? (LOWER is better)

    fitness = total_cost
            + W_PEAK * peak_grid_import
            + W_VIOLATION * window_violation_hours
            + W_LIMIT * over_limit_hours          (extra, documented term)

Definitions (all per hour h of the 24-hour day):
  home load[h]      = base load[h] + power of every appliance running at h
  net grid import[h] = max(0, home load[h] - solar generation[h])
                       (a solar surplus is simply not exported/sold)
  total_cost        = sum over h of net_import[h] * price[h]   [EGP]
  peak_grid_import  = max over h of net_import[h]              [kW]
  window_violation_hours = appliance-hours running outside the allowed
                      window or past hour 23 (from schedule_utils)
  over_limit_hours  = hours in which the grid import exceeds the home's
                      power limit (smart meter / main breaker, config.py)

Why these weights: see config.py. Short version - window violations are
punished so hard that ANY violating schedule is worse than EVERY valid
schedule (user comfort first, then optimisation), over-limit hours are
punished strongly enough that the final schedule never trips the breaker,
and the peak weight makes peak reduction a first-class goal.
"""

from config import (N_HOURS, W_PEAK, W_VIOLATION, GRID_LIMIT_KW, W_LIMIT)
from schedule_utils import (compute_load_profile,
                            count_violation_hours)


def count_limit_violation_hours(net_import):
    """Count the hours in which the home draws MORE power than its limit.

    Input: net_import (24 floats, kW of grid import per hour).
    Returns an integer >= 0 (one point per hour above GRID_LIMIT_KW).
    This models the main-breaker limit of a real house; see config.py.
    """
    over_limit_hours = 0
    for hour in range(N_HOURS):
        if net_import[hour] > GRID_LIMIT_KW:
            over_limit_hours += 1
    return over_limit_hours


def net_import_profile(home_load, solar_array):
    """Compute the grid import per hour (kW) that the house buys.

    Inputs: home_load (24 floats, kW) and solar_array (24 floats, kW).
    Returns a list of 24 floats: max(0, load - solar) for each hour.
    Solar surplus above the home load is wasted (not sold) - documented
    simplification.
    """
    net = []
    for hour in range(N_HOURS):
        import_now = home_load[hour] - solar_array[hour]
        if import_now < 0:
            import_now = 0
        net.append(import_now)
    return net


def compute_total_cost(net_import, price_array):
    """Multiply grid import by the per-hour price and sum it up.

    Inputs: net_import (24 floats, kW) and price_array (24 floats, EGP/kWh).
    Returns one float: the total electricity cost of the day in EGP.
    """
    total_cost = 0.0
    for hour in range(N_HOURS):
        total_cost = total_cost + net_import[hour] * price_array[hour]
    return total_cost


def solar_self_consumption(home_load, solar_array):
    """Share of the produced solar energy that the home actually uses, %.

    Inputs: home_load (24 floats) and solar_array (24 floats).
    Returns a float 0..100. Solar used per hour = min(load, solar):
    solar can only feed the house at the SAME hour it is produced
    (no battery in this model).
    """
    solar_produced = 0.0
    solar_used = 0.0
    for hour in range(N_HOURS):
        solar_produced = solar_produced + solar_array[hour]
        solar_used = solar_used + min(home_load[hour], solar_array[hour])
    if solar_produced <= 0:
        return 0.0
    return 100.0 * solar_used / solar_produced


def evaluate(schedule, appliances, price_array, solar_array):
    """Evaluate one schedule completely - THE central function of the project.

    Inputs:
      schedule   : chromosome (list of int genes)
      appliances : appliance list
      price_array: 24 EGP/kWh values
      solar_array: 24 kW values
    Returns a dictionary with the fitness AND all its ingredients:
      fitness      -> the single number the GA minimises
      cost         -> daily electricity bill, EGP
      peak_import_kw, par, violations, solar_self_consumption_pct,
      load_profile (24 kW), net_import (24 kW), home_load? (see keys below)
    """
    # 1) when does the home consume how much?
    load_profile = compute_load_profile(schedule, appliances)

    # 2) constraint violations: appliance-hours outside the window / day
    violations = count_violation_hours(schedule, appliances)

    # 3) what the grid must supply, its cost and its peak
    net_import = net_import_profile(load_profile, solar_array)
    total_cost = compute_total_cost(net_import, price_array)
    peak_import = max(net_import)
    limit_violations = count_limit_violation_hours(net_import)

    # 4) Peak-to-Average Ratio of the grid import (typical DSM metric)
    mean_import = sum(net_import) / float(N_HOURS)
    if mean_import > 1e-9:
        par = peak_import / mean_import
    else:
        par = 0.0

    # 5) solar usage
    solar_pct = solar_self_consumption(load_profile, solar_array)

    # 6) the actual fitness value (lower = better)
    fitness_value = (total_cost
                     + W_PEAK * peak_import
                     + W_VIOLATION * violations
                     + W_LIMIT * limit_violations)

    return {
        "fitness": fitness_value,
        "cost": total_cost,
        "peak_import_kw": peak_import,
        "par": par,
        "violations": violations,
        "limit_violations": limit_violations,
        "solar_self_consumption_pct": solar_pct,
        "load_profile": load_profile,
        "net_import": net_import,
    }
