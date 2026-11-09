"""Schedule helpers: decode genes, build load profiles, check the windows.

A "schedule" (chromosome) is a plain list of integers:
    genes[0 .. N-1]  = the start hour of appliance number 0 .. N-1
    genes[N ..]      = one pause gene for each INTERRUPTIBLE appliance
                       (0 = no pause, 1..D-1 = which of its run hours to
                       skip - full explanation in config.py)

No other file needs to know this layout: they just call functions here.
"""

from appliances import get_base_load
from config import N_HOURS


def chromosome_length(appliances):
    """Return the number of genes a chromosome needs for these appliances.

    Input: the appliance list.
    Returns one integer: N start genes + one gene per interruptible appliance.
    """
    return len(appliances) + sum(1 for app in appliances if app["interruptible"])


def allowed_start_range(appliance):
    """Return the earliest and latest start hour (inclusive) that keeps
    the whole run inside the appliance window AND inside the 24-hour day.

    Input: one appliance dictionary.
    Returns a tuple (min_start, max_start).

    Why interruptible reach one hour less: a pausing run spans D+1 hours
    (D running hours + 1 pause hour), so the start must be small enough
    that start + D is still within the window/day.
    """
    window_start = appliance["window_start"]
    window_end = min(appliance["window_end"], N_HOURS - 1)  # never past 23:00
    duration = appliance["duration_hours"]
    if appliance["interruptible"]:
        span = duration + 1    # one pause hour can be placed inside the span
    else:
        span = duration
    min_start = window_start
    max_start = window_end - span + 1
    if max_start < min_start:
        # Invalid appliance data; run_checks.py detects this immediately.
        raise ValueError("Window too short for %s" % appliance["name"])
    return min_start, max_start


def get_running_hours(appliance, start_hour, pause_index):
    """Return the list of hours (integers) at which the appliance draws power.

    Inputs:
      appliance    : the appliance dictionary (power/duration only matter)
      start_hour   : the start gene
      pause_index  : the pause gene (0 = no pause, else which run hour to skip)
    Returns a list of exactly `duration_hours` integers (may contain hours
    >= 24 if the schedule is invalid - the violation counter will catch it).

    This is the single definition of "when is an appliance on", used by the
    load profile, the violation counter and the energy check, so they can
    never disagree.
    """
    duration = appliance["duration_hours"]
    # Defensive clamp keeps the meaning of a bad pause gene well-defined.
    pause = min(max(int(pause_index), 0), duration - 1)
    if pause == 0:
        # No pause: one continuous block of D hours [start, start + D - 1].
        return list(range(start_hour, start_hour + duration))
    # One pause: block 1 = the first `pause` hours, off for one hour,
    # then block 2 lasting the remaining duration - pause hours.
    first_block = list(range(start_hour, start_hour + pause))
    second_block = list(range(start_hour + pause + 1, start_hour + 1 + duration))
    return first_block + second_block


def interpret_schedule(schedule, appliances):
    """Split a chromosome into (start_hour, pause_index) pairs.

    Inputs: the chromosome (list of ints) and the appliance list.
    Returns a list with one (start, pause) tuple per appliance, in the same
    order as the appliances.
    """
    n_apps = len(appliances)
    parsed = []
    extra_gene_index = n_apps   # pause genes live after all the start genes
    for i in range(n_apps):
        start_hour = schedule[i]
        pause_index = 0
        if appliances[i]["interruptible"]:
            pause_index = schedule[extra_gene_index]
            extra_gene_index += 1
        parsed.append((start_hour, pause_index))
    return parsed


def count_violation_hours(schedule, appliances):
    """Count appliance-hours that run OUTSIDE the allowed window or the day.

    Inputs: chromosome and appliance list.
    Returns an integer >= 0. This is the "violation_amount" used by the
    fitness penalty: every appliance-hour running before/after its window
    or past 23:00 counts as exactly one violation.
    """
    total_violations = 0
    parsed = interpret_schedule(schedule, appliances)
    for app, (start_hour, pause_index) in zip(appliances, parsed):
        for hour in get_running_hours(app, start_hour, pause_index):
            outside_window = hour < app["window_start"] or hour > app["window_end"]
            outside_day = hour < 0 or hour >= N_HOURS
            if outside_window or outside_day:
                total_violations += 1
    return total_violations


def schedule_is_valid(schedule, appliances):
    """Check that a schedule breaks no window and no day-boundary rule.

    Inputs: chromosome and appliance list.
    Returns a tuple (is_valid_boolean, number_of_violating_hours).
    """
    violations = count_violation_hours(schedule, appliances)
    return violations == 0, violations


def compute_load_profile(schedule, appliances):
    """Build the total home load profile (kW) for one schedule.

    Inputs: chromosome and appliance list.
    Returns a list of 24 floats: base load + every appliance drawing power
    in that hour. Hours outside 0..23 are ignored here (they are already
    punished by the violation penalty, and no grid exists past midnight).
    """
    load = get_base_load()  # start from the fixed fridge/lights/cooking load
    parsed = interpret_schedule(schedule, appliances)
    for app, (start_hour, pause_index) in zip(appliances, parsed):
        for hour in get_running_hours(app, start_hour, pause_index):
            if 0 <= hour < N_HOURS:
                load[hour] = load[hour] + app["power_kw"]
    return load


def get_run_blocks(app, start_hour, pause_index):
    """Return the appliance run times as contiguous blocks for plotting.

    Inputs: one appliance, its start gene and pause gene.
    Returns a list of (first_hour, last_hour) tuples (inclusive).
    Used to draw the Gantt charts.
    """
    hours = get_running_hours(app, start_hour, pause_index)
    blocks = []
    current_first = None
    current_last = None
    for hour in sorted(hours):
        if current_first is None:
            current_first = hour
            current_last = hour
        elif hour == current_last + 1:
            current_last = hour
        else:
            blocks.append((current_first, current_last))
            current_first = hour
            current_last = hour
    if current_first is not None:
        blocks.append((current_first, current_last))
    return blocks


def total_appliance_energy(schedule, appliances):
    """Return how much energy (kWh) each appliance actually draws.

    Inputs: chromosome and appliance list.
    Returns a dict {appliance_name: kWh}. By definition this equals
    power_kw * duration_hours for every schedule, which is exactly what
    run_checks.py verifies (energy conservation).
    """
    energy = {}
    parsed = interpret_schedule(schedule, appliances)
    for app, (start_hour, pause_index) in zip(appliances, parsed):
        hours = get_running_hours(app, start_hour, pause_index)
        energy[app["name"]] = app["power_kw"] * len(hours)
    return energy
