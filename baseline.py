"""The naive "no optimisation" schedule used as the BEFORE case.

Rule: every appliance simply starts at the FIRST hour of its allowed
window (window_start) and runs without pausing. This is how a house
without smart controls would behave: the user just turns things on as
soon as their rules allow.

The alternative "preferred start time" rule would work the same way -
just replace window_start by the preferred hour; the rest of the project
does not care where the baseline comes from.
"""

from schedule_utils import (interpret_schedule, chromosome_length,
                            get_running_hours)
from appliances import get_appliances


def get_baseline_schedule(appliances):
    """Build the naive baseline schedule.

    Input: the appliance list.
    Returns a chromosome (list of ints): every start gene = the appliance's
    PREFERRED start time (the hour a normal user would pick with no smart
    scheduling - see appliances.py), and every pause gene = 0 (run
    continuously, no pause).
    """
    schedule = []
    for app in appliances:
        schedule.append(app["preferred_start"])
    for app in appliances:
        if app["interruptible"]:
            schedule.append(0)   # no pause in the baseline
    return schedule


def baseline_already_valid(appliances):
    """Sanity helper: the baseline must respect all windows by construction.

    Input: the appliance list. Returns True if every window can hold the
    continuous run of its appliance (starting at the preferred hour).
    Used by run_checks.py.
    """
    schedule = get_baseline_schedule(appliances)
    parsed = interpret_schedule(schedule, appliances)
    for app, (start_hour, pause_index) in zip(appliances, parsed):
        for hour in get_running_hours(app, start_hour, pause_index):
            if hour < app["window_start"] or hour > app["window_end"] or hour >= 24:
                return False
    return True


if __name__ == "__main__":
    # Tiny manual demo: print the baseline chromosome and its running hours.
    apps = get_appliances()
    schedule = get_baseline_schedule(apps)
    print("Baseline chromosome (%d genes): %s"
          % (chromosome_length(apps), schedule))
    for app, (start_hour, pause_index) in zip(apps, interpret_schedule(schedule, apps)):
        print("%-15s start=%2d pause=%d -> running at %s"
              % (app["name"], start_hour, pause_index,
                 get_running_hours(app, start_hour, pause_index)))
