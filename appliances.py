"""Definitions of the shiftable appliances plus the fixed base load.

Each appliance is a plain dictionary (no classes needed) with 7 fields:
    name            : short human readable name
    power_kw        : how much power it draws while running, in kW
    duration_hours  : how many hours it must run in TOTAL, as an integer
    window_start    : earliest hour it may start running (inclusive)
    window_end      : latest hour it may still be running (inclusive)
    interruptible   : True -> may pause for one hour inside its run
                      (see the chromosome notes in config.py)
    preferred_start : the hour the user would pick WITHOUT any
                      optimisation - this is what the naive baseline
                      schedule uses (a typical "dumb home" behaviour:
                      run the dishwasher right after dinner, charge the
                      EV as soon as you think of it, and so on).

Base load: fridge, lights, standby, some cooking... things the user can
not reschedule. Given as a 24-value kW array, added to every load profile.
"""

from config import N_HOURS


def _appliance(name, power_kw, duration_hours, window_start, window_end,
               interruptible, preferred_start):
    """Small helper so the big list below stays short and easy to read.

    Inputs are the 7 fields described in the module docstring.
    Returns one appliance dictionary.
    """
    return {
        "name": name,
        "power_kw": power_kw,
        "duration_hours": duration_hours,
        "window_start": window_start,
        "window_end": window_end,
        "interruptible": interruptible,
        "preferred_start": preferred_start,
    }


# ---------------------------------------------------------------------------
# The 13 shiftable appliances of the smart home (the task asks for at
# least 8; more appliances = more realistic interaction between them).
# Prices to remember: off-peak 0-6 = 1.15, mid 6-17 = 2.30,
#                     peak 17-22 = 4.80, late 22-24 = 1.60 EGP/kWh.
# The preferred_start column is written like a real user would behave:
# "wash right after dinner", "charge the EV over night", ...
# Narrow windows: washing machine, water heater, lawn mower, oven...
# Wide windows:   HVAC, vacuum robot, EV charger, cooling compressor.
# The daytime EV top-up was added on purpose: its late-morning window
# makes it COMPETE with HVAC and the pool pump for the limited solar
# generation (only ~3.9 kW), which creates the interesting interactions
# the optimizers have to untangle.
# ---------------------------------------------------------------------------
APPLIANCE_DEFS = [
    # --- EV charger: very high power, long run; user charges over night
    #     already (already near-optimal - a good discussion point) ---
    _appliance("ev_charger", 7.4, 5, 0, 9, True, 0),
    # --- EV top-up: second short charge before an afternoon outing;
    #     fights for the same solar energy as HVAC and the pool ---
    _appliance("ev_topup", 2.2, 3, 9, 17, False, 15),
    # --- Water heater: early morning showers; narrow window, pausable ---
    _appliance("water_heater", 3.0, 3, 4, 9, True, 5),
    # --- Washing machine: narrow late-evening window; the user starts it
    #     a bit too early (the last peak hour), it can slide into 22-23 ---
    _appliance("washing_machine", 2.0, 2, 20, 23, False, 21),
    # --- Dishwasher: "right after dinner" = peak hours! Late hours still
    #     allow pushing it into the cheap 22-24 zone ---
    _appliance("dishwasher", 1.8, 2, 16, 23, False, 18),
    # --- Dryer: usually started right after the dishwasher -> also peak.
    #     A 4.6 kW electric dryer is realistic (US-style resistive heater)
    #     and it is the appliance that makes the cheap 22-24 zone OVERFLOW
    #     the home power limit - the optimizers must find out who runs where ---
    _appliance("dryer", 4.6, 2, 16, 23, False, 19),
    # --- HVAC / air conditioner: big flexibility. The user turns it on at
    #     14:00 (hot afternoon); pre-cooling later in the solar midday is
    #     much cheaper. Wide window, pausable ---
    _appliance("hvac", 2.5, 6, 12, 23, True, 14),
    # --- Robot vacuum: cleans in the late afternoon; can also run at
    #     night or at the sunny midday; one continuous block ---
    _appliance("vacuum_robot", 0.8, 2, 12, 23, False, 16),
    # --- Pool pump: long daily cycle, wide window (8-17), pausable; the
    #     naive start at 8:00 partially misses the solar midday ---
    _appliance("pool_pump", 1.5, 6, 8, 17, True, 8),
    # --- Iron: user irons right after work; a few cheaper evening hours
    #     exist inside its window ---
    _appliance("iron", 1.1, 1, 16, 22, False, 17),
    # --- Oven: cooking dinner; preferred start = peak hour ---
    _appliance("oven", 2.0, 1, 16, 21, False, 18),
    # --- Rice cooker: also dinner; preferred start = peak hour ---
    _appliance("rice_cooker", 0.6, 1, 16, 21, False, 19),
    # --- Robot lawn mower: morning job; solar makes late morning free ---
    _appliance("lawn_mower", 0.5, 3, 6, 12, False, 7),
]

# ---------------------------------------------------------------------------
# Fixed base load in kW for each of the 24 hours (fridge + lights + small
# cooking spikes at breakfast and dinner). Can NOT be moved by the GA.
# ---------------------------------------------------------------------------
BASE_LOAD_KW = [
    0.55, 0.50, 0.45, 0.45, 0.50, 0.60,   # 00-05  night (fridge only + few lights)
    0.85, 1.05, 0.95, 0.75, 0.65, 0.65,   # 06-11  breakfast bump
    0.70, 0.75, 0.80, 0.85, 1.05, 1.50,   # 12-17  afternoon + start of evening
    1.80, 1.90, 1.70, 1.35, 0.95, 0.70,   # 18-23  dinner + evening lights
]


def get_appliances():
    """Return a fresh copy of the appliance list.

    No inputs. Returns a list of dictionaries (one per appliance).
    A copy is returned so experiment code can not accidentally edit the
    master definitions above.
    """
    return [dict(app) for app in APPLIANCE_DEFS]


def get_base_load():
    """Return the fixed 24-hour base load array in kW.

    No inputs. Returns a list of 24 floats.
    """
    return list(BASE_LOAD_KW)


def describe_appliances(appliances):
    """Print a small human readable table of the appliances.

    Input: the appliance list. No return value (just prints).
    Useful at the top of main.py so the scenario is visible in the log.
    """
    print("Appliances (name, power, duration, window, interruptible):")
    for app in appliances:
        kind = "can pause" if app["interruptible"] else "continuous"
        print("  %-15s %4.1f kW  %d h  window %2d-%2d  %s"
              % (app["name"], app["power_kw"], app["duration_hours"],
                 app["window_start"], app["window_end"], kind))
    print("  base load total: %.1f kWh (fixed, not shiftable)"
          % sum(get_base_load()))
