"""Time-of-use (TOU) price array and the rooftop solar generation curve.

Both are just 24-value lists (kW or EGP/kWh), one value per hour.
Everything that depends on them reads these arrays, so swapping in real
data (NREL PVWatts, PVGIS, a real tariff sheet) only means replacing the
two builder functions below - no other file has to change.
"""

import math

from config import (N_HOURS, PRICE_OFF_PEAK, PRICE_MID, PRICE_PEAK,
                    PRICE_LATE, OFF_PEAK_END, MID_END, PEAK_END,
                    SOLAR_PEAK_KW, SOLAR_CENTER_HOUR, SOLAR_SIGMA_HOURS)


def get_price_array():
    """Build the 24-hour time-of-use price array in EGP per kWh.

    No inputs. Returns a list of 24 floats (EGP/kWh for each hour).
    Tier boundaries come from config.py so they are easy to change.
    """
    prices = []
    for hour in range(N_HOURS):
        if hour < OFF_PEAK_END:
            price = PRICE_OFF_PEAK        # night: cheap
        elif hour < MID_END:
            price = PRICE_MID             # working hours: medium
        elif hour < PEAK_END:
            price = PRICE_PEAK            # evening peak: expensive
        else:
            price = PRICE_LATE            # late night: cheap again
        prices.append(price)
    return prices


def get_solar_array():
    """Build the 24-hour solar generation curve of the rooftop PV.

    No inputs. Returns a list of 24 floats (kW generated in each hour).

    We use a Gaussian (bell) curve centred just after noon. It is zero at
    night because the bell is so narrow - 12 hours away from the centre the
    value is below 1e-6 kW. To use REAL measured solar data instead:
    export an hourly profile from NREL PVWatts or PVGIS (CSV), read it here
    with the csv module, and return its 24 values. Nothing else changes.
    """
    solar = []
    for hour in range(N_HOURS):
        distance = hour - SOLAR_CENTER_HOUR
        value = SOLAR_PEAK_KW * math.exp(-(distance ** 2) / (2 * SOLAR_SIGMA_HOURS ** 2))
        # keep a clean 2-decimal profile, and clamp negatives just in case
        value = max(0.0, round(value, 2))
        solar.append(value)
    return solar


def describe_pricing():
    """Print the price tiers and a mini summary of the solar curve.

    No inputs, no return value (just prints). Handy for the project log.
    """
    prices = get_price_array()
    solar = get_solar_array()
    print("Price tiers (EGP/kWh): off-peak(00-06)=%.2f  mid(06-17)=%.2f  "
          "peak(17-22)=%.2f  late(22-24)=%.2f"
          % (PRICE_OFF_PEAK, PRICE_MID, PRICE_PEAK, PRICE_LATE))
    print("Solar: peak %.1f kW at hour %d, total %.1f kWh/day"
          % (max(solar), solar.index(max(solar)), sum(solar)))
    print("Solar array:", solar)
