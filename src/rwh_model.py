import numpy as np


def calculate_rwh(rainfall_mm, roof_area_m2, runoff_coeff):
    """
    Calculate rooftop water harvesting potential.
    RWH = P * A * C
    """
    return rainfall_mm * roof_area_m2 * runoff_coeff


def calculate_per_capita_supply(annual_rwh_liters, population):
    """
    Calculate per-capita daily water supply.
    """
    pop_safe = np.where(population > 0, population, np.nan)
    annual_per_capita = annual_rwh_liters / pop_safe
    daily_per_capita = annual_per_capita / 365.0
    return daily_per_capita


def calculate_reliability(daily_rwh, daily_demand, storage_capacity=0):
    """
    Calculate reliability as the percentage of days demand is met.
    """
    met_demand = daily_rwh >= daily_demand
    reliability = np.mean(met_demand)
    return reliability
