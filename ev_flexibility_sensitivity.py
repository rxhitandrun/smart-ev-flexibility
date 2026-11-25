"""Sensitivity to fleet size, initial battery energy and arrival time.

Fleet-size cases scale charging capacity and battery requirements together.
Each case is compared against its own no-EV-reserve baseline. All inputs
are synthetic.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import pandas as pd

import ev_operating_reserves as model

BASE_FLEET_VEHICLES = 2000
FLEET_SIZES = [500, 1000, 2000, 3000, 4000]
INITIAL_SOC_FRACTIONS = [0.25, 0.40, 0.50]  # 25%, 40%, 50% of usable capacity
ARRIVAL_HOURS = [16, 17, 18]              # departure remains at 22:00
COMPENSATION_EUR_PER_MW_HOUR = 3.0
WIND_CAPACITY_MW = 160

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Save references/values of every module setting we're about to change.
ORIGINAL = {
    "WIND_MW": model.WIND_MW,
    "EV_RESERVE_PRICE": model.EV_RESERVE_PRICE,
    "EV_POWER_MW": model.EV_POWER_MW,
    "EV_CAPACITY_MWH": model.EV_CAPACITY_MWH,
    "EV_INITIAL_MWH": model.EV_INITIAL_MWH,
    "EV_MIN_MWH": model.EV_MIN_MWH,
    "EV_DEPARTURE_MWH": model.EV_DEPARTURE_MWH,
    "connected": model.connected.copy(),
    "reserve_eligible": model.reserve_eligible.copy(),
    "store_lower_bound": model.store_lower_bound.copy(),
}


def restore_base_model():
    """Return to the exact initial imported module settings."""
    for parameter, original_value in ORIGINAL.items():
        if isinstance(original_value, pd.Series):
            setattr(model, parameter, original_value.copy())
        else:
            setattr(model, parameter, original_value)
    model.WIND_MW = WIND_CAPACITY_MW
    model.EV_RESERVE_PRICE = COMPENSATION_EUR_PER_MW_HOUR


def update_battery_lower_bound():
    """Rebuild Store's per-unit lower bound after changing battery assumptions.

    The corrected model creates this pandas Series at import time.
    Changing EV_CAPACITY_MWH or EV_MIN_MWH alone would leave it STALE.
    The first 21 snapshots require the protected minimum; hour 21 onward
    requires the departure target because departure is at 22:00.
    """
    if not (0 <= model.EV_MIN_MWH <= model.EV_INITIAL_MWH
            <= model.EV_CAPACITY_MWH):
        raise ValueError("Invalid starting SoC or battery minimum/capacity.")
    if not (model.EV_MIN_MWH <= model.EV_DEPARTURE_MWH
            <= model.EV_CAPACITY_MWH):
        raise ValueError("Invalid departure energy target.")
    bound = pd.Series(model.EV_MIN_MWH / model.EV_CAPACITY_MWH,
                      index=model.hours, dtype=float)
    bound.iloc[21:] = model.EV_DEPARTURE_MWH / model.EV_CAPACITY_MWH
    model.store_lower_bound = bound


def run_pair(experiment, value, label):
    """Solve identical inputs with and without the EV reserve option."""
    print(f"\n===== {experiment}: {label} =====", flush=True)
    print("Solving matching no-EV-reserve baseline...", flush=True)
    baseline_hourly, baseline = model.run_scenario(ev_reserve_enabled=False)
    print("Solving EV-reserve-enabled case...", flush=True)
    enabled_hourly, enabled = model.run_scenario(ev_reserve_enabled=True)

    saving = float(baseline["Total_Cost_EUR"] - enabled["Total_Cost_EUR"])
    if saving < -1e-4:
        raise AssertionError(f"EV option raised optimal costs for {label}: {saving}")

    # Both scenarios must meet the same fleet's charging and departure target.
    for hourly in (baseline_hourly, enabled_hourly):
        assert hourly["Battery_End_MWh"].min() >= model.EV_MIN_MWH - 1e-5
        assert hourly["Battery_End_MWh"].max() <= model.EV_CAPACITY_MWH + 1e-5
        assert hourly["Battery_End_MWh"].iloc[21] >= model.EV_DEPARTURE_MWH - 1e-5

    row = {
        "Experiment": experiment,
        "Value": value,
        "Description": label,
        "EV_Fleet_Vehicles": round(BASE_FLEET_VEHICLES * model.EV_POWER_MW /
                                    ORIGINAL["EV_POWER_MW"]),
        "EV_Power_MW": model.EV_POWER_MW,
        "EV_Usable_Capacity_MWh": model.EV_CAPACITY_MWH,
        "EV_Initial_MWh": model.EV_INITIAL_MWH,
        "EV_Initial_SOC_Percent": 100 * model.EV_INITIAL_MWH / model.EV_CAPACITY_MWH,
        "EV_Minimum_MWh": model.EV_MIN_MWH,
        "EV_Departure_Target_MWh": model.EV_DEPARTURE_MWH,
        "Connected_From_Hour": int(model.connected[model.connected > 0].index[0].hour),
        "Reserve_Compensation_EUR_per_MW_hour": model.EV_RESERVE_PRICE,
        "Baseline_Cost_EUR": float(baseline["Total_Cost_EUR"]),
        "With_EV_Reserve_Cost_EUR": float(enabled["Total_Cost_EUR"]),
        "Savings_EUR": saving,
        "EV_Reserve_MW_hours": float(enabled["EV_Reserve_MW_hours"]),
        "Backup_Reserve_MW_hours": float(enabled["Backup_Reserve_MW_hours"]),
        "EV_Availability_Payment_EUR": float(enabled["EV_Reserve_MW_hours"] *
                                              model.EV_RESERVE_PRICE),
        "With_EV_Charging_MWh": float(enabled["EV_Charging_MWh"]),
        "With_EV_V2G_Export_MWh": float(enabled["V2G_Export_MWh"]),
        "Battery_At_Departure_MWh": float(enabled["Battery_At_Departure_MWh"]),
        "Hydro_With_EV_MWh": float(enabled_hourly["hydro"].sum()),
        "Backup_Generation_With_EV_MWh": float(enabled_hourly["backup"].sum()),
    }
    print(f"EV reserve: {row['EV_Reserve_MW_hours']:.2f} MW-hours | "
          f"savings: EUR {saving:.2f} | "
          f"departure energy: {row['Battery_At_Departure_MWh']:.2f} MWh")

    # Record dispatch for inspection and additional checks.
    # Use stable filenames, no spaces or decimals from floating-point values.
    tag = (str(value).replace(".", "p")
           if not isinstance(value, int) else str(value))
    name_prefix = f"lesson16_{experiment}_{tag}"
    baseline_hourly.to_csv(OUTPUT / f"{name_prefix}_baseline_hourly.csv")
    enabled_hourly.to_csv(OUTPUT / f"{name_prefix}_ev_reserve_hourly.csv")
    return row


def save_plot(frame, x_field, filename, title, xlabel):
    """One clean chart per result, with no assumptions about the outcome."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frame[x_field], frame["EV_Reserve_MW_hours"], marker="o",
            label="EV reserve committed (MW-hours)")
    ax.set(xlabel=xlabel, ylabel="EV reserve commitment (MW-hours)", title=title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=180)
    plt.close(fig)


def save_savings_plot(frame, x_field, filename, title, xlabel):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(frame[x_field], frame["Savings_EUR"], marker="o")
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set(xlabel=xlabel, ylabel="Daily system-cost savings (EUR)", title=title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / filename, dpi=180)
    plt.close(fig)


records = []
try:
    # EXPERIMENT A: Fleet size.
    # Per-vehicle power, usable battery energy, minimum, starting energy,
    # and required departure energy are held constant via proportional scaling.
    # the same 20 MWh of aggregate charging as 2,000 vehicles.
    for fleet in FLEET_SIZES:
        restore_base_model()
        scale = fleet / BASE_FLEET_VEHICLES
        model.EV_POWER_MW = ORIGINAL["EV_POWER_MW"] * scale
        model.EV_CAPACITY_MWH = ORIGINAL["EV_CAPACITY_MWH"] * scale
        model.EV_INITIAL_MWH = ORIGINAL["EV_INITIAL_MWH"] * scale
        model.EV_MIN_MWH = ORIGINAL["EV_MIN_MWH"] * scale
        model.EV_DEPARTURE_MWH = ORIGINAL["EV_DEPARTURE_MWH"] * scale
        update_battery_lower_bound()
        records.append(run_pair("fleet_size", fleet, f"{fleet:,} vehicles"))

    # EXPERIMENT B: Initial charge, at exactly the same 2,000-vehicle scale.
    # Protected minimum stays at 25% and departure target at 75%.
    # Charging ENERGY NEED is therefore different between cases; compare
    # ONLY enabled vs baseline within each SoC scenario.
    for fraction in INITIAL_SOC_FRACTIONS:
        restore_base_model()
        model.EV_INITIAL_MWH = model.EV_CAPACITY_MWH * fraction
        update_battery_lower_bound()
        records.append(run_pair("initial_soc", int(round(100 * fraction)),
                                f"Initial usable battery SoC: {100*fraction:.0f}%"))

    # EXPERIMENT C: Connection availability, on the 2,000-vehicle base case.
    # Departure stays at 22:00. Reserve eligible only until 20:59, not
    # the final charging hour 21:00-21:59.
    for arrival in ARRIVAL_HOURS:
        restore_base_model()
        model.connected = pd.Series(0.0, index=model.hours)
        model.connected.iloc[arrival:22] = 1.0
        model.reserve_eligible = pd.Series(0.0, index=model.hours)
        model.reserve_eligible.iloc[arrival:21] = 1.0
        records.append(run_pair("arrival_hour", arrival,
                                f"Connected from {arrival:02d}:00 to 22:00"))
finally:
    # Restore everything, even if a solver run fails midway.
    for parameter, original_value in ORIGINAL.items():
        if isinstance(original_value, pd.Series):
            setattr(model, parameter, original_value.copy())
        else:
            setattr(model, parameter, original_value)

results = pd.DataFrame(records)
results.to_csv(OUTPUT / "lesson16_ev_flexibility_detailed.csv", index=False)

columns = ["Description", "EV_Reserve_MW_hours", "Savings_EUR",
           "With_EV_Charging_MWh", "Battery_At_Departure_MWh"]
for experiment, heading in [
    ("fleet_size", "A. FLEET SIZE SENSITIVITY"),
    ("initial_soc", "B. INITIAL BATTERY CHARGE SENSITIVITY"),
    ("arrival_hour", "C. CONNECTION TIME SENSITIVITY"),
]:
    view = results.loc[results["Experiment"] == experiment]
    view.to_csv(OUTPUT / f"lesson16_{experiment}_comparison.csv", index=False)
    print(f"\n{heading}\n")
    print(view[columns].round(2).to_string(index=False))

# The same 2,000-car, initial-25%, 16:00 connection configuration
benchmarks = results.loc[
    ((results.Experiment == "fleet_size") & (results.Value == 2000))
    | ((results.Experiment == "initial_soc") & (results.Value == 25))
    | ((results.Experiment == "arrival_hour") & (results.Value == 16))
]
print("\n160 MW WIND / EUR 3 COMPENSATION BENCHMARK")
for _, row in benchmarks.iterrows():
    print(f"{row['Description']}: reserve "
          f"{row['EV_Reserve_MW_hours']:.2f} MW-hours; "
          f"saving EUR {row['Savings_EUR']:.2f}")
    if (not math.isclose(row["EV_Reserve_MW_hours"], 20.0, abs_tol=0.1)
            or not math.isclose(row["Savings_EUR"], 34.19, abs_tol=0.1)):
        print("WARNING: Benchmark differs from /15. Check inputs.")

fleet = results[results.Experiment == "fleet_size"].copy()
soc = results[results.Experiment == "initial_soc"].copy()
arrival = results[results.Experiment == "arrival_hour"].copy()

save_plot(fleet, "Value", "lesson16_fleet_vs_reserve.png",
          "Reserve commitment versus number of EVs", "Connected vehicles")
save_savings_plot(fleet, "Value", "lesson16_fleet_vs_savings.png",
                  "EV reserve savings versus fleet size", "Connected vehicles")
save_plot(soc, "Value", "lesson16_initial_soc_vs_reserve.png",
          "Reserve commitment versus initial battery charge", "Initial usable battery SoC (%)")
save_savings_plot(soc, "Value", "lesson16_initial_soc_vs_savings.png",
                  "EV reserve savings versus initial SoC", "Initial usable battery SoC (%)")
save_plot(arrival, "Value", "lesson16_arrival_vs_reserve.png",
          "Reserve commitment versus EV connection time", "First connected hour (24-hour clock)")
save_savings_plot(arrival, "Value", "lesson16_arrival_vs_savings.png",
                  "EV reserve savings versus connection time", "First connected hour (24-hour clock)")

print("\nSaved 3 comparisons, complete results, individual hourly schedules and 6 PNG plots in results/.")
print("No Icelandic measurement data were used. No frequency-response simulation was performed.")
