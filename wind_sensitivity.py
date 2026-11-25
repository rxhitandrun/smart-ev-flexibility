"""Evaluate how installed wind capacity affects EV reserve and system cost.

The wind profile and other system assumptions remain fixed across cases.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import ev_operating_reserves as model

# Only wind nameplate capacity changes across experiments.
WIND_CAPACITIES_MW = [80, 120, 160, 200, 240]
EV_COMPENSATION_EUR_PER_MW_HOUR = 3.0

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

# Preserve the imported module's original settings after the experiment.
original_wind_capacity = model.WIND_MW
original_ev_price = model.EV_RESERVE_PRICE

records = []

try:
    model.EV_RESERVE_PRICE = EV_COMPENSATION_EUR_PER_MW_HOUR
    for wind_mw in WIND_CAPACITIES_MW:
        print(f"\n===== WIND CAPACITY: {wind_mw} MW =====")
        model.WIND_MW = wind_mw

        # Both scenarios receive exactly the same wind capacity and inputs.
        print("Solving baseline: no EV reserve...")
        baseline_hourly, baseline = model.run_scenario(
            ev_reserve_enabled=False
        )
        print("Solving EV-reserve scenario...")
        ev_hourly, with_ev = model.run_scenario(
            ev_reserve_enabled=True
        )

        # Available wind energy over 24 one-hour snapshots (MWh).
        available_wind_mwh = wind_mw * sum(model.wind_cf)

        # The objective already includes generation, charging-link throughput,
        # and reserve-availability costs (including the EV compensation).
        savings_eur = baseline["Total_Cost_EUR"] - with_ev["Total_Cost_EUR"]

        # EVs can only be another reserve option. With identical physical
        # inputs and a cost-minimising model, permitting them cannot increase
        # the optimal total cost beyond numerical tolerances.
        assert savings_eur >= -1e-4, (
            f"EV option unexpectedly raises optimal cost at {wind_mw} MW"
        )

        # Store identical metrics for the two operating cases.
        for case_name, hourly, summary in [
            ("No EV reserve", baseline_hourly, baseline),
            ("With EV reserve", ev_hourly, with_ev),
        ]:
            wind_used_mwh = float(hourly["wind"].sum())
            wind_curtailed_mwh = max(0.0, available_wind_mwh - wind_used_mwh)
            # Net demand seen by generators: normal demand plus EV charge
            total_grid_demand_mwh = (
                sum(model.normal_demand)
                + float(hourly["EV_Charging_MW"].sum())
                - float(hourly["V2G_Export_MW"].sum())
            )
            records.append({
                "Wind_Capacity_MW": wind_mw,
                "Scenario": case_name,
                "EV_Reserve_Payment_Rate_EUR_per_MW_hour": model.EV_RESERVE_PRICE,
                "Wind_Available_MWh": available_wind_mwh,
                "Wind_Used_MWh": wind_used_mwh,
                "Wind_Curtailed_MWh": wind_curtailed_mwh,
                "Wind_Utilisation_Percent": 100 * wind_used_mwh / available_wind_mwh,
                "Wind_Share_of_Grid_Supply_Percent": 100 * wind_used_mwh / total_grid_demand_mwh,
                "Hydro_MWh": float(hourly["hydro"].sum()),
                "Backup_Generation_MWh": float(hourly["backup"].sum()),
                "EV_Charging_MWh": float(hourly["EV_Charging_MW"].sum()),
                "V2G_Export_MWh": float(hourly["V2G_Export_MW"].sum()),
                "EV_Reserve_MW_hours": summary["EV_Reserve_MW_hours"],
                "Backup_Reserve_MW_hours": summary["Backup_Reserve_MW_hours"],
                "Reserve_Cost_EUR": summary["Reserve_Cost_EUR"],
                "Total_System_Cost_EUR": summary["Total_Cost_EUR"],
                "Savings_vs_No_EV_EUR": savings_eur if case_name == "With EV reserve" else 0.0,
            })

            # Save the entire hourly dispatch and reserve schedule.
            label = "no_ev" if case_name == "No EV reserve" else "with_ev"
            hourly.to_csv(OUT / f"lesson15_wind_{wind_mw}_{label}_hourly.csv")

        print(f"At {wind_mw} MW wind: EV reserve = "
              f"{with_ev['EV_Reserve_MW_hours']:.2f} MW-hours; "
              f"system savings = EUR {savings_eur:,.2f}")

finally:
    model.WIND_MW = original_wind_capacity
    model.EV_RESERVE_PRICE = original_ev_price

results = pd.DataFrame(records)
results.to_csv(OUT / "wind_sensitivity_detailed.csv", index=False)

# Build compact 5-row comparison across wind capacities.
base = results[results["Scenario"] == "No EV reserve"].set_index("Wind_Capacity_MW")
ev = results[results["Scenario"] == "With EV reserve"].set_index("Wind_Capacity_MW")
comparison = pd.DataFrame({
    "Wind_Capacity_MW": WIND_CAPACITIES_MW,
    "Wind_Curtailed_With_EV_MWh": [ev.loc[w, "Wind_Curtailed_MWh"] for w in WIND_CAPACITIES_MW],
    "Wind_Used_With_EV_MWh": [ev.loc[w, "Wind_Used_MWh"] for w in WIND_CAPACITIES_MW],
    "Hydro_With_EV_MWh": [ev.loc[w, "Hydro_MWh"] for w in WIND_CAPACITIES_MW],
    "Backup_Generation_With_EV_MWh": [ev.loc[w, "Backup_Generation_MWh"] for w in WIND_CAPACITIES_MW],
    "EV_Reserve_MW_hours": [ev.loc[w, "EV_Reserve_MW_hours"] for w in WIND_CAPACITIES_MW],
    "Baseline_Cost_EUR": [base.loc[w, "Total_System_Cost_EUR"] for w in WIND_CAPACITIES_MW],
    "With_EV_Cost_EUR": [ev.loc[w, "Total_System_Cost_EUR"] for w in WIND_CAPACITIES_MW],
    "Savings_EUR": [ev.loc[w, "Savings_vs_No_EV_EUR"] for w in WIND_CAPACITIES_MW],
})
comparison.to_csv(OUT / "wind_sensitivity_comparison.csv", index=False)

print("\nWIND CAPACITY SENSITIVITY\n")
print(comparison.round(2).to_string(index=False))

# Check the previously observed EUR 3/MW-hour, 160 MW benchmark.
benchmark = comparison.loc[comparison["Wind_Capacity_MW"] == 160].iloc[0]
print("\n160 MW benchmark, at EUR 3/MW/hour EV reserve compensation:")
print(f"EV reserve: {benchmark['EV_Reserve_MW_hours']:.2f} MW-hours")
print(f"Savings: EUR {benchmark['Savings_EUR']:.2f}")
if (abs(benchmark["EV_Reserve_MW_hours"] - 20) > 0.1
        or abs(benchmark["Savings_EUR"] - 34.19) > 0.1):
    print("NOTE: Results differ from . Check whether model inputs changed.")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(comparison["Wind_Capacity_MW"], comparison["EV_Reserve_MW_hours"], marker="o")
ax.set(xlabel="Installed wind capacity (MW)",
       ylabel="EV reserve availability (MW-hours)",
       title="EV reserve commitment versus wind capacity")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "lesson15_ev_reserve_vs_wind.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(comparison["Wind_Capacity_MW"], comparison["Savings_EUR"], marker="o")
ax.axhline(0, linestyle="--", linewidth=1)
ax.set(xlabel="Installed wind capacity (MW)", ylabel="Savings (EUR/day)",
       title="Value of EV reserve versus wind capacity")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "lesson15_savings_vs_wind.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(comparison["Wind_Capacity_MW"], comparison["Wind_Curtailed_With_EV_MWh"], marker="o")
ax.set(xlabel="Installed wind capacity (MW)", ylabel="Wind curtailed (MWh/day)",
       title="Wind curtailment versus installed wind capacity")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "lesson15_curtailment_vs_wind.png", dpi=200)
plt.close(fig)

print("\nSaved detailed CSVs, 5-row comparison and three plots in results/.")
print("Reminder: this is an hourly synthetic reserve-capacity model, not a frequency-dynamics simulation.")
