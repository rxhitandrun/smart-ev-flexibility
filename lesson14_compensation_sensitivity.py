"""Compare EV reserve participation under synthetic availability payments.

Each scenario uses the same physical system and a matched no-EV baseline.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import ev_operating_reserves_corrected as model

# Assumed fleet/aggregator availability compensation in EUR per MW per hour.
RATES_EUR_PER_MW_HOUR = [1.0, 3.0, 5.0, 10.0]
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# The baseline cannot sell EV reserve, so changing the EV payment cannot affect it.
print("\nSolving the no-EV-reserve baseline...")
_, baseline = model.run_scenario(ev_reserve_enabled=False)
baseline_cost = baseline["Total_Cost_EUR"]

records = []
for rate in RATES_EUR_PER_MW_HOUR:
    print(f"\nSolving EV reserve scenario at EUR {rate:g}/MW/hour...")
    model.EV_RESERVE_PRICE = rate
    hourly, summary = model.run_scenario(ev_reserve_enabled=True)

    # Hourly snapshots: sum of reserve MW = MW-hours of capacity commitment.
    ev_reserve_mw_hours = summary["EV_Reserve_MW_hours"]
    ev_availability_payment = rate * ev_reserve_mw_hours
    savings = baseline_cost - summary["Total_Cost_EUR"]

    # Verify payment is based on exactly what the solver committed.
    assert abs(
        ev_reserve_mw_hours - hourly["EV_Reserve_MW"].sum()
    ) < 1e-5

    records.append({
        "EV_Payment_Rate_EUR_per_MW_hour": rate,
        "EV_Reserve_MW_hours": ev_reserve_mw_hours,
        "Backup_Reserve_MW_hours": summary["Backup_Reserve_MW_hours"],
        "EV_Availability_Payment_EUR": ev_availability_payment,
        "Reserve_Cost_EUR": summary["Reserve_Cost_EUR"],
        "Total_System_Cost_EUR": summary["Total_Cost_EUR"],
        "Savings_vs_No_EV_EUR": savings,
        "EV_Charging_MWh": summary["EV_Charging_MWh"],
        "V2G_Export_MWh": summary["V2G_Export_MWh"],
    })

    # Keep an hourly trace for each rate, useful when interpreting results.
    rate_tag = str(rate).replace(".", "p")
    hourly.to_csv(RESULTS_DIR / f"lesson14_hourly_rate_{rate_tag}.csv")

results = pd.DataFrame(records)
results.to_csv(RESULTS_DIR / "lesson14_compensation_sensitivity.csv", index=False)

print("\nDRIVER/AGGREGATOR COMPENSATION SENSITIVITY\n")
print(f"No-EV-reserve baseline total cost: EUR {baseline_cost:,.2f}\n")
print(results.round(3).to_string(index=False))

# The EUR 3 case should reproduce the previous corrected model's cost,
# provided model parameters and file contents have not changed.
three_euro = results.loc[
    results["EV_Payment_Rate_EUR_per_MW_hour"] == 3.0
].iloc[0]
print("\nBenchmark check at EUR 3/MW/hour:")
print(f"EV reserve: {three_euro['EV_Reserve_MW_hours']:.2f} MW-hours")
print(f"System savings: EUR {three_euro['Savings_vs_No_EV_EUR']:.2f}")
print("Expected approximately 20 MW-hours and EUR 34.19, if the corrected model is unchanged.")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(
    results["EV_Payment_Rate_EUR_per_MW_hour"],
    results["EV_Reserve_MW_hours"],
    marker="o",
)
ax.set(
    title="EV Reserve Participation vs Compensation",
    xlabel="EV compensation (EUR per MW per hour)",
    ylabel="EV reserve capacity commitments (MW-hours)",
)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "lesson14_reserve_vs_compensation.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(
    results["EV_Payment_Rate_EUR_per_MW_hour"],
    results["Savings_vs_No_EV_EUR"],
    marker="o",
)
ax.axhline(0, linestyle="--", linewidth=1)
ax.set(
    title="System Cost Savings vs EV Compensation",
    xlabel="EV compensation (EUR per MW per hour)",
    ylabel="Savings relative to no EV reserve (EUR/day)",
)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "lesson14_savings_vs_compensation.png", dpi=200)
plt.close(fig)

print("\nSaved tables, hourly schedules and plots to results/.")
