"""Audit reserve schedules across activation durations and requirements.

Checks hourly feasibility for 15-, 30- and 60-minute assumed activations.
It does not verify dynamic response or repeated-event recovery.
"""

from pathlib import Path
import math

import pandas as pd
import matplotlib.pyplot as plt

import ev_operating_reserves as model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

# Exact combinations: 3 x 3 x 2 = 18 optimisations.
DURATIONS_HOURS = [0.25, 0.50, 1.00]
RESERVE_REQUIREMENTS_MW = [10.0, 15.0, 20.0]
BASE_WIND_CAPACITY_MW = 160.0
BASE_EV_COMPENSATION_EUR_PER_MW_HOUR = 3.0
TOL = 1e-4

ORIGINAL = {
    "RESERVE_HOURS": model.RESERVE_HOURS,
    "RESERVE_MW": model.RESERVE_MW,
    "HYDRO_RESERVE_RESPONSE_MW": model.HYDRO_RESERVE_RESPONSE_MW,
    "WIND_MW": model.WIND_MW,
    "EV_RESERVE_PRICE": model.EV_RESERVE_PRICE,
}


def set_assumptions(duration_hours: float, reserve_requirement_mw: float):
    """Change reserve scenario only, leaving EV fleet and profiles unchanged."""
    model.RESERVE_HOURS = float(duration_hours)
    model.RESERVE_MW = float(reserve_requirement_mw)
    model.WIND_MW = BASE_WIND_CAPACITY_MW
    model.EV_RESERVE_PRICE = BASE_EV_COMPENSATION_EUR_PER_MW_HOUR
    # Hydro's assumed reachable response is ramp rate times allowed time.
    model.HYDRO_RESERVE_RESPONSE_MW = min(
        model.HYDRO_MW,
        model.HYDRO_MW * model.HYDRO_RAMP_PU_PER_HOUR * duration_hours,
    )


def audit_hourly(hourly: pd.DataFrame, ev_reserve_enabled: bool):
    """Independent post-solve checks, using hourly CSV-like output only."""
    h = hourly
    ev_r = h["EV_Reserve_MW"]
    hydro_r = h["Hydro_Reserve_MW"]
    backup_r = h["Backup_Reserve_MW"]
    all_r = hydro_r + backup_r + ev_r

    # The model's EV charging is grid-side MW. V2G_Export_MW is GRID-side
    # MW, so divide by discharge efficiency to obtain battery-side MW.
    charging = h["EV_Charging_MW"]
    v2g_battery_side = h["V2G_Export_MW"] / model.ETA_DISCHARGE
    reserve_battery_power = ev_r / model.ETA_DISCHARGE
    activation_battery_energy = reserve_battery_power * model.RESERVE_HOURS

    demand = pd.Series(model.normal_demand, index=h.index, dtype=float)
    production = h["wind"] + h["hydro"] + h["backup"]
    grid_balance = production + h["V2G_Export_MW"] - charging - demand

    # Reserve reserve requirement and physical headroom.
    residuals = {
        "hourly_grid_balance_MW": float(grid_balance.abs().max()),
        "reserve_shortfall_MW": float((model.RESERVE_MW - all_r).clip(lower=0).max()),
        "hydro_headroom_violation_MW": float(
            (h["hydro"] + hydro_r - model.HYDRO_MW).clip(lower=0).max()),
        "hydro_response_violation_MW": float(
            (hydro_r - model.HYDRO_RESERVE_RESPONSE_MW).clip(lower=0).max()),
        "backup_headroom_violation_MW": float(
            (h["backup"] + backup_r - model.BACKUP_MW).clip(lower=0).max()),
        "shared_converter_violation_MW": float(
            (charging + v2g_battery_side + reserve_battery_power
             - model.EV_POWER_MW).clip(lower=0).max()),
        "battery_start_reserve_margin_violation_MWh": float(
            (model.EV_MIN_MWH + activation_battery_energy
             - h["Battery_Start_MWh"]).clip(lower=0).max()),
        "battery_end_reserve_margin_violation_MWh": float(
            (model.EV_MIN_MWH + activation_battery_energy
             - h["Battery_End_MWh"]).clip(lower=0).max()),
        "departure_one_activation_violation_MWh": float(
            max(0.0, model.EV_DEPARTURE_MWH + activation_battery_energy.max()
                - float(h["Battery_End_MWh"].iloc[21]))),
        "hydro_energy_budget_violation_MWh": float(
            max(0.0, h["hydro"].sum() - model.HYDRO_BUDGET_MWH)),
        "hydro_hourly_ramp_violation_MW": float(
            (h["hydro"].diff().abs().dropna()
             - model.HYDRO_MW * model.HYDRO_RAMP_PU_PER_HOUR)
            .clip(lower=0).max()),
        "ev_reserve_outside_window_MW": float(
            ev_r[model.reserve_eligible.to_numpy() == 0].abs().max()),
    }
    if not ev_reserve_enabled:
        residuals["ev_reserve_when_disabled_MW"] = float(ev_r.abs().max())

    bad = {k: v for k, v in residuals.items() if not math.isfinite(v) or v > TOL}
    if bad:
        raise AssertionError(f"Post-solve reserve audit FAILED: {bad}")
    return residuals


def simulate_all():
    scenario_rows = []
    audit_rows = []
    try:
        for duration in DURATIONS_HOURS:
            for required_mw in RESERVE_REQUIREMENTS_MW:
                set_assumptions(duration, required_mw)
                print("\n" + "=" * 64)
                print(f"ACTIVATION DURATION: {duration*60:.0f} minutes; "
                      f"RESERVE REQUIREMENT: {required_mw:.0f} MW", flush=True)

                # Solve two MATCHED cases with the same inputs.
                print("Baseline: no EV reserve...", flush=True)
                baseline_hourly, baseline_summary = model.run_scenario(False)
                baseline_audit = audit_hourly(baseline_hourly, False)

                print("Allow EV reserve...", flush=True)
                ev_hourly, ev_summary = model.run_scenario(True)
                ev_audit = audit_hourly(ev_hourly, True)

                # Longer duration can be infeasible in some energy systems:
                # if so, the optimiser raises an informative exception.
                saving_eur = (baseline_summary["Total_Cost_EUR"]
                              - ev_summary["Total_Cost_EUR"])
                if saving_eur < -TOL:
                    raise AssertionError("Allowing a reserve option raised optimum cost")

                ev_commitment = float(ev_summary["EV_Reserve_MW_hours"])
                peak_ev_r = float(ev_hourly["EV_Reserve_MW"].max())
                # This reports battery draw IF the largest MW commitment
                # is fully activated once for the assumed duration.
                worst_single_draw = peak_ev_r * duration / model.ETA_DISCHARGE
                scenario_rows.append({
                    "Reserve_Duration_Minutes": int(round(duration * 60)),
                    "Reserve_Requirement_MW": required_mw,
                    "Hydro_Assumed_Response_MW": model.HYDRO_RESERVE_RESPONSE_MW,
                    "EV_Reserve_Commitment_MW_hours": ev_commitment,
                    "EV_Peak_Reserve_MW": peak_ev_r,
                    "Max_Single_Activation_Battery_Draw_MWh": worst_single_draw,
                    "Hydro_Reserve_Commitment_MW_hours": float(
                        ev_hourly["Hydro_Reserve_MW"].sum()),
                    "Backup_Reserve_Commitment_MW_hours": float(
                        ev_hourly["Backup_Reserve_MW"].sum()),
                    "EV_Availability_Payment_EUR": ev_commitment * model.EV_RESERVE_PRICE,
                    "No_EV_Reserve_Total_Cost_EUR": float(baseline_summary["Total_Cost_EUR"]),
                    "With_EV_Reserve_Total_Cost_EUR": float(ev_summary["Total_Cost_EUR"]),
                    "Savings_EUR": float(saving_eur),
                    "Battery_Departure_MWh": float(ev_summary["Battery_At_Departure_MWh"]),
                    "Audit": "PASS",
                })

                for name, checks in (("no_ev", baseline_audit), ("with_ev", ev_audit)):
                    audit_rows.append({
                        "Reserve_Duration_Minutes": int(round(duration * 60)),
                        "Reserve_Requirement_MW": required_mw,
                        "Scenario": name,
                        **checks,
                    })

                prefix = (f"lesson17_duration_{int(round(duration*60))}min"
                          f"_requirement_{int(required_mw)}MW")
                baseline_hourly.to_csv(OUT / f"{prefix}_without_ev.csv")
                ev_hourly.to_csv(OUT / f"{prefix}_with_ev.csv")
                print(f"EV reserve commitment: {ev_commitment:.2f} MW-hours; "
                      f"saving: EUR {saving_eur:.2f}; audit PASS")

        results = pd.DataFrame(scenario_rows)
        audits = pd.DataFrame(audit_rows)
        results.to_csv(OUT / "reserve_validation_summary.csv", index=False)
        audits.to_csv(OUT / "lesson17_independent_constraint_audit.csv", index=False)

        print("\nRESERVE DURATION AND REQUIREMENT VALIDATION\n")
        display_cols = [
            "Reserve_Duration_Minutes", "Reserve_Requirement_MW",
            "EV_Reserve_Commitment_MW_hours",
            "EV_Peak_Reserve_MW", "Savings_EUR", "Audit",
        ]
        print(results[display_cols].round(3).to_string(index=False))

        benchmark = results.loc[
            (results["Reserve_Duration_Minutes"] == 15)
            & (results["Reserve_Requirement_MW"] == 15)
        ].iloc[0]
        is_benchmark = (
            math.isclose(benchmark["EV_Reserve_Commitment_MW_hours"], 20.0, abs_tol=0.1)
            and math.isclose(benchmark["Savings_EUR"], 34.19, abs_tol=0.1)
        )
        print("\nREFERENCE BENCHMARK (15 minutes, 15 MW):")
        print(f"EV reserve: {benchmark['EV_Reserve_Commitment_MW_hours']:.2f} MW-hours")
        print(f"Savings: EUR {benchmark['Savings_EUR']:.2f}")
        print(f"Benchmark: {'PASS' if is_benchmark else 'CHECK INPUTS'}")

        # One chart per metric, avoiding confusion between MW and MW-hours.
        for key, title, ylabel, filename in [
            ("EV_Reserve_Commitment_MW_hours", "EV reserve commitment",
             "Reserved capacity x time (MW-hours)", "lesson17_ev_reserve_sensitivity.png"),
            ("Savings_EUR", "Cost savings with EV reserve",
             "Daily savings (EUR)", "lesson17_reserve_savings_sensitivity.png"),
        ]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for mins, grp in results.groupby("Reserve_Duration_Minutes"):
                ax.plot(grp["Reserve_Requirement_MW"], grp[key],
                        marker="o", label=f"{mins} min activation")
            ax.set(title=title, xlabel="Required upward reserve (MW)", ylabel=ylabel)
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUT / filename, dpi=180)
            plt.close(fig)
        print("\nSaved summary, hourly scenarios, independent audit and charts to results/.")
        print("IMPORTANT: Passing an hourly reserve audit does NOT establish "
              "frequency stability, response time or repeated-event feasibility.")
        if not is_benchmark:
            print("WARNING: Reference results changed. Check base assumptions.")
    finally:
        # Never leave modified parameter values in an imported module.
        for name, value in ORIGINAL.items():
            setattr(model, name, value)


if __name__ == "__main__":
    simulate_all()
