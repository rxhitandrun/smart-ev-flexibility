"""Hourly hydro-wind dispatch with EV charging, V2G and upward reserve.

Uses synthetic data. Reserve feasibility is modelled through power and
battery-energy constraints, not through dynamic frequency simulation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pypsa
import xarray as xr

# 24 one-hour snapshots. All MW sums over these snapshots are MWh.
hours = pd.date_range("2026-01-01", periods=24, freq="h")
normal_demand = [
    65, 60, 58, 56, 55, 60, 70, 85, 95, 100, 105, 108,
    110, 108, 105, 100, 105, 115, 120, 115, 105, 90, 80, 70,
]
wind_cf = [
    .70, .75, .78, .80, .72, .65, .60, .55, .50, .40, .35, .30,
    .25, .20, .25, .35, .45, .55, .65, .70, .75, .80, .78, .72,
]

# Generator specifications: capacities in MW; costs in EUR/MWh.
WIND_MW = 160
HYDRO_MW = 100
BACKUP_MW = 100
HYDRO_BUDGET_MWH = 900
HYDRO_RAMP_PU_PER_HOUR = .20
HYDRO_COST = 20
BACKUP_COST = 200

# Aggregate EV-fleet assumptions, not data measured in Iceland.
EV_POWER_MW = 14
EV_CAPACITY_MWH = 40
EV_INITIAL_MWH = 10
EV_MIN_MWH = 10
EV_DEPARTURE_MWH = 30
ETA_CHARGE = .95
ETA_DISCHARGE = .95
LINK_THROUGHPUT_EUR_PER_MWH = .05

# Reserve parameters: assumed 15-minute delivery capability.
RESERVE_MW = 15
RESERVE_HOURS = .25
HYDRO_RESERVE_RESPONSE_MW = HYDRO_MW * HYDRO_RAMP_PU_PER_HOUR * RESERVE_HOURS
HYDRO_RESERVE_PRICE = .10   # EUR per MW held for one hour
BACKUP_RESERVE_PRICE = 5.00
EV_RESERVE_PRICE = 3.00     # Illustrative driver/aggregator allowance

# Connected during 16:00 ... 21:59, departure at 22:00.
connected = pd.Series(0., index=hours)
connected.iloc[16:22] = 1.
# Reserve may be offered during 16:00 ... 20:59.
reserve_eligible = pd.Series(0., index=hours)
reserve_eligible.iloc[16:21] = 1.

# PyPSA Store end-of-hour minimum energy (per unit of e_nom).
store_lower_bound = pd.Series(EV_MIN_MWH / EV_CAPACITY_MWH, index=hours)
store_lower_bound.iloc[21:] = EV_DEPARTURE_MWH / EV_CAPACITY_MWH


def run_scenario(ev_reserve_enabled: bool):
    """Build, solve and validate one scenario."""
    n = pypsa.Network()
    n.set_snapshots(hours)
    n.add("Carrier", "AC")
    n.add("Carrier", "EV")
    n.add("Bus", "grid", carrier="AC")
    n.add("Bus", "ev_bus", carrier="EV")
    n.add("Load", "normal_demand", bus="grid", p_set=normal_demand)
    n.add("Generator", "wind", bus="grid", p_nom=WIND_MW,
          p_max_pu=wind_cf, marginal_cost=0)
    n.add("Generator", "hydro", bus="grid", p_nom=HYDRO_MW,
          e_sum_max=HYDRO_BUDGET_MWH,
          ramp_limit_up=HYDRO_RAMP_PU_PER_HOUR,
          ramp_limit_down=HYDRO_RAMP_PU_PER_HOUR,
          marginal_cost=HYDRO_COST)
    n.add("Generator", "backup", bus="grid", p_nom=BACKUP_MW,
          marginal_cost=BACKUP_COST)

    # Separate links represent charging and discharging losses accurately.
    n.add("Link", "ev_charger", bus0="grid", bus1="ev_bus",
          p_nom=EV_POWER_MW, p_min_pu=0, p_max_pu=connected,
          efficiency=ETA_CHARGE,
          marginal_cost=LINK_THROUGHPUT_EUR_PER_MWH)
    n.add("Link", "ev_v2g", bus0="ev_bus", bus1="grid",
          p_nom=EV_POWER_MW, p_min_pu=0, p_max_pu=connected,
          efficiency=ETA_DISCHARGE,
          marginal_cost=LINK_THROUGHPUT_EUR_PER_MWH)
    n.add("Store", "ev_battery", bus="ev_bus",
          e_nom=EV_CAPACITY_MWH, e_initial=EV_INITIAL_MWH,
          e_cyclic=False, e_min_pu=store_lower_bound)

    # First create PyPSA's standard hourly dispatch/storage model.
    m = n.optimize.create_model(include_objective_constant=False)

    # drop=True removes scalar 'name' coordinates from selected components.
    gen_p = m.variables["Generator-p"]
    link_p = m.variables["Link-p"]
    store_e = m.variables["Store-e"]
    hydro_p = gen_p.sel(name="hydro", drop=True)
    backup_p = gen_p.sel(name="backup", drop=True)
    charging_p = link_p.sel(name="ev_charger", drop=True)
    v2g_battery_p = link_p.sel(name="ev_v2g", drop=True)
    battery_end = store_e.sel(name="ev_battery", drop=True)

    # Named time dimension must exactly match PyPSA's 'snapshot' dimension.
    coords = {"snapshot": n.snapshots}
    r_h = m.add_variables(lower=0, coords=coords, name="hydro_upward_reserve")
    r_b = m.add_variables(lower=0, coords=coords, name="backup_upward_reserve")
    r_ev = m.add_variables(lower=0, coords=coords, name="ev_upward_reserve")

    # Hydro and backup reserve must fit unused capacity.
    m.add_constraints(hydro_p + r_h <= HYDRO_MW,
                      name="hydro_reserve_headroom")
    m.add_constraints(r_h <= HYDRO_RESERVE_RESPONSE_MW,
                      name="hydro_reserve_response")
    m.add_constraints(backup_p + r_b <= BACKUP_MW,
                      name="backup_reserve_headroom")

    # Shared aggregate converter capacity: scheduled charging + scheduled
    # V2G battery-side power + newly committed reserve battery-side power.
    # This is deliberately conservative. Scheduled charging is assumed to
    # CONTINUE if reserve activates, so no interrupted-charging penalty.
    m.add_constraints(
        charging_p + v2g_battery_p + r_ev / ETA_DISCHARGE <= EV_POWER_MW,
        name="shared_ev_converter_capacity",
    )

    # EV power cannot be promised when vehicles are disconnected.
    # All three decision variables are 1-D and share 'snapshot'.
    limit = xr.DataArray(
        reserve_eligible.to_numpy() * EV_POWER_MW * ETA_DISCHARGE,
        dims=["snapshot"], coords=coords,
    )
    if not ev_reserve_enabled:
        limit = limit * 0
    m.add_constraints(r_ev <= limit, name="ev_reserve_availability")

    # The energy in the Store variable is end-of-hour energy.
    # The start value follows from the one-hour Store energy balance.
    # V2G input is battery-side power, so it requires no /ETA factor here.
    battery_start = battery_end - ETA_CHARGE * charging_p + v2g_battery_p

    # Reserve energy withdrawn from the battery in a 15-min activation.
    extra_battery_energy = r_ev * RESERVE_HOURS / ETA_DISCHARGE

    # Conservative start/end tests: a reserve call anywhere in the hour
    # must not draw the battery below its protected minimum.
    m.add_constraints(
        battery_start >= EV_MIN_MWH + extra_battery_energy,
        name="ev_reserve_start_energy",
    )
    m.add_constraints(
        battery_end >= EV_MIN_MWH + extra_battery_energy,
        name="ev_reserve_end_energy",
    )

    # Scenario-based departure protection for ONE full 15-minute activation
    # in ANY eligible hour. No recovery charging is credited.
    # Crucially, if r_ev == 0, this is simply the ordinary 30 MWh target.
    departure_end = battery_end.sel(snapshot=n.snapshots[21], drop=True)
    m.add_constraints(
        departure_end >= EV_DEPARTURE_MWH + extra_battery_energy,
        name="ev_departure_after_one_activation",
    )

    # Total upward reserve must be >= 15 MW every simulated hour.
    m.add_constraints(r_h + r_b + r_ev >= RESERVE_MW,
                      name="system_upward_reserve_requirement")

    # Charge capacity availability (one-hour commitments) is paid even
    # when no emergency activation occurs. These prices are assumptions.
    reserve_cost_expr = (
        HYDRO_RESERVE_PRICE * r_h.sum()
        + BACKUP_RESERVE_PRICE * r_b.sum()
        + EV_RESERVE_PRICE * r_ev.sum()
    )
    m.add_objective(m.objective.expression + reserve_cost_expr, overwrite=True)

    status, condition = n.optimize.solve_model(solver_name="highs")
    if (status, condition) != ("ok", "optimal"):
        raise RuntimeError(f"Optimisation failed: {status}, {condition}")

    generation = n.generators_t.p
    charging = n.links_t.p0["ev_charger"]
    discharge_from_battery = n.links_t.p0["ev_v2g"]
    export_to_grid = -n.links_t.p1["ev_v2g"]
    battery_soc = n.stores_t.e["ev_battery"]

    rh = pd.Series(r_h.solution.to_numpy(), index=hours)
    rb = pd.Series(r_b.solution.to_numpy(), index=hours)
    rev = pd.Series(r_ev.solution.to_numpy(), index=hours)
    rtotal = rh + rb + rev
    start_soc = battery_soc.shift(1)
    start_soc.iloc[0] = EV_INITIAL_MWH
    reserve_energy = rev * RESERVE_HOURS / ETA_DISCHARGE
    normal_load = pd.Series(normal_demand, index=hours, dtype=float)
    tol = 1e-5

    # Check power balance and all our main physical screening constraints.
    assert (generation.sum(axis=1) + export_to_grid - charging - normal_load).abs().max() < tol
    assert (rtotal >= RESERVE_MW - tol).all()
    assert (generation["hydro"] + rh <= HYDRO_MW + tol).all()
    assert (rh <= HYDRO_RESERVE_RESPONSE_MW + tol).all()
    assert (generation["backup"] + rb <= BACKUP_MW + tol).all()
    assert (charging + discharge_from_battery + rev / ETA_DISCHARGE
            <= EV_POWER_MW + tol).all()
    assert (start_soc >= EV_MIN_MWH + reserve_energy - tol).all()
    assert (battery_soc >= EV_MIN_MWH + reserve_energy - tol).all()
    assert (battery_soc <= EV_CAPACITY_MWH + tol).all()
    assert (battery_soc.iloc[21] >= EV_DEPARTURE_MWH + reserve_energy.max() - tol)
    assert rev.iloc[:16].abs().max() < tol
    assert rev.iloc[21:].abs().max() < tol
    if not ev_reserve_enabled:
        assert rev.abs().max() < tol
    assert generation["hydro"].sum() <= HYDRO_BUDGET_MWH + tol
    assert (generation["hydro"].diff().dropna().abs()
            <= HYDRO_MW * HYDRO_RAMP_PU_PER_HOUR + tol).all()

    generation_cost = (generation["hydro"].sum() * HYDRO_COST
                       + generation["backup"].sum() * BACKUP_COST)
    link_cost = ((charging.sum() + discharge_from_battery.sum())
                 * LINK_THROUGHPUT_EUR_PER_MWH)
    reserve_cost = (rh.sum() * HYDRO_RESERVE_PRICE
                    + rb.sum() * BACKUP_RESERVE_PRICE
                    + rev.sum() * EV_RESERVE_PRICE)
    assert abs(generation_cost + link_cost + reserve_cost - n.objective) < 1e-4

    hourly = generation.copy()
    hourly["EV_Charging_MW"] = charging
    hourly["V2G_Export_MW"] = export_to_grid
    hourly["Battery_Start_MWh"] = start_soc
    hourly["Battery_End_MWh"] = battery_soc
    hourly["EV_Reserve_MW"] = rev
    hourly["Hydro_Reserve_MW"] = rh
    hourly["Backup_Reserve_MW"] = rb
    hourly["Total_Reserve_MW"] = rtotal
    hourly["Reserve_Battery_Energy_MWh"] = reserve_energy
    hourly["Start_Margin_MWh"] = start_soc - EV_MIN_MWH - reserve_energy
    hourly["End_Margin_MWh"] = battery_soc - EV_MIN_MWH - reserve_energy

    summary = {
        "Scenario": "With EV reserve" if ev_reserve_enabled else "Without EV reserve",
        "EV_Reserve_MW_hours": rev.sum(),
        "Backup_Reserve_MW_hours": rb.sum(),
        "EV_Charging_MWh": charging.sum(),
        "V2G_Export_MWh": export_to_grid.sum(),
        "Battery_At_Departure_MWh": battery_soc.iloc[21],
        "Generation_Cost_EUR": generation_cost,
        "Reserve_Cost_EUR": reserve_cost,
        "Total_Cost_EUR": n.objective,
    }
    return hourly, summary


if __name__ == "__main__":
    print("\nRunning WITHOUT EV reserve...")
    no_ev, no_ev_summary = run_scenario(False)
    print("\nRunning WITH EV reserve...")
    with_ev, with_ev_summary = run_scenario(True)

    comparison = pd.DataFrame([no_ev_summary, with_ev_summary])
    print("\nDELIVERABILITY-CORRECTED COMPARISON\n")
    print(comparison.round(3).to_string(index=False))
    savings = no_ev_summary["Total_Cost_EUR"] - with_ev_summary["Total_Cost_EUR"]
    print(f"\nRevised total system cost savings: EUR {savings:,.2f}")

    print("\nEV reserve and battery states, 16:00-21:00:\n")
    print(with_ev.iloc[16:22][[
        "EV_Charging_MW", "V2G_Export_MW", "Battery_Start_MWh",
        "Battery_End_MWh", "EV_Reserve_MW", "Start_Margin_MWh",
        "End_Margin_MWh",
    ]].round(3).to_string())

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    no_ev.to_csv(output_dir / "lesson12_v2_without_ev_reserve.csv")
    with_ev.to_csv(output_dir / "lesson12_v2_with_ev_reserve.csv")
    comparison.to_csv(output_dir / "lesson12_v2_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    with_ev[["Hydro_Reserve_MW", "Backup_Reserve_MW", "EV_Reserve_MW"]].plot.area(ax=ax)
    ax.axhline(RESERVE_MW, color="black", linestyle="--", label="Required reserve")
    ax.set(title="Upward reserve allocation (corrected)", xlabel="Time", ylabel="Reserve capacity (MW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lesson12_v2_reserve_allocation.png", dpi=200)
    plt.show()

    fig, ax = plt.subplots(figsize=(11, 5))
    with_ev["Battery_Start_MWh"].plot(ax=ax, label="Start-of-hour energy")
    with_ev["Battery_End_MWh"].plot(ax=ax, label="End-of-hour energy")
    ax.axhline(EV_MIN_MWH, color="red", linestyle=":", label="Protected minimum")
    ax.axhline(EV_DEPARTURE_MWH, color="green", linestyle="--", label="Departure target")
    ax.set(title="EV battery energy", xlabel="Time", ylabel="Energy (MWh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "lesson12_v2_battery_energy.png", dpi=200)
    plt.show()

    print("\ndeliverability correction completed.")
