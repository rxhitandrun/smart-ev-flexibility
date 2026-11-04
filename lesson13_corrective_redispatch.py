"""Re-optimise hourly dispatch after a single 15-minute EV reserve activation.

A 10 MW disturbance is assumed at either 17:00 or 18:00. Dispatch before
19:00 remains fixed; subsequent EV and generator schedules can change.
The model preserves hydro energy and ramp limits and conventional reserves.
This is an hourly recovery study, not a dynamic frequency simulation.
"""

from pathlib import Path
import pandas as pd
import pypsa

RESULTS = Path('results')
INPUT = RESULTS / 'lesson12_v2_with_ev_reserve.csv'
if not INPUT.exists():
    raise FileNotFoundError(
        f'Missing {INPUT}. Run python ev_operating_reserves_corrected.py first.'
    )
base = pd.read_csv(INPUT, index_col=0, parse_dates=True)
base.index.name = 'snapshot'

from ev_operating_reserves_corrected import (
    hours, normal_demand, wind_cf, WIND_MW, HYDRO_MW, BACKUP_MW,
    HYDRO_BUDGET_MWH, HYDRO_RAMP_PU_PER_HOUR, HYDRO_COST, BACKUP_COST,
    EV_POWER_MW, EV_CAPACITY_MWH, EV_MIN_MWH, EV_DEPARTURE_MWH,
    ETA_CHARGE, ETA_DISCHARGE, LINK_THROUGHPUT_EUR_PER_MWH,
    RESERVE_MW, HYDRO_RESERVE_RESPONSE_MW,
    HYDRO_RESERVE_PRICE, BACKUP_RESERVE_PRICE, EV_RESERVE_PRICE,
)

assert list(base.index) == list(hours), 'Unexpected snapshots in CSV'

TOL = 1e-5
activation_times = [hours[17], hours[18]]
reserve_activation_mw = 10.0
reserve_duration_hours = 0.25
extra_energy_used_mwh = (
    reserve_activation_mw * reserve_duration_hours / ETA_DISCHARGE
)

# Re-optimise only hours 19:00 through 23:00.
future_hours = hours[19:]
past_hours = hours[:19]
hydro_used_past = base.loc[past_hours, 'hydro'].sum()
hydro_remaining_mwh = HYDRO_BUDGET_MWH - hydro_used_past
hydro_previous_mw = float(base.loc[hours[18], 'hydro'])
base_energy_start_19 = float(base.loc[hours[18], 'Battery_End_MWh'])

if hydro_remaining_mwh < -TOL:
    raise ValueError('Historical hydro use already exceeds daily budget')


def solve_recovery(event_time):
    """Check immediate EV activation, then re-optimise the rest of the day."""
    committed = float(base.loc[event_time, 'EV_Reserve_MW'])
    if committed + TOL < reserve_activation_mw:
        raise ValueError(f'Not enough EV reserve committed at {event_time}')

    battery_start_at_event = float(base.loc[event_time, 'Battery_Start_MWh'])
    battery_end_at_event = float(base.loc[event_time, 'Battery_End_MWh'])
    if battery_start_at_event - extra_energy_used_mwh < EV_MIN_MWH - TOL:
        raise ValueError(f'Immediate battery reserve infeasible at {event_time}')
    if battery_end_at_event - extra_energy_used_mwh < EV_MIN_MWH - TOL:
        raise ValueError(f'End-of-hour battery reserve infeasible at {event_time}')

    # Both candidate events end before 19:00. We assume the scheduled
    # 17:00-18:59 operations remain unchanged except for this battery draw.
    battery_at_19 = base_energy_start_19 - extra_energy_used_mwh
    if battery_at_19 < EV_MIN_MWH - TOL:
        raise ValueError('Battery below minimum at start of recovery')

    n = pypsa.Network()
    n.set_snapshots(future_hours)
    n.add('Carrier', 'AC')
    n.add('Carrier', 'EV')
    n.add('Bus', 'grid', carrier='AC')
    n.add('Bus', 'ev_bus', carrier='EV')
    n.add('Load', 'normal_demand', bus='grid',
          p_set=pd.Series(normal_demand, index=hours).loc[future_hours])
    n.add('Generator', 'wind', bus='grid', p_nom=WIND_MW,
          p_max_pu=pd.Series(wind_cf, index=hours).loc[future_hours],
          marginal_cost=0)
    n.add('Generator', 'hydro', bus='grid', p_nom=HYDRO_MW,
          e_sum_max=max(0, hydro_remaining_mwh),
          ramp_limit_up=HYDRO_RAMP_PU_PER_HOUR,
          ramp_limit_down=HYDRO_RAMP_PU_PER_HOUR,
          marginal_cost=HYDRO_COST)
    n.add('Generator', 'backup', bus='grid', p_nom=BACKUP_MW,
          marginal_cost=BACKUP_COST)

    # Reconnection/charging window ends at 22:00. No V2G reserve is
    connected_future = pd.Series(0., index=future_hours)
    connected_future.loc[hours[19]:hours[21]] = 1.
    n.add('Link', 'ev_charger', bus0='grid', bus1='ev_bus',
          p_nom=EV_POWER_MW, p_min_pu=0, p_max_pu=connected_future,
          efficiency=ETA_CHARGE,
          marginal_cost=LINK_THROUGHPUT_EUR_PER_MWH)
    n.add('Link', 'ev_v2g', bus0='ev_bus', bus1='grid',
          p_nom=EV_POWER_MW, p_min_pu=0, p_max_pu=connected_future,
          efficiency=ETA_DISCHARGE,
          marginal_cost=LINK_THROUGHPUT_EUR_PER_MWH)

    # Already connected EVs start at 19:00 with reserve energy consumed.
    soc_min = pd.Series(EV_MIN_MWH / EV_CAPACITY_MWH, index=future_hours)
    soc_min.loc[hours[21]:] = EV_DEPARTURE_MWH / EV_CAPACITY_MWH
    n.add('Store', 'ev_battery', bus='ev_bus', e_nom=EV_CAPACITY_MWH,
          e_initial=battery_at_19, e_cyclic=False, e_min_pu=soc_min)

    m = n.optimize.create_model(include_objective_constant=False)
    gp = m.variables['Generator-p']
    lp = m.variables['Link-p']
    hydro = gp.sel(name='hydro', drop=True)
    backup = gp.sel(name='backup', drop=True)
    charge = lp.sel(name='ev_charger', drop=True)
    discharge = lp.sel(name='ev_v2g', drop=True)

    # Shared charger and inverter budget, no simultaneous charging/V2G
    # at a combined rate exceeding the fleet's charger rating.
    m.add_constraints(charge + discharge <= EV_POWER_MW,
                      name='shared_ev_converter')

    # IMPORTANT: first recovery hour must be reachable from actual 18:00
    # hydro output. Ordinary within-horizon ramp constraints omit this edge.
    hydro_at_19 = hydro.sel(snapshot=hours[19], drop=True)
    ramp_mw = HYDRO_MW * HYDRO_RAMP_PU_PER_HOUR
    m.add_constraints(hydro_at_19 <= hydro_previous_mw + ramp_mw,
                      name='hydro_ramp_from_previous_up')
    m.add_constraints(hydro_at_19 >= hydro_previous_mw - ramp_mw,
                      name='hydro_ramp_from_previous_down')

    # Preserve conventional reserve adequacy after the EV reserve event.
    # The 15 MW requirement is retained at all five recovery snapshots.
    rh = m.add_variables(lower=0, coords={'snapshot': n.snapshots},
                         name='recovery_hydro_reserve')
    rb = m.add_variables(lower=0, coords={'snapshot': n.snapshots},
                         name='recovery_backup_reserve')
    m.add_constraints(hydro + rh <= HYDRO_MW, name='hydro_headroom')
    m.add_constraints(rh <= HYDRO_RESERVE_RESPONSE_MW,
                      name='hydro_response_limit')
    m.add_constraints(backup + rb <= BACKUP_MW,
                      name='backup_headroom')
    m.add_constraints(rh + rb >= RESERVE_MW,
                      name='maintain_recovery_reserve')
    reserve_cost = HYDRO_RESERVE_PRICE * rh.sum() + BACKUP_RESERVE_PRICE * rb.sum()
    m.add_objective(m.objective.expression + reserve_cost, overwrite=True)

    status, termination = n.optimize.solve_model(solver_name='highs')
    if (status, termination) != ('ok', 'optimal'):
        raise RuntimeError(f'Corrective dispatch infeasible: {status}, {termination}')

    dispatch = n.generators_t.p
    charging = n.links_t.p0['ev_charger']
    v2g_export = -n.links_t.p1['ev_v2g']
    battery = n.stores_t.e['ev_battery']
    rh_result = pd.Series(rh.solution.to_numpy(), index=future_hours)
    rb_result = pd.Series(rb.solution.to_numpy(), index=future_hours)
    load = pd.Series(normal_demand, index=hours).loc[future_hours]

    assert (dispatch.sum(axis=1) + v2g_export - charging - load).abs().max() < TOL
    assert battery.min() >= EV_MIN_MWH - TOL
    assert battery.max() <= EV_CAPACITY_MWH + TOL
    assert battery.loc[hours[21]] >= EV_DEPARTURE_MWH - TOL
    assert (charging + n.links_t.p0['ev_v2g'] <= EV_POWER_MW + TOL).all()
    assert hydro_used_past + dispatch['hydro'].sum() <= HYDRO_BUDGET_MWH + TOL
    assert abs(float(dispatch.loc[hours[19], 'hydro']) - hydro_previous_mw) <= ramp_mw + TOL
    assert (dispatch['hydro'].diff().dropna().abs() <= ramp_mw + TOL).all()
    assert (rh_result + rb_result >= RESERVE_MW - TOL).all()

    result = dispatch.copy()
    result['EV_Charging_MW'] = charging
    result['V2G_Export_MW'] = v2g_export
    result['Battery_End_MWh'] = battery
    result['Hydro_Reserve_MW'] = rh_result
    result['Backup_Reserve_MW'] = rb_result

    # Incremental cost counts only the rescheduled 19:00-23:00 operation,
    # NOT a new total day-wide optimum or a reserve-activation market price.
    baseline_future_generation = (
        base.loc[future_hours, 'hydro'].sum() * HYDRO_COST
        + base.loc[future_hours, 'backup'].sum() * BACKUP_COST
    )
    baseline_future_link = (
        base.loc[future_hours, 'EV_Charging_MW'].sum()
        + base.loc[future_hours, 'V2G_Export_MW'].sum() / ETA_DISCHARGE
    ) * LINK_THROUGHPUT_EUR_PER_MWH
    baseline_future_reserve = (
        base.loc[future_hours, 'Hydro_Reserve_MW'].sum() * HYDRO_RESERVE_PRICE
        + base.loc[future_hours, 'Backup_Reserve_MW'].sum() * BACKUP_RESERVE_PRICE
    )
    baseline_future_cost = (baseline_future_generation
                            + baseline_future_link
                            + baseline_future_reserve)
    # Additional direct throughput from activation (battery-side input).
    activation_link_cost = extra_energy_used_mwh * LINK_THROUGHPUT_EUR_PER_MWH

    # If activation happens at 17:00, the 18:00 EV reserve commitment
    # cannot remain as planned: its initial battery energy is lower by
    # 2.632 MWh. Replace the 18:00 EV reserve with backup reserve.
    # This is a one-hour commitment change, not additional energy output.
    replacement_reserve_cost = 0.0
    if event_time == hours[17]:
        next_hour = hours[18]
        lost_ev_reserve = float(base.loc[next_hour, 'EV_Reserve_MW'])
        if lost_ev_reserve > TOL:
            backup_headroom = BACKUP_MW - float(base.loc[next_hour, 'backup'])
            existing_backup_reserve = float(base.loc[next_hour, 'Backup_Reserve_MW'])
            if backup_headroom + TOL < existing_backup_reserve + lost_ev_reserve:
                raise ValueError('Insufficient backup headroom to replace 18:00 EV reserve')
            replacement_reserve_cost = lost_ev_reserve * (
                BACKUP_RESERVE_PRICE - EV_RESERVE_PRICE
            )

    return result, {
        'Activation_Hour': str(event_time),
        'Immediate_Delivery': 'PASS',
        'Recovery_Optimisation': 'OPTIMAL',
        'Minimum_Battery_MWh': round(float(battery.min()), 4),
        'Departure_Battery_MWh': round(float(battery.loc[hours[21]]), 4),
        'Hydro_Budget_Used_MWh': round(float(hydro_used_past + dispatch['hydro'].sum()), 4),
        'Baseline_Future_Cost_EUR': round(float(baseline_future_cost), 4),
        'Recovery_Future_Cost_EUR': round(float(n.objective), 4),
        '18h_Reserve_Replacement_Cost_EUR': round(float(replacement_reserve_cost), 4),
        'Incremental_Recovery_Cost_EUR': round(float(
            n.objective - baseline_future_cost + activation_link_cost
            + replacement_reserve_cost), 4),
    }


if __name__ == '__main__':
    print('\nCORRECTIVE REDISPATCH AFTER ONE ACTIVATION\n')
    rows = []
    for event in activation_times:
        print(f'\nTesting reserve activation at {event} ...')
        recovered, info = solve_recovery(event)
        rows.append(info)
        recovered.to_csv(RESULTS / f'lesson13_recovery_{event.hour:02d}00.csv')
        print(recovered[['EV_Charging_MW', 'V2G_Export_MW', 'Battery_End_MWh',
                         'hydro', 'backup']].round(3).to_string())
    table = pd.DataFrame(rows)
    print('\nCORRECTIVE RECOVERY SUMMARY\n')
    print(table.to_string(index=False))
    table.to_csv(RESULTS / 'lesson13_corrective_recovery_summary.csv', index=False)
    print('\nSaved results/lesson13_corrective_recovery_summary.csv')
    print('NOTE: This is an hourly redispatch feasibility test, not frequency stability.')
