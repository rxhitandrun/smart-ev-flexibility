"""Test single EV reserve activations without changing subsequent dispatch.

The unchanged schedule is intentionally used to identify battery-energy
violations before corrective redispatch is considered.
"""

from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("results")
INPUT_FILE = RESULTS_DIR / "lesson12_v2_with_ev_reserve.csv"
OUTPUT_FILE = RESULTS_DIR / "lesson13_activation_stress_test.csv"

INITIAL_MWH = 10.0
MIN_MWH = 10.0
DEPARTURE_MWH = 30.0
DISCHARGE_EFFICIENCY = 0.95
ACTIVATION_DURATION_HOURS = 0.25
DEPARTURE_HOUR = 21  # End of 21:00 interval; vehicles depart at 22:00
TOLERANCE = 1e-4


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_FILE}. First run: python ev_operating_reserves.py"
        )

    data = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True)
    needed = ["EV_Reserve_MW", "Battery_Start_MWh", "Battery_End_MWh"]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")

    # The baseline is the *optimised, nonactivated* battery trajectory.
    start = data["Battery_Start_MWh"].astype(float)
    end = data["Battery_End_MWh"].astype(float)
    offered = data["EV_Reserve_MW"].astype(float).clip(lower=0)

    assert abs(start.iloc[0] - INITIAL_MWH) < TOLERANCE
    assert abs(end.iloc[DEPARTURE_HOUR] - data["Battery_End_MWh"].iloc[DEPARTURE_HOUR]) < TOLERANCE
    assert (start.iloc[1:].to_numpy() - end.iloc[:-1].to_numpy()).max() < TOLERANCE
    assert (end.iloc[:-1].to_numpy() - start.iloc[1:].to_numpy()).max() < TOLERANCE

    cases = []
    for idx, (timestamp, reserve_mw) in enumerate(offered.items()):
        if reserve_mw <= TOLERANCE:
            continue

        # A single full reserve activation at the beginning of this hour.
        extra_draw_mwh = (
            reserve_mw * ACTIVATION_DURATION_HOURS / DISCHARGE_EFFICIENCY
        )

        # Assumption: after activation, the original charging and scheduled
        # V2G dispatch remain unchanged. This deliberately disallows recovery.
        # The activated trajectory is the baseline less the extra discharge.
        activated_end = end.copy()
        activated_end.iloc[idx:] -= extra_draw_mwh

        immediate_energy = start.iloc[idx] - extra_draw_mwh
        immediate_ok = immediate_energy >= MIN_MWH - TOLERANCE

        # Check hourly battery trajectory from activation to departure.
        through_departure = activated_end.iloc[idx:DEPARTURE_HOUR + 1]
        minimum_after_activation = float(through_departure.min())
        trajectory_ok = minimum_after_activation >= MIN_MWH - TOLERANCE
        final_energy = float(activated_end.iloc[DEPARTURE_HOUR])
        departure_ok = final_energy >= DEPARTURE_MWH - TOLERANCE

        # Locate first scheduled end-of-hour SOC violation, if any.
        violations = through_departure[through_departure < MIN_MWH - TOLERANCE]
        first_violation = str(violations.index[0]) if len(violations) else "None"

        cases.append({
            "Activation_Hour": str(timestamp),
            "Reserve_MW": round(float(reserve_mw), 3),
            "Extra_Battery_Draw_MWh": round(extra_draw_mwh, 3),
            "Immediate_Energy_MWh": round(float(immediate_energy), 3),
            "Immediate_Check": bool(immediate_ok),
            "Lowest_Energy_Through_Departure_MWh": round(minimum_after_activation, 3),
            "Unchanged_Schedule_Check": bool(trajectory_ok),
            "First_SOC_Violation": first_violation,
            "Departure_Energy_MWh": round(final_energy, 3),
            "Departure_Check": bool(departure_ok),
        })

    if not cases:
        print("No positive EV reserve commitments found; nothing to activate.")
        return

    result = pd.DataFrame(cases)
    RESULTS_DIR.mkdir(exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print("\nSINGLE RESERVE ACTIVATION STRESS TEST\n")
    print(result.to_string(index=False))
    print("\nCases checked:", len(result))
    print("Immediate-delivery passes:", int(result["Immediate_Check"].sum()))
    print("Unchanged-schedule passes:", int(result["Unchanged_Schedule_Check"].sum()))
    print("Departure-energy passes:", int(result["Departure_Check"].sum()))
    print("\nSaved:", OUTPUT_FILE)
    print(
        "\nNOTE: A failed unchanged-schedule check does not establish that "
        "recovery is impossible. It means scheduled charging/discharging "
        "must be adjusted after activation."
    )


if __name__ == "__main__":
    main()
