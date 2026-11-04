"""Illustrative aggregate frequency-response sensitivity analysis.

This separate swing-equation model uses synthetic inertia, disturbances and
response times. It is not calibrated to Icelandic grid measurements or
coupled dynamically to the dispatch optimisation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

NOMINAL_FREQUENCY_HZ = 50.0
SYSTEM_BASE_MW = 200.0       # Illustrative aggregate synchronous power base.
DAMPING_MW_PER_HZ = 2.0      # Simplified frequency-sensitive demand response.

# The following 10 MW EV and 5 MW hydro capacities are inspired by the
# We now ASSUME illustrative response timings solely for sensitivity work.
EV_RESERVE_MW = 10.0
HYDRO_RESERVE_MW = 5.0
HYDRO_DELAY_S = 5.0
HYDRO_RAMP_S = 30.0
EV_RAMP_S = 10.0

# Only analyse the FIRST 120 SECONDS of frequency response here.
# Battery-energy sustainability for 15-minute EV activation was investigated
# separately in Lessons 12-13. This file does NOT revalidate that recovery.
SIMULATION_SECONDS = 120.0
TIME_STEP_SECONDS = 0.1

# All variations are hypothetical, not uncertainty distributions measured
# from an electricity grid. We do NOT assign probabilities to these cases.
INERTIA_CASES_SECONDS = (3.0, 5.0, 7.0)
DISTURBANCE_CASES_MW = (5.0, 10.0, 15.0)
EV_DELAY_CASES_SECONDS = (1.0, 5.0, 10.0)


def response_ramp(t_seconds, delay_seconds, ramp_seconds, max_response_mw):
    """Return a simple delayed linear ramp of delivered response in MW.

    At t < delay: no response.
    During the next ramp_seconds: response rises linearly.
    After that: response stays at max_response_mw.

    This is NOT a physical inverter controller or a hydro governor model.
    """
    elapsed = max(0.0, t_seconds - delay_seconds)
    fraction = min(elapsed / ramp_seconds, 1.0)
    return max_response_mw * fraction


def simulate_frequency(disturbance_mw, inertia_seconds, ev_delay_seconds, use_ev):
    """Solve a very simple aggregate swing-equation approximation.

    d(Delta f)/dt = f0 / (2 * H * Sbase) * ( -P_loss + P_hydro
                        + P_ev - D * Delta f )

    Delta f is Hz relative to nominal, H is seconds, and the other
    quantities are expressed in MW. The proportional damping term is an
    *assumed* simplified aggregate load response.

    Forward Euler integration with 0.1-second steps is sufficient for this
    classroom illustration, not engineering-grade validation.
    """
    times = np.arange(0, SIMULATION_SECONDS + TIME_STEP_SECONDS / 2,
                      TIME_STEP_SECONDS)
    delta_f = np.zeros(len(times), dtype=float)
    hydro_mw = np.zeros(len(times), dtype=float)
    ev_mw = np.zeros(len(times), dtype=float)

    for i, t in enumerate(times):
        hydro_mw[i] = response_ramp(
            t, HYDRO_DELAY_S, HYDRO_RAMP_S, HYDRO_RESERVE_MW
        )
        if use_ev:
            ev_mw[i] = response_ramp(
                t, ev_delay_seconds, EV_RAMP_S, EV_RESERVE_MW
            )

        if i == len(times) - 1:
            break

        # Negative disturbance means lost generation and falling frequency.
        power_imbalance_mw = (
            -disturbance_mw + hydro_mw[i] + ev_mw[i]
            - DAMPING_MW_PER_HZ * delta_f[i]
        )
        dfdt_hz_per_s = (
            NOMINAL_FREQUENCY_HZ / (2.0 * inertia_seconds * SYSTEM_BASE_MW)
            * power_imbalance_mw
        )
        delta_f[i + 1] = delta_f[i] + TIME_STEP_SECONDS * dfdt_hz_per_s

    frequency_hz = NOMINAL_FREQUENCY_HZ + delta_f
    # Early RoCoF approximation from frequency difference over 0.5 seconds.
    early_steps = int(round(0.5 / TIME_STEP_SECONDS))
    rocof_first_half_second = (
        frequency_hz[early_steps] - frequency_hz[0]
    ) / (times[early_steps] - times[0])

    series = pd.DataFrame({
        "Time_s": times,
        "Frequency_Hz": frequency_hz,
        "Hydro_Response_MW": hydro_mw,
        "EV_Response_MW": ev_mw,
    })
    summary = {
        "Disturbance_MW": disturbance_mw,
        "Inertia_s": inertia_seconds,
        "EV_Delay_s": ev_delay_seconds if use_ev else np.nan,
        "EV_Enabled": bool(use_ev),
        "Frequency_Nadir_Hz": float(frequency_hz.min()),
        "Time_of_Nadir_s": float(times[np.argmin(frequency_hz)]),
        "RoCoF_First_0p5s_Hz_per_s": float(rocof_first_half_second),
        "Frequency_at_120s_Hz": float(frequency_hz[-1]),
    }
    return series, summary


def main():
    print("\nILLUSTRATIVE FREQUENCY RESPONSE AND UNCERTAINTY")
    print("All system parameters are SYNTHETIC; no Icelandic input data.")
    print("This is an educational aggregate swing-equation approximation.\n")

    # Reference example: one 15 MW generation loss, assumed H=5 s,
    # and an EV controller delayed by 5 s before starting a 10 s ramp.
    base = (15.0, 5.0, 5.0)
    no_ev_curve, no_ev_summary = simulate_frequency(*base, use_ev=False)
    with_ev_curve, with_ev_summary = simulate_frequency(*base, use_ev=True)

    print("REFERENCE EXAMPLE: 15 MW loss, H=5 s, EV response delay=5 s")
    print("(Hydro provides up to 5 MW; EVs provide up to 10 MW.)")
    print(pd.DataFrame([no_ev_summary, with_ev_summary])[
        ["EV_Enabled", "Frequency_Nadir_Hz", "Time_of_Nadir_s",
         "RoCoF_First_0p5s_Hz_per_s", "Frequency_at_120s_Hz"]
    ].round(4).to_string(index=False))

    no_ev_curve.to_csv(OUTPUT_DIR / "lesson18_reference_without_ev.csv", index=False)
    with_ev_curve.to_csv(OUTPUT_DIR / "lesson18_reference_with_ev.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(no_ev_curve["Time_s"], no_ev_curve["Frequency_Hz"],
             label="Without EV dynamic response", linewidth=2)
    plt.plot(with_ev_curve["Time_s"], with_ev_curve["Frequency_Hz"],
             label="With assumed EV dynamic response", linewidth=2)
    plt.axhline(NOMINAL_FREQUENCY_HZ, color="gray", linestyle=":",
                label="Nominal frequency")
    plt.xlabel("Seconds after illustrative generation loss")
    plt.ylabel("System frequency (Hz)")
    plt.title("Illustrative aggregate frequency response")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lesson18_reference_frequency.png", dpi=160)
    plt.close()

    # Full sensitivity matrix: 3 inertia x 3 disturbances x 3 EV delays.
    # Each pair compares the SAME disturbance and inertia, with/without EV.
    records = []
    for inertia in INERTIA_CASES_SECONDS:
        for disturbance in DISTURBANCE_CASES_MW:
            for delay in EV_DELAY_CASES_SECONDS:
                _, no_ev = simulate_frequency(
                    disturbance, inertia, delay, use_ev=False
                )
                _, yes_ev = simulate_frequency(
                    disturbance, inertia, delay, use_ev=True
                )
                records.append({
                    "Inertia_s": inertia,
                    "Disturbance_MW": disturbance,
                    "EV_Delay_s": delay,
                    "Nadir_Without_EV_Hz": no_ev["Frequency_Nadir_Hz"],
                    "Nadir_With_EV_Hz": yes_ev["Frequency_Nadir_Hz"],
                    "Nadir_Improvement_Hz": (
                        yes_ev["Frequency_Nadir_Hz"]
                        - no_ev["Frequency_Nadir_Hz"]
                    ),
                    "RoCoF_Without_EV_Hz_per_s": (
                        no_ev["RoCoF_First_0p5s_Hz_per_s"]
                    ),
                    "RoCoF_With_EV_Hz_per_s": (
                        yes_ev["RoCoF_First_0p5s_Hz_per_s"]
                    ),
                })

    sensitivity = pd.DataFrame(records)
    sensitivity.to_csv(
        OUTPUT_DIR / "lesson18_frequency_uncertainty_summary.csv", index=False
    )

    print("\nUNCERTAINTY MATRIX (27 synthetic cases)")
    print("Minimum/maximum nadir with EV (Hz):",
          f"{sensitivity['Nadir_With_EV_Hz'].min():.3f} / "
          f"{sensitivity['Nadir_With_EV_Hz'].max():.3f}")
    print("Minimum/maximum nadir improvement (Hz):",
          f"{sensitivity['Nadir_Improvement_Hz'].min():.3f} / "
          f"{sensitivity['Nadir_Improvement_Hz'].max():.3f}")

    print("\nFIXED 15 MW LOSS AND H=5 s: EFFECT OF EV RESPONSE DELAY")
    selected = sensitivity.loc[
        (sensitivity["Disturbance_MW"] == 15.0)
        & (sensitivity["Inertia_s"] == 5.0),
        ["EV_Delay_s", "Nadir_Without_EV_Hz", "Nadir_With_EV_Hz",
         "Nadir_Improvement_Hz"],
    ].copy()
    print(selected.round(4).to_string(index=False))

    plt.figure(figsize=(9, 5))
    for delay in EV_DELAY_CASES_SECONDS:
        sub = sensitivity.loc[sensitivity["EV_Delay_s"] == delay]
        plt.scatter(sub["Inertia_s"], sub["Nadir_With_EV_Hz"],
                    label=f"EV delay {delay:.0f} s", alpha=0.8)
    plt.xlabel("Assumed inertia constant H (seconds)")
    plt.ylabel("Minimum frequency (Hz)")
    plt.title("Frequency nadir across synthetic uncertainties")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "lesson18_uncertainty_scatter.png", dpi=160)
    plt.close()

    print("\nSaved 3 CSV files and 2 PNG figures in results/.")
    print("LIMITATIONS: simplified single-area swing model; illustrative response")
    print("timings; no governor dynamics, inverter control, protection, network")
    print("location, measured inertia, real disturbances or grid-code assessment.")
    print("No claim about actual Icelandic frequency stability can be inferred.\n")


if __name__ == "__main__":
    main()
