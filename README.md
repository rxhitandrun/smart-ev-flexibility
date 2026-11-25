# Smart EV Flexibility for Ancillary Services in a Wind-Rich Isolated Power System

A small research prototype exploring how aggregated electric vehicles (EVs) could contribute upward operating reserve in an isolated electricity system with wind and hydro generation. I developed the model to examine the relationship between EV charging schedules, available battery energy, conventional reserves and operating cost.

The work is exploratory. All demand, wind, fleet and price inputs are **synthetic**. The model does not represent the Icelandic grid, and the results should not be interpreted as estimates of its operating costs or frequency stability.

## Approach

The main model uses [PyPSA](https://pypsa.org/), Linopy and HiGHS to optimise 24 hourly operating periods. It includes dispatchable hydro, wind generation, a backup generator and an aggregated EV fleet with charging and vehicle-to-grid (V2G) operation. EV reserve commitments compete with hydro and backup reserve while respecting battery-energy, converter-capacity and departure requirements.

The reference configuration assumes 160 MW of wind, 100 MW of hydro capacity, a 900 MWh daily hydro energy allowance and a 15 MW hourly upward reserve requirement. The EV fleet represents 2,000 connected vehicles with 14 MW aggregate converter capacity, 40 MWh usable stored-energy capacity and a 30 MWh departure target. Charging and discharging efficiencies are each set to 95%. Reserve availability costs are illustrative, including €3 per MW per hour for EVs.

A separate recovery model examines what happens if EV reserve is activated at 17:00 or 18:00.

## Repository contents

| Script | Purpose |
| --- | --- |
| `ev_operating_reserves.py` | Reference dispatch and reserve co-optimisation |
| `validate_reserve_activation.py` | Checks activation against the unchanged operating schedule |
| `corrective_redispatch.py` | Re-optimises the remaining hours following reserve activation |
| `compensation_sensitivity.py` | EV reserve availability payment sensitivity |
| `wind_sensitivity.py` | Wind capacity sensitivity |
| `ev_flexibility_sensitivity.py` | Fleet size, initial charge and connection-window sensitivity |
| `reserve_validation.py` | Reserve duration, requirement and independent schedule checks |


## Running the model

From the repository directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash reproduce.sh
```

On macOS, use `python` from the activated virtual environment rather than a separate system Python installation. The scripts write tables and figures to `results/`, which is ignored by Git. Run the reference model before the activation and recovery checks, since those scripts read the reference CSV output.

The repository has been tested locally using the listed Python dependencies. Exact cross-machine reproducibility is not guaranteed because dependency versions are not pinned.

## Results from the original local runs

The following results were obtained in the original working environment with synthetic inputs:

| Experiment | Selected finding |
| --- | --- |
| Reference reserve case | 20 MW·h of EV reserve commitment; €34.19 lower daily model objective than a matched no-EV-reserve case |
| EV compensation | At €1/MW/hour, EV reserve increased to 36.6 MW·h; at €5 and €10/MW/hour, the optimiser selected none |
| Wind capacity | Reserve participation was not monotonic across 80–240 MW wind cases; the 240 MW case committed 40 MW·h of EV reserve |
| Connection time | Arriving at 16:00 yielded 20 MW·h of reserve; arriving at 18:00 yielded none |
| Activation duration | At a 15 MW reserve requirement, commitment declined from 20 MW·h at 15 minutes to 10 MW·h at 30 minutes and zero at 60 minutes, when the assumed hydro response reaches the full 15 MW reserve requirement |
| Corrective redispatch | Two separately tested 15-minute reserve activations had feasible subsequent hourly recovery schedules, conditional on the model assumptions |

Here, **MW·h of reserve commitment** means MW of capacity held available across hourly snapshots. It is not the electrical energy delivered during an activation, which is measured in MWh.


## Limitations and next steps

The hourly optimisation captures energy scheduling and simplified reserve deliverability, but does not resolve seconds-scale EV inverter or hydro-governor behaviour. The reserve analysis assumes one activation at a time; corrective redispatch was tested only for two reference cases. Repeated disturbances, uncertain EV participation, degradation, aggregator expenses and probability-weighted activation costs remain outside the current model.

A stronger study would use measured or otherwise validated input profiles, detailed hydro and inverter dynamics, realistic EV availability uncertainty and an explicitly coupled dispatch-and-frequency assessment. Those are proposed research directions rather than completed results.

## Reuse

Released under the MIT License; see `LICENSE`.
