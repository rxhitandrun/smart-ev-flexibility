#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python ev_operating_reserves_corrected.py
python validate_reserve_activation.py
python lesson13_corrective_redispatch.py
python lesson14_compensation_sensitivity.py
python lesson15_wind_sensitivity.py
python lesson16_ev_flexibility_sensitivity.py
python lesson17_reserve_validation.py
python lesson18_frequency_uncertainty.py
printf '\nCompleted. Outputs are saved in results/.\n'
