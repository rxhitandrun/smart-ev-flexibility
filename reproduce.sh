#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python ev_operating_reserves.py
python validate_reserve_activation.py
python corrective_redispatch.py
python compensation_sensitivity.py
python wind_sensitivity.py
python ev_flexibility_sensitivity.py
python reserve_validation.py
printf '\nCompleted. Outputs are saved in results/.\n'
