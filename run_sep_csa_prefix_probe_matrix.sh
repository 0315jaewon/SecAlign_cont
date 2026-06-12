#!/usr/bin/env bash

set -euo pipefail

OUT_DIR="${OUT_DIR:-sep_csa_prefix_probe_outputs}" \
PLACEMENTS="prefix" \
RUN_NATURAL_FILLER="0" \
bash run_sep_csa_position_and_filler_experiments.sh
