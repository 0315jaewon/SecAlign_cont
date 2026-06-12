#!/usr/bin/env bash

set -euo pipefail

OUT_DIR="${OUT_DIR:-sep_natural_suffix_outputs}" \
PLACEMENTS="" \
RUN_NATURAL_FILLER="1" \
NATURAL_FILLER_NUM_SAMPLES="${NATURAL_FILLER_NUM_SAMPLES:-256}" \
NATURAL_FILLER_JUDGE="${NATURAL_FILLER_JUDGE:-gemini-2.5-flash}" \
bash run_sep_csa_position_and_filler_experiments.sh
