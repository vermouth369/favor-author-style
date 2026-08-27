#!/usr/bin/env bash
# ============================================================
# train_asce_full_k50.sh — Train ArcFace classifiers for K=50
# ============================================================
# Produces ArcFace-ready artifacts in:
#   runs/exp1_asce_full/K=50/author_classifier/
#   runs/exp1_asce_full/assistant_classifier/
#
# These are the directories pointed to by favor_main.yaml and
# consumed by 47b_compute_metrics_continuation.py and
# 53_direct_style_eval_continuation.py.
#
# Run this before the held-out continuation evaluation pipeline.
#
# Uses 07_train_classifiers.py with a patched config that outputs
# to the asce_full directory. Requires a local classifier config
# (config/exp1.yaml) and the raw corpus prepared under exp1/data/;
# neither is redistributed in this archive.
# ============================================================
set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXP1_DIR="${PROJ_ROOT}/exp1"
SCRIPTS_DIR="${EXP1_DIR}/scripts"
OUTPUT_BASE="${EXP1_DIR}/runs/exp1_asce_full"

echo "============================================================"
echo " Training ArcFace K=50 Classifiers"
echo " Output: ${OUTPUT_BASE}"
echo "============================================================"

cd "${EXP1_DIR}"
echo "  Working directory: $(pwd)"

# The existing 07_train_classifiers.py reads exp1.yaml and
# outputs to the runs_dir specified there. We override runs_dir
# to point to our asce_full target directory.

# Create a temporary config override
OVERRIDE_CONFIG="${OUTPUT_BASE}/train_config_override.yaml"
mkdir -p "${OUTPUT_BASE}"

python3 -c "
import yaml, copy, os, sys

with open('config/exp1.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Override runs_dir to write to asce_full
cfg['paths']['runs_dir'] = 'runs/exp1_asce_full'

# Force arcface backend
cfg.setdefault('classifiers', {})['backend'] = 'arcface'

# Only train K=50
cfg.setdefault('sweep', {})['k_values'] = [50]

with open('${OVERRIDE_CONFIG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)

print('  Override config written to ${OVERRIDE_CONFIG}')
print(f'  runs_dir = {cfg[\"paths\"][\"runs_dir\"]}')
print(f'  backend = {cfg[\"classifiers\"][\"backend\"]}')
print(f'  k_values = {cfg[\"sweep\"][\"k_values\"]}')
"

# ---- Step 1: Train both classifiers ----
echo ""
echo ">>> Training authorship (K=50) and assistant classifiers ..."
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU:-0} python3 "${SCRIPTS_DIR}/07_train_classifiers.py" \
    --config "${OVERRIDE_CONFIG}" \
    2>&1 | tee "${OUTPUT_BASE}/training.log"

TRAIN_EXIT=${PIPESTATUS[0]}
if [ "${TRAIN_EXIT}" -ne 0 ]; then
    echo "WARNING: Training exited with code ${TRAIN_EXIT}"
fi

# ---- Verify artifacts ----
echo ""
echo "============================================================"
echo " Verifying ArcFace artifacts"
echo "============================================================"

ALL_OK=true

AUTHOR_DIR="${OUTPUT_BASE}/K=50/author_classifier"
echo ""
echo "  --- Authorship classifier: ${AUTHOR_DIR}"
for artifact in backend_meta.json model_state.pt label_map.json; do
    if [ -f "${AUTHOR_DIR}/${artifact}" ]; then
        echo "    ✓ ${artifact}"
    else
        echo "    ✗ MISSING: ${artifact}"
        ALL_OK=false
    fi
done
for artifact in class_prototypes.npy class_weight_vectors.npy; do
    if [ -f "${AUTHOR_DIR}/${artifact}" ]; then
        echo "    ✓ ${artifact}"
    fi
done

ASST_DIR="${OUTPUT_BASE}/assistant_classifier"
echo ""
echo "  --- Assistant classifier: ${ASST_DIR}"
for artifact in backend_meta.json model_state.pt label_map.json calibration.json; do
    if [ -f "${ASST_DIR}/${artifact}" ]; then
        echo "    ✓ ${artifact}"
    else
        echo "    ✗ MISSING: ${artifact}"
        ALL_OK=false
    fi
done
for artifact in class_prototypes.npy class_weight_vectors.npy; do
    if [ -f "${ASST_DIR}/${artifact}" ]; then
        echo "    ✓ ${artifact}"
    fi
done

# ---- Summary ----
echo ""
echo "============================================================"
if [ "${ALL_OK}" = true ]; then
    echo " ✓ ArcFace K=50 Training Complete — all artifacts present"
else
    echo " ⚠ ArcFace Training Complete — some artifacts missing"
fi
echo "============================================================"
echo ""
echo "  Authorship: ${AUTHOR_DIR}"
echo "  Assistant:  ${ASST_DIR}"
echo ""
echo "  Next step: run the held-out continuation generation and"
echo "  metric scripts (46b_generate_exp2_continuation.py,"
echo "  47b_compute_metrics_continuation.py)."
echo "============================================================"
