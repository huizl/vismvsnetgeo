#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-2}"
MODEL_TYPES="${MODEL_TYPES:-range oa_range range_hyp}"
DATAPATH="${DATAPATH:-/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu}"
TRAINLIST="${TRAINLIST:-lists/dtu/train.txt}"
VALLIST="${VALLIST:-lists/dtu/val.txt}"
LOG_ROOT="${LOG_ROOT:-./checkpoints/dtu}"
TRAIN_NVIEWS="${TRAIN_NVIEWS:-5}"
EVAL_NVIEWS="${EVAL_NVIEWS:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-16}"
VISIBILITY_GT_DOWNSAMPLE="${VISIBILITY_GT_DOWNSAMPLE:-2}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"

echo "======================================================================"
echo "factorial queue GPU: ${GPU}"
echo "models:              ${MODEL_TYPES}"
echo "log root:            ${LOG_ROOT}"
echo "views:               train=${TRAIN_NVIEWS}, eval=${EVAL_NVIEWS}"
echo "======================================================================"

read -r -a MODEL_QUEUE <<< "${MODEL_TYPES}"
if [[ ${#MODEL_QUEUE[@]} -eq 0 ]]; then
    echo "MODEL_TYPES must contain at least one model name" >&2
    exit 2
fi

for MODEL_TYPE in "${MODEL_QUEUE[@]}"; do
    LOGDIR="${LOG_ROOT}/${MODEL_TYPE}_view${TRAIN_NVIEWS}"
    if [[ "${SKIP_COMPLETED}" == "1" && -f "${LOGDIR}/latest.ckpt" ]]; then
        echo "SKIP completed: ${MODEL_TYPE} (${LOGDIR}/latest.ckpt)"
        continue
    fi

    echo
    echo "QUEUE START: ${MODEL_TYPE} on GPU ${GPU}"
    GPU="${GPU}" \
    MODEL_TYPE="${MODEL_TYPE}" \
    DATAPATH="${DATAPATH}" \
    TRAINLIST="${TRAINLIST}" \
    VALLIST="${VALLIST}" \
    LOGDIR="${LOGDIR}" \
    TRAIN_NVIEWS="${TRAIN_NVIEWS}" \
    EVAL_NVIEWS="${EVAL_NVIEWS}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    EPOCHS="${EPOCHS}" \
    VISIBILITY_GT_DOWNSAMPLE="${VISIBILITY_GT_DOWNSAMPLE}" \
        bash tools/train_view5.sh
done

echo
echo "Queue completed on GPU ${GPU}: ${MODEL_TYPES}"
