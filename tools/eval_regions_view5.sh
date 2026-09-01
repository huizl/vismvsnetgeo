#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-0}"
MODEL_TYPE="${MODEL_TYPE:-oa}"
DATAPATH="${DATAPATH:-/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu}"
TESTLIST="${TESTLIST:-lists/dtu/test.txt}"
CHECKPOINT="${CHECKPOINT:-}"
LABEL="${LABEL:-model}"
LIGHT="${LIGHT:-3}"
OUTDIR="${OUTDIR:-./outputs/region_metrics/${LABEL}_light${LIGHT}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
RANGE_SIGMA_SCALE="${RANGE_SIGMA_SCALE:-2.0}"
RANGE_MIN_SCALE="${RANGE_MIN_SCALE:-1.0}"
RANGE_MAX_SCALE="${RANGE_MAX_SCALE:-2.0}"
HYPOTHESIS_RESIDUAL_SCALE="${HYPOTHESIS_RESIDUAL_SCALE:-1.0}"

case "${MODEL_TYPE}" in
    vis)       ABLATION_CODE="000" ;;
    oa)        ABLATION_CODE="100" ;;
    range)     ABLATION_CODE="010" ;;
    hyp)       ABLATION_CODE="001" ;;
    oa_range)  ABLATION_CODE="110" ;;
    oa_hyp)    ABLATION_CODE="101" ;;
    range_hyp) ABLATION_CODE="011" ;;
    oa_full)   ABLATION_CODE="111" ;;
    *)
        echo "Unknown MODEL_TYPE: ${MODEL_TYPE}" >&2
        echo "Use: vis oa range hyp oa_range oa_hyp range_hyp oa_full" >&2
        exit 2
        ;;
esac

if [[ -z "${CHECKPOINT}" ]]; then
    echo "Set CHECKPOINT to a model checkpoint." >&2
    echo "Example: CHECKPOINT=./checkpoints/dtu/oa_view5/best_2mm.ckpt bash tools/eval_regions_view5.sh" >&2
    exit 2
fi

echo "======================================================================"
echo "label:       ${LABEL}"
echo "model:       ${MODEL_TYPE}"
echo "A/B/C:       ${ABLATION_CODE}"
echo "checkpoint:  ${CHECKPOINT}"
echo "test list:   ${TESTLIST}"
echo "light:       ${LIGHT} (-1 means all lights)"
echo "output:      ${OUTDIR}"
echo "======================================================================"

CUDA_VISIBLE_DEVICES="${GPU}" python tools/eval_region_metrics_dtu_yao.py \
    --model_type "${MODEL_TYPE}" \
    --loadckpt "${CHECKPOINT}" \
    --label "${LABEL}" \
    --testpath "${DATAPATH}" \
    --testlist "${TESTLIST}" \
    --outdir "${OUTDIR}" \
    --eval_nviews 5 \
    --region_nviews 5 \
    --light "${LIGHT}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --vismode soft \
    --numdepth 192 \
    --interval_scale 1.06 \
    --stage1_dnum 48 \
    --stage1_iscale 4 \
    --stage2_dnum 32 \
    --stage2_iscale 2 \
    --stage3_dnum 16 \
    --stage3_iscale 1 \
    --range_sigma_scale "${RANGE_SIGMA_SCALE}" \
    --range_min_scale "${RANGE_MIN_SCALE}" \
    --range_max_scale "${RANGE_MAX_SCALE}" \
    --hypothesis_residual_scale "${HYPOTHESIS_RESIDUAL_SCALE}" \
    --boundary_pct 10 \
    --large_disp_pct 80 \
    --occ_abs_tol 2.0 \
    --occ_rel_tol 0.01
