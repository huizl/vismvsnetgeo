#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-0}"
MODEL_TYPE="${MODEL_TYPE:-vis}"
DATAPATH="${DATAPATH:-/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu}"
TESTLIST="${TESTLIST:-lists/dtu/val.txt}"
CHECKPOINT="${CHECKPOINT:-}"
LABEL="${LABEL:-model}"
LIGHT="${LIGHT:-3}"
OUTDIR="${OUTDIR:-./outputs/region_metrics/${LABEL}_light${LIGHT}}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
HYPOTHESIS_RESIDUAL_SCALE="${HYPOTHESIS_RESIDUAL_SCALE:-1.0}"
VISIBILITY_FUSION_BETA="${VISIBILITY_FUSION_BETA:-0.2}"
HYBRID_STAGE2_WIDE_NUM="${HYBRID_STAGE2_WIDE_NUM:-8}"
HYBRID_STAGE3_WIDE_NUM="${HYBRID_STAGE3_WIDE_NUM:-4}"
HYBRID_SIGMA_SCALE="${HYBRID_SIGMA_SCALE:-2.0}"
HYBRID_MAX_SCALE="${HYBRID_MAX_SCALE:-2.0}"
HYBRID_CLIP_MODE="${HYBRID_CLIP_MODE:-global}"

case "${MODEL_TYPE}" in
    vis)           ABLATION_CODE="000" ;;
    m1_hyp)        ABLATION_CODE="100" ;;
    m2_visibility) ABLATION_CODE="010" ;;
    m3_hybrid)     ABLATION_CODE="001" ;;
    m1_m2)         ABLATION_CODE="110" ;;
    m1_m3)         ABLATION_CODE="101" ;;
    m2_m3)         ABLATION_CODE="011" ;;
    full)          ABLATION_CODE="111" ;;
    *)
        echo "Unknown MODEL_TYPE: ${MODEL_TYPE}" >&2
        echo "Use: vis m1_hyp m2_visibility m3_hybrid m1_m2 m1_m3 m2_m3 full" >&2
        exit 2
        ;;
esac

if [[ -z "${CHECKPOINT}" ]]; then
    echo "Set CHECKPOINT to a model checkpoint." >&2
    echo "Example: MODEL_TYPE=m1_hyp CHECKPOINT=./checkpoints/dtu/m1_hyp_view5/best_2mm.ckpt bash tools/eval_regions_view5.sh" >&2
    exit 2
fi

echo "======================================================================"
echo "label:       ${LABEL}"
echo "model:       ${MODEL_TYPE}"
echo "M1/M2/M3:    ${ABLATION_CODE}"
echo "checkpoint:  ${CHECKPOINT}"
echo "validation:  ${TESTLIST}"
echo "M3 clipping: ${HYBRID_CLIP_MODE}"
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
    --hypothesis_residual_scale "${HYPOTHESIS_RESIDUAL_SCALE}" \
    --visibility_fusion_beta "${VISIBILITY_FUSION_BETA}" \
    --hybrid_stage2_wide_num "${HYBRID_STAGE2_WIDE_NUM}" \
    --hybrid_stage3_wide_num "${HYBRID_STAGE3_WIDE_NUM}" \
    --hybrid_sigma_scale "${HYBRID_SIGMA_SCALE}" \
    --hybrid_max_scale "${HYBRID_MAX_SCALE}" \
    --hybrid_clip_mode "${HYBRID_CLIP_MODE}" \
    --boundary_pct 10 \
    --large_disp_pct 80 \
    --occ_abs_tol 2.0 \
    --occ_rel_tol 0.01
