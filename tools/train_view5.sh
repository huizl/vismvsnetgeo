#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-0}"
MODEL_TYPE="${MODEL_TYPE:-vis}"
DATAPATH="${DATAPATH:-/home/disk_10T/lzh_data/dtu_training/mvs_training/dtu}"
TRAINLIST="${TRAINLIST:-lists/dtu/train.txt}"
VALLIST="${VALLIST:-lists/dtu/val.txt}"
TRAIN_NVIEWS="${TRAIN_NVIEWS:-5}"
EVAL_NVIEWS="${EVAL_NVIEWS:-5}"
LOGDIR="${LOGDIR:-./checkpoints/dtu/${MODEL_TYPE}_view${TRAIN_NVIEWS}}"
INIT_CKPT="${INIT_CKPT:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-16}"
TRAIN_WORKERS="${TRAIN_WORKERS:-8}"
TEST_WORKERS="${TEST_WORKERS:-4}"
VISIBILITY_GT_DOWNSAMPLE="${VISIBILITY_GT_DOWNSAMPLE:-2}"
HYPOTHESIS_RESIDUAL_SCALE="${HYPOTHESIS_RESIDUAL_SCALE:-1.0}"
HYPOTHESIS_VISIBILITY_WEIGHT="${HYPOTHESIS_VISIBILITY_WEIGHT:-0.1}"
VISIBILITY_FUSION_BETA="${VISIBILITY_FUSION_BETA:-0.2}"
HYBRID_STAGE2_WIDE_NUM="${HYBRID_STAGE2_WIDE_NUM:-8}"
HYBRID_STAGE3_WIDE_NUM="${HYBRID_STAGE3_WIDE_NUM:-4}"
HYBRID_SIGMA_SCALE="${HYBRID_SIGMA_SCALE:-2.0}"
HYBRID_MAX_SCALE="${HYBRID_MAX_SCALE:-2.0}"

if ! [[ "${TRAIN_NVIEWS}" =~ ^[0-9]+$ ]] || (( TRAIN_NVIEWS < 2 )); then
    echo "TRAIN_NVIEWS must be an integer >= 2, got: ${TRAIN_NVIEWS}" >&2
    exit 2
fi
if ! [[ "${EVAL_NVIEWS}" =~ ^[0-9]+$ ]] || (( EVAL_NVIEWS < 2 )); then
    echo "EVAL_NVIEWS must be an integer >= 2, got: ${EVAL_NVIEWS}" >&2
    exit 2
fi

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

LOAD_ARGS=()
if [[ -n "${INIT_CKPT}" ]]; then
    LOAD_ARGS+=(--loadckpt "${INIT_CKPT}")
fi

echo "======================================================================"
echo "model:       ${MODEL_TYPE}"
echo "M1/M2/M3:    ${ABLATION_CODE}"
echo "GPU:         ${GPU}"
echo "train list:  ${TRAINLIST}"
echo "val list:    ${VALLIST}"
echo "views:       train=${TRAIN_NVIEWS}, eval=${EVAL_NVIEWS}"
echo "batch size:  ${BATCH_SIZE}"
echo "output:      ${LOGDIR}"
if [[ -n "${INIT_CKPT}" ]]; then
    echo "init ckpt:   ${INIT_CKPT}"
else
    echo "init ckpt:   from scratch"
fi
echo "======================================================================"

CUDA_VISIBLE_DEVICES="${GPU}" python train.py \
    --mode train \
    --model_type "${MODEL_TYPE}" \
    --dataset dtu_yao \
    --trainpath "${DATAPATH}" \
    --testpath "${DATAPATH}" \
    --trainlist "${TRAINLIST}" \
    --testlist "${VALLIST}" \
    --logdir "${LOGDIR}" \
    --batch_size "${BATCH_SIZE}" \
    --nviews "${TRAIN_NVIEWS}" \
    --eval_nviews "${EVAL_NVIEWS}" \
    --numdepth 192 \
    --interval_scale 1.06 \
    --epochs "${EPOCHS}" \
    --lr 0.001 \
    --lrepochs "10,12,14:2" \
    --wd 0.0 \
    --vismode soft \
    --stage1_dnum 48 \
    --stage1_iscale 4 \
    --stage2_dnum 32 \
    --stage2_iscale 2 \
    --stage3_dnum 16 \
    --stage3_iscale 1 \
    --visibility_gt_downsample "${VISIBILITY_GT_DOWNSAMPLE}" \
    --pair_l1_weight 1.0 \
    --uncertainty_weight 1.0 \
    --visibility_weight 0.2 \
    --visibility_focal_gamma 2.0 \
    --hypothesis_residual_scale "${HYPOTHESIS_RESIDUAL_SCALE}" \
    --hypothesis_visibility_weight "${HYPOTHESIS_VISIBILITY_WEIGHT}" \
    --visibility_fusion_beta "${VISIBILITY_FUSION_BETA}" \
    --hybrid_stage2_wide_num "${HYBRID_STAGE2_WIDE_NUM}" \
    --hybrid_stage3_wide_num "${HYBRID_STAGE3_WIDE_NUM}" \
    --hybrid_sigma_scale "${HYBRID_SIGMA_SCALE}" \
    --hybrid_max_scale "${HYBRID_MAX_SCALE}" \
    --occ_abs_tol 2.0 \
    --occ_rel_tol 0.01 \
    --summary_freq 20 \
    --save_freq 1 \
    --train_workers "${TRAIN_WORKERS}" \
    --test_workers "${TEST_WORKERS}" \
    --seed 1 \
    "${LOAD_ARGS[@]}"
