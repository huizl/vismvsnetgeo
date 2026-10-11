#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=3 bash tools/run_blendedmvs_paper.sh BLENDED_ROOT CHECKPOINT_ROOT [View5|View3|all]
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash tools/run_blendedmvs_paper.sh BLENDED_ROOT CHECKPOINT_ROOT [View5|View3|all]" >&2
  exit 2
fi
data_root="$1"
checkpoint_root="$2"
series="${3:-View5}"
case "$series" in
  View5|View3) sets=("$series") ;;
  all) sets=(View5 View3) ;;
  *) echo "Series must be View5, View3, or all" >&2; exit 2 ;;
esac
extra=()
if [[ "$series" == all ]] && [[ -n "${BLENDED_BASE_CHECKPOINT:-}" || -n "${BLENDED_FULL_CHECKPOINT:-}" ]]; then
  echo "Explicit checkpoint paths require View5 or View3 separately; do not reuse one checkpoint for both series." >&2
  exit 2
fi
[[ -z "${BLENDED_TESTLIST:-}" ]] || extra+=(--testlist "$BLENDED_TESTLIST")
[[ -z "${BLENDED_SCENE:-}" ]] || extra+=(--scene "$BLENDED_SCENE")
[[ "${BLENDED_CHECK_ONLY:-0}" != 1 ]] || extra+=(--check_only)
[[ "${BLENDED_NO_FIGURES:-0}" != 1 ]] || extra+=(--no_figures)
[[ -z "${BLENDED_BASE_CHECKPOINT:-}" ]] || extra+=(--base_checkpoint "$BLENDED_BASE_CHECKPOINT")
[[ -z "${BLENDED_FULL_CHECKPOINT:-}" ]] || extra+=(--full_checkpoint "$BLENDED_FULL_CHECKPOINT")
for set_name in "${sets[@]}"; do
  "${PAPER_PYTHON:-python}" tools/eval_blendedmvs.py \
    --testpath "$data_root" --checkpoint_root "$checkpoint_root" --series "$set_name" \
    --checkpoint_name "${BLENDED_CHECKPOINT_NAME:-best_2mm.ckpt}" \
    --outdir "${BLENDED_OUTPUT:-outputs/cross_dataset/BlendedMVS}" \
    --max_samples "${BLENDED_MAX_SAMPLES:-0}" \
    --max_width "${BLENDED_MAX_WIDTH:-768}" --max_height "${BLENDED_MAX_HEIGHT:-576}" \
    --examples_per_category "${BLENDED_EXAMPLES_PER_CATEGORY:-2}" \
    "${extra[@]}"
done
