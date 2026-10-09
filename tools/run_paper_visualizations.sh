#!/usr/bin/env bash
# Usage: CUDA_VISIBLE_DEVICES=0 bash tools/run_paper_visualizations.sh DTU_ROOT CHECKPOINT_ROOT [all|View5|View3]
# CSV statistics retain all eight configurations; inference/qualitative figures compare Base and Ours.
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash tools/run_paper_visualizations.sh DTU_ROOT CHECKPOINT_ROOT [all|View5|View3]' >&2
  exit 2
fi
paper_data_root="$1"
paper_checkpoint_root="$2"
paper_series="${3:-all}"
paper_python="${PAPER_PYTHON:-python}"
paper_output="${PAPER_OUTPUT:-outputs/paper_visualizations}"
paper_eval_args="${PAPER_EVAL_ARGS_JSON:-tools/paper_eval_args.example.json}"
paper_verify_mode="${PAPER_VERIFY_MODE:-strict}"
paper_batch_size="${PAPER_BATCH_SIZE:-1}"
case "$paper_series" in
  all) paper_sets=(View5 View3) ;;
  View5|View3) paper_sets=("$paper_series") ;;
  *) echo 'Series must be all, View5 or View3' >&2; exit 2 ;;
esac
"$paper_python" tools/visualize_paper_results.py stats --series "$paper_series" --outdir "$paper_output"
for paper_set in "${paper_sets[@]}"; do
  "$paper_python" tools/visualize_paper_results.py export \
    --series "$paper_set" --testpath "$paper_data_root" \
    --checkpoint_root "$paper_checkpoint_root" --testlist lists/dtu/test.txt \
    --configs Base 'Base+A+B+C' \
    --verify_mode "$paper_verify_mode" --batch_size "$paper_batch_size" \
    --selection "$paper_output/$paper_set/selection.json" \
    --eval_args_json "$paper_eval_args" --outdir "$paper_output"
  "$paper_python" tools/visualize_paper_results.py render \
    --series "$paper_set" --outdir "$paper_output" --layout paper
done
