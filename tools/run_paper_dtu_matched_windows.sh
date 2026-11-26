#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash tools/run_paper_dtu_matched_windows.sh DTU_ROOT CHECKPOINT_ROOT [all|View5|View3]' >&2
  exit 2
fi

python_bin="${PAPER_PYTHON:-python}"
output="${PAPER_OUTPUT:-outputs/first_batch_existing_weights/dtu_matched_windows}"
eval_args="${PAPER_EVAL_ARGS_JSON:-tools/paper_eval_args.example.json}"
case "${3:-all}" in
  all) series=(View5 View3) ;;
  View5|View3) series=("$3") ;;
  *) echo 'Series must be all, View5 or View3' >&2; exit 2 ;;
esac

for split in "${series[@]}"; do
  common=(--series "$split" --testpath "$1" --checkpoint_root "$2"
          --configs 'Base+B' --verify_mode warn --eval_args_json "$eval_args")
  "$python_bin" tools/visualize_paper_results.py export "${common[@]}" \
    --validation --selection tools/paper_dtu_window_validation.json --outdir "$output/validation"
  "$python_bin" tools/visualize_paper_results.py export "${common[@]}" \
    --selection tools/paper_dtu_window_test.json --outdir "$output/test"
done

"$python_bin" tools/paper_search_windows.py "$output/test" \
  --validation_dir "$output/validation" --series "${series[@]}"
