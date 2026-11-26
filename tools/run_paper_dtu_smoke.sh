#!/usr/bin/env bash
# Run the first DTU check with existing checkpoints.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo 'Usage: bash tools/run_paper_dtu_smoke.sh DTU_ROOT CHECKPOINT_ROOT [all|View5|View3]' >&2
  exit 2
fi

dtu_root="$1"
checkpoint_root="$2"
python_bin="${PAPER_PYTHON:-python}"
output="${PAPER_OUTPUT:-outputs/first_batch_existing_weights/dtu_smoke}"
eval_args="${PAPER_EVAL_ARGS_JSON:-tools/paper_eval_args.example.json}"
selection="${PAPER_SELECTION:-tools/paper_dtu_smoke_samples.json}"
case "${3:-all}" in
  all) series=(View5 View3) ;;
  View5|View3) series=("$3") ;;
  *) echo 'Series must be all, View5 or View3' >&2; exit 2 ;;
esac

for split in "${series[@]}"; do
  "$python_bin" tools/visualize_paper_results.py export \
    --series "$split" --testpath "$dtu_root" --checkpoint_root "$checkpoint_root" \
    --configs Base 'Base+B' 'Base+A+B+C' \
    --selection "$selection" --eval_args_json "$eval_args" \
    --verify_mode warn --outdir "$output"
done

"$python_bin" tools/paper_search_windows.py "$output" --series "${series[@]}"
echo "Inference arrays and per-configuration reproduction_status.json reports: $output"
