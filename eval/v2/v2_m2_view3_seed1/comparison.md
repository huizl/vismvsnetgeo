# v2 Val trained candidates

Each row uses its own best_2mm checkpoint. Val light=3, five inference/region views.
Candidate results; a positive effect must be established from the metrics and repeated training.

| Model | Region | Abs mm | Delta Abs vs v2_vis | Acc2 % | Delta Acc2 pp | Acc4 % | Acc8 % | S2 coverage % | S3 coverage % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_m2 | full | 5.0780 | - | 80.9239 | - | 88.7851 | 92.7290 | 97.6023 | 95.4293 |
| v2_m2 | boundary | 20.0864 | - | 40.3870 | - | 54.3732 | 66.5763 | 92.1249 | 79.2397 |
| v2_m2 | large_disparity | 12.9587 | - | 68.5650 | - | 78.9393 | 84.6056 | 91.5741 | 88.6057 |
| v2_m2 | occluded_any | 22.9718 | - | 44.3906 | - | 57.2777 | 68.1210 | 89.7214 | 78.4106 |
| v2_m2 | occluded_majority | 32.8129 | - | 34.7717 | - | 46.5998 | 57.7716 | 86.0749 | 69.9999 |
| v2_m2 | large_disp_and_occluded | 35.8717 | - | 41.3817 | - | 53.3735 | 63.0121 | 81.8800 | 71.8930 |
| v2_m2 | boundary_and_occluded | 29.5229 | - | 30.4183 | - | 42.6817 | 55.4013 | 88.4592 | 70.4399 |