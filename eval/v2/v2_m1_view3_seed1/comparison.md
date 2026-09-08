# v2 Val trained candidates

Each row uses its own best_2mm checkpoint. Val light=3, five inference/region views.
Candidate results; a positive effect must be established from the metrics and repeated training.

| Model | Region | Abs mm | Delta Abs vs v2_vis | Acc2 % | Delta Acc2 pp | Acc4 % | Acc8 % | S2 coverage % | S3 coverage % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_m1 | full | 5.0196 | - | 81.3403 | - | 88.9812 | 92.7531 | 97.5854 | 95.3654 |
| v2_m1 | boundary | 19.8265 | - | 40.8931 | - | 54.7271 | 66.7300 | 92.1810 | 79.2480 |
| v2_m1 | large_disparity | 12.5258 | - | 69.6525 | - | 79.4050 | 84.7391 | 91.6298 | 88.5777 |
| v2_m1 | occluded_any | 22.6753 | - | 44.8783 | - | 57.5079 | 68.1759 | 89.8238 | 78.4114 |
| v2_m1 | occluded_majority | 32.5448 | - | 35.4614 | - | 47.0982 | 57.9613 | 86.2246 | 69.9156 |
| v2_m1 | large_disp_and_occluded | 35.0589 | - | 42.5621 | - | 54.0690 | 63.1321 | 82.1585 | 71.9857 |
| v2_m1 | boundary_and_occluded | 29.2145 | - | 30.8874 | - | 43.0702 | 55.5790 | 88.5684 | 70.3531 |