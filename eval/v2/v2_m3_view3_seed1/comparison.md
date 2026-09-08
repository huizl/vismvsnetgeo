# v2 Val trained candidates

Each row uses its own best_2mm checkpoint. Val light=3, five inference/region views.
Candidate results; a positive effect must be established from the metrics and repeated training.

| Model | Region | Abs mm | Delta Abs vs v2_vis | Acc2 % | Delta Acc2 pp | Acc4 % | Acc8 % | S2 coverage % | S3 coverage % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_m3 | full | 5.0970 | - | 81.4989 | - | 89.1400 | 92.8225 | 97.6824 | 95.2690 |
| v2_m3 | boundary | 21.3444 | - | 40.9985 | - | 54.8138 | 66.4654 | 92.4772 | 77.8919 |
| v2_m3 | large_disparity | 12.5590 | - | 70.3072 | - | 80.0450 | 85.1205 | 91.9621 | 88.5716 |
| v2_m3 | occluded_any | 23.2448 | - | 45.2343 | - | 58.0289 | 68.3188 | 90.1704 | 77.8361 |
| v2_m3 | occluded_majority | 33.2731 | - | 35.9732 | - | 47.6223 | 58.1529 | 86.6619 | 69.3577 |
| v2_m3 | large_disp_and_occluded | 34.8243 | - | 43.7172 | - | 55.4803 | 64.4239 | 83.3981 | 72.4790 |
| v2_m3 | boundary_and_occluded | 31.1542 | - | 31.1092 | - | 43.1517 | 55.2307 | 88.9582 | 68.7110 |