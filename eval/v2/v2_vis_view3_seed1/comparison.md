# v2 Val trained candidates

Each row uses its own best_2mm checkpoint. Val light=3, five inference/region views.
Candidate results; a positive effect must be established from the metrics and repeated training.

| Model | Region | Abs mm | Delta Abs vs v2_vis | Acc2 % | Delta Acc2 pp | Acc4 % | Acc8 % | S2 coverage % | S3 coverage % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2_vis | full | 5.0226 | +0.0000 | 80.5503 | +0.0000 | 88.7577 | 92.7775 | 97.6223 | 95.4095 |
| v2_vis | boundary | 19.9799 | +0.0000 | 40.0970 | +0.0000 | 54.2484 | 66.6095 | 92.2822 | 79.1230 |
| v2_vis | large_disparity | 12.6562 | +0.0000 | 68.0382 | +0.0000 | 78.9748 | 84.7221 | 91.6286 | 88.5554 |
| v2_vis | occluded_any | 22.7834 | +0.0000 | 44.3081 | +0.0000 | 57.3788 | 68.2389 | 89.8281 | 78.2526 |
| v2_vis | occluded_majority | 32.7356 | +0.0000 | 34.8047 | +0.0000 | 46.8113 | 57.9583 | 86.2050 | 69.7196 |
| v2_vis | large_disp_and_occluded | 35.0638 | +0.0000 | 41.4978 | +0.0000 | 53.6950 | 63.2750 | 82.2180 | 71.6871 |
| v2_vis | boundary_and_occluded | 29.4795 | +0.0000 | 30.4384 | +0.0000 | 42.8617 | 55.6164 | 88.6152 | 70.2119 |