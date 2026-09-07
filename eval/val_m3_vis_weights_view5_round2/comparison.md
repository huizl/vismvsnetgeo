# Val inference diagnostics

Fixed val.txt, light=3, five inference/region views. Pixel-weighted metrics.
These are inference diagnostics, not retrained ablations. Checkpoint sources are listed below.
M2 beta=0 disables the inference gate; it does not undo visibility supervision during training.
M3 scale=1 disables expansion; clipping is controlled independently.

| Setting | Checkpoint |
| --- | --- |
| vis | /home/disk_10T/lzh/triple/vismvsnetgeo/checkpoints/dtu/vis_view5/best_2mm.ckpt |
| vis_weights_m3_none_scale1.0 | /home/disk_10T/lzh/triple/vismvsnetgeo/checkpoints/dtu/vis_view5/best_2mm.ckpt |
| vis_weights_m3_none_scale2.0 | /home/disk_10T/lzh/triple/vismvsnetgeo/checkpoints/dtu/vis_view5/best_2mm.ckpt |

## full

| Setting | Abs mm | Delta Abs vs vis | Acc2 % | Delta Acc2 pp | Acc4 % | Acc8 % | S2 coverage % | S3 coverage % | S2 width/base | S3 width/base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vis | 4.5424 | 0.0000 | 82.6992 | 0.0000 | 89.6477 | 93.1708 | 97.7247 | 95.7871 | 62.0000 | 15.0000 |
| vis_weights_m3_none_scale1.0 | 4.5424 | 0.0000 | 82.6991 | -0.0001 | 89.6480 | 93.1708 | 97.7247 | 95.7871 | 62.0000 | 15.0000 |
| vis_weights_m3_none_scale2.0 | 4.5504 | 0.0080 | 82.7065 | 0.0073 | 89.6582 | 93.1863 | 97.8541 | 96.1169 | 62.2270 | 15.2013 |

## boundary

| Setting | Abs mm | Delta Abs vs vis | Acc2 % | Delta Acc2 pp | Acc4 % | Acc8 % | S2 coverage % | S3 coverage % | S2 width/base | S3 width/base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vis | 19.1283 | 0.0000 | 41.2059 | 0.0000 | 55.0218 | 67.0943 | 92.4576 | 80.2536 | 62.0000 | 15.0000 |
| vis_weights_m3_none_scale1.0 | 19.1284 | 0.0001 | 41.2049 | -0.0010 | 55.0236 | 67.0937 | 92.4576 | 80.2540 | 62.0000 | 15.0000 |
| vis_weights_m3_none_scale2.0 | 19.2260 | 0.0977 | 41.2339 | 0.0280 | 55.0682 | 67.1742 | 93.0809 | 82.4037 | 62.9525 | 16.2117 |

## large_disparity

| Setting | Abs mm | Delta Abs vs vis | Acc2 % | Delta Acc2 pp | Acc4 % | Acc8 % | S2 coverage % | S3 coverage % | S2 width/base | S3 width/base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vis | 10.9525 | 0.0000 | 72.4265 | 0.0000 | 81.1569 | 86.0799 | 91.9996 | 89.7636 | 62.0000 | 15.0000 |
| vis_weights_m3_none_scale1.0 | 10.9526 | 0.0001 | 72.4265 | 0.0000 | 81.1569 | 86.0797 | 91.9995 | 89.7636 | 62.0000 | 15.0000 |
| vis_weights_m3_none_scale2.0 | 10.9357 | -0.0168 | 72.4533 | 0.0268 | 81.1875 | 86.1308 | 92.3070 | 90.3185 | 62.5247 | 15.4218 |

## occluded_any

| Setting | Abs mm | Delta Abs vs vis | Acc2 % | Delta Acc2 pp | Acc4 % | Acc8 % | S2 coverage % | S3 coverage % | S2 width/base | S3 width/base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vis | 21.1716 | 0.0000 | 46.5677 | 0.0000 | 59.0919 | 69.3838 | 90.3115 | 79.6775 | 62.0000 | 15.0000 |
| vis_weights_m3_none_scale1.0 | 21.1719 | 0.0002 | 46.5678 | 0.0002 | 59.0936 | 69.3834 | 90.3115 | 79.6775 | 62.0000 | 15.0000 |
| vis_weights_m3_none_scale2.0 | 21.2339 | 0.0623 | 46.6059 | 0.0382 | 59.1584 | 69.4768 | 90.9782 | 81.2995 | 63.1049 | 16.0444 |

## occluded_majority

| Setting | Abs mm | Delta Abs vs vis | Acc2 % | Delta Acc2 pp | Acc4 % | Acc8 % | S2 coverage % | S3 coverage % | S2 width/base | S3 width/base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vis | 30.3983 | 0.0000 | 36.9160 | 0.0000 | 48.4758 | 59.1521 | 87.0164 | 71.5129 | 62.0000 | 15.0000 |
| vis_weights_m3_none_scale1.0 | 30.3986 | 0.0003 | 36.9166 | 0.0006 | 48.4793 | 59.1521 | 87.0159 | 71.5128 | 62.0000 | 15.0000 |
| vis_weights_m3_none_scale2.0 | 30.5248 | 0.1265 | 36.9595 | 0.0435 | 48.5635 | 59.2840 | 87.9249 | 73.6836 | 63.5299 | 16.4358 |

## large_disp_and_occluded

| Setting | Abs mm | Delta Abs vs vis | Acc2 % | Delta Acc2 pp | Acc4 % | Acc8 % | S2 coverage % | S3 coverage % | S2 width/base | S3 width/base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vis | 31.7532 | 0.0000 | 44.6854 | 0.0000 | 56.3203 | 65.2083 | 83.1770 | 74.0478 | 62.0000 | 15.0000 |
| vis_weights_m3_none_scale1.0 | 31.7536 | 0.0004 | 44.6859 | 0.0005 | 56.3208 | 65.2085 | 83.1765 | 74.0480 | 62.0000 | 15.0000 |
| vis_weights_m3_none_scale2.0 | 31.7358 | -0.0175 | 44.7660 | 0.0805 | 56.4400 | 65.4136 | 84.1813 | 75.8363 | 63.6185 | 16.3885 |

## boundary_and_occluded

| Setting | Abs mm | Delta Abs vs vis | Acc2 % | Delta Acc2 pp | Acc4 % | Acc8 % | S2 coverage % | S3 coverage % | S2 width/base | S3 width/base |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vis | 28.1604 | 0.0000 | 31.3994 | 0.0000 | 43.6362 | 56.0985 | 88.9750 | 71.5980 | 62.0000 | 15.0000 |
| vis_weights_m3_none_scale1.0 | 28.1606 | 0.0002 | 31.3991 | -0.0003 | 43.6385 | 56.0982 | 88.9753 | 71.5984 | 62.0000 | 15.0000 |
| vis_weights_m3_none_scale2.0 | 28.3264 | 0.1660 | 31.4322 | 0.0328 | 43.7043 | 56.2076 | 89.8356 | 74.3806 | 63.3015 | 16.6873 |
