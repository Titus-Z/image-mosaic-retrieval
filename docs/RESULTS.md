# Historical experiment results

The table below was transcribed from the original course report. The experiment
used a high-resolution target and an approximately 50,000-image Tiny
ImageNet-derived gallery. The source dataset, output images, hardware details,
and exact environment are not included in the public repository, so these rows
must be read as historical evidence rather than a currently reproducible
benchmark.

| Run | Feature weights | PSNR | SSIM | LPIPS | Time (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| ex 01 | CNN 1.0 | 17.60 | **0.4227** | 0.2656 | 142.25 |
| ex 02 | Color 1.0 | 17.23 | 0.4117 | 0.2580 | 42.75 |
| ex 06 | HOG 0.1, Color 0.9 | 17.26 | 0.4076 | 0.2563 | 93.52 |
| ex 04 | Edge 0.3, Color 0.7 | **18.79** | 0.4072 | 0.2494 | 54.82 |
| ex 00 | Color 0.6, HOG 0.3, Edge 0.1 | 17.50 | 0.3999 | 0.2898 | 106.73 |
| ex 05 | CNN 0.5, Color 0.5 | 17.66 | 0.3912 | **0.2249** | 197.12 |
| ex 03 | HOG 0.5, Color 0.5 | 16.88 | 0.3856 | 0.3623 | 89.01 |

## What the table supports

- Shallow CNN features produced the highest SSIM in this run.
- Edge plus color produced the highest PSNR and was substantially faster than
  the CNN-only run.
- CNN plus color produced the lowest LPIPS but also the longest runtime.
- Metric winners disagree, so declaring one universally best configuration
  would be unjustified.

## What the table does not support

- Generalization to other target images or galleries
- Statistical significance
- A current FAISS-versus-KMeans speed ratio on modern hardware
- Exact reproduction without the original data and environment

The public synthetic demo is designed to validate pipeline behavior. It is not
an attempt to reproduce these quality numbers.
