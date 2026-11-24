# row_major_flat_tiles — l1_sharded reduce (x - mean_row), tilize_untilize vs flat_tiles

box=bh-50-special-adamjanovic-for-reservation-177924  arch=BH  grid=13x10  clock=unknown MHz  N=3 (median)  kernel-iters=1 (per-launch latency)  date=2026-11-24  sha=e3f6587e285

HEIGHT_SHARDED L1 ROW_MAJOR bf16 [H, W] in and out, S sticks per Tensix core, H / S Tensix cores. Baseline: stage each shard into 32-stick tile-rows (zero rows past S, zero columns past W), tilize, REDUCE_ROW mean, subtract, untilize, copy back. Lever: flat tiles in the mode W picks.

| H | W | S | Tensix cores | flat mode | tilize_untilize ns | flat_tiles ns | speedup |
|---:|---:|---:|---:|---|---:|---:|---:|
| 32 | 4096 | 1 | 32 | whole | 18705 | 1173 | 15.95x |
| 64 | 8192 | 1 | 64 | whole | 32982 | 1349 | 24.45x |
| 256 | 4096 | 4 | 64 | whole | 18978 | 2549 | 7.45x |
| 512 | 2048 | 8 | 64 | whole | 12133 | 2973 | 4.08x |
| 2048 | 2048 | 32 | 64 | whole | 13424 | 9200 | 1.46x |
| 8192 | 1024 | 128 | 64 | whole | 21644 | 19385 | 1.12x |
| 8192 | 8 | 128 | 64 | narrow | 6057 | 6649 | 0.91x |
| 32768 | 8 | 512 | 64 | narrow | 18576 | 7087 | 2.62x |
| 65536 | 8 | 1024 | 64 | narrow | 35472 | 7721 | 4.59x |
| 4096 | 16 | 64 | 64 | narrow | 3913 | 11447 | 0.34x |
| 8192 | 16 | 128 | 64 | narrow | 6016 | 11625 | 0.52x |
| 32768 | 16 | 512 | 64 | narrow | 18609 | 12507 | 1.49x |
| 131072 | 16 | 2048 | 64 | narrow | 69141 | 16364 | 4.23x |
| 64 | 544 | 1 | 64 | padded | 5247 | 1324 | 3.96x |
| 256 | 544 | 4 | 64 | padded | 5339 | 2207 | 2.42x |
| 512 | 544 | 8 | 64 | padded | 5487 | 3415 | 1.61x |
| 1024 | 544 | 16 | 64 | padded | 5702 | 5762 | 0.99x |
| 2048 | 544 | 32 | 64 | padded | 6212 | 10481 | 0.59x |
| 64 | 1056 | 1 | 64 | padded | 7618 | 1584 | 4.81x |
| 512 | 1056 | 8 | 64 | padded | 7854 | 5424 | 1.45x |
| 2048 | 1056 | 32 | 64 | padded | 8681 | 14045 | 0.62x |
| 64 | 96 | 1 | 64 | padded | 2228 | 1283 | 1.74x |
| 512 | 96 | 8 | 64 | padded | 2352 | 3373 | 0.70x |
| 2048 | 96 | 32 | 64 | padded | 3161 | 10467 | 0.30x |
