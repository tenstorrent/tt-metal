# width_split — measured report

| stamp | value |
|---|---|
| box | `bgd-lab-16-special-dstoiljkovic-for-reservation-44175` |
| arch | Wormhole B0 (8×8 = 64 compute grid) |
| commit | `6f13238bd69` |
| date | 2026-07-24 |
| metric | `DEVICE KERNEL DURATION [ns]`, read **in-process** (`ttnn.ReadDeviceProfiler` + `ttnn.get_latest_programs_perf_data`) |
| method | 5 warmup + 20 timed launches per case, flush-bracketed, on-device duration averaged; whole `width × variant` matrix in one device session |

> Numbers are illustrative of the *effect*, not a CI bound — single-box, single-arch.
> Re-run `python -m ttnn.operations.examples.width_split` to measure your own widths.
> A different arch (e.g. Blackhole) should be **appended** as a new block, not overwritten.

The concept is one thing: a **wide, short tensor has no parallelism along its height**, so a
"split by tile-rows" strategy strands all the work on one core. **Width-splitting** — handing each
core a contiguous range of tile-columns — fills the grid. Compute is held trivial (one relu per
tile), inputs are interleaved DRAM, so the measured delta is purely **work distribution**: how many
cores the `Wt` tiles are spread over. `single_core` is what a tile-row split degenerates to for a
one-tile-row-tall tensor; `width_split` spreads the tiles over `min(Wt, grid)` cores.

## bfloat16, H=32 (one tile-row), interleaved DRAM, iters=1, trials=20

|   W  | Wt | variant | cores | ns/op | speedup vs single_core |
|-----:|---:|---|---:|---:|---:|
|   32 |  1 | single_core | 1 | 1817.0 | (base) |
|   32 |  1 | width_split | 1 | 1822.0 | 1.00× |
|  256 |  8 | single_core | 1 | 4292.4 | (base) |
|  256 |  8 | width_split | 8 | 1919.0 | **2.24×** |
| 1024 | 32 | single_core | 1 | 9613.2 | (base) |
| 1024 | 32 | width_split | 32 | 2262.8 | **4.25×** |
| 2048 | 64 | single_core | 1 | 16655.2 | (base) |
| 2048 | 64 | width_split | 64 | 2665.9 | **6.25×** |
| 4096 | 128 | single_core | 1 | 30885.2 | (base) |
| 4096 | 128 | width_split | 64 | 4202.7 | **7.35×** |
| 8192 | 256 | single_core | 1 | 59260.8 | (base) |
| 8192 | 256 | width_split | 64 | 7636.6 | **7.76×** |

### Reading it
- **Wt=1 → no benefit (1.00×).** A single tile-column can't be split — width_split also lands on 1
  core, so it correctly matches the baseline. This is the crossover: below the point where there's
  more than one chunk to spread, width-splitting has nothing to do.
- **Speedup scales with the grid as Wt grows.** Wt=8 → 8 cores → 2.24×; Wt=32 → 32 cores → 4.25×;
  Wt=64 → the full 64-core grid → 6.25×. `single_core` time grows ~linearly with Wt (one core does
  every tile serially); `width_split` stays nearly flat while it still has spare cores.
- **Saturation at the grid (64 cores).** Beyond Wt=64 every core is busy, so `width_split` time
  starts rising too (each core now owns >1 tile), but the gap keeps widening (7.35× → 7.76×) because
  `single_core` keeps growing linearly while `width_split` grows at 1/64 the rate.

**Takeaway:** for a wide-short tensor, splitting along the width to fill the grid is up to **~7.8×**
faster than the single-core (tile-row-split) baseline on this 64-core WH B0, and the benefit grows
with width. The only regime where it doesn't help is `Wt ≤ 1` (nothing to split).

## 2026-10-12 — helps / does-not-help calibration (Wormhole B0 n150)

| stamp | value |
|---|---|
| board | Wormhole B0 n150 (8×8 = 64 Tensix cores) |
| base commit | `adce58258c3` |
| test | `test_width_split_regimes` (bf16, H=32, interleaved DRAM) |
| method | 5 warmup launches per variant; then 12 trials per variant, interleaved single_core/width_split; a trial = mean `DEVICE KERNEL DURATION [ns]` over 5 launches (one profiler window); median compared |

Run 1 (min–max over the 12 trials):

| regime | W | Wt (tiles) | width_split Tensix cores | single_core median ns | width_split median ns | speedup |
|---|--:|--:|--:|---|---|--:|
| helps_wt64 | 2048 | 64 | 64 | 16831.8 (16766.0–16880.6) | 2719.9 (2635.2–2796.2) | 6.19× |
| does_not_help_wt2 | 64 | 2 | 2 | 2075.9 (2042.8–2113.2) | 2105.3 (1994.8–2513.6) | 0.99× |

Speedups over four runs: helps_wt64 6.19 / 6.10 / 6.07 / 6.06×; does_not_help_wt2 0.99 / 0.91 / 0.98 / 0.89×.
The Wt=2 width_split trials are bimodal (≈2.0 µs or ≈2.4 µs), which is why its median moves between runs.

Crossover sweep (`test_width_split_device_perf`, `WS_WIDTHS=32,64,96,128,256,2048`, mean of 20 launches):

| W | Wt | single_core ns | width_split ns | speedup |
|--:|--:|--:|--:|--:|
| 32 | 1 | 1823.5 | 1828.2 | 1.00× |
| 64 | 2 | 2106.2 | 2159.1 | 0.98× |
| 96 | 3 | 2512.8 | 1933.6 | 1.30× |
| 128 | 4 | 2872.9 | 1966.0 | 1.46× |
| 256 | 8 | 4305.4 | 1932.2 | 2.23× |
| 2048 | 64 | 16800.8 | 2778.7 | 6.05× |

**Correction:** the earlier "only `Wt ≤ 1` does not benefit" is too optimistic — `Wt = 2` is also flat
or slower (one 8-tile read block on one Tensix core is already at the ~1.8 µs launch floor). The split
pays from `Wt = 3`.

## Blackhole p150b — bh-qb-15-special-adamjanovic-for-reservation-139410 (2026-10-19)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid 12×10 = 120 Tensix cores, 8 DRAM banks, AICLK 1350 MHz |
| commit | `f74ee7aad64` + arch-agnostic regimes tests, Release build |
| metric / method | `DEVICE KERNEL DURATION [ns]`, same as the blocks above: `test_width_split_device_perf` (mean of 20 launches per cell) and `test_width_split_regimes` (5 warmup launches; 12 trials per variant interleaved, a trial = mean of 5 launches; medians compared) |

### Sweep (`test_width_split_device_perf`) — bfloat16, H=32 (one tile-row), interleaved DRAM, iters=1, trials=20

|   W  | Wt (tiles) | variant | Tensix cores | ns/op | speedup vs single_core |
|-----:|---:|---|---:|---:|---:|
|   32 |   1 | single_core | 1 | 1025.4 | (base) |
|   32 |   1 | width_split | 1 | 1010.2 | 1.02× |
|  256 |   8 | single_core | 1 | 2480.2 | (base) |
|  256 |   8 | width_split | 8 | 1144.9 | **2.17×** |
| 1024 |  32 | single_core | 1 | 6159.8 | (base) |
| 1024 |  32 | width_split | 32 | 1239.5 | **4.97×** |
| 2048 |  64 | single_core | 1 | 11074.0 | (base) |
| 2048 |  64 | width_split | 64 | 1533.5 | **7.22×** |
| 4096 | 128 | single_core | 1 | 20885.5 | (base) |
| 4096 | 128 | width_split | 120 | 1866.8 | **11.19×** |
| 8192 | 256 | single_core | 1 | 40518.2 | (base) |
| 8192 | 256 | width_split | 120 | 2961.2 | **13.68×** |

Boundary probe (same interleaved A/B as the regimes test, ratios only): Wt = 2 tiles 1.17×, Wt = 3 tiles 1.36×,
Wt = 4 tiles 1.54×.

### Regimes (`test_width_split_regimes` on BH p150b)

| regime | W | Wt (tiles) | width_split Tensix cores | single_core median ns | width_split median ns | speedup | current test asserts |
|---|--:|--:|--:|---|---|--:|---|
| helps_wt64 | 2048 | 64 | 64 | 11092.4 (11078.4–11112.2) | 1536.4 (1513.6–1548.8) | 7.22× | ≥ 3.5× |
| measure_wt2 | 64 | 2 | 2 | 1217.1 (1206.0–1237.8) | 1031.2 (1023.2–1039.6) | 1.18× | measure (logged only — verdict differs by arch) — WH 0.99× |
| does_not_help_wt1 | 32 | 1 | 1 | 1025.4 | 1010.2 | 1.02× (device_perf sweep) | ≤ 1.05× |

### Reading it
- **Wt = 2 tiles already pays on BH p150b (1.17–1.18×)**, where WH B0 n150 is flat (0.89–0.99×). The launch
  floor is ~1.0 µs (Wt = 1 tile: 1025 ns) versus ~1.8 µs on WH B0, and the second tile adds ~190 ns to
  `single_core`; that saving survives the second Tensix core's startup on BH p150b and does not on WH B0.
  Only Wt = 1 tile (nothing to split) is flat on BH p150b.
- **The grid keeps scaling past 64 Tensix cores.** WH B0 n150 saturates at 64 Tensix cores (7.35× at
  Wt = 128 tiles); BH p150b uses 120 Tensix cores and reaches 11.19× at Wt = 128 tiles and 13.68× at 256 tiles.
- **Per-tile cost** on `single_core` is ~160 ns on BH p150b vs ~235 ns on WH B0 n150, so the absolute times
  are ~1.5× lower on BH p150b at every width; at equal Tensix-core count (Wt = 8–64 tiles) the speedups are
  similar (BH p150b 2.17–7.22× vs WH B0 n150 2.24–6.25×).


## Reduce case (per-row combine) — Blackhole p150b, bh-50-special-adamjanovic-for-reservation-143315 (2026-10-29)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid **13×10 = 130 Tensix cores**, AICLK max 1350 MHz |
| commit | `8104cced9bc` + the reduce-case variants, Release build |
| op | `y = x − mean_W(x)` per row; bf16 `Layout::TILE`, `TensorMemoryLayout::INTERLEAVED` DRAM in and out, shape `[32·R_t, 32·Wt]` |
| compute | HiFi4 `REDUCE_ROW` (scaler 1/W, fp32 DST) + `sub_tiles_bcast_cols`, 4 tiles per DST block; slice resident in L1 between the two |
| combine | `k > 1`: symmetric all-gather of the 1-tile partials in one round (faces 0 and 2 only: 2 × 512 B writes per member per destination, flush then semaphore increment), sum in member order with FPU accumulate-to-DST |
| metric / method | `DEVICE KERNEL DURATION [ns]`; sweep `test_width_split_reduce_device_perf` = 5 warmup + mean of 20 launches per cell, correctness asserted per cell; regimes `test_width_split_regimes` = 12 trials per variant interleaved, a trial = mean of 5 launches, medians |
| variants | `row_split_reduce` (baseline): whole tile-rows per Tensix core over min(R_t, 130) Tensix cores. `width_split_reduce`: groups of k consecutive Tensix cores (row-major), min(R_t, 130 // k) groups, slice = Wt / k tiles |

### Sweep — R_t × Wt × k (kept: k divides Wt, R_t · k ≤ 2 × 130)

DRAM GB/s = (input + output bytes) / time = 2 · R_t · Wt · 2048 B / ns. Bold: ≥ 1.10×.

| R_t (tile-rows) | Wt (tiles) | variant | k | Tensix cores | tiles per slice | ns/op | DRAM GB/s | vs k = 1 |
|--:|--:|---|--:|--:|--:|--:|--:|--:|
| 1 | 4 | row_split_reduce | 1 | 1 | 4 | 2083.1 | 7.9 | (base) |
| 1 | 4 | width_split_reduce | 2 | 2 | 2 | 2299.1 | 7.1 | 0.91× |
| 1 | 4 | width_split_reduce | 4 | 4 | 1 | 2421.6 | 6.8 | 0.86× |
| 1 | 8 | row_split_reduce | 1 | 1 | 8 | 3017.6 | 10.9 | (base) |
| 1 | 8 | width_split_reduce | 2 | 2 | 4 | 2571.0 | 12.7 | **1.17×** |
| 1 | 8 | width_split_reduce | 4 | 4 | 2 | 2573.1 | 12.7 | **1.17×** |
| 1 | 8 | width_split_reduce | 8 | 8 | 1 | 3102.3 | 10.6 | 0.97× |
| 1 | 16 | row_split_reduce | 1 | 1 | 16 | 4899.9 | 13.4 | (base) |
| 1 | 16 | width_split_reduce | 2 | 2 | 8 | 3534.1 | 18.5 | **1.39×** |
| 1 | 16 | width_split_reduce | 4 | 4 | 4 | 2858.9 | 22.9 | **1.71×** |
| 1 | 16 | width_split_reduce | 8 | 8 | 2 | 3308.8 | 19.8 | **1.48×** |
| 1 | 16 | width_split_reduce | 16 | 16 | 1 | 5499.1 | 11.9 | 0.89× |
| 1 | 32 | row_split_reduce | 1 | 1 | 32 | 8566.5 | 15.3 | (base) |
| 1 | 32 | width_split_reduce | 2 | 2 | 16 | 5329.9 | 24.6 | **1.61×** |
| 1 | 32 | width_split_reduce | 4 | 4 | 8 | 3928.4 | 33.4 | **2.18×** |
| 1 | 32 | width_split_reduce | 8 | 8 | 4 | 3566.7 | 36.7 | **2.40×** |
| 1 | 32 | width_split_reduce | 16 | 16 | 2 | 5721.6 | 22.9 | **1.50×** |
| 1 | 32 | width_split_reduce | 32 | 32 | 1 | 10091.2 | 13.0 | 0.85× |
| 1 | 64 | row_split_reduce | 1 | 1 | 64 | 16004.4 | 16.4 | (base) |
| 1 | 64 | width_split_reduce | 2 | 2 | 32 | 9064.7 | 28.9 | **1.77×** |
| 1 | 64 | width_split_reduce | 4 | 4 | 16 | 5779.8 | 45.4 | **2.77×** |
| 1 | 64 | width_split_reduce | 8 | 8 | 8 | 4633.1 | 56.6 | **3.45×** |
| 1 | 64 | width_split_reduce | 16 | 16 | 4 | 6089.0 | 43.1 | **2.63×** |
| 1 | 64 | width_split_reduce | 32 | 32 | 2 | 10403.0 | 25.2 | **1.54×** |
| 1 | 64 | width_split_reduce | 64 | 64 | 1 | 19465.8 | 13.5 | 0.82× |
| 16 | 4 | row_split_reduce | 1 | 16 | 4 | 2542.6 | 103.1 | (base) |
| 16 | 4 | width_split_reduce | 2 | 32 | 2 | 2675.8 | 98.0 | 0.95× |
| 16 | 4 | width_split_reduce | 4 | 64 | 1 | 2842.8 | 92.2 | 0.89× |
| 16 | 8 | row_split_reduce | 1 | 16 | 8 | 3742.1 | 140.1 | (base) |
| 16 | 8 | width_split_reduce | 2 | 32 | 4 | 3306.0 | 158.6 | **1.13×** |
| 16 | 8 | width_split_reduce | 4 | 64 | 2 | 3392.9 | 154.5 | **1.10×** |
| 16 | 8 | width_split_reduce | 8 | 128 | 1 | 4098.5 | 127.9 | 0.91× |
| 16 | 16 | row_split_reduce | 1 | 16 | 16 | 5869.3 | 178.7 | (base) |
| 16 | 16 | width_split_reduce | 2 | 32 | 8 | 5265.3 | 199.1 | **1.11×** |
| 16 | 16 | width_split_reduce | 4 | 64 | 4 | 4915.8 | 213.3 | **1.19×** |
| 16 | 16 | width_split_reduce | 8 | 128 | 2 | 5149.6 | 203.6 | **1.14×** |
| 16 | 16 | width_split_reduce | 16 | 128 | 1 | 11194.4 | 93.7 | 0.52× |
| 16 | 32 | row_split_reduce | 1 | 16 | 32 | 9674.0 | 216.8 | (base) |
| 16 | 32 | width_split_reduce | 2 | 32 | 16 | 8014.2 | 261.7 | **1.21×** |
| 16 | 32 | width_split_reduce | 4 | 64 | 8 | 8114.9 | 258.4 | **1.19×** |
| 16 | 32 | width_split_reduce | 8 | 128 | 4 | 8347.2 | 251.2 | **1.16×** |
| 16 | 32 | width_split_reduce | 16 | 128 | 2 | 12579.4 | 166.7 | 0.77× |
| 16 | 64 | row_split_reduce | 1 | 16 | 64 | 17834.0 | 235.2 | (base) |
| 16 | 64 | width_split_reduce | 2 | 32 | 32 | 13986.9 | 299.9 | **1.28×** |
| 16 | 64 | width_split_reduce | 4 | 64 | 16 | 14424.0 | 290.8 | **1.24×** |
| 16 | 64 | width_split_reduce | 8 | 128 | 8 | 14832.5 | 282.8 | **1.20×** |
| 16 | 64 | width_split_reduce | 16 | 128 | 4 | 17902.7 | 234.3 | 1.00× |
| 64 | 4 | row_split_reduce | 1 | 64 | 4 | 3658.6 | 286.6 | (base) |
| 64 | 4 | width_split_reduce | 2 | 128 | 2 | 4506.7 | 232.7 | 0.81× |
| 64 | 4 | width_split_reduce | 4 | 128 | 1 | 5340.2 | 196.4 | 0.69× |
| 64 | 8 | row_split_reduce | 1 | 64 | 8 | 7217.4 | 290.6 | (base) |
| 64 | 8 | width_split_reduce | 2 | 128 | 4 | 7010.9 | 299.1 | 1.03× |
| 64 | 8 | width_split_reduce | 4 | 128 | 2 | 6894.6 | 304.2 | 1.05× |
| 64 | 16 | row_split_reduce | 1 | 64 | 16 | 13533.6 | 309.9 | (base) |
| 64 | 16 | width_split_reduce | 2 | 128 | 8 | 13875.0 | 302.3 | 0.98× |
| 64 | 16 | width_split_reduce | 4 | 128 | 4 | 13172.4 | 318.4 | 1.03× |
| 64 | 32 | row_split_reduce | 1 | 64 | 32 | 25725.2 | 326.1 | (base) |
| 64 | 32 | width_split_reduce | 2 | 128 | 16 | 25396.5 | 330.3 | 1.01× |
| 64 | 32 | width_split_reduce | 4 | 128 | 8 | 25602.8 | 327.6 | 1.00× |
| 64 | 64 | row_split_reduce | 1 | 64 | 64 | 50874.8 | 329.8 | (base) |
| 64 | 64 | width_split_reduce | 2 | 128 | 32 | 46477.1 | 361.0 | 1.09× |
| 64 | 64 | width_split_reduce | 4 | 128 | 16 | 47102.7 | 356.2 | 1.08× |
| 128 | 4 | row_split_reduce | 1 | 128 | 4 | 7048.9 | 297.5 | (base) |
| 128 | 4 | width_split_reduce | 2 | 130 | 2 | 8182.1 | 256.3 | 0.86× |
| 128 | 8 | row_split_reduce | 1 | 128 | 8 | 14122.8 | 297.0 | (base) |
| 128 | 8 | width_split_reduce | 2 | 130 | 4 | 14015.0 | 299.3 | 1.01× |
| 128 | 16 | row_split_reduce | 1 | 128 | 16 | 25309.7 | 331.4 | (base) |
| 128 | 16 | width_split_reduce | 2 | 130 | 8 | 25693.0 | 326.5 | 0.99× |
| 128 | 32 | row_split_reduce | 1 | 128 | 32 | 46022.2 | 364.5 | (base) |
| 128 | 32 | width_split_reduce | 2 | 130 | 16 | 47519.0 | 353.1 | 0.97× |
| 128 | 64 | row_split_reduce | 1 | 128 | 64 | 122388.2 | 274.2 | (base) |
| 128 | 64 | width_split_reduce | 2 | 130 | 32 | 114530.1 | 293.0 | 1.07× |

`R_t = 128, Wt = 64` `row_split_reduce` measured 90 295 ns in an earlier identical sweep (122 388 ns here) — the
DRAM-bound large cells move ±15% between runs; the `width_split_reduce` / baseline ratio stayed 1.00–1.07×.

### Regimes (`test_width_split_regimes`, this board)

| regime | R_t | Wt | k | Tensix cores | tiles per slice | row_split_reduce median ns | width_split_reduce median ns | speedup (runs) | test asserts |
|---|--:|--:|--:|---|--:|---|---|---|---|
| reduce_helps_rt1_wt32_k8 | 1 | 32 | 8 | 1 → 8 | 4 | 8 737.0 (8 692.0–8 759.8) | 3 697.7 (3 668.2–3 757.8) | 2.36× (2.34–2.37, 5 runs) | ≥ 1.68× |
| reduce_does_not_help_rt64_wt16_k2 | 64 | 16 | 2 | 64 → 128 | 8 | 13 583.7 (13 417.2–13 691.4) | 13 833.7 (13 609.0–13 985.2) | 0.98× (0.98–0.99, 2 runs) | ≤ 1.05× |
| reduce_measure_rt64_wt8_k2 | 64 | 8 | 2 | 64 → 128 | 4 | 7 233.6 (6 996.4–7 339.6) | 6 956.8 (6 889.8–7 069.8) | 1.04× (1.03–1.04, 5 runs) | measure (logged only) |

The elementwise regimes on this 130-Tensix-core board: helps_wt64 7.08–7.11×, measure_wt2 1.18×,
does_not_help_wt1 1.00–1.01× (same verdicts as the 120-Tensix-core p150b block above).

### Combine cost, isolated

Same 4 tiles per Tensix core, so the extra time over `k = 1, Wt = 4` is the combine:

| k | Wt | ns | combine ≈ |
|--:|--:|--:|--:|
| 1 | 4 | 2 083.1 | — |
| 2 | 8 | 2 571.0 | 0.49 µs |
| 4 | 16 | 2 858.9 | 0.78 µs |
| 8 | 32 | 3 566.7 | 1.48 µs |
| 16 | 64 | 6 089.0 | 4.0 µs |

Partial size, measured: replacing the two 512 B face writes by one 1536 B write (faces 0–2) gave
`R_t = 1`: k = 2, Wt = 4 2 244 vs 2 270 ns (−1%); k = 16, Wt = 16 6 271 vs 5 491 ns (+14%); k = 64, Wt = 64
24 672 vs 19 408 ns (+27%). At large k each member ingests k partials, so bytes per partial dominate; the
example keeps the two-face (1 KB) partial.

Precision note: the first combine summed with a dest-reuse add (DST → SrcA each step); SrcA's 19-bit format
truncated the running sum by ~1 ulp per add, max abs error 0.019 / 0.042 / 0.067 / 0.17 at k = 8 / 16 / 32 /
64. Summing pairs straight from the CB with FPU accumulate-to-DST: ≤ 0.010 at every k (the test asserts < 0.03).
A bf16 DST (no fp32 accumulation) failed `R_t = 1, Wt = 64, k = 1` at PCC 0.9966: the row-sum across 64 tiles
loses the small per-tile increments.

### Reading it
- **Grid nearly empty (`R_t = 1`): the split pays from Wt = 8** (k = 2, 4 tiles per slice, 1.17×) and grows
  with Wt: 1.71× at Wt = 16 (k = 4), 2.40× at Wt = 32 (k = 8), 3.45× at Wt = 64 (k = 8). Best slice is 4–8 tiles.
  Wt = 4 loses at every k (0.86–0.91×): the ~0.5 µs combine is more than the 2 tiles × ~230 ns it saves.
- **1-tile slices always lose** (0.52–0.97×), and **k ≥ 16** loses its gain: the all-gather's cost grows with k
  (each member sends and receives k partials), ~4 µs at k = 16, ~19 µs at k = 64.
- **Tile-rows already on ~half the grid (`R_t = 64`): flat, 0.98–1.09× for 4–32 tiles per slice.** The 64-Tensix-core
  baseline already moves 287–330 GB/s, ~70–80% of the ~415–430 GB/s DRAM read + write ceiling; doubling to 128
  Tensix cores halves per-core work but not the bytes. `R_t = 128` (128 of 130 Tensix cores): 0.86–1.07×.
- **`R_t = 16` (16 of 130 Tensix cores): small gains, 1.11–1.28× at k = 2–8** with ≥ 4 tiles per slice; the baseline
  already reaches 140–235 GB/s at Wt ≥ 8, so the DRAM ceiling caps the gain long before the grid runs out.
- **Not measured:** splitting a `Layout::ROW_MAJOR` stick (512 B → 2 × 256 B doubles the DRAM transfer count) —
  that is a transfer-size concept, outside this example's two.

## Reduce case on ROW_MAJOR sticks — Blackhole p150b, bh-50-special-adamjanovic-for-reservation-143315 (2026-10-30)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid **13×10 = 130 Tensix cores**, AICLK max 1350 MHz |
| commit | `30351ce33f5` + the ROW_MAJOR reduce path (uncommitted), Release build |
| op | the same `y = x − mean_W(x)`; bf16 `Layout::ROW_MAJOR`, `TensorMemoryLayout::INTERLEAVED` DRAM in and out, shape `[32·R_t, W]`; one DRAM page = one stick of W datums = 2·W bytes |
| kernels | `ws_reduce_*.cpp` built with `WS_ROW_MAJOR`: per tile-row, the reader issues **32 reads of 2·W / k bytes** (its piece of each stick) and one barrier; compute tilizes the slice (S = Wt / k tiles), reduces / combines / applies exactly as the TILE path, untilizes; the writer issues 32 writes of 2·W / k bytes and one barrier. Combine unchanged (1 KB partials, one symmetric round) |
| metric / method | `DEVICE KERNEL DURATION [ns]`; sweep `test_width_split_reduce_rm_device_perf` = 5 warmup + mean of 20 launches per cell, correctness asserted per cell; regimes `test_width_split_regimes` = 12 trials per variant interleaved, a trial = mean of 5 launches, medians |
| variants | `row_split_reduce` (baseline, k = 1): whole sticks, whole tile-rows per Tensix core. `width_split_reduce`: groups of k Tensix cores, each owning a 2·W / k byte piece of every stick of its tile-rows |

### Sweep — R_t × W × k (k ∈ {1, 2, 4}, kept: k divides Wt, R_t · k ≤ 2 × 130)

Speedup over `k = 1`; in brackets the bytes per DRAM transfer (stick piece). DRAM transfers per launch
= 2 · 32 · R_t · k (read + write), whatever the stick width.

| R_t | W (stick B) | `k = 1` ns (Tensix cores) | k = 2 | k = 4 |
|--:|--:|--:|--:|--:|
| 1 | 64 (128) | 4 203 (1) | 0.90× (64 B) | |
| 1 | 128 (256) | 4 439 (1) | 0.94× (128 B) | 0.92× (64 B) |
| 1 | 256 (512) | 4 976 (1) | 1.01× (256 B) | 1.02× (128 B) |
| 1 | 512 (1 024) | 5 935 (1) | 1.09× (512 B) | 1.15× (256 B) |
| 1 | 1 024 (2 048) | 7 846 (1) | 1.22× (1 024 B) | 1.36× (512 B) |
| 1 | 2 048 (4 096) | 12 626 (1) | 1.51× (2 048 B) | **1.88×** (1 024 B) |
| 4 | 64 (128) | 4 272 (4) | 0.91× | |
| 4 | 128 (256) | 4 479 (4) | 0.94× | 0.92× |
| 4 | 256 (512) | 5 048 (4) | 1.01× | 1.02× |
| 4 | 512 (1 024) | 6 041 (4) | 1.08× | 1.15× |
| 4 | 1 024 (2 048) | 7 979 (4) | 1.21× | 1.37× |
| 4 | 2 048 (4 096) | 13 187 (4) | 1.38× | 1.70× |
| 16 | 64 (128) | 4 320 (16) | 0.82× | |
| 16 | 128 (256) | 4 531 (16) | 0.85× | 0.64× |
| 16 | 256 (512) | 5 107 (16) | 0.90× | 0.71× |
| 16 | 512 (1 024) | 6 893 (16) | 1.05× (512 B) | 0.90× (256 B) |
| 16 | 1 024 (2 048) | 11 375 (16) | 1.16× (1 024 B) | 1.22× (512 B) |
| 16 | 2 048 (4 096) | 20 283 (16) | 1.25× (2 048 B) | 1.38× (1 024 B) |
| 47 | 64 (128) | 5 652 (47) | 0.65× | |
| 47 | 128 (256) | 5 983 (47) | 0.66× | 0.40× |
| 47 | 256 (512) | 7 403 (47) | **0.69×** (256 B, 94 Tensix cores) | 0.44× |
| 47 | 512 (1 024) | 11 184 (47) | 0.82× | 0.61× |
| 47 | 1 024 (2 048) | 19 681 (47) | 0.98× | 0.83× |
| 47 | 2 048 (4 096) | 38 226 (47) | 1.08× (2 048 B) | 1.02× (1 024 B) |
| 64 | 64 (128) | 6 493 (64) | 0.61× | |
| 64 | 128 (256) | 6 936 (64) | 0.63× | 0.38× |
| 64 | 256 (512) | 8 424 (64) | **0.65×** (256 B, 128 Tensix cores) | 0.43× |
| 64 | 512 (1 024) | 13 594 (64) | 0.82× | 0.61× |
| 64 | 1 024 (2 048) | 24 961 (64) | 0.95× | 0.83× |
| 64 | 2 048 (4 096) | 50 530 (64) | 1.07× (2 048 B) | 1.02× (1 024 B) |
| 128 | 64 / 128 / 256 / 512 / 1 024 / 2 048 | 10 600 / 12 619 / 16 704 / 25 980 / 46 403 / 97 780 (128) | 0.57× / 0.62× / 0.72× / 0.86× / 0.94× / 1.08× (130 Tensix cores, 1–2 tile-rows per group) | |

R_t = 1 and 4 come from a second run of the same test (`WSRM_ROWS=1,4,16,64`); its R_t = 16 and 64 cells
matched the first run within 1% (largest: R_t = 64, W = 2 048, 1.07× both runs).

The field shapes: `2048 × 256` (R_t = 64, 512 B sticks) **0.65×** (k = 2 is 1.54× slower) and
`4 × 8 × 47 × 256` = `[1504, 256]` (R_t = 47) **0.69×** (1.45× slower) — the field's 1.3–1.5× slower.

### Regimes (`test_width_split_regimes`, this board)

| regime | R_t | W | k | Tensix cores | piece | row_split_reduce median ns | width_split_reduce median ns | speedup | test asserts |
|---|--:|--:|--:|---|--:|---|---|---|---|
| reduce_rm_helps_rt1_w2048_k4 | 1 | 2 048 | 4 | 1 → 4 | 1 024 B | 12 802.5 (12 751.6–12 827.2) | 6 871.6 (6 844.4–6 913.4) | 1.86× | ≥ 1.43× |
| reduce_rm_does_not_help_rt64_w256_k2 | 64 | 256 | 2 | 64 → 128 | 256 B | 8 566.1 (8 429.0–8 690.2) | 13 031.6 (12 899.0–13 182.0) | 0.66× | ≤ 1.05× |
| reduce_rm_measure_rt64_w2048_k2 | 64 | 2 048 | 2 | 64 → 128 | 2 048 B | 50 470.8 (50 107.8–50 806.6) | 46 952.5 (46 574.2–47 712.4) | 1.07× | measure (logged only) |

The six earlier regimes in the same run: helps_wt64 7.09×, measure_wt2 1.18×, does_not_help_wt1 1.00×,
reduce_helps_rt1_wt32_k8 2.36×, reduce_does_not_help_rt64_wt16_k2 0.98×, reduce_measure_rt64_wt8_k2 1.05×.

### What the transfer count costs, isolated

- **A Tensix core's transfer count does not shrink with k.** It issues 32 reads + 32 writes per tile-row
  for any k; only their size shrinks. `R_t = 1, W = 256`: k = 1 / 2 / 4 = 4 976 / 4 910 / 4 891 ns while the
  tiles per Tensix core go 8 → 4 → 2. (TILE path, same tiles: `R_t = 1, Wt = 8`, k = 2 is 1.17×, because
  there the per-core reads halve with the tiles.)
- **Each extra DRAM transfer costs the grid ~1 ns.** Bytes held constant, `R_t = 64, W = 128` (1 MB in +
  out): 4 096 / 8 192 / 16 384 transfers at k = 1 / 2 / 4 → 6 936 / 10 929 / 18 484 ns, **+0.97 and +0.92 ns
  per added transfer**. Per-core work held constant, Tensix cores doubled: `W = 64, k = 1`, R_t = 64 → 128
  (4 096 → 8 192 transfers of 128 B) 6 493 → 10 600 ns, +1.0 ns per transfer. So small transfers run at
  ~1 G transfers/s across the grid at the margin (0.5–0.9 G/s averaged over a launch: the "Mxfer/s" column
  of the sweep; the field's ~690 M/s sits in that range). At ~1 ns each, a transfer below ~400 B is
  count-bound against the ~415 GB/s DRAM ceiling.
- **The ROW_MAJOR baseline carries a larger fixed cost** than the TILE one: `R_t = 1, W = 64` 4.2 µs vs TILE
  `R_t = 1, Wt = 4` 2.1 µs (tilize + untilize, and one barrier per 32 sticks: compute cannot start a
  tile-row before all its 32 sticks arrived). Above it, ~150 ns per tile (`R_t = 1`, W = 1 024 → 2 048).
- Probe, not shipped: rotating each Tensix core's first stick (so the cores do not all start on the same DRAM
  bank) moved both variants by ≤ 5% and no speedup by more than 0.04×. The result is not a bank-order artifact.

### Reading it
- **The split pays only where the TILE rule already says so, and later in W.** Tile-rows on ≤ 4 of 130
  Tensix cores: flat at 512 B sticks (1.01–1.02×), pays from 1 KB sticks (1.08–1.15×), 1.36–1.37× at 2 KB, 1.51–1.88× at
  4 KB. R_t = 16: pays from 2 KB sticks (1.16–1.22×), 1 KB sticks only at k = 2 (512 B pieces, 1.05×). The
  TILE path pays from Wt = 8 tiles (W = 256) at R_t = 1.
- **Tile-rows on ≥ ~1/3 of the grid (R_t = 47, 64, 128): loses or is flat at every piece ≤ 1 KB (0.38–1.02×;
  the 1.02× is k = 4 on 4 KB sticks)**, ~flat at 2 KB pieces (1.07–1.08×, k = 2 on 4 KB sticks). Same place where the TILE path is
  flat (0.98–1.09×), but here the k-fold transfer count turns flat into a loss.
- **k = 4 beats k = 2 only while R_t ≤ 16 and pieces ≥ 256 B at R_t ≤ 4, ≥ 512 B at R_t = 16.**
