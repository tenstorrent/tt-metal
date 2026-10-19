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
| branch | `adamjanovic/perf-example-improvements` (base `adce58258c3`) |
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
| commit | `f74ee7aad64` + arch-agnostic regimes tests (branch `adamjanovic/ab-tilize-run-scoreboard-catalog`), Release build |
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
