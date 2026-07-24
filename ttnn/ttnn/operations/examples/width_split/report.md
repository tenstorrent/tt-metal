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
