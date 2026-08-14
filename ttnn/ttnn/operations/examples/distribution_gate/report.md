# distribution_gate — measured report

| stamp | value |
|---|---|
| box | `bgd-lab-16-special-dstoiljkovic-for-reservation-44175` |
| arch | Wormhole B0 (8×8 = 64 compute grid) |
| commit | `b9e384c5753` |
| date | 2026-07-24 |
| metric | `DEVICE KERNEL DURATION [ns]`, read **in-process** (`ttnn.ReadDeviceProfiler` + `ttnn.get_latest_programs_perf_data`) |
| method | 5 warmup + 20 timed launches per case, flush-bracketed, on-device duration averaged; whole `shape × variant` matrix in one device session |

> Numbers are illustrative of the *effect*, not a CI bound — single-box, single-arch.
> Re-run `python -m ttnn.operations.examples.distribution_gate` to measure your own shapes.
> A different arch (e.g. Blackhole) should be **appended** as a new block, not overwritten.

Splitting a tile op across the grid means partitioning its tiles along **some** axis. A **height**
(tile-row) split fills the grid when there are many tile-rows but strands a **wide-short** tensor on one
core; a **width** (tile-column) split fills the grid when there are many tile-columns but strands a
**tall-narrow** tensor on one core. The trap is symmetric, so "fix wide-short by switching to a width
split" just trades one collapse for the other. Compute is held trivial (one relu per tile), inputs are
interleaved DRAM tile tensors, and all three variants share byte-identical kernels — so the measured
delta is purely **work distribution**: which tiles each core owns, and how many cores run.

## bfloat16, interleaved DRAM, relu, iters=1, trials=20

|   H  |   W  | Ht | Wt | variant | cores | ns/op | vs gated |
|-----:|-----:|---:|---:|---|---:|---:|---:|
|   32 | 4096 |  1 | 128 | height_split | 1  | 30921.5 | 7.25× |
|   32 | 4096 |  1 | 128 | width_split  | 64 |  4190.4 | 0.98× |
|   32 | 4096 |  1 | 128 | gated        | 64 |  4262.1 | (ref) |
| 2048 |   32 | 64 |  1 | height_split | 64 |  2724.6 | 1.00× |
| 2048 |   32 | 64 |  1 | width_split  | 1  | 16702.3 | 6.15× |
| 2048 |   32 | 64 |  1 | gated        | 64 |  2714.2 | (ref) |
| 2048 | 2048 | 64 | 64 | height_split | 64 | 90666.6 | 1.00× |
| 2048 | 2048 | 64 | 64 | width_split  | 64 | 83740.4 | 0.92× |
| 2048 | 2048 | 64 | 64 | gated        | 64 | 90665.1 | (ref) |
| 1024 | 1024 | 32 | 32 | height_split | 32 | 23481.8 | 1.00× |
| 1024 | 1024 | 32 | 32 | width_split  | 32 | 22220.7 | 0.94× |
| 1024 | 1024 | 32 | 32 | gated        | 32 | 23543.0 | (ref) |

### Reading it
- **Each fixed axis collapses on its bad regime.** `height_split` on wide-short `32×4096` (Ht=1) lands
  on **1 core → 30921 ns, 7.25× slower**. `width_split` on tall-narrow `2048×32` (Wt=1) lands on
  **1 core → 16702 ns, 6.15× slower**. Neither fixed choice is safe across aspect ratios.
- **`gated` fills the grid on both.** It diverts to width on wide-short (4262 ns, matches width_split)
  and stays on height for tall-narrow (2714 ns, matches height_split) — 64 cores in both.
- **No regression, measured.** On the two squares — where the default height split already saturates
  the grid — `gated` lands on the **same code path and the same number** as `height_split`
  (90665 vs 90666 ns; 23543 vs 23482 ns, within noise). The gate does not trip, so the baseline path
  is untouched and *cannot* regress.
- **The gate optimizes the worst case, not the marginal one.** `width_split` is ~0.92–0.94× (a few %
  faster) on the squares — a real but small edge. Switching wholesale to width to grab it would cost a
  **6.15× collapse** on tall-narrow. The gate correctly keeps the default and forgoes the marginal gain.

**Takeaway:** when a distribution scheme under-fills the grid on some regime, do not replace it
wholesale — gate the specialized scheme behind a utilization predicate and keep the conventional scheme
as the default. When the gate doesn't trip, the default is byte-for-byte unchanged (no regression); when
it does, only the regime that needed help changes. On this 64-core WH B0, that buys the grid-filling
win on the bad regime (up to ~7×) with a *measured zero* regression on the shapes the default handled.
