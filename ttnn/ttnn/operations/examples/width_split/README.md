# width_split — filling the grid on a wide, short tensor

**Problem type:** grid not filled  ·  **Concept(s):** work distribution — splitting along width vs. height
**First profiled on:** `bgd-lab-16-special-dstoiljkovic-for-reservation-44175` · WH B0 · 8×8=64 grid · 2026-07-24 · `6f13238bd69`
**Measured on:** WH B0 n150 (`bgd-lab-16-special-dstoiljkovic-for-reservation-44175`, 2026-07-24; regimes 2026-10-12) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19)

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
You have a **wide, short** tensor — few tile-rows, many tile-columns (extreme case: a single
tile-row tall, e.g. `[32, 8192]`). The obvious way to spread a tile op over the grid is to give
each core some tile-**rows**. But a one-tile-row-tall tensor has exactly one row to hand out, so
*one* core does all the work and the other 63 sit idle. The op then runs at single-core speed no
matter how wide the tensor is — and wide-short tensors are common (folded activations, flat feature
maps). The fix is to distribute along the **width** instead.

## What this isolates — and how
- **Concept:** work distribution — parallelizing a wide-short tensor along its **width**
  (tile-columns) instead of its height (tile-rows).
- **Isolation setup:** *work distribution / multi-core grid* — compute is a trivial per-tile op
  (one relu), inputs and outputs are interleaved DRAM, and the tensor is fixed at **one tile-row
  tall (H=32)**. The reader/compute/writer kernels are byte-identical across variants; only the
  per-core `(start_page, num_pages)` — and therefore how many cores run — changes. So the measured
  delta is purely *how the Wt tiles are spread across cores*, nothing else.
- **Why it's kernel-level:** how you assign work-units to cores (the core grid + per-core runtime
  args) is a decision the kernel/program author makes, not a model choice.

## The methods being compared
| Variant | What it does | Why it should differ |
|---|---|---|
| `single_core` *(baseline)* | all `Wt` tiles on ONE core | what a tile-row split degenerates to for a 1-tile-row tensor — no width parallelism, runs serially |
| `width_split` | the `Wt` tiles spread contiguously across `min(Wt, grid)` cores (each core owns a `WT_CHUNK`-bounded column range) | fills the whole compute grid; per-core work drops to `Wt / cores`, so it should be ~`cores`× faster until the grid saturates |

## Measured result (WH B0, 64-core grid, bf16, H=32)
| W | Wt | single_core (ns) | width_split (ns) | cores | speedup |
|--:|--:|--:|--:|--:|--:|
| 32 | 1 | 1817 | 1822 | 1 | **1.00×** (nothing to split) |
| 256 | 8 | 4292 | 1919 | 8 | **2.24×** |
| 1024 | 32 | 9613 | 2263 | 32 | **4.25×** |
| 2048 | 64 | 16655 | 2666 | 64 | **6.25×** |
| 4096 | 128 | 30885 | 4203 | 64 | **7.35×** |
| 8192 | 256 | 59261 | 7637 | 64 | **7.76×** |

**On WH B0 n150 the crossover is at `Wt=2`–`3`:** with a single tile-column there's nothing to spread, so
`width_split` correctly matches the baseline (both use 1 Tensix core); at `Wt=2` it is still flat on WH B0 (0.89–0.99×),
while on BH p150b `Wt=2` already gains 1.18× (see [When it helps / when it does not](#when-it-helps--when-it-does-not)). From `Wt=8` up it scales with the grid — the
speedup tracks the number of cores it can fill (8→2.24×, 32→4.25×, 64→6.25×) — and past the 64-core
saturation point the gap keeps widening because `single_core` grows linearly with width while
`width_split` grows at 1/64 the rate. Full table + method in [`report.md`](report.md).

## CLI — measure your own widths
```bash
python -m ttnn.operations.examples.width_split [--widths 32,256,1024,2048,4096,8192]
                                               [--variant all|single_core|width_split]
                                               [--dtype bfloat16|float32|bfloat8_b]
                                               [--iters K] [--trials N]
```
`H` is fixed at 32 (one tile-row — the wide-short case); only the width sweeps. `--iters 1` measures
per-launch latency; large `--iters` measures steady-state throughput.

## Takeaway for your own kernels
If a tensor is **wider than it is tall in tiles** (`Wt > nt_h`, and especially `nt_h` small), do not
split work by tile-rows — you will strand it on `nt_h` cores. Split along the width (assign each
core a contiguous tile-column range, capped by a `WT_CHUNK` constant so per-core L1 stays bounded)
so the whole grid runs in parallel. The benefit grows with width, up to ~grid× at saturation; it
does not pay for `Wt ≤ 2` tiles on WH B0 n150 and for `Wt = 1` tile on BH p150b (see below — the earlier
WH claim of "only `Wt ≤ 1`" was corrected by measurement there, and holds on BH p150b).

## When it helps / when it does not
**Rule:** split along the width once the single Tensix core's baseline work is more than the launch floor can
hide — from **Wt ≥ 3 tiles on WH B0 n150** and from **Wt ≥ 2 tiles on BH p150b**; at Wt = 1 tile there is nothing
to split on either board. bf16, H = 32 (one tile-row), interleaved DRAM (`TensorMemoryLayout::INTERLEAVED`) in
and out; baseline = `single_core`, lever = `width_split` (1 tile per Tensix core up to the grid).

| Regime (same definition on every board) | WH B0 n150 · 8×8 = 64 Tensix cores | BH p150b · 12×10 = 120 Tensix cores |
|---|---|---|
| **Helps:** Wt = 64 tiles → 64 Tensix cores × 1 tile `[32, 2048]` | 16 832 → 2 720 ns **6.19×** | 11 092 → 1 536 ns **7.22×** |
| **Does not help:** Wt = 2 tiles → 2 Tensix cores × 1 tile `[32, 64]` | 2 076 → 2 105 ns **0.99×** (runs 0.89–0.99×) | 1 217 → 1 031 ns **1.18× — helps on BH** |
| **Does not help:** Wt = 1 tile, both variants on 1 Tensix core `[32, 32]` | 1 824 → 1 828 ns **1.00×** | 1 025 → 1 010 ns **1.02×** |
| Sweep: Wt = 3 tiles `[32, 96]` | 2 513 → 1 934 ns **1.30×** | **1.36×** |
| Sweep: Wt = 4 tiles `[32, 128]` | 2 873 → 1 966 ns **1.46×** | **1.54×** |
| Sweep: Wt = 8 tiles `[32, 256]` | 4 305 → 1 932 ns **2.23×** | 2 480 → 1 145 ns **2.17×** |
| Sweep: Wt = 32 tiles `[32, 1024]` | 9 613 → 2 263 ns **4.25×** | 6 160 → 1 240 ns **4.97×** |
| Sweep: Wt = 128 tiles `[32, 4096]` | 64 Tensix cores × 2 tiles: 30 885 → 4 203 ns **7.35×** | 120 Tensix cores: 20 886 → 1 867 ns **11.19×** |
| Sweep: Wt = 256 tiles `[32, 8192]` | 64 Tensix cores × 4 tiles: 59 261 → 7 637 ns **7.76×** | 120 Tensix cores: 40 518 → 2 961 ns **13.68×** |
| **Boundary** | Wt ≈ 3 tiles | Wt = 2 tiles |

**Why the boards differ:**
- **Launch floor.** A 1-tile launch costs ~1.8 µs on WH B0 n150 and ~1.0 µs on BH p150b (AICLK 1000 vs
  1350 MHz). The second tile adds ~280 ns to `single_core` on WH B0 and ~190 ns on BH p150b; on WH B0 the
  second Tensix core's own startup cancels that saving, on BH p150b it does not, so the split pays one
  tile earlier.
- **Grid size.** Past Wt = 64 tiles WH B0 n150 has no more Tensix cores and each one takes 2–4 tiles (7.35–7.76×),
  while BH p150b spreads up to 120 Tensix cores and keeps scaling (11.19× at Wt = 128 tiles, 13.68× at 256 tiles).

- **Helps:** `single_core` reads, relus and writes every tile serially (~235 ns per tile on WH B0 n150,
  ~160 ns on BH p150b, on top of the launch floor), while `width_split` gives every Tensix core one tile, so
  the device time collapses to about one tile's round trip plus the launch floor.
- **Does not help (WH B0 n150, Wt = 2 tiles):** the single Tensix core moves both tiles in **one** 8-tile read
  block (one NoC barrier), ~2.1 µs of which ~1.8 µs is the launch floor; a second Tensix core cannot shorten
  that floor and only adds its own startup, so the split is flat to ~10% slower.
- **Limitation:** the example's API fixes H at one tile-row (`nt_h = 1`), so the "tile-row split vs width
  split as `nt_h` grows toward `Wt`" crossover cannot be measured here; the portable rule is about **tiles per
  Tensix core**: the split pays once the busiest Tensix core's baseline work is ≳3 tiles above the launch
  floor on WH B0 n150, ≳2 tiles on BH p150b.

*How measured:* `DEVICE KERNEL DURATION [ns]`, baseline → lever.
- **WH B0 n150:** Wt = 64 and Wt = 2 tiles from `test_width_split_regimes` (median of 12 trials
  per variant, a trial = mean of 5 launches, variants interleaved), branch `adamjanovic/perf-example-improvements`,
  2026-10-12; Wt = 1, 3, 4, 8 tiles from that day's `test_width_split_device_perf` crossover sweep, Wt = 32, 128,
  256 tiles from the 2026-07-24 sweep on `bgd-lab-16-…` (mean of 20 launches each; [`report.md`](report.md)).
- **BH p150b:** Wt = 64 and Wt = 2 tiles from `test_width_split_regimes` (same method; Wt = 2
  re-measured 1.17× in a probe); Wt = 1, 8, 32, 128, 256 tiles from `test_width_split_device_perf` (mean of 20
  launches); Wt = 3, 4 tiles from a probe of the same interleaved A/B (ratios only).
  `bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19, commit `f74ee7aad64`.

Reproduce: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_width_split.py::test_width_split_regimes`
(same regimes on any board; asserts only what held on every measured board, `measure` regimes only log).
