# width_split — filling the grid on a wide, short tensor

**Problem type:** grid not filled  ·  **Concept(s):** work distribution — splitting along width vs. height
**First profiled on:** `bgd-lab-16-special-dstoiljkovic-for-reservation-44175` · WH B0 · 8×8=64 grid · 2026-07-24 · `6f13238bd69`

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

**The crossover is at `Wt=2`–`3`:** with a single tile-column there's nothing to spread, so
`width_split` correctly matches the baseline (both use 1 core); at `Wt=2` it is still flat (0.89–0.99×,
see [When it helps / when it does not](#when-it-helps--when-it-does-not)). From `Wt=8` up it scales with the grid — the
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
does not pay for `Wt ≤ 2` (see below — the earlier claim of "only `Wt ≤ 1`" was corrected by
measurement).

## When it helps / when it does not
Board: **Wormhole B0 n150** (8×8 = 64-Tensix-core compute grid), bf16, H=32, interleaved DRAM
(`TensorMemoryLayout::INTERLEAVED`). Numbers from `test_width_split_regimes` — median of 12 trials
per variant (a trial = mean device kernel duration of 5 launches), baseline/lever trials interleaved.

| Condition | Shape | single_core (ns) | width_split (ns) | Speedup | Trials | Board |
|---|---|--:|--:|--:|--:|---|
| **Helps:** Wt ≫ 1, one tile-column per Tensix core up to the 64-core grid | `[32, 2048]` (Wt=64 tiles → 64 Tensix cores × 1 tile) | 16832 (16766–16881) | 2720 (2635–2796) | **6.19×** (runs: 6.19 / 6.10 / 6.07 / 6.06) | 12 | WH B0 n150 |
| **Does not help:** Wt ≤ 2 tiles total | `[32, 64]` (Wt=2 tiles → 2 Tensix cores × 1 tile) | 2076 (2043–2113) | 2105 (1995–2514) | **0.99×** (runs: 0.99 / 0.91 / 0.98 / 0.89) | 12 | WH B0 n150 |

- **Helps:** single_core reads, relus and writes all 64 tiles serially (~230 ns per tile on top of a
  ~1.8 µs launch floor), while width_split gives every Tensix core one tile, so the device time
  collapses to about one tile's round trip plus the launch floor.
- **Does not help:** with only 2 tiles the single Tensix core moves the whole tensor in **one**
  8-tile read block (one NoC barrier), costing ~2.1 µs of which ~1.8 µs is the fixed per-launch
  floor; a second Tensix core cannot shorten that floor and only adds its own startup, so the split
  is flat to ~10% slower. Width-splitting starts to pay at Wt=3 (1.30× in a sweep of the existing
  `test_width_split_device_perf`; Wt=4 → 1.46×, Wt=8 → 2.23×).
- **Limitation:** the example's API fixes H at one tile-row (`nt_h = 1`), so the "tile-row split vs
  width split as `nt_h` grows toward `Wt`" crossover cannot be measured here; the portable rule is
  about **tiles per Tensix core**: the split pays once the busiest Tensix core's work under the
  baseline is ≳3 tiles above the ~1.8 µs launch floor.

Reproduce: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_width_split.py::test_width_split_regimes`
