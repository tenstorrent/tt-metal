# width_split — filling the grid on a wide, short tensor

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** work distribution — splitting along width vs. height
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

**The crossover is at `Wt=1`:** with a single tile-column there's nothing to spread, so `width_split`
correctly matches the baseline (both use 1 core). From `Wt=8` up it scales with the grid — the
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
so the whole grid runs in parallel. The benefit grows with width, up to ~grid× at saturation; the
only case where it doesn't pay is `Wt ≤ 1`.
