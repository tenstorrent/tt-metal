# distribution_gate — gate the split axis so fixing one regime doesn't regress the other

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** work distribution — choosing (and *gating*) the axis you split across the grid
**First profiled on:** `bgd-lab-16-special-dstoiljkovic-for-reservation-44175` · WH B0 · 8×8=64 grid · 2026-07-24 · `b9e384c5753`

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
To spread a tile op over the grid you split its tiles across cores along **some** axis. There are two
natural choices, and each is a trap for the opposite aspect ratio:

- **height split** — partition the tile-**rows** across cores. Fills the grid when there are many
  tile-rows; strands a **wide-short** tensor (few rows, many columns — extreme: one tile-row tall) on
  as few as **one** core.
- **width split** — partition the tile-**columns** across cores. Fills the grid when there are many
  tile-columns; strands a **tall-narrow** tensor (many rows, few columns) on as few as **one** core.

The trap is symmetric. So the tempting "fix" for a wide-short tensor — *just switch to a width split* —
**regresses every tall-narrow tensor** the height split already handled. Swapping one collapse for the
other is not a fix.

## What this isolates — and how
- **Concept:** work distribution — which axis to split, and gating that choice by aspect ratio so a
  specialization for one regime never regresses the other.
- **Isolation setup:** *work distribution / multi-core grid* — compute is a trivial per-tile op (one
  relu), inputs and outputs are interleaved DRAM tile tensors, and the reader/compute/writer kernels
  are **byte-identical across all three variants**. Only the per-core tile rectangle — and therefore
  how many cores run — changes. The measured delta is purely *how the tiles are spread across cores*.
- **Why it's kernel-level:** how you assign work-units to cores (the core grid + per-core runtime args)
  is a decision the kernel/program author makes, not a model choice.

## The methods being compared
| Variant | What it does | Why it should differ |
|---|---|---|
| `height_split` | split tile-**rows** across `min(Ht, grid)` cores (each core: a row band, full width) | fills the grid when `Ht` is large; collapses to 1 core when `Ht=1` (wide-short) |
| `width_split` | split tile-**columns** across `min(Wt, grid)` cores (each core: a column strip, full height) | fills the grid when `Wt` is large; collapses to 1 core when `Wt=1` (tall-narrow) |
| `gated` *(the discipline)* | **height by default; divert to width ONLY when width fills strictly more cores** | fills the grid on both regimes; when the gate doesn't trip it is *byte-identical* to `height_split`, so shapes the default already handled cannot regress |

## Measured result (WH B0, 64-core grid, bf16, relu, iters=1)
| H | W | Ht | Wt | height_split | width_split | gated | note |
|--:|--:|--:|--:|--:|--:|--:|---|
| 32 | 4096 | 1 | 128 | 30921 ns *(1 core)* | 4190 ns *(64)* | 4262 ns *(64)* | **wide-short**: height strands → **7.25× slower**; gated diverts to width |
| 2048 | 32 | 64 | 1 | 2725 ns *(64)* | 16702 ns *(1 core)* | 2714 ns *(64)* | **tall-narrow**: width strands → **6.15× slower**; gated stays on height |
| 2048 | 2048 | 64 | 64 | 90667 ns *(64)* | 83740 ns *(64)* | 90665 ns *(64)* | square: both fill grid; **gated == height_split** (no regression) |
| 1024 | 1024 | 32 | 32 | 23482 ns *(32)* | 22221 ns *(32)* | 23543 ns *(32)* | square: both fill grid; **gated == height_split** |

**Read it:** each fixed axis is catastrophic on its bad regime (7.25× / 6.15× slower, stranded on one
core). `gated` fills the grid on **both**. And on the two square shapes — where the default height split
already saturates the grid — `gated` lands on the *same* number and same code path as `height_split`
(90665 vs 90666 ns): the specialization left the baseline untouched. Note `width_split` is marginally
faster on the squares (~0.92×), yet the gate correctly **refuses to chase that ~8% square gain** at the
cost of a 6.15× tall-narrow collapse — it optimizes the worst case, not the marginal one. Full table +
method in [`report.md`](report.md).

## CLI — measure your own aspect ratios
```bash
python -m ttnn.operations.examples.distribution_gate [--shapes 32x4096,2048x32,2048x2048,1024x1024]
                                                      [--variant all|height_split|width_split|gated]
                                                      [--dtype bfloat16|float32|bfloat8_b]
                                                      [--iters K] [--trials N]
```
Shapes are `HxW`. `--iters 1` measures per-launch latency; large `--iters` measures steady-state throughput.

## Takeaway for your own kernels
When one distribution scheme under-fills the grid on some regime, **do not replace it wholesale** with a
scheme tuned for that regime — you will strand the shapes the original handled well. Instead **gate** the
specialized scheme behind an explicit utilization predicate (e.g. "the default split fills ≤ 1/K of the
grid"), keeping the conventional scheme as the default. When the gate doesn't trip, the default path is
byte-for-byte unchanged, so it *cannot* regress; the new path is confined to exactly the regime that
needed it. A gate that ties in favor of the default is how you get "fast on the new regime **and**
provably no slower on the old one."
