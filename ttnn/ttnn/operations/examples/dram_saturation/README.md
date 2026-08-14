# dram_saturation — how many cores? DRAM bandwidth saturates; more cores stop paying

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** work distribution — the core-count sweet spot for a DRAM-bound op (bandwidth saturation), and how placement sets where it saturates
**First profiled on:** `bgd-lab-16-special-dstoiljkovic-for-reservation-44175` · WH B0 · 8×8=64 grid · 2026-07-24 · `e1dc0e60d9a`

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
"More cores = faster" is only true until the **DRAM interface saturates**. For a data-movement-bound
op (the bytes, not the math, are the whole job), achieved bandwidth rises as you add cores — then
**plateaus** once the banks / NoC are full. Past that knee, extra cores add **no** bandwidth: they're
wasted, and if they pile onto shared links they *congest*. So the right answer for a bandwidth-bound
op is the **minimum well-placed cores that saturate the bus**, not the whole grid — the opposite
instinct from a compute- or latency-bound op, where filling the grid keeps paying.

## What this isolates — and how
- **Concept:** achieved DRAM bandwidth as a function of **core count** (the saturation knee = the
  sweet spot), plus how **placement** decides where that knee falls.
- **Isolation setup:** *work distribution / NoC contention* — a **pure DRAM→DRAM copy** of a fixed
  large interleaved tensor: reader on NoC0, writer on NoC1, **no compute kernel at all**, so the op
  is unambiguously DRAM-bandwidth-bound. Achieved bandwidth = `2 × tensor_bytes / device_kernel_ns`
  (read + write). The reader/writer kernels are **byte-identical** across every point; only the
  **core count** (swept) and the **placement** (the variant) change.
- **Why it's kernel-level:** how many cores to launch and where to put them is the program author's
  call, not a model choice.

## The methods being compared
| Variant | Core placement | Why it should differ |
|---|---|---|
| `spread` | the N cores row-major across the grid (traffic spread over the DRAM-facing axis) | saturates *early* — hits peak bandwidth with the fewest cores |
| `stacked` | the N cores column-major (piled onto one axis, sharing NoC links) | congests — saturates *late*, needs far more cores for the same bandwidth |

The measured axis is the **core count**; the variant is the placement.

## Measured result (WH B0, 64-core grid, bf16, 2048×2048 = 8.4 MB, iters=1)
| cores | spread GB/s | spread GB/s/core | stacked GB/s |
|--:|--:|--:|--:|
| 1 | 21.7 | 21.7 | 21.7 |
| 2 | 43.3 | 21.7 | 43.2 |
| 4 | 85.7 | 21.4 | 64.8 |
| 8 | 146.3 | 18.3 | 71.8 |
| 16 | **190.9** | 11.9 | 130.5 |
| 32 | 192.5 | 6.0 | 179.6 |
| 48 | 194.9 | 4.1 | 191.8 |
| 64 | 193.8 | 3.0 | 191.9 |

**Read it:**
- **Saturation knee ≈ 16 cores.** `spread` climbs ~linearly (~21.7 GB/s/core) through 4 cores, then
  bends and **plateaus at ~191–195 GB/s from 16 cores on**. Going **16 → 64 cores (4× more) buys
  ~1.5%** — and per-core efficiency collapses **21.7 → 3.0 GB/s/core**. Those extra 48 cores did
  essentially nothing; on a real op they'd be better spent elsewhere (or left idle to cut power/heat).
- **Placement sets where the knee falls.** `stacked` reaches only **71.8 GB/s at 8 cores vs spread's
  146.3** (2× worse), and needs **~48 cores to reach what spread hits at 16** — the cost of piling
  traffic onto shared NoC links. A well-placed 16 cores beats a badly-placed 32.
- **Honest note:** on this shape/arch `stacked` doesn't go strictly *slower* as cores grow (no hard
  rollover) — it just saturates far later. The penalty for over-subscribing here is **wasted cores**,
  not a slowdown; on a more contended pattern the same mechanism can tip into an actual rollover.

Full tables + method in [`report.md`](report.md).

## The exploit — cap at the knee, free the rest
The bench derives the sweet spot automatically (`sweet_spot_cores`: the smallest core count within 3%
of peak on the `spread` curve) and quantifies the gain:

```
--- EXPLOIT: cap a DRAM-bound op at the bandwidth knee ---
sweet spot (spread, within 3% of peak):  16 cores @ 191.9 GB/s
full grid:                               64 cores @ 192.7 GB/s
=> same bandwidth at 4.0x fewer cores: 48 cores freed at +0.4% perf cost.
```

That's the gain you exploit: a DRAM-bandwidth-bound op gets its **full bandwidth on ~1/4 of the grid**,
so **48 cores come free at ~0% cost** to the op. In a real kernel or model those freed cores are the
prize — assign them other work, or cut power/heat. The rule is concrete: **measure the curve once, cap
the op at the knee.** `sweet_spot_cores()` is reusable on any `{cores: GB/s}` sweep, and the placement
lever below decides how low the knee can go.

## CLI — measure your own sweep
```bash
python -m ttnn.operations.examples.dram_saturation [--shape 2048x2048]
                                                    [--cores 1,2,4,8,16,32,48,64]
                                                    [--variant all|spread|stacked]
                                                    [--dtype bfloat16|float32|bfloat8_b]
                                                    [--iters K] [--trials N]
```

## Takeaway for your own kernels
First classify the bound (ablate compute vs DM). If the op is **DRAM-bandwidth-bound**, don't reflexively
fill the grid: sweep the core count, find the knee where achieved GB/s plateaus, and use the **minimum
well-placed cores that reach it** (spread across the DRAM-facing axis — see the placement lever). Cores
past the knee add no bandwidth and only risk contention. Only when the op is **not** bandwidth-bound
(compute- or latency-bound, or the grid isn't yet full of independent work) does adding more cores keep
paying — that's the grid-filling regime, the opposite of this one.
