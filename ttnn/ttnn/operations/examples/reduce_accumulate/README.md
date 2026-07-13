# reduce_accumulate — a SUM/mean reduce as accumulate + SFPU-finalize vs the standard reduce library

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** reduce split into cross-tile FPU accumulate + within-tile SFPU finalize, across row/col/scalar, vs the matmul-reduce library — with a width-and-dim dispatch
**First profiled on:** `bgd-lab-t3003-special-mstaletovic-for-reservation-40918` · WH B0 · single core · 2026-07-13 · `54a202a35d5`

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
A `SUM`/mean reduce of `N` tiles is usually done with the reduce library, whose datapath folds the cross-tile
sum and the within-tile reduction together in one matmul-with-ones per tile — so it pays that datapath `N`
times. An alternative splits the work: **elementwise-accumulate the `N` tiles into one DEST tile** (cheap
pairwise `add_tiles`), then do a **single within-tile finalize** — and do that finalize on the **SFPU**, which
reads DEST in place (no L1 round-trip). This example implements that fast path for all three reduce
dimensions, checks it against the library, and works out where it wins.

## What this isolates — and how
- **Concept:** reduce = cross-tile accumulate (FPU `add_tiles`) + within-tile finalize (SFPU `sfpu_reduce`),
  vs the library's matmul-reduce. The **accumulate stage is identical for every dim**; only the finalize
  differs: `sfpu_reduce<REDUCE_ROW>` (width), `<REDUCE_COL>` (height), or `ROW` then `COL` (scalar).
- **Isolation setup (pure compute):** input sharded in L1 on one Tensix core, no DRAM. The measured `ns` is
  the on-core reduce pipeline only.
- **Why it's kernel-level:** "fold the reduce in the matmul datapath" vs "accumulate then finalize on the
  SFPU" — and which to dispatch — are kernel-author decisions.

## The variants
| Variant | What it does |
|---|---|
| `helper` *(baseline)* | the standard reduce library (`compute_kernel_lib::reduce`, FPU matmul-with-ones), AVG so the `1/N` scaler is applied per dim |
| `fast` | pairwise `add_tiles(acc_to_dest)` accumulate (copy-seed for odd `N`) → SFPU `sfpu_reduce` finalize in DEST → `mul_unary_tile` for the `1/N` mean |
| `dispatch` | picks `fast` when `N ≥` a **per-dim** threshold (measured), else `helper` — so it's never slower than the library |

## Measured result — one overview
*Illustrative — see the **First profiled on** stamp and [`report.md`](report.md); re-run the CLI for your box.
Single core, bf16 input, fp32 output/accumulation, HiFi4.*

| reduce dim | `fast` vs `helper` @ 32t | crossover (`dispatch` ≥) | max win | accuracy (fast vs helper) |
|---|---:|---:|---:|---|
| `row` (width) | **2.87×** | 4t | 2.87× @32t | ≈equal fp32; **5.5×** better bf16 @32t |
| `col` (height) | **1.71×** | 8t | 1.71× @32t | ≈equal fp32; ~**3×** better bf16 @32t |
| `scalar` (both) | **2.94×** | 8t | 2.94× @32t | **~100×** better fp32 (see below) |

```
  fast ÷ helper, median ns per reduce (bf16 in, fp32 acc, single WH-B0 core):
  N tiles →         1t      2t      4t      8t     16t     32t
  row   helper     485     649     968    1587    2832    5328
  row   fast       906     917     943    1071    1336    1858     (0.54 → 2.87×;  wins ≥4t)
  col   helper     383     472     641     978    1629    2962
  col   fast       773     793     819     951    1200    1731     (0.50 → 1.71×;  wins ≥8t)
  scal  helper     474     648    1004    1696    3076    5850
  scal  fast      1031    1066    1073    1214    1472    1990     (0.46 → 2.94×;  wins ≥8t)
  dispatch = helper below the per-dim crossover, fast at/above it → never slower than the library.
```

**Reading of the result:**
- **It generalizes to all three dims.** The accumulate is identical; the SFPU `sfpu_reduce` does `REDUCE_ROW`
  and `REDUCE_COL`, and scalar is `ROW` then `COL` (the total lands at `[0,0]`).
- **The crossover is dim-dependent** — the one caveat for a shared helper. The fast path amortizes its
  finalize over the accumulate, so it only wins once there are enough tiles; and the FPU **`REDUCE_COL`
  datapath is cheaper** than `REDUCE_ROW`, so **col needs more tiles (≥8) and wins the least (1.71× vs
  ~2.9×)**. A flat threshold would mis-dispatch col/scalar at ~4 tiles (fast is *slower* there), so the
  dispatch is per-dim (row=4, col=8, scalar=8) and `dispatch` is consequently never slower than the library.
- **Accuracy: equal in fp32, better in bf16.** In fp32 both are near-exact. In bf16 the fast path is *more*
  accurate (the SFPU collapses the 32 columns in fp32 internally before one output rounding): row/col bf16
  @32t are ~3–5.5× lower error. For **scalar** it is dramatically better even in fp32 — the fast path
  multiplies by `1/N` **once**, whereas the library's AVG-scalar applies a `1/√N` scaler **twice** (row then
  col), adding rounding: **8e-6 vs 1.5e-3** max error @32t (~100×).
- **Takeaway:** the accumulate+SFPU-finalize path is a strong fast path for wide SUM/mean reductions and a
  clear accuracy win for scalar — but it is *not* a universal replacement for the reduce library: it loses
  below the (dim-dependent) crossover, benefits least on col, and this microbench is compute-bound and
  single-core (most real reductions are data-movement-bound, where the compute delta won't show). The right
  shape is a **dispatched fast path** inside the helper — exactly what the `dispatch` variant models.

## CLI
```bash
python -m ttnn.operations.examples.reduce_accumulate [options]
```
| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `{all,helper,fast,dispatch}…` | `all` | which path(s) |
| `--dim` | `{all,row,col,scalar}…` | `all` | reduce dimension(s) |
| `--accum` | `{all,fp32,bf16}…` | `all` | accumulation (DEST/SFPU) dtype for the accuracy tables |
| `--widths` | ints (tiles) | `1 2 4 8 16 32` | tile counts to reduce |
| `--trials` | int | `5` | timed passes; median ± std |
| `--kernel-iters` | int | `200` | in-kernel loop count |
| `--report` | path | *(print only)* | write the report |

```bash
# is the fast path worth it for a wide height reduction on my box?
python -m ttnn.operations.examples.reduce_accumulate --dim col --widths 8 16 32
```

## Run the predefined sweep (regenerates `report.md`)
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_reduce_accumulate.py
```

## Code
`program_descriptor_with_inline_kernels.py` — the `fast` kernel (shared accumulate + per-dim SFPU finalize),
the `helper` kernel (standard reduce), and the AVG scaler dataflow kernel; `dispatch` routes between them
per (dim, width).
