# reduce_accumulate — a SUM/mean reduce as accumulate + SFPU-finalize vs the standard reduce library

**Problem type:** compute-bound: reduction  ·  **Concept(s):** reduce split into cross-tile FPU accumulate + within-tile SFPU finalize, across row/col/scalar, vs the matmul-reduce library — with a width-and-dim dispatch
**First profiled on:** `bgd-lab-t3003-special-mstaletovic-for-reservation-40918` · WH B0 · single core · 2026-07-13 · `54a202a35d5`
**Measured on:** WH B0 n150 (overview 2026-07-13, regimes 2026-10-12) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19)

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

*Board scope (added 2026-10-19):* the overview above is WH B0 on an older library. "REDUCE_COL is cheaper, so col
wins least" is a WH B0 property — on BH p150b the library's REDUCE_ROW is the cheaper one (~36 vs ~62 ns per
reduced tile), so **col wins the most (2.97× @32t) and row needs 16 tiles** before `fast` pays. The scalar fp32
accuracy gap is ~50× on BH p150b (helper 4.3e-04 vs fast 8.3e-06 @32t), not ~100×. Per-arch crossovers are in
*When it helps* below.

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

## When it helps / when it does not
**Rule:** `fast` (accumulate + SFPU finalize) beats the reduce library once enough tiles are reduced per output tile
to amortize its fixed finalize; the crossover is per dim **and** per arch — WH B0 n150: row 8, col 8 (marginal,
1.05×), scalar 8 tiles; BH p150b: **row 16**, col 8, scalar 8 tiles. Below the crossover `fast` loses (0.47–0.97×).

| Regime (same definition on every board; 1 Tensix core, sharded L1, bf16 in, fp32 accumulation, HiFi4) | WH B0 n150 · 8×8 = 64 Tensix cores | BH p150b · 12×10 = 120 Tensix cores |
|---|---|---|
| **Helps:** many tiles reduced per output, REDUCE_ROW — `[32, 1024]` (32 tiles) | 3 652 → 1 865 ns **1.96×** | 1 385 → 800 ns **1.73×** |
| **Helps:** many tiles reduced, REDUCE_COL — `[1024, 32]` (32 tiles) | 2 978 → 1 736 ns **1.72×** | 2 157 → 727 ns **2.97×** |
| **Helps:** many tiles reduced, REDUCE_SCALAR — 32 tiles | 5 861 → 2 000 ns **2.93×** | 3 286 → 880 ns **3.73×** |
| **Helps:** 8 tiles, REDUCE_COL | 994 → 945 ns **1.05×** (sweep; edge) | 679 → 505 ns **1.35×** |
| **Does not help:** 8 tiles, REDUCE_ROW | 1 188 → 1 077 ns **1.10× — helps on WH** (sweep) | 514 → 578 ns **0.89×** |
| **Does not help:** 4 tiles, REDUCE_ROW | 782 → 940 ns **0.83×** | 368 → 536 ns **0.69×** |
| **Does not help:** 4 tiles, REDUCE_COL | 650 → 807 ns **0.81×** | 438 → 458 ns **0.96×** |
| **Does not help:** 4 tiles, REDUCE_SCALAR | 1 013 → 1 071 ns **0.95×** | 589 → 609 ns **0.97×** |
| **Does not help:** 1 tile, REDUCE_ROW | 467 → 940 ns **0.50×** | 254 → 535 ns **0.47×** |
| Sweep row, 1 / 2 / 4 / 8 / 16 / 32 tiles | 0.50 / 0.64 / 0.85 / 1.10 / 1.51 / 1.96× | 0.47 / 0.56 / 0.69 / 0.89 / 1.24 / 1.73× |
| Sweep col, same widths | 0.51 / 0.61 / 0.81 / 1.05 / 1.37 / 1.71× | 0.54 / 0.69 / 0.96 / 1.35 / 2.04 / 2.97× |
| Sweep scalar, same widths | 0.46 / 0.64 / 0.94 / 1.40 / 2.11 / 2.92× | 0.48 / 0.66 / 0.97 / 1.48 / 2.39 / 3.73× |
| **Boundary** (first width where `fast` ≥ 1.05×) | row 8 · col 8 (1.05×; 1.37× at 16) · scalar 8 tiles | **row 16** · col 8 · scalar 8 tiles |
| `dispatch` thresholds in the code (row = 4, col = 8, scalar = 8) | row = 4 is wrong: `dispatch` row @4t = 0.83× of `helper`; use row = 8 | row = 4 is wrong: 0.69× @4t and 0.89× @8t; use row = 16 |

**Why the boards differ:** per reduced tile (from the 1 → 32 tile sweeps), the library costs ~102 ns (row), ~83 ns
(col), ~173 ns (scalar) on WH B0 n150 but ~36 / ~62 / ~97 ns on BH p150b, while `fast` costs ~30 ns per tile on
WH B0 n150 and ~8.5 ns on BH p150b for every dim. BH's library REDUCE_ROW is therefore relatively cheap (the WH
"REDUCE_COL is the cheaper datapath" ordering is reversed), so row needs twice the tiles to amortize `fast`'s
~530 ns fixed finalize, and col — the dearest library path on BH — gains the most.

- **Helps (>= ~8 tiles reduced per output tile; row >= 16 on BH p150b):** the library pays its matmul-with-ones datapath once per input tile (about 100 ns per tile for REDUCE_ROW on WH B0 n150, ~36 ns on BH p150b), while the fast path pays about 30 ns per tile on WH B0 n150 (~8.5 ns on BH p150b) for `add_tiles` plus a fixed finalize, so the gap grows with tile count.
- **Does not help (< ~8 tiles reduced):** the fast path has a fixed cost of roughly 800-900 ns on WH B0 n150 (~460-610 ns on BH p150b) (pack, SFPU finalize, `mul_unary_tile`) that the library's 1-4 tile reduce never pays, so at 4 tiles the fast path is still 0.8-0.95x (0.69-0.97x on BH p150b).
- **Measured crossover on WH B0 (this build):** row 8 tiles (1.10x), col 8 tiles (1.05x), scalar 8 tiles (1.40x); at 4 tiles all three dims are still slower (0.81-0.95x). The row crossover is ~8 tiles, not 4: the table at the top (row `helper` 968 ns at 4 tiles) was measured on an older library, whose REDUCE_ROW is now ~20-30% cheaper (here 3652 ns at 32 tiles, not 5328 ns). The per-dim `dispatch` thresholds in the code (row=4) are therefore stale on this build: `dispatch` row at 4 tiles measured 0.83x of `helper`.
- **Measured crossover on BH p150b:** row 16 tiles (1.24x; 0.89x at 8), col 8 tiles (1.35x), scalar 8 tiles (1.48x); col and scalar are just under 1.0x at 4 tiles (0.96x / 0.97x). The code's row=4 threshold is wrong here too (`dispatch` row 0.69x at 4 tiles, 0.89x at 8).

*How measured:* WH B0 n150 rows — `test_reduce_accumulate_regimes` (single Tensix core, sharded L1, bf16 input, fp32
accumulation, HiFi4, 200 in-kernel iters, 11 interleaved trials per variant, median; `helper` = baseline, `fast` =
lever, speedup = helper_ns / fast_ns; min-max in `report.md`) and the `RA_ACCUMS=fp32` device_perf sweep, 2026-10-12.
BH p150b rows — the same regimes test and device_perf sweep on that board, `bh-qb-15-special-adamjanovic-for-reservation-139410`,
2026-10-19 (details in `report.md`).
Reproduce: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_reduce_accumulate.py::test_reduce_accumulate_regimes`
(same regimes on any board; asserts only what held on every measured board, `measure` regimes only log).
