# row_reduce_accumulate — four ways to accumulate a row of tiles for a mean, and their perf/accuracy tradeoff

**Problem type:** compute-bound: reduction  ·  **Concept(s):** cross-tile accumulation strategy for a row-mean reduce · precision (input dtype × accumulation dtype), swept over input distributions
**First profiled on:** `bgd-lab-t3003-special-mstaletovic-for-reservation-40918` · WH B0 · single core · 2026-07-13 · `54a202a35d5`
**Measured on:** WH B0 n150 (overview 2026-07-13, regimes 2026-10-12) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19)

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
You need the mean across a **row of `W` tiles** — a `[32, 32·W]` strip collapsed to one output tile whose
column 0 holds the per-row mean (`REDUCE_ROW`). The obvious way is a single reduce over the whole strip.
But a reduce carries the reduce datapath's fixed per-tile cost, and it pays that cost `W` times as the row
gets wide. There are cheaper ways to do the *cross-tile* part of the sum (a plain add, or the packer's L1
adder) and leave only the final *within-tile* collapse to the reduce. This example bakes off four ways and
measures both **speed and accuracy** of each, across two precision axes and the width from 1 to 32 tiles.

## What this isolates — and how
- **Concept:** how the `W` tiles are summed together before the within-tile collapse, and at what precision.
- **Isolation setup (pure compute):** the input row is **sharded in L1 on one Tensix core** — no DRAM in the
  fast path, so the measured `ns` is only the on-core compute pipeline. Every variant does the identical math
  and is gated on correctness; only the accumulation mechanism and its precision vary. The output CB is fixed
  fp32, so accuracy deltas reflect the input/accumulation path, not output rounding.
- **Why it's kernel-level:** "fold the cross-tile sum into the reduce vs. `add_tiles` vs. packer L1-acc", and
  "accumulate in fp32 vs bf16 DEST", are decisions the kernel author makes — not model choices.

## The methods being compared
| Variant | What it does | Why it should differ |
|---|---|---|
| `reduce_fold` *(baseline)* | one `reduce<SUM, REDUCE_ROW>` over the whole row — accumulates the running row-sum across the width in DEST and does the within-tile column-sum in the same pass | pays the reduce datapath's per-tile cost `W` times; accumulates the full magnitude in DEST |
| `l1_accum` | copy each tile to DEST and **pack it onto one L1 tile with the packer's L1-accumulator** (`pack_tile<true>` + `pack_reconfig_l1_acc`), then one finalize reduce | moves the cross-tile sum onto the packer; L1-accumulate is **fp32-DEST-only** hardware |
| `dest_accum` | `add_tiles(acc_to_dest)` sums each input tile into one DEST register (one tile per add, against a zero tile), pack the sum, then one **FPU** finalize reduce | cross-tile sum is `W` cheap FPU adds + **one** reduce instead of `W` reduces |
| `dest_accum_pairs` | same, but **two input tiles per add** (`DEST += cb[k] + cb[k+1]`); an odd width seeds one tile via `copy_tile` (parity-at-seed, no phantom zero CB — see below) | halves the add count and the bf16 rounding steps |
| `dest_accum_sfpu` | `dest_accum` accumulation, but the finalize collapse runs on the **SFPU in DEST** (`sfpu_reduce<SUM>` + a scalar-multiply for `1/N`) — no pack-to-L1 + FPU-reduce round-trip | SFPU reads DEST natively; tests whether skipping the round-trip wins, and how SFPU vs FPU reduce rounding differs |
| `dest_accum_pairs_sfpu` | `dest_accum_pairs` accumulation + the same **SFPU** finalize | the finalize-engine twin of the fastest method |

### Handling an odd tile count (parity-at-seed vs a phantom zero CB)
Pairwise accumulation (`DEST += cb[k] + cb[k+1]`) hits a parity problem when `W` is odd: one tile has no
partner. The naive fix is a **phantom zero CB** — add the unpaired tile against a tile of zeros
(`add_tiles(cb[last], cb_zero)`), which costs an extra L1 CB and a dataflow zero-fill. The more general fix is
to **resolve the parity at the *seed*** — `add_tiles` is binary, but `copy_tile` is *unary*, so it can seed
DEST with a single real tile and no partner:
- **odd `W`** → seed DEST with `copy_tile(cb[0])`, leaving an **even** remainder;
- **even `W`** → seed with the first `add_tiles(cb[0], cb[1])` pair, leaving an even remainder.

After the seed the remainder is *always even*, so the pair loop is uniform with **no leftover and no zero CB**
(and `W==1` falls out for free). It's strictly better: no phantom CB, no dataflow zero-fill, handles every
parity, and — measured at odd widths (bf16-bf16, positive) — a touch **faster** (the tail zero-add, a binary op
that also loads the zero tile, becomes a unary seed copy):

```
  dest_accum_pairs, ns @ odd widths      3t     7t    15t    31t
  phantom-zero-CB leftover              815    944   1207   1726
  parity-at-seed (copy_tile)            798    931   1191   1715   (~1–2% faster, no zero CB)
```
Accuracy is equivalent (the unpaired tile is a real value either way; `copy` is exact like adding zero). The
example's `dest_accum_pairs{,_sfpu}` use the parity-at-seed approach; only the 1-tile-per-add `dest_accum`
methods still need the zero CB (each add is inherently binary with one real operand).

## The precision axes
`precision = <input>-<accum>` — the **input tensor dtype** and the **accumulation (DEST / intermediate-CB)
dtype** are independent. Three configs are measured (fp32-input into a bf16 accumulator is omitted — you'd
never build it):

| config | isolates |
|---|---|
| `fp32-fp32` | baseline — both precise |
| `bf16-fp32` | the **input-quantization** floor (lossy input, accurate accumulation) |
| `bf16-bf16` | **+ accumulation** loss on top of the input floor |

`fp32-fp32` vs `bf16-fp32` = the input-precision effect; `bf16-fp32` vs `bf16-bf16` = the
accumulation-precision effect. (`l1_accum` keeps fp32 DEST regardless — the packer L1-acc is fp32-DEST-only —
so its `-bf16` rounds only the L1 accumulator CB.)

## CLI — measure your own width / precision
```bash
python -m ttnn.operations.examples.row_reduce_accumulate [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `{all,reduce_fold,l1_accum,dest_accum,dest_accum_pairs}…` | `all` | which method(s) |
| `--precision` | `{all,fp32-fp32,bf16-fp32,bf16-bf16}…` | `all` | `<input>-<accum>` precision config(s) |
| `--distribution` | `{all,signal,uniform,positive}…` | `all` | input value distribution(s) (accuracy only) |
| `--widths` | ints (elements, ×32) | `32 64 128 256 512 1024` | accumulation widths, in **elements** |
| `--trials` | int | `5` | measured trials; median ± std |
| `--kernel-iters` | int | `200` | in-kernel loop count — large = steady-state throughput |
| `--report` | path | *(print only)* | also write the report (perf + accuracy tables) |

```bash
# accuracy of the accumulation axis on the widest row, all methods
python -m ttnn.operations.examples.row_reduce_accumulate --precision bf16-fp32 bf16-bf16 --widths 1024
```

## Measured result — one overview table
*Illustrative — see the **First profiled on** stamp and [`report.md`](report.md) (full per-precision perf +
per-distribution `max_abs | mean_abs | ULP` tables); re-run the CLI for your box.*

Perf = median ns/row-mean at **1t (narrow)** and **32t (wide, × vs `reduce_fold`)**, bf16 input, 200 in-kernel
iters. Accuracy = `max_abs | max ULP_bf16` of the **bf16-bf16** config at 32t, per input distribution (`signal`
= per-row linspace+noise, large; `uniform` = [-1,1) zero-mean; `positive` = [0,1)). This is where precision is
actually lost — **fp32 accumulation is ~exact** (≤3e-3 max_abs everywhere).

| method | ns @1t | ns @32t (×) | signal `max\|ULP` | uniform `max\|ULP` | positive `max\|ULP` |
|---|---:|---:|---:|---:|---:|
| `reduce_fold` (baseline) | 467 | 5312 | 2.1e-01 \| 13u | 7.1e-04 \| 40u | 1.1e-02 \| 6u |
| `l1_accum` | 814 | 5153 (1.03×) | 3.7e-03 \| 0u | 2.1e-04 \| 12u | 8.6e-04 \| 0u |
| `dest_accum` | 846 | 2868 (1.85×) | 3.3e-02 \| 2u | 6.3e-04 \| 328u | 5.7e-03 \| 3u |
| `dest_accum_pairs` | 826 | **1811 (2.93×)** | 2.1e-02 \| 1u | 3.7e-04 \| 112u | 4.5e-03 \| 2u |
| `dest_accum_sfpu` | 913 | 2913 (1.82×) | 2.4e-02 \| 2u | 5.4e-04 \| 319u | 4.0e-03 \| 2u |
| `dest_accum_pairs_sfpu` | 870 | 1864 (2.85×) | **8.2e-03** \| 1u | 2.8e-04 \| 132u | **2.0e-03** \| 1u |

**Reading of the result:**
- **Perf** — at a narrow row (1–2 tiles) `reduce_fold` wins (one reduce call; the others add finalize
  overhead). As the row widens `reduce_fold` loses (it pays the reduce datapath *per tile*) while the others do
  `W` cheap adds + **one** reduce; `dest_accum_pairs` wins for `W ≥ 4` (**2.93×** bf16 @32t). fp32 input ≈
  halves the win (pairs 1.87×) since the add path unpacks 2× the bytes; `reduce_fold`'s cost is
  input-dtype-insensitive.
- **Input precision is nearly free for a wide mean** (`fp32-fp32` → `bf16-fp32`): bf16-input error *averages
  DOWN* with width (`reduce_fold` bf16-fp32: 0.17 → 0.04 ULP as W:1→32) and stays sub-ULP — a mean over many
  elements washes out input quantization.
- **Accumulation precision is what costs you** (`bf16-fp32` → `bf16-bf16`), and it *grows UP* with width. On
  nonzero-mean data (`signal`, `positive`) the loss ordering is **`reduce_fold` (worst) > `dest_accum` >
  `dest_accum_pairs` > `l1_accum` (best)**: the running sum swamps small increments in bf16, worst when the
  whole sum lives in one bf16 DEST (`reduce_fold`, **13 ULP**); pairing halves the rounding steps; `l1_accum`
  is best (**0.24 ULP**) only because the packer forces fp32 DEST so its finalize reduce stays fp32.
- **SFPU vs FPU finalize** (`dest_accum{,_pairs}` → `dest_accum{,_pairs}_sfpu`): doing the within-tile collapse
  on the SFPU in DEST (`sfpu_reduce` + scalar-mul) reads DEST natively and skips the pack→L1→unpack round-trip,
  yet is **not faster** (32t: pairs_sfpu 1864 ns vs pairs 1811 ns) — the SFPU vector reduce costs more than the
  FPU matmul-reduce, just outweighing the saved round-trip. But it is **more accurate in bf16** (pairs_sfpu
  8.2e-3 vs pairs 2.1e-2 max_abs @32t, signal) because the SFPU collapses the 32 columns in fp32 internally
  before a single output rounding. So the SFPU finalize buys accuracy, not speed.
- **Distribution matters.** On zero-mean `uniform` every method keeps `max_abs` tiny (~1e-3) — a near-zero mean
  has little magnitude to lose, so the method choice barely matters in absolute terms; the large ULP there is
  just that tiny error divided by the ~0 mean (max_abs, not ULP, is the honest metric for cancellation-heavy
  data). All-positive `positive` is the monotonic-swamping case; `signal` is worst in absolute terms only
  because its magnitudes are largest.
- **Bottom line:** `dest_accum_pairs` is the fastest wide-row method and the more accurate of the plain
  DEST-add methods; its **SFPU-finalize twin** matches its speed and is the most accurate bf16 DEST-add option.
  Reach for `l1_accum` only when you need the best bf16 accuracy and can pay the latency; keep the single
  `reduce_fold` for narrow rows (≤2 tiles).

## Run the predefined sweep (regenerates `report.md`)
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_row_reduce_accumulate.py
```

## Code
`program_descriptor_with_inline_kernels.py` — one compute kernel (`method` compile-time arg selects the path),
plus a tiny dataflow kernel that fills the reduce scaler (`1/(W·32)`) and the zero tile.

## When it helps / when it does not
**Rule:** `dest_accum_pairs` beats the single `reduce_fold` once the row is wide enough to amortize its fixed
finalize — from 16 tiles for bf16 input and 8 tiles for fp32 input on WH B0 n150, from 8 tiles for either input on
BH p150b (marginal there: 1.05–1.06×; clearly from 16 tiles). At 1–4 tiles it loses on both boards.

| Regime (same definition on every board; 1 Tensix core, input sharded in L1, HiFi4, 200 in-kernel iters) | WH B0 n150 · 8×8 = 64 Tensix cores | BH p150b · 12×10 = 120 Tensix cores |
|---|---|---|
| **Helps:** wide row, fp32 input — 32 tiles, `fp32-fp32` | 4 478 → 2 850 ns **1.57×** | 1 386 → 784 ns **1.77×** |
| **Helps:** wide row, bf16 input — 32 tiles, `bf16-fp32` | 2 419 → 1 803 ns **1.34×** | 1 369 → 694 ns **1.97×** |
| **Does not help:** 8-tile row — 8 tiles, `bf16-fp32` | 886 → 1 014 ns **0.87×** | 499 → 470 ns **1.06× — marginal gain on BH** (`measure` regime: verdict differs by arch) |
| **Does not help:** 4-tile row — 4 tiles, `bf16-fp32` | 633 → 877 ns **0.72×** (sweep) | 356 → 430 ns **0.83×** |
| **Does not help:** 2-tile row — 2 tiles, `bf16-fp32` | 497 → 834 ns **0.60×** | 282 → 403 ns **0.70×** |
| **Does not help:** 1-tile row — 1 tile, `bf16-fp32` | 433 → 837 ns **0.52×** | 240 → 413 ns **0.58×** |
| Sweep `bf16-fp32`, 1 / 2 / 4 / 8 / 16 / 32 tiles | 0.52 / 0.61 / 0.72 / 0.87 / 1.09 / 1.34× | 0.58 / 0.70 / 0.83 / 1.06 / 1.46 / 1.97× |
| Sweep `fp32-fp32`, same widths | 0.58 / 0.71 / 0.87 / 1.10 / 1.35 / 1.58× | 0.61 / 0.71 / 0.85 / 1.05 / 1.37 / 1.77× |
| Sweep `bf16-bf16`, same widths | 0.60 / 0.71 / 0.82 / 0.99 / 1.22 / 1.46× | 0.63 / 0.74 / 0.89 / 1.14 / 1.55 / 2.08× |
| Single-tile-per-add `dest_accum`, 32 tiles (`fp32-fp32` / `bf16-fp32`) | 0.93× / 0.86× — never wins | 1.20× / 1.33× — wins at 32 tiles |
| **Boundary** (first width ≥ 1.05×) | bf16 input 16 tiles · fp32 input 8 tiles | 8 tiles for both (1.06× / 1.05×) |

**Why the boards differ:** per extra tile (from the 1 → 32 tile `bf16-fp32` sweeps) `reduce_fold` costs ~64 ns on
WH B0 n150 vs ~36 ns on BH p150b, but `dest_accum_pairs` costs ~31 ns vs ~9 ns, and its fixed cost halves
(~830 → ~410 ns at 1 tile). The FPU add-into-DEST path sped up far more on BH p150b than the reduce datapath did —
the same pattern as in `reduce_accumulate` and `eltwise_l1_vs_dest_accumulate` — so the accumulate methods pay
sooner and more. Input dtype also flips: on BH p150b `reduce_fold` costs the same per tile for fp32 and bf16 input
(~36 ns) while the add path pays ~30 % more for fp32, so bf16 input gives the larger win (1.97× vs 1.77×); on WH B0
n150 `reduce_fold` doubles its per-tile cost for fp32 input (~129 vs ~64 ns), so fp32 gives the larger win.

- **Helps (wide row):** `reduce_fold` costs about 70 ns (bf16) to 130 ns (fp32) per tile on WH B0 n150 (~36 ns for either on BH p150b), `dest_accum_pairs` about 30-60 ns per tile on WH B0 n150 (~9-12 ns on BH p150b) plus a fixed cost, so the gap widens with width; on WH B0 n150 the win is larger for fp32 input (more bytes per `reduce_fold` tile), on BH p150b for bf16 input.
- **Does not help (narrow row):** `dest_accum_pairs` has a fixed ~800 ns finalize on WH B0 n150 (~410 ns on BH p150b) (pack to L1, FPU reduce) that a 1-2 tile single reduce never pays.
- **Measured crossover on this build (WH B0 n150):** `dest_accum_pairs` first wins at **16 tiles for bf16 input** (1.09x `bf16-fp32`, 1.22x `bf16-bf16`) and at **8 tiles for fp32 input** (1.10x); at 8 tiles bf16 it is 0.87x. Single-tile-per-add `dest_accum` never wins on WH B0 n150 (0.93x even at 32 tiles, fp32); on BH p150b it does at 32 tiles (1.20x fp32, 1.33x bf16).
- **Measured crossover on BH p150b:** 8 tiles for both inputs, but only just (1.06x `bf16-fp32`, 1.05x `fp32-fp32`, 1.14x `bf16-bf16`); 1.37-1.55x from 16 tiles. 4 tiles still loses (0.83x).
- **Correction to the text above (measured 2026-07-13 on WH B0; WH B0 n150 only):** the overview table and the "wins for W >= 4 (2.93x bf16 @32t)" claim do not reproduce on this build. `reduce_fold` is now ~2.2x cheaper at 32 tiles bf16 (2419 vs 5312 ns), so the win at 32 tiles is **1.34x (bf16) / 1.57x (fp32)**, not 2.93x / 1.87x, the crossover is 8-16 tiles, not 4, and fp32 input now gives the *larger* win rather than half of it. On BH p150b the win at 32 tiles is 1.97x (bf16) / 1.77x (fp32), the crossover is 8 tiles, and bf16 input gives the larger win, as the overview said for WH originally (though fp32 is ~0.9x of the bf16 win there, not half).

*How measured:* WH B0 n150 rows — `test_row_reduce_accumulate_regimes` (single Tensix core, input sharded in L1,
`positive` distribution, HiFi4, 200 in-kernel iters, 11 interleaved trials per variant, median (min-max in `report.md`);
baseline = `reduce_fold`, lever = `dest_accum_pairs`; speedup = baseline_ns / lever_ns) and the `RRA_DISTS=positive`
device_perf sweep, 2026-10-12. BH p150b rows — the same regimes test and device_perf sweep on that board (distribution
`signal`; perf is data-independent), `bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19 (details in `report.md`).
Reproduce: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_row_reduce_accumulate.py::test_row_reduce_accumulate_regimes`
(same regimes on any board; asserts only what held on every measured board, `measure` regimes only log).
