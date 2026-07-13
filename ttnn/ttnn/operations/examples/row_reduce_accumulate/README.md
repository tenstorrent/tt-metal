# row_reduce_accumulate — four ways to accumulate a row of tiles for a mean, and their perf/accuracy tradeoff

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** cross-tile accumulation strategy for a row-mean reduce · precision (input dtype × accumulation dtype), swept over input distributions
**First profiled on:** `bgd-lab-t3003-special-mstaletovic-for-reservation-40918` · WH B0 · single core · 2026-07-13 · `54a202a35d5`

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
