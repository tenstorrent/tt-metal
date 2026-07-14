# reduce_block — a 2-D `(Ht, Wt, NC)` reduce as per-output accumulate + SFPU-finalize vs the reduce library

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** an accumulate (FPU `add_tiles`) + within-tile SFPU-finalize reduce, applied per output tile over a full 2-D tile block (many output tiles), vs the matmul-reduce library — with a per-dim, per-width dispatch
**First profiled on:** `bh-50-special-mstaletovic-for-reservation-44580` · BH · single core · 2026-07-14 · `d52e6bc64b8` (working tree, `mstaletovic/agent_eval`)

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
A single-strip reduce that collapses a `1 × W` (or `H × 1`) row of tiles into **one** output tile is easy — but real reductions run over a full 2-D tile block `(Ht, Wt, NC)` and emit **many** output tiles (one per non-reduced position, per batch). The reduce library handles this, but its datapath folds the cross-tile sum and the within-tile reduction together in one matmul-with-ones per input tile, so it pays that datapath once per input tile. This example takes the "accumulate the tiles, then finalize once on the SFPU" fast path — which wins on a single strip — and drives it over general 2-D blocks to see whether it still wins when the output is many tiles, and where the crossover is per dim.

## What this isolates — and how
- **Concept:** a whole-block reduce built as a **loop over output tiles**, each one an independent *accumulate its input subset into a single DEST register (pairwise FPU `add_tiles(acc_to_dest)`) → finalize within-tile on the SFPU (`sfpu_reduce`, reads DEST in place) → `mul_unary_tile` for the `1/N` mean*, vs the library's matmul-reduce over the same block. The only per-dim difference is which input tiles feed each output:
  - **row** (reduce width): output `(nc, h)` ← the `Wt` tiles contiguous from `(nc*Ht+h)*Wt` → `Ht*NC` output tiles.
  - **col** (reduce height): output `(nc, w)` ← the `Ht` tiles strided by `Wt` → `Wt*NC` output tiles.
  - **scalar**: output `(nc)` ← all `Ht*Wt` tiles of the batch → `NC` output tiles.
- **Isolation setup (pure compute):** input sharded in L1 on one Tensix core, no DRAM. The measured `ns` is the on-core reduce pipeline only.
- **Why it's kernel-level:** "fold the reduce into the matmul datapath, per input tile" vs "accumulate then finalize on the SFPU, per output tile" — and which to dispatch — are kernel-author decisions.
- **A structural bonus, not just speed:** because each output tile uses **one** DEST register (accumulate, then finalize in place), the fast path reduces an arbitrary block one DEST at a time — it never hits the DEST/chunk limit the library's `REDUCE_COL` datapath chunks around.

## The methods being compared
| Variant | What it does | Why it should differ |
|---|---|---|
| `reduce_tile` *(baseline)* | the reduce library, default datapath (`ReduceAlgorithm::Auto → ReduceTile`, FPU matmul-with-ones), AVG so the `1/N` is per dim | — |
| `accumulate_via_add` | the reduce library with the opt-in `ReduceAlgorithm::AccumulateViaAdd` (accumulate + SFPU-finalize inside the library) | trades `N` matmul-reduce datapaths for `N` cheap adds + one SFPU finalize; pays a per-`reduce()`-call init |
| `accumulate_via_add_inline` | the same algorithm as a standalone hand-written kernel, with the one-time init **hoisted out** of the perf loop | the init-hoisted reference — shows the algorithm's steady-state cost without the per-call init `accumulate_via_add` pays |
| `dispatch` | picks `accumulate_via_add` when the reduced tile-count per output (`row=Wt`, `col=Ht`, `scalar=Ht*Wt`) `≥` a **per-dim** threshold, else `reduce_tile` | so it is never slower than the library |

## CLI — measure your own shapes/params
This example is runnable on **any** block you care about, not just the predefined sweep:

```bash
python -m ttnn.operations.examples.reduce_block [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--dim` | `{all,row,col,scalar}…` | `all` | reduce dimension(s) |
| `--variant` | `{all,reduce_tile,accumulate_via_add,accumulate_via_add_inline}…` | `all` | which path(s) to compare |
| `--shape` | `Ht Wt NC` (repeatable) | *(built-in sweep)* | a block to reduce; overrides the sweep, applied to every selected dim |
| `--trials` | int | `5` | timed passes; median ± std |
| `--kernel-iters` | int | `200` | in-kernel loop count (`1` = per-launch latency; large = steady-state) |
| `--report` | path | *(print only)* | also write the report |

```bash
# is the fast path worth it for a wide (32-tile) scalar reduce on my box?
python -m ttnn.operations.examples.reduce_block --dim scalar --shape 4 8 1

# a batched row reduce, both library and inline paths
python -m ttnn.operations.examples.reduce_block --dim row --shape 2 8 4 --variant reduce_tile accumulate_via_add_inline
```

## Measured result
*Illustrative — see the **First profiled on** stamp above and [`report.md`](report.md); re-run the CLI for your box.
Single core, bf16 input, fp32 output/accumulation, HiFi4.*

| reduce dim | `inline` vs library @ 32 reduced tiles | dispatch crossover (reduced tiles) | note |
|---|---:|---:|---|
| `row` (width) | **5.35×** | ≥4 | wins the most — `REDUCE_ROW` is the library's dearest datapath |
| `col` (height) | **3.37×** | ≥8 | wins least — the FPU `REDUCE_COL` datapath is already cheap |
| `scalar` (both) | **4.88×** | ≥8 | also removes the library's `REDUCE_COL` DST/chunk limit |

```
  inline ÷ reduce_tile, median ns per reduce (bf16 in, fp32 acc, single BH core):
  reduced tiles →     1t      2t      4t      8t     16t     32t
  row   reduce_tile   296     417     647    1109    2033    3884
  row   inline        441     442     463     502     574     726     (0.67 → 5.35×;  wins ≥4t)
  col   reduce_tile   235     298     421     665    1155    2140
  col   inline        352     352     368     412     483     634     (0.67 → 3.37×;  wins ≥4t)
  scal  reduce_tile   290     414     651    1117    2057    3928
  scal  inline        522     523     540     582     656     806     (0.56 → 4.88×;  wins ≥4t)
```

**Reading of the result:**
- **It generalizes to a full 2-D block, and the per-output loop is linear.** A multi-output block costs ≈ `out_tiles × the single-output cost` (see [`report_reduced_sweep.md`](report_reduced_sweep.md)); the reduce library's cost, by contrast, climbs with every input tile.
- **The crossover is dim-dependent** — the one caveat for a shared helper. The fast path amortizes its finalize over the accumulate, so it only wins once there are enough reduced tiles. The FPU `REDUCE_COL` datapath is cheaper than `REDUCE_ROW`, so **row wins the most and col the least**. The `dispatch` thresholds are per-dim (row=4, col=8, scalar=8) and chosen with the library opt-in path's per-call init in mind, so `dispatch` is never slower than the library.
- **Accuracy: equal in fp32, better in bf16.** The SFPU collapses the 32 columns in fp32 internally before one output rounding, so the fast path's bf16 error is lower (e.g. `row` 3.3e-3 vs 5.8e-3, ~0.8 ULP vs 1.6 ULP @ `2×8`). Both are near-exact in fp32.
- **Takeaway:** for reducing a wide 2-D block, the per-output accumulate+SFPU-finalize loop is a strong fast path *and* sidesteps the `REDUCE_COL` DST/chunk limit — but it is a **dispatched fast path, not a replacement**: it loses below the (dim-dependent) crossover, wins least on col, and this microbench is compute-bound and single-core (most real reductions are data-movement-bound, where the compute delta won't show).

## Beyond the bake-off — other `AccumulateViaAdd` capabilities (correctness-validated)
The same `accumulate + SFPU-finalize` datapath handles three further cases the test bench exercises. They are
**correctness** tests (matched against torch), not part of the perf table:

- **Partial (non-tile-aligned) row/col reduce.** When the reduce dimension isn't a tile multiple, the last
  reduce-dim tile is folded in with a **masked accumulating broadcast-mul** (a `0/1` mask tile: row-0 for row,
  col-0 for col), so the padding contributes `0`; the bulk stays pure `add_tiles` (fidelity-flat) and only the
  one partial tile is a mul, with the mean dividing by the true element count. Tests: `test_reduce_block_partial_{row,col}`;
  on-device cost of the masked tile in `test_reduce_block_partial_perf`.
- **Streaming (`WaitAndPopPerTile`).** Instead of a resident block, the reduce-dim tiles stream through DST in
  pairs — **DST itself is the accumulator** (`add_tiles(acc_to_dest)`), so only ~2 input tiles are resident at a
  time and the reduce dim can be arbitrarily large with no L1 bulk (contiguous row/scalar). Test: `test_reduce_block_streaming`.
- **Cross-call accumulate.** A large reduce split across several `reduce()` calls: a **raw** partial-sum tile per
  output lives in a CB and each later chunk folds it into the *same* pairwise `add_tiles` **natively** (no
  `binary_dest_reuse`) — by parity of the chunk's tile count (even → one `copy_tile` reload as the seed; odd →
  the accumulator is the last add's operand) — and the SFPU finalize runs only on the last chunk. The library
  reconfigures the unpacker data format around each accumulator read, so the accumulator CB keeps full fp32
  precision. Test: `test_reduce_block_accumulate` (`SUM`, `num_chunks` chunks ⇒ `num_chunks × sum`).

## Run the predefined sweep (regenerates the reports)
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_reduce_block.py
```
Writes [`report.md`](report.md) (the 2-D block comparison) and [`report_reduced_sweep.md`](report_reduced_sweep.md) (the crossover-vs-reduced-tiles sweep + the linearity check) when `REDBLK_REPORT` / `REDBLK_SWEEP_REPORT` point at them.

## Code
`program_descriptor_with_inline_kernels.py` — the inline fast kernel (per-output-tile accumulate with the per-dim input-subset map + SFPU finalize), the library kernel (`algo` selects `ReduceTile` vs `AccumulateViaAdd`), and the AVG scaler dataflow kernel; `dispatch` routes between the library paths per `(dim, reduced-tile-count)`.
