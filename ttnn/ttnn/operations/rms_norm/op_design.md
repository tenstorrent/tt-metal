# Operation Design: rms_norm

Root-mean-square layer normalization over the last dimension:

```
RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma
```

`gamma` is an optional `[1,1,1,W]` scale (ROW_MAJOR). Output shape, dtype and layout match the input.

---

## Blocking Model

The op is a **parameterized row-parallel streaming reduction**. The math reduces over `W` (the last
dim); every row's result is independent. That fixes the scheme before anything else:

| Axis | Character | Block-factor knob | Phase-1 value | Core-assignment | Later unlock |
|------|-----------|-------------------|---------------|-----------------|--------------|
| **Row** (flattened `N·C·…·H`, counted in **tile-rows**) | **independent** — each row's RMS is computed in isolation, no cross-row dependency | `row_block_tiles` (tile-rows processed per outer pass) | `1` | all tile-rows on **one core** (phase-1); grid mapping = `split_work_to_cores(grid, total_tile_rows)` | **knob-turn** — assign tile-rows to more cores (no comms); also `HEIGHT_SHARDED` |
| **W** (hidden / reduced dim, counted in **W-tiles** `Wt = ceil(W/32)`) | **dependent** — the RMS denominator spans the whole W (a `SUM` reduce) | `w_block_tiles` (W-tiles streamed per reduce chunk) | `1` | single core owns the full `W` of each of its rows (phase-1) | **scheme-change** — split `W` across cores → cross-core partial-`Σx²` combine (mcast); `WIDTH_SHARDED` / `BLOCK_SHARDED` |
| **Gamma-W** (weight, `[1,W]`, reused across every row) | **reuse-shared** — the same gamma feeds every row along the Row axis | (streamed with `w_block_tiles`) | re-read per row | streamed from DRAM per row (phase-1) | **reuse scheme-change** — hold gamma resident when it fits, or mcast once |

**Buffer-depth knobs** (distinct from block factors — they buy data-movement↔compute overlap, not reuse):
`cb_input_tiles` = `2·w_block_tiles` (double-buffered, phase-1 depth 2), `cb_output_tiles` = `2·w_block_tiles`,
`cb_gamma_*` = `2·w_block_tiles`, `cb_input_rm`/`cb_output_rm` = 2 stick-blocks. Sequential compute-to-compute
intermediates are single-depth full-block: `cb_xsq` = `w_block_tiles`, `cb_norm` = `w_block_tiles`, accumulator
`cb_sumsq` = `row_block_tiles` (=1). `cb_scaler` = 2 (full + partial scaler tiles).

**Every knob is a parameter, never a constant.** No CB is sized to `Wt` or any whole-op dimension — the input
is *streamed* over `W` in `w_block_tiles` chunks and reduce-accumulated into a 1-tile running sum, so per-core L1
stays bounded for arbitrarily wide `W` (incl. the `W=32768` loose cases). Each knob has **one source of truth**:
`w_block_tiles` and `row_block_tiles` are host constants → every dependent (CB page counts, loop trip counts,
`num_w_blocks = ceil(Wt / w_block_tiles)`) is derived from them, never restated.

**Bandwidth ranking (why Row is the primary split axis):** splitting the **Row** axis across cores moves each
core's `x` rows exactly once per pass with **no cross-core traffic** (independent axis) and keeps each row's `W`
contiguous; the only redundancy is re-reading the small `[1,W]` gamma per core. Splitting the **W** axis instead
forces a cross-core gather of partial `Σx²` plus a broadcast-back of the combined statistic (extra NoC bytes +
semaphore sync) and fragments the contiguous per-row `W` transfer. Fewer bytes + no comms ⇒ **split rows first**;
split `W` only when a single row's `W` cannot fill/fit the grid (the lamp). This matches the golden
`feature_spec.py` note ("HEIGHT_SHARDED → reduction stays LOCAL … WIDTH/BLOCK → reduction is CROSS-CORE").

**Lamp — scheme-changes phase-1 leaves reachable (does not foreclose):**
1. **Cross-core W-split** (dependent-axis combine): phase-1's streaming reduce already materializes a *per-core
   local partial `Σx²`* (the `cb_sumsq` accumulator). A cross-core combine only has to gather those partials,
   fold them (sum, or Welford for stability), and broadcast the finalized `1/rms` back — via `mcast_pipe`
   (`ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp`) as described in `.claude/references/cross_core_reduction_design.md`.
   `WIDTH_SHARDED` / `BLOCK_SHARDED` land here.
2. **HEIGHT_SHARDED** (row axis sharded): the reduction stays local per core; this is the Row knob-turn with the
   tile-row→core assignment pinned by the shard spec instead of `split_work_to_cores`.
3. **Gamma reuse**: hold gamma resident (or mcast once) instead of re-reading per row.

Sharding, when the op accepts it, is *this same scheme* with block factors/core-assignment pinned by the shard
spec — HEIGHT is a knob-turn, WIDTH/BLOCK is lamp #1.

**Catalog evidence** (`ttnn/ttnn/operations/examples/master.md`) informing the knobs:
- `double_buffer` (T2) → `cb_input_tiles`/`cb_output_tiles` depth `2·w_block_tiles`; issue a *block* of reads
  then one barrier. Sets the phase-1 double-buffer depth and the "raise `w_block_tiles` to ~4–8" refinement.
- `compute_block_size` (T2) → raising `row_block_tiles`/`w_block_tiles` amortizes per-call reconfig+init over
  more tiles (measured 1.65× whole-pass vs tile-row-by-tile-row). Justifies these as first perf knob-turns.
- `row_reduce_accumulate` / `reduce_block` (T2) → the `Σx²` row reduce; accumulate + SFPU-finalize is the
  perf/accuracy refinement for wide rows and the `AccumulateViaAdd` partial path is the alignment fallback.
- `reader_placement` (T1) → when the Row split goes multi-core, spread the reader line `row_wise=True`.
- `tensix_all_reduce*` (T3) → the transport/topology for lamp #1's cross-core combine.

---

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (reduction + broadcast eltwise) |
| Goal | Normalize each row of `input` by the RMS over the last dim, optionally scaled by `gamma`. |
| Math | `out[..., i, j] = x[..., i, j] / sqrt( (1/W) * Σ_k x[..., i, k]^2 + epsilon ) * gamma[j]` |
| Mode | Hybrid (streaming reduce-accumulate + broadcast eltwise) |
| References | `ttnn/cpp/ttnn/kernel_lib/{reduce_helpers_compute,reduce_helpers_dataflow,streaming_reduce_helpers,eltwise_convenience,eltwise_math,eltwise_scalar,tilize_helpers,untilize_helpers,dest_helpers,mcast_pipe}.hpp`; `.claude/references/cross_core_reduction_design.md`; `references/precision_convention.md`; `eval/golden_tests/rms_norm/feature_spec.py` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank ≥ 2; bf16/fp32; RM or TILE; INTERLEAVED (phase-0) | — | — |
| `gamma` | `Optional[ttnn.Tensor]` | no (kw-only) | ROW_MAJOR `[1,1,1,W]`, last dim == input W | `None` | — |
| `epsilon` | `float` | no (kw-only) | `> 0` | `1e-6` | RT (compute) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | no (kw-only) | phase-0: `fp32_dest_acc_en=True`; any `math_fidelity`/`math_approx_mode` | resolved via `default_compute_kernel_config()` | build |
| `w_block_tiles` | host constant (block-factor knob) | — | `1 … Wt` | `1` | CT (compute/reader) |
| `row_block_tiles` | host constant (block-factor knob) | — | `≥ 1` | `1` | CT (compute) |

`default_compute_kernel_config()` is the single exported factory (`HiFi4`, `fp32_dest_acc_en=True`,
`math_approx_mode=False`); `None` resolves through it and the golden axis-tagger reads the same factory
(`references/precision_convention.md`). Config is passed through as-is to the compute-kernel descriptor
(`config=compute_kernel_config`) after `validate()`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank ≥ 2, `(..., H, W)`. Non-tile-aligned `H` and/or `W` supported natively (both layouts). |
| Dtype | `bfloat16`, `float32` (phase-0). `bfloat8_b` is TARGET ambition (refinement). |
| Layout | `ROW_MAJOR` or `TILE` (both native — **no host-side layout conversion**). |
| Memory | INTERLEAVED (phase-0). Sharded = lamp. |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input |
| Dtype | same as input |
| Layout | **matches input layout** |
| Memory | matches input placement (INTERLEAVED phase-0) |

### Gamma (optional)

| Property | Requirement |
|----------|-------------|
| Shape | `[1,1,1,W]`, `W == input.shape[-1]` (else `ValueError`) |
| Dtype/Layout | `ROW_MAJOR` (phase-0 contract); TILE/mixed-dtype gamma are TARGET ambition |
| Absence | canonicalized to the `"none"` sentinel for `gamma_dtype`/`gamma_layout` (always legal) |

## Compute-regime selection

Phase-1 selects **one** regime, on input layout (compile-time define built by the host):

| Predicate | Regime | Effect |
|-----------|--------|--------|
| `input_tensor.layout == ROW_MAJOR_LAYOUT` | `rm` | compute tilizes input on the way in and untilizes output on the way out |
| `input_tensor.layout == TILE_LAYOUT` | `tile` | compute reads/writes tiles directly (no tilize/untilize) |

Gamma is **always** `ROW_MAJOR` → the gamma-tilize step is present whenever `gamma` is supplied, independent of
the input-layout regime. **Regime-pinned tests required**: the acceptance test parametrizes both layouts so a
regime that only triggers on some inputs cannot pass silently. `dim` is fixed to `-1` (single reduce direction),
so there is no reduce-direction regime.

## Work Distribution

Realizes the Blocking Model's Row-axis core-assignment.

| Field | Value |
|-------|-------|
| Work unit | a **block** = `row_block_tiles` tile-rows (phase-1: 1 tile-row); each block streams its full `W` in `w_block_tiles` chunks |
| Grid | phase-1 = **single core** (`CoreCoord(1,1)`). Multi-core = `ttnn.split_work_to_cores(grid, total_tile_rows)` (`ttnn-python-utility-bindings.md:194`) — a runtime-arg change, no loop-nest change |
| Total tile-rows | `total_tile_rows = Σ_images ceil(H_image / 32)` over the flattened leading dims — **alignment-aware `ceil`, per-image** (each `(N,C,H,W)` image is tile-padded independently; **never** `floor(N·C·H / 32)`) |
| W-tiles | `Wt = ceil(W / 32)`; `num_w_blocks = ceil(Wt / w_block_tiles)`; last block's last tile is the partial one when `W % 32 != 0` |
| Per-core work | its assigned contiguous span of tile-rows; loop `for rb in blocks_of(my_rows, row_block_tiles)` |
| Remainder | `split_work_to_cores` returns two core groups (`g1`/`g2`) covering the `total_tile_rows % num_cores` remainder; handle both groups explicitly. Phase-1 single core ⇒ no remainder. |

## Circular Buffers

Page counts are functions of the block/buffer knobs only — never of `Wt`/`W`/sequence length. Producer/consumer
each name exactly one **thread**; where a row is regime-conditional it is annotated (the two regimes are separate
compiled programs, so the invariant holds per build).

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_rm` | 0 | stick (`W·elt`, aligned) | `2` stick-blocks | input dtype | reader | compute *(rm regime: tilize)* | rm regime only |
| `cb_input_tiles` | 1 | tile | `2·w_block_tiles` | input dtype | reader *(tile)* / compute-tilize *(rm)* | compute (square / normalize-mul) | streamed, both passes |
| `cb_scaler` | 2 | tile | `2` (full + partial) | **bfloat16** | reader | compute (reduce) | persistent per program |
| `cb_gamma_rm` | 3 | stick (`W·elt`) | `2` stick-blocks | gamma dtype | reader | compute (tilize gamma) | gamma present only |
| `cb_gamma_tiles` | 4 | tile | `2·w_block_tiles` | compute dtype | compute (tilize gamma) | compute (gamma mul) | gamma present only |
| `cb_output_tiles` | 16 | tile | `2·w_block_tiles` | output dtype | compute (final mul) | writer *(tile)* / compute-untilize *(rm)* | streamed |
| `cb_output_rm` | 17 | stick (`W·elt`, aligned) | `2` stick-blocks | output dtype | compute (untilize) | writer | rm regime only |
| `cb_xsq` | 24 | tile | `w_block_tiles` | compute dtype (fp32 if fp32-in) | compute (square) | compute (reduce) | pass-1 scratch |
| `cb_sumsq` (= `cb_rms_recip`) | 25 | tile | `row_block_tiles` (=1) | compute dtype (fp32 if fp32-in) | compute (reduce-accum → in-place rsqrt) | compute (normalize mul) | held across pass-2 |
| `cb_norm` | 26 | tile | `w_block_tiles` | compute dtype | compute (x·rms mul) | compute (gamma mul) | gamma present only; pass-2 scratch |

`cb_sumsq` is touched only by the **compute** thread (reduce-accumulate → `transform_in_place` rsqrt → read by
the normalize mul): a single-thread accumulator/scratch, so the one-producer-thread/one-consumer-thread invariant
holds (no cross-thread sharing). When `gamma` is absent, the x·rms mul writes straight to `cb_output_tiles` and
`cb_norm`/`cb_gamma_*` are not allocated.

## API Mapping

Every mechanism is a kernel-lib helper; the one raw block (the rsqrt+eps finalizer) is inside a helper's
documented lambda hook. Block/chunk knobs called out.

| Phase | Type | Function | File:Line | Template Params / Args (knobs **bold**) | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|----------------------------------------|----------|-----------|--------------|
| boot | helper | `compute_kernel_hw_startup(...)` | `reduce_helpers_compute.hpp:40` (contract) | — | — | — | called once before any helper; one boot per pack-target stage |
| tilize input *(rm)* | helper | `tilize()` | `tilize_helpers.hpp:187` | **`block_width_tiles = w_block_tiles`**, `cb_input_rm→cb_input_tiles`; `Fp32Mode::Fast` | `cb_input_rm` | `cb_input_tiles` | per W-block; `num_blocks=row_block_tiles` |
| square | helper | `square()` | `eltwise_convenience.hpp:116` | `<cb_input_tiles, cb_xsq>`; `EltwiseShape::tiles(`**`w_block_tiles`**`)` | `cb_input_tiles` | `cb_xsq` | streaming (DEST=1 tile/iter) |
| Σx² reduce (stream) | helper | `accumulate_reduce_block<SUM,REDUCE_ROW>()` | `streaming_reduce_helpers.hpp:53` | `ReduceInputBlockShape::row(`**`w_block_tiles`**`)`, `b`, `num_w_blocks`, `partial=last_tile_at(1)` when `W%32` | `cb_xsq` | `cb_sumsq` | scaler CB preloaded; partial scaler routed only on last block |
| mean+eps → rsqrt | helper + raw hook | `transform_in_place()` + `add_unary_tile` / `rsqrt_tile` | `streaming_reduce_helpers.hpp:110`; hook LLK `rsqrt`/`add_unary` | lambda: `add_unary_tile(0, epsilon); rsqrt_tile(0)` | `cb_sumsq` | `cb_sumsq` | 1 tile in place; `epsilon` is RT arg |
| tilize gamma *(gamma)* | helper | `tilize()` | `tilize_helpers.hpp:187` | **`w_block_tiles`**, `cb_gamma_rm→cb_gamma_tiles` | `cb_gamma_rm` | `cb_gamma_tiles` | per W-block, pass-2 |
| normalize x·(1/rms) | helper | `mul<…, BroadcastDim::Col>()` | `eltwise_convenience.hpp:94` | `<cb_input_tiles, cb_sumsq, cb_norm/cb_output_tiles, BroadcastDim::Col>`; B = `OperandKind::Scalar`, held (non-pop) | `cb_input_tiles`,`cb_sumsq` | `cb_norm` or `cb_output_tiles` | rms held across all pass-2 W-blocks; popped once at row end |
| gamma scale *(gamma)* | helper | `mul<…, BroadcastDim::Row>()` | `eltwise_convenience.hpp:94` | `<cb_norm, cb_gamma_tiles, cb_output_tiles, BroadcastDim::Row>` | `cb_norm`,`cb_gamma_tiles` | `cb_output_tiles` | gamma row0 broadcast down rows |
| untilize output *(rm)* | helper | `untilize()` | `untilize_helpers.hpp:145` | **`block_width_tiles=w_block_tiles`**, `cb_output_tiles→cb_output_rm` | `cb_output_tiles` | `cb_output_rm` | per W-block |
| scaler prep | helper | `prepare_reduce_scaler<cb_scaler,SUM,REDUCE_ROW>()` (+ partial tile) | `reduce_helpers_dataflow.hpp:58` | pool-type-aware `<SUM, REDUCE_ROW>`; `scaler_f = 1.0f/W`; `valid = 32` (tile0) / `W%32` (tile1) | — | `cb_scaler` | **bfloat16 CB**; NOT legacy `prepare_reduce_scaler<cb>` |
| addressing | helper | `TensorAccessor` | `tech_reports/tensor_accessor/tensor_accessor.md` | interleaved DRAM; TensorAccessorArgs last in CT args | — | — | reader/writer |
| **lamp** cross-core W | helper | `SenderPipe`/`ReceiverPipe`/`McastRect` | `mcast_pipe.hpp` | (not phase-1) | — | — | combine per-core partial `Σx²` |

**Helpers considered and rejected — none.** Every compute phase is covered by a kernel-lib helper. The only raw
LLK calls are `add_unary_tile` + `rsqrt_tile` **inside** `transform_in_place`'s lambda hook, which is exactly the
helper's documented use ("a chain like `mul_unary_tile, add_unary_tile, rsqrt_tile`", `streaming_reduce_helpers.hpp:104-105`);
these are not a helper bypass. (An `eltwise_chain(single, CopyTile, AddUnary<>{eps}, Rsqrt<>, PackTile)` finalizer
is an equivalent alternative but `transform_in_place` is purpose-built for the in-place 1-tile rsqrt-with-eps and
is preferred.)

## Compute Phases

Per assigned tile-row (`row_block_tiles = 1`). Two passes over `W` (the input is re-read for pass 2 — the price
of bounded L1; documented reuse-lamp candidate is holding the row/gamma resident).

| # | Operation | Helper? | Input CB (name, tiles, state) | Output CB (name, tiles) | CB State After |
|---|-----------|---------|-------------------------------|-------------------------|----------------|
| 0 | boot `compute_kernel_hw_startup` | contract | — | — | HW init done |
| — | **Pass 1: statistics** (loop `b = 0 … num_w_blocks-1`) | | | | |
| 1a | *(rm)* tilize input W-block | `tilize` | `cb_input_rm` (`w_block_tiles` sticks) | `cb_input_tiles` (`w_block_tiles`) | tiles ready |
| 1b | square W-block | `square` | `cb_input_tiles` (`w_block_tiles`) | `cb_xsq` (`w_block_tiles`) | `cb_input_tiles` popped |
| 1c | reduce-accumulate `Σx²` | `accumulate_reduce_block` | `cb_xsq` (`w_block_tiles`) | `cb_sumsq` (1) | `cb_xsq` popped; `cb_sumsq` accumulates |
| 2 | mean+eps → `1/rms` | `transform_in_place` | `cb_sumsq` (1, = `Σx²/W`) | `cb_sumsq` (1) | holds `1/rms`, persists through pass 2 |
| — | **Pass 2: normalize** (loop `b = 0 … num_w_blocks-1`, re-read x) | | | | |
| 3a | *(rm)* tilize input W-block | `tilize` | `cb_input_rm` | `cb_input_tiles` (`w_block_tiles`) | |
| 3b | *(gamma)* tilize gamma W-block | `tilize` | `cb_gamma_rm` | `cb_gamma_tiles` (`w_block_tiles`) | |
| 3c | `x · (1/rms)` (bcast Col) | `mul<Col>` | `cb_input_tiles`, `cb_sumsq` (held) | `cb_norm` or `cb_output_tiles` | `cb_input_tiles` popped; `cb_sumsq` NOT popped |
| 3d | *(gamma)* `· gamma` (bcast Row) | `mul<Row>` | `cb_norm`, `cb_gamma_tiles` | `cb_output_tiles` (`w_block_tiles`) | `cb_norm`,`cb_gamma_tiles` popped |
| 3e | *(rm)* untilize output W-block | `untilize` | `cb_output_tiles` | `cb_output_rm` (`w_block_tiles` sticks) | pushed to writer |
| 4 | end of row | — | pop `cb_sumsq` (1) | — | ready for next tile-row |

## Broadcast Verification

Binary ops used (both `mul`). Reduce output for `REDUCE_ROW` is column-shaped `(rows,1)`, valid in **col 0**.

| Phase | Op | CB_A valid region | CB_B valid region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| 3c | `mul` `x·(1/rms)` | `cb_input_tiles` — all `[32,32]` | `cb_sumsq` — Col0 (per-row scalar, `(rows,1)`) | **Col** (broadcast the per-row scalar across W columns) |
| 3d | `mul` `·gamma` | `cb_norm` — all `[32,32]` | `cb_gamma_tiles` — Row0 (`[1,W]` weight, valid in tile row 0) | **Row** (broadcast the single gamma row down all 32 rows) |

`BroadcastDim` names the axis broadcast (per `eltwise_chain.hpp:501-513`): a `REDUCE_ROW` result broadcasts back
across columns via `Col`; a `[1,W]` weight broadcasts down rows via `Row`.

## Key Risks and Gotchas

- **No `Wt`-sized CB.** The reduce **streams** `W` in `w_block_tiles` chunks and accumulates into the 1-tile
  `cb_sumsq`; the input CB holds only `2·w_block_tiles`. Never widen a CB to `Wt` — that is a latent OOM on the
  wide-W cells (`W=4096…32768`) and breaks the blocking model.
- **Non-tile-aligned W = masked reduce, not zero-padding.** The reader emits two scaler tiles (`cb_scaler`
  pages 0/1), both carrying `1/W`; page 1 zeroes columns beyond `W%32`. The reduce uses page 1 (partial) only on
  the last W-tile of the last block (`ReducePartialScaler::last_tile_at(1)`); the mean divides by the *true* `W`.
  This is what makes the RMS denominator reflect only valid elements (requirement).
- **Non-tile-aligned H is a reader/writer concern only.** Padding tile-rows compute their own (discarded) RMS;
  the per-row reduce over `W` is unaffected. Reader/writer bound stick counts by the true `H` per image.
- **Scaler CB must be bfloat16 and pool-type-aware.** Use `prepare_reduce_scaler<cb_scaler, PoolType::SUM,
  ReduceDim::REDUCE_ROW>(1.0f/W, valid)`; `1/W` is a runtime float (W varies per input) so the caller-value
  overload (not `calculate_and_prepare_*`'s compile-time `reduce_factor`) is correct. Never the legacy
  `prepare_reduce_scaler<cb>`.
- **`cb_sumsq` must persist across all of pass 2.** It holds `1/rms` for the whole row; the pass-2 `mul` reads it
  with a non-consuming (`Scalar`/held) lifecycle and it is popped exactly once at row end. Producer push count
  (1 from reduce-accum) = consumer wait count for every CB.
- **DEST budget = fp32 limit from the start.** Phase-0 is `fp32_dest_acc_en=True` → half-sync DEST = **4 tiles**
  (`DEST_AUTO_LIMIT`). All streaming chains use 1 DEST tile/iter and the reduce/rsqrt stay ≤ 4; `w_block_tiles`
  inflates only L1 depth, never DEST. `float32` input requires `fp32_dest_acc_en=True` (validate rejects
  `float32 + False`); intermediate CBs (`cb_xsq`, `cb_sumsq`, `cb_norm`) are fp32 for fp32 input, bf16 otherwise.
- **Sequential compute intermediates full-block.** `cb_xsq`/`cb_norm` are `w_block_tiles` (both producer and
  consumer are compute → cannot pipeline), single-depth.
- **`compute_kernel_hw_startup()` before any helper**, one boot per distinct pack-target stage; never inside a
  loop.
- **Both layouts native.** The Python entry point performs **no** `to_layout`/`tilize`/`untilize`/`pad`/`slice`;
  tilize/untilize live in the compute kernel, gated by the `rm` regime define. Output layout = input layout.

## Validation (Python entry point)

| Check | Raises | Golden axis |
|-------|--------|-------------|
| `rank(input) < 2` | `ValueError` | (structural) |
| `gamma is not None and gamma.shape[-1] != input.shape[-1]` | `ValueError` | (structural) |
| axis value ∉ `SUPPORTED` (dtype/layout/alignment/rank/gamma_*/memory_layout) | `UnsupportedAxisValue` (`NotImplementedError`) | all TARGET axes |
| `float32` input + `fp32_dest_acc_en=False` | `ExcludedCell` (`NotImplementedError`) — **EXCLUSIONS**, not INVALID (`precision_convention.md`) | `dtype`×`fp32_dest_acc_en` |

Axis names mirror `feature_spec.py` TARGET exactly: `dtype`, `fp32_dest_acc_en`, `layout`, `alignment`
(`tile_aligned`/`w_non_aligned`/`h_non_aligned` via `tag_alignment`), `rank` (via `tag_rank`), `gamma_mode`,
`gamma_dtype`, `gamma_layout` (`"none"` sentinel always legal), `memory_layout`. Index axis `dim` is fixed to
`-1` (reduce over last dim).

## Structural impossibilities

None beyond those already in `feature_spec.py` INVALID (`bfloat8_b`+`ROW_MAJOR` for input and gamma; the
`no_gamma` ⇄ `"none"` sentinel coupling). No op-specific INVALID candidate to fold in. `float32 +
fp32_dest_acc_en=False` is an **EXCLUSION** (legal but refused), not INVALID.
