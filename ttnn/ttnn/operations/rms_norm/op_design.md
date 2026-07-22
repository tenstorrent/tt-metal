# Operation Design: rms_norm

> **This document is the single source of truth for the implementer.** Every statement is a
> decision. The kernel `.cpp` files, the `SUPPORTED`/`EXCLUSIONS` block, `op_requirements.md`,
> and `feature_spec.py` are authored by others — this file fixes the blocking scheme, CB layout,
> helper selection, and work-split so no structural design remains.

---

## 1. Blocking Model (decided first — everything downstream references it)

RMSNorm normalizes each row of the input along the last dim `W`:

```
out[..., h, :] = x[..., h, :] * rsqrt( mean(x[..., h, :]^2) + eps ) * gamma[:]
```

The op's data is a rank-≥2 tensor viewed as `(NC, H, W)` where `NC = prod(shape[:-2])`. In tile
geometry each `(NC, H)` image is tile-padded independently, so:

* **tile-rows** `R = NC * ceil(H/32)` — each tile-row is 32 logical rows.
* **W-tiles** `Wt = ceil(W/32)` per tile-row.

### Axis table

| Axis | Character | Block-factor knob | Phase-1 value | Core-assignment | Later unlock |
|------|-----------|-------------------|---------------|-----------------|--------------|
| **Row axis** (`R = NC·ceil(H/32)` tile-rows) | **independent** — each row's RMS depends only on its own `W` elements; no cross-row dependency | `ROWS_PER_CALL` (tile-rows batched into one helper invocation) | `1` (one tile-row per pass; rows are independent so batching buys no reuse, only L1) | **spread across the whole grid** — `split_work_to_cores(R, grid, row_wise=True)`; each core owns a contiguous run of `R` tile-rows | **knob-turn** (assign rows to more cores / raise `ROWS_PER_CALL`; loop nest unchanged) |
| **W (hidden) axis** (`Wt` tiles) | **dependent** — the RMS sum `Σx²` spans all of `W`; a result cannot be produced until the whole row is reduced | `BLOCK_SIZE` (W-tiles per square/reduce/normalize invocation) | coarse: `pick_block_size(Wt)` = largest divisor of `Wt` that is `≤ 8` (matches the `double_buffer` 4–8 sweet spot; **not** `1`) | **single-core within a row** in phase-1 — one core reduces the whole `W` sequentially (cheap accumulate across `NUM_BLOCKS = Wt/BLOCK_SIZE` blocks) | **scheme-change** (split `W` across cores → cross-core partial-`Σx²` combine via mcast/all-gather) |
| **gamma** (`[1, W]`, `Wt` tiles) | **reuse-shared** — the same `gamma` vector multiplies *every* tile-row | (streamed in `BLOCK_SIZE` chunks, one CB) | streamed per-row in pass 2 (re-read once per tile-row) | replicated: each core reads its own copy from DRAM | **knob-turn** (hold `gamma` resident per core, or mcast it once) — bounded only under the fits-in-L1 predicate; see Lamp |

**Buffer-depth knobs (per streaming CB):** `cb_x_in`, `cb_gamma`, `cb_out` are double-buffered
(`DEPTH = 2`, `num_pages = DEPTH * BLOCK_SIZE`). `DEPTH` is a parameter; phase-1 value `2`
(the `double_buffer` catalog result: `block≈4` + depth-2 reaches the single-core NoC ceiling;
depth>2 does not help).

**Single source of truth for each knob.** `BLOCK_SIZE` is computed once on the host
(`pick_block_size(Wt)`), passed as one CT arg to reader + compute; `NUM_BLOCKS = Wt / BLOCK_SIZE`,
every CB page count (`DEPTH*BLOCK_SIZE`, `2*BLOCK_SIZE`), and every loop bound derive from it.
`DEPTH`, `ROWS_PER_CALL`, and the core grid are host constants/args — turning any knob later is a
one-line change, no duplicated literal.

### Bandwidth ranking of candidate splits (qualitative, no ns)

| Candidate split | Bytes moved / fan-out | Combine cost | Verdict |
|---|---|---|---|
| **Row axis (independent)** — chosen primary | each row read (2× for the 2-pass), written once; disjoint per core | **none** | **primary split.** Fills the grid whenever `R ≥ #cores`. Zero cross-core traffic. |
| **W axis (dependent)** | same input bytes, but each core streams only `W/K`; adds a cross-core combine of `K` partial-`Σx²` tiles (1 tile/core — tiny) | 1 gather+combine+broadcast round (mcast/all-gather) | **fallback lamp.** The available parallelism when `R < #cores` (decode: `R=1` tile-row → row-split uses 1 core). Cheap combine (stats are tiny) but only pays when rows under-fill the grid. |
| **gamma reuse (mcast)** | saves re-reading `gamma` (`Wt` tiles) per row | 1 mcast of a small vector | **not worth it in phase-1** (catalog `shared_input_reuse`: gamma is small, per-core resident/streamed is fine). Lamp. |

Row axis wins: it is the only split with **zero** cross-core combine, and it saturates DRAM
bandwidth once enough independent tile-rows fill the grid (catalog `double_buffer`: 64 cores hit
~DRAM peak untuned — grid utilization dominates).

### Scheme phase-1 commits to

**Row-parallel, bounded two-pass streaming reduce over `W`, multi-core from day 1.** Independent
tile-rows are spread across the whole grid (the easy parallelization, realized now). Within a
core, each tile-row streams its `W` twice through fixed-size CBs (pass 1 = `Σx²`→rstd; pass 2 =
`x·rstd·gamma`), so **no CB is sized by an op dimension** and the op runs at any `W` without OOM.
Every knob (`BLOCK_SIZE`, `DEPTH`, `ROWS_PER_CALL`, grid) is a parameter at its trivial value, not
an inlined constant.

### Lamps (scheme-changes phase-1 leaves room for; structure keeps them reachable)

1. **Resident single-read fast-path (knob-turn / dual-path).** When a size predicate holds
   (`Wt·tile_bytes + intermediates ≤ L1 budget`), load the row's `Wt` tiles **once**, hold
   resident, and do both passes over L1 (one DRAM read instead of two; and `gamma` held resident,
   reused across the core's rows instead of re-read). Model as a dual-path dispatched on an
   explicit `fits_in_l1` predicate with the streaming path as fallback. Reachable because the
   compute phase sequence is identical — only the input policy (`WaitUpfrontNoPop` vs streaming)
   and CB sizing change. **This is what the decode/small-`W` perf cases want.**
2. **Cross-core W-split (scheme-change).** When `R < #cores` (rows under-fill the grid) or
   `Wt` exceeds one core's L1 (the `LOOSE_CASES` `W=16384/32768`), split `W` across `K` cores:
   each core reduces its `W/K` slice to a partial `Σx²`, then one cross-core round combines the
   `K` partials (gather + add + broadcast) before every core normalizes its own slice. Built on
   `mcast_pipe` (`ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp`) + a partial-stat gather
   (see `cross_core_reduction_design.md`, Pattern A→B). **Reachable independent of whether the
   caller pre-shards** — phase-1 already computes a *partial* `Σx²` per block via
   `accumulate_reduce`, so a cross-core combine is an added finalize round, not a loop-nest
   rewrite. This is the **logical width/block-shard** of the input.
3. **Physical sharded inputs (the lamps expressed as `memory_layout` TARGET values).**
   * `HEIGHT_SHARDED` → the row-shard is pre-placed in each core's L1; reduction stays **local**
     per core → **knob-turn**: back `cb_x_in` on the sharded buffer via
     `ttnn.cb_descriptor_from_sharded_tensor` (zero-copy, **no NoC read**) and consume the whole
     resident shard as the per-core block. Same scheme, block+placement pinned by the shard.
   * `WIDTH_SHARDED` / `BLOCK_SHARDED` → the hidden dim is split across cores; the reduction is
     **cross-core** → **scheme-change** = lamp 2 with geometry pinned and the data slice
     pre-placed in L1 (consumed locally; only the tiny stat-combine crosses the NoC).

---

## 2. Overview

| Field | Value |
|-------|-------|
| Classification | compute (reduction + broadcast elementwise) |
| Goal | RMS-normalize a tensor along its last dim, optional per-feature scale `gamma`. Native RM+TILE, native non-tile-aligned H/W, no host-side layout/pad workarounds. |
| Math | `out[...,h,w] = x[...,h,w] * rsqrt( (1/W)·Σ_w x[...,h,w]² + eps ) * gamma[w]` (gamma optional) |
| Mode | Derivative (built from `reduce` / `eltwise` / `tilize` kernel-lib helpers) |
| References | `streaming_reduce_helpers.hpp`, `reduce_helpers_compute.hpp`, `reduce_helpers_dataflow.hpp`, `eltwise_convenience.hpp`, `tilize_helpers.hpp`, `untilize_helpers.hpp`, `dest_helpers.hpp`; analog op `ttnn/ttnn/operations/toy_variance/`; `.claude/references/cross_core_reduction_design.md`; `references/precision_convention.md` |

## 3. Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank ≥ 2; bf16/f32 (bf8b tile-aligned is a later dtype); TILE or RM; interleaved (phase-1) | — | — |
| `gamma` | `Optional[ttnn.Tensor]` | no (kw-only) | RM `(1,1,1,W)`, last dim == input last dim | `None` | — |
| `epsilon` | `float` | no (kw-only) | > 0 | `1e-6` | RT scalar → compute |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | no (kw-only) | resolved through `default_compute_kernel_config()` when `None` | `None` → `default_compute_kernel_config()` | passed as-is to `ComputeConfigDescriptor(config=...)` |
| `BLOCK_SIZE` | host constant | — | divisor of `Wt`, coarse | `pick_block_size(Wt)` (≤8) | CT (reader, compute) |
| `NUM_BLOCKS` | derived | — | `Wt / BLOCK_SIZE` | — | CT (reader, compute) |
| `DEPTH` | host constant | — | ≥1 | `2` | host (CB sizing) |
| `has_partial_w`, `partial_w` | derived | — | `W % 32` | — | CT (reader, compute) |
| grid | `CoreRangeSet` | — | up to device compute grid | full grid | host |

**`default_compute_kernel_config()`** (exported single source of truth; `validate()` and the golden
axis-tagger both read it — never inline the default elsewhere):

```python
def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        math_approx_mode=False,
    )
```

## 4. Tensors

### Input
| Property | Requirement |
|----------|-------------|
| Shape | rank ≥ 2, `(…, H, W)` |
| Dtype | bfloat16, float32 (phase-1); bfloat8_b (later, tile-aligned only) |
| Layout | ROW_MAJOR or TILE (both native) |
| Memory | interleaved (phase-1); `*_SHARDED` are lamps (§1) |

### Output
| Property | Value |
|----------|-------|
| Shape | identical to input |
| Dtype | same as input |
| Layout | **same as input** (RM→RM, TILE→TILE) |
| Memory | same class as input (interleaved phase-1) |

### gamma
`(1,1,1,W)` ROW_MAJOR, one value per feature column. Consumed by the reader; the reader tilizes
it on read for the RM case. May differ in dtype from the input (mixed-precision LLM pattern).

## 5. Dataflow Strategy

Data path (phase-1, interleaved DRAM):

```
DRAM x ──reader──▶ cb_x_in ──compute(square)──▶ cb_xsq ──compute(accumulate_reduce SUM,1/W)──▶ cb_rstd
                                                                     (partial scaler zeros padded W)
cb_rstd ──compute(transform_in_place: +eps, rsqrt)──▶ cb_rstd (=1/RMS, 1 tile/row, held)
DRAM x ──reader(pass2)──▶ cb_x_in ──compute(mul<Col> x·rstd)──▶ cb_norm
DRAM γ ──reader(pass2)──▶ cb_gamma ──compute(mul<Row> norm·γ)──▶ cb_out ──writer──▶ DRAM out
```

**Layout regimes** (the reader/writer boundary; compute core is layout-agnostic — it always works
on tiles in CBs):

| Regime (selected at host time) | Reader | Compute boundary | Writer |
|---|---|---|---|
| **TILE input** | `noc_async_read_tile` → `cb_x_in` (tiles) | consumes `cb_x_in` directly | `noc_async_write_tile` from `cb_out` |
| **ROW_MAJOR input** | read sticks → `cb_x_sticks` (row pages); compute `tilize<BLOCK_SIZE>` → `cb_x_in` | extra `tilize` phase per block per pass | compute `untilize<BLOCK_SIZE>` `cb_out`→`cb_out_sticks`; writer writes sticks |

**gamma tilize.** gamma is RM `(1,1,1,W)`. Reader reads its sticks; compute `tilize` → `cb_gamma`
tiles (phase-1 supports RM gamma — the stated contract). Tiled gamma is a knob-turn (skip the
tilize).

**Cross-core contract for the unlocked W-split (lamp 2), described though phase-1 does not use it.**
Cores in one reduction group own disjoint `W/K` slices of the same tile-rows. Each core computes a
partial `Σx²` tile (already produced by `accumulate_reduce`). One gather leg (workers →
group-master, unicast `noc_async_write` + `noc_semaphore_inc`) collects the `K` partials; the
master adds them (FPU) and multicasts the finalized `1/RMS` back to the group via
`mcast_pipe::SenderPipe`/`ReceiverPipe` (`mcast_pipe.hpp:169/247`); every core then normalizes its
own slice. Group geometry must be **rectangular** (NoC mcast addresses a rectangle) — the host core
assignment guarantees it. Semaphore budget ~2–3 of 16. See `cross_core_reduction_design.md` §2–§4.

## 6. Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one **tile-row** (all `Wt` W-tiles of 32 logical rows), processed as `NUM_BLOCKS` blocks of `BLOCK_SIZE` tiles |
| Grid | full device compute grid (a runtime-arg parameter — core count stays tunable) |
| Per-core work | `split_work_to_cores(R, grid, row_wise=True)` → each core a contiguous run of tile-rows `[row_start, row_start+rows_local)`; per core loops its rows, each row a 2-pass streaming reduce |
| Remainder | `split_work_to_cores` returns two core groups (`rows_local` and `rows_local+1`); handle both explicitly via per-core runtime args (`rows_local`, `row_start`) |

**Alignment-aware tile geometry (per-image `ceil`, never `floor`/`//`):**
`Ht_img = ceil(H/32)`, `R = NC * Ht_img` (per-image ceil — **not** `floor(NC*H/32)`),
`Wt = ceil(W/32)`, `partial_w = W % 32`, `has_partial_w = partial_w != 0`.
Phase-1 exercises tile-aligned shapes but the formulas are `ceil` from the start so the alignment
refinement hits no boundary rewrite.

**Regime-selection function (pinned; regime-pinned tests required).** The program factory selects
the kernel path by two host predicates read off the tensors, not by shape magnitude:

| Predicate | Value → regime |
|---|---|
| `input_tensor.layout` | `TILE_LAYOUT` → tile reader/writer; `ROW_MAJOR_LAYOUT` → tilize-on-read + untilize-on-write |
| `has_partial_w = (W % 32 != 0)` | `True` → partial-scaler reduce (2-tile scaler CB, `ReducePartialScaler::last_tile_at(1)`); `False` → single full scaler |
| `partial_h = (H % 32 != 0)` | RM only: reader pads the last stick-group to 32, writer writes exactly `H` valid sticks; transparent in TILE (padding rows discarded on write-back) |

Because these route to distinct kernel behavior, tests MUST pin each regime: `{TILE, RM} ×
{tile_aligned, w_non_aligned, h_non_aligned}` are all exercised in the acceptance test so a regime
that only triggers on some inputs cannot pass on one device and silently fail on another.

## 7. Circular Buffers

All page counts are functions of `BLOCK_SIZE`/`DEPTH` — **no CB grows with an op dimension**
(`Wt`, `W`, `H`, `R`). `cb_rstd` is 1 tile per row (a reduced scalar, not `Wt`).

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_x_sticks` | 0 | stick (`W·elem`) | `DEPTH * 32` sticks (RM only) | input dtype | reader | compute | RM regime: raw sticks awaiting tilize |
| `cb_x_in` | 1 | tile | `DEPTH * BLOCK_SIZE` | input dtype | reader (TILE) / compute-tilize (RM) | compute | streaming input tiles, both passes |
| `cb_scaler` | 2 | tile | `2 if has_partial_w else 1` | **bfloat16** | reader | compute | `1/W` reduce scaler (+partial), wait-not-pop, spans all rows |
| `cb_xsq` | 24 | tile | `2 * BLOCK_SIZE` | input dtype | compute (square) | compute (reduce) | pass-1 intermediate; sequential helpers → holds a full block |
| `cb_rstd` | 25 | tile | `DEPTH` (≥2) | fp32 (accum) | compute (accumulate_reduce + transform) | compute (mul<Col>) | 1 tile/row `= 1/RMS`; held across pass 2, popped at row end |
| `cb_gamma` | 3 | tile | `DEPTH * BLOCK_SIZE` | gamma dtype | reader (TILE γ) / compute-tilize (RM γ) | compute (mul<Row>) | pass-2 gamma tiles, streamed per block |
| `cb_norm` | 26 | tile | `2 * BLOCK_SIZE` | input dtype | compute (mul<Col>) | compute (mul<Row>) | pass-2 intermediate `x·rstd`; sequential helpers → full block |
| `cb_out` | 16 | tile | `DEPTH * BLOCK_SIZE` | output dtype | compute (mul<Row>, or untilize) | writer | output tiles (TILE) / feeds untilize (RM) |
| `cb_out_sticks` | 17 | stick | `DEPTH * 32` sticks (RM only) | output dtype | compute (untilize) | writer | RM regime: row-major output sticks |

**CB sync (push == wait) & ownership (one producer, one consumer):**
* `cb_xsq`: producer = compute `square`, consumer = compute `reduce`. Sequential compute helpers →
  sized to a **full block** (`≥ BLOCK_SIZE`), else `square` blocks on `reserve_back`.
* `cb_norm`: producer = compute `mul<Col>`, consumer = compute `mul<Row>` — same full-block rule.
* `cb_rstd`: single producer (compute pass-1), single consumer (compute pass-2 `mul<Col>`). It is
  **not** read by any dataflow kernel → one consumer. The pass-1 `accumulate_reduce` accumulator
  and the `transform_in_place` (+eps, rsqrt) both act on it in-compute; the mcast handoff of lamp 2
  would use a **separate** `cb_stat_handoff`, never `cb_rstd` (avoids the two-consumer race).
* `cb_scaler`: reader pushes `2 if has_partial_w else 1`; compute waits the same count and pops once
  at kernel end (reused across every row and both passes).
* In-place `square<cb_xsq,cb_xsq>` is **avoided** (it would make compute both producer & consumer in
  one call, which is legal, but we keep a clean `cb_x_in → square → cb_xsq` so the reused `cb_x_in`
  is not consumed by square). `cb_x_in` is popped by whichever compute op reads it, per pass.

## 8. API Mapping

Every mechanism has a verified `file:line`. Paths under
`ttnn/cpp/ttnn/kernel_lib/` unless noted.

| Phase | Type | Function | File:Line | Template Params / Args (knobs **bold**) | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------------------------|----------|-----------|--------------|
| boot | raw_api | `compute_kernel_hw_startup(icb0,icb1,ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:71` | `(cb_x_in, cb_scaler, cb_out)` | — | — | called once, first stmt of MAIN() before any helper |
| scaler prep | helper | `dataflow_kernel_lib::prepare_reduce_scaler<cb,SUM,REDUCE_ROW>` | `reduce_helpers_dataflow.hpp:58` | `<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1/W)` | — | `cb_scaler` | bf16 CB; pool-type-aware overload (checklist) |
| scaler prep (partial W) | helper | `dataflow_kernel_lib::prepare_partial_reduce_scalers<cb,SUM,REDUCE_ROW,partial_w>` | `reduce_helpers_dataflow.hpp:134` | `<cb_scaler,SUM,REDUCE_ROW,**partial_w**>(1/W)` | — | `cb_scaler` | emits full(tile0)+partial(tile1); pair with `last_tile_at(1)` |
| tilize (RM in) | helper | `compute_kernel_lib::tilize<bw,in,out>` | `tilize_helpers.hpp:187` | `<**BLOCK_SIZE**, cb_x_sticks, cb_x_in>(num_blocks, total_input_pages)` | `cb_x_sticks` | `cb_x_in` | `Fp32Mode::Fast` (default, correct even at max precision) |
| square | helper | `compute_kernel_lib::square<in,out>` | `eltwise_convenience.hpp:115` | `<cb_x_in, cb_xsq>(EltwiseShape::tiles(**BLOCK_SIZE**))` | `cb_x_in` | `cb_xsq` | FPU `x*x` |
| streaming reduce (pass 1) | helper | `compute_kernel_lib::accumulate_reduce<SUM,REDUCE_ROW>` | `streaming_reduce_helpers.hpp:85` | `(cb_xsq, cb_scaler, cb_rstd, ReduceInputBlockShape::of(1,**BLOCK_SIZE**), **NUM_BLOCKS**, partial_scaler)` | `cb_xsq` | `cb_rstd` | scaler `1/W`; partial routed to last block only |
| per-block reduce variant | helper | `compute_kernel_lib::accumulate_reduce_block<SUM,REDUCE_ROW>` | `streaming_reduce_helpers.hpp:53` | (used if per-block work interleaved) | `cb_xsq` | `cb_rstd` | index-aware `Accumulate::at(cb,b)` |
| +eps, rsqrt | helper | `compute_kernel_lib::transform_in_place` | `streaming_reduce_helpers.hpp:111` | `(cb_rstd, [eps](dst){ add eps (SFPU scalar); rsqrt_tile(dst); })` | `cb_rstd` | `cb_rstd` | 1-tile in-DST finalize; bundles reconfig |
| reduce shapes | helper | `ReduceInputBlockShape::of` | `reduce_helpers_compute.hpp:215` | `of(rows=1, cols=**BLOCK_SIZE**, batches=1)` | — | — | rows=1 = per-tile-row |
| partial scaler sel | helper | `ReducePartialScaler::last_tile_at` | `reduce_helpers_compute.hpp:260` | `last_tile_at(1)` when `has_partial_w` else `none()` | — | — | selects tile1 on last W-tile |
| normalize `x·rstd` | helper | `compute_kernel_lib::mul<A,B,Out,Col,…>` | `eltwise_convenience.hpp:93` | `<cb_x_in,cb_rstd,cb_norm, **BroadcastDim::Col**, Streaming, HeldBulk, …, Scalar, Scalar>(EltwiseShape::tiles(**BLOCK_SIZE**))` | `cb_x_in`,`cb_rstd` | `cb_norm` | rstd is REDUCE_ROW (col-shaped) → `Col`; rstd held (`HeldBulk`, popped at row end) |
| scale `·gamma` | helper | `compute_kernel_lib::mul<A,B,Out,Row,…>` | `eltwise_convenience.hpp:93` | `<cb_norm,cb_gamma,cb_out, **BroadcastDim::Row**, …>(EltwiseShape::tiles(**BLOCK_SIZE**))` | `cb_norm`,`cb_gamma` | `cb_out` | gamma is `[1,W]` (row-shaped) → `Row` |
| no-gamma passthrough | helper | `compute_kernel_lib::copy<in,out>` | `eltwise_convenience.hpp:179` | `<cb_norm,cb_out>(EltwiseShape::tiles(**BLOCK_SIZE**))` | `cb_norm` | `cb_out` | when `gamma is None`, skip the `·gamma` mul |
| untilize (RM out) | helper | `compute_kernel_lib::untilize<bw,in,out>` | `untilize_helpers.hpp:154` | `<**BLOCK_SIZE**, cb_out, cb_out_sticks>(num_blocks)` | `cb_out` | `cb_out_sticks` | symmetric tile pages |
| DEST limit | helper | `compute_kernel_lib::DEST_AUTO_LIMIT` | `dest_helpers.hpp:103` | — | — | — | auto-halves to 4 (fp32, half-sync); helpers chunk to it |
| DRAM I/O | raw_api | `TensorAccessor` / `noc_async_read_tile` / `noc_async_write_tile` | `.claude/references/ttnn-cb-memory-fundamentals.md` TensorAccessor pattern | interleaved page addressing | — | — | CT `TensorAccessorArgs` last |
| W-split transport (lamp 2) | helper | `mcast_pipe::SenderPipe::send` / `ReceiverPipe::receive` | `mcast_pipe.hpp:188` / `:265` | rectangle mcast + handshake | — | — | cross-core combine only; not phase-1 |
| sharded CB (lamp 3) | helper | `ttnn.cb_descriptor_from_sharded_tensor` | `ttnn/ttnn/types.py:115` | back `cb_x_in` on a sharded L1 buffer | — | — | HEIGHT_SHARDED zero-copy local read |

**Helpers considered and rejected (raw-API fallbacks):**
* **DRAM tile read/write (raw `TensorAccessor` + `noc_async_read_tile`)** — considered: no
  kernel-lib dataflow helper covers interleaved per-tile DRAM streaming with the custom
  `tile_id = row_base + b*BLOCK_SIZE + wt` order this op needs (two passes, per-core row offset).
  `tilize_helpers_dataflow.hpp` covers *stick* reads for tilize but not the generic tiled stream.
  Concrete reason: the reader must emit tiles in the streaming-reduce's expected order across two
  passes with a per-core `row_start` offset — no helper parameterizes that. Raw accessor is the
  documented pattern (`ttnn-cb-memory-fundamentals.md` → TensorAccessor Pattern).
* **`+eps` then `rsqrt` (via `transform_in_place` lambda using raw `rsqrt_tile` / scalar add)** —
  considered `unary<Rsqrt<>,…>`: rejected because eps must be added *before* rsqrt on the same DST
  tile and eps is a runtime scalar; `transform_in_place` (`streaming_reduce_helpers.hpp:111`) is the
  sanctioned 1-tile in-DST finalize hook for exactly "a chain like add_unary_tile, rsqrt_tile"
  (its doc, line 104-105). The lambda's inner ops are raw SFPU calls by design of that helper.

Every other compute phase uses a helper.

## 9. Compute Phases (per tile-row; sequential within a core)

| # | Operation | Helper? | Input CB (tiles, state) | Output CB (tiles) | CB State After |
|---|-----------|---------|-------------------------|-------------------|----------------|
| 0 | (RM only) tilize x block | `tilize` | `cb_x_sticks` (32 sticks/block) | `cb_x_in` (`BLOCK_SIZE`) | `cb_x_in` filled |
| 1 | square `x²` | `square` | `cb_x_in` (`BLOCK_SIZE`, streamed/popped) | `cb_xsq` (`BLOCK_SIZE`) | pass-1 consumes `cb_x_in` |
| 2 | streaming `Σx²·(1/W)` over `NUM_BLOCKS` | `accumulate_reduce<SUM,REDUCE_ROW>` | `cb_xsq` (block-by-block) | `cb_rstd` (1 tile) | `cb_rstd` = mean(x²); partial scaler zeros padded W |
| 3 | `+eps`, `rsqrt` | `transform_in_place` | `cb_rstd` (1 tile) | `cb_rstd` (1 tile) | `cb_rstd` = `1/RMS`, **held for pass 2** |
| 4 | (RM only) re-tilize x block (pass 2) | `tilize` | `cb_x_sticks` | `cb_x_in` | reader re-streams x |
| 5 | `x·rstd` (bcast Col) | `mul<…,Col,…>` | `cb_x_in`(`BLOCK_SIZE`), `cb_rstd`(1, held) | `cb_norm` (`BLOCK_SIZE`) | `cb_norm` = normalized (pre-gamma) |
| 6a | `·gamma` (bcast Row) — gamma present | `mul<…,Row,…>` | `cb_norm`, `cb_gamma`(`BLOCK_SIZE`) | `cb_out` (`BLOCK_SIZE`) | `cb_out` ready |
| 6b | passthrough — no gamma | `copy` | `cb_norm` | `cb_out` | `cb_out` ready |
| 7 | (RM only) untilize out block | `untilize` | `cb_out` (`BLOCK_SIZE`) | `cb_out_sticks` | writer writes sticks |
| — | row end | raw | pop `cb_rstd` (1) | — | `cb_rstd` freed for next row |
| — | kernel end | raw | pop `cb_scaler` (`2 if has_partial_w else 1`) | — | — |

**Reduce-direction note.** Single reduce dim only (`W`, `REDUCE_ROW`). The REDUCE_ROW result is
column-shaped (`[R,1]`), broadcast back across columns in phase 5 via `BroadcastDim::Col`. (No
multi-dim reduce table — the op reduces one axis.)

## 10. Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| 5 `x·rstd` | mul | `cb_x_in` — All `[32,32]` | `cb_rstd` — **Col0** (REDUCE_ROW result, per-row scalar) | **Col** |
| 6a `·gamma` | mul | `cb_norm` — All `[32,32]` | `cb_gamma` — **Row0** (`[1,W]` per-feature vector) | **Row** |

Rule check: REDUCE_ROW output → broadcast across columns → `BroadcastDim::Col` (matches
`eltwise_chain.hpp:526-533` — "a REDUCE_ROW result is column-shaped and broadcasts back across
columns via BroadcastDim::Col"). A `[1,W]` operand → broadcast across rows → `BroadcastDim::Row`.

## 11. Key Risks and Gotchas

* **CBs that MUST hold a full block:** `cb_xsq` (square→reduce) and `cb_norm` (mul→mul) sit between
  two sequential compute helpers (all 3 TRISCs, no overlap) → size `≥ BLOCK_SIZE` (we use
  `2*BLOCK_SIZE`). Undersizing hangs on `reserve_back`.
* **`cb_scaler` must be bfloat16 and packed via the pool-type-aware overload**
  (`prepare_reduce_scaler<cb,SUM,REDUCE_ROW>` / `prepare_partial_reduce_scalers<…,partial_w>`) —
  SUM+REDUCE_ROW has a specific row-0 fill pattern; the legacy single-arg form is wrong.
* **Mean divides by the TRUE element count** `origin_W`, not `Wt*32`: `scaler = 1/origin_W`, and the
  partial-scaler tile zeros the `32 - partial_w` padded lanes of the last W-tile so padding
  contributes `x²·0 = 0`. Padded values must be **finite** (randn/zero-pad are) — inf/nan padding
  would give `inf*0 = nan`; note in docs, not a phase-1 concern for the test distribution.
* **`cb_rstd` persists across pass 2** (held, `mul<…,HeldBulk,…>` on the B operand or a
  wait-not-pop), then popped once per row. It is 1 tile — never `Wt`. Two-consumer trap: the
  lamp-2 mcast handoff uses a **separate** `cb_stat_handoff`, never `cb_rstd`.
* **fp32 DEST budget:** `fp32_dest_acc_en=True` halves DEST to 4 (half-sync). Helpers read
  `DEST_AUTO_LIMIT` and chunk to it, so `BLOCK_SIZE > 4` is safe (extra inner iterations, no
  overflow). Pass `config=compute_kernel_config` unchanged into `ComputeConfigDescriptor`.
* **`float32 + fp32_dest_acc_en=False` is rejected** (EXCLUSION, xfail-strict): f32 with non-fp32
  accumulation is lossy/nonsensical. `validate()` reads `compute_kernel_config.fp32_dest_acc_en`
  (resolved through `default_compute_kernel_config()`) and refuses the combination. Not INVALID —
  it is legal-but-refused. (`references/precision_convention.md`.)
* **Output layout must match input** (RM→RM via untilize, TILE→TILE). The entry point must **not**
  call `to_layout`/`tilize`/`untilize`/`pad`/`slice` on host — the kernel does the RM↔tile
  conversion natively.
* **H non-alignment is transparent in TILE** (padding tile-rows produce garbage that lands in the
  output tensor's padding region, discarded on read-back). In **RM** the reader pads the last
  stick-group to 32 and the writer writes exactly `origin_H` sticks.
* **Under-fill:** shapes with `R < #cores` (decode `R=1`) only occupy a few cores under
  row-parallel — correct but slow. That is the trigger for lamp 2 (W-split); do not "fix" it by
  collapsing `BLOCK_SIZE` to 1.

## 12. Structural impossibilities (op-specific candidates for `feature_spec.py` INVALID)

`feature_spec.py` already encodes the needed INVALID cells (bf8b+RM on activation and gamma; the
`no_gamma`↔`"none"` sentinel coupling; RM+sharded+tiled-gamma author-scoped exclusions; bf8b on
non-aligned). No additional op-specific structural impossibility is identified — do not edit
`feature_spec.py`.

## 13. Support-axis mapping (for the implementer's `SUPPORTED` / `validate`; not authored here)

INPUT_TAGGERS the op must export (feature_spec references them):
* `tag_alignment(inputs, axes)` → `"tile_aligned"` (both H,W %32==0) / `"w_non_aligned"` (W%32≠0) /
  `"h_non_aligned"` (W aligned, H%32≠0).
* `tag_rank(inputs, axes)` → `len(inputs[0])`.
* `tag_gamma_dtype` / `tag_gamma_layout` → real dtype/layout when gamma present, else `"none"`.

Precision axes (`dtype`, `fp32_dest_acc_en`) gate per `precision_convention.md`: maxed corner
(`fp32_dest_acc_en=True`) first; `{float32, False}` → EXCLUSION. `gamma_dtype`/`gamma_layout` must
accept `"none"` as always-legal. `memory_layout` SUPPORTED starts at `[INTERLEAVED]`; the sharded
values are the §1 lamps promoted refinement-by-refinement. Index/dim axes: n/a (rms_norm always
reduces the last dim).
