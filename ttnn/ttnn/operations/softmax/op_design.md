# Operation Design: softmax

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (SFPU + FPU reduce + broadcast) |
| Goal | Numerically-stable row-wise (`dim=-1`) or column-wise (`dim=-2`) softmax for fp32 TILE-layout tensors, returning the same shape/dtype/layout. |
| Math | `output[..., i, j] = exp(x[..., i, j] − M) / Σ_k exp(x[..., i, k] − M)` where M is `max(x, dim)` if `numeric_stable=True`, else `M = 0` (the subtraction is skipped). |
| Mode | Derivative (mirrors the existing `ttnn.softmax` numeric-stable pipeline: MAX → sub+exp → SUM → mul-by-recip) |
| References | `build_Release/libexec/tt-metalium/ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention/compute/softmax.cpp` (reference pipeline, lines 22–61 and 277–315), `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`, `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp`, `eval/golden_tests/softmax/feature_spec.py` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | float32, TILE_LAYOUT, rank-4, H%32=0, W%32=0, on device | — | input |
| `dim` | `int` | no | `-1` or `-2` (any other value rejected by `validate()`) | `-1` | CT (selects compute kernel reduce-direction and broadcast-direction template specialization) |
| `numeric_stable` | `bool` | no (keyword-only in the public signature; we treat dim/numeric_stable as positional-or-keyword) | `True` / `False` | `True` | CT (selects between the 4-phase numeric-stable pipeline and the 2-phase fast path) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | no, keyword-only | Either `None` (entry point installs the Phase-0 default), or an explicit descriptor with `math_fidelity=ttnn.MathFidelity.HiFi4` and `fp32_dest_acc_en=True`. Other configs rejected. | `None` → `ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)` | RT-from-Python (handed straight to the compute `KernelDescriptor`) |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(N, C, H, W)` — rank exactly 4 |
| Dtype | `float32` |
| Layout | `TILE_LAYOUT` |
| Memory | interleaved (DRAM or L1) |
| Alignment | `H % 32 == 0`, `W % 32 == 0` |

### Output

| Property | Value |
|----------|-------|
| Shape | `(N, C, H, W)` — identical to input |
| Dtype | `float32` |
| Layout | `TILE_LAYOUT` |
| Memory | default `DRAM_MEMORY_CONFIG` (interleaved) |

## Dataflow Strategy

| Stage | Where | Format | Notes |
|-------|-------|--------|-------|
| Source | DRAM (or L1) interleaved | fp32 tiles | Tile index `tid = nc*Ht*Wt + ht*Wt + wt`. |
| Reader (NCRISC) | DRAM → `cb_input_tiles` | fp32 tiles | For each work-item (one reduce-strip), reads `reduce_dim_tiles` tiles from DRAM via `TensorAccessor`. Reader is also responsible for emitting one scaler tile into each of `cb_max_scaler` and `cb_sum_scaler` ONCE per core at startup via `calculate_and_prepare_reduce_scaler<...>()`. |
| Compute (TRISCs) | `cb_input_tiles` + scaler CBs → intermediates → `cb_output_tiles` | fp32 tiles | Four sequential helper calls per work-item (see Compute Phases). Each helper owns all 3 TRISCs for its duration. |
| Writer (BRISC) | `cb_output_tiles` → DRAM | fp32 tiles | For each work-item, writes `reduce_dim_tiles` tiles back to the corresponding tile indices. |
| Sink | DRAM (or L1) interleaved | fp32 tiles | Output tile indices mirror input tile indices (shape unchanged). |

**Single-Tensix; no inter-Tensix communication.** Each Tensix processes its assigned work-items end-to-end. There are no semaphores, no multicast, and no ring topology. Cores receive disjoint slices of the work-item set via runtime args and proceed independently.

**Reduce-strip definition (one work-item):**

| `dim` | Strip shape (tiles) | Tile index pattern | `reduce_dim_tiles` | Output strip |
|-------|---------------------|--------------------|---------------------|--------------|
| `-1` (REDUCE_ROW) | `1 × Wt` (one tile-row) | `nc*Ht*Wt + ht*Wt + wt`, wt = 0..Wt-1 | `Wt` | same `1 × Wt` strip back to the same indices |
| `-2` (REDUCE_COL) | `Ht × 1` (one tile-column) | `nc*Ht*Wt + ht*Wt + wt`, ht = 0..Ht-1 (wt fixed) | `Ht` | same `Ht × 1` strip back to the same indices |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | One reduce-strip: `1 × Wt` tiles for `dim=-1`, `Ht × 1` tiles for `dim=-2`. |
| Total work items | `dim=-1`: `num_strips = N * C * Ht`. `dim=-2`: `num_strips = N * C * Wt`. |
| Grid | `device.compute_with_storage_grid_size()` (full available Tensix grid; on Wormhole typically 8×8). |
| Per-core formula | `ttnn.split_work_to_cores(num_strips, grid)` → returns `(num_cores_total, all_cores, core_group_1, core_group_2, num_per_core_group_1, num_per_core_group_2)`. Group 1 cores get `num_per_core_group_1` strips, group 2 gets `num_per_core_group_2`. Each core gets a contiguous `[strip_start, strip_start + per_core)` runtime range. |
| Remainder | Handled by `split_work_to_cores` (two-group split). Cores that get zero strips are excluded from `all_cores`. |
| Inter-core | None. |

## Circular Buffers

`reduce_dim_tiles` = `Wt` when `dim=-1`, `Ht` when `dim=-2`. `input_tile_size = ttnn.tile_size(ttnn.float32)` = 4096 bytes. `scaler_tile_size = ttnn.tile_size(ttnn.bfloat16)` = 2048 bytes.

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_tiles` | 0 | `input_tile_size` (fp32 tile) | `2 * reduce_dim_tiles` | fp32 | Reader | Compute (Phase A waits upfront, Phase B pops at end) | Refilled once per work-item. Double-block sized to pipeline the next strip's reader fill against the current strip's compute. |
| `cb_max_scaler` | 1 | `scaler_tile_size` (bf16 tile) | `1` | bf16 | Reader (one-shot at boot) | Phase A reduce (`WaitUpfrontNoPop` — never popped) | Persistent for entire core lifetime. |
| `cb_sum_scaler` | 2 | `scaler_tile_size` (bf16 tile) | `1` | bf16 | Reader (one-shot at boot) | Phase C reduce (`WaitUpfrontNoPop` — never popped) | Persistent for entire core lifetime. |
| `cb_output_tiles` | 16 | `input_tile_size` (fp32 tile) | `2 * reduce_dim_tiles` | fp32 | Compute Phase D | Writer | Refilled once per work-item. Double-block sized so the writer can drain strip N-1 while Phase D produces strip N. |
| `cb_max` | 24 | `input_tile_size` (fp32 tile) | `2` | fp32 | Phase A reduce | Phase B sub (pops at end) | Single-tile intermediate per work-item. 2 pages so the producer for strip N+1 can fill while the consumer drains strip N. |
| `cb_exps` | 25 | `input_tile_size` (fp32 tile) | `reduce_dim_tiles` | fp32 | Phase B sub+exp postop | Phase C reduce (`WaitUpfrontNoPop`), Phase D mul (`WaitUpfrontPopAtEnd`) | Sized to a single full strip. Phases B→C→D are sequential within compute (all own TRISCs), so the CB only needs to hold one work-item's worth — but ALL `reduce_dim_tiles` tiles must be resident simultaneously because Phase C waits upfront and Phase D reuses the same Wt/Ht tiles via broadcast. |
| `cb_inv_sum` | 26 | `input_tile_size` (fp32 tile) | `2` | fp32 | Phase C reduce + recip postop | Phase D mul (pops at end) | Single-tile intermediate per work-item. 2 pages for the same producer/consumer overlap rationale as `cb_max`. |

**CB sync verification (per work-item, after one-shot scaler setup):**

| CB | Reader pushes | Compute waits | Compute pops | Compute pushes | Writer pops | Net |
|----|---------------|---------------|--------------|----------------|-------------|-----|
| `cb_input_tiles` | `reduce_dim_tiles` | `reduce_dim_tiles` (Phase A upfront, Phase B reuses) | `reduce_dim_tiles` (Phase B pop-at-end) | — | — | balanced |
| `cb_max_scaler` | 1 (boot only) | 1 (Phase A, NoPop) | 0 | — | — | balanced (persistent) |
| `cb_sum_scaler` | 1 (boot only) | 1 (Phase C, NoPop) | 0 | — | — | balanced (persistent) |
| `cb_max` | — | 1 (Phase B) | 1 | 1 (Phase A) | — | balanced |
| `cb_exps` | — | `reduce_dim_tiles` (Phase C upfront, Phase D reuses) | `reduce_dim_tiles` (Phase D pop-at-end) | `reduce_dim_tiles` (Phase B) | — | balanced |
| `cb_inv_sum` | — | 1 (Phase D) | 1 | 1 (Phase C) | — | balanced |
| `cb_output_tiles` | — | — | — | `reduce_dim_tiles` (Phase D) | `reduce_dim_tiles` | balanced |

**`cb_exps` sizing rule.** Phases B (producer) and C (consumer with `WaitUpfrontNoPop`) are sequential helpers — they cannot pipeline because each helper owns all 3 TRISCs (see `.claude/references/ttnn-cb-memory-fundamentals.md` "Intermediate CB Sizing Between Compute Helpers"). Therefore `cb_exps` MUST hold a full strip's worth of tiles (`reduce_dim_tiles`); a smaller CB would deadlock with Phase B blocked on `cb_reserve_back` and Phase C never starting.

**`fp32_dest_acc_en=True` DEST budget**: half-sync + fp32 → **4 tiles**. `DEST_AUTO_LIMIT` resolves to 4 automatically (`dest_helpers.hpp:97`). All helpers respect this internally.

## API Mapping

| Phase | Type | Function | File:Line | Template Params / Args | Input CB (semantic name) | Output CB (semantic name) | Requirements |
|-------|------|----------|-----------|------------------------|--------------------------|---------------------------|--------------|
| Boot | helper | `compute_kernel_hw_startup(cb_input_tiles, cb_max_scaler, cb_output_tiles)` | `ttnn/cpp/ttnn/kernel_lib/cb_helpers_compute.hpp` (declared) | — | — | — | Called exactly once at the start of `kernel_main`. Helper prerequisites in `reduce_helpers_compute.hpp:30–33`, `binary_op_helpers.hpp:17–19`. |
| Boot (reader) | helper | `calculate_and_prepare_reduce_scaler<cb_max_scaler, PoolType::MAX, ReduceDim::REDUCE_ROW>()` (when `dim=-1`) or `<..., REDUCE_COL>` (when `dim=-2`) | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:94–101` | `cb_id=cb_max_scaler`, `pool_type=MAX`, `reduce_dim=REDUCE_ROW` or `REDUCE_COL`, `reduce_factor=1` (MAX) | — | `cb_max_scaler` | Pool-type/reduce-dim-aware overload — different combos use different tile fill patterns (`reduce_helpers_dataflow.hpp:46–48`). Scaler value = 1.0 for MAX. Tile filled once at boot; never popped. |
| Boot (reader) | helper | `calculate_and_prepare_reduce_scaler<cb_sum_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>()` or `<..., REDUCE_COL>` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:94–101` | `pool_type=SUM`, `reduce_dim` matches phase C, `reduce_factor=1` | — | `cb_sum_scaler` | Pool-type/reduce-dim-aware overload. Scaler = 1.0 for SUM. One-shot at boot. |
| Phase A | helper | `reduce<PoolType::MAX, ReduceDim::REDUCE_ROW \| REDUCE_COL, ReduceInputPolicy::WaitUpfrontNoPop, ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(cb_input_tiles, cb_max_scaler, cb_max, shape)` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:400–416` | `reduce_type=MAX`, `reduce_dim=REDUCE_ROW` (dim=-1) or `REDUCE_COL` (dim=-2), `input_policy=WaitUpfrontNoPop` (tiles persist for Phase B); shape = `ReduceInputBlockShape::of(1, Wt)` (dim=-1) or `of(Ht, 1)` (dim=-2) | `cb_input_tiles` (Wt or Ht), `cb_max_scaler` | `cb_max` (1 tile pushed) | Manages its own `cb_wait_front`/`cb_reserve_back`/`cb_push_back` (does NOT pop input — that's the WaitUpfrontNoPop contract; Phase B pops). Pattern matches reference softmax (`softmax.cpp:29–34`). |
| Phase B | helper | `sub<BroadcastDim::COL \| ROW, BinaryInputPolicy::WaitUpfrontPopAtEnd, BinaryInputPolicy::WaitUpfrontPopAtEnd, BinaryOutputPolicy::PerTile, BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(cb_input_tiles, cb_max, cb_exps, shape, post_op_lambda)` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:283–293` (declaration); composition seam at `binary_op_helpers.hpp:248–250` ("`post_op` lambda receives `dst_idx` and runs inside the dst-sync window before pack") | `bcast_dim=COL` (dim=-1) or `ROW` (dim=-2); `input_a_policy=WaitUpfrontPopAtEnd` (pops cb_input_tiles after Phase A's WaitUpfrontNoPop); `input_b_policy=WaitUpfrontPopAtEnd` (pops cb_max); shape = `BinaryInputBlockShape::of(1, Wt)` or `of(Ht, 1)`; post-op = `[](uint32_t d){ exp_tile_init(); exp_tile(d); }` | `cb_input_tiles` (Wt or Ht), `cb_max` (1) | `cb_exps` (Wt or Ht tiles pushed) | Composition seam — the postop fuses `exp` onto each subtracted DEST register inside the same dst-sync window, no extra CB hop. Matches reference softmax's `calc_numeric_stable` fused sub+exp loop (`softmax.cpp:41–57`). `cb_input_tiles` must hold ≥ `reduce_dim_tiles` (helper asserts via `assert_binary_input_cb_size`, `binary_op_helpers.inl:62–70`). |
| Phase C | helper | `reduce<PoolType::SUM, ReduceDim::REDUCE_ROW \| REDUCE_COL, ReduceInputPolicy::WaitUpfrontNoPop, ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(cb_exps, cb_sum_scaler, cb_inv_sum, shape, mem_layout, NoAccumulation{}, recip_post_op)` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:400–416`; recip-postop pattern documented at `reduce_helpers_compute.hpp:377–385` | `reduce_type=SUM`, `reduce_dim` matches Phase A, `input_policy=WaitUpfrontNoPop` (tiles persist for Phase D); shape = `ReduceInputBlockShape::row(Wt)` (dim=-1) or `::col(Ht)` (dim=-2); post-op = `[](uint32_t d){ recip_tile_init(); recip_tile(d); }` | `cb_exps` (Wt or Ht), `cb_sum_scaler` | `cb_inv_sum` (1 tile pushed) | Direct lift from the reference softmax `reduce<SUM,REDUCE_ROW,WaitUpfrontNoPop>` + recip postop pattern (`softmax.cpp:278–289`). The postop produces `1/Σexp` so Phase D can multiply instead of divide. |
| Phase D | helper | `mul<BroadcastDim::COL \| ROW, BinaryInputPolicy::WaitUpfrontPopAtEnd, BinaryInputPolicy::WaitUpfrontPopAtEnd>(cb_exps, cb_inv_sum, cb_output_tiles, shape)` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:296–306` | `bcast_dim` matches Phase B; `input_a_policy=WaitUpfrontPopAtEnd` (pops cb_exps after Phase C's WaitUpfrontNoPop); `input_b_policy=WaitUpfrontPopAtEnd` (pops cb_inv_sum); shape matches Phase B | `cb_exps` (Wt or Ht), `cb_inv_sum` (1) | `cb_output_tiles` (Wt or Ht tiles pushed) | Mirror of reference softmax's `mul_bcast_cols` loop (`softmax.cpp:298–313`). Helper owns `wait_front`/`pop_front`/`reserve_back`/`push_back`. |
| Reader | raw_api | `TensorAccessor` for input | `tech_reports/tensor_accessor/tensor_accessor.md` | constructed from `TensorAccessorArgs<idx>()` CT arg; `get_noc_addr(tile_id)` | DRAM (input buffer) | `cb_input_tiles` | Per-strip loop: `reserve_back(reduce_dim_tiles)` → `reduce_dim_tiles` × `noc_async_read` → `noc_async_read_barrier` → `push_back(reduce_dim_tiles)`. Helpers considered and rejected: `cb_helpers_dataflow` provides `cb_push_*` utilities but they wrap the same primitives — no dedicated "read N tile-strip into CB" helper exists. Verified by `grep -n "noc_async_read\|tile_strip" ttnn/cpp/ttnn/kernel_lib/cb_helpers_dataflow.hpp` returns nothing wrapping a per-strip read sequence. |
| Writer | raw_api | `TensorAccessor` for output | `tech_reports/tensor_accessor/tensor_accessor.md` | constructed from `TensorAccessorArgs<idx>()` CT arg; `get_noc_addr(tile_id)` | `cb_output_tiles` | DRAM (output buffer) | Per-strip loop: `wait_front(reduce_dim_tiles)` → `reduce_dim_tiles` × `noc_async_write` → `noc_async_write_barrier` → `pop_front(reduce_dim_tiles)`. Same rejection rationale as reader. |

**Numeric-stable=False path:** Phases A and B collapse to a single helper:

| Phase (unstable path only) | Type | Function | File:Line | Template Params | Input CB | Output CB | Requirements |
|-----|------|----------|-----------|------------------|----------|-----------|--------------|
| Phase B′ (replaces A+B when `numeric_stable=False`) | helper | `sfpu_op<cb_input_tiles, SfpuBatching::Auto, SfpuInputPolicy::WaitAndPopPerTile, SfpuOutputPolicy::PerTile, SfpuDataFormatReconfig::INPUT_AND_OUTPUT>(cb_exps, reduce_dim_tiles, Exp<Approx::Exact, Approx::Fast>{})` (or the named alias `sfpu_exp<cb_input_tiles>(cb_exps, reduce_dim_tiles)`) | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:1438–1466` | ICB=`cb_input_tiles`; default batching auto-fills DEST | `cb_input_tiles` (Wt or Ht) | `cb_exps` (Wt or Ht) | Streams input tiles, applies `exp` on SFPU, packs into `cb_exps`. Phases C and D are unchanged. `cb_max`, `cb_max_scaler` are not used in this path (allocated but idle; harmless). |

## Compute Phases

`reduce_dim_tiles` = `Wt` (dim=-1) or `Ht` (dim=-2). All phases below execute per-work-item, wrapped in `for (uint32_t s = 0; s < num_strips_for_this_core; ++s) { ... }`.

### Numeric-stable = True (default)

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | MAX reduce | `reduce<MAX, REDUCE_ROW\|REDUCE_COL, WaitUpfrontNoPop>` | `cb_input_tiles` (`reduce_dim_tiles`, waited upfront, NOT popped); `cb_max_scaler` (1, persistent, NoPop) | `cb_max` (1 tile) | `cb_input_tiles` still holds the strip (Phase B will consume); `cb_max` holds 1 tile; scalers untouched. |
| 2 | sub + exp postop | `sub<COL\|ROW, WaitUpfrontPopAtEnd, WaitUpfrontPopAtEnd>` with `[](d){ exp_tile_init(); exp_tile(d); }` | `cb_input_tiles` (`reduce_dim_tiles`, popped at end); `cb_max` (1, popped at end) | `cb_exps` (`reduce_dim_tiles`) | `cb_input_tiles` drained; `cb_max` drained; `cb_exps` holds `reduce_dim_tiles` tiles of `exp(x − max)`. |
| 3 | SUM reduce + recip postop | `reduce<SUM, REDUCE_ROW\|REDUCE_COL, WaitUpfrontNoPop>` with `[](d){ recip_tile_init(); recip_tile(d); }` | `cb_exps` (`reduce_dim_tiles`, waited upfront, NOT popped); `cb_sum_scaler` (1, persistent, NoPop) | `cb_inv_sum` (1 tile) | `cb_exps` still holds the strip (Phase D will consume); `cb_inv_sum` holds `1/Σexp`; scalers untouched. |
| 4 | mul by `1/Σ` (broadcast) | `mul<COL\|ROW, WaitUpfrontPopAtEnd, WaitUpfrontPopAtEnd>` | `cb_exps` (`reduce_dim_tiles`, popped at end); `cb_inv_sum` (1, popped at end) | `cb_output_tiles` (`reduce_dim_tiles`) | All intermediates drained; `cb_output_tiles` holds the final softmax strip ready for the writer. |

### Numeric-stable = False (skip subtraction)

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | exp on input | `sfpu_op<cb_input_tiles>(..., Exp{})` (or alias `sfpu_exp<cb_input_tiles>`) | `cb_input_tiles` (`reduce_dim_tiles`, WaitAndPopPerTile) | `cb_exps` (`reduce_dim_tiles`) | `cb_input_tiles` drained tile-by-tile; `cb_exps` holds `exp(x)` tiles. |
| 2 | SUM reduce + recip postop | identical to Phase 3 above | `cb_exps` (WaitUpfrontNoPop), `cb_sum_scaler` | `cb_inv_sum` (1) | `cb_exps` still held; `cb_inv_sum` holds `1/Σexp`. |
| 3 | mul by `1/Σ` | identical to Phase 4 above | `cb_exps` (WaitUpfrontPopAtEnd), `cb_inv_sum` (WaitUpfrontPopAtEnd) | `cb_output_tiles` (`reduce_dim_tiles`) | Output drained; ready for writer. |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| Phase B (dim=-1) | `sub` | `cb_input_tiles`: All (`1 × Wt`, every tile fully valid) | `cb_max`: Col0 of the single tile (REDUCE_ROW MAX produces column-vector output) | `BroadcastDim::COL` — B has Ht=1 tile, replicated across columns of each input tile |
| Phase B (dim=-2) | `sub` | `cb_input_tiles`: All (`Ht × 1`) | `cb_max`: Row0 of the single tile (REDUCE_COL MAX produces row-vector output) | `BroadcastDim::ROW` — B has Wt=1 tile, replicated across rows of each input tile |
| Phase D (dim=-1) | `mul` | `cb_exps`: All (`1 × Wt`) | `cb_inv_sum`: Col0 (REDUCE_ROW SUM + recip produces column-vector output) | `BroadcastDim::COL` |
| Phase D (dim=-2) | `mul` | `cb_exps`: All (`Ht × 1`) | `cb_inv_sum`: Row0 (REDUCE_COL SUM + recip produces row-vector output) | `BroadcastDim::ROW` |

Valid-region key (per `op-design-template.md:102–103`): REDUCE_ROW output → Col0; REDUCE_COL output → Row0. The post-reduce broadcast direction is the conjugate of the reduce direction (`binary_op_helpers.hpp:37–43`): REDUCE_ROW → COL broadcast, REDUCE_COL → ROW broadcast.

## Reduce Direction Verification

| Logical Dim | Tile ReduceDim | Output Valid Region | BroadcastDim | ReduceInputBlockShape | BinaryInputBlockShape |
|-------------|----------------|---------------------|--------------|-----------------------|-----------------------|
| `dim = -1` | `ReduceDim::REDUCE_ROW` | Col0 of output tile (column-vector) | `BroadcastDim::COL` | `ReduceInputBlockShape::of(1, Wt)` (rows=1 tile-row, cols=Wt tile-cols, batches=1) | `BinaryInputBlockShape::of(1, Wt)` |
| `dim = -2` | `ReduceDim::REDUCE_COL` | Row0 of output tile (row-vector) | `BroadcastDim::ROW` | `ReduceInputBlockShape::of(Ht, 1)` | `BinaryInputBlockShape::of(Ht, 1)` |

## Hardware Constraints

- [x] CB sync: push count = wait count for every CB (see CB sync table above)
- [x] Reduce scaler CB is bfloat16 (`cb_max_scaler` and `cb_sum_scaler` are bf16 even though input is fp32)
- [x] Reduce scaler uses pool-type-aware API (`calculate_and_prepare_reduce_scaler<cb, PoolType, ReduceDim>` — NOT the legacy single-template-arg overload)
- [x] DEST budget: `fp32_dest_acc_en=True` → half-sync limit = 4 tiles. `DEST_AUTO_LIMIT` handles this automatically inside helpers. The plan never names a specific DEST tile count — all helpers (`reduce`, `binary_op`/`sub`/`mul`, `sfpu_op`) auto-batch against `DEST_AUTO_LIMIT`.
- [x] Sequential helper intermediate `cb_exps` sized to a full strip (`reduce_dim_tiles`), see "`cb_exps` sizing rule" above. `cb_input_tiles` similarly sized (`2 × reduce_dim_tiles`) because Phase A waits upfront for the full strip.
- [x] Page sizes aligned: fp32 tile pages use `ttnn.tile_size(ttnn.float32)` (4096 B); scaler pages use `ttnn.tile_size(ttnn.bfloat16)` (2048 B).
- [x] All `cb_wait_front` calls on the same CB use the same page count: helpers compute `total_tiles` from the `ReduceInputBlockShape` / `BinaryInputBlockShape` and assert via `assert_binary_input_cb_size` (`binary_op_helpers.inl:62–70`).
- [x] Helpers are not wrapped with extra CB operations (Phases A–D delegate all CB management to the helpers; the reader/writer handle their own CB ops only for `cb_input_tiles`/`cb_output_tiles` and the one-shot scaler fills).
- [x] Every compute phase uses a helper (Phase A: `reduce`; Phase B: `sub` with exp postop; Phase C: `reduce` with recip postop; Phase D: `mul`; unstable Phase B′: `sfpu_op`/`sfpu_exp`). No raw compute APIs needed except inside the `post_op` lambdas, which is the documented composition seam (`binary_op_helpers.hpp:248–250`, `reduce_helpers_compute.hpp:309–313`).
- [x] `compute_kernel_hw_startup()` called exactly once at the start of the compute kernel, before any helper invocation.

## Key Risks and Gotchas

| Risk | Why it matters | Mitigation |
|------|----------------|------------|
| `cb_exps` must be sized to a full strip | Sequential helpers (Phase B producer → Phase C consumer with WaitUpfrontNoPop) cannot pipeline because each owns all 3 TRISCs. An undersized `cb_exps` deadlocks Phase B on `cb_reserve_back`. | Allocate `cb_exps` with `num_pages = reduce_dim_tiles` (not double-buffered). |
| `cb_input_tiles` must hold a full strip | Phase A's `WaitUpfrontNoPop` requires the entire strip resident before reduce starts. Same hazard as `cb_exps`. | Allocate `num_pages = 2 * reduce_dim_tiles` (double-block enables reader pipelining against compute). |
| Scaler CBs are bfloat16 even for fp32 input | The LLK reduce uses the scaler tile as SrcB which has a fixed format. Mismatched format produces silently wrong results. | `cb_max_scaler` and `cb_sum_scaler` are bf16; `calculate_and_prepare_reduce_scaler` deduces format from the CB and writes the correct representation. |
| Pool-type-aware scaler API is mandatory | The non-pool-type-aware overload uses a default fill pattern that is wrong for MAX (which needs row-0 fill) or for SUM REDUCE_ROW (which needs col-0 fill / matmul layout). | Both scaler fills go through the `<cb, PoolType, ReduceDim>` overload. |
| Phase A uses WaitUpfrontNoPop on `cb_input_tiles`, Phase B uses WaitUpfrontPopAtEnd on the same CB | The double-wait is intentional: Phase A leaves data in the CB for Phase B. Phase B's `wait_front(reduce_dim_tiles)` is idempotent — data is already there. The pop happens at the end of Phase B. | This is the documented "softmax pattern" in both `reduce_helpers_compute.hpp:366–369` and `binary_op_helpers.hpp:65–70`. No special handling needed. |
| `fp32_dest_acc_en=True` halves DEST capacity (8 → 4 in half-sync) | Hand-rolled `tile_regs_acquire`/loops sized for bf16 would overflow DEST. | All helpers respect `DEST_AUTO_LIMIT` (`dest_helpers.hpp:88–99`). The design never hand-codes a DEST loop. |
| `dim=-2` reduce direction is the "transpose" of `dim=-1` | Mixing up `REDUCE_ROW`/`REDUCE_COL` or `BroadcastDim::ROW`/`COL` produces wrong output silently. | Both directions encoded in the Reduce Direction Verification table. The program factory selects the compile-time template specialization based on the validated `dim` value. |
| `numeric_stable=False` path leaves `cb_max` / `cb_max_scaler` unused | Allocating them is harmless (a few extra bytes of L1) but the program factory must still set them up so the same kernel binary handles both paths — or the kernel branches and the factory only allocates what's needed. | Simpler: allocate both scaler CBs and `cb_max` unconditionally; the compute kernel branches on a CT arg `numeric_stable` to choose Phase A+B vs Phase B′. |
| Validation must reject everything outside the Phase 0 envelope | `compute_kernel_config` is the easy one to get wrong: `None` is accepted (entry point fills the default), and an explicit `ComputeConfigDescriptor` with the exact Phase 0 fields is accepted; everything else must raise. | The Python entry point fills the default before `validate()`; `validate()` reads `compute_kernel_config.math_fidelity` and `.fp32_dest_acc_en` and raises `NotImplementedError` if they don't match the Phase 0 pair. Rank, layout, dtype, H/W alignment, and `dim ∈ {-1, -2}` are checked in the same function before construction proceeds. |

## Structural Impossibilities (notes for the golden-tests skill)

`feature_spec.py` (already authored in pipeline mode) declares `INVALID = []`. The current `TARGET` axes are precision-bundled (each precision name baked dtype + math_fidelity + fp32_dest_acc_en), layout, alignment, rank, dim, numeric_stable. There is one structural couple worth flagging for the next `/golden-tests` invocation:

- **`(precision = bf16_*, layout = ROW_MAJOR_LAYOUT)`**: bfloat16 (and any non-fp32 precision name in this op) tensors only have a meaningful tile representation through TILE_LAYOUT in this kernel — the helpers operate on tile-format CBs. ROW_MAJOR + bf16 would either require an extra tilize stage (not in Phase 0 scope) or is a structural mismatch with the kernel's CB format declarations. Phase 0 SUPPORTED is restricted to fp32 + TILE only, so this distinction does not bite Phase 0 acceptance, but the golden-test universe should still skip these cells rather than xfail them.

If the user re-runs the `/golden-tests` skill to refresh `feature_spec.py`, they may want to encode that bullet as an `INVALID` entry. Otherwise it remains a note here.
