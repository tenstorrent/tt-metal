# Operation Design: layer_norm_rm

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (fused reduce + binary + SFPU + tilize/untilize) |
| Goal | Per-row (final-dim) LayerNorm on a ROW_MAJOR fp32 tensor with optional affine. Output shape, dtype, and layout match input. Tilize/untilize happen entirely in-kernel — the entry point does NOT cast the tensor to TILE_LAYOUT. |
| Math | `mean = sum_w(x) / W`; `var = sum_w((x-mean)^2) / W`; `inv_std = 1 / sqrt(var + epsilon)`; `y = (x - mean) * inv_std`; if gamma is given, `y *= gamma`; if beta is given, `y += beta`. |
| Mode | Derivative (single-tile-row pipeline assembled from kernel-lib helpers: tilize → reduce-SUM(mean) → sub-bcast(centered) → square → reduce-SUM(var) + postop(+eps, rsqrt) → mul-bcast(normalize) → optional gamma `mul_in_place` + beta `add_in_place` → untilize). |
| References | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp`, `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`, `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp`, `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`, `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h`, `tt_metal/hw/inc/api/compute/eltwise_unary/rsqrt.h`, `ttnn/ttnn/operations/toy_variance/` (two-pass mean/var template), `ttnn/ttnn/operations/toy_tilize_untilize/` (RM-in / RM-out tilize+untilize template), `ttnn/ttnn/operations/softmax/op_design.md` (multi-phase reduce + broadcast template), `eval/golden_tests/layer_norm_rm/feature_spec.py` (TARGET / INPUTS / INVALID — already authored, pipeline mode). |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | fp32, ROW_MAJOR_LAYOUT, rank ≥ 2, on device, H ≥ 32 and W ≥ 32 with H%32 == 0 and W%32 == 0 (Phase 0 tile-aligned-only) | — | input |
| `gamma` | `ttnn.Tensor` or `None` | no, positional | If non-`None`: fp32, ROW_MAJOR_LAYOUT, total element count equal to input's W (logical shape `(1, 1, 1, W)`) | `None` | CT flag `has_gamma` (bool) selects whether gamma kernels run; input slot is the second tensor |
| `beta` | `ttnn.Tensor` or `None` | no, positional | If non-`None`: fp32, ROW_MAJOR_LAYOUT, total element count equal to input's W (logical shape `(1, 1, 1, W)`) | `None` | CT flag `has_beta` (bool) selects whether beta kernels run; input slot is the third tensor when gamma is absent it is still positional-third |
| `epsilon` | `float` | no, keyword-only | finite, > 0 (strictly positive; default 1e-5) | `1e-5` | CT (`uint32_t` bit-cast of fp32) — baked into the variance-reduce post-op as the add-unary scalar |

**Compute config policy.** Phase 0 hard-codes the compute config inside the program factory: `ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)`. The Python entry point does NOT expose a `compute_kernel_config` parameter at this phase — that knob is a later refinement.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(..., H, W)` with `len(shape) >= 2`, `H % 32 == 0`, `W % 32 == 0`, `H >= 32`, `W >= 32` |
| Dtype | `float32` |
| Layout | `ROW_MAJOR_LAYOUT` |
| Memory | interleaved (DRAM, default) |

### Gamma (optional)

| Property | Requirement |
|----------|-------------|
| Shape | `(1, 1, 1, W)` — total element count equal to input's `W`; rank may be any so long as `numel == W` and the only non-1 dim is the last |
| Dtype | `float32` |
| Layout | `ROW_MAJOR_LAYOUT` |
| Memory | interleaved |

### Beta (optional)

Same requirements as gamma.

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input (`(..., H, W)`) |
| Dtype | `float32` |
| Layout | `ROW_MAJOR_LAYOUT` |
| Memory | `ttnn.DRAM_MEMORY_CONFIG` (interleaved, default) |

## Dataflow Strategy

`NC` = product of `shape[:-2]`. `Ht = H/32`, `Wt = W/32`. The natural work unit is **one tile-row** (32 contiguous logical rows, each of width W — i.e. `Wt` tiles wide after tilization). LayerNorm is per-logical-row, so processing 32 rows together via REDUCE_ROW is exactly the right granularity and avoids any cross-row dependency.

| Stage | Where | Format | Notes |
|-------|-------|--------|-------|
| Source: input | DRAM interleaved | RM sticks (`W * 4` bytes each) | Stick index `sid = nc*H + h`. For each work-item the reader emits 32 contiguous sticks. |
| Source: gamma / beta (one-shot, optional) | DRAM interleaved | RM stick (`W * 4` bytes each) | A single stick at stick index 0. Read ONCE at boot, broadcast 32 times into the reader's gamma/beta RM CB so the in-kernel `tilize` helper sees a full tile-row (every row identical to gamma/beta). |
| Reader (NCRISC) → cb_input_rm | DRAM → L1 (TILE-granularity tilize CB) | RM sticks packed into tile-pages | Reader uses `dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rm, TilizeGranularity::TILE>(accessor, /*total_num_rows=*/32, row_bytes, start_page = tile_row_index * 32)` once per work-item. CB page = `ttnn.tile_size(float32)` = 4096 B. Reader pushes `Wt` pages per work-item (32 sticks × W*4 B = Wt × 4096 B). |
| Reader (boot) → cb_gamma_rm / cb_beta_rm | DRAM → L1 | RM stick replicated 32× | Per the spec ("kernels must handle row-major data natively"), the reader replicates the one gamma/beta row into 32 L1 rows so the standard `tilize<Wt>(1)` helper consumes it. Read once, push `Wt` pages, never refilled. |
| Reader (boot) → cb_scaler | (compute-thread-visible) L1 | bf16 reduce scaler tile (col-0 fill) | `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f / static_cast<float>(W))`. Pool-type-/reduce-dim-aware overload — SUM + REDUCE_ROW takes the matmul col-0 fill path (`reduce_helpers_dataflow.hpp:46-48`). Pushed ONCE at boot, never popped. |
| Compute (3 TRISCs) | per work-item | RM ↔ tiles ↔ RM | See *Compute Phases* below. Reuses the same scaler tile across both reductions (mean and variance). Optional gamma/beta `mul_in_place` / `add_in_place` execute only when their CT flag is set. |
| Writer (BRISC) ← cb_output_tiles | L1 → DRAM | RM sticks extracted from tile-pages | `dataflow_kernel_lib::write_sticks_after_untilize<cb_output_tiles>(accessor, /*total_num_rows=*/32, row_bytes, start_page = tile_row_index * 32)` once per work-item. CB page = `ttnn.tile_size(float32)` = 4096 B; writer pops `Wt` pages per work-item and writes 32 sticks back. |
| Sink: output | DRAM interleaved | RM sticks | Same stick layout as input. |

**Single-Tensix dataflow within a core, no inter-Tensix communication.** Each Tensix consumes a contiguous range of tile-rows assigned via runtime args. There are no semaphores, no multicast, and no ring topology. Cores process disjoint slices of the work-item set independently.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | One tile-row = 32 sticks × W elements = `Wt` tiles after tilization. |
| Total work items | `num_tile_rows = NC * Ht` where `NC = prod(shape[:-2])` and `Ht = H/32`. |
| Grid | `device.compute_with_storage_grid_size()` (full Wormhole compute grid; Wormhole = 8×8 = 64 cores). |
| Per-core formula | `(num_cores_total, all_cores, core_group_1, core_group_2, rows_per_core_g1, rows_per_core_g2) = ttnn.split_work_to_cores(grid, num_tile_rows)`. Group 1 cores receive `rows_per_core_g1` tile-rows; group 2 cores receive `rows_per_core_g2`. Each core receives runtime args `(start_tile_row, num_tile_rows_for_this_core)` so it knows which contiguous range to read/process/write. |
| Remainder | Handled by `ttnn.split_work_to_cores` (two-group split). Cores that would receive zero tile-rows are excluded from `all_cores`. |
| Inter-core | None. |

## Circular Buffers

`Wt = W/32`. `tile_size_f32 = ttnn.tile_size(ttnn.float32) = 4096 B`. `tile_size_bf16 = ttnn.tile_size(ttnn.bfloat16) = 2048 B`. CBs with "Only if X" exist iff the corresponding CT flag is set (the program factory skips the descriptor when X is false; the kernels gate the corresponding code with `if constexpr (X)`).

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_rm` | 0 | `tile_size_f32` (4096 B) | `2 * Wt` | fp32 | Reader (per work-item, `read_sticks_for_tilize<TILE>`) | Compute Phase 1 (`tilize`) | Refilled once per work-item; double-block sized so the reader can fill the next tile-row while compute consumes the current one. |
| `cb_gamma_rm` | 1 | `tile_size_f32` | `Wt` | fp32 | Reader (one-shot at boot, replicates gamma stick × 32) | Compute boot (`tilize`) | Filled and consumed exactly once. **Only if `has_gamma`.** |
| `cb_beta_rm` | 2 | `tile_size_f32` | `Wt` | fp32 | Reader (one-shot at boot, replicates beta stick × 32) | Compute boot (`tilize`) | Filled and consumed exactly once. **Only if `has_beta`.** |
| `cb_scaler` | 8 | `tile_size_bf16` (2048 B) | `1` | bf16 | Reader (one-shot at boot, `prepare_reduce_scaler<SUM, REDUCE_ROW>(1/W)`) | Compute Phases 2 and 5 (`reduce<SUM, REDUCE_ROW>`, both via `WaitUpfrontNoPop`-like behavior — helper waits on 1 tile, never pops) | Persistent for entire core lifetime; the same scaler tile drives both the mean and the variance reductions. |
| `cb_gamma_tiles` | 9 | `tile_size_f32` | `Wt` | fp32 | Compute boot (`tilize<Wt>(1)` from `cb_gamma_rm`) | Compute Phase 7 (`mul_in_place<ROW, NoWaitNoPop>`) every work-item | Persistent across all work-items. **Only if `has_gamma`.** |
| `cb_beta_tiles` | 10 | `tile_size_f32` | `Wt` | fp32 | Compute boot (`tilize<Wt>(1)` from `cb_beta_rm`) | Compute Phase 8 (`add_in_place<ROW, NoWaitNoPop>`) every work-item | Persistent across all work-items. **Only if `has_beta`.** |
| `cb_output_tiles` | 16 | `tile_size_f32` | `2 * Wt` | fp32 | Compute Phase 9 (`untilize<Wt>(1)` from `cb_norm`) | Writer (per work-item, `write_sticks_after_untilize`) | Refilled once per work-item; double-block sized for compute/writer pipelining. |
| `cb_input_tiles` | 24 | `tile_size_f32` | `Wt` | fp32 | Compute Phase 1 (`tilize<Wt>(1)`) | Compute Phase 2 (`reduce<SUM,REDUCE_ROW,WaitUpfrontNoPop>`) and Phase 3 (`sub<COL,WaitUpfrontPopAtEnd>`) | Filled once per work-item; sized to one full tile-row because two sequential helpers (reduce, then sub) both need the strip resident — Phase 2's WaitUpfrontNoPop leaves data in the CB and Phase 3 pops it. **Sequential helpers cannot pipeline (each owns all 3 TRISCs) — see CB-fundamentals.** |
| `cb_mean` | 25 | `tile_size_f32` | `2` | fp32 | Compute Phase 2 (`reduce<SUM, REDUCE_ROW>` writes 1 tile) | Compute Phase 3 (`sub<COL, WaitUpfrontPopAtEnd>` waits 1, pops 1) | 1 tile per work-item; double-paged so next work-item's mean can start producing while current one drains. |
| `cb_centered` | 26 | `tile_size_f32` | `Wt` | fp32 | Compute Phase 3 (`sub<COL>` writes Wt tiles) | Compute Phase 4 (`square<WaitUpfrontNoPop>` reads Wt, leaves them) and Phase 6 (`mul<COL, WaitUpfrontPopAtEnd>` reads Wt, pops Wt) | Holds one full tile-row of `(x - mean)`. **MUST hold Wt tiles** — Phase 4's WaitUpfrontNoPop requires all Wt resident, AND Phase 6 reuses the same Wt tiles for the final normalize multiply. Phases 3→4→5→6 all sequential within compute. |
| `cb_centered_sq` | 27 | `tile_size_f32` | `Wt` | fp32 | Compute Phase 4 (`square` writes Wt tiles) | Compute Phase 5 (`reduce<SUM, REDUCE_ROW, WaitAndPopPerTile>` streams Wt tiles) | One full tile-row of `(x-mean)^2`. Streaming reduce pops 1 tile at a time, but the producer (square) is a sequential helper that writes all Wt before yielding the TRISCs — so the CB must still hold Wt pages to avoid a `cb_reserve_back` deadlock. |
| `cb_inv_std` | 28 | `tile_size_f32` | `2` | fp32 | Compute Phase 5 (`reduce<SUM, REDUCE_ROW>` with `(+eps → rsqrt)` post-op writes 1 tile) | Compute Phase 6 (`mul<COL, WaitUpfrontPopAtEnd>` waits 1, pops 1) | 1 tile per work-item; double-paged for the same producer/consumer-overlap rationale as `cb_mean`. |
| `cb_norm` | 29 | `tile_size_f32` | `Wt` | fp32 | Compute Phase 6 (`mul<COL>` writes Wt tiles), then Phase 7 (`mul_in_place` pops/pushes 1 at a time over Wt) and/or Phase 8 (`add_in_place` pops/pushes 1 at a time over Wt) | Compute Phase 9 (`untilize<Wt>(1)` waits Wt, pops Wt) | Holds one full tile-row of normalized data; **exclusively owned by compute** (no reader/writer push or pop) — required by `binary_op_in_place` (`binary_op_helpers.hpp:340-348`). Each in-place op preserves the Wt-tile occupancy invariant. |

### CB sync verification

`Wt = W/32`. Boot phase fills `cb_scaler`, `cb_gamma_rm`/`cb_gamma_tiles` (if `has_gamma`), `cb_beta_rm`/`cb_beta_tiles` (if `has_beta`). Main loop runs `num_tile_rows_for_this_core` iterations.

| CB | Producer pushes | Consumer waits | Consumer pops | Net per work-item |
|----|-----------------|----------------|---------------|------------------|
| `cb_input_rm` | `Wt` (reader, per work-item) | `Wt` (`tilize` helper) | `Wt` (`tilize` helper) | balanced |
| `cb_gamma_rm` | `Wt` (reader, boot only) | `Wt` (`tilize`, boot only) | `Wt` (`tilize`, boot only) | balanced (one-shot) |
| `cb_beta_rm` | `Wt` (reader, boot only) | `Wt` (`tilize`, boot only) | `Wt` (`tilize`, boot only) | balanced (one-shot) |
| `cb_scaler` | `1` (reader, boot only) | `1` per reduce call × 2 reduces per work-item (Phase 2, Phase 5; helper does not pop with `WaitUpfrontNoPop`-equivalent on scaler) | `0` | balanced (persistent) |
| `cb_gamma_tiles` | `Wt` (compute, boot only via `tilize`) | `Wt` once at boot (`cb_wait_front(cb_gamma_tiles, Wt)`) — then `NoWaitNoPop` in every Phase 7 call | `0` | balanced (persistent) |
| `cb_beta_tiles` | `Wt` (compute, boot only) | `Wt` once at boot — then `NoWaitNoPop` in every Phase 8 call | `0` | balanced (persistent) |
| `cb_input_tiles` | `Wt` (Phase 1 `tilize`) | `Wt` (Phase 2 reduce upfront — does NOT pop; Phase 3 sub upfront pops at end) | `Wt` (Phase 3) | balanced |
| `cb_mean` | `1` (Phase 2 reduce) | `1` (Phase 3 sub on B input, `WaitUpfrontPopAtEnd`) | `1` (Phase 3) | balanced |
| `cb_centered` | `Wt` (Phase 3 sub) | `Wt` (Phase 4 square upfront — does NOT pop; Phase 6 mul upfront pops at end) | `Wt` (Phase 6) | balanced |
| `cb_centered_sq` | `Wt` (Phase 4 square) | `Wt` (Phase 5 reduce streaming, `WaitAndPopPerTile`) | `Wt` (Phase 5) | balanced |
| `cb_inv_std` | `1` (Phase 5 reduce) | `1` (Phase 6 mul on B input, `WaitUpfrontPopAtEnd`) | `1` (Phase 6) | balanced |
| `cb_norm` | `Wt` (Phase 6 mul) + (Phase 7 if `has_gamma`: Wt pushes via `mul_in_place`) + (Phase 8 if `has_beta`: Wt pushes via `add_in_place`) | matched pop counts per in-place op + `Wt` (Phase 9 `untilize`) | matched pops per in-place op + `Wt` (Phase 9) | balanced (each in-place op preserves Wt-resident invariant) |
| `cb_output_tiles` | `Wt` (Phase 9 untilize) | `Wt` (writer) | `Wt` (writer) | balanced |

### Memory budget (Phase 0)

L1 budget on Wormhole = 1.5 MB per core. The CBs above add up to (for `Wt`):

`bytes ≈ 2*Wt*4096 + Wt*4096*(has_gamma + has_beta) + 2048 + Wt*4096*(has_gamma + has_beta) + 2*Wt*4096 + Wt*4096 + 2*4096 + Wt*4096 + Wt*4096 + 2*4096 + Wt*4096`

Worst case (`has_gamma = has_beta = 1`):
`= (2 + 1 + 1 + 2 + 1 + 1 + 1 + 1) * Wt * 4096 + 2 * (2 * 4096) + 2048`
`= 10 * Wt * 4096 + 18432`
`≈ Wt * 40 KB + 18 KB`.

| W | Wt | Approx CB footprint | Fits L1 (1.5 MB)? |
|---|----|---------------------|--------------------|
| 32 | 1 | ~58 KB | yes |
| 64 | 2 | ~98 KB | yes |
| 128 | 4 | ~178 KB | yes |
| 256 | 8 | ~338 KB | yes |
| 512 | 16 | ~658 KB | yes |
| 1024 | 32 | ~1.30 MB | tight but yes |
| 2048 | 64 | ~2.58 MB | **NO — Phase 0 does not support** |

Phase 0 supports `W <= 1024` (`Wt <= 32`). Wider W is a later refinement that introduces W-axis chunking (the softmax program descriptor's Refinement 1 is the template). The op file's `SUPPORTED` rejects W > 1024 explicitly; the test universe (`INPUTS` in `feature_spec.py`) includes wider shapes but they will xfail under Phase 0.

## API Mapping

Every mechanism — helper or raw — has an exact file:line reference. All helpers are namespaced under `compute_kernel_lib` (`ckl::`) or `dataflow_kernel_lib`. `ckernel::PoolType::SUM` and `ckernel::ReduceDim::REDUCE_ROW` are passed as template args.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB (semantic name) | Output CB (semantic name) | Requirements |
|-------|------|----------|-----------|------------------------|--------------------------|---------------------------|--------------|
| Compute boot | helper | `compute_kernel_hw_startup(cb_input_rm, cb_input_tiles)` | `ttnn/cpp/ttnn/kernel_lib/cb_helpers_compute.hpp` (declared) | — | — | — | Called exactly once at the start of the compute `kernel_main`, before any other helper. Sets srcA = srcB = cb_input_rm and dst = cb_input_tiles for the first op (tilize). Helper prerequisites in `tilize_helpers.hpp:86-90`, `reduce_helpers_compute.hpp:30-33`, `binary_op_helpers.hpp:17-19`. |
| Reader boot (scaler) | helper | `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f / static_cast<float>(W))` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:65-67` | `cb_id=cb_scaler`, `pool_type=SUM`, `reduce_dim=REDUCE_ROW`, scaler `1/W` | — | `cb_scaler` | Pool-type-/reduce-dim-aware overload — selects matmul col-0 fill for SUM + REDUCE_ROW (`reduce_helpers_dataflow.hpp:46-48`). **Use `prepare_reduce_scaler`, NOT `calculate_and_prepare_reduce_scaler`**: the latter forces scaler = 1.0 for SUM (`reduce_helpers_dataflow.hpp:77`) and would compute total instead of mean. The same scaler (1/W) feeds both Phase 2 and Phase 5 — pushed once at boot, never popped. |
| Reader (per work-item) | helper | `dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rm, TilizeGranularity::TILE>(accessor, /*total_num_rows=*/32, row_bytes, /*start_page=*/(start_tile_row + i) * 32)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:87-93` | `cb_id=cb_input_rm`, `granularity=TILE` (CB pages = tile-size) | DRAM input buffer (via `TensorAccessor`) | `cb_input_rm` (`Wt` pages per call) | TILE granularity matches the compute-side `tilize<Wt, cb_input_rm, cb_input_tiles>(1)` symmetric mode (`tilize_helpers.hpp:138-139`). The helper batches the 32 sticks into `Wt` tile-pages and pushes once per work-item. |
| Reader boot (gamma) | raw_api | NoC read-and-replicate sequence into `cb_gamma_rm` | `tt_metal/hw/inc/dataflow_api.h` (`noc_async_read`, `cb_reserve_back`, `cb_push_back`, `get_write_ptr`); pattern lives in this op file | `cb_reserve_back(cb_gamma_rm, Wt);` then 32× `noc_async_read(noc_addr_of_gamma_stick_0, l1_addr + r * padded_row_bytes, row_bytes);` then `noc_async_read_barrier();` `cb_push_back(cb_gamma_rm, Wt);` | DRAM gamma buffer | `cb_gamma_rm` | **Helpers considered and rejected**: `read_sticks_for_tilize` (`tilize_helpers_dataflow.hpp:87-93`) reads `total_num_rows` **distinct** sticks from the accessor — it has no "replicate one source stick across N L1 rows" mode. Source stick is `accessor.get_noc_addr(start_page + row_offset)`, advancing row by row (verified by reading the inline `tilize_helpers_dataflow.inl`). Calling it with `total_num_rows=32` would attempt to read 32 distinct sticks from a single-stick gamma tensor — invalid. The raw replicate-32× pattern is the minimum-correctness expansion of `read_sticks_for_tilize` for a 1-stick source; everything else (CB sync, tile-page boundary, padding) is identical. Only if `has_gamma`. Reader runs this once at boot before the per-work-item loop. |
| Reader boot (beta) | raw_api | Identical replicate-32× pattern into `cb_beta_rm` | (same as gamma) | (same as gamma, swap `cb_beta_rm` and beta accessor) | DRAM beta buffer | `cb_beta_rm` | Same rejection rationale as gamma. Only if `has_beta`. |
| Compute boot (gamma tilize) | helper | `compute_kernel_lib::tilize<Wt, cb_gamma_rm, cb_gamma_tiles>(1)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:178-187` | `block_width_tiles=Wt`, `input_cb=cb_gamma_rm`, `output_cb=cb_gamma_tiles`; defaults (`InitAndUninit`, `WaitBlock`, `UnpackAndPackReconfigure`, `Fast`) | `cb_gamma_rm` (`Wt` pages) | `cb_gamma_tiles` (`Wt` pages) | Helper handles wait/pop on input, reserve/push on output. After this call, kernel does `cb_wait_front(cb_gamma_tiles, Wt)` once so subsequent `NoWaitNoPop` uses are safe (`binary_op_helpers.hpp:141-146`). Only if `has_gamma`. |
| Compute boot (beta tilize) | helper | `compute_kernel_lib::tilize<Wt, cb_beta_rm, cb_beta_tiles>(1)` | `tilize_helpers.hpp:178-187` | Same template-arg pattern as gamma | `cb_beta_rm` (`Wt` pages) | `cb_beta_tiles` (`Wt` pages) | Same as gamma. Only if `has_beta`. |
| Phase 1 (per work-item) | helper | `compute_kernel_lib::tilize<Wt, cb_input_rm, cb_input_tiles>(1)` | `tilize_helpers.hpp:178-187` | `block_width_tiles=Wt`, defaults; `num_blocks=1` | `cb_input_rm` (`Wt` pages) | `cb_input_tiles` (`Wt` pages) | Symmetric TILE-granularity path — both CBs have tile-size pages (`tilize_helpers.hpp:138-139`). Helper handles wait/pop/reserve/push internally. |
| Phase 2 (per work-item) | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, ReduceInputPolicy::WaitUpfrontNoPop>(cb_input_tiles, cb_scaler, cb_mean, ReduceInputBlockShape::of(1, Wt))` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:400-415` | `reduce_type=SUM`, `reduce_dim=REDUCE_ROW`, `input_policy=WaitUpfrontNoPop`; shape `of(1, Wt)` | `cb_input_tiles` (Wt tiles, waited upfront, **NOT popped**), `cb_scaler` (1 bf16 tile, persistent) | `cb_mean` (1 tile pushed) | `WaitUpfrontNoPop` leaves `cb_input_tiles` intact for Phase 3 — softmax-style pattern (`reduce_helpers_compute.hpp:366-369`). Scaler is 1/W → SUM produces row means. Output is a column-vector tile (REDUCE_ROW SUM via matmul path writes col-0 valid; verified in `reduce_helpers_compute.inl` SUM/REDUCE_ROW dispatch). |
| Phase 3 (per work-item) | helper | `compute_kernel_lib::sub<BroadcastDim::COL, BinaryInputPolicy::WaitUpfrontPopAtEnd, BinaryInputPolicy::WaitUpfrontPopAtEnd>(cb_input_tiles, cb_mean, cb_centered, BinaryInputBlockShape::of(1, Wt))` | `binary_op_helpers.hpp:283-293` (declaration) | `bcast_dim=COL`, `input_a_policy=WaitUpfrontPopAtEnd` (pairs with Phase 2's WaitUpfrontNoPop on the same CB), `input_b_policy=WaitUpfrontPopAtEnd`; shape `of(1, Wt)` | `cb_input_tiles` (Wt, popped at end), `cb_mean` (1, popped at end) | `cb_centered` (Wt tiles pushed) | `BroadcastDim::COL` because `cb_mean` is a column-vector (REDUCE_ROW output) broadcast across columns of each input tile (`binary_op_helpers.hpp:36-43`). Pattern lifts directly from softmax Phase B (`ttnn/ttnn/operations/softmax/op_design.md`) and toy_variance Pass-2 sub (`toy_variance/kernels/compute.cpp:79-82`). |
| Phase 4 (per work-item) | helper | `compute_kernel_lib::square<BinaryInputPolicy::WaitUpfrontNoPop>(cb_centered, cb_centered_sq, BinaryInputBlockShape::of(1, Wt))` | `binary_op_helpers.hpp:309-316` (declaration) | `input_policy=WaitUpfrontNoPop` — leaves `cb_centered` intact for Phase 6's `mul` | `cb_centered` (Wt, waited upfront, **NOT popped**) | `cb_centered_sq` (Wt tiles pushed) | `square` is the single-CB variant of `binary_op<SQUARE>` (computes `cb_in * cb_in`); we use it precisely to preserve `cb_centered` for later reuse — `toy_variance` does the analogous thing in-place via `square_in_place`, but here we cannot consume `cb_centered` because Phase 6 still needs it (mul-by-`inv_std`). |
| Phase 5 (per work-item) | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_centered_sq, cb_scaler, cb_inv_std, ReduceInputBlockShape::of(1, Wt), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, postop)` where `postop = [eps_bits](uint32_t dst) { add_unary_tile_init(); add_unary_tile(dst, eps_bits); rsqrt_tile_init(); rsqrt_tile(dst); }` | `reduce_helpers_compute.hpp:400-415` (declaration); post-op composition seam at `reduce_helpers_compute.hpp:309-313` and the `recip_tile` example at `reduce_helpers_compute.hpp:377-385`; `add_unary_tile` at `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:28`; `rsqrt_tile_init` / `rsqrt_tile` at `tt_metal/hw/inc/api/compute/eltwise_unary/rsqrt.h:18` and `:37` | `reduce_type=SUM`, `reduce_dim=REDUCE_ROW`, default `input_policy=WaitAndPopPerTile` (streaming); `accumulate=NoAccumulation{}`; `post_reduce_op=postop` callable with the captured `eps_bits` from CT arg | `cb_centered_sq` (Wt, streamed), `cb_scaler` (1, persistent) | `cb_inv_std` (1 tile pushed) | Composition seam: the post-op runs inside the reduce's dst-sync window before pack (analogous to softmax's `recip` post-op, `softmax.cpp:289`). One LLK fusion produces `1 / sqrt(var + eps)` directly into the output CB — no separate "eps add" or "rsqrt" pass / CB. **`add_unary_tile` and `rsqrt_tile` are raw SFPU APIs**, not helpers; **Helpers considered and rejected for the post-op body**: `sfpu_op` / `sfpu_chain` (`sfpu_helpers.hpp:1141`+) operate on whole CBs of tiles (their `apply()` does its own CB wait/reserve/push) — they can't be invoked on a single DST register inside another helper's dst-sync window. The reduce helper documents calling raw LLK SFPU functions inside the post-op lambda (`reduce_helpers_compute.hpp:377-385` shows `recip_tile_init(); recip_tile(dst_idx);`). The post-op pattern is the documented seam; using helpers there would violate their CB protocol. |
| Phase 6 (per work-item) | helper | `compute_kernel_lib::mul<BroadcastDim::COL, BinaryInputPolicy::WaitUpfrontPopAtEnd, BinaryInputPolicy::WaitUpfrontPopAtEnd>(cb_centered, cb_inv_std, cb_norm, BinaryInputBlockShape::of(1, Wt))` | `binary_op_helpers.hpp:296-306` (declaration) | `bcast_dim=COL`, `input_a_policy=WaitUpfrontPopAtEnd` (pairs with Phase 4's WaitUpfrontNoPop on `cb_centered`), `input_b_policy=WaitUpfrontPopAtEnd`; shape `of(1, Wt)` | `cb_centered` (Wt, popped at end), `cb_inv_std` (1, popped at end) | `cb_norm` (Wt tiles pushed) | Same broadcast direction as Phase 3 (REDUCE_ROW output is column-vector → COL broadcast). Pattern matches softmax Phase D mul. |
| Phase 7 (per work-item, optional) | helper | `compute_kernel_lib::mul_in_place<BroadcastDim::ROW, BinaryInputPolicy::NoWaitNoPop>(cb_norm, cb_gamma_tiles, BinaryInputBlockShape::of(1, Wt))` | `binary_op_helpers.hpp:456-462` | `bcast_dim=ROW`, `input_b_policy=NoWaitNoPop` (gamma is pre-waited at boot, persistent across work-items) | `cb_norm` (in-place: pops 1 → packs 1 over Wt iterations), `cb_gamma_tiles` (Wt, no wait, no pop) | `cb_norm` (Wt tiles, modified in place) | `BroadcastDim::ROW` broadcasts B's row-0 across the 32 rows of each A tile (`binary_op_helpers.hpp:27-31`) — correct because gamma's logical shape is (1, W): the same gamma row applies to all 32 logical rows in this tile-row. In-place pattern preserves Wt-tile occupancy of `cb_norm` (`binary_op_helpers.hpp:333-339`). Only if `has_gamma`. |
| Phase 8 (per work-item, optional) | helper | `compute_kernel_lib::add_in_place<BroadcastDim::ROW, BinaryInputPolicy::NoWaitNoPop>(cb_norm, cb_beta_tiles, BinaryInputBlockShape::of(1, Wt))` | `binary_op_helpers.hpp:438-444` | `bcast_dim=ROW`, `input_b_policy=NoWaitNoPop` (beta pre-waited at boot, persistent) | `cb_norm` (in-place), `cb_beta_tiles` (Wt, no wait, no pop) | `cb_norm` (Wt tiles, modified in place) | Same in-place + ROW-bcast rationale as Phase 7. Only if `has_beta`. |
| Phase 9 (per work-item) | helper | `compute_kernel_lib::untilize<Wt, cb_norm, cb_output_tiles>(1)` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:132-140` | `block_width_tiles=Wt`, defaults; `num_blocks=1` | `cb_norm` (Wt) | `cb_output_tiles` (Wt) | Both CBs have tile-size pages (symmetric, the only mode the untilize helper supports — `untilize_helpers.hpp:96`). Compute pops `cb_norm` and pushes `cb_output_tiles` via the helper. |
| Writer (per work-item) | helper | `dataflow_kernel_lib::write_sticks_after_untilize<cb_output_tiles>(accessor, /*total_num_rows=*/32, row_bytes, /*start_page=*/(start_tile_row + i) * 32)` | `tilize_helpers_dataflow.hpp:129-135` | `cb_id=cb_output_tiles` | `cb_output_tiles` (Wt pages per call) | DRAM output buffer | Helper waits Wt pages, writes 32 sticks (W*4 bytes each) to DRAM, pops Wt. |

## Compute Phases

`Wt = W / 32`. The per-work-item loop runs `num_tile_rows_for_this_core` iterations. All boot work (scaler, gamma/beta tilize) happens before the loop.

### Boot (executed once per core)

| # | Operation | Helper? | Input CB | Output CB | CB State After |
|---|-----------|---------|----------|-----------|----------------|
| B0 | `compute_kernel_hw_startup(cb_input_rm, cb_input_tiles)` | helper | — | — | hardware initialized for tilize as first op |
| B1 (only if `has_gamma`) | tilize gamma RM → tiles | `tilize<Wt, cb_gamma_rm, cb_gamma_tiles>(1)` | `cb_gamma_rm` (Wt) | `cb_gamma_tiles` (Wt) | `cb_gamma_rm` drained; `cb_gamma_tiles` holds Wt tiles permanently |
| B2 (only if `has_gamma`) | one-shot `cb_wait_front(cb_gamma_tiles, Wt)` | raw CB op | `cb_gamma_tiles` | — | makes subsequent `NoWaitNoPop` uses legal |
| B3 (only if `has_beta`) | tilize beta RM → tiles | `tilize<Wt, cb_beta_rm, cb_beta_tiles>(1)` | `cb_beta_rm` (Wt) | `cb_beta_tiles` (Wt) | `cb_beta_rm` drained; `cb_beta_tiles` permanent |
| B4 (only if `has_beta`) | one-shot `cb_wait_front(cb_beta_tiles, Wt)` | raw CB op | `cb_beta_tiles` | — | makes subsequent `NoWaitNoPop` uses legal |

### Per work-item loop (executed `num_tile_rows_for_this_core` times)

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | tilize input | `tilize<Wt, cb_input_rm, cb_input_tiles>(1)` | `cb_input_rm` (Wt sticks-batched-into-tile-pages, WaitBlock) | `cb_input_tiles` (Wt tiles) | `cb_input_rm` drained; `cb_input_tiles` holds the tile-row |
| 2 | MEAN reduce | `reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>` | `cb_input_tiles` (Wt, waited upfront, **NOT popped**); `cb_scaler` (1, persistent, NoPop) | `cb_mean` (1) | `cb_input_tiles` still holds the strip (Phase 3 will consume); `cb_mean` holds 1 tile (column-vector of per-row means); scaler persistent |
| 3 | SUB (centered) | `sub<COL, WaitUpfrontPopAtEnd, WaitUpfrontPopAtEnd>` | `cb_input_tiles` (Wt, popped at end); `cb_mean` (1, popped at end) | `cb_centered` (Wt) | `cb_input_tiles` drained; `cb_mean` drained; `cb_centered` holds Wt tiles of `(x − mean)` |
| 4 | SQUARE (preserving) | `square<WaitUpfrontNoPop>` | `cb_centered` (Wt, waited upfront, **NOT popped**) | `cb_centered_sq` (Wt) | `cb_centered` still holds Wt tiles (Phase 6 will consume); `cb_centered_sq` holds Wt tiles of `(x − mean)^2` |
| 5 | VAR reduce + (+eps, rsqrt) post-op | `reduce<SUM, REDUCE_ROW>` with post-op `[eps_bits](dst){ add_unary_tile_init(); add_unary_tile(dst, eps_bits); rsqrt_tile_init(); rsqrt_tile(dst); }` | `cb_centered_sq` (Wt, streamed via default WaitAndPopPerTile); `cb_scaler` (1, persistent, NoPop) | `cb_inv_std` (1) | `cb_centered_sq` drained; `cb_inv_std` holds 1 tile of `1/sqrt(var + epsilon)` (column-vector); scaler persistent |
| 6 | MUL by inv_std (normalize) | `mul<COL, WaitUpfrontPopAtEnd, WaitUpfrontPopAtEnd>` | `cb_centered` (Wt, popped at end); `cb_inv_std` (1, popped at end) | `cb_norm` (Wt) | `cb_centered` drained; `cb_inv_std` drained; `cb_norm` holds Wt tiles of normalized output |
| 7 (only if `has_gamma`) | MUL by gamma in place | `mul_in_place<ROW, NoWaitNoPop>` | `cb_norm` (in-place: 1-tile pop/pack cycle × Wt); `cb_gamma_tiles` (Wt, no wait, no pop) | `cb_norm` (Wt, modified) | `cb_norm` holds Wt tiles of `norm * gamma`; `cb_gamma_tiles` persistent |
| 8 (only if `has_beta`) | ADD beta in place | `add_in_place<ROW, NoWaitNoPop>` | `cb_norm` (in-place); `cb_beta_tiles` (Wt, no wait, no pop) | `cb_norm` (Wt, modified) | `cb_norm` holds Wt tiles of `norm * gamma + beta` (or `norm + beta` if `!has_gamma`); `cb_beta_tiles` persistent |
| 9 | untilize → output | `untilize<Wt, cb_norm, cb_output_tiles>(1)` | `cb_norm` (Wt, WaitBlock) | `cb_output_tiles` (Wt) | `cb_norm` drained; `cb_output_tiles` holds Wt tile-pages ready for the writer to extract 32 sticks |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| Phase 3 | `sub` | `cb_input_tiles`: All (`1 × Wt` tile-row, every tile fully valid) | `cb_mean`: Col0 of the single tile (REDUCE_ROW SUM via matmul path produces column-vector output, col-0 valid) | `BroadcastDim::COL` — B has Ht=1 tile, replicated across columns of each A tile |
| Phase 6 | `mul` | `cb_centered`: All (`1 × Wt`) | `cb_inv_std`: Col0 of the single tile (REDUCE_ROW SUM + post-op produces column-vector output) | `BroadcastDim::COL` — same broadcast direction as Phase 3 |
| Phase 7 (optional) | `mul_in_place` | `cb_norm`: All (`1 × Wt`) | `cb_gamma_tiles`: Row0 of each tile (`(1, Wt)` row-vector, valid only in row-0 of each gamma tile — well-defined regardless of replication because `BroadcastDim::ROW` instructs the LLK to apply B's row-0 across all 32 rows of A's corresponding tile, `binary_op_helpers.hpp:27-31`) | `BroadcastDim::ROW` — B has Ht=1 tile-row of width Wt, replicated down rows of each A tile |
| Phase 8 (optional) | `add_in_place` | `cb_norm`: All (`1 × Wt`) | `cb_beta_tiles`: Row0 of each tile (same shape as gamma) | `BroadcastDim::ROW` — same broadcast direction as Phase 7 |

Valid-region key (per `op-design-template.md`): REDUCE_ROW output → Col0; REDUCE_COL output → Row0; (1, W) row vector → Row0. The post-reduce broadcast direction is the *conjugate* of the reduce direction (`binary_op_helpers.hpp:36-43`): REDUCE_ROW → COL broadcast; a (1, W) row-vector applied across H → ROW broadcast.

## Key Risks and Gotchas

| Risk | Why it matters | Mitigation |
|------|----------------|------------|
| `cb_input_tiles`, `cb_centered`, `cb_centered_sq`, `cb_norm` MUST be sized to a full tile-row (Wt) | Sequential compute helpers (tilize → reduce → sub → square → reduce → mul → in-place ops → untilize) cannot pipeline because each helper owns all 3 TRISCs (CB-fundamentals "Intermediate CB Sizing Between Compute Helpers"). Undersized CBs deadlock with the producing helper blocked on `cb_reserve_back`. | All four CBs allocated with `num_pages = Wt` (the full tile-row). Verified in the CB sizing table. |
| `cb_norm` must be EXCLUSIVELY owned by compute | `binary_op_in_place` documents the rule (`binary_op_helpers.hpp:340-348`): cb_a must NOT be touched by reader or writer while in-place ops run, or `cb_push_back`/`fifo_wr_ptr` races corrupt data. | The reader never touches `cb_norm`; the writer never touches it (writer reads from `cb_output_tiles` via the untilize helper). The compute kernel is the sole producer and consumer. |
| Scaler CB is bfloat16 even for fp32 input | The reduce LLK uses the scaler tile as SrcB, which is bf16 by hardware contract. Mismatched format produces silently wrong results. | `cb_scaler` is bf16; `prepare_reduce_scaler` deduces format from the CB and writes the correct bit representation (`reduce_helpers_dataflow.hpp:42-67`). |
| Pool-type-/reduce-dim-aware scaler API is mandatory | The default-template overload of `prepare_reduce_scaler` (without explicit `PoolType`/`ReduceDim`) defaults to MAX + REDUCE_COL, which uses row-0 fill — wrong for SUM + REDUCE_ROW, which needs col-0 (matmul) fill (`reduce_helpers_dataflow.hpp:46-48`). | The scaler call passes `<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>` explicitly. Hardware Constraints checklist below also flags this. |
| `prepare_reduce_scaler` (caller-provided value) vs `calculate_and_prepare_reduce_scaler` (auto value) | `calculate_and_prepare_reduce_scaler` forces scaler = 1.0 for SUM (`reduce_helpers_dataflow.hpp:73-77`). We need 1/W. | Use `prepare_reduce_scaler<cb, SUM, REDUCE_ROW>(1.0f / W)` — caller-provided value. (Same choice as `toy_variance/kernels/reader.cpp:50-52`.) |
| `fp32_dest_acc_en = True` halves DEST capacity (8 → 4 in half-sync) | Hand-rolled `tile_regs_acquire` loops sized for bf16 would overflow DEST. | All helpers respect `DEST_AUTO_LIMIT` (`dest_helpers.hpp:88-99`). The design never hand-codes a DEST loop. The Phase-5 post-op operates on a single DST register — safe regardless of capacity. |
| Phase 2's `WaitUpfrontNoPop` on `cb_input_tiles` paired with Phase 3's `WaitUpfrontPopAtEnd` on the same CB | Double-wait is intentional: Phase 2 leaves data in the CB for Phase 3. Phase 3's `wait_front(Wt)` is idempotent — data is already there. The pop happens at the end of Phase 3. Same applies to Phase 4 (`WaitUpfrontNoPop` on `cb_centered`) → Phase 6 (`WaitUpfrontPopAtEnd` on same CB). | Documented softmax pattern (`reduce_helpers_compute.hpp:366-369` and `binary_op_helpers.hpp:65-70`). No special handling needed. |
| Gamma/beta tilization requires 32 input rows but gamma/beta is 1 row | The `tilize<Wt>(1)` helper requires `Wt` tile-pages = 32 sticks of input. A single gamma stick would underfill. | The reader replicates gamma/beta 32 times into the RM CB before tilize. Combined with `BroadcastDim::ROW` on the downstream `mul_in_place`/`add_in_place`, only row-0 of each gamma/beta tile matters — but having all 32 rows identical is the simplest correctness contract and the path of least code in the reader. |
| `has_gamma` / `has_beta` CT flags must drive the program factory (CB allocation), reader (NoC reads), AND compute (helper calls) consistently | Allocating a CB without filling it would deadlock on first wait; filling without allocating is a memory error. | Single source of truth: the program factory computes `has_gamma = (gamma is not None)` once, then (a) skips the descriptor for the unused CB(s), (b) passes the CT flag to the reader and compute kernels, (c) the kernels gate their boot/loop sections on `if constexpr (has_gamma)`. The writer is unconditional (always drains `cb_output_tiles`). |
| `epsilon > 0` strictly | `rsqrt(x)` is undefined at `x = 0` (variance of a constant row is 0); without epsilon, the result is `±inf`. | Validation rejects `epsilon <= 0`. Default `1e-5` is the standard PyTorch LayerNorm default. |
| Wide W (W > 1024) exceeds L1 budget | At Wt = 32 the CB footprint is ~1.3 MB; at Wt = 64 it is ~2.6 MB → no fit. | Phase 0 `SUPPORTED` restricts W ≤ 1024 explicitly; wider shapes in `feature_spec.py:INPUTS` will xfail. The widening refinement adds W-axis chunking (softmax Refinement 1 template): split the tile-row into BLOCK_SIZE-tile blocks, hold `cb_mean`/`cb_inv_std` across all blocks, run the per-block sub+square+sum twice (mean pass, then var+normalize pass). Out of Phase 0 scope. |
| Non-tile-aligned H or W | Phase 0 requires `H % 32 == 0` and `W % 32 == 0`. The reader's `read_sticks_for_tilize` would otherwise need partial-block handling; the reduce would need the partial-scaler API. | `validate()` rejects non-aligned shapes (Phase 0). The non-aligned axes (`w_non_aligned`, `h_non_aligned` in `feature_spec.py`) are TARGET universe but not in Phase-0 SUPPORTED, so they will xfail — refinement candidates. |

## Hardware Constraints

- [x] CB sync: push count = wait count for every CB (see CB sync table above).
- [x] Reduce scaler CB is bfloat16 (`cb_scaler` is bf16 even though input/output are fp32).
- [x] Reduce scaler uses pool-type-aware API: `prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f / W)` — NOT the legacy default-template-arg overload. Caller-provided value (1/W) is required because `calculate_and_prepare_reduce_scaler` forces scaler = 1.0 for SUM.
- [x] DEST budget: `fp32_dest_acc_en=True` → half-sync limit = 4 tiles. `DEST_AUTO_LIMIT` resolves to 4 automatically (`dest_helpers.hpp:88-99`). All helpers respect this internally. The Phase-5 post-op operates on a single DST register, well under the limit.
- [x] Sequential helper intermediates sized to full block: `cb_input_tiles`, `cb_centered`, `cb_centered_sq`, `cb_norm` each hold `Wt` tiles.
- [x] Page sizes aligned: tile CBs use `ttnn.tile_size(ttnn.float32) = 4096 B`; scaler CB uses `ttnn.tile_size(ttnn.bfloat16) = 2048 B`. RM CBs (`cb_input_rm`, `cb_gamma_rm`, `cb_beta_rm`, `cb_output_tiles`) use tile-size pages because the dataflow helpers operate at TILE granularity (page = tile_size; 32 sticks pack into Wt pages).
- [x] All `cb_wait_front` calls on the same CB use the same page count: tile-CB waits use `Wt` (matched by tilize/untilize Wt push, reduce/binary helpers compute total from `ReduceInputBlockShape::of(1, Wt)` / `BinaryInputBlockShape::of(1, Wt)`); 1-tile CBs (`cb_mean`, `cb_inv_std`, `cb_scaler`) wait for 1.
- [x] Helpers are not wrapped with extra CB operations. Every CB is owned by exactly one helper (or one pair of cooperating helpers), and the helpers internally manage `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front`. The only "raw" CB ops in the compute kernel are the boot-time `cb_wait_front` on `cb_gamma_tiles` / `cb_beta_tiles` (needed by the `NoWaitNoPop` policy contract — `binary_op_helpers.hpp:141-146`).
- [x] Every compute phase uses a helper. Phase 1: `tilize`. Phase 2: `reduce<SUM, REDUCE_ROW, WaitUpfrontNoPop>`. Phase 3: `sub<COL, WaitUpfrontPopAtEnd, WaitUpfrontPopAtEnd>`. Phase 4: `square<WaitUpfrontNoPop>`. Phase 5: `reduce<SUM, REDUCE_ROW>` with raw-SFPU post-op (`add_unary_tile` + `rsqrt_tile` — documented post-op composition seam, see API Mapping rationale). Phase 6: `mul<COL, WaitUpfrontPopAtEnd, WaitUpfrontPopAtEnd>`. Phase 7: `mul_in_place<ROW, NoWaitNoPop>` (conditional). Phase 8: `add_in_place<ROW, NoWaitNoPop>` (conditional). Phase 9: `untilize`. The only raw-API fallback is the reader's gamma/beta replicate-32× sequence, with rejection rationale documented in the API Mapping table.
- [x] `compute_kernel_hw_startup()` called exactly once at the start of the compute kernel (with `(cb_input_rm, cb_input_tiles)` since tilize is the first op), before any helper invocation. Never re-called.

## Structural Impossibilities (notes for the golden-tests skill)

`feature_spec.py` (already authored in pipeline mode, `eval/golden_tests/layer_norm_rm/feature_spec.py`) declares `INVALID` with five entries:

- `{dtype: bfloat8_b, layout: ROW_MAJOR}` — block-quantized + RM has no meaning (input tensor).
- `{affine_dtype: bfloat8_b, affine_layout: ROW_MAJOR}` — same impossibility on the affine tensors.
- `{affine: no_affine, affine_dtype: bfloat16}` — canonicalization: when no gamma/beta is supplied, the affine_dtype × affine_layout cartesian collapses.
- `{affine: no_affine, affine_dtype: bfloat8_b}` — same canonicalization.
- `{affine: no_affine, affine_layout: ROW_MAJOR_LAYOUT}` — same canonicalization.

These are correct and sufficient for the existing TARGET universe; no additional structural impossibilities to flag at design time.

A note for the `/golden-tests` skill if `feature_spec.py` is regenerated: the current TARGET does **not** include `epsilon` as a finite axis — it's left as a single value driven by helpers/run_layer_norm. That's appropriate for now (epsilon is continuous; the cartesian-tester axis surface should stay categorical), but if a later refinement adds named precision bundles like in softmax, the planner should re-examine whether `compute_kernel_config` enters TARGET as a bundled axis.
