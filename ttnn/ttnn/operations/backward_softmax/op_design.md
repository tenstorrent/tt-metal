# Operation Design: backward_softmax

## Overview

| Field | Value |
|-------|-------|
| Classification | compute |
| Goal | Vector-Jacobian product (VJP) of softmax. Given upstream gradient `grad_output` (dy) and forward softmax output `output` (y), produce the gradient with respect to the softmax input. |
| Math | `grad_input = output * (grad_output - sum(output * grad_output, dim))` |
| Mode | Derivative |
| References | `ttnn/ttnn/operations/toy_variance/` (two-pass streaming reduce + per-block binary fan-out), `ttnn/ttnn/operations/toy_binary_in_place/` (dual-input reader pattern), `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `grad_output` | `ttnn.Tensor` | yes | float32, TILE_LAYOUT, 4D, H%32==0, W%32==0, on-device | — | runtime (buffer addr) |
| `output` | `ttnn.Tensor` | yes | float32, TILE_LAYOUT, 4D, H%32==0, W%32==0, identical shape & dtype to `grad_output`, on-device | — | runtime (buffer addr) |
| `dim` | `int` | no | `{-1, -2}` (any other value rejected) | `-1` | compile-time (DIM_IS_W flag) |

### Compute Config (hard-coded internally — NOT a caller parameter)

| Field | Value |
|-------|-------|
| `math_fidelity` | `ttnn.MathFidelity.HiFi4` |
| `fp32_dest_acc_en` | `True` |
| Effective DEST capacity | 4 tiles (half-sync + fp32 acc) — kernel-lib helpers honor this via `DEST_AUTO_LIMIT` |

This is the Phase 0 maximum-precision configuration. The compute kernel never reads a caller-supplied compute config; the API surface does not expose one.

## Tensors

### Input — `grad_output` (dy) and `output` (y), identical metadata

| Property | Requirement |
|----------|-------------|
| Shape | `(N, C, H, W)` — rank == 4 (validated). Identical between the two inputs. |
| Dtype | `float32` (both tensors). Mismatched dtypes → `ValueError`. |
| Layout | `TILE_LAYOUT` (both). `ROW_MAJOR_LAYOUT` → `ValueError`. |
| Memory | DRAM or L1 interleaved |
| Tile-alignment | `H % 32 == 0`, `W % 32 == 0`. Non-aligned → `ValueError`. |

### Output — `grad_input` (dx)

| Property | Value |
|----------|-------|
| Shape | identical to inputs `(N, C, H, W)` |
| Dtype | `float32` |
| Layout | `TILE_LAYOUT` |
| Memory | inherited from `memory_config` arg, defaults to DRAM interleaved |

## Validation (Python side, before launch)

| Check | Failure |
|-------|---------|
| `grad_output.dtype == float32` | `ValueError` |
| `output.dtype == float32` | `ValueError` |
| `grad_output.dtype == output.dtype` | `ValueError` (mismatched dtypes) |
| `grad_output.layout == TILE_LAYOUT` | `ValueError` |
| `output.layout == TILE_LAYOUT` | `ValueError` |
| `len(grad_output.shape) == 4` | `ValueError` (rank must be 4) |
| `len(output.shape) == 4` | `ValueError` (rank must be 4) |
| `tuple(grad_output.shape) == tuple(output.shape)` | `ValueError` (shape mismatch) |
| `grad_output.shape[-1] % 32 == 0 and grad_output.shape[-2] % 32 == 0` | `ValueError` |
| `dim in {-1, -2}` | `ValueError` |

## Dataflow Strategy

The reduction is over a single dimension, but each tensor element of `grad_input` depends on (a) its own `output[i]`, (b) its own `grad_output[i]`, and (c) the per-row/per-col sum of `output * grad_output`. That sum cannot be computed in parallel with its consumer because the consumer reads it broadcasted; therefore this op is a **two-pass streaming algorithm** over the reduction axis, with the inputs re-read from DRAM in pass 2.

| Stage | Role | Data path |
|-------|------|-----------|
| **DRAM → reader** | NCRISC reader streams `grad_output` and `output` tiles from DRAM into two separate input CBs, one tile from each per loop iteration (lockstep). The full lane is streamed twice (once per pass). | `grad_output` DRAM → `cb_grad_output`; `output` DRAM → `cb_output` |
| **Pass 1 (compute)** | TRISCs multiply `dy * y` element-wise into `cb_prod` (one block at a time), then accumulate-reduce that block into `cb_sum`. After all blocks, `cb_sum` holds 1 tile (the lane's sum). | `cb_grad_output`, `cb_output` → `cb_prod` → `cb_sum` |
| **Pass 2 (compute)** | TRISCs subtract the persistent scalar `cb_sum` from streaming `dy` into `cb_centered`, then multiply by streaming `y` to produce the final `grad_input` block, pushed to `cb_grad_input`. `cb_sum` is held with `WaitUpfrontNoPop` across all pass-2 blocks and popped once after the lane completes. | `cb_grad_output`, `cb_sum` → `cb_centered`; `cb_centered`, `cb_output` → `cb_grad_input` |
| **compute → writer** | BRISC writer drains `cb_grad_input` one tile at a time and writes back to DRAM at the same logical tile_id as the corresponding `grad_output` tile that produced it. | `cb_grad_input` → `grad_input` DRAM |

**No inter-Tensix communication.** Each core processes its assigned reduction lanes end-to-end; there is no multicast, semaphore, or ring topology. The operation is embarrassingly parallel across lanes.

**Tensor format does not change.** Inputs arrive tiled (TILE_LAYOUT), all CBs hold tiles, and the output is written tiled. No tilize/untilize step is needed.

## Work Distribution

The work unit is a **reduction lane** — the slice of tiles whose values collapse to a single reduced sum. Each lane is processed end-to-end (both passes) on a single core before the core moves to its next lane.

| Field | dim = -1 (reduce over W) | dim = -2 (reduce over H) |
|-------|--------------------------|--------------------------|
| Work unit | One row of tiles (size `Wt` tiles) within one (n, c, h)-slice | One column of tiles (size `Ht` tiles) within one (n, c, w)-slice |
| Total lanes | `N * C * Ht` | `N * C * Wt` |
| Grid | `device.compute_with_storage_grid_size()` |  same |
| Per-core work | `pages_per_core_g{1,2}` lanes, computed by `ttnn.split_work_to_cores(grid_size, total_lanes)` | same |
| Remainder | Handled by `ttnn.split_work_to_cores`'s two-group split (`core_group_1` + `core_group_2`); each group has a uniform per-core lane count differing by at most 1. | same |
| Per-core RT args | `start_lane`, `num_lanes`, `grad_output.buffer_address()`, `output.buffer_address()`, `grad_input.buffer_address()` | same |

`split_work_to_cores` returns `(num_cores, all_cores, core_group_1, core_group_2, pages_per_core_g1, pages_per_core_g2)`. `all_cores` is the union and is used for CB descriptors. The two compute/reader/writer KernelDescriptors have one entry each, but each kernel sets per-core RT args by walking `core_group_1` and `core_group_2` and assigning the appropriate `num_lanes`.

### Lane → tile_id decomposition (for reader/writer)

Let `NC = N * C`, `nc = lane / reduce_lanes_per_nc`, and `idx = lane % reduce_lanes_per_nc`. With `tile_id_origin = nc * Ht * Wt`:

| dim | reduce_lanes_per_nc | block-tile offset within lane (block `b`, position `k` in `[0, BLOCK_SIZE)`) |
|-----|---------------------|------------------------------------------------------------------------------|
| -1  | `Ht`                | `tile_id = tile_id_origin + idx * Wt + (b * BLOCK_SIZE + k)`                 |
| -2  | `Wt`                | `tile_id = tile_id_origin + (b * BLOCK_SIZE + k) * Wt + idx`                 |

The writer uses the identical decomposition (output shape == input shape).

### Block size selection

`BLOCK_SIZE` divides the reduce-direction tile count (`Wt` for dim=-1, `Ht` for dim=-2). Default selection: largest divisor of the reduce-dim tile count that is `≤ 8`. This matches the toy_variance pattern and stays well inside `DEST_AUTO_LIMIT = 4` (helper internally batches DST across `BLOCK_SIZE` tiles). Implementer may expose a `block_size` override but defaults must be deterministic.

`NUM_BLOCKS = reduce_dim_tiles / BLOCK_SIZE`. Both compile-time.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_grad_output` | 0 | `tile_size(float32)` = 4096 B | 2 (double-buffer for streaming) | float32 | reader | compute (`mul` in pass 1, `sub` in pass 2) | streamed per-tile, drained twice across passes |
| `cb_output` | 1 | `tile_size(float32)` = 4096 B | 2 (double-buffer for streaming) | float32 | reader | compute (`mul` in pass 1, `mul` in pass 2) | streamed per-tile, drained twice across passes |
| `cb_scaler` | 2 | `tile_size(bfloat16)` = 2048 B | 1 (one shared scaler tile, lifetime = whole program) | bfloat16 | reader (writes 1.0 once at startup via `calculate_and_prepare_reduce_scaler`) | compute (`accumulate_reduce_block`) | persistent — not popped between lanes |
| `cb_grad_input` | 16 | `tile_size(float32)` = 4096 B | 2 (double-buffer for streaming) | float32 | compute (`mul` in pass 2) | writer | streamed per-tile to DRAM |
| `cb_prod` | 24 | `tile_size(float32)` = 4096 B | `2 * BLOCK_SIZE` (full block, doubled for headroom — sequential helpers `mul` and `accumulate_reduce_block` cannot pipeline within compute since they share the math/pack threads) | float32 | compute (`mul` pass 1) | compute (`accumulate_reduce_block` pass 1) | reused per block within a lane |
| `cb_sum` | 25 | `tile_size(float32)` = 4096 B | 2 (sized for 1 tile + double-buffer; held with `WaitUpfrontNoPop` across pass-2 blocks then popped once per lane) | float32 | compute (`accumulate_reduce_block` writes 1 tile per lane after final block) | compute (`sub<SCALAR>` reads it `NUM_BLOCKS` times in pass 2 with no pop) | persists across pass 2 of a single lane; popped once at end of lane |
| `cb_centered` | 26 | `tile_size(float32)` = 4096 B | `2 * BLOCK_SIZE` (full block, doubled for headroom — sequential helpers `sub` and `mul` cannot pipeline) | float32 | compute (`sub<SCALAR>` pass 2) | compute (`mul` pass 2) | reused per block within a lane |

CB ranges: `core_ranges = all_cores` (the union returned by `split_work_to_cores`) for every CB.

### CB sync (push/wait counts must match)

| CB | Producer pushes per lane | Consumer waits per lane | Match |
|----|--------------------------|--------------------------|-------|
| `cb_grad_output` | reader: `2 * NUM_BLOCKS * BLOCK_SIZE` (one push per tile × 2 passes) | compute: `NUM_BLOCKS * BLOCK_SIZE` for pass-1 `mul` + `NUM_BLOCKS * BLOCK_SIZE` for pass-2 `sub` (per-tile wait/pop, helpers manage internally) | ✓ |
| `cb_output` | reader: `2 * NUM_BLOCKS * BLOCK_SIZE` | compute: `NUM_BLOCKS * BLOCK_SIZE` (pass-1 `mul`) + `NUM_BLOCKS * BLOCK_SIZE` (pass-2 `mul`) | ✓ |
| `cb_prod` | compute (`mul`): `NUM_BLOCKS * BLOCK_SIZE` | compute (`accumulate_reduce_block`): `NUM_BLOCKS * BLOCK_SIZE` | ✓ |
| `cb_sum` | compute (`accumulate_reduce_block`): 1 (final block of pass 1 pushes the lane's sum tile) | compute (`sub<SCALAR>` with `WaitUpfrontNoPop` waits 1 each block × `NUM_BLOCKS`) + 1 explicit `cb_pop_front(cb_sum, 1)` after pass 2 | ✓ (push count 1 = pop count 1; helper `WaitUpfrontNoPop` does not pop) |
| `cb_centered` | compute (`sub<SCALAR>`): `NUM_BLOCKS * BLOCK_SIZE` | compute (`mul`): `NUM_BLOCKS * BLOCK_SIZE` | ✓ |
| `cb_grad_input` | compute (`mul` pass 2): `NUM_BLOCKS * BLOCK_SIZE` | writer: `NUM_BLOCKS * BLOCK_SIZE` (one tile per wait/pop) | ✓ |
| `cb_scaler` | reader: 1 push at startup (whole program, not per lane) | compute: `accumulate_reduce_block` waits but never pops (kernel-lib reduce holds the scaler indefinitely) | ✓ (push 1 ≡ no pop ever) |

## API Mapping

Every mechanism below has an exact file:line reference.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|
| **Reader, startup (once per program)** | helper | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_id, PoolType, ReduceDim>()` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:100` | `<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>` for `dim=-1`; `<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_COL>` for `dim=-2`. `reduce_factor` defaults to `SUM_AND_MAX_REDUCE_FACTOR=1` since the scaler is 1.0 for SUM. | — | `cb_scaler` (push 1 tile) | Pool-type-aware overload (selects correct row-0 vs col-0 fill pattern). Must be the first thing the reader does. Tiles are tile-aligned per validation, so partial-scaler variants are NOT used. |
| **Reader, per lane** | raw_api | `cb_reserve_back` / `noc_async_read_tile` / `noc_async_read_barrier` / `cb_push_back` (paired across `cb_grad_output` and `cb_output`) | `tt_metal/hw/inc/dataflow_api.h` (standard NoC primitives) | — | DRAM (`grad_output`, `output` via `TensorAccessor`) | `cb_grad_output`, `cb_output` (1 tile each per inner iteration, in lockstep) | **Helpers considered and rejected:** No reader helper exists for "stream 2 input tensors in lockstep through 2 separate CBs". The helpers under `tilize_helpers_dataflow.hpp` cover RM→tile conversion (not relevant — input is already TILE_LAYOUT). `cb_helpers_dataflow.hpp` is for in-CB compute-thread coordination. The reader-side TensorAccessor + per-tile push loop is the canonical idiom for tiled-input streaming (see `ttnn/ttnn/operations/toy_binary_in_place/kernels/reader.cpp:24-51` and `ttnn/ttnn/operations/toy_variance/kernels/reader.cpp:54-70`). |
| **Compute, init** | raw_api | `compute_kernel_hw_startup(cb_grad_output, cb_scaler, cb_grad_input)` | required prologue per all `kernel_lib` helpers; documented at `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:29-31` and `binary_op_helpers.hpp:380` | passes the CBs whose data formats anchor unpacker/packer config | — | — | Must be called exactly once before any helper. Pass any representative input CB (`cb_grad_output`), the scaler CB (`cb_scaler`), and the output CB (`cb_grad_input`). Subsequent helpers reconfig as needed. |
| **Pass 1, per block — multiply** | helper | `compute_kernel_lib::mul()` (alias for `binary_op<ADD,...>` family) | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:303` | `<BroadcastDim::NONE, BinaryInputPolicy::WaitAndPopPerTile, BinaryInputPolicy::WaitAndPopPerTile, BinaryOutputPolicy::PerTile, BinaryDataFormatReconfig::INPUT_AND_OUTPUT, init=true>`. Both inputs streamed per-tile; output streamed per-tile. Block shape: `BinaryInputBlockShape::of(1, BLOCK_SIZE)` for `dim=-1`, `::of(BLOCK_SIZE, 1)` for `dim=-2`. | `cb_grad_output`, `cb_output` (each pops `rows*cols` tiles) | `cb_prod` (pushes `rows*cols` tiles) | Helper owns DEST acquire/commit/wait/release, CB wait/pop/reserve/push, and `binary_init`. No external CB ops needed. |
| **Pass 1, per block — accumulating reduce** | helper | `compute_kernel_lib::accumulate_reduce_block<PoolType, ReduceDim>()` | `ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp:53` | `<PoolType::SUM, ReduceDim::REDUCE_ROW>` for `dim=-1`; `<PoolType::SUM, ReduceDim::REDUCE_COL>` for `dim=-2`. Block shape: `ReduceInputBlockShape::of(1, BLOCK_SIZE, 1)` for `dim=-1`; `::of(BLOCK_SIZE, 1, 1)` for `dim=-2`. Args: `b` (current block index), `NUM_BLOCKS`, `partial = ReducePartialScaler::none()` (tile-aligned input). Default `WaitAndPopPerTile` input policy. | `cb_prod` (pops block tiles), `cb_scaler` (waits, never pops) | `cb_sum` (writes 1 tile after final block; intermediate blocks reload-and-re-emit via `Accumulate::at(cb_sum, b)`) | Helper bundles `Accumulate::at(cb_sum, b)` index-aware reload, last-block routing, and DEST/CB choreography. Caller owns popping `cb_sum` (deferred until end of pass 2 — see explicit `cb_pop_front` below). |
| **Pass 2, per block — broadcast subtract** | helper | `compute_kernel_lib::sub()` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:290` | `<BroadcastDim::SCALAR, BinaryInputPolicy::WaitAndPopPerTile, BinaryInputPolicy::WaitUpfrontNoPop, BinaryOutputPolicy::PerTile, BinaryDataFormatReconfig::INPUT_AND_OUTPUT, init=true>`. A is streamed per-tile; B (`cb_sum`) is held across all `NUM_BLOCKS` blocks and never popped by the helper. Block shape: same as the matching pass-1 mul. | `cb_grad_output` (pops `rows*cols` per block), `cb_sum` (waits 1, no pop) | `cb_centered` (pushes `rows*cols`) | SCALAR broadcast: helper consumes element [0,0] of the single `cb_sum` tile; `accumulate_reduce_block` SUM/REDUCE_ROW (or REDUCE_COL) places the lane's sum exactly at [0,0]. `WaitUpfrontNoPop` is essential — popping inside the per-block loop would discard the sum needed by later blocks. |
| **Pass 2, per block — final multiply** | helper | `compute_kernel_lib::mul()` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:303` | `<BroadcastDim::NONE, BinaryInputPolicy::WaitAndPopPerTile, BinaryInputPolicy::WaitAndPopPerTile, BinaryOutputPolicy::PerTile, BinaryDataFormatReconfig::INPUT_AND_OUTPUT, init=true>`. Block shape: same as previous mul. | `cb_output`, `cb_centered` | `cb_grad_input` | Streams every block tile to writer. |
| **Compute, end of lane — release sum** | raw_api | `cb_pop_front(cb_sum, 1)` | `tt_metal/hw/inc/cb_api.h` | — | `cb_sum` (pop 1) | — | **Helpers considered and rejected:** No helper "release a CB held by `WaitUpfrontNoPop`" exists. `transform_in_place` (`streaming_reduce_helpers.hpp:111`) does pop+pack but only as part of a 1-tile DST transform, which we don't need here — the sum has already been consumed by the per-block `sub`. A naked `cb_pop_front(cb_sum, 1)` matches the toy_variance precedent at `kernels/compute.cpp:106`. |
| **Writer, per lane** | raw_api | `cb_wait_front` / `noc_async_write_tile` / `noc_async_write_barrier` / `cb_pop_front` | `tt_metal/hw/inc/dataflow_api.h` | — | `cb_grad_input` | DRAM (`grad_input` via `TensorAccessor`) | **Helpers considered and rejected:** No helper for "drain a tiled CB to DRAM in tile_id order". `untilize_helpers_dataflow.hpp` is RM-output-only; output is TILE_LAYOUT here. The standard tiled writer loop matches `ttnn/ttnn/operations/toy_variance/kernels/writer.cpp:20-28`. |

## Compute Phases

Per lane on each core, the compute kernel executes the following sequence. `cb_state_after` is annotated only where state is non-obvious (persistent CBs, ownership transfers).

| # | Operation | Helper? | Input CB (state) | Output CB (state after) | CB State After |
|---|-----------|---------|------------------|--------------------------|----------------|
| 0 | `compute_kernel_hw_startup(cb_grad_output, cb_scaler, cb_grad_input)` | raw_api | — | — | (init only) |
| 1a | **Pass 1, block b** — `mul(cb_grad_output, cb_output, cb_prod, block_shape)` | helper | `cb_grad_output`, `cb_output`: `BLOCK_SIZE` tiles consumed | `cb_prod`: `BLOCK_SIZE` tiles | reader keeps refilling input CBs |
| 1b | **Pass 1, block b** — `accumulate_reduce_block<SUM, REDUCE_DIM>(cb_prod, cb_scaler, cb_sum, block_shape, b, NUM_BLOCKS)` | helper | `cb_prod`: `BLOCK_SIZE` tiles consumed; `cb_scaler`: 1 tile (no pop) | `cb_sum`: 1 tile written after the final block | After loop completes (b == NUM_BLOCKS-1), `cb_sum` holds the lane sum at `[0, 0]` of its single tile. |
| — | (loop 1a → 1b for `b in [0, NUM_BLOCKS)`) |  |  |  |  |
| 2a | **Pass 2, block b** — `sub<SCALAR, A=WaitAndPopPerTile, B=WaitUpfrontNoPop>(cb_grad_output, cb_sum, cb_centered, block_shape)` | helper | `cb_grad_output`: `BLOCK_SIZE` tiles consumed; `cb_sum`: 1 tile waited, NOT popped | `cb_centered`: `BLOCK_SIZE` tiles | `cb_sum` survives for the next block |
| 2b | **Pass 2, block b** — `mul(cb_output, cb_centered, cb_grad_input, block_shape)` | helper | `cb_output`: `BLOCK_SIZE` tiles consumed; `cb_centered`: `BLOCK_SIZE` tiles consumed | `cb_grad_input`: `BLOCK_SIZE` tiles | writer drains in parallel |
| — | (loop 2a → 2b for `b in [0, NUM_BLOCKS)`) |  |  |  |  |
| 3 | `cb_pop_front(cb_sum, 1)` | raw_api | `cb_sum`: 1 tile popped | — | `cb_sum` empty, ready for next lane |

The above sequence (steps 1a..3) repeats for each of the `num_lanes` lanes assigned to this core. `cb_scaler` is loaded once at program start (by the reader) and never popped, so the reduce helper's `cb_wait_front(cb_scaler, 1)` succeeds on every lane.

## Build Order

The implementer should bring the op up incrementally rather than implementing the full kernel and debugging end-to-end. Order:

| Stage | Goal | What to verify | DPRINT hints |
|-------|------|----------------|--------------|
| 1. **Data pipeline only** | Reader pushes `grad_output` and `output` tiles in lockstep through `cb_grad_output` / `cb_output`; writer drains `cb_grad_input`; compute is a passthrough that copies `cb_grad_output → cb_grad_input` (using `compute_kernel_lib::copy_tiles`). Test with `dim=-1`, single shape `(1, 1, 32, 32)`. | Output equals `grad_output` exactly. Confirms tile_id arithmetic and CB sync. | `DPRINT << "lane=" << lane << " pass=" << pass << " tile_id=" << tile_id << ENDL();` in reader. |
| 2. **Pass 1 — pointwise multiply only** | Compute does only `mul(cb_grad_output, cb_output, cb_grad_input, ...)` (using `cb_grad_input` as direct sink) and skips reduction; reader streams *one* pass instead of two; writer drains `Wt` tiles per lane. | Output == `grad_output * output` element-wise. Confirms `mul` helper wiring with two streamed inputs. | DPRINT first 4 elements of the tile in DST after `mul` to verify the product. |
| 3. **Pass 1 — full reduce** | Add `cb_prod` and `accumulate_reduce_block`; route output to `cb_grad_input` so the writer drains `1` tile per lane (debug only). Verify the lane sum matches `(grad_output * output).sum(dim)`. | The single output tile per lane has the correct sum at element [0,0]. | DPRINT `cb_sum`'s [0,0] after the final block. Use deterministic input (`torch.ones(...)`) to make the sum predictable: e.g. for `(1,1,32,64)` with all ones, sum should be 64.0. |
| 4. **Pass 2 — sub broadcast** | Restore reader's two-pass streaming. Compute does pass 1 (write to `cb_sum`) then per block: `sub<SCALAR>(cb_grad_output, cb_sum, cb_grad_input, ...)`, drain to writer. Verify output == `grad_output - sum(grad_output*output, dim)` broadcast. | Subtraction matches the torch reference. Confirms `WaitUpfrontNoPop` policy and SCALAR broadcast. | DPRINT one tile of `cb_centered` after the first sub block. |
| 5. **Pass 2 — full** | Add the second `mul(cb_output, cb_centered, cb_grad_input)` and the explicit `cb_pop_front(cb_sum, 1)` at end of lane. End-to-end correctness for `dim=-1` shape `(1, 1, 32, 32)`. | Output matches torch reference of the full formula. | None — measure via test. |
| 6. **Multi-block / multi-tile / multi-batch** | Bump shape to `(1, 1, 32, 256)` then `(2, 4, 64, 256)`. Verify multi-block reduction (via `BLOCK_SIZE < Wt`) and multi-lane operation. | Tolerance-bounded match across all parametrized shapes. | — |
| 7. **dim=-2 path** | Add the compile-time `DIM_IS_W` flag (or equivalent) and the alternate tile_id arithmetic + `REDUCE_COL` + transposed block shape. Verify against torch. | Tolerance-bounded match for `dim=-2` shapes. | — |
| 8. **Multi-core** | Wire `split_work_to_cores` and per-core RT args. Verify that each core handles the correct lane range. | Same correctness with multiple cores; runs faster on bigger shapes. | DPRINT `start_lane` and `num_lanes` per core. |

## Key Risks and Gotchas

| # | Risk | Mitigation |
|---|------|------------|
| 1 | `cb_sum` popped too early would leave subsequent pass-2 blocks reading stale memory. | Use `BinaryInputPolicy::WaitUpfrontNoPop` for the B input of `sub`; explicit `cb_pop_front(cb_sum, 1)` ONLY after pass 2's last block. Verified against toy_variance precedent (`kernels/compute.cpp:78–106`). |
| 2 | Sequential helpers `mul` and `accumulate_reduce_block` (and `sub`/`mul` in pass 2) cannot pipeline — they share the unpack/math/pack threads. Sizing `cb_prod` and `cb_centered` to less than a full block deadlocks. | Size both intermediate CBs to `2 * BLOCK_SIZE` pages (full block + headroom), per `ttnn/ttnn/operations/toy_variance/kernels/compute.cpp` precedent. |
| 3 | Scaler CB content/format mismatch with reduce LLK requirements (e.g. wrong fill pattern for the chosen `PoolType`/`ReduceDim`). | Use the **pool-type-aware** overload `calculate_and_prepare_reduce_scaler<cb, PoolType::SUM, ReduceDim::REDUCE_ROW or REDUCE_COL>()`. The legacy single-template overload is forbidden — different `(PoolType, ReduceDim)` combinations need different fill patterns (row-0 vs col-0). |
| 4 | `fp32_dest_acc_en=True` halves DEST capacity from 8→4 tiles in half-sync. Hand-rolled DEST loops would overrun. | Use kernel-lib helpers exclusively for compute. Helpers reference `DEST_AUTO_LIMIT` (`ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:102`) which evaluates to 4 under this config. No raw `tile_regs_acquire` calls. |
| 5 | dim=-1 vs dim=-2 use different `ReduceDim`, different `BinaryInputBlockShape` orientation, and different reader tile_id arithmetic. Mixing them up silently produces garbage. | Single compile-time flag `DIM_IS_W` (true for dim=-1, false for dim=-2) selects: `ReduceDim`, block_shape rows/cols, and reader tile-id formula. All three flip together — implementer must wire all three from the same flag. See **Reduce Direction Verification** table below. |
| 6 | Reader pushing all `cb_grad_output` tiles before any `cb_output` tiles deadlocks (compute's `mul` requires both per tile). | Reader pushes tiles in lockstep: in the inner loop, read+push one `dy` tile then read+push one `y` tile to the matching CB. Same per-tile cadence as in toy_binary_in_place reader (`kernels/reader.cpp:30-50` modified to alternate). |
| 7 | Two-pass requires reading the inputs twice. Skipping the second read would produce only the partial result `dy*y`, not the centered gradient. | Reader has an explicit `for pass in [0, 2)` loop wrapping the per-lane streaming (toy_variance pattern, `kernels/reader.cpp:57-70`). |
| 8 | Scaler tile is bfloat16 while data CBs are float32. Without correct format reconfig, the helper would unpack the scaler with the wrong format. | `BinaryDataFormatReconfig::INPUT_AND_OUTPUT` (default) on every binary helper handles this. Reduce helpers default to `INPUT_AND_OUTPUT` reconfig. |

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| Pass 1, mul | `mul(cb_grad_output, cb_output, cb_prod, of(rows, cols))` | All `rows*cols` tiles fully populated (`grad_output` block) | All `rows*cols` tiles fully populated (`output` block) | `NONE` |
| Pass 2, sub | `sub<SCALAR>(cb_grad_output, cb_sum, cb_centered, of(rows, cols))` | All `rows*cols` tiles fully populated (`grad_output` block, second pass) | 1 tile, scalar value at element [0,0] (placed there by `accumulate_reduce_block` SUM/REDUCE_ROW or SUM/REDUCE_COL — both pool-type-aware variants land the result at face[0,0] = tile[0,0]) | `SCALAR` |
| Pass 2, mul | `mul(cb_output, cb_centered, cb_grad_input, of(rows, cols))` | All `rows*cols` tiles fully populated (`output` block, second pass) | All `rows*cols` tiles (output of previous sub) | `NONE` |

## Reduce Direction Verification

| Logical Dim | Tile ReduceDim | Output Valid Region (cb_sum) | BroadcastDim (sub) | ReduceInputBlockShape | BinaryInputBlockShape | Reader tile_id formula |
|-------------|----------------|------------------------------|--------------------|-----------------------|------------------------|------------------------|
| `-1` (W)    | `REDUCE_ROW`   | Single tile, value at face [0,0] (col-0 fill) | `SCALAR`           | `of(1, BLOCK_SIZE, 1)`| `of(1, BLOCK_SIZE)`   | `nc * Ht * Wt + ht * Wt + (b * BLOCK_SIZE + k)` |
| `-2` (H)    | `REDUCE_COL`   | Single tile, value at face [0,0] (row-0 fill) | `SCALAR`           | `of(BLOCK_SIZE, 1, 1)`| `of(BLOCK_SIZE, 1)`   | `nc * Ht * Wt + (b * BLOCK_SIZE + k) * Wt + wt` |

Both REDUCE_ROW (SUM/AVG normally use matmul-path col-0 fill) and REDUCE_COL (reduce-path row-0 fill) place the scalar at tile element [0,0] — `SCALAR` broadcast picks up [0,0] correctly in either case. The `compute_kernel_lib::reduce` dispatcher routes REDUCE_ROW SUM through the appropriate path automatically; the scaler-prep helper uses the pool-type-aware overload to match.
