# Operation Design: layer_norm_rm

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (fused: tilize → reduce → broadcast-sub → square → reduce → eps/rsqrt → broadcast-mul → optional gamma/beta → untilize) |
| Goal | Row-wise (last-dim) layer normalization on a ROW_MAJOR_LAYOUT float32 tensor. The kernels accept RM input directly (in-kernel tilize) and produce RM output (in-kernel untilize); no host-side layout conversion. Optional affine `gamma` (scale) and `beta` (shift) of shape `(1,1,1,W)` are applied per W column. |
| Math | `y[..., h, w] = ((x[..., h, w] - mean(x[..., h, :])) / sqrt(var(x[..., h, :]) + epsilon)) * (gamma[w] if gamma else 1) + (beta[w] if beta else 0)`, where `mean` and `var` (population variance) are over the last axis. |
| Mode | Derivative — extends the TILE-layout layer_norm pipeline by absorbing the host-side tilize/untilize into the kernel, and chunks the W axis to bound L1 footprint independent of W. |
| References | `ttnn/ttnn/operations/toy_variance/` (streaming variance via `accumulate_reduce_block` + `sub<COL>` + `square_in_place`), `ttnn/ttnn/operations/toy_tilize_untilize/` (RM-in / RM-out via `dataflow_kernel_lib::read_sticks_for_tilize` + `compute_kernel_lib::tilize` + `compute_kernel_lib::untilize` + `dataflow_kernel_lib::write_sticks_after_untilize`), `ttnn/ttnn/operations/softmax/softmax_program_descriptor.py` (chunked CB sizing pattern, fp32 scaler decision, `ttnn.split_work_to_cores`), `tt_metal/third_party/tt_ops_code_gen/eval/golden_tests/layer_norm_rm/feature_spec.py` (test universe). |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | float32, ROW_MAJOR_LAYOUT, on-device, rank ≥ 2, `shape[-2] % 32 == 0`, `shape[-1] % 32 == 0` | — | tensor (buffer address via RT) |
| `gamma` | `ttnn.Tensor \| None` | no (positional) | If not None: float32, ROW_MAJOR_LAYOUT, on-device, shape `(1,1,1,W)` with `W == input_tensor.shape[-1]` | `None` | tensor (HAS_GAMMA CT flag + buffer address RT) |
| `beta` | `ttnn.Tensor \| None` | no (positional) | If not None: float32, ROW_MAJOR_LAYOUT, on-device, shape `(1,1,1,W)` with `W == input_tensor.shape[-1]` | `None` | tensor (HAS_BETA CT flag + buffer address RT) |
| `epsilon` | `float` | no (kw-only) | finite, positive | `1e-5` | RT (uint32 fp32 bit-pattern, compute kernel) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor \| None` | no (kw-only) | `None` (entry installs Phase 0 default), or explicit with `math_fidelity=HiFi4`, `fp32_dest_acc_en=True`, `math_approx_mode=False` | `None` | passed to `ttnn.KernelDescriptor.config` for the compute kernel |

Notes:
- Function signature exactly: `layer_norm(input_tensor, gamma=None, beta=None, *, epsilon=1e-5, compute_kernel_config=None)`. Import path: `from ttnn.operations.layer_norm_rm import layer_norm`.
- `validate()` rejects everything outside the table above with `NotImplementedError` (per the registry-model contract used by the `softmax` and other modern ops).
- The Phase 0 default config is installed when `compute_kernel_config is None`; any user-supplied config that does not equal the Phase 0 contract (HiFi4 + fp32_dest_acc_en=True + math_approx_mode=False) is rejected.

## Tensors

### Input — `input_tensor`

| Property | Requirement |
|----------|-------------|
| Shape | rank ≥ 2; treated as `(NC..., H, W)` where `NC = ∏ shape[:-2]`, `H = shape[-2]`, `W = shape[-1]` |
| Dtype | float32 |
| Layout | ROW_MAJOR_LAYOUT |
| Memory | interleaved (DRAM or L1; planner does not restrict). One row of W elements = 1 stick = 1 page (`page_size = W * 4` bytes). |
| Alignment | `H % 32 == 0` and `W % 32 == 0` (so `NC * H` is a multiple of 32 and the strip-of-32 work unit is exact). |

### Input — `gamma` (optional)

| Property | Requirement |
|----------|-------------|
| Shape | `(1, 1, 1, W)` exactly (4D, length-1 in all leading axes). |
| Dtype | float32 |
| Layout | ROW_MAJOR_LAYOUT |
| Memory | interleaved. 1 stick total, `page_size = W * 4` bytes. |

### Input — `beta` (optional)

Same as `gamma` row above (independent CT/RT flag, independent CB pair).

### Output

| Property | Value |
|----------|-------|
| Shape | identical to `input_tensor.shape` |
| Dtype | float32 |
| Layout | ROW_MAJOR_LAYOUT |
| Memory | DRAM interleaved (default) or whatever the entry point's `memory_config` resolves to. |
| Stick layout | one stick per row, `page_size = W * 4` bytes, `num_pages = NC * H`. |

## Dataflow Strategy

The op is single-Tensix per work unit (each strip lives on exactly one core, no inter-core fan-out for the normalization itself). At the Tensix level the contract is the standard reader → compute → writer pipeline via CBs.

| Stage | Producer | Consumer | Path | Format at this stage |
|-------|----------|----------|------|----------------------|
| Read input rows | DRAM (interleaved) | Reader (NCRISC) | `noc_async_read` via `TensorAccessor` → `dataflow_kernel_lib::read_sticks_for_tilize<TILE>` | RM sticks (32 sticks × `chunk_bytes` per chunk push) |
| Tilize | Compute (TRISCs) | Compute (TRISCs) | `compute_kernel_lib::tilize<BLOCK_SIZE, cb_input_rm, cb_tilized_x>(1)` | Tile (BLOCK_SIZE tiles per chunk) |
| Reduce → mean | Compute | Compute | `compute_kernel_lib::accumulate_reduce_block<SUM, REDUCE_ROW>(cb_tilized_x, cb_scaler, cb_mean, …)` | 1 mean tile (per-row mean in column 0) |
| Subtract mean (Pass 2 / Pass 3) | Compute | Compute | `compute_kernel_lib::sub<BroadcastDim::COL, WaitAndPopPerTile, WaitUpfrontNoPop>` | `(x − mean)` tiles in `cb_centered` |
| Square + reduce → variance | Compute | Compute | `compute_kernel_lib::square_in_place` + `accumulate_reduce_block<SUM, REDUCE_ROW>` | 1 variance tile |
| eps + rsqrt → inv_std | Compute | Compute | `compute_kernel_lib::transform_in_place(cb_inv_std, lambda: add_unary_tile(eps) + rsqrt_tile)` | 1 `1/sqrt(var+eps)` tile |
| Normalize (Pass 3) | Compute | Compute | `mul_in_place<COL>(cb_centered, cb_inv_std)` | normalized tiles in `cb_centered` |
| Apply gamma (optional, Pass 3) | Reader → Compute | Compute | Reader tilizes `gamma` chunk; `mul_in_place<ROW>(cb_centered, cb_gamma_tilized)` | scaled tiles in `cb_centered` |
| Apply beta (optional, Pass 3) | Reader → Compute | Compute | Reader tilizes `beta` chunk; `add_in_place<ROW>(cb_centered, cb_beta_tilized)` | shifted tiles in `cb_centered` |
| Untilize | Compute | Compute → Writer | `compute_kernel_lib::untilize<BLOCK_SIZE, cb_centered, cb_output_rm>(1)` | RM sticks (tile-sized pages on `cb_output_rm`) |
| Write output rows | Writer (BRISC) | DRAM | `dataflow_kernel_lib::write_sticks_after_untilize<cb_output_rm>` | RM sticks |

**Reader passes per strip.** The reader streams the input strip THREE times: Pass A (mean), Pass B (variance), Pass C (output). Each pass iterates `NUM_BLOCKS` width-chunks, reading 32 sticks × `chunk_bytes` per chunk via `read_sticks_for_tilize<TILE>` with `byte_offset_within_page = chunk_id * chunk_bytes`. Three reader passes (re-reading the DRAM tensor 3×) is the deliberate Phase-0 trade-off for L1-bounded CBs: keeping the tilized input persistent across mean+variance+output would require `Wt` fp32 tiles in L1, which overflows the 1.5 MB budget at the wide-W shapes in the test universe (W=8192 ⇒ `Wt`=256 ⇒ 1 MB just for `cb_tilized_x`). The chunked design caps every CB at `BLOCK_SIZE` tiles regardless of W.

**Gamma/beta reads.** Reader makes ONE pass through gamma (during Pass C) and ONE pass through beta (during Pass C), each in width-chunks of `chunk_bytes`. Gamma and beta are 1-row tensors so the chunked read uses `read_sticks_for_tilize<ROW>` with `total_num_rows=1` (pairing with the asymmetric `tilize<BLOCK_SIZE, …>(num_blocks=1, total_input_pages=1)` overload). Reading gamma/beta from DRAM once per strip is acceptable (small constant data; gamma/beta are tiny relative to the input strip and the NCRISC overlaps the read with compute on the TRISCs).

**No inter-Tensix communication.** Each strip is independent; there is no multicast, semaphore, or ring topology. Work distribution is purely the `ttnn.split_work_to_cores` static partition over the strip index space.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | One strip = 32 logical input rows (= 1 tile-row tall × `Wt` tiles wide). Each strip's W rows are normalized independently. |
| Total work units | `num_strips = NC * Ht` where `Ht = H / 32`. |
| Grid | `device.compute_with_storage_grid_size()` (typically 8 × 8 on Wormhole, the full compute_with_storage grid). |
| Partition | `ttnn.split_work_to_cores(grid, num_strips)` returns `(num_cores_total, all_cores, core_group_1, core_group_2, strips_per_core_group_1, strips_per_core_group_2)`. |
| Per-core work | `strips_per_core` consecutive strips, with the per-core start index assigned in (x, y)-row-major order across each core_group's `CoreRangeSet`. |
| Remainder | `split_work_to_cores` returns two groups: `core_group_1` gets `strips_per_core_group_1 = ceil(num_strips / num_cores)`, `core_group_2` gets `strips_per_core_group_2 = strips_per_core_group_1 − 1`. Cores in `core_group_1` therefore process one extra strip when the division is not exact. |
| Per-strip math | Strip `s` operates on input rows `[s*32, (s+1)*32)` and writes output rows `[s*32, (s+1)*32)`. No cross-strip dependency. |
| Chunking inside a strip | `NUM_BLOCKS = Wt / BLOCK_SIZE`, where `BLOCK_SIZE = largest divisor of Wt that is ≤ BLOCK_SIZE_CAP=8`. With `fp32_dest_acc_en=True` half-sync DEST cap is 4 tiles, so `BLOCK_SIZE ≤ 8` gives the helpers enough headroom (the helpers internally chunk DST batches via `DEST_AUTO_LIMIT`). |

## Circular Buffers

CB index convention: 0–7 inputs (reader-produced RM streams), 8–15 special (scaler), 16–23 output (writer-consumed RM stream), 24–31 intermediate (compute-only).

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_rm` | 0 | `tile_size(fp32) = 4096` B | `2 * BLOCK_SIZE` | fp32 | Reader (`read_sticks_for_tilize<TILE>`, pushes `BLOCK_SIZE` pages per 32-stick block) | Compute (`tilize<BLOCK_SIZE>(1)`, waits & pops `BLOCK_SIZE` pages per chunk) | Per-chunk streaming across all 3 passes per strip × all strips on this core. Double-buffered for reader/compute pipelining. |
| `cb_gamma_rm` | 1 | `padded_row_bytes(chunk) = BLOCK_SIZE * 32 * 4 = BLOCK_SIZE * 128` B | `2` | fp32 | Reader (`read_sticks_for_tilize<ROW>`, 1 page per chunk = 1 row × `chunk_bytes`) | Compute (`tilize<BLOCK_SIZE>(1, /*total_input_pages=*/1)`, waits 1 page, pops 1) | Allocated only when `HAS_GAMMA=1`. Per-chunk during Pass C. |
| `cb_beta_rm` | 2 | `padded_row_bytes(chunk) = BLOCK_SIZE * 128` B | `2` | fp32 | Reader (same as `cb_gamma_rm`) | Compute (same as `cb_gamma_rm`) | Allocated only when `HAS_BETA=1`. Per-chunk during Pass C. |
| `cb_scaler` | 8 | `tile_size(fp32) = 4096` B | `1` | fp32 | Reader (`dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f / W)` once at kernel start) | Compute (every `accumulate_reduce_block<SUM, REDUCE_ROW>` call, `WaitAndPopPerTile` internally never pops the scaler — `reduce<>` issues `cb_wait_front` only) | Pushed once at startup, persists for the full kernel. fp32 chosen to preserve precision for fp32 input + fp32 dest accumulator (mirrors the rationale in `softmax_program_descriptor.py:160-167`). |
| `cb_output_rm` | 16 | `tile_size(fp32) = 4096` B | `2 * BLOCK_SIZE` | fp32 | Compute (`untilize<BLOCK_SIZE>(1)` pushes `BLOCK_SIZE` tile pages per chunk) | Writer (`write_sticks_after_untilize<cb_output_rm>`, waits & pops `BLOCK_SIZE` pages per chunk) | Per-chunk streaming, double-buffered for compute/writer pipelining. |
| `cb_tilized_x` | 24 | `tile_size(fp32) = 4096` B | `BLOCK_SIZE` | fp32 | Compute (`tilize` pushes `BLOCK_SIZE` tile pages per chunk) | Compute (the next helper in the same chunk — `accumulate_reduce_block` in Pass A; `sub<COL>` in Pass B; `sub<COL>` in Pass C) | Sequential within compute → sized to the full block (`BLOCK_SIZE` tiles), NOT double-buffered. Empty between chunks. |
| `cb_centered` | 25 | `tile_size(fp32) = 4096` B | `BLOCK_SIZE` | fp32 | Compute (`sub<COL>` pushes `BLOCK_SIZE` tiles) | Compute (`square_in_place` in Pass B; `mul_in_place<COL>` + optional `mul_in_place<ROW>` + optional `add_in_place<ROW>` + `untilize` in Pass C) | Sequential intermediate. Used in Pass B (centered → squared → reduced) and in Pass C (centered → normalized → optional gamma/beta → untilized). |
| `cb_mean` | 26 | `tile_size(fp32) = 4096` B | `1` | fp32 | Compute (`accumulate_reduce_block` writes the final mean tile on the last block of Pass A) | Compute (`sub<COL>` in Pass B and Pass C, with `BinaryInputPolicy::WaitUpfrontNoPop` for input B; explicit `cb_pop_front(cb_mean, 1)` after Pass C) | Persists across Pass A → Pass B → Pass C of one strip. |
| `cb_inv_std` | 27 | `tile_size(fp32) = 4096` B | `1` | fp32 | Compute (`accumulate_reduce_block` of Pass B writes variance; `transform_in_place` overwrites with `1/sqrt(variance + eps)`) | Compute (`mul_in_place<COL>` in Pass C, `WaitUpfrontNoPop` for input B; explicit `cb_pop_front(cb_inv_std, 1)` after Pass C) | Persists across Pass B (variance accumulation) → eps+rsqrt transform → Pass C (multiply). |
| `cb_gamma_tilized` | 28 | `tile_size(fp32) = 4096` B | `BLOCK_SIZE` | fp32 | Compute (`tilize<BLOCK_SIZE, cb_gamma_rm, cb_gamma_tilized>(1, 1)`) | Compute (`mul_in_place<ROW>(cb_centered, cb_gamma_tilized)`, `WaitAndPopPerTile` input B) | Allocated only when `HAS_GAMMA=1`. Per-chunk in Pass C — emptied after the in-place multiply. |
| `cb_beta_tilized` | 29 | `tile_size(fp32) = 4096` B | `BLOCK_SIZE` | fp32 | Compute (`tilize<BLOCK_SIZE, cb_beta_rm, cb_beta_tilized>(1, 1)`) | Compute (`add_in_place<ROW>(cb_centered, cb_beta_tilized)`, `WaitAndPopPerTile` input B) | Allocated only when `HAS_BETA=1`. Per-chunk in Pass C. |

**CB sync verification** (push count = wait count for every CB):

| CB | Producer push count per chunk | Consumer wait count per chunk | Match? |
|----|-------------------------------|-------------------------------|--------|
| `cb_input_rm` | Reader pushes `BLOCK_SIZE` pages | `tilize<BLOCK_SIZE>(1)` waits/pops `BLOCK_SIZE` pages | ✅ |
| `cb_gamma_rm` | Reader pushes 1 page | `tilize<BLOCK_SIZE, …>(1, 1)` waits/pops 1 page | ✅ |
| `cb_beta_rm` | Reader pushes 1 page | `tilize<BLOCK_SIZE, …>(1, 1)` waits/pops 1 page | ✅ |
| `cb_scaler` | Reader pushes 1 page once at startup | `accumulate_reduce_block` waits 1 page (no pop — held across all calls) | ✅ (final explicit `cb_pop_front(cb_scaler, 1)` at end of compute kernel) |
| `cb_output_rm` | `untilize<BLOCK_SIZE>(1)` pushes `BLOCK_SIZE` pages | Writer waits/pops `BLOCK_SIZE` pages | ✅ |
| `cb_tilized_x` | `tilize` pushes `BLOCK_SIZE` | Next helper waits/pops `BLOCK_SIZE` | ✅ |
| `cb_centered` | `sub<COL>` pushes `BLOCK_SIZE`; in-place helpers preserve the count | `untilize` (Pass C) or `accumulate_reduce_block` (Pass B) waits/pops `BLOCK_SIZE` | ✅ |
| `cb_mean` | `accumulate_reduce_block` push of 1 final tile | `sub<COL>` waits 1 (no pop, B policy `WaitUpfrontNoPop`); final explicit `cb_pop_front(cb_mean, 1)` | ✅ |
| `cb_inv_std` | `accumulate_reduce_block` pushes 1 tile (variance) + `transform_in_place` re-pushes 1 tile (inv_std) | `mul_in_place<COL>` waits 1 (no pop); final explicit `cb_pop_front(cb_inv_std, 1)` | ✅ |
| `cb_gamma_tilized` | `tilize` pushes `BLOCK_SIZE` per chunk | `mul_in_place<ROW>` waits/pops `BLOCK_SIZE` per chunk | ✅ |
| `cb_beta_tilized` | `tilize` pushes `BLOCK_SIZE` per chunk | `add_in_place<ROW>` waits/pops `BLOCK_SIZE` per chunk | ✅ |

## API Mapping

Every mechanism — helper or raw — has an exact file:line reference. All file paths below are absolute under the repo root `/localdev/dnijemcevic/tt-metal/`.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB (semantic name) | Output CB (semantic name) | Requirements |
|-------|------|----------|-----------|------------------------|--------------------------|---------------------------|--------------|
| Reader prelude — scaler | helper | `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f / W)` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:65-67` | `cb_id = cb_scaler (8)`, `pool_type = SUM`, `reduce_dim = REDUCE_ROW`, `compute_uses_reduce_tile = false` (default). Runtime arg: `scaler_f = 1.0f / W` (caller-provided because `calculate_and_prepare_reduce_scaler` only emits 1.0 for SUM — see `reduce_helpers_dataflow.hpp:81-82`). | — | `cb_scaler` | Called ONCE per core at the top of the reader. Pool-type-aware overload (pool + reduce dim template args), satisfying the planner checklist. |
| Reader — input streaming | helper | `dataflow_kernel_lib::read_sticks_for_tilize<cb_input_rm, TilizeGranularity::TILE>(accessor, /*total_num_rows=*/32, /*row_bytes=*/chunk_bytes, /*start_page=*/strip_id*32, /*byte_offset_within_page=*/chunk_id*chunk_bytes)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:87-93` (declaration); `tilize_helpers_dataflow.inl:96-128` (TILE-mode body) | `cb_id = cb_input_rm (0)`, `granularity = TILE` (32 sticks per push, page = tile_size). | DRAM (via `TensorAccessor`) | `cb_input_rm` | Called inside a `for strip in strips_per_core` × `for pass in 3` × `for chunk in NUM_BLOCKS` loop. The helper itself owns the `cb_reserve_back` / `noc_async_read` / `cb_push_back` cycle. |
| Reader — gamma streaming (if `HAS_GAMMA`) | helper | `dataflow_kernel_lib::read_sticks_for_tilize<cb_gamma_rm, TilizeGranularity::ROW>(gamma_accessor, /*total_num_rows=*/1, /*row_bytes=*/chunk_bytes, /*start_page=*/0, /*byte_offset_within_page=*/chunk_id*chunk_bytes)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:87-93` (declaration); `tilize_helpers_dataflow.inl:129-158` (ROW-mode body) | `cb_id = cb_gamma_rm (1)`, `granularity = ROW`. | DRAM (via `gamma_accessor`) | `cb_gamma_rm` | Called inside `for chunk in NUM_BLOCKS` of Pass C only. ROW granularity is the natural fit because gamma is a single row; the asymmetric `tilize<…>(1, 1)` consumes exactly 1 row-sized page. |
| Reader — beta streaming (if `HAS_BETA`) | helper | `dataflow_kernel_lib::read_sticks_for_tilize<cb_beta_rm, TilizeGranularity::ROW>(beta_accessor, 1, chunk_bytes, 0, chunk_id*chunk_bytes)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:87-93` | Same as gamma row but with `cb_beta_rm`. | DRAM | `cb_beta_rm` | Called inside `for chunk in NUM_BLOCKS` of Pass C only. |
| Compute — hardware init | helper | `compute_kernel_hw_startup(cb_input_rm, cb_scaler, cb_centered)` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:29-33` (prerequisite documented); LLK declaration is in the per-arch `compute_kernel_api/common.h`. | 3-arg form: srcA, srcB, dst — pick `cb_input_rm` for srcA so the first `tilize` does not need a `UnpackReconfigure`, `cb_scaler` for srcB (used by every reduce), and `cb_centered` for the packer (the most-written intermediate). All subsequent helpers reconfigure as needed via their `reconfig_mode` template params (default `INPUT_AND_OUTPUT`). | — | — | Called EXACTLY ONCE at the top of the compute kernel. Mid-kernel re-init is forbidden per the helper docstring. |
| Compute — Pass A tilize | helper | `compute_kernel_lib::tilize<BLOCK_SIZE, cb_input_rm, cb_tilized_x>(/*num_blocks=*/1)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:178-187` (declaration); `tilize_helpers.hpp:131-176` (examples / behavior) | Template: `block_width_tiles = BLOCK_SIZE`, `input_cb = cb_input_rm (0)`, `output_cb = cb_tilized_x (24)`, `init_uninit_mode = InitAndUninit` (default), `wait_mode = WaitBlock` (default), `reconfig_mode = UnpackAndPackReconfigure` (default), `fp32_mode = Fast` (default — correct for fp32 input flowing into FPU reduce per the Fp32Mode docstring at `tilize_helpers.hpp:64-72`). Runtime: `num_blocks = 1`. | `cb_input_rm` (BLOCK_SIZE pages of tile_size each) | `cb_tilized_x` (BLOCK_SIZE tiles) | Called inside the chunk loop of Pass A. |
| Compute — Pass A reduce (mean) | helper | `compute_kernel_lib::accumulate_reduce_block<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_tilized_x, cb_scaler, cb_mean, ReduceInputBlockShape::of(1, BLOCK_SIZE, 1), b, NUM_BLOCKS, ReducePartialScaler::none())` | `ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp:47-61` | Template: `pool = SUM`, `rdim = REDUCE_ROW`, `in_policy = WaitAndPopPerTile` (default), `reconfig_mode = INPUT_AND_OUTPUT` (default). Runtime: `b = chunk_id`, `num_blocks = NUM_BLOCKS`, `partial = none()` (Phase 0 is tile-aligned). The scaler embeds `1/W`, so SUM produces the mean directly. | `cb_tilized_x` (BLOCK_SIZE tiles, popped per tile) + `cb_scaler` (1 tile, never popped) | `cb_mean` (1 tile, written on the LAST chunk; intermediate chunks accumulate inside DST) | The helper owns the per-block Accumulate::at routing — on the last block it packs the mean tile out, on earlier blocks it keeps the running sum in DST. |
| Compute — Pass B tilize | helper | `compute_kernel_lib::tilize<BLOCK_SIZE, cb_input_rm, cb_tilized_x>(1)` | `tilize_helpers.hpp:178-187` | Same as Pass A tilize. | `cb_input_rm` | `cb_tilized_x` | Re-tilize the same input strip (reader Pass B re-streamed it). |
| Compute — Pass B subtract mean | helper | `compute_kernel_lib::sub<BroadcastDim::COL, BinaryInputPolicy::WaitAndPopPerTile, BinaryInputPolicy::WaitUpfrontNoPop>(cb_tilized_x, cb_mean, cb_centered, BinaryInputBlockShape::of(1, BLOCK_SIZE))` | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:282-293` | Template: `bcast_dim = COL` (broadcast B's column 0 across the W direction — `cb_mean` is a `[Ht=1, Wt=1]` tile holding per-row means in column 0), `input_a_policy = WaitAndPopPerTile` (cb_tilized_x streams), `input_b_policy = WaitUpfrontNoPop` (cb_mean persists across chunks), `output_policy = PerTile` (default), `reconfig = INPUT_AND_OUTPUT` (default), `init = true` (default). | `cb_tilized_x` (popped) + `cb_mean` (not popped, B policy `WaitUpfrontNoPop`) | `cb_centered` (BLOCK_SIZE tiles) | Helper handles all binary_init / acquire / commit / wait / pack / release. |
| Compute — Pass B square | helper | `compute_kernel_lib::square_in_place(cb_centered, BinaryInputBlockShape::of(1, BLOCK_SIZE))` | `binary_op_helpers.hpp:464-469` | Template: `reconfig = INPUT_AND_OUTPUT` (default), `init = true` (default). | `cb_centered` (BLOCK_SIZE tiles, in-place) | `cb_centered` (BLOCK_SIZE tiles, in-place) | Uses the canonical `cb_a` in-place pattern (wait → pop → reserve → pack → push per tile). cb_centered must be exclusively compute-owned (yes — it lives in CB index 25, the intermediate range, and only compute touches it). |
| Compute — Pass B accumulate variance | helper | `compute_kernel_lib::accumulate_reduce_block<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_centered, cb_scaler, cb_inv_std, ReduceInputBlockShape::of(1, BLOCK_SIZE, 1), b, NUM_BLOCKS, ReducePartialScaler::none())` | `streaming_reduce_helpers.hpp:47-61` | Same as Pass A reduce but reading from `cb_centered` and writing into `cb_inv_std` (we reuse this CB slot for variance during Pass B; `transform_in_place` overwrites it with `1/sqrt(var+eps)` immediately after). | `cb_centered` (popped) + `cb_scaler` | `cb_inv_std` (1 tile of variance on the last chunk) | The output is `mean((x − mean)^2) = population variance`. With `1/W` already in the scaler, SUM produces variance directly. |
| Compute — eps + rsqrt transform | helper | `compute_kernel_lib::transform_in_place(cb_inv_std, [epsilon_bits](uint32_t dst) { binop_with_scalar_tile_init(); add_unary_tile(dst, epsilon_bits); rsqrt_tile_init(); rsqrt_tile(dst); })` | `ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp:110-111` (declaration); `streaming_reduce_helpers.hpp:94-109` (semantics docstring) | Runtime: `cb = cb_inv_std`, `t` = lambda capturing `epsilon_bits` (fp32 bit-pattern of `epsilon`, passed via compute-kernel RT arg 1). | `cb_inv_std` (1 tile of variance, popped and re-pushed) | `cb_inv_std` (1 tile of `1/sqrt(var+eps)`) | Lambda body uses raw `add_unary_tile` (`tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:28-30`) for the eps add and raw `rsqrt_tile` (`tt_metal/hw/inc/api/compute/eltwise_unary/rsqrt.h`) for the inverse-sqrt. **Helpers considered and rejected for the lambda body:** the `sfpu_chain` / `sfpu_pipeline` machinery in `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:1140-1180` is designed for full pipeline iteration (it owns the CB wait/reserve/push cycle and DST acquire/commit/release lifecycle — see the `sfpu_pipeline` examples in `sfpu_helpers.hpp:151-172`). The eps-add-then-rsqrt step here is two LLK ops on a *single tile already in DST*, inside the DST scope owned by `transform_in_place`. Wrapping it in `sfpu_chain`/`sfpu_pipeline` would re-acquire DST and re-orchestrate CB wait/pop/reserve/push around a tile the surrounding helper has already accounted for — i.e. the helper is structurally a worse fit than the in-DST raw calls. `transform_in_place`'s docstring (`streaming_reduce_helpers.hpp:104-107`) explicitly invites this pattern: "`t` is a callable taking a single `uint32_t dst_idx`. It can issue any number of in-DST init+op pairs (e.g. `rsqrt_tile_init + rsqrt_tile`, or a chain like `mul_unary_tile, add_unary_tile, rsqrt_tile`)." This is the documented usage. |
| Compute — Pass C tilize input | helper | `compute_kernel_lib::tilize<BLOCK_SIZE, cb_input_rm, cb_tilized_x>(1)` | `tilize_helpers.hpp:178-187` | Same as Pass A / Pass B. | `cb_input_rm` | `cb_tilized_x` | Third re-tilize of the strip. |
| Compute — Pass C tilize gamma (if `HAS_GAMMA`) | helper | `compute_kernel_lib::tilize<BLOCK_SIZE, cb_gamma_rm, cb_gamma_tilized>(/*num_blocks=*/1, /*total_input_pages=*/1)` | `tilize_helpers.hpp:178-187` (asymmetric overload demonstrated at `tilize_helpers.hpp:141-145`) | Template: same defaults. Runtime: `num_blocks = 1`, `total_input_pages = 1` (one row of gamma per chunk → asymmetric mode). Only row 0 of each output tile carries gamma; rows 1–31 are uninitialised (irrelevant for `BroadcastDim::ROW` which only reads row 0). | `cb_gamma_rm` (1 page) | `cb_gamma_tilized` (BLOCK_SIZE tiles) | |
| Compute — Pass C tilize beta (if `HAS_BETA`) | helper | `compute_kernel_lib::tilize<BLOCK_SIZE, cb_beta_rm, cb_beta_tilized>(1, 1)` | `tilize_helpers.hpp:178-187` | Same as gamma row. | `cb_beta_rm` | `cb_beta_tilized` | |
| Compute — Pass C subtract mean | helper | `compute_kernel_lib::sub<BroadcastDim::COL, BinaryInputPolicy::WaitAndPopPerTile, BinaryInputPolicy::WaitUpfrontNoPop>(cb_tilized_x, cb_mean, cb_centered, BinaryInputBlockShape::of(1, BLOCK_SIZE))` | `binary_op_helpers.hpp:282-293` | Same as Pass B subtract. | `cb_tilized_x` + `cb_mean` | `cb_centered` | |
| Compute — Pass C multiply by inv_std | helper | `compute_kernel_lib::mul_in_place<BroadcastDim::COL, BinaryInputPolicy::WaitUpfrontNoPop>(cb_centered, cb_inv_std, BinaryInputBlockShape::of(1, BLOCK_SIZE))` | `binary_op_helpers.hpp:454-462` | Template: `bcast_dim = COL`, `input_b_policy = WaitUpfrontNoPop` (cb_inv_std persists across chunks), `reconfig = INPUT_AND_OUTPUT` (default), `init = true` (default). | `cb_centered` (in-place, BLOCK_SIZE tiles) + `cb_inv_std` (not popped) | `cb_centered` (in-place) | `cb_centered` is compute-owned (CB index 25 in the intermediate range), satisfying the in-place ownership rule from `binary_op_helpers.hpp:344-352`. |
| Compute — Pass C multiply by gamma (if `HAS_GAMMA`) | helper | `compute_kernel_lib::mul_in_place<BroadcastDim::ROW, BinaryInputPolicy::WaitAndPopPerTile>(cb_centered, cb_gamma_tilized, BinaryInputBlockShape::of(1, BLOCK_SIZE))` | `binary_op_helpers.hpp:454-462` | Template: `bcast_dim = ROW` (B is `[1, Wt]`, broadcasts row 0 across all H positions — exactly the gamma shape `(1,1,1,W)`), `input_b_policy = WaitAndPopPerTile` (gamma chunk is consumed and freed). | `cb_centered` (in-place) + `cb_gamma_tilized` (popped) | `cb_centered` (in-place) | |
| Compute — Pass C add beta (if `HAS_BETA`) | helper | `compute_kernel_lib::add_in_place<BroadcastDim::ROW, BinaryInputPolicy::WaitAndPopPerTile>(cb_centered, cb_beta_tilized, BinaryInputBlockShape::of(1, BLOCK_SIZE))` | `binary_op_helpers.hpp:443-444` | Template: `bcast_dim = ROW`, `input_b_policy = WaitAndPopPerTile`. | `cb_centered` (in-place) + `cb_beta_tilized` (popped) | `cb_centered` (in-place) | |
| Compute — Pass C untilize | helper | `compute_kernel_lib::untilize<BLOCK_SIZE, cb_centered, cb_output_rm>(1)` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:132-140` (declaration) | Template: `block_width_tiles = BLOCK_SIZE`, `input_cb = cb_centered`, `output_cb = cb_output_rm`, `init_uninit_mode = InitAndUninit` (default), `wait_mode = WaitBlock` (default), `reconfig_mode = UnpackAndPackReconfigure` (default). Runtime: `num_blocks = 1`. | `cb_centered` (popped, BLOCK_SIZE tiles) | `cb_output_rm` (BLOCK_SIZE tile pages) | Output CB is tile-paged because `write_sticks_after_untilize` expects TILE granularity (`tilize_helpers_dataflow.hpp:103-107`). |
| Compute — end-of-strip cleanup | raw_api | `cb_pop_front(cb_mean, 1); cb_pop_front(cb_inv_std, 1);` | `tt_metalium/circular_buffer_constants.h` (raw API; the in-place helpers do not pop B when `WaitUpfrontNoPop` is set, so we drain the persistent CBs at strip end) | — | `cb_mean`, `cb_inv_std` | — | **Helpers considered and rejected:** the binary helpers' `WaitUpfrontNoPop` policy explicitly delegates the pop to the caller (see `binary_op_helpers.hpp:138-141` for B policy semantics — "WaitUpfrontNoPop: Wait for all tiles upfront, don't pop (persistent, for tile reuse)"). There is no "drain a persistent NoPop CB" helper — this is the documented escape hatch. |
| Compute — kernel teardown | raw_api | `cb_pop_front(cb_scaler, 1);` | same as above | — | `cb_scaler` | — | **Helpers considered and rejected:** same as the mean/inv_std drain — the scaler is `WaitAndPopPerTile`-waited but `accumulate_reduce_block`'s internal reduce uses `cb_wait_front` without `cb_pop_front` on the scaler (it stays for every block). At the end of all strips we drop it. |
| Writer — output streaming | helper | `dataflow_kernel_lib::write_sticks_after_untilize<cb_output_rm>(output_accessor, /*total_num_rows=*/32, /*row_bytes=*/chunk_bytes, /*start_page=*/strip_id*32, /*byte_offset_within_page=*/chunk_id*chunk_bytes)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:129-135` (declaration); `tilize_helpers_dataflow.inl:185-241` (body) | `cb_id = cb_output_rm (16)`. | `cb_output_rm` | DRAM (via `output_accessor`) | Called inside `for strip` × `for chunk in NUM_BLOCKS` of Pass C (writer mirrors reader pass count = 1 output pass). |

**Justification for every non-helper raw API.** Only two phases use raw APIs (`cb_pop_front` for `cb_mean`, `cb_inv_std`, `cb_scaler` drains), both already justified in the table above with file:line citations to the helper-policy documentation that mandates caller-side pop on `WaitUpfrontNoPop` and "no helper exists for drain-only" cases.

## Compute Phases

For one strip (the per-strip body of the compute kernel's outer `for strip in strips_per_core` loop). All CB capacities reflect the layout from the previous table. `BLOCK_SIZE` and `NUM_BLOCKS` are compile-time constants.

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | **(prelude, once at kernel start)** Hardware init: `compute_kernel_hw_startup(cb_input_rm, cb_scaler, cb_centered)`. | helper | — | — | LLK registers initialized. |
| 2 | **Pass A, chunk b (for b in [0, NUM_BLOCKS)):** Tilize chunk b of input → BLOCK_SIZE tiles. | helper | `cb_input_rm` (BLOCK_SIZE pages — produced by reader Pass A chunk b) | `cb_tilized_x` (BLOCK_SIZE tiles) | `cb_input_rm` chunk-b drained; `cb_tilized_x` holds BLOCK_SIZE tiles for the reduce. |
| 3 | **Pass A, chunk b:** `accumulate_reduce_block<SUM, REDUCE_ROW>(cb_tilized_x, cb_scaler, cb_mean, of(1, BLOCK_SIZE, 1), b, NUM_BLOCKS)`. | helper | `cb_tilized_x` (BLOCK_SIZE tiles, popped) + `cb_scaler` (1 tile, no-pop) | `cb_mean` (1 tile, only on last chunk; earlier chunks accumulate in DST inside the helper) | After last chunk: `cb_mean` has 1 tile with per-row means in column 0. |
| 4 | **Pass B, chunk b:** Tilize chunk b of input again. | helper | `cb_input_rm` (BLOCK_SIZE pages — reader Pass B chunk b) | `cb_tilized_x` (BLOCK_SIZE tiles) | `cb_tilized_x` populated. `cb_mean` persistent across all of Pass B. |
| 5 | **Pass B, chunk b:** `sub<COL, WaitAndPopPerTile, WaitUpfrontNoPop>(cb_tilized_x, cb_mean, cb_centered, of(1, BLOCK_SIZE))`. | helper | `cb_tilized_x` (popped) + `cb_mean` (held) | `cb_centered` (BLOCK_SIZE tiles of `(x − mean)`) | `cb_mean` still in CB. `cb_centered` has BLOCK_SIZE tiles. |
| 6 | **Pass B, chunk b:** `square_in_place(cb_centered, of(1, BLOCK_SIZE))`. | helper | `cb_centered` (BLOCK_SIZE tiles, in-place) | `cb_centered` (BLOCK_SIZE tiles of `(x − mean)^2`) | Same count, same slots; values are now squared. |
| 7 | **Pass B, chunk b:** `accumulate_reduce_block<SUM, REDUCE_ROW>(cb_centered, cb_scaler, cb_inv_std, of(1, BLOCK_SIZE, 1), b, NUM_BLOCKS)`. | helper | `cb_centered` (BLOCK_SIZE tiles, popped) + `cb_scaler` (no-pop) | `cb_inv_std` (1 tile of variance, only on last chunk) | After last chunk: `cb_inv_std` has variance. `cb_centered` is drained. |
| 8 | **eps + rsqrt:** `transform_in_place(cb_inv_std, lambda(dst){ binop_with_scalar_tile_init(); add_unary_tile(dst, epsilon_bits); rsqrt_tile_init(); rsqrt_tile(dst); })`. | helper (lambda body uses raw in-DST ops; justified above) | `cb_inv_std` (1 tile of variance, popped and re-pushed) | `cb_inv_std` (1 tile of `1/sqrt(var+eps)`) | `cb_inv_std` semantically becomes the inv-std tile. `cb_mean` still held. |
| 9 | **Pass C, chunk b:** Tilize chunk b of input once more. | helper | `cb_input_rm` (BLOCK_SIZE pages — reader Pass C chunk b) | `cb_tilized_x` (BLOCK_SIZE tiles) | |
| 10 | **Pass C, chunk b (if `HAS_GAMMA`):** Tilize gamma chunk b (asymmetric, 1 row). | helper | `cb_gamma_rm` (1 page) | `cb_gamma_tilized` (BLOCK_SIZE tiles, row 0 valid, rows 1-31 garbage for ROW broadcast) | |
| 11 | **Pass C, chunk b (if `HAS_BETA`):** Tilize beta chunk b. | helper | `cb_beta_rm` (1 page) | `cb_beta_tilized` (BLOCK_SIZE tiles) | |
| 12 | **Pass C, chunk b:** `sub<COL, WaitAndPopPerTile, WaitUpfrontNoPop>(cb_tilized_x, cb_mean, cb_centered, of(1, BLOCK_SIZE))`. | helper | `cb_tilized_x` (popped) + `cb_mean` (held) | `cb_centered` (BLOCK_SIZE tiles of `(x − mean)`) | |
| 13 | **Pass C, chunk b:** `mul_in_place<COL, WaitUpfrontNoPop>(cb_centered, cb_inv_std, of(1, BLOCK_SIZE))`. | helper | `cb_centered` (in-place) + `cb_inv_std` (held) | `cb_centered` (BLOCK_SIZE tiles of `(x − mean) / sqrt(var + eps)`) | |
| 14 | **Pass C, chunk b (if `HAS_GAMMA`):** `mul_in_place<ROW, WaitAndPopPerTile>(cb_centered, cb_gamma_tilized, of(1, BLOCK_SIZE))`. | helper | `cb_centered` (in-place) + `cb_gamma_tilized` (popped) | `cb_centered` | After: `cb_gamma_tilized` drained. |
| 15 | **Pass C, chunk b (if `HAS_BETA`):** `add_in_place<ROW, WaitAndPopPerTile>(cb_centered, cb_beta_tilized, of(1, BLOCK_SIZE))`. | helper | `cb_centered` (in-place) + `cb_beta_tilized` (popped) | `cb_centered` | After: `cb_beta_tilized` drained. |
| 16 | **Pass C, chunk b:** `untilize<BLOCK_SIZE, cb_centered, cb_output_rm>(1)`. | helper | `cb_centered` (BLOCK_SIZE tiles, popped) | `cb_output_rm` (BLOCK_SIZE tile pages) | `cb_centered` empty. `cb_output_rm` ready for writer to drain. |
| 17 | **End of strip:** `cb_pop_front(cb_mean, 1); cb_pop_front(cb_inv_std, 1);` | raw_api (justified — `WaitUpfrontNoPop` caller pop) | `cb_mean` (1 tile), `cb_inv_std` (1 tile) | — | Both CBs are empty, ready for the next strip. |
| 18 | **End of kernel (after the strip loop):** `cb_pop_front(cb_scaler, 1);` | raw_api (justified — final scaler drain) | `cb_scaler` | — | Clean exit. |

## Broadcast Verification

Layer-norm uses three broadcast operations.

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| Pass B subtract / Pass C subtract | `sub<BroadcastDim::COL>` | `cb_tilized_x` — all `[Ht=1, Wt=BLOCK_SIZE]` positions are valid | `cb_mean` — `[Ht=1, Wt=1]`; result of `REDUCE_ROW` populates **column 0** of the tile (`reduce<>` packs the row-reduction result into col 0; see `binary_op_helpers.hpp:36-43`). `BroadcastDim::COL` reads B as `[Ht=1, Wt=1]` and replicates **column 0 of each B-tile** across the W direction. ✅ matches |
| Pass C multiply by inv_std | `mul_in_place<BroadcastDim::COL>` | `cb_centered` — all `[Ht=1, Wt=BLOCK_SIZE]` valid | `cb_inv_std` — `[Ht=1, Wt=1]`; populated by `REDUCE_ROW` then overwritten by `transform_in_place` (1-tile in-DST op preserves col-0 valid region). `BroadcastDim::COL` reads col 0. ✅ matches |
| Pass C multiply by gamma | `mul_in_place<BroadcastDim::ROW>` | `cb_centered` — all `[Ht=1, Wt=BLOCK_SIZE]` valid | `cb_gamma_tilized` — `[Ht=1, Wt=BLOCK_SIZE]`; only **row 0** of each tile is valid (asymmetric tilize from a 1-row source, per `tilize_helpers.hpp:141-145`). `BroadcastDim::ROW` reads row 0 of each B-tile and replicates down. ✅ matches |
| Pass C add beta | `add_in_place<BroadcastDim::ROW>` | `cb_centered` — all valid | `cb_beta_tilized` — row 0 valid (same construction as gamma). ✅ matches |

## Key Risks and Gotchas

1. **CBs that must hold full blocks (sequential-helper deadlock).** `cb_tilized_x`, `cb_centered`, `cb_gamma_tilized`, `cb_beta_tilized` are written by one helper and consumed by the next helper in the same `kernel_main()`. The two helpers cannot pipeline (each owns all three TRISCs for its full duration — see `.claude/references/ttnn-cb-memory-fundamentals.md:86-117`). Each of these CBs is therefore sized to the full block (`BLOCK_SIZE` pages), NOT double-buffered. Reader → compute CBs (`cb_input_rm`, `cb_gamma_rm`, `cb_beta_rm`) and compute → writer CB (`cb_output_rm`) ARE double-buffered because reader/writer run on independent processors.

2. **Scaler CB format and source.** `cb_scaler` is fp32 (NOT bf16), even though the planner checklist's default-row says bf16. Rationale: with fp32 input + `fp32_dest_acc_en=True`, a bf16 scaler downcasts the multiply-accumulate result, costing ~3e-3 relative error per reduce. The softmax op makes the identical choice for the same reason (`softmax_program_descriptor.py:158-167`). The scaler value is `1.0f / W` (caller-provided), packed via `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f / W)` — the pool-type-aware overload. The legacy single-arg `prepare_reduce_scaler<cb>` is NOT used.

3. **DEST budget under `fp32_dest_acc_en=True`.** Half-sync + fp32_dest_acc → 4 tiles, full-sync + fp32_dest_acc → 8 tiles (per `dest_helpers.hpp:88-99`). `DEST_AUTO_LIMIT` evaluates to 4 in our Phase-0 config (half-sync is the default `ComputeConfigDescriptor` setting). All helpers used here (`reduce`, `accumulate_reduce_block`, `binary_op` family, `tilize`, `untilize`, `transform_in_place`, `square_in_place`) batch their DST usage via `DEST_AUTO_LIMIT` internally — the caller does not need to size differently for fp32 vs fp16. `BLOCK_SIZE ≤ 8` keeps the helper's batch loops well-behaved.

4. **`cb_mean` and `cb_inv_std` persistence is paid for by explicit caller-side pops.** Pass B's `sub<COL>` and Pass C's `sub<COL>` use `WaitUpfrontNoPop` for input B (`cb_mean`); Pass C's `mul_in_place<COL>` uses `WaitUpfrontNoPop` for `cb_inv_std`. The helper docstrings (`binary_op_helpers.hpp:138-141`) make caller-side popping mandatory when this policy is selected — without the explicit `cb_pop_front(cb_mean, 1)` and `cb_pop_front(cb_inv_std, 1)` at the end of each strip, the CBs fill up after a few strips and the reader stalls on `cb_reserve_back`.

5. **`cb_centered` ownership for in-place ops.** `binary_op_in_place`'s docstring (`binary_op_helpers.hpp:344-352`) requires `cb_a` to be exclusively compute-owned. `cb_centered` lives in CB index 25 (the intermediate-only range, never written by the reader and never read by the writer). The kernel-side untilize that drains `cb_centered` runs on the same TRISCs as the in-place ops, so the ordering is sequential within compute and there is no cross-processor race.

6. **Three reader passes per strip is the L1-budget trade.** Re-reading the input from DRAM three times costs DRAM bandwidth but yields a constant per-strip L1 footprint (all CBs are bounded by `BLOCK_SIZE` tiles, not by `Wt`). At wide W (4096–8192 in the test universe) this is the only way to fit in 1.5 MB L1. Single-pass alternatives would require either holding `Wt` tiles of the tilized input in L1 (1 MB at W=8192) or a chunked algorithm that fuses mean+variance into a single pass via Welford's method — both are larger Phase-1+ refinements out of scope here.

7. **`tilize` `Fp32Mode::Fast` is correct here despite "max precision" framing.** The `tilize_helpers.hpp:64-72` docstring is explicit: `Lossless` is only correct when the tiled output flows exclusively to SFPU-Dest consumers, which is not our case (our consumers are `reduce`, `sub`, `mul_in_place`, `add_in_place`, `untilize` — all FPU-bearing). Using `Lossless` here would slow the tilize without recovering precision because every downstream FPU op re-reads through SrcA/SrcB and truncates fp32 → tf32 anyway.

8. **Non-tile-aligned shapes are out of Phase 0.** `validate()` rejects shapes with `H % 32 != 0` or `W % 32 != 0`. The `feature_spec.py` TARGET universe includes `w_non_aligned` and `h_non_aligned` alignment buckets — those become refinement candidates that the implementer files in `op_requirements.md` (they require the partial-scaler API and a chunked `read_sticks_for_tilize<TILE>` that handles partial last blocks). Phase 0 stays minimal.

## Structural impossibilities

The `feature_spec.py` `INVALID` list at `tt_metal/third_party/tt_ops_code_gen/eval/golden_tests/layer_norm_rm/feature_spec.py:98-113` is taken as authoritative. The four entries cover:
- `{precision: bf8b_hifi4_bf16acc, layout: ROW_MAJOR_LAYOUT}` — block-float in ROW_MAJOR has no blocks (single-tensor coupling on input).
- `{affine_dtype: bfloat8_b, affine_layout: ROW_MAJOR_LAYOUT}` — same impossibility on the affine (gamma/beta) tensors (single-tensor coupling on the affine tensors).
- Three `no_affine` canonicalisation entries that fold the `affine_dtype × affine_layout` cartesian onto the canonical `(float32, TILE_LAYOUT)` cell when there is no affine tensor (canonicalisation-of-redundant-cells, the only multi-axis exception per the skill rules).

Phase-0-specific additional impossibilities the planner sees that the skill may have missed: **none**. The four entries above are sufficient; this op does not introduce additional structural cells where the universe would have to change.
