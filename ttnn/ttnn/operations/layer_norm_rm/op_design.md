# Operation Design: layer_norm (layer_norm_rm)

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (fused statistics + normalization + affine) |
| Goal | Per-row LayerNorm of an interleaved tensor with optional gamma scale and beta shift. Normalize the last dimension, output shape == input shape. |
| Math | `out[..., i, j] = gamma[j] * (x[..., i, j] - mean_i) / sqrt(var_i + eps) + beta[j]` where `mean_i = mean_j(x[..., i, :])` and `var_i = mean_j((x[..., i, :] - mean_i)^2)`. gamma/beta optional. |
| Mode | Derivative — composes existing tilize / untilize / streaming reduce / binary broadcast / SFPU transform helpers from `ttnn/cpp/ttnn/kernel_lib`. |
| References | `.claude/references/generic_op_template/`, `ttnn/ttnn/operations/toy_variance/` (closest existing pattern: two-pass mean/variance with partial-scaler handling), `ttnn/ttnn/operations/toy_tilize_untilize/` (in-kernel tilize+untilize), `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`, `ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`, `eval/golden_tests/layer_norm_rm/feature_spec.py` (authoritative TARGET / INVALID). |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank ≥ 2; dtype ∈ {bfloat16, float32, bfloat8_b}; layout ∈ {ROW_MAJOR, TILE}; INVALID combo {bfloat8_b, ROW_MAJOR} rejected by validate() | — | RT (buffer addr) + CT (shape) |
| `gamma` | `ttnn.Tensor` or `None` | no | shape `(1, 1, 1, W)` where `W == input.shape[-1]`; layout MUST be `ROW_MAJOR_LAYOUT`; dtype MUST equal `input_tensor.dtype` | `None` | RT (addr) + CT (shape, present flag) |
| `beta` | `ttnn.Tensor` or `None` | no | same shape/layout/dtype constraints as `gamma` | `None` | RT (addr) + CT (shape, present flag) |
| `epsilon` | `float` (keyword-only) | no | finite > 0; recommended ≥ 1e-12 | `1e-5` | CT (bit-cast to fp32 uint32) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` (keyword-only) | no | any `ComputeConfigDescriptor`; controls `math_fidelity`, `fp32_dest_acc_en`, `math_approx_mode`, … | `None` → `ttnn.ComputeConfigDescriptor()` (HiFi4, bf16 dest) | passed to `KernelDescriptor.config` |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| `input_tensor` shape | `(..., H, W)` — at least rank 2; arbitrary leading batch dims |
| `input_tensor` dtype | `bfloat16` (default), `float32`, or `bfloat8_b` |
| `input_tensor` layout | `ROW_MAJOR_LAYOUT` or `TILE_LAYOUT` (handled natively; no host-side conversion) |
| `input_tensor` memory | DRAM interleaved (only memory layout the op accepts in Phase 0; sharded inputs fall to `EXCLUSIONS` in the op file) |
| `gamma` / `beta` shape | `(1, 1, 1, W)`; flattened width must equal `input_tensor.shape[-1]` (using padded `W`'s logical value) |
| `gamma` / `beta` dtype | exactly equal to `input_tensor.dtype` |
| `gamma` / `beta` layout | always `ROW_MAJOR_LAYOUT` regardless of input layout |
| `gamma` / `beta` memory | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to `input_tensor.shape` |
| Dtype | identical to `input_tensor.dtype` |
| Layout | identical to `input_tensor.layout` — TILE→TILE, RM→RM |
| Memory | DRAM interleaved (same as input default) |

## Algorithm and Dataflow Strategy

LayerNorm is a per-row statistics + element-wise rescale. We use a **three-pass streaming algorithm** (input read three times from DRAM) that gives independent control over numerical accuracy, L1 footprint, and gamma/beta fusion:

| Pass | Reads | Writes | Helper(s) |
|------|-------|--------|-----------|
| 1 (mean) | `cb_input_tiles` (Wt × Ht_local per tile-row group) | `cb_mean` (Ht_local persistent) | `streaming_reduce_helpers::accumulate_reduce<SUM, REDUCE_ROW>` with scaler `1/W` (SUM × 1/W = mean) |
| 2 (variance → inv_std) | `cb_input_tiles`, `cb_mean` (persistent) | `cb_inv_std` (Ht_local persistent, in place over `cb_variance`) | per-block: `binary_op_helpers::sub<COL>` (x − mean → cb_centered), `square_in_place`, `streaming_reduce_helpers::accumulate_reduce_block<SUM, REDUCE_ROW>` (accum variance); after loop: `streaming_reduce_helpers::transform_in_place` to compute `rsqrt(var + eps)` per Ht_local tile |
| 3 (normalize + affine + drain) | `cb_input_tiles`, `cb_mean`, `cb_inv_std`, `cb_gamma_tiles?`, `cb_beta_tiles?` | `cb_output` (block by block) | per-block: `sub<COL>` (x − mean → cb_centered), `mul_in_place<COL>` (× inv_std), optional `mul_in_place<ROW>` (× gamma), optional `add_in_place<ROW>` (+ beta), then drain via `copy_tiles` (TILE output) or `untilize` (RM output) |

**Why three passes (not Welford, not E[x²] − E[x]²):**
- Welford requires per-tile sequential accumulation that the streaming-reduce helpers don't support; would need raw LLK code, violating the helper-first invariant.
- `E[x²] − E[x]²` has catastrophic-cancellation issues for activations with mean magnitudes ≫ variance and is rejected as a numerical accuracy regression versus the two-pass mean/variance reference in `pytorch_layer_norm` (helpers/helpers.py:33-50).
- Three passes re-read the input from DRAM 3×; bandwidth is the cost, but each pass uses well-tested helpers and the on-chip footprint stays bounded (no need to cache `(x − mean)` for the entire row).

### DRAM → Tensix → DRAM dataflow

Per Tensix core:
1. **Reader (NCRISC)** streams the per-core slice of `input_tensor` from DRAM into `cb_input_*` THREE times (one full sweep per pass). On the first pass it also: tilizes gamma/beta to `cb_gamma_tiles` / `cb_beta_tiles` (in-kernel), and writes the `1/W` scaler tile(s) to `cb_scaler`. The scaler tile is written ONCE; it is never popped (reduce<> waits but never consumes), so it serves all three passes.
2. **Compute (TRISC unpack/math/pack)** runs Pass 1 → Pass 2 → Pass 3 as described above. After Pass 1, `cb_mean` is held with `WaitUpfrontNoPop` policy until end of Pass 3. After Pass 2, `cb_inv_std` is similarly held until end of Pass 3.
3. **Writer (BRISC)** consumes `cb_output` block-by-block and writes to DRAM. For TILE output, the writer reads tile-sized pages and `noc_async_write_tile`s. For RM output, the writer reads stick-sized pages (one stick per page) and writes `row_bytes` (= origin_W × element_size) per stick, skipping any L1 padding past origin_W.

**Layout adaptation:**
- For ROW_MAJOR input, reader uses `dataflow_kernel_lib::read_sticks_for_tilize<cb_input_sticks, TilizeGranularity::ROW>` and compute calls `tilize<BLOCK_SIZE, cb_input_sticks, cb_input_tiles>(Ht_local, rows_this_pass)` per pass to tilize the streamed sticks into `cb_input_tiles`.
- For TILE input, reader pushes tiles directly into `cb_input_tiles` (no tilize step); `cb_input_sticks` is not allocated.
- For ROW_MAJOR output, compute uses `untilize<BLOCK_SIZE, cb_centered, cb_output>(Ht_local)` per output block; the writer drains sticks.
- For TILE output, compute uses `copy_tiles(cb_centered, cb_output, Ht_local * BLOCK_SIZE)` per output block; the writer drains tiles.

Gamma/beta are ALWAYS row-major (per user spec). When the input is TILE layout, gamma/beta still arrive as RM sticks (a single stick of `padded_W * element_size` bytes), and the compute kernel still calls `tilize<Wt, cb_gamma_sticks, cb_gamma_tiles>(1, 1)` (one block of one stick) to tilize them into `Wt` tiles. The tilized tiles are retained in `cb_gamma_tiles` for all Pass 3 blocks via `BinaryInputPolicy::WaitUpfrontNoPop` on the `mul_in_place<ROW>` call.

**Tensix-to-Tensix communication:** none. Each Tensix core processes a disjoint slice of tile-rows; gamma and beta are read independently from DRAM by every core (they're small — Wt tiles after tilize). No semaphores, no multicast, no rings.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | One **tile-row group** = 32 consecutive rows of the input, processed as `Wt` tiles wide. For RM input, 32 consecutive sticks per work unit; for TILE input, one row-block of `Wt` tiles. |
| Total work units | `total_tile_rows = ceil(product(input.shape[:-1]) / 32)` — i.e., the height tile count over all leading dims flattened, including any partial-H rounded up. |
| Grid | `device.compute_with_storage_grid_size()` (e.g., 8×8 = 64 cores on Wormhole). Caller via `ttnn.split_work_to_cores(grid, total_tile_rows)` returns `(num_cores, all_cores, core_group_1, core_group_2, units_per_core_g1, units_per_core_g2)`. |
| Per-core work | A core in `core_group_1` gets `Ht_per_core_g1` tile-row groups; group_2 gets `Ht_per_core_g2`. Each core's runtime args carry `start_tile_row` (offset into the flattened height) and `Ht_local`. The kernels iterate purely within their slice — no cross-core synchronization. |
| Remainder | Naturally handled by `split_work_to_cores`. Cores in core_group_2 get one fewer tile-row each when `total_tile_rows % num_cores != 0`. When `total_tile_rows < num_cores`, only `total_tile_rows` cores get work and the rest sit idle. |
| Small-tensor fallback | When `total_tile_rows == 1` (i.e., one tile-row total), only 1 core is used — `split_work_to_cores` returns `num_cores=1` automatically. No special-case code path needed in the Python factory; let the helper do its job. |
| BLOCK_SIZE | Streaming block in tiles along the W direction. Determined per program: largest divisor of `Wt` that is ≤ 8 (i.e., `_pick_block_size(Wt, requested=None)` per the toy_variance:27-36 pattern). `NUM_BLOCKS = Wt / BLOCK_SIZE`. |

## Circular Buffers

CBs are allocated based on the input layout (`IS_RM_INPUT`), output layout (`IS_RM_OUTPUT == IS_RM_INPUT` since the spec ties them), and which of gamma/beta are present (`HAS_GAMMA`, `HAS_BETA`). The unconditional CBs are listed first; conditional ones note their allocation predicate.

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_sticks` | 0 | `padded_row_bytes = ceil(W / 32) * 32 * element_size` | `2 * 32 * Ht_local` (≥ 2 tile-row groups of 32 sticks double-buffered) | input dtype (RM) | Reader (NCRISC) — streams input sticks | Compute — tilizes via `read_sticks_for_tilize<ROW>` + `tilize<..>(num_blocks, total_rows)` | Allocated only when **input is RM**. Reused for all 3 passes (reader rewinds DRAM addr each pass). |
| `cb_input_tiles` | 1 | `ttnn.tile_size(input.dtype)` | `2 * Ht_local * BLOCK_SIZE` (one block double-buffered) | input dtype (tile) | Compute (RM input) via `tilize`, OR Reader directly (TILE input) | Compute — Pass 1 reduce, Pass 2 sub<COL>, Pass 3 sub<COL> | Allocated unconditionally. For TILE input, reader pushes here directly; for RM input, compute fills it from `cb_input_sticks`. Reused all 3 passes. |
| `cb_gamma_sticks` | 2 | `padded_row_bytes` | 2 sticks (double-buffered single-stick CB) | input dtype (RM) | Reader — single read of gamma | Compute — tilizes once into `cb_gamma_tiles` | Allocated only when `HAS_GAMMA`. Lifetime: one pre-pass tilize step at kernel start. |
| `cb_beta_sticks` | 3 | `padded_row_bytes` | 2 sticks | input dtype (RM) | Reader — single read of beta | Compute — tilizes once into `cb_beta_tiles` | Allocated only when `HAS_BETA`. Lifetime: one pre-pass tilize step at kernel start. |
| `cb_scaler` | 4 | `ttnn.tile_size(ttnn.bfloat16)` | `2` when `has_partial_w` else `1` (full + partial scaler tiles, both must coexist) | `bfloat16` (REQUIRED — `prepare_partial_reduce_scalers` / `prepare_reduce_scaler` emit bf16 layout) | Reader — writes `1/W` scaler tile pair on startup | Compute — `reduce<>` waits, never pops | Allocated unconditionally. Producer pushes once at startup; both passes' reduce<> calls wait on the same tiles. |
| `cb_output` | 16 | TILE output: `ttnn.tile_size(input.dtype)`. RM output: `padded_row_bytes`. | `2 * Ht_local * BLOCK_SIZE` (TILE) or `2 * 32 * Ht_local` (RM) | output dtype (== input dtype) | Compute — Pass 3 drain via `copy_tiles` (TILE) or `untilize<>` (RM) | Writer | Allocated unconditionally; page size + count switches on output layout. |
| `cb_gamma_tiles` | 24 | `ttnn.tile_size(input.dtype)` | `2 * Wt` (Wt tiles, held with `WaitUpfrontNoPop` across all Pass 3 blocks; 2× for headroom) | input dtype | Compute — `tilize<Wt, cb_gamma_sticks, cb_gamma_tiles>(1, 1)` once | Compute — Pass 3 `mul_in_place<ROW>` (per block, `WaitUpfrontNoPop`) | Allocated only when `HAS_GAMMA`. Filled once before Pass 1. Persistent through end of Pass 3; popped at kernel end. |
| `cb_beta_tiles` | 25 | `ttnn.tile_size(input.dtype)` | `2 * Wt` | input dtype | Compute — `tilize<Wt, cb_beta_sticks, cb_beta_tiles>(1, 1)` once | Compute — Pass 3 `add_in_place<ROW>` (per block, `WaitUpfrontNoPop`) | Allocated only when `HAS_BETA`. Same lifetime as `cb_gamma_tiles`. |
| `cb_mean` | 26 | `ttnn.tile_size(input.dtype)` | `max(2 * Ht_local, 2)` | input dtype | Compute — Pass 1 `accumulate_reduce` accumulator (writes Ht_local tiles) | Compute — Pass 2 `sub<COL>` (B = `WaitUpfrontNoPop`) and Pass 3 `sub<COL>` (B = `WaitUpfrontNoPop`) | Filled at end of Pass 1; held through Pass 2 and Pass 3; popped (`cb_pop_front(cb_mean, Ht_local)`) at kernel end. |
| `cb_inv_std` | 27 | `ttnn.tile_size(input.dtype)` | `max(2 * Ht_local, 2)` | input dtype | Compute — Pass 2 `accumulate_reduce_block` accumulator (Ht_local tiles), then mutated in place by `transform_in_place` (var + eps; rsqrt) — same CB used in two phases | Compute — Pass 3 `mul_in_place<COL>` (B = `WaitUpfrontNoPop`) | Filled at end of Pass 2 (post-transform); held through Pass 3; popped at kernel end. The same physical CB transiently holds variance values mid-Pass-2 before the per-tile `transform_in_place` rewrites each slot with `rsqrt(var + eps)`. |
| `cb_centered` | 28 | `ttnn.tile_size(input.dtype)` | `2 * Ht_local * BLOCK_SIZE` (sized to hold one full Pass-3 block for the in-place chain) | input dtype | Compute — `sub<COL>` (both Pass 2 and Pass 3) | Compute — `square_in_place` (Pass 2), in-place `mul`/`add` chain + drain (Pass 3) | Sized for one full Pass-3 block (per `binary_op_in_place` contract — see binary_op_helpers.hpp:336-342: "All Ht*Wt tiles of A must already be present in cb_a before calling"). After Pass 3 in-place chain, drained into `cb_output` block by block. |

### CB sizing rationale

| CB | Sizing rule | Why |
|----|-------------|-----|
| `cb_input_sticks` | `2 * 32 * Ht_local` sticks | Reader pushes one stick at a time (ROW granularity); compute tilizes one block (32 sticks per tile-row) at a time. Double-buffer one block-equivalent of sticks for reader/compute pipelining. |
| `cb_input_tiles` | `2 * Ht_local * BLOCK_SIZE` tiles | Compute consumes one block (`Ht_local * BLOCK_SIZE` tiles) per inner step. Double-buffer for pipelining with the producer (reader for TILE, prior tilize step for RM). |
| `cb_gamma_sticks` / `cb_beta_sticks` | 2 sticks each | One stick of gamma/beta total; double-buffer the single stick so the reader can pre-stage while the compute kernel finishes prior work. |
| `cb_scaler` | 1 or 2 tiles | Single full scaler when `has_partial_w == false`; full + partial pair (`prepare_partial_reduce_scalers` emits exactly these two tiles, in this order) when `has_partial_w == true`. |
| `cb_output` | `2 * Ht_local * BLOCK_SIZE` tiles (TILE) / `2 * 32 * Ht_local` sticks (RM) | Compute drains one block at a time; writer consumes. Double-buffer for compute/writer pipelining. |
| `cb_gamma_tiles` / `cb_beta_tiles` | `2 * Wt` tiles | Wt tiles per row of gamma/beta; held with `WaitUpfrontNoPop` for the entire Pass 3 so each of the `NUM_BLOCKS` blocks can reuse them. 2× for headroom. |
| `cb_mean` | `max(2 * Ht_local, 2)` tiles | Holds Ht_local mean tiles across Pass 2 and Pass 3. The `max(., 2)` floor avoids 1-page CB pathologies on the smallest test (Ht_local == 1). |
| `cb_inv_std` | `max(2 * Ht_local, 2)` tiles | Same as `cb_mean`. Transiently holds variance during Pass 2; rewritten to `rsqrt(var + eps)` by `transform_in_place` before Pass 3. |
| `cb_centered` | `2 * Ht_local * BLOCK_SIZE` tiles | The Pass 3 `*_in_place` chain requires the full Ht*Wt block to be present at call time (helper invariant — see binary_op_helpers.hpp:336-342). Double-buffer one block for the post-chain drain to overlap with the next block's `sub<COL>`. |

### CB sync verification

Every CB satisfies producer push count == consumer wait count:

| CB | Push count per pass | Wait count per pass | Comment |
|----|---------------------|---------------------|---------|
| `cb_input_sticks` | Reader pushes `32 * Ht_local` sticks per pass × 3 passes (RM input only) | Compute (`tilize<..>(Ht_local, 32 * Ht_local)`) waits per-block; total waits == total pushes per pass | Per `tilize_helpers_dataflow.hpp:60-65`, ROW granularity pushes 1 page per row; matched by tilize's per-block-with-total wait semantics. |
| `cb_input_tiles` | TILE input: reader pushes `Ht_local * Wt` tiles per pass × 3. RM input: compute (via tilize) pushes `Ht_local * Wt` tiles per pass × 3. | Compute downstream pops `Ht_local * Wt` per pass: Pass 1 reduce, Pass 2 sub<COL>, Pass 3 sub<COL>. | Match. |
| `cb_gamma_sticks` / `cb_beta_sticks` | Reader pushes 1 stick (gamma/beta is one row) | Compute tilize pops 1 stick | Match — one-shot, pre-Pass-1. |
| `cb_scaler` | Reader pushes 1 or 2 tiles (depending on `has_partial_w`) | Compute reduces in Pass 1 and Pass 2 both wait `1 + (has_partial_w ? 1 : 0)` — but reduce<> never pops, so the same tiles serve all passes. Pop at kernel end: `cb_pop_front(cb_scaler, has_partial_w ? 2 : 1)`. | Match (single push, single late pop). |
| `cb_mean` | Pass 1: `accumulate_reduce` pushes `Ht_local` tiles | Pass 2: `sub<COL>` B-side waits `Ht_local` (`WaitUpfrontNoPop`); Pass 3: `sub<COL>` B-side waits `Ht_local` (`WaitUpfrontNoPop`). Compute pops `Ht_local` at kernel end. | Match (single push, single late pop; `WaitUpfrontNoPop` is wait-only). |
| `cb_inv_std` | Pass 2 `accumulate_reduce_block` pushes `Ht_local` tiles (variance); `transform_in_place` pops + pushes `Ht_local` times (net 0 change) | Pass 3: `mul_in_place<COL>` B-side waits `Ht_local` (`WaitUpfrontNoPop`). Compute pops `Ht_local` at kernel end. | Match. `transform_in_place` is pop-then-push per tile so the net count is preserved. |
| `cb_centered` | Pass 2: `sub<COL>` pushes `Ht_local * BLOCK_SIZE` per block × `NUM_BLOCKS`. Pass 3: same. | Pass 2: `square_in_place` pops/pushes same count (in-place); `accumulate_reduce_block` pops `Ht_local * BLOCK_SIZE` per block. Pass 3: `mul_in_place<COL>`, `mul_in_place<ROW>?`, `add_in_place<ROW>?` are each pop-then-push (in-place); final drain (`copy_tiles` or `untilize`) pops `Ht_local * BLOCK_SIZE` per block. | Match per block. |
| `cb_gamma_tiles` / `cb_beta_tiles` | Compute pushes `Wt` tiles once at startup | Pass 3: `mul_in_place<ROW>` (gamma) / `add_in_place<ROW>` (beta) waits `Wt` per block (`WaitUpfrontNoPop`); never pops mid-pass. Compute pops `Wt` at kernel end. | Match. |
| `cb_output` | Compute drain pushes `Ht_local * BLOCK_SIZE` tiles (TILE) or `32 * Ht_local` sticks (RM) per block × `NUM_BLOCKS` | Writer pops same per block | Match. |

## API Mapping

Every mechanism — helper or raw — has an exact file:line reference.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB (semantic name) | Output CB (semantic name) | Requirements |
|-------|------|----------|-----------|------------------------|--------------------------|---------------------------|--------------|
| Reader: scaler tile setup | helper | `dataflow_kernel_lib::prepare_partial_reduce_scalers` (when partial W) or `dataflow_kernel_lib::prepare_reduce_scaler` (when not) | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:136-142` (partial) / `:65-67` (single) | `<cb_id=cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW, partial_positions=partial_w>` for partial; `<cb_id=cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>` for full. Argument: `scaler_f = 1.0f / float(origin_W)`. | — | `cb_scaler` (1 or 2 tiles) | Pool-type-aware overload (not legacy `prepare_reduce_scaler<cb>`). `bfloat16` CB. |
| Reader: RM input streaming (RM input only) | helper | `dataflow_kernel_lib::read_sticks_for_tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:77-79` | `<cb_id=cb_input_sticks, TilizeGranularity::ROW>` (template); call args: `accessor`, `total_num_rows=32 * Ht_local`, `row_bytes=W * element_size`, `start_page=start_tile_row * 32`. Called once per pass × 3 passes. | DRAM (via `TensorAccessor`) | `cb_input_sticks` | ROW granularity matches asymmetric `tilize<>(num_blocks, total_rows)` on the compute side. |
| Reader: TILE input streaming (TILE input only) | raw_api | `noc_async_read_tile` + `cb_reserve_back` / `cb_push_back` per tile (loop) | `tt_metal/include/api/dataflow/dataflow_api.h` (raw NoC API), pattern documented in `.claude/references/ttnn-cb-memory-fundamentals.md:208-232` and shown working in `ttnn/ttnn/operations/toy_variance/kernels/reader.cpp:57-70` | Loop: for pass in [0,3): for ht in [0, Ht_local): for wt in [0, Wt): `tile_id = (start_tile_row + ht) * Wt + wt`, `cb_reserve_back(cb_input_tiles, 1)`, `noc_async_read_tile(tile_id, accessor, get_write_ptr(cb_input_tiles))`, `noc_async_read_barrier()`, `cb_push_back(cb_input_tiles, 1)`. | DRAM | `cb_input_tiles` | **Helpers considered and rejected**: `read_sticks_for_tilize` (tilize_helpers_dataflow.hpp:60-65) — emits sticks not tiles, intended for the RM→tilize path. **There is no dataflow helper that reads tiles into a tile CB**; `noc_async_read_tile` is the canonical raw mechanism (used in every TILE-layout reader in the repo, e.g. toy_variance/kernels/reader.cpp:62-66). |
| Reader: gamma stick read (HAS_GAMMA only) | raw_api | `noc_async_read_page` (one page = one gamma stick) | Same as above; one-shot stick read | One stick read: `cb_reserve_back(cb_gamma_sticks, 1)`, `noc_async_read_page(0, gamma_accessor, get_write_ptr(cb_gamma_sticks))`, `noc_async_read_barrier()`, `cb_push_back(cb_gamma_sticks, 1)`. | DRAM | `cb_gamma_sticks` | **Helpers considered and rejected**: `read_sticks_for_tilize<TilizeGranularity::ROW>` requires `total_num_rows ≥ 32` for a full tile-row; here we have exactly 1 row (one stick of gamma). The helper assumes block-of-32-rows semantics — using it for a single-row read would push a malformed block and the downstream `tilize<>(1, 1)` would mismatch. **File:line for the mismatch**: tilize_helpers_dataflow.hpp:60-65 documents the row-block invariant. So we use raw `noc_async_read_page` here — same one-shot pattern used for scaler tile prep. |
| Reader: beta stick read (HAS_BETA only) | raw_api | `noc_async_read_page` | same as gamma | same as gamma, into `cb_beta_sticks` | DRAM | `cb_beta_sticks` | Same justification as gamma. |
| Compute: HW init | raw_api | `compute_kernel_hw_startup(icb0, icb1, ocb)` | `ttnn/cpp/ttnn/kernel_lib/*` doc strings (e.g. `tilize_helpers.hpp:83-87`, `binary_op_helpers.hpp:16-19`) — required exactly once per kernel | `compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output)` — 3-arg form for srcA/srcB/dst | — | — | Helpers' contract — must be the FIRST call in `kernel_main`, never re-called. |
| Compute: tilize gamma (HAS_GAMMA only) | helper | `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:178-187` | `<block_width_tiles=Wt, input_cb=cb_gamma_sticks, output_cb=cb_gamma_tiles>` (default `InitAndUninit`, `WaitBlock`, `UnpackAndPackReconfigure`, `Fast`). Call args: `num_blocks=1`, `total_input_pages=1` (asymmetric — 1 stick → Wt tiles). | `cb_gamma_sticks` | `cb_gamma_tiles` | One-shot pre-Pass-1 step. Wt is compile-time so the helper instantiates correctly. |
| Compute: tilize beta (HAS_BETA only) | helper | `compute_kernel_lib::tilize` | `tilize_helpers.hpp:178-187` | `<Wt, cb_beta_sticks, cb_beta_tiles>`; args `(1, 1)` | `cb_beta_sticks` | `cb_beta_tiles` | Same as gamma. |
| Compute: tilize input (RM input only, per pass) | helper | `compute_kernel_lib::tilize` | `tilize_helpers.hpp:178-187` | `<BLOCK_SIZE, cb_input_sticks, cb_input_tiles>`, `InitUninitMode::InitAndUninit`, `WaitMode::WaitBlock`, `Fp32Mode::Fast`. Asymmetric call: `num_blocks=Ht_local * NUM_BLOCKS` (one block per Ht-tile per W-block ... actually `num_blocks=Ht_local` per W-block iteration). Total input pages = `32 * Ht_local`. | `cb_input_sticks` | `cb_input_tiles` | Called once per pass (3 times total). |
| Compute Pass 1: streaming mean | helper | `compute_kernel_lib::accumulate_reduce` | `ttnn/cpp/ttnn/kernel_lib/streaming_reduce_helpers.hpp:79-92` | `<PoolType::SUM, ReduceDim::REDUCE_ROW>` (default policies). Args: `cb_in=cb_input_tiles`, `cb_scaler=cb_scaler`, `cb_acc=cb_mean`, `block_shape=ReduceInputBlockShape::of(Ht_local, BLOCK_SIZE, /*NC=*/1)`, `num_blocks=NUM_BLOCKS`, `partial = has_partial_w ? ReducePartialScaler::last_tile_at(1) : ReducePartialScaler::none()`. | `cb_input_tiles`, `cb_scaler` | `cb_mean` | Standard mean reduction via `1/W * SUM`. Streaming chunked along W direction so wide W (e.g. 64000) doesn't exceed L1. |
| Compute Pass 2 (per block b): centered subtract | helper | `compute_kernel_lib::sub` (binary_op alias) | `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp:282-293` | `<BroadcastDim::COL, BinaryInputPolicy::WaitAndPopPerTile, BinaryInputPolicy::WaitUpfrontNoPop>`. Args: `icb_a=cb_input_tiles`, `icb_b=cb_mean`, `ocb=cb_centered`, `shape=BinaryInputBlockShape::of(Ht_local, BLOCK_SIZE)`. | `cb_input_tiles` (A, streaming), `cb_mean` (B, persistent) | `cb_centered` | Mean is a single column-vector per Ht-tile (REDUCE_ROW output) — `BroadcastDim::COL` replicates across W. |
| Compute Pass 2 (per block b): square in place | helper | `compute_kernel_lib::square_in_place` | `binary_op_helpers.hpp:466-469` | Args: `cb_a=cb_centered`, `shape=BinaryInputBlockShape::of(Ht_local, BLOCK_SIZE)`. | `cb_centered` (in/out) | `cb_centered` (in/out) | Squares in place so we keep one CB intermediate; the helper handles the dst-sync + format-reconfig dance. |
| Compute Pass 2 (per block b): variance accumulator | helper | `compute_kernel_lib::accumulate_reduce_block` | `streaming_reduce_helpers.hpp:47-61` | `<PoolType::SUM, ReduceDim::REDUCE_ROW>`. Args: `cb_in=cb_centered`, `cb_scaler=cb_scaler`, `cb_acc=cb_inv_std`, `block_shape=ReduceInputBlockShape::of(Ht_local, BLOCK_SIZE, 1)`, `b`, `num_blocks=NUM_BLOCKS`, `partial = has_partial_w ? last_tile_at(1) : none()`. | `cb_centered`, `cb_scaler` | `cb_inv_std` (holds variance pre-transform) | Routes partial scaler only to last block; helper owns accumulator reload via `Accumulate::at`. |
| Compute Pass 2 finalize: variance → inv_std | helper | `compute_kernel_lib::transform_in_place` (looped Ht_local times) | `streaming_reduce_helpers.hpp:94-111` | Lambda `[eps_bits](uint32_t dst){ binop_with_scalar_tile_init(); add_unary_tile(dst, eps_bits); rsqrt_tile_init(); rsqrt_tile(dst); }` where `eps_bits = bit_cast<uint32_t>(epsilon)`. Call: `for (ht=0; ht<Ht_local; ++ht) transform_in_place(cb_inv_std, lambda);`. | `cb_inv_std` (variance) | `cb_inv_std` (rsqrt(var+eps)) | Helper pops one tile, runs lambda in DST[0], packs back. `add_unary_tile(dst, eps_bits)` adds epsilon scalar (see `eltwise_scalar.hpp`). |
| Compute Pass 3 (per block): centered subtract | helper | `compute_kernel_lib::sub` | `binary_op_helpers.hpp:282-293` | Same template + arg shape as Pass 2 centered subtract. A-policy `WaitAndPopPerTile`, B-policy `WaitUpfrontNoPop`. | `cb_input_tiles` (A), `cb_mean` (B persistent) | `cb_centered` | Same operation as Pass 2 step 1, re-streaming the input through cb_input_tiles. |
| Compute Pass 3 (per block): * inv_std | helper | `compute_kernel_lib::mul_in_place` | `binary_op_helpers.hpp:455-462` | `<BroadcastDim::COL, BinaryInputPolicy::WaitUpfrontNoPop>`. Args: `cb_a=cb_centered`, `icb_b=cb_inv_std`, `shape=BinaryInputBlockShape::of(Ht_local, BLOCK_SIZE)`. | `cb_centered` (in/out), `cb_inv_std` (B persistent) | `cb_centered` (in/out) | Per binary_op_helpers.hpp:336-342, full block of A must be pre-populated — guaranteed by the preceding `sub<COL>`. |
| Compute Pass 3 (per block, HAS_GAMMA only): * gamma | helper | `compute_kernel_lib::mul_in_place` | `binary_op_helpers.hpp:455-462` | `<BroadcastDim::ROW, BinaryInputPolicy::WaitUpfrontNoPop>`. Args: `cb_a=cb_centered`, `icb_b=cb_gamma_tiles`, `shape=BinaryInputBlockShape::of(Ht_local, BLOCK_SIZE)`. | `cb_centered`, `cb_gamma_tiles` | `cb_centered` | Gamma is a single row of W elements (1 × Wt tiles) — `BroadcastDim::ROW` replicates down H. |
| Compute Pass 3 (per block, HAS_BETA only): + beta | helper | `compute_kernel_lib::add_in_place` | `binary_op_helpers.hpp:438-444` | `<BroadcastDim::ROW, BinaryInputPolicy::WaitUpfrontNoPop>`. Args: `cb_a=cb_centered`, `icb_b=cb_beta_tiles`, `shape=BinaryInputBlockShape::of(Ht_local, BLOCK_SIZE)`. | `cb_centered`, `cb_beta_tiles` | `cb_centered` | Same broadcast semantics as gamma. |
| Compute Pass 3 (per block): drain — TILE output | helper | `compute_kernel_lib::copy_tiles` | `ttnn/cpp/ttnn/kernel_lib/copy_tile_helpers.hpp:146-150` | `<CopyInputPolicy::WaitAndPop, CopyDataFormatReconfig::INPUT_AND_OUTPUT>`. Args: `cb_centered`, `cb_output`, `num_tiles = Ht_local * BLOCK_SIZE`. | `cb_centered` | `cb_output` | Only when `IS_RM_OUTPUT == false`. |
| Compute Pass 3 (per block): drain — RM output | helper | `compute_kernel_lib::untilize` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:132-140` | `<BLOCK_SIZE, cb_centered, cb_output>` (default `InitAndUninit`, `WaitBlock`, `UnpackAndPackReconfigure`). Args: `num_blocks=Ht_local`. | `cb_centered` | `cb_output` | Only when `IS_RM_OUTPUT == true`. Block-style untilize: one tile-row of `BLOCK_SIZE` tiles per LLK call × `Ht_local`. |
| Compute Pass 2 → 3 cleanup | raw_api | `cb_pop_front` | `dataflow_api.h` standard | `cb_pop_front(cb_mean, Ht_local)`, `cb_pop_front(cb_inv_std, Ht_local)`, `cb_pop_front(cb_gamma_tiles, Wt)` (if HAS_GAMMA), `cb_pop_front(cb_beta_tiles, Wt)` (if HAS_BETA), `cb_pop_front(cb_scaler, has_partial_w ? 2 : 1)` | persistent CBs | — | **Helpers considered and rejected**: no helper covers explicit late-pop of CBs that were held via `WaitUpfrontNoPop`. The toy_variance kernel has the same pattern (`compute.cpp:106` and `:113`). Raw `cb_pop_front` is the documented mechanism for this. |
| Writer: RM stick write (RM output only) | helper | `dataflow_kernel_lib::write_sticks_after_untilize` | `tilize_helpers_dataflow.hpp:107-109` | `<cb_id=cb_output>`. Args: `accessor`, `total_num_rows=32 * Ht_local`, `row_bytes=W * element_size`, `start_page=start_tile_row * 32`. | `cb_output` (sticks) | DRAM | Writes only the valid `row_bytes` per stick, skipping L1 padding past `W`. |
| Writer: tile write (TILE output only) | raw_api | `noc_async_write_tile` + `cb_wait_front` / `cb_pop_front` per tile | `dataflow_api.h`; pattern in `toy_variance/kernels/writer.cpp:21-27` | Loop: for ht in [0, Ht_local): for wt in [0, Wt): `tile_id = (start_tile_row + ht) * Wt + wt`, `cb_wait_front(cb_output, 1)`, `noc_async_write_tile(tile_id, accessor, get_read_ptr(cb_output))`, `noc_async_write_barrier()`, `cb_pop_front(cb_output, 1)`. | `cb_output` | DRAM | **Helpers considered and rejected**: `write_sticks_after_untilize` (tilize_helpers_dataflow.hpp:107-109) — writes sticks, not tiles. No tile-mode writer helper exists in the repo's kernel_lib (verified by grepping `kernel_lib/*.hpp` for "tile_write" / "write_tile" — only the raw NoC mechanism is exposed). The toy_variance writer (writer.cpp:21-27) uses the same loop. |

## Compute Phases

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 0a | (HAS_GAMMA only) tilize gamma row | yes (`tilize`) | `cb_gamma_sticks` (1 stick) | `cb_gamma_tiles` (Wt tiles) | `cb_gamma_tiles` filled and held until kernel end. `cb_gamma_sticks` empty. |
| 0b | (HAS_BETA only) tilize beta row | yes (`tilize`) | `cb_beta_sticks` (1 stick) | `cb_beta_tiles` (Wt tiles) | `cb_beta_tiles` filled and held until kernel end. `cb_beta_sticks` empty. |
| 1 | (RM input only, pre-Pass-1) tilize input pass 1 | yes (`tilize`) | `cb_input_sticks` (32 × Ht_local sticks) | `cb_input_tiles` (Ht_local × Wt tiles) | Note: in practice the per-pass tilize is interleaved with the consumer block-by-block so `cb_input_tiles` never holds more than one block. |
| 2 | Pass 1: streaming mean | yes (`accumulate_reduce<SUM, REDUCE_ROW>`) | `cb_input_tiles` (Ht_local × Wt streamed in NUM_BLOCKS blocks), `cb_scaler` (1 or 2 tiles, persistent) | `cb_mean` (Ht_local tiles) | `cb_input_tiles` consumed (popped by helper). `cb_scaler` retained (helper waits-never-pops). `cb_mean` holds Ht_local mean tiles. |
| 3 | Pass 2 (per block b ∈ [0, NUM_BLOCKS)): centered subtract | yes (`sub<COL>`) | `cb_input_tiles` (Ht_local × BLOCK_SIZE tiles, A streaming) + `cb_mean` (Ht_local tiles, B `WaitUpfrontNoPop`) | `cb_centered` (Ht_local × BLOCK_SIZE tiles) | `cb_input_tiles` block popped. `cb_mean` retained. `cb_centered` holds the block. |
| 4 | Pass 2 (per block b): square in place | yes (`square_in_place`) | `cb_centered` (Ht_local × BLOCK_SIZE) | `cb_centered` (same count, in place) | `cb_centered` overwritten with `(x − mean)^2`. |
| 5 | Pass 2 (per block b): variance accumulator | yes (`accumulate_reduce_block<SUM, REDUCE_ROW>`) | `cb_centered` (Ht_local × BLOCK_SIZE), `cb_scaler` | `cb_inv_std` (Ht_local tiles — transiently holds variance) | `cb_centered` popped. `cb_inv_std` accumulating; only the last block's call uses the partial scaler. |
| 6 | Pass 2 finalize: variance → inv_std (looped Ht_local times) | yes (`transform_in_place`) | `cb_inv_std` (Ht_local tiles, variance) | `cb_inv_std` (Ht_local tiles, `1/sqrt(var + eps)`) | Each iteration pops one tile, computes `rsqrt(var + eps)`, pushes back into same slot. After loop, `cb_inv_std` semantically holds inv_std. |
| 7 | Pass 3 (per block b): centered subtract | yes (`sub<COL>`) | `cb_input_tiles` (re-streamed for pass 3), `cb_mean` (still persistent) | `cb_centered` | Same as phase 3 — `cb_input_tiles` consumed block by block. |
| 8 | Pass 3 (per block b): × inv_std | yes (`mul_in_place<COL>`) | `cb_centered` (full block, in/out), `cb_inv_std` (B `WaitUpfrontNoPop`) | `cb_centered` (in place) | `cb_centered` now holds normalized values for the block. |
| 9 | (HAS_GAMMA only) Pass 3 (per block b): × gamma | yes (`mul_in_place<ROW>`) | `cb_centered`, `cb_gamma_tiles` (B `WaitUpfrontNoPop`) | `cb_centered` (in place) | `cb_centered` scaled. |
| 10 | (HAS_BETA only) Pass 3 (per block b): + beta | yes (`add_in_place<ROW>`) | `cb_centered`, `cb_beta_tiles` (B `WaitUpfrontNoPop`) | `cb_centered` (in place) | `cb_centered` shifted. |
| 11 | Pass 3 (per block b): drain | yes (`copy_tiles` TILE / `untilize` RM) | `cb_centered` (Ht_local × BLOCK_SIZE) | `cb_output` (TILE: same tile count; RM: 32 × Ht_local sticks) | `cb_centered` popped block-by-block; writer drains `cb_output`. |
| 12 | Kernel teardown: late pops | raw (`cb_pop_front`) | persistent CBs | — | `cb_pop_front(cb_mean, Ht_local)`, `cb_pop_front(cb_inv_std, Ht_local)`, `cb_pop_front(cb_gamma_tiles, Wt)` if HAS_GAMMA, `cb_pop_front(cb_beta_tiles, Wt)` if HAS_BETA, `cb_pop_front(cb_scaler, has_partial_w ? 2 : 1)`. All persistent CBs drained. |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| 3 (Pass 2 centered subtract) | `sub` | `cb_input_tiles` [Ht_local, BLOCK_SIZE] — All | `cb_mean` [Ht_local, 1] — Col0 (REDUCE_ROW output of Pass 1) | `BroadcastDim::COL` |
| 7 (Pass 3 centered subtract) | `sub` | `cb_input_tiles` [Ht_local, BLOCK_SIZE] — All | `cb_mean` [Ht_local, 1] — Col0 | `BroadcastDim::COL` |
| 8 (× inv_std) | `mul_in_place` | `cb_centered` [Ht_local, BLOCK_SIZE] — All | `cb_inv_std` [Ht_local, 1] — Col0 (REDUCE_ROW output of Pass 2) | `BroadcastDim::COL` |
| 9 (× gamma) | `mul_in_place` | `cb_centered` [Ht_local, BLOCK_SIZE] — All | `cb_gamma_tiles` [1, BLOCK_SIZE] — Row0 (gamma is a single row of width W; for the per-block view this is one row across BLOCK_SIZE tiles; for the full Wt the helper picks the relevant block slice via the iteration loop) | `BroadcastDim::ROW` |
| 10 (+ beta) | `add_in_place` | `cb_centered` [Ht_local, BLOCK_SIZE] — All | `cb_beta_tiles` [1, BLOCK_SIZE] — Row0 | `BroadcastDim::ROW` |

**Per-block iteration over gamma/beta tiles:** `mul_in_place<ROW>` / `add_in_place<ROW>` with `cb_b = cb_gamma_tiles` and shape `(Ht_local, BLOCK_SIZE)` use the first `BLOCK_SIZE` tiles of `cb_b`. For NUM_BLOCKS > 1 we need each block's call to see the right `BLOCK_SIZE`-wide slice of gamma. The helper's `BinaryInputPolicy::WaitUpfrontNoPop` waits for `Wt` tiles total in `cb_gamma_tiles`. However, the broadcast addressing for `BroadcastDim::ROW` in `binary_op_helpers.hpp` reads `cb_b` tile `wt` indexed within `[0, cols)` of the shape — which is `BLOCK_SIZE` per call. **To correctly index a different slice per block, the in-place call needs to address `cb_gamma_tiles + b * BLOCK_SIZE`.** Inspecting `binary_op_helpers.inl` confirms there is no built-in offset for the B-side CB. **Decision:** call `mul_in_place<ROW>` once per Pass-3 block with `shape = BinaryInputBlockShape::of(Ht_local, BLOCK_SIZE)`, and ensure the gamma tiles are arranged so the helper indexes correctly. The implementer must verify `binary_op_helpers.inl` handles per-block B-side advancement; if not, the fallback is to issue the per-block `mul_in_place<ROW>` over the FULL `Wt`-wide row at the end of Pass 3 (one call with `shape = BinaryInputBlockShape::of(Ht_local, Wt)` AFTER all blocks' inv_std multiplications), which requires holding the entire Ht_local × Wt result in `cb_centered` — that's `Ht_local × Wt` tiles and exceeds the per-block sizing. **This must be resolved at implementation time** — see "Key Risks and Gotchas" below.

## Key Risks and Gotchas

| Risk | Mitigation / Note |
|------|-------------------|
| **CB sync mismatch on `cb_scaler`** | Reader pushes 1 (no partial) or 2 (partial) tiles; reduce<> waits-never-pops; compute kernel must `cb_pop_front(cb_scaler, has_partial_w ? 2 : 1)` at kernel end. The push count and the final pop count must match. |
| **Scaler CB format** | MUST be `bfloat16` regardless of input dtype. `prepare_*_reduce_scalers` writes `bfloat16` tile layout (reduce_helpers_dataflow.hpp:50-67); other formats cause silent corruption of the scaler tile values. |
| **Pool-type-aware scaler API** | Use `prepare_reduce_scaler<cb, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f/W)` — the FOUR-template-arg overload at reduce_helpers_dataflow.hpp:65-67. NOT the legacy `prepare_reduce_scaler<cb>(value)` — that overload doesn't compile against the pool-type-aware API and the LLK fill pattern differs (row-0 vs col-0). |
| **`transform_in_place` finalizer correctness** | The lambda `add_unary_tile(dst, eps_bits); rsqrt_tile(dst);` requires `add_unary_tile_init()` and `rsqrt_tile_init()` to be issued inside the lambda — these are per-op short-inits, not the full HW init. The helper bundles a copy_tile-to-dst init so the unpacker is ready; the user lambda owns the SFPU-op-specific inits. |
| **`cb_centered` must hold full Pass-3 block before in-place chain** | `binary_op_helpers.hpp:336-342` — "All Ht*Wt tiles of A must already be present in cb_a before calling." Pass 3's `sub<COL>` is `PerTile` output by default (`BinaryOutputPolicy::PerTile`), which pushes after every tile; by the time the in-place chain begins, all Ht_local × BLOCK_SIZE tiles are in `cb_centered`. Confirmed by binary_op_helpers.hpp:163-170. |
| **`Wt < BLOCK_SIZE`** | When `Wt` is small (e.g., W=64 → Wt=2), `BLOCK_SIZE` is chosen as the largest divisor of Wt that is ≤ 8 — for Wt=2 this gives 2, NUM_BLOCKS=1. No degenerate case. |
| **`Wt` not divisible by BLOCK_SIZE** | The `_pick_block_size` helper (per toy_variance:27-36) guarantees BLOCK_SIZE divides Wt. If user-provided BLOCK_SIZE doesn't divide, raise `ValueError`. |
| **Padded W in last W-tile (partial scaler)** | Reader emits two scaler tiles (full + partial). Compute uses `ReducePartialScaler::last_tile_at(1)` on the **last block** of both Pass 1 and Pass 2 reductions — `accumulate_reduce` and `accumulate_reduce_block` route the partial scaler internally. Padded positions contribute 0 to mean and 0 to variance. |
| **Padded W in tile output** | For TILE output, the implicit tile padding (positions `[origin_W, ceil_pad_W)` in the last W-tile) goes through `(garbage − mean) * inv_std * gamma_padded + beta_padded` and ends up with garbage values. TTNN tile-padding convention treats these as "don't care". The validate caller (or `ttnn.fill_implicit_tile_padding(input, 0.0)` upstream) is responsible if upstream `inf`/`nan` would contaminate the variance calculation. The acceptance test must NOT seed the input with non-finite values. |
| **Padded W in RM output** | For RM output, the writer writes only `origin_W * element_size` bytes per stick; padded positions are never written to DRAM. No contamination. |
| **`bfloat8_b` + `ROW_MAJOR_LAYOUT`** | `INVALID` per `feature_spec.py:54` — bfloat8_b is block-quantized and has no row-major layout. The op's `validate()` should raise `ValueError` if this combination is requested. The test harness skips INVALID cells. |
| **`compute_kernel_config` is None** | Python entry point defaults to `ttnn.ComputeConfigDescriptor()` (HiFi4, bf16 dest) before passing to `KernelDescriptor.config`. Do NOT pass `None` through to the descriptor — the pybind layer expects a concrete object. |
| **`fp32_dest_acc_en=True` and DEST limit** | When `fp32_dest_acc_en=True`, `DEST_AUTO_LIMIT` drops from 8 to 4 (per sfpu_helpers.hpp:91-95). The helpers we use respect this automatically. The kernel does NOT pre-acquire more than 4 dst tiles. The `transform_in_place` finalizer uses 1 dst slot only, so it's safe. |
| **`compute_kernel_hw_startup` called exactly once** | Helpers' contract (binary_op_helpers.hpp:16-19, reduce_helpers_compute.hpp:30-34). Never re-call inside a loop. Pick three representative CBs: `compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output)`. |
| **Per-block gamma indexing for `mul_in_place<ROW>`** | See Broadcast Verification note. If `binary_op_helpers.inl` does not advance the B-side CB pointer per block, the implementer must either (a) read all `Wt` tiles into a single full-W pass at the end (raising L1 footprint by `Ht_local * Wt` tiles), or (b) post-pass-3 `mul_in_place<ROW>` over the whole row using a separate `cb_full_result` buffer. **Default plan:** issue `mul_in_place<ROW>` per Pass-3 block with the implicit assumption that the helper's B-indexing is per-block; if testing reveals incorrect output, fall back to plan (b). |
| **Late pops** | `cb_mean`, `cb_inv_std`, `cb_gamma_tiles`, `cb_beta_tiles`, `cb_scaler` are all held with `WaitUpfrontNoPop`. The kernel MUST issue explicit `cb_pop_front` for each before exiting — otherwise the next program launch sees stale `front` pointers. See toy_variance/kernels/compute.cpp:106 and :113 for the pattern. |

## Structural impossibilities

Already encoded in `eval/golden_tests/layer_norm_rm/feature_spec.py:51-64`:
- `{dtype: bfloat8_b, layout: ROW_MAJOR_LAYOUT}` — bf8b has no row-major encoding.
- `{affine_dtype: bfloat8_b, affine_layout: ROW_MAJOR_LAYOUT}` — same.
- `{affine: no_affine, affine_dtype: bfloat16}`, `{affine: no_affine, affine_dtype: bfloat8_b}`, `{affine: no_affine, affine_layout: ROW_MAJOR_LAYOUT}` — canonicalization: when no affine tensors are supplied, the affine_dtype/affine_layout axes have no effect; pick `{affine: no_affine, affine_dtype: float32, affine_layout: TILE_LAYOUT}` as the canonical cell and mark the other 5 combos INVALID to avoid expanding the test matrix redundantly.

No additional structural impossibilities discovered during planning. Note one **inconsistency between user prompt and feature_spec.py**: the prompt states "Gamma and beta are always ROW_MAJOR layout, regardless of whether the input is ROW_MAJOR or TILE layout"; feature_spec.py TARGET allows both layouts for gamma/beta. The user prompt takes precedence for op behavior — the op's `validate()` should reject non-RM gamma/beta. Cells in TARGET where `affine_layout == TILE_LAYOUT` will then fail validate, which is expected to be modeled in the op's `EXCLUSIONS` (or, more correctly, by aligning `feature_spec.py` once `/golden-tests` is re-run).

## Hardware Constraints Checklist

- [x] CB sync: push count = wait count for every CB (verified in "CB sync verification" table).
- [x] Reduce scaler CB is bfloat16 (`cb_scaler` format = `ttnn.bfloat16`).
- [x] Reduce scaler uses pool-type-aware API (`prepare_partial_reduce_scalers<cb, PoolType::SUM, ReduceDim::REDUCE_ROW, partial_w>` / `prepare_reduce_scaler<cb, PoolType::SUM, ReduceDim::REDUCE_ROW>`).
- [x] DEST: max 8 tiles (bf16) / 4 tiles (fp32-dest-acc). `transform_in_place` and `square_in_place` use 1 dst slot; in-place chain ops batch by `DEST_AUTO_LIMIT`. No raw pre-acquires.
- [x] Sequential helper intermediates sized to full block (`cb_centered` = `2 * Ht_local * BLOCK_SIZE`).
- [x] Page sizes aligned: tile CBs use `ttnn.tile_size(dtype)`; RM CBs use `padded_row_bytes = ceil(W * elem_size, dram_alignment)` via `tensor.buffer_aligned_page_size()` or explicit ceil.
- [x] RM CBs count pages in sticks (`cb_input_sticks`: 2 * 32 * Ht_local sticks), tile CBs count in tiles.
- [x] All `cb_wait_front` calls on same CB use same page count (helpers enforce this internally — verified by reading binary_op_helpers.inl and reduce_helpers_compute.inl).
- [x] `compute_kernel_hw_startup` called exactly once, at start of `kernel_main`, with `(cb_input_tiles, cb_scaler, cb_output)`.
