# Operation Design: avg_pool2d

The specification governs: `eval/specs/avg_pool2d/spec.py` (signature, `reference`, `TARGET`,
`INPUTS`, `INVALID`, `AXES`, `PHASE0`, `RULES` R1–R12, `REJECTIONS` X1–X3, `NUMERIC`). This
document is how to build it. Where the two disagree, the specification is right.

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (a reduction), bound by data movement |
| Goal | Global average pool: for one image of H×W positions with C channels, flattened as `[1, 1, H*W, C]`, return the per-channel mean `[1, 1, 1, C]` as one native device program on Quasar. |
| Math | `out[n, c] = (1 / (H*W)) * Σ_{i<H, j<W} in[(n*H + i)*W + j, c]` (Phase 0: n = 0 only) |
| Reduction in tile terms | Each input tile is 32 positions × 32 channels. A channel tile column `ct` is reduced over its `Ht = ceil(H*W/32)` tiles along the tile's H direction: `REDUCE_COL`, result in row 0 of one output tile. The 1/(H*W) and the R1 padding mask are both carried by the reduce's scaler row. |
| References | spec `eval/specs/avg_pool2d/spec.py`; target `.claude/references/targets/quasar.md`; backend `.claude/references/backends/ttsim.md`; examples `.claude/references/quasar_op_examples/` (`toy_row_passes`, `toy_thread_plan/kernels/reader_mixed.cpp`); helpers `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`, `reduce_helpers_dataflow.hpp`, `dest_helpers.hpp`; catalog `ttnn/ttnn/operations/examples/master.md` (`width_split`, `double_buffer`, `compute_block_size`, `reduce_block`, `reduce_accumulate`) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on the ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | `quasar` | target page |
| Compute grid | 8×4 = 32 clusters, fixed on this part; queried at run time with `device.compute_with_storage_grid_size()` | target page |
| L1 per core | 4,194,304 B (4 MiB) per cluster, shared by its 4 NEOs and 8 DM cores | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes`; on this target ≤ 4,194,304 B (exact value is the query's, not probed). No predicate in this design spends it: the footprint is ≤ 69,632 B per cluster, independent of every op dimension (`l1_ledger.md`). | target page |
| Host program | `ttnn.ProgramSpec` (Metal 2.0: `ttnn.experimental.ProgramSpec`, `KernelSpec`, `DataflowBufferSpec`, `TensorParameter`, `WorkUnitSpec`, `ProgramRunArgs`) dispatched with one `ttnn.generic_op` call — reference `.claude/references/quasar_op_examples/` (README, then `toy_row_passes/toy_row_passes_program_artifacts.py`) | target page |
| Threads per core | 4 NEO compute engines (one compute thread each) and 6 user DM harts; plan (R, C, W) with C ∈ {1, 2, 4}, R + W ≤ 6, R | C and W | C on STRIDED rings; ≤ 24 active DFBs per NEO; DFB ring ≤ 1,048,560 B | target page, *Cluster model* / *Thread plan* |
| Unavailable here | (1) `ttnn.ProgramDescriptor` / Metal 1 kernels and the legacy C++ program factory: refused at kernel creation (target page). (2) `bfloat8_b`, `bfloat4_b`: no path on the part (target page). (3) `ReduceFp32Mode::Accurate`: static_assert on Quasar (`reduce_helpers_common.hpp:41-50`). (4) `ReduceAlgorithm::AccumulateViaAdd` (and `reduce_mean<>`, which defaults to it): not used — only `ReduceTile` is used. (5) An SFPU `post_reduce_op` inside `reduce<>`: not used — the 1/(H*W) is carried by the scaler, never by a post-op. (6) The reduce helper's indexed policies (`BulkWaitBulkPop`, `WaitUpfrontNoPop`) under a thread plan with C > 1: not used — this design streams with `WaitAndPopPerTile`. (7) `UnpackToDest` of a bfloat16 page into a 32-bit DEST: not used — every ring keeps the default unpack mode. (8) uint16/uint32 DFB formats: refused (not needed). (9) An untagged NoC transfer after an implicit-sync program — see Key Risks: the scaler fill's `write_zeros` is untagged. (10) `fpu_math_fidelity`: always set from the caller's config. | target page; helper headers |

Every worked number below is for this target (32 clusters) and says so; every predicate the op
evaluates queries the device.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | `[1, 1, batch_size*input_h*input_w, channels]` (X1); Phase 0: bfloat16, TILE, DRAM/L1 interleaved or L1 width-sharded | — | tensor |
| `batch_size` | int | yes | ≥ 1; Phase 0: 1 (`batch` = single) | — | host (validate) |
| `input_h` | int | yes | ≥ 1 | — | host → `Ht`, `inv_hw` |
| `input_w` | int | yes | ≥ 1 | — | host → `Ht`, `inv_hw` |
| `channels` | int | yes | ≥ 1; Phase 0: multiple of 32 | — | host → `tensor_ct_tiles` |
| `kernel_size` | list/tuple of 2 ints | yes | each ≥ 1 (X2, X3); `== (input_h, input_w)` with zero padding is the only window (`window` = global) | — | host (validate only) |
| `stride` | list/tuple of 2 ints | yes | each ≥ 1 (X2, X3); any value accepted, no effect for the global window (R6) | — | host (validate only) |
| `padding` | list/tuple of 2 or 4 ints | yes | each ≥ 0 (X2, X3); all zero for the global window | — | host (validate only) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` or device config (`ttnn.WormholeComputeKernelConfig`) or None | kw-only | any `math_fidelity`, any `math_approx_mode` (R3); `fp32_dest_acc_en` ∈ {True, False} (R5) | `None` → `default_compute_kernel_config()` = HiFi4, `math_approx_mode=False`, `fp32_dest_acc_en=True` (R2) | host → `ComputeHardwareConfig` |
| `dtype` | `ttnn.DataType` | kw-only | bfloat16 only (`output_dtype`, R8) | `ttnn.bfloat16` | host |
| `output_layout` | `ttnn.Layout` | kw-only | Phase 0: TILE | `ttnn.ROW_MAJOR_LAYOUT` (the signature's default; refused in Phase 0 by `output_layout` gating) | host |

Host normalization, in this order, before the support check: (1) `kernel_size`, `stride`, `padding`
are converted to tuples whatever their form (R6); `padding` of 2 values becomes `(top, bottom,
left, right) = (p0, p0, p1, p1)`. (2) Malformed calls raise (X1 `ValueError`: shape is not `[1, 1,
N*H*W, C]`; X2 `TypeError`/`ValueError`: wrong arity or non-int; X3 `ValueError`: kernel/stride < 1
or padding < 0). (3) Axes: `eval.memory.memory_axes(input_tensor)` for `memory_config`/`sharding`
(R11), `AXES` from the specification for `window`/`batch`/`channel_alignment` (imported, never
redefined: `INPUT_TAGGERS = AXES`), `fp32_dest_acc_en` read as an attribute off the resolved config
(R4); every axis is gated against `SUPPORTED` with `NotImplementedError` (R12) — this is where a
`sliding` window, a block/nd-sharded input, a row-major layout, a batch, or a non-bfloat16 `dtype`
is refused by name. (4) The compute config is normalized to a `ttnn.ComputeConfigDescriptor`
(`math_fidelity`, `math_approx_mode`, `fp32_dest_acc_en` read as attributes; `packer_l1_acc` and
`throttle_level` ignored; R4) and mapped to `ttnn.experimental.ComputeHardwareConfig(
fpu_math_fidelity=math_fidelity, enable_32_bit_dest=fp32_dest_acc_en, double_buffer_dest=True,
sfpu_precision_mode=<Approximate if math_approx_mode else Precise>)`. `fpu_math_fidelity` is always
set explicitly.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[1, 1, N*H*W, C]`, row `(n*H + i)*W + j` = position (i, j) of image n (R7). Phase 0: N = 1. |
| Dtype | bfloat16 |
| Layout | TILE (Phase 0). Padded shape `[1, 1, 32*Ht, 32*tensor_ct_tiles]`; rows ≥ H*W and columns ≥ C are padding of undefined content (R1). |
| Memory | DRAM interleaved, L1 interleaved, L1 width-sharded on any shard grid (Phase 0, R10). All read through the tensor's `TensorAccessor` (one path for every placement). |

### Output

| Property | Value |
|----------|-------|
| Shape | `[1, 1, N, C]` (Phase 0: `[1, 1, 1, C]`); padded `[1, 1, 32, 32*tensor_ct_tiles]` in TILE |
| Dtype | bfloat16 (`dtype`) |
| Layout | `output_layout` (Phase 0: TILE) |
| Memory | Interleaved, in the input's buffer type (R9): `ttnn.MemoryConfig(INTERLEAVED, input.memory_config().buffer_type)`, allocated with `ttnn.allocate_tensor_on_device(spec, device)` — L1 for any L1 input (interleaved or sharded), DRAM for a DRAM input. Never sharded. |

## Blocking Model

The first design section. Everything below it is downstream of these decisions.
Semantics: `.claude/references/blocking-model.md`.

### Axes

The op's data has three axes. Images (`batch`) and positions are contiguous in the flattened row
index, but they are *not* folded: positions are reduced and images are not.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `channel` (tile columns, `tensor_ct_tiles = ceil(C/32)`) | **independent** — every output channel is the mean of its own column only; no value crosses columns | `block_ct_tiles` (channel tiles per block) | the core's whole assignment: `block_ct_tiles = core_ct_tiles` → `num_blocks_this_core = 1` (model, 32 clusters: 2 tiles) | host constant `BLOCK_CT_TILES_CAP` (Phase 0 = unbounded → whole assignment), clamped per core: `block_ct_tiles = min(core_ct_tiles, BLOCK_CT_TILES_CAP)`; passed as RT arg | **spread across the full grid**: `ttnn.split_work_to_cores(grid, tensor_ct_tiles)` → each engaged cluster owns a contiguous range `[ct_start, ct_start + core_ct_tiles)`; grid from `device.compute_with_storage_grid_size()` | knob-turn (more/fewer cores, larger/smaller blocks) |
| `positions` (tile rows of one image, `Ht = ceil(H*W/32)`) | **dependent** — every output is a sum over all positions (a reduction) | `block_pos_tiles` | the whole extent: `block_pos_tiles = Ht` (model: 2; Phase 0 max 7) — the whole column accumulates in DEST inside one reduce call, no cross-call accumulator | derived on host, `Ht = ceil(input_h*input_w / 32)`; RT arg | **not assigned across cores** — cutting it needs a cross-core combine that moves more bytes than the channel split (Traffic ranking); deferred row `positions_split_combine` | scheme-change |
| `batch` (images, N) | **independent** — images are averaged separately | `block_images` | 1 (Phase 0 has N = 1 only: `batch` = single; a block cannot exceed the one image there is) | host, `batch_size`; RT arg | not assigned in Phase 0 (one image); in the batched regime images are a second independent axis spread with channels | scheme-change (output-row placement and straddling tiles: regime `batched_averaging_matrix`) |

**Inside a block (the DEST chunk).** `reduce<..., REDUCE_COL>` holds one output tile per channel
column in DEST for the whole positions walk, so the helper walks a block's channels in chunks of
`dest_chunk_tiles = DEST_AUTO_LIMIT` columns (`reduce_helpers_compute.inl:1163-1182`). This is the
helper's internal decomposition, not a block knob, but the reader must emit tiles in the helper's
streaming order (chunk, then position tile row, then channel within the chunk;
`reduce_helpers_compute.inl:1157`), so `dest_chunk_tiles` is a declared mechanism cap below with a
single host source.

**Stages and their data.** The scheme has one compute stage (the column reduce) and two constant
producers. The reduce's output (`C_t` tiles, row 0 valid) has the `channel` axis only — independent,
and already split with the input's channel split; nothing is collected on a designated core, so no
stage computes on one cluster while the others wait. The scaler pair is per cluster, built locally
by each engaged cluster's writer (no communication role, no shared operand crossing the NoC).

**Every knob is a parameter.** `block_ct_tiles`, `Ht`, `core_ct_tiles`, `ct_start`,
`num_blocks_this_core` are runtime args; `dest_chunk_tiles`, `INPUT_DEPTH_CHUNK_ROWS`,
`OUTPUT_DEPTH_CHUNKS` and the thread-plan counts are host constants passed as compile-time args;
the core count is the queried grid, never a literal. Loops are written as
`for (block_idx = 0; block_idx < num_blocks_this_core; ++block_idx)` with the block's extents as
parameters — no stride-over-element-count loops.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `INPUT_DEPTH_CHUNK_ROWS` (host constant; capacity = `INPUT_DEPTH_CHUNK_ROWS × dest_chunk_tiles` tiles) | 2 | The reader keeps two chunk rows of reads in flight while compute unpacks the previous one (catalog `double_buffer`: ~4–8 reads in flight, capacity 2× the in-flight group). |
| `cb_scaler` | none (constant pair) | 2 pages, filled once | — (constant: full scaler + partial scaler) |
| `cb_output_tiles` | `OUTPUT_DEPTH_CHUNKS` (host constant; capacity = `OUTPUT_DEPTH_CHUNKS × dest_chunk_tiles` tiles) | 2 | Compute packs the next DEST chunk while the writer drains the previous one (relevant where a core owns more than one chunk; e.g. C = 8192 in TARGET). |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `reduce<..., REDUCE_COL>` holds one DEST tile per output column of a chunk: chunk = `DEST_AUTO_LIMIT` (`dest_helpers.hpp:89-103`: half-sync 4 for 32-bit DEST, 8 for 16-bit; full-sync 8/16) and the streaming order is (chunk, position row, column) (`reduce_helpers_compute.inl:1157-1207`) | the reader's chunk width (the order it emits tiles in), and the capacity unit of `cb_input_tiles` / `cb_output_tiles` | host derives `dest_chunk_tiles = (4 if fp32_dest_acc_en else 8) × (1 if double_buffer_dest else 2)` — the same rule as `dest_helpers.hpp:95-100` — from the same `ComputeHardwareConfig` it hands the compute kernel (`double_buffer_dest=True` in Phase 0); passed as CT arg to reader and compute; the compute kernel `static_assert(DEST_AUTO_LIMIT == dest_chunk_tiles)` | tiles of one column are reduced into another column's DEST slot: silently wrong channels |
| Partial scaler (`ReducePartialScaler::with_partial()`, `reduce_helpers_compute.hpp:350-357`; fill `prepare_reduce_scaler(…, valid)`, `reduce_helpers_dataflow.inl:171-236`) masks only the **last** tile of the reduce dim of each `reduce()` call, and only a **prefix** of its rows | `block_pos_tiles` must be the whole image column (`= Ht`); `last_tile_valid_rows ∈ [1, 32]` | `last_tile_valid_rows = H*W − 32·(Ht − 1)` (∈ [1, 32] by construction of `Ht = ceil`); a value of 32 fills the second scaler fully, so `with_partial()` is used unconditionally and is a no-op mask for tile-aligned H*W | a positions block shorter than `Ht` would mask valid rows of a middle tile; a 0 would zero a whole tile; padding rows would enter the sum (R1, P1 offset) |
| DEST accumulation length through the FPU reduce (tf32 round trip, ~1.8e-4 bias per tile) | `block_pos_tiles = Ht` per column | none needed in Phase 0 (Ht ≤ 7 in PHASE0 ∩ INPUTS); TARGET's 1024-position entry (Ht = 32) is outside Phase 0 (C = 50) — a refinement admitting it re-checks the tolerance | relative error grows with Ht, mostly with a 16-bit DEST |
| Output page = one tile; the one-image output is one tile row | `block_images` | 1 in Phase 0 (`batch` = single is gated) | n/a in Phase 0; images > 1 need their means in rows n%32 of shared output tiles (regime `batched_averaging_matrix`) |
| DFB limits: ≤ 24 active DFBs per NEO, ≤ 1,048,560 B per ring | capacities | 3 rings, ≤ 32 KB each | program build refuses |

### Regimes

| Regime | Status | Predicate | Block | Data movement (vs. minimum) | What a bigger block buys |
|--------|--------|-----------|-------|-----------------------------|--------------------------|
| `col_split_stream` | **built** (Phase 0) | `window == "global"` and `batch == "single"` (every Phase 0 cell, every placement) | `block_pos_tiles (= Ht) × block_ct_tiles (= core_ct_tiles)`, one image | **Minimum** against the input's own tier: each input tile crosses once (DRAM→cluster for DRAM inputs; L1→L1 NoC for L1 interleaved/sharded inputs), each output tile crosses once, scaler is built in place (0 B moved). Model, 32 clusters: input 262,144 B once, output 131,072 B once (31 of 32 rows of each output tile are tile padding — unavoidable in TILE output). | Per block: one `reduce_init` + `pack_init` + scaler `wait_front` + one `reduce_uninit` (intended **once per block** = once per cluster in Phase 0), the unpack/math/pack pipeline fill/drain once per DEST chunk. A block below the whole assignment would repeat these per block with no traffic saved (catalog `compute_block_size`). |
| `shard_in_place_width` | **deferred** — the built regime serves every width-sharded shape on any shard grid (R10) through one accessor path, and on the spec's traced call (2 shards) it is the better trade: this regime would confine the reduce to the caller's shard clusters (2 of 32), while `col_split_stream` spends one L1→L1 crossing of the input (262,144 B, no DRAM) to engage 32. Bytes are traded for occupancy explicitly (blocking-model §4). It wins where the shard grid already spans ≥ the clusters the channel split engages. | (when built) `sharding == "width"` and `num_shard_cores ≥ min(tensor_ct_tiles, grid_cores)` | the whole resident shard: `Ht × shard_ct_tiles` (one block, the sharded default) | input moves 0 B (consumed in place from a ring borrowed on the shard, `ttnn.experimental.dfb_spec_from_sharded_tensor(..., borrowed_from=...)`); output once. Saves the input's one L1→L1 crossing vs `col_split_stream`. | same fixed costs, once per shard |
| `positions_split_combine` | **deferred** — it moves strictly more bytes than the channel split (input once + one partial tile per column per extra cluster + a rendezvous) and only adds occupancy where `tensor_ct_tiles` < grid, which in Phase 0 are the shapes with ≤ 7 tiles per column (C = 32: 2–7 tiles in total); no residency gain exists to buy (the input is read once in every regime, nothing is re-read). It is also the in-place regime for a **height-sharded** input (the shard cuts the dependent axis), which `col_split_stream` already serves without a combine by reading across shard cores. | (when built) `tensor_ct_tiles < grid_cores` and `Ht ≥ positions_split_min_tiles`, or `sharding == "height"` in place | `block_pos_tiles = ceil(Ht / pos_splits) × block_ct_tiles` per cluster, partials combined on one cluster per channel block | input once; adds `(pos_splits − 1) × block_ct_tiles` partial tiles over the NoC per channel block plus one semaphore round trip | fewer combine rounds; the combine's partial count scales with clusters per column, not with block size |
| `batched_averaging_matrix` | **deferred** — Phase 0 is one image (`batch` = single is the PHASE0 rectangle); the regime needs the averaging matrix built and a matmul primitive written with LLK calls | (when built) `batch == "batched"` | `(32 images × their tile rows) × block_ct_tiles`: the block is an output tile row (32 images) by a channel block | input once, except a tile shared by two images when `H*W % 32 ≠ 0` (read once per image block that overlaps it: ≤ 2 crossings of those tiles); output once | one matrix build per row group, amortized over every channel block of the cluster |
| `reduce_via_add_sfpu_finalize` | **rejected** — superseded by `col_split_stream`'s FPU matmul-reduce: the catalog (`reduce_accumulate`, `reduce_block`) puts its col crossover at ≥ 8 reduced tiles and Phase 0 has ≤ 7; `ReduceFp32Mode::Accurate` is guarded out | — | — | same as `col_split_stream` | — |
| `single_cluster` | **rejected** — a single-core Phase 0 is not allowed; superseded by `col_split_stream`, whose `channel` split is the same kernels with more clusters | — | — | — | — |

**Regime selection (Phase 0):** exactly one built regime; every call that passes `validate()`
dispatches `col_split_stream`. No selection function, no regime-pinned tests needed until a second
regime is built; the deferred rows carry their predicates for that day.

**Reachability of the deferred rows.** `shard_in_place_width`: the compute kernel and its block
operation are unchanged (it already reduces an `Ht × block_ct_tiles` block streamed tile by tile
from a ring); the reader becomes a credit-only producer on a borrowed ring, and the host split is
read off the shard spec instead of `split_work_to_cores`. `positions_split_combine`: the reduce
call already takes `block_pos_tiles` as a runtime extent; the regime adds a gather of partial tiles
(`ttnn.Mcast` / semaphores) and one SUM reduce of the partials on the owning cluster, with the
`1/(H*W)` scaler applied to the grand total only. `batched_averaging_matrix`: the scaler ring
becomes the averaging-matrix ring (built locally by the writer, the same way), the reader's stream
and the channel split are unchanged, and the compute primitive changes from `reduce_tile` to an LLK
matmul of the matrix tiles with the input tiles (each output row n = mean of image n; padding rows
and the other image's rows get matrix entries 0 — the same masking idea as the partial scaler).

### Traffic ranking

Counted per tensor crossing, for one image `Ht × tensor_ct_tiles` input tiles (B = 2048 B/tile).

| Candidate split | Input crossings | Output crossings | Cross-core traffic | Rank |
|-----------------|-----------------|------------------|--------------------|------|
| `channel` across clusters (`col_split_stream`) | 1 (`Ht·C_t·B`) | 1 (`C_t·B`) | none (for L1 inputs the input crossing *is* L1→L1) | **1 (chosen)**: minimum, no combine |
| `channel` across the caller's shard grid, zero copy (`shard_in_place_width`) | 0 for width-sharded inputs | 1 | none | cheapest bytes for width-sharded inputs only; traded against occupancy (deferred row) |
| `positions` across clusters + combine (`positions_split_combine`) | 1 | 1 | `(splits − 1)·block_ct_tiles·B` partials + semaphore round trip per channel block | 3: strictly more bytes than rank 1 |
| `batch` across clusters | n/a in Phase 0 (N = 1) | | | (batched regime) |

No operand is re-read: the only shared operand is the scaler pair, which does not vary along any
split axis — the **operand-reuse check** finds it reuse-shared by construction, and the answer is to
*build* it per cluster in L1 (2 tiles, 0 NoC bytes), not to broadcast it. **Stall-shadow check:**
the compute waits on the reader's first tiles; the scaler fill is placed on the **writer**, whose
first real work (draining output) cannot start before the reduce finishes, so the fill runs in the
writer's stall while the reader's first reads are in flight. Nothing else is independent of the
wait. Summation order changes nothing beyond float rounding (associative/commutative sum), and the
order is the helper's fixed order — no reorder to gate on the precision baseline.

### Block schedule

```cpp
// Per engaged cluster. Logical schedule: reader, compute and writer run asynchronously.
fill_scaler_pair();                                   // writer, once per cluster
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // Phase 0: 1
    load_column_block(block_idx);                     // reader
    reduce_column_block(block_idx);                   // compute
    store_mean_block(block_idx);                      // writer
}
release_scaler_pair();                                // compute, once per cluster
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `fill_scaler_pair` | 2 constant tiles: tile 0 = `inv_hw` in row 0 of every face (all 32 position lanes), tile 1 = `inv_hw` in the first `last_tile_valid_rows` position lanes, zero elsewhere | stays at the front of `cb_scaler` for every block of the cluster | once per cluster; one `write_zeros` + barrier + L1 cache flush per tile (inside the helper) |
| `load_column_block` | `block_pos_tiles × block_ct_tiles` input tiles, emitted in the helper's streaming order: for each DEST chunk of the block (`dest_chunk_tiles` wide, last one narrower), for each position tile row `ht < Ht`, for each channel tile of the chunk: page `ht·tensor_ct_tiles + ct` | nothing (streamed through a `INPUT_DEPTH_CHUNK_ROWS × dest_chunk_tiles` window) | one tagged implicit-sync read per tile (no barrier, no push: DM0 posts credits); no per-block setup |
| `reduce_column_block` | one `reduce<SUM, REDUCE_COL>` call over `ReduceInputBlockShape::of(Ht, block_ct_tiles, 1)` | the scaler pair; each DEST chunk's `chunk` accumulators for the whole positions walk | `reduce_init` + `pack_init` + `reduce_uninit` **once per block**; DEST acquire/commit/release once per DEST chunk; one `reduce_tile` per input tile; partial scaler on the last position row of each chunk |
| `store_mean_block` | `block_ct_tiles` output tiles, pages `ct_start + block offset + j` of the output's single tile row | nothing | one tagged implicit-sync write per tile; tag handed back once at the end of the kernel |
| `release_scaler_pair` | 2 pages | — | one `pop_front(2)` after the last block |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization | Maximum participation leaves 2–4 tiles per cluster for many Phase 0 shapes (model: 4 tiles per cluster on 32 clusters; C = 64: 2 clusters × 2 tiles); per-cluster fixed cost (launch, scaler fill, reduce init) may dominate | cap the engaged clusters so each owns ≥ `dest_chunk_tiles` channel tiles (e.g. model: 8 or 16 clusters), against the full grid |
| Thread plan (data-movement bound, plan (1,1,1)) | The bound is argued, not measured; with the input already in L1 (sharded/L1-interleaved) the reduce's per-tile unpack+GAPOOL may come close to the read cost | (1,4,1) with whole DEST chunks dealt to NEOs (a NEO per chunk of columns, rounds padded), measured on a timed backend (ttsim-perf) against (1,1,1) |
| Reconfig mode | `INPUT_AND_OUTPUT` reconfig per call is wasted MMIO when the reduce is the kernel's only operation (catalog `compute_block_size`, second lever) | `ReduceDataFormatReconfigMode::NONE` (formats are set once by `compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles)` and never change) — Phase 0 uses NONE; measure against INPUT_AND_OUTPUT only if a format question arises |
| Width-sharded read shape | Contiguous channel tiles of one shard row sit back to back in one cluster's L1; per-page reads fragment them (catalog `read_coalescing`) | one wide read per (shard, position row) range instead of one per tile, where the accessor reports physical contiguity |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input → `cb_input_tiles` | bfloat16 tiles (2048 B) | reader, `noc.async_read<NocOptions::TXN_ID>(input_accessor, cb_input_tiles, {.page_id = ht*tensor_ct_tiles + ct}, {})`: **implicit sync**, one tagged read per ring entry | the same accessor read for DRAM-interleaved, L1-interleaved and L1-width-sharded inputs (the accessor maps a page to its bank or shard core) |
| Writer → `cb_scaler` | bfloat16 tiles, row 0 of each face | writer, `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_COL>(scaler, inv_hw, 32)` then `(..., inv_hw, last_tile_valid_rows)`: **explicit sync** (the helper reserves/pushes and writes the entry itself), ring listed in the writer's `disable_dfb_implicit_sync_for` | `inv_hw = 1.0f / (H*W)` computed on host in fp32, passed as its bit pattern (RT arg); rounded to bfloat16 by the fill (≤ 2⁻⁹ relative) |
| `cb_input_tiles` + `cb_scaler` → DEST → `cb_output_tiles` | bfloat16 in; DEST fp32 or 16-bit per `fp32_dest_acc_en`; bfloat16 out | compute, one `compute_kernel_lib::reduce<SUM, REDUCE_COL, ...>` call per block | the FPU's two sources are both genuine bfloat16 data (input, scaler) — no fp32 data is unpacked under a bf16 source format |
| `cb_output_tiles` → output | bfloat16 tiles | writer, `noc.async_write<NocOptions::TXN_ID>(cb_output_tiles, output_accessor, {}, {.page_id = ct})`: **implicit sync**; last statement `noc_async_write_set_trid(0, noc.get_noc_id())` | output interleaved in the input's buffer type (R9) |

**As built in Phase 0 (verifier note).** The `cb_input_tiles` row above is superseded: the reader runs that ring in **explicit sync** (opted out on the host, user transaction id 0 on every transfer), one reserve / one barrier / one push per chunk row, and zeroes rows `[last_tile_valid_rows, 32)` of each last-position-row tile in L1 before the push. The partial scaler stays (harmless with zeroed padding). `l1_ledger.md` carries the same note; footprint and bytes are unchanged.

**TARGET axes as realizations of the scheme** (each `TARGET − PHASE0` value is a refinement):

| TARGET value outside Phase 0 | Which axis / mechanism it touches | Cost class |
|-------------------------------|-----------------------------------|------------|
| `sharding` = height (L1) | cuts `positions` (dependent). `col_split_stream` reads it through the accessor across shard cores with no combine: a validation change. In place it is `positions_split_combine`. | knob-turn (via accessor); scheme-change (in place) |
| `sharding` = width (Phase 0 already) zero copy | cuts `channel` (independent): `shard_in_place_width` | knob-turn on the placement (reader swap + split read off the shard) |
| `channel_alignment` = non_tile_aligned | `channel` padding columns: a column never enters another column's sum, so no mask is needed; `tensor_ct_tiles = ceil(C/32)` already | knob-turn (validation) |
| `layout` = ROW_MAJOR input | pages become sticks (one position each): the reader assembles 32 sticks × a chunk of channels into tile-shaped entries and a `tilize` phase precedes the reduce, or the reader sums sticks with a row-major reduce | refinement (adds a tilize phase; blocks unchanged) |
| `output_layout` = ROW_MAJOR | `untilize` of the output tiles, the writer stores row 0 (one stick of `block_ct_tiles·32` channels per image) | refinement (adds an untilize phase) |
| `batch` = batched | `batched_averaging_matrix` | scheme-change |
| `fp32_dest_acc_en` | both values in Phase 0; sets `dest_chunk_tiles` (4 / 8) | — |

**Deferred inter-cluster contracts.** `positions_split_combine`: each non-owner cluster of a channel
block packs its `block_ct_tiles` partial sums (scaler 1.0, unmasked except its own last tile) into
a partial ring, unicasts them to the owner's gather ring and increments the owner's semaphore; the
owner waits for `pos_splits − 1` arrivals and runs one SUM reduce of the partials with scaler
`inv_hw`. `shard_in_place_width`: no inter-cluster traffic.

**Structural impossibilities:** none to add to the specification's `INVALID`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a channel tile column of one image: `Ht` input tiles → 1 output tile |
| Grid | `device.compute_with_storage_grid_size()` (32 clusters on this target), passed to `ttnn.split_work_to_cores(grid, tensor_ct_tiles, row_wise=True)` |
| Per-core work | `core_ct_tiles` ∈ {`per_core_1`, `per_core_2`} channel tiles starting at `ct_start` (running sum over the groups, as `split_rows` in `toy_row_passes_program_artifacts.py:42-58`); `num_blocks_this_core = ceil(core_ct_tiles / block_ct_tiles)` = 1 in Phase 0; the last block's runtime extent is `core_ct_tiles − (num_blocks_this_core − 1)·block_ct_tiles` |
| Remainder | `tensor_ct_tiles = ceil(C/32)`, `Ht = ceil(H*W/32)` per image (never `floor` or `//` on a flattened row count); `split_work_to_cores` gives group 2 one tile less; DEST chunks inside a block: last chunk `block_ct_tiles − k·dest_chunk_tiles` wide, same in reader and helper |
| Worked Phase 0 values (32 clusters) | model `7×7×2048`: C_t = 64 → 32 clusters × 2 tiles, Ht = 2, `last_tile_valid_rows` = 17; `7×7×512`: 16 × 1; `7×7×256`: 8 × 1; `7×7×64`: 2 × 1; `4×8×64`: 2 × 1, Ht = 1, valid = 32; `8×8×32`: 1 × 1, Ht = 2, valid = 32; `14×14×32`: 1 × 1, Ht = 7, valid = 4; `3×5×96`: 3 × 1, Ht = 1, valid = 15 |
| Bound | **data movement**: each input tile (2048 B, moved once) receives one FPU `reduce_tile` and nothing else — one FPU op per tile, the target page's data-movement case |
| Thread plan | **(R, C, W) = (1, 1, 1)**, host constant `THREAD_PLAN = (1, 1, 1)` passed to every kernel as compile-time args `reader_threads`, `compute_threads`, `writer_threads`; the kernels size their loops from them. Reason: bound by data movement. |
| Unit dealt to compute threads | n/a at C = 1 (the one NEO takes every entry). The host split is in channel tiles; if a refinement raises C, the unit is a DEST chunk of columns (larger than an entry), rounds padded and whole rounds of C chunks dealt to clusters (target page rule) |
| Tests the plan needs | (1) a test that reads the plan off the host program artifacts and checks (1, 1, 1) and that every engaged cluster's channel range is covered by the output (each output tile's channels nonzero-checked per cluster range with a channel-ramped input); (2) one device session: the op after another program (e.g. a `ttnn.to_layout` round trip of an unrelated tensor) and the op followed by `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)` of its own output — both must finish and be correct |

## Circular Buffers

Dataflow buffers (DFB rings). Capacity counted per compute thread × C (C = 1).

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `cb_input_tiles` | 0 | 2048 B (bf16 tile) | `INPUT_DEPTH_CHUNK_ROWS × dest_chunk_tiles` (Phase 0: 2 × 4 = 8 at fp32 DEST, 2 × 8 = 16 at 16-bit DEST) | streams over `positions` and `channel` (window = two chunk rows in flight); live set for compute is 1 tile (`WaitAndPopPerTile`) | Float16_b | reader | compute | reader: implicit (tagged read per entry); compute: wait/pop per tile (helper) | whole program |
| `cb_scaler` | 1 | 2048 B | 2 | constant: [full scaler, partial scaler] | Float16_b | writer | compute | writer: explicit (`prepare_reduce_scaler` reserves/pushes; ring in writer's `disable_dfb_implicit_sync_for`); compute: waits 2 (helper, never pops), compute pops 2 after the last block | whole program |
| `cb_output_tiles` | 2 | 2048 B | `OUTPUT_DEPTH_CHUNKS × dest_chunk_tiles` (Phase 0: 8 / 16) | spans one DEST chunk of `channel` (the chunk's tiles are packed back to back), double-buffered by chunk | Float16_b (output ABI: bfloat16 result; the sum is complete in DEST before the single pack) | compute | writer | compute: reserve/push per tile (helper); writer: implicit (tagged write per entry) | whole program |

Push count = wait count per ring: `cb_input_tiles` reader pushes (via DM0 credit) `Ht × block_ct_tiles`
per block, compute pops the same 1 by 1; `cb_scaler` writer pushes 2, compute waits 2 and pops 2;
`cb_output_tiles` compute pushes `block_ct_tiles` per block, writer drains the same.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `fill_scaler_pair` | 2 tiles | yes (`prepare_reduce_scaler` ×2) | — | `cb_scaler`, 2 | 2 pages at the front, held for the whole program |
| 1 | `load_column_block` | `Ht × block_ct_tiles` | raw NoC API (implicit-sync tagged reads, as the Quasar examples) | input tensor pages | `cb_input_tiles`, 1 per read | — |
| 2 | `reduce_column_block` | `Ht × block_ct_tiles` (chunks of `dest_chunk_tiles` internal) | yes (`reduce<SUM, REDUCE_COL, cb_input_tiles, cb_scaler, cb_output_tiles, WaitAndPopPerTile, NONE>`) | `cb_input_tiles`, 1 at a time, popped; `cb_scaler`, 2, waited not popped | `cb_output_tiles`, `block_ct_tiles` | input consumed; scaler still held |
| 3 | `store_mean_block` | `block_ct_tiles` | raw NoC API (implicit-sync tagged writes) | `cb_output_tiles`, 1 per write | output tensor pages | — |
| 4 | `release_scaler_pair` | 2 | raw (`cb_scaler.pop_front(2)`) | `cb_scaler`, 2 | — | empty |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| compute boot | raw_api | `compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60` | once, first statement of the compute kernel | — | — | — |
| `fill_scaler_pair` | helper | `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_COL>(scaler_dfb, inv_hw, valid)` | decl `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:64-71`; impl `reduce_helpers_dataflow.inl:171-236` (Quasar L1-cache flush `:230-235`) | call 1: `valid = 32`; call 2: `valid = last_tile_valid_rows`; `scaler_dfb` constructed once at kernel scope | — | `cb_scaler` | none (constant) |
| `load_column_block` | raw_api | `noc.async_read<NocOptions::TXN_ID>(input_accessor, cb_input_tiles, {.page_id = ht*tensor_ct_tiles + ct}, {})`; end: `noc_async_read_set_trid(0, noc.get_noc_id())` | pattern `.claude/references/quasar_op_examples/toy_row_passes/kernels/reader.cpp:45,53` | loop order: chunk → ht → ct in chunk | — | `cb_input_tiles` | `block_ct_tiles`, `Ht` (RT), `dest_chunk_tiles` (CT) |
| `reduce_column_block` | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_COL, cb_input_tiles, cb_scaler, cb_output_tiles, ReduceInputPolicy::WaitAndPopPerTile, ReduceDataFormatReconfigMode::NONE>(in, scaler, out, ReduceInputBlockShape::of(Ht, block_ct_tiles, 1), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, NoOp{}, ReducePartialScaler::with_partial())` | decl `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:653-681`; partial `:350-357`; COL path `reduce_helpers_compute.inl:1155-1271`; Quasar `pack_init` `:961-964` | `fp32_mode`/`algorithm` left at defaults (`Fast`, `Auto` → `ReduceTile`); buffers constructed once at kernel scope | `cb_input_tiles`, `cb_scaler` | `cb_output_tiles` | `ReduceInputBlockShape` rows = `Ht`, cols = `block_ct_tiles`; DEST chunk = `DEST_AUTO_LIMIT` (static_assert == `dest_chunk_tiles`) |
| `store_mean_block` | raw_api | `noc.async_write<NocOptions::TXN_ID>(cb_output_tiles, output_accessor, {}, {.page_id = ct})`; end: `noc_async_write_set_trid(0, noc.get_noc_id())` | pattern `.claude/references/quasar_op_examples/toy_row_passes/kernels/writer.cpp:39,44` | | `cb_output_tiles` | — | `block_ct_tiles` |
| writer tag preset | raw_api | `noc_async_write_set_trid(0, noc.get_noc_id())` and `noc_async_read_set_trid(0, noc.get_noc_id())` as the writer's first statements | — | before `fill_scaler_pair` (its `write_zeros` is untagged) | — | — | — |
| `release_scaler_pair` | raw_api | `scaler.pop_front(2)` | contract `reduce_helpers_compute.hpp:39-43` ("caller owns that page and must pop it once") | | `cb_scaler` | — | — |

**Raw-API entries — helpers considered and rejected.**
- `load_column_block` / `store_mean_block`: there is no kernel-library helper for an
  accessor-driven implicit-sync tile stream on Quasar; `dfb_helpers_dataflow.hpp` and
  `tile_column_dataflow.hpp` were considered — the target page mandates implicit sync with
  `NocOptions::TXN_ID` per entry for a NoC-filled ring, which is what the Quasar examples write
  raw (`toy_row_passes/kernels/reader.cpp:45`).
- `release_scaler_pair`: one `pop_front` mandated by the reduce helper's contract
  (`reduce_helpers_compute.hpp:39-43`); no helper exists for it.
- `compute_kernel_hw_startup`: the required boot, not a phase.

**Helpers considered for the compute phase and not used:** `reduce_mean<>` (`reduce_helpers_compute.hpp:792-809`)
defaults to `AccumulateViaAdd` and applies 1/N on the SFPU after
the reduce; `post_reduce_op` with a `mul_unary` is not used.
The scaler carries `1/(H*W)` instead — one multiply per element in the FPU reduce.

## Broadcast Verification

Omitted: the op uses no binary eltwise operation.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Padding rows of the last tile row enter the sum (R1, P1) | The model's 49 rows sit in 64: 15 padding rows of undefined content; a whole-tile reduce gives every channel the same offset, which PCC does not see | `with_partial()` + the second scaler tile filled for `last_tile_valid_rows` (17 for the model) only: padding rows multiply by 0. The divisor is `H*W` (49), carried by `inv_hw`, never `32·Ht`. Regression: a test with the padding rows of the input tiles poisoned with large finite values (write through a tile tensor whose logical rows are padded) must still pass. |
| Non-finite padding | Multiplication by a zero scaler masks any finite value; an Inf/NaN in a padding row may propagate through the FPU (R1: "whatever it holds") | Accepted for Phase 0: inputs produced by `from_torch` and by tt-metal ops carry finite tile padding. If a non-finite padding case is ever specified, the reader zeroes rows `[last_tile_valid_rows, 32)` of each last-row tile in L1 before handing it over (explicit-sync ring, tagged transfers) — a reader-only change. |
| Untagged `write_zeros` in `prepare_reduce_scaler` (`reduce_helpers_dataflow.inl:200-201`) | An untagged transfer reuses the hart's sticky tag | The scaler fill is on the writer, whose first statements set its read and write trids to 0 (user id) before the fill; `cb_scaler` is in the writer's `disable_dfb_implicit_sync_for`; the output writes after it are tagged implicit; last statement `noc_async_write_set_trid(0, …)`. Reader: only tagged reads, last statement `noc_async_read_set_trid(0, …)`. Verified by the one-session tests (Work Distribution). |
| Reader order ≠ helper order | REDUCE_COL streams (chunk, position row, column); a reader emitting row-major over the whole block puts tiles in the wrong DEST slots when `block_ct_tiles > DEST_AUTO_LIMIT` | `dest_chunk_tiles` derived once on host from the compute config and shared by reader and compute; compute `static_assert(DEST_AUTO_LIMIT == dest_chunk_tiles)`. Phase 0 cells have `core_ct_tiles ≤ 2 <` any chunk, so a regression test with a narrowed grid (forcing `core_ct_tiles > dest_chunk_tiles`, e.g. C = 2048 on 2 clusters) is required to exercise more than one chunk. |
| Scaler page never popped | `reduce()` waits but never pops the scaler (`reduce_helpers_compute.hpp:39-43`); an unbalanced ring at program end | compute pops 2 after the last block |
| DFB constructed inside a helper | On Quasar a DataflowBuffer drains when it goes out of scope; a helper-built buffer blocks at every call end | construct `in`, `scaler`, `out` once at compute-kernel scope and pass by reference (`reduce_helpers_compute.hpp:46-49`); the writer's scaler buffer likewise |
| `fpu_math_fidelity` | set on every compute kernel | always set from the normalized config |
| Output placement (R9) | A width-sharded input must give an L1 *interleaved* output; a DRAM input a DRAM one | `MemoryConfig(INTERLEAVED, input buffer type)`; never copy the input's memory config |
| Engaged clusters vary with C | `split_work_to_cores` engages `min(tensor_ct_tiles, grid)` clusters (1 to 32 in Phase 0); runtime args are only given to engaged clusters | the `WorkUnitSpec` target nodes are exactly `all_cores` from the split; every per-core RT arg map is keyed by that set |
