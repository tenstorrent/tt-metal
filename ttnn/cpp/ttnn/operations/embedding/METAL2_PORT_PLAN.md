# Port Plan - `ttnn/cpp/ttnn/operations/embedding`

Port plan for the `embedding` op, ported from the legacy `ProgramDescriptor` (`create_descriptor`) API to Metal 2.0 (`create_program_artifacts` / `MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

The device-operation `EmbeddingsDeviceOperation` (`ttnn::prim`) holds three factories in its `program_factory_t` variant. All three are ported together in this pass. Because the port removes each factory's `create_descriptor` and adds `create_program_artifacts`, every variant alternative moves to `MetalV2FactoryConcept` at once (the `AllFactoriesValid` concept requires each alternative to satisfy exactly one factory concept).

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` - each of the three factories is a struct with a single static `create_descriptor(...)` returning `tt::tt_metal::ProgramDescriptor`.
- Variants (device-op `program_factory_t`): `EmbeddingsFusedProgramFactory`, `EmbeddingsRMProgramFactory`, `EmbeddingsTilizedIndicesProgramFactory`. Selection in `embedding_device_operation.cpp:17-26`: TILE-layout index tensor → TilizedIndices; else `tilized` attr → Fused; else → RM.
- Custom `compute_program_hash`: none - already the default reflection-based hash (audit + code confirmed; whole-op grep for `compute_program_hash` is empty). No deletion required.

*(Target Metal 2.0 concept chosen in the audit: `MetalV2FactoryConcept`. Carried forward in [TTNN ProgramFactory](#ttnn-programfactory).)*

### Shared kernel helper: `device/kernels/dataflow/embeddings_common.hpp`
Included by all three readers. Two device helpers, both templated on the weights `TensorAccessor` type:
- `prepare_local_cache(...)` - fills the per-core weight cache for `PADDED` (pad-token stick) / `BINARY` (index-0 and index-1 sticks); no-op for `GENERIC`. Legacy takes `uint32_t local_cache_cb` (the c_2/c_3 CB id) and `pad_token_arg_idx` (an RTA index).
- `read_token_async(...)` - issues the weight-stick (or chunk) read for one token: through the weights accessor for `GENERIC`/`BFP16`/`PADDED`-non-pad, or from the local cache for `PADDED`-pad / `BINARY`. Takes `weight_offset_bytes` (chunk offset within the stick).
- Globals: `pad_token`, `pad_local_addr`, `zero_local_addr`, `one_local_addr`; `input_token_t` = `uint16_t` under `BFP16`, else `uint32_t`.

### Variant: RM (`embeddings_rm_program_factory.cpp`) - row-major output

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs (per node) | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/embeddings.cpp` | all_cores | `[out_cb(0), idx_cb(1), cache_cb(2), input_page_size(3), weight_stick_size(4), rows_per_block(5), input_block_size_bytes(6), chunk_size(7), num_chunks(8), last_chunk_size(9)]` + `TensorAccessorArgs(a)` + `TensorAccessorArgs(weights)` | `[a_buf, weights_buf, batch_offset, weights_offset, num_rows, index_idx, (pad_token if PADDED)]` | `{<EmbeddingsType>:1, <IndexType>:1}` | `ReaderConfigDescriptor{}` |
| writer (chunked; `!output_sharded && rounded_weight_page_size>1MB`) | `device/kernels/dataflow/embeddings_rm_writer_chunked.cpp` | all_cores | `[out_cb(0), output_page_size(1), chunk_size(2), num_chunks(3), last_chunk_size(4)]` + `TensorAccessorArgs(output)` | `[output_buf, num_sticks, start_id]` | - | `WriterConfigDescriptor{}` |
| writer (non-chunked; `!output_sharded`) | **donor** `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | all_cores | `[out_cb(0), output_page_size(1)]` + `TensorAccessorArgs(output)` | `[output_buf, stick_size, num_sticks, start_id]` | - | `WriterConfigDescriptor{}` |

No writer is created when `output_sharded` (HEIGHT_SHARDED RM); the reader fills the resident output shard directly.

#### CBs
| index | total_size | page_size | data_format | touchers → disposition |
|---|---|---|---|---|
| c_0 (out) | interleaved: `buffering*chunk_size`; sharded: `aligned_size_per_bank`, `.buffer=out_buffer` | `chunk_size` | weights fmt (bf16) | interleaved: reader P → writer C (**1:1**); sharded: reader only (**self-loop**, borrowed) |
| c_1 (index scratch) | `block_height*index_page_size` | same | input fmt | reader only (**self-loop**) |
| c_2 (weight cache; PADDED×1 / BINARY×2) | `(1|2)*round_up_to_mul32(weight_page_size)` | `round_up_to_mul32(weight_page_size)` | weights fmt | reader only (**self-loop**), **conditional** on PADDED/BINARY |

#### Work split
- Non-sharded: `split_work_to_cores(compute_with_storage_grid_size, num_blocks)` where `num_blocks = a.padded_shape()[-1]*a.padded_shape()[0]` (`num_output_rows`); `block_height = alignment/input_element_size`.
- Sharded: `all_cores = shard.grid`; `core_group_1 = all_cores`; `units_per_core_group_1 = shard.shape[0]`; group 2 empty; `row_major = shard.orientation == ROW_MAJOR`.

### Variant: Fused (`embeddings_fused_program_factory.cpp`) - tilized output

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs (per node) | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/embeddings_tilize.cpp` | all_cores | `[src0(0), idx(1), cache(2), input_page_size(3), weight_page_size(4)ᴰ, weight_block_size(5), tiles_per_chunk(6), input_block_size_bytes(7), num_chunks(8), last_chunk_tiles(9)]` + `TA(a)` + `TA(weights)` | `[a_buf, weights_buf, input_start_id, input_start_offset, weight_offset, num_blocks, (pad_token if PADDED)]` | `{<EmbeddingsType>:1, <IndexType>:1}` | `ReaderConfigDescriptor{}` |
| compute_g1 / compute_g2 (one per non-empty core group) | chunked: `device/kernels/compute/tilize_chunked.cpp`; else **donor** `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp` | core_group_1 / core_group_2 | chunked: `[src0(0), out(1), per_core_block_cnt(2), tiles_per_chunk(3), num_chunks(4), last_chunk_tiles(5)]`; donor: reads `[in(0), out(1), per_core_block_cnt(2), per_core_block_tile_cnt(3)]` (CTAs 4,5 dead on donor path) | - | - | `ComputeConfigDescriptor{}` |
| writer (`!output_sharded`) | **donor** `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | all_cores | `[out_cb(0)]` + `TA(output)` | `[output_buf, num_pages, start_id]` | - | `WriterConfigDescriptor{}` |

`ᴰ` = dead CTA (emitted by the factory, never read by the kernel). No writer when `output_sharded`.

#### CBs
| index | total_size | page_size | data_format | touchers → disposition |
|---|---|---|---|---|
| c_0 (src0, weights-in) | `buffering*tiles_per_chunk*weights_tile_size` | `weights_tile_size` | weights fmt (bf16) | reader P → compute C (**1:1**) |
| c_1 (index scratch) | `TILE_HEIGHT*input_element_size` | same | input fmt | reader only (**self-loop**) |
| c_2 (out) | interleaved: `buffering*tiles_per_chunk*output_tile_size`; sharded: `aligned_size_per_bank`, `.buffer=out_buffer` | `output_tile_size` | output fmt (bf16) | interleaved: compute P → writer C (**1:1**); sharded: compute only (**self-loop**, borrowed) |
| c_3 (weight cache; PADDED×1 / BINARY×2) | `(1|2)*round_up_to_mul32(weight_page_size)` | `round_up_to_mul32(weight_page_size)` | weights fmt | reader only (**self-loop**), **conditional** on PADDED/BINARY |

#### Work split
- Non-sharded: `split_work_to_cores(grid, num_blocks)`, `num_blocks = num_output_rows/TILE_HEIGHT`, `num_tiles_per_block = weights.padded_shape()[-1]/TILE_WIDTH`.
- Sharded: `all_cores = shard.grid`; `units_per_core_group_1 = shard.shape[0]/TILE_HEIGHT`; `num_tiles_per_block = shard.shape[1]/TILE_WIDTH`; `weight_block_size = shard.shape[1]*weights_element_size` (else `weight_page_size`).
- `weight_offset` RTA advances by `weight_block_size` per core, non-zero only for block/width-sharded output (see Applied Patterns → weights accessor base offset).

### Variant: TilizedIndices (`embeddings_tilized_indices_program_factory.cpp`) - TILE-layout index input

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs (per node) | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/embedding_ind_tilized.cpp` | all_cores | `[src0(0), idx(1), cache(2), input_page_size(3)ᴰ, weight_stick_size(4), row_length(5), input_block_size_bytesᴰ(6)]` + `TA(a)` + `TA(weights)` | `[a_buf, weights_buf, tile_offset, face_offset, num_rows, curr_col, starting_index, (pad_token if PADDED)]` | `{<EmbeddingsType>:1, <IndexType>:1, (ONLY_ONE_FACE_COLUMN if row≤FACE_HEIGHT)}` | `ReaderConfigDescriptor{}` |
| writer | **donor** `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | all_cores | `[out_cb(0), output_page_size(1)ᴰ]` + `TA(output)` | `[output_buf, stick_size, num_sticks, start_id]` | - | `WriterConfigDescriptor{}` |

`ᴰ` dead CTAs: `input_page_size`(3) and `input_block_size_bytes`(6) are emitted but the reader never reads them (it uses `input.get_aligned_page_size()` and hardcoded face/tile constants); `output_page_size`(1) is unread by the stick-layout writer. `output_cb_index = src0_cb_index` (c_0 doubles as weights-in and output).

#### CBs
| index | total_size | page_size | data_format | touchers → disposition |
|---|---|---|---|---|
| c_0 (src0, weights-in **and** output) | `2*rounded_weight_page_size` | `rounded_weight_page_size` | weights fmt (bf16) | reader P → writer C (**1:1**) |
| c_1 (index scratch) | `FACE_HEIGHT*index_page_size` | same | input fmt | reader only (**self-loop**) |
| c_2 (weight cache; PADDED×1 / BINARY×2) | `(1|2)*round_up_to_mul32(weight_page_size)` | `round_up_to_mul32(weight_page_size)` | weights fmt | reader only (**self-loop**), **conditional** on PADDED/BINARY |

No sharded path (output always interleaved / Case 1).

#### Work split
`split_work_to_cores_aligned(grid, volume, FACE_HEIGHT)` (op-local helper in `embedding_program_factory_common.cpp`); `volume = a.logical_shape()[0]*a.logical_shape()[-1]`.

### Cross-op kernels (out of the op's own directory)
Three donor kernels are instantiated by file path and shared beyond this op. Each needs a Metal 2.0 rewrite (CB→DFB, named-token bindings). Per the audit and the [shared-dataflow-kernel Caution](../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md), the consumers cannot co-migrate now, so each is **forked with a `_metal2` suffix alongside the original** (legacy copy retained for unmigrated consumers). Recorded under Open items in the port report.

| donor | fork | also used by (unmigrated) |
|---|---|---|
| `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | `…_metal2.cpp` (same dir) | `data_movement/concat`, `data_movement/slice` |
| `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp` | `tilize_metal2.cpp` (same dir) | `tilize`, `tilize_with_val_padding`, `untilize`, `untilize_with_unpadding`, `moreh/moreh_getitem`, `pool/upsample`, `sliding_window/halo`, `deepseek_prefill/combine`, `quasar/tilize_with_val_padding` |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `…_metal2.cpp` (same dir) | ~22 op families |

### Semaphores
None (whole op uses no semaphores).

### Tensor accessors
| host site | originating tensor | reached by |
|---|---|---|
| RM reader `embeddings.cpp:39,40` | input `a`, weights | `TensorAccessor(args, addr)` → Case 1 |
| RM writers | output | Case 1 (interleaved) / borrowed-DFB (sharded, no accessor) |
| Fused reader `embeddings_tilize.cpp:35,36` | input `a`, weights (base + `weight_offset`) | Case 1; weights base carries a live scalar offset (see Applied Patterns) |
| Fused writer (donor) | output | Case 1 (interleaved) / borrowed-DFB (sharded) |
| TilizedIndices reader `embedding_ind_tilized.cpp:35,36` | input `a`, weights | Case 1 |
| TilizedIndices writer (donor) | output | Case 1 |

3rd-argument site: `embeddings_rm_writer_chunked.cpp:26` `TensorAccessor(dst0_args, dst_addr, output_page_size)` - Class 2 (redundant, interleaved), drop the 3rd arg.

### Flags
- Latent bug (audit Misc anomaly): TilizedIndices reader passes `pad_token_arg_idx=6`, but the factory places the pad token at RTA 7 (arg 6 is `starting_index`). Named-arg conversion reads the pad token by name and cannot reproduce the positional aliasing → the bug is corrected as a side effect on the (untested) PADDED + TILE-layout path. Approved by the invoker; documented in the report.
- Dead code the port leaves alone (routed to report, not fixed): `#define RISC_CORES_PER_TENSIX 2` (`embedding_device_operation.cpp:13`), vestigial `api/debug/dprint.h` include in `embedding_ind_tilized.cpp`.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MetalV2FactoryConcept` - each factory's `create_descriptor` (returning `ProgramDescriptor`) is replaced by `create_program_artifacts` (returning `ttnn::device_operation::ProgramArtifacts`). Signature: `(const EmbeddingParams&, const EmbeddingInputs&, Tensor&)`.
- **Custom `compute_program_hash`**: none - nothing to delete.
- **Op-owned tensors**: none - the factories allocate no device tensors beyond the op's io.
- **Pybind**: `embedding_nanobind.cpp` binds only `&ttnn::embedding` (the user op). No `create_descriptor` / `create_program_descriptor` pybind exposure exists, so no pybind surface is removed.
- **Implementation notes**:
  - Extract `const MeshTensor&` for input / weights / output at the top of each factory (`.mesh_tensor()`), and build every `TensorParameter` / `TensorArgument` from those.
  - Name constants (`KernelSpecName`, `DFBSpecName`, `TensorParamName`) are declared **function-local** inside each `create_program_artifacts` so the three factory TUs never collide under unity build (sidesteps the [unity-build hygiene](../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md) issue without a shared header).
  - `DFBBinding` written in full designated-initializer form (not `ProducerOf`/`ConsumerOf`).

## Planned Spec Shape

### Variant: RM
- **KernelSpecs**: `reader` (`embeddings.cpp`); plus, when `!output_sharded`, one `writer` (chunked source or donor stick-layout fork). Sharded: reader only.
- **DataflowBufferSpecs**: `OUT` (c_0; borrowed_from `OUTPUT` when sharded), `IDX` (c_1), `WCACHE` (c_2, conditional on PADDED/BINARY).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT` (a), `WEIGHTS`, `OUTPUT`.
  - interleaved: INPUT+WEIGHTS on reader; OUTPUT on writer.
  - sharded: INPUT+WEIGHTS on reader; OUTPUT used only as `OUT`'s `borrowed_from` backing (no kernel binding - referential-integrity satisfied per `program_spec.cpp:545`).
- **WorkUnitSpecs**: one - `{reader (+writer)}` on `all_cores`.

### Variant: Fused
- **KernelSpecs**: `reader` (`embeddings_tilize.cpp`); `compute_g1` and (if group 2 non-empty) `compute_g2` - same source (chunked own kernel or donor `tilize` fork), differing only by the `per_core_block_cnt` CTA; plus, when `!output_sharded`, one `writer` (donor `writer_unary_interleaved_start_id` fork).
- **DataflowBufferSpecs**: `SRC0` (c_0, weights-in), `IDX` (c_1), `OUT` (c_2; borrowed_from `OUTPUT` when sharded), `WCACHE` (c_3, conditional on PADDED/BINARY).
- **TensorParameters**: `INPUT`, `WEIGHTS`, `OUTPUT` (OUTPUT: writer TensorBinding when interleaved; `OUT` borrowed_from backing when sharded).
- **WorkUnitSpecs**: `wu_g1 = {reader, compute_g1 (+writer)}` on `core_group_1`; `wu_g2 = {reader, compute_g2 (+writer)}` on `core_group_2` (only if group 2 non-empty). See Preserved Multiplicity.

### Variant: TilizedIndices
- **KernelSpecs**: `reader` (`embedding_ind_tilized.cpp`), `writer` (donor stick-layout fork).
- **DataflowBufferSpecs**: `OUT` (c_0, weights-in + output), `IDX` (c_1), `WCACHE` (c_2, conditional on PADDED/BINARY).
- **TensorParameters**: `INPUT`, `WEIGHTS`, `OUTPUT`.
- **WorkUnitSpecs**: one - `{reader, writer}` on `all_cores`.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| Fused: `compute_desc_1` (core_group_1) + `compute_desc_2` (core_group_2), same compute source, differing `per_core_block_cnt` CTA | `compute_g1`, `compute_g2` | `wu_g1` (core_group_1), `wu_g2` (core_group_2) - **disjoint** node sets | `SRC0` (each CONSUMER), `OUT` (each PRODUCER; +CONSUMER self-loop when sharded). Disjoint nodes ⇒ each node sees one compute instance ⇒ ordinary single-role bindings, **no** multi-binding flag. |

RM and TilizedIndices have no work-split multiplicity (single reader/writer over `all_cores`).

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| RM reader CTAs 0,1,2 | `out_cb`, `idx_cb`, `cache_cb` magic indices | `DFBBinding` OUT (PRODUCER, or PRODUCER+CONSUMER when sharded), IDX (self-loop), WCACHE (self-loop, conditional) |
| RM reader `TensorAccessorArgs(a)`, `(weights)` CTAs; RTAs 0,1 (`a_buffer`,`weights_buffer`) | address plumbing + buffer-ptr RTAs | `TensorParameter`/`TensorBinding` INPUT, WEIGHTS → `TensorAccessor(tensor::…)` |
| RM writer CTA 0; `TensorAccessorArgs(output)`; RTA `output_buffer` | out_cb index, output address plumbing | `DFBBinding` OUT (CONSUMER); `TensorBinding` OUTPUT |
| RM chunked writer RTA `output_page_size`; donor writer RTAs `stick_size`,`num_sticks`,`start_id` | positional RTAs | named RTAs |
| Fused reader CTAs 0,1,2; `TA` CTAs; RTAs 0,1 | cb indices + address plumbing | DFB + tensor bindings |
| Fused reader CTA 4 `weight_page_size` | dead CTA (unread) | **dropped** (named model emits only read args) |
| Fused reader RTA 4 `weight_offset` | separate scalar folded into weights accessor **base** | kept as named RTA `weight_offset`; **relocated into the accessor read `offset_bytes`** (Applied Patterns) |
| Fused compute CTAs 0,1 | src0/out cb indices | DFB bindings SRC0 (CONSUMER), OUT (PRODUCER) |
| Fused writer CTA 0; `TA(output)`; RTA `output_buffer` | out_cb index + address | DFB OUT (CONSUMER); TensorBinding OUTPUT |
| TilizedIndices reader CTAs 0,1,2; CTA 3 (`input_page_size`, dead); CTA 6 (`input_block_size_bytes`, dead) | cb indices + dead CTAs | DFB + tensor bindings; dead CTAs dropped |
| TilizedIndices writer CTA 0; CTA 1 (`output_page_size`, dead) | out_cb index + dead CTA | DFB OUT (CONSUMER); dead CTA dropped |
| `embeddings_rm_writer_chunked.cpp:26` 3rd accessor arg `output_page_size` | `TensorAccessor(args, addr, page_size)` | drop 3rd arg → `TensorAccessor(tensor::output)` (Class 2) |
| `prepare_local_cache` params `local_cache_cb`, `pad_token_arg_idx` | cb id + RTA index | `dfb::weight_cache` referenced internally (gated by PADDED/BINARY); pad token read as `get_arg(args::pad_token)` |
| all readers: pad-token RTA (positional, PADDED) | `get_arg_val<uint32_t>(idx)` | named RTA `pad_token` |

All remaining RTAs become named (no varargs anywhere - every arg is a distinct field read a fixed number of times).

## Applied Patterns

- **[Sync-free / single-ended CB → self-loop DFB](../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)**: every index-scratch CB (`IDX`, all three variants); every weight-cache CB (`WCACHE`); the borrowed output CB on sharded configs (RM `OUT`, Fused `OUT`). Each is a single-toucher → bind its one kernel PRODUCER + CONSUMER (shared accessor name).
- **[Borrowed-memory DFB](../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/migration_guide.md#dataflowbufferspec)**: sharded output - `OUT.borrowed_from = OUTPUT` (RM + Fused). The `OUTPUT` TensorParameter is used only as backing on the sharded path (no kernel TensorBinding), which the referential-integrity check accepts (`program_spec.cpp:545`).
- **[Conditional / optional DFB binding](../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)**: `WCACHE` bound only when `embeddings_type ∈ {PADDED, BINARY}`. The gating `#define` is the **existing** `PADDED`/`BINARY` type define (already emitted by the reader), which gates the `dfb::weight_cache` reference inside `prepare_local_cache` - no new define needed. The pad-token named RTA is likewise added to the reader schema only for PADDED, and referenced only under `#if defined PADDED`.
- **[Two-toucher / dual-instance work-split → 1P+1C, no flag](../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)**: Fused compute over two **disjoint** core groups - each node sees one compute instance, ordinary single-role bindings (not the multi-binding flag). See Preserved Multiplicity.
- **[Multi-variant factory](../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories)**: three separate factory structs (device-op variant), each its own `create_program_artifacts`.
- **[Pass DFB handles to LLKs](../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)**: fused compute - `compute_kernel_hw_startup(dfb::in, dfb::out)`, `compute_kernel_lib::tilize<…, dfb::in, dfb::out, …>`, `is_fp32_input_format<dfb::in>()` (implicit `DFBAccessor → uint32_t`).
- **Weights accessor base offset (fused, sharded)** - the load-bearing heads-up: `weight_offset` (RTA) was folded into the weights accessor **base** (`weight_buffer_src_addr + weight_offset`). Under a `tensor::weights` binding the base is fixed, so `weight_offset` is **relocated into the accessor reads' `offset_bytes`** at every site that read through the weights accessor: `read_token_async`'s GENERIC/BFP16/PADDED-non-pad reads (`weight_offset_bytes + weight_base_offset`) **and** `prepare_local_cache`'s cache-fill reads (`offset_bytes = weight_base_offset`). Local-cache reads keep only `weight_offset_bytes` (the cache already holds the offset). `weight_offset` is non-zero only for block/width-sharded fused output; RM and TilizedIndices pass `weight_base_offset = 0` (unchanged). The brief's one-liner named only `read_token_async`; extending to `prepare_local_cache` is required for faithful PADDED/BINARY + sharded behavior (see report Friction).
- **[Removing pybound legacy factory entry points](../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-removing-pybound-legacy-factory-entry-points)**: N/A - no factory entry point is pybound.
- **Hardware config**: all DM kernels resolve to the reader/writer defaults → `ttnn::create_reader_datamovement_config(arch)` / `create_writer_datamovement_config(arch)`. Fused compute uses the legacy `ComputeConfigDescriptor{}` (all defaults, Style B) → `ComputeGen1Config{}` (defaults coincide: HiFi4, `enable_32_bit_dest=false`, `double_buffer_dest=true`, `sfpu_precision_mode=Precise`, `bfp_pack_precision_mode=Approximate`, empty `unpack_modes`). Output/weights are bf16 (no Float32), so no `unpack_modes` entry is required.

## Deferred / Flagged
- New findings during planning:
  - Fused reader dead CTA `weight_page_size` (index 4) and TilizedIndices reader dead CTAs `input_page_size`(3) + `input_block_size_bytes`(6): unread by their kernels; the named model naturally drops them. (Extends the audit's single dead-CTA note, which covered only the stick-layout writer.)
  - `prepare_local_cache` also reads through the weights accessor, so the `weight_offset` relocation must extend there for the untested PADDED/BINARY + sharded-fused case (above).
  - No PADDED/BINARY path is exercised by any test (all use GENERIC) → the weight-cache conditional binding and the pad-token correction are compile-covered only. Recorded in report Open items.
