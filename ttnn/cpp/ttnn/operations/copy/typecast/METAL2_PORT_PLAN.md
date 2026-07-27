# Port Plan — `ttnn/cpp/ttnn/operations/copy/typecast`

Port plan for `copy/typecast`, ported from the legacy `ProgramDescriptor` API to Metal 2.0
(`MetalV2FactoryConcept` / `create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

**Scope of this pass: all four factories.** They are one atomic unit — every factory binds the
*same* compute kernel entry point (`device/kernels/compute/eltwise_typecast.cpp`), so
Metal-2.0-ifying that source for one factory breaks the others (recipe § *the atomic unit of a
port*, "Shared top-level entry point"). There is no tractable single-factory sub-target.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (all four factories: `static ProgramDescriptor create_descriptor(...)`).
- Variants: four — `TypecastProgramFactory` (interleaved/tiled + non-optimized-sharded fallback),
  `TypecastSubgridProgramFactory` (`sub_core_grids`, tiled), `TypecastShardedProgramFactory`
  (L1-sharded optimized, borrowed-memory CBs), `TypecastRowMajorChunkedProgramFactory`
  (ROW_MAJOR chunked DRAM path).
- Custom `compute_program_hash`: **none** — `TypecastDeviceOperation` already uses the default
  reflection-based hash (audit confirmed; no device-op-class edit forced).

*(The Metal 2.0 factory concept the port targets was chosen during the audit — `MetalV2FactoryConcept`.
Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section below.)*

### Variant: `TypecastProgramFactory`  (`device/typecast_program_factory.cpp:17-190`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` (**cross-op**) | `all_cores` | `TensorAccessorArgs(*src_buffer)` only (slots 0..N) | none | `{src_buffer, num_items_per_core, num_items_written}` | none | none | `ReaderConfigDescriptor{}` |
| writer | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (**cross-op**) | `all_cores` | `[0]=output_cb_index(c_2)`, then `TensorAccessorArgs(*dst_buffer)` | none | `{dst_buffer, num_items_per_core, num_items_written}` | none | none | `WriterConfigDescriptor{}` |
| compute_group_1 | `copy/typecast/device/kernels/compute/eltwise_typecast.cpp` | `core_group_1` | `[0]=num_items_per_core_group_1`, `[1]=1`, `[2]=c_0`, `[3]=c_2` | none | none | none | `TYPECAST_LLK_INIT`, `TYPECAST_LLK` | `ComputeConfigDescriptor{HiFi4, fp32_dest_acc_en, unpack_to_dest_mode, bfp8_pack_precise, math_approx_mode=false}` |
| compute_group_2 *(only if `core_group_2` non-empty)* | same source | `core_group_2` | `[0]=num_items_per_core_group_2`, `[1]=1`, `[2]=c_0`, `[3]=c_2` | none | none | none | same | same |

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| c_0 | `2 * input_page_size` | `all_cores` | `datatype_to_dataformat_converter(input.dtype())` | `is_row_major ? src_buffer->page_size() : tile_size(input_df)` | not set |
| c_2 | `2 * output_page_size` | `all_cores` | `datatype_to_dataformat_converter(output.dtype())` | `is_row_major ? dst_buffer->page_size() : tile_size(output_df)` | not set |

Neither CB is borrowed (`.buffer` unset), neither is a GlobalCircularBuffer.

#### Semaphores
none — the op uses no semaphores in any factory.

#### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `typecast_program_factory.cpp:78` (`TensorAccessorArgs(*src_buffer).append_to`) | `tensor_args.input` | reader RTA[0] (`Buffer*` — `BufferBinding` auto-patched) |
| `typecast_program_factory.cpp:80` | `output` | writer RTA[0] (`Buffer*`) |

#### Work split
- Driver: `split_work_to_cores(compute_with_storage_grid_size, num_pages, is_row_major)` (`:49-50`)
- `num_pages = input.buffer()->num_pages()` (tiles for TILE, rows for ROW_MAJOR)
- `core_group_1`, count `num_items_per_core_group_1`; `core_group_2`, count `num_items_per_core_group_2`
- RTA loop over `corerange_to_cores(all_cores, nullopt, is_row_major)`, accumulating `num_items_written`.

### Variant: `TypecastSubgridProgramFactory`  (`device/typecast_program_factory.cpp:193-334`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `…/reader_unary_interleaved_start_id.cpp` (**cross-op**) | `all_cores` | `TensorAccessorArgs(*src_buffer)` | none | `{src_buffer, ntiles_per_core, tile_start_id}` | none | none | `ReaderConfigDescriptor{}` |
| writer | `…/writer_unary_interleaved_start_id.cpp` (**cross-op**) | `all_cores` | `[0]=c_2`, then `TensorAccessorArgs(*dst_buffer)` | none | `{dst_buffer, ntiles_per_core, tile_start_id}` | none | none | `WriterConfigDescriptor{}` |
| compute | `…/eltwise_typecast.cpp` | `all_cores` | `[0]=ntiles_per_core`, `[1]=1`, `[2]=c_0`, `[3]=c_2` | none | none | none | `TYPECAST_LLK_INIT`, `TYPECAST_LLK` | `ComputeConfigDescriptor{…}` (identical field set to `TypecastProgramFactory`) |

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| c_0 | `2 * tile_size(input_df)` | `all_cores` | input df | `tile_size(input_df)` | not set |
| c_2 | `2 * tile_size(output_df)` | `all_cores` | output df | `tile_size(output_df)` | not set |

#### Tensor accessors
`typecast_program_factory.cpp:260` (input → reader RTA[0]), `:262` (output → writer RTA[0]).

#### Work split
- No `split_work_to_cores`. `ncores` is shrunk from `sub_core_grids->num_cores()` until
  `ntiles % ncores == 0`; `cores = corerange_to_cores(sub_core_grids, ncores, true)`;
  `all_cores = num_cores_to_corerangeset_in_subcoregrids(...)` (or a single `CoreRange` when `ncores == 1`).
- Uniform `ntiles_per_core = ntiles / ncores` — one compute descriptor, no group split.

### Variant: `TypecastShardedProgramFactory`  (`device/typecast_sharded_program_factory.cpp:17-205`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` (**cross-op**) | `all_cores` (= `shard_spec.grid`) | `[0]=in_cb_id(c_0)` | none | `{num_tile_per_core}` (no address arg) | none | none (empty `kernel_defines`) | `ReaderConfigDescriptor{}` |
| compute | `…/eltwise_typecast.cpp` | `all_cores` | `[0]=1`, `[1]=num_tile_per_core`, `[2]=c_0`, `[3]=c_2` | none | none | none | `TYPECAST_LLK_INIT`, `TYPECAST_LLK` | `ComputeConfigDescriptor{…}` (identical field set) |

No writer kernel: the borrowed output CB *is* the result.

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) | buffer (borrowed) |
|---|---|---|---|---|---|---|
| c_0 | `in_cb_pagesize * in_cb_npages` | `all_cores` | `act_df` | `round_up_to_mul32(input_tile_size)` | not set | `input.buffer()` (`:94`) |
| c_2 | `out_cb_pagesize * out_cb_npages` | `all_cores` | `out_df` | `round_up_to_mul32(output_tile_size)` | not set | `output.buffer()` (`:111`) |

`in_cb_npages = out_cb_npages = num_tile_per_core * buffering_factor(=1)`.

#### Tensor accessors
none — no kernel in this factory constructs a `TensorAccessor`; the tensors reach the kernels
only as borrowed CB memory.

#### Work split
n/a — every core does `num_tile_per_core` tiles (derived from the shard spec); the RTA loop over
`corerange_to_cores(all_cores)` writes the same value to every core.

### Variant: `TypecastRowMajorChunkedProgramFactory`  (`device/typecast_rm_chunked_program_factory.cpp:65-277`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `copy/typecast/device/kernels/dataflow/reader_typecast_rm_chunked.cpp` | `all_cores` | `[0]=c_0`, `[1]=full_chunks_per_row`, `[2]=partial_chunks_per_row`, `[3]=input_full_chunk_size_bytes`, `[4]=input_partial_chunk_size_bytes`, then `TensorAccessorArgs(*src_buffer)` (from slot 5) | none | `{src_buffer, num_rows_for_core, start_row_id}` | none | none | `ReaderConfigDescriptor{}` |
| writer | `copy/typecast/device/kernels/dataflow/writer_typecast_rm_chunked.cpp` | `all_cores` | `[0]=c_2`, `[1]=full_chunks_per_row`, `[2]=partial_chunks_per_row`, `[3]=output_full_chunk_size_bytes`, `[4]=output_partial_chunk_size_bytes`, then `TensorAccessorArgs(*dst_buffer)` (from slot 5) | none | `{dst_buffer, num_rows_for_core, start_row_id}` | none | none | `WriterConfigDescriptor{}` |
| compute_group_1 *(if `core_group_1` non-empty)* | `…/eltwise_typecast.cpp` | `core_group_1` | `[0]=num_rows_per_core_group_1 * chunks_per_row_total`, `[1]=1`, `[2]=c_0`, `[3]=c_2` | none | none | none | `TYPECAST_LLK_INIT`, `TYPECAST_LLK` | `ComputeConfigDescriptor{…}` |
| compute_group_2 *(if `core_group_2` non-empty)* | same source | `core_group_2` | `[0]=num_rows_per_core_group_2 * chunks_per_row_total`, `[1]=1`, `[2]=c_0`, `[3]=c_2` | none | none | none | same | same |

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| c_0 | `2 * input_cb_page_size_bytes` | `all_cores` | input df | `align(padded_input_full_chunk_size_bytes, src_buffer->alignment())` | not set |
| c_2 | `2 * output_cb_page_size_bytes` | `all_cores` | output df | `align(padded_output_full_chunk_size_bytes, dst_buffer->alignment())` | not set |

#### Tensor accessors
`typecast_rm_chunked_program_factory.cpp:155` (input → reader RTA[0]), `:164` (output → writer RTA[0]).
Kernel-side the accessor page reads carry a **device-computed** `.offset_bytes`
(`chunk_idx * full_chunk_size_bytes`) — a kernel-side NoC page offset, not a host-folded base;
it stays exactly as-is (audit *Offset base pointers* GREEN).

#### Work split
- Driver: `split_work_to_cores(compute_with_storage_grid_size, num_rows, /*row_wise=*/true)` (`:109-110`)
- Two groups with `num_rows_per_core_group_{1,2}`; RTA loop over `corerange_to_cores(all_cores, nullopt, true)`.

### Cross-op kernels
Three dataflow kernels live in the **eltwise/unary** family and are instantiated by file path:

| kernel file | used by (this op) | co-borrowers |
|---|---|---|
| `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | `TypecastProgramFactory`, `TypecastSubgridProgramFactory` | ~70 factories across data_movement, reduction, matmul, embedding, kv_cache, transformer, examples… |
| `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `TypecastProgramFactory`, `TypecastSubgridProgramFactory` | same broad set |
| `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | `TypecastShardedProgramFactory` | broad set |

**Decision: fork, do not modify in place.** Every co-borrower is still on a legacy descriptor
factory emitting positional CTAs and address RTAs; a named-arg / `dfb::` rewrite in place breaks
all of them at JIT time. The forks land inside the op's own directory as
`device/kernels/dataflow/{reader_unary_interleaved_start_id,writer_unary_interleaved_start_id,reader_unary_sharded}_metal2.cpp`,
which also keeps the port's writeable surface inside the op directory
(catalog: *Caution: Modifying a shared dataflow kernel*; precedent for the `_metal2` suffix:
the earlier quasar-dir forks). Recorded in the port report under *Open items for downstream*
as the sunset checklist for the legacy twins.

### Flags
- `device/kernels/compute/eltwise_typecast.cpp:31` — `TYPECAST_LLK_INIT()` is invoked inside the
  innermost per-tile loop rather than once before it (audit *Misc anomalies*). Carried through
  **unchanged**; not port work.
- The forked interleaved reader/writer keep their `BACKWARDS` / `OUT_SHARDED` `#ifdef` blocks even
  though typecast never defines either, so the forks stay diffable against their legacy twins.
- No unreferenced kernel files in the op directory; every kernel source is bound by some factory.
- `TypecastProgramFactory` computes `const auto* device = input.device()` and drops `num_cores`
  via `(void)num_cores` — untouched by the port except where the new API needs `device->arch()`.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MetalV2FactoryConcept` — each factory replaces
  `static ProgramDescriptor create_descriptor(...)` with
  `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const TypecastParams&, const TypecastInputs&, Tensor& output)`.
- **Custom `compute_program_hash`**: none — nothing to delete.
- **Implementation notes**:
  - No pybind surface change: `ttnn-nanobind/operations/copy.cpp` binds `typecast` as a plain
    function; no `create_descriptor` entry point is exposed, so nothing to remove.
  - `TypecastDeviceOperation` itself is untouched (`program_factory_t` keeps all four alternatives;
    all four flip to the new concept together, so `AllFactoriesValid` stays satisfied).
  - Op-owned tensors: none.

## Planned Spec Shape

Shared across all four variants: DFB spec names `in` / `out`, tensor parameter names
`input` / `output`, kernel accessor names `in` / `out` / `input` / `output` (the compute kernel and
the forked dataflow kernels are shared, so the accessor names must agree across factories).

### Variant: `TypecastProgramFactory`
- KernelSpecs: `reader`, `writer`, `compute_group_1`, and `compute_group_2` when `core_group_2` is
  non-empty (multiplicity preserved — per-group tile count stays a CTA).
- DataflowBufferSpecs: `in` (`entry_size = input_page_size`, `num_entries = 2`,
  `data_format_metadata = input df`), `out` (`entry_size = output_page_size`, `num_entries = 2`,
  `data_format_metadata = output df`). No `tile_format_metadata` (legacy set no `.tile`).
- SemaphoreSpecs: none.
- TensorParameters: `input` (`input.tensor_spec()`), `output` (`output.tensor_spec()`).
- WorkUnitSpecs: `{reader, writer, compute_group_1}` on `core_group_1`;
  `{reader, writer, compute_group_2}` on `core_group_2` (when non-empty). Their union is
  `all_cores`, reproducing the legacy reader/writer placement.
- Op-owned tensors: none.

### Variant: `TypecastSubgridProgramFactory`
- KernelSpecs: `reader`, `writer`, `compute`.
- DataflowBufferSpecs: `in` / `out`, `num_entries = 2`, entry sizes = the respective tile sizes.
- TensorParameters: `input`, `output`. WorkUnitSpecs: one — `{reader, writer, compute}` on `all_cores`.

### Variant: `TypecastShardedProgramFactory`
- KernelSpecs: `reader` (forked sharded reader), `compute`.
- DataflowBufferSpecs: `in` (`entry_size = in_cb_pagesize`, `num_entries = in_cb_npages`,
  `data_format_metadata = act_df`, **`borrowed_from = input`**), `out` (`entry_size = out_cb_pagesize`,
  `num_entries = out_cb_npages`, `data_format_metadata = out_df`, **`borrowed_from = output`**).
- TensorParameters: `input`, `output` — declared for the borrowed-memory backing only; no kernel
  `TensorBinding` (the validator counts `borrowed_from` as a use).
- WorkUnitSpecs: one — `{reader, compute}` on `all_cores`.

### Variant: `TypecastRowMajorChunkedProgramFactory`
- KernelSpecs: `reader`, `writer`, plus `compute_group_1` / `compute_group_2` under the same
  non-empty guards the legacy factory used.
- DataflowBufferSpecs: `in` (`entry_size = input_cb_page_size_bytes`, `num_entries = 2`),
  `out` (`entry_size = output_cb_page_size_bytes`, `num_entries = 2`), with data formats as legacy.
- TensorParameters: `input`, `output`.
- WorkUnitSpecs: `{reader, writer, compute_group_1}` on `core_group_1`;
  `{reader, writer, compute_group_2}` on `core_group_2` — each under its legacy guard.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| `TypecastProgramFactory`: `compute_desc_group_1` + optional `compute_desc_group_2` of `eltwise_typecast.cpp` | `compute_group_1`, `compute_group_2` | `typecast_group_1` (`core_group_1`), `typecast_group_2` (`core_group_2`) | `in` — CONSUMER on each; `out` — PRODUCER on each. Disjoint node sets ⇒ one instance per node per role, legal without any flag. |
| `TypecastRowMajorChunkedProgramFactory`: `compute_desc_group_1` + `compute_desc_group_2` of `eltwise_typecast.cpp` | `compute_group_1`, `compute_group_2` | `typecast_rm_group_1`, `typecast_rm_group_2` | same as above |
| `TypecastSubgridProgramFactory`, `TypecastShardedProgramFactory` | none — single compute descriptor each | — | — |

The two `reader` / `writer` KernelSpecs are single instances that appear in *both* WorkUnitSpecs
(a kernel may be a member of several work units); that is placement, not multiplicity.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `typecast_program_factory.cpp:78` | `TensorAccessorArgs(*src_buffer).append_to(reader_cta)` | `TensorParameter{input}` + `TensorBinding{INPUT,"input"}` on `reader` |
| `typecast_program_factory.cpp:80` | `TensorAccessorArgs(*dst_buffer).append_to(writer_cta)` | `TensorParameter{output}` + `TensorBinding{OUTPUT,"output"}` on `writer` |
| `typecast_program_factory.cpp:79` | writer CTA[0] `= output_cb_index` (magic CB index) | `DFBBinding{OUT_DFB,"out",CONSUMER}` |
| `typecast_program_factory.cpp:177` | reader RTA[0] `src_buffer` (`Buffer*` / `BufferBinding`) | `TensorBinding` (address auto-injected per enqueue) |
| `typecast_program_factory.cpp:178` | writer RTA[0] `dst_buffer` | `TensorBinding` |
| `typecast_program_factory.cpp:98-102, 140-144` | compute CTA[2]=`c_0`, CTA[3]=`c_2` (magic CB indices) | `DFBBinding{IN_DFB,"in",CONSUMER}` + `DFBBinding{OUT_DFB,"out",PRODUCER}` |
| `typecast_program_factory.cpp:98-102, 140-144` | compute CTA[0], CTA[1] positional | named CTAs `per_core_block_cnt`, `per_core_block_dim` |
| `typecast_program_factory.cpp:260, 262, 261, 324, 325, 281-285` | same six shapes in the subgrid factory | same replacements |
| `typecast_sharded_program_factory.cpp:140-142` | reader CTA[0] `= in_cb_id` | `DFBBinding{IN_DFB,"in",PRODUCER}` |
| `typecast_sharded_program_factory.cpp:159-163` | compute CTA[0..3] (two scalars + two CB indices) | named CTAs `per_core_block_cnt`, `per_core_block_dim` + `in`/`out` DFB bindings (`out` self-looped) |
| `typecast_sharded_program_factory.cpp:94, 111` | `CBDescriptor.buffer = input.buffer()/output.buffer()` | `DataflowBufferSpec.borrowed_from = INPUT / OUTPUT` |
| `typecast_sharded_program_factory.cpp:198` | reader RTA[0] `num_tile_per_core` (positional) | named RTA `num_tiles_per_core` |
| `typecast_rm_chunked_program_factory.cpp:148-155` | reader CTA[0] `= input_cb_index`; CTA[1..4] positional; `TensorAccessorArgs` slots 5+ | `DFBBinding{IN_DFB,"in",PRODUCER}`; named CTAs `full_chunks_per_row`, `partial_chunks_per_row`, `full_chunk_size_bytes`, `partial_chunk_size_bytes`; `TensorBinding{INPUT,"input"}` |
| `typecast_rm_chunked_program_factory.cpp:157-164` | writer CTA[0] `= output_cb_index`; CTA[1..4] positional; `TensorAccessorArgs` slots 5+ | `DFBBinding{OUT_DFB,"out",CONSUMER}`; same four named CTAs; `TensorBinding{OUTPUT,"output"}` |
| `typecast_rm_chunked_program_factory.cpp:261-262` | reader/writer RTA[0] `src_buffer` / `dst_buffer` | `TensorBinding` |
| `typecast_rm_chunked_program_factory.cpp:185-195` | compute CTA[0..3] | named CTAs + DFB bindings |
| all four factories | reader/writer RTA[1..2] positional (`num_pages`/`num_rows`, `start_id`/`start_row_id`) | named RTAs of the same names |
| kernel side: `reader_unary_interleaved_start_id.cpp:17` | `constexpr uint32_t cb_id_in0 = 0;` (hardcoded CB index) | `dfb::in` (comment about the buffer's role relocated to the DFB construction) |
| kernel side: `reader_unary_interleaved_start_id.cpp:20`, `writer_unary_interleaved_start_id.cpp:19` | `get_local_cb_interface(cb_id).fifo_page_size` | `DataflowBuffer::get_entry_size()` (whitelist §B: query the object) |
| kernel side: all five dataflow kernels | `TensorAccessorArgs<N>()` + address RTA + `TensorAccessor(args, addr)` | `TensorAccessor(tensor::input)` / `(tensor::output)` |

- **Page-size 3rd-argument CTAs/RTAs**: none — no accessor site passes a third argument (audit GREEN).
- **Semaphore-ID RTAs**: none — the op has no semaphores.
- **Case 2 (raw pointer) bindings**: none — every accessor is Case 1 (page access through
  `TensorAccessor`). No `get_bank_base_address` bridge is used anywhere in this port.

## Applied Patterns

- **Two-toucher / disjoint-node same-source KernelSpecs** — `compute_group_1` + `compute_group_2`
  over disjoint node sets each bind one endpoint role on `in` / `out`; no
  `allow_instance_multi_binding`, no self-loop. (`TypecastProgramFactory`,
  `TypecastRowMajorChunkedProgramFactory`.)
- **Self-loop DFB binding** — `TypecastShardedProgramFactory`'s `out` DFB: compute is the only
  toucher (it `reserve_back`/`push_back`es into the borrowed output buffer; there is no writer
  kernel to drain it), so compute binds it as **both** PRODUCER and CONSUMER. No kernel-side
  `wait_front`/`pop_front` is added to "balance" it.
- **Borrowed-memory DFB** — `TypecastShardedProgramFactory`'s `in` and `out`
  (`borrowed_from = INPUT / OUTPUT`); the backing L1 address refreshes from the `TensorArgument`
  each enqueue, so no `dfb_run_overrides` entry is needed.
- **Fork of a shared dataflow kernel** — the three eltwise/unary donors, per the *Caution*
  entry; forks named `*_metal2.cpp` inside this op's directory.
- **Multi-variant factory** — four factories in the `program_factory_t` variant, ported together
  because they share the compute entry point.

## Deferred / Flagged

- **`unpack_modes` needs a rule the legacy vector didn't encode.** Legacy sets
  `unpack_to_dest_mode[c_0] = UnpackToDestFp32` whenever `preserve_fp32_precision`, and legacy JIT
  ignores that entry unless the format is `Float32`. Metal 2.0 additionally *requires* an explicit
  entry for a consumed `Float32` DFB when `enable_32_bit_dest` is true. Plan: emit
  `{in → UnpackToDest}` iff `preserve_fp32_precision` (faithful to legacy), else
  `{in → UnpackToSrc}` when `fp32_dest_acc_en && input df == Float32` (the newly-required explicit
  entry, whose value is the legacy `Default`). For the sharded factory the `out` self-loop makes
  compute a *consumer* of `out` too, so `out` also gets an explicit `UnpackToSrc` there.
  `UnpackToSrc` lowers to the legacy `UnpackToDestMode::Default`, so this is byte-identical.
- **Assumption recorded**: `preserve_fp32_precision ⇒ fp32_dest_acc_en` for every reachable call
  (`typecast.cpp:38` derives the latter from the former, and `ttnn::prim::typecast` has exactly one
  caller). If that ever decouples, a `preserve_fp32_precision && !fp32_dest_acc_en` call on a
  ≤16-bit input would now be *rejected* by the Metal 2.0 validator where legacy silently no-op'd.
  Flagged in the port report; no code change made.
- No structural issue the audit missed; no feature gate fired during planning.
