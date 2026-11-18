# Port Plan — concat (`ConcatS2SMultiProgramFactory`)

Port plan for `data_movement/concat`'s `ConcatS2SMultiProgramFactory`, ported from the ProgramDescriptor API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

Scope: this one factory and its kernel. `ConcatProgramFactory`, `ConcatS2SRMProgramFactory`, `ConcatS2STiledProgramFactory` are already Metal 2.0; `ConcatS2IProgramFactory`, `ConcatBlockShardedProgramFactory`, `ConcatTiledUnalignedProgramFactory` stay on `create_descriptor` (the variant mixes concepts, which is legal). Port base: `0de8c7f3aad` (origin/main, 2026-11-08), branch `edwinlee/port_concat`.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`create_descriptor` returning `ProgramDescriptor`, `concat_s2s_multi_program_factory.hpp:14`), a struct in `ConcatDeviceOperation::program_factory_t` (`concat_device_operation.hpp:35`) — not the direct-descriptor shape.
- Variants: single factory; two runtime axes (RM vs TILE page size, height vs width concat) change only *values*, not kernel source or bindings.
- Custom `compute_program_hash`: present at `concat_device_operation.cpp:102-136` (factory index + input-aliasing signature mixed into the default hash of attributes + tensor args) — **left intact**. It hashes the full tensor args, so every value the factory derives from shard specs is cache-key-determined.

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_s2s_tensor_concat.cpp` | `all_cores` = input 0 shard grid | `[0] cb_dst_id=16, [1] page_size, [2] output_stride, [3] num_input_tensors, [4] output_block_stride, [5] block_start=0, [6] block_count=reader_block_count` (`:239-240`) | none | 5N per core, identical on every core: per input `{pages_per_stick, reader_sticks, write_offset, read_offset=0, block_stride}` (`:262-266`) | none | none | unset → **O2** (DM) | `ReaderConfigDescriptor{}` → RISCV_1 / NOC_0 / DEDICATED |
| writer | same source | `all_cores` | same, `[5] writer_block_start, [6] writer_block_count` (`:241-248`) | none | 5N per core: `{pages_per_stick, writer_sticks, write_offset + output_stride*writer_stick_offset, page_size*pages_per_stick*writer_stick_offset, block_stride}` (`:267-271`) | none | none | unset → **O2** | `WriterConfigDescriptor{}` → RISCV_0 / NOC_1 / DEDICATED |

### CBs
| index | total_size | core_ranges | data_format | page_size | tile | backing |
|---|---|---|---|---|---|---|
| 0..N-1 (one per input) | `page_size * input_num_pages` | `all_cores` | output dtype format | `page_size` | not set | `.buffer = input_tensors[i].buffer()` (borrowed) |
| 16 | `page_size * output_num_sticks * output_num_pages_per_stick` | `all_cores` | output dtype format | `page_size` | not set | `.buffer = output.buffer()` (borrowed) |

Kernel-touch census (both instances run the same source on every core): input CB *i* — `get_read_ptr()` only (`reader_s2s_tensor_concat.cpp:45-46`), id = loop variable; output CB — `get_write_ptr()` only (`:31-32`). No `reserve_back`/`push_back`/`wait_front`/`pop_front` anywhere. Every CB is a sync-free address source.

### Semaphores
none

### Tensor accessors
none — no `TensorAccessor`/`TensorAccessorArgs`; tensor memory is reached only through the borrowed CBs (`Buffer*` in `CBDescriptor::buffer`, `:168`, `:189`).

### Work split
n/a — no `split_work_to_cores`. Every core of the shard grid builds its own output shard from its own input shards. The two instances split the work by `choose_risc_block_partition` (`:48-65`): by leading block (CTAs `block_start`/`block_count`) or by sticks within a block (RTAs).

### Shared kernels
none — `grep -rl reader_s2s_tensor_concat ttnn/cpp/ttnn/operations/` hits only `concat_s2s_multi_program_factory.cpp` (and the audit doc). Not bound by any other factory or op.

### Flags
- `concat_device_operation.cpp:385-404` (`calculate_max_tensors_per_concat`) describes S2SMulti's CTA/RTA layout and its "CB 0..N-1, output at 16" ceiling. Off-limits device-op code; goes stale with this port → report.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` — `create_program_artifacts` returning `ProgramArtifacts`.
- **Custom `compute_program_hash`**: present at `concat_device_operation.cpp:102-136` — leave intact.
- **Implementation notes**: no `override_runtime_arguments`, no pybind `create_descriptor`, no op-owned tensors. The header switches to `ttnn/metal_v2_artifacts.hpp` like the S2SRM sibling.

## Planned Spec Shape

Departure from the recipe's default 1:1 CB→DFB mapping, per the brief: borrowed CBs selected by a runtime index have no DFB translation (no variadic DFB bindings), so the inputs and output become `LocalTensorAccessor`s built from tensor bindings instead. They meet the sync-free criterion trivially (no credit calls; one source binds them).

- KernelSpecs: `reader`, `writer` — same source, 1:1 with the legacy descriptors. Each binds `input_0..input_{N-1}` and `output` as `TensorBinding`s, and declares `TensorBindingSequence{.sequence_name = "inputs", .members = {input_0, …}}`.
- DataflowBufferSpecs: **none** (all N+1 legacy CBs become tensor bindings; see Dropped Plumbing).
- SemaphoreSpecs: none.
- TensorParameters: `input_0..input_{N-1}` (one per input, `tensor_spec()`), `output`. Names function-local (unity build).
- WorkUnitSpecs: one, `main` = `{reader, writer}` over `all_cores`.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared resources |
|---|---|---|---|
| reader (`:282-294`), writer (`:296-308`) of `reader_s2s_tensor_concat.cpp`, distinct `block_start`/`block_count` CTAs | `reader`, `writer`, each with its own CTA table | `main` (both, `all_cores`) | no DFBs; both bind every input/output `TensorParameter` (two kernels binding one parameter is legal) |

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:160-169`, kernel `:45-46` | input CB *i* (`buffer_index = i`, `.buffer = input.buffer()`), `DataflowBuffer input_dfb(input_id)` | `TensorBinding{input_i, "input_i"}` + sequence `inputs`; kernel `make_local_tensor_accessors<uint32_t>(tensor::inputs)[input_id]` |
| factory `:181-190`, CTA 0, kernel `:19,31-32` | output CB 16, `cb_dst_id` CTA, `DataflowBuffer output_dfb(output_dfb_id)` | `TensorBinding{output, "output"}`; kernel `LocalTensorAccessor<uint32_t> output(tensor::output)` |
| CTA 3 / kernel `:22` | `num_input_tensors` | dropped — `inputs.size()` (sequence length) |
| CTAs 1, 2, 4, 5, 6 | positional `page_size, output_stride, output_block_stride, block_start, block_count` | named CTAs, same names; kept compile-time and per-kernel |
| RTAs `0..5N-1`, kernel `:39-43` `get_arg_val(arg_idx++)` | 5 interleaved values per input | five runtime **array** args of length N, per node: `input_num_pages_per_stick`, `input_num_sticks`, `input_write_offset`, `input_read_offset`, `input_block_stride`; kernel `get_arg(args::x)[input_id]` |
| kernel `:56-61, 72-88` | `UnicastEndpoint{my_x, my_y, .addr}` NoC source built from the CB read pointer | `LocalTensorAccessor` as the NoC source (loopback), `{.offset_bytes = read offset}`; destination `output` LTA `{.offset_bytes = write offset}` |

No buffer-address RTAs, semaphore RTAs, `TensorAccessorArgs`, or page-size 3rd args exist in legacy.

## Applied Patterns

- Tensor binding sequence for a runtime-indexed input set — modelled on `ConcatProgramFactory` (`concat_program_factory.cpp:227-254`), with `make_local_tensor_accessors` instead of `make_tensor_accessors` because the shards are node-local.
- Sync-free borrowed CB → `LocalTensorAccessor` (normally the post-port `sync_free_dfbs.md` pass; done in the port per the brief because the borrowed-DFB route cannot express a runtime-selected input).
- Dual-instance same-source work split (reader + writer over one grid), shape from `ConcatS2SRMProgramFactory`.
- Runtime array arguments (#63805) in place of RTA varargs, per the brief. Values per node (Edwin's decision), every node identical, reader and writer different.

## Deferred / Flagged

- Kernel-side whitelist deviation, authorised by the brief: the transfer endpoints change (`UnicastEndpoint{my_x,my_y,addr}` → `LocalTensorAccessor` + `offset_bytes`). The loopback NoC address an LTA resolves (`internal/local_mem/noc_traits.h:28-38`) is built from exactly the same `UnicastEndpoint{my_x, my_y, addr}`, so the transfers are byte-identical.
- `TT_FATAL(num_input_tensors <= 16, …)` (`:88`) stays with its limit; its message (which blames CB ids) is reworded per the brief.
- Unproven seams (brief): LTA into `set_async_read_state`/`async_read_with_state` (no prior user), `make_local_tensor_accessors` (no ttnn caller), runtime array args (no ttnn caller). Stop and report if any fails.

---

# Port Plan — concat (`ConcatBlockShardedProgramFactory`)

Second port in this op, 2026-11-18. Scope: `ConcatBlockShardedProgramFactory` and its kernel `reader_writer_block_sharded_concat.cpp` only. Port base: `e38dfa66343` (origin/main, 2026-11-18, includes #62362), branch `edwinlee/port_concat`. Brief: `BlockShardedHandoff.md`, appended to `METAL2_PORT_BRIEF.md`. Audit: RED, waived for this factory (see the audit's waiver note).

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`create_descriptor`, `concat_block_sharded_program_factory.hpp:13`), a struct in `program_factory_t` (`concat_device_operation.hpp:36`).
- Variants: one factory; RM vs TILE (unit size) and width vs height concat change only values.
- Custom `compute_program_hash` at `concat_device_operation.cpp:102-136` — left intact.

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `kernels/dataflow/reader_writer_block_sharded_concat.cpp` | `all_cores` = input 0 shard grid (one rectangle) | `[0] cb_dst_id=16` (`:443`) | none | per core: `num_transfers`, then 9 per transfer `{src_noc_x, src_noc_y, src_cb_id, src_l1_offset, src_stride, dst_offset, dst_stride, copy_size, num_rows}` — first `ceil(total/2)` of the core's transfers (`:463-491`) | none | none | unset → O2 | `ReaderConfigDescriptor{}` → RISCV_1 / NOC_0 |
| writer | same source | `all_cores` | same | none | same layout, the remaining transfers | none | none | unset → O2 | `WriterConfigDescriptor{}` → RISCV_0 / NOC_1 |

### CBs
| index | total_size | core_ranges | data_format | page_size | backing |
|---|---|---|---|---|---|
| 0..N-1 | `unit_size * input shard units` | `all_cores` | output dtype | `unit_size` | `.buffer = input_tensors[i].buffer()` (borrowed) |
| 16 | `unit_size * output shard units` | `all_cores` | output dtype | `unit_size` | `.buffer = output.buffer()` (borrowed) |

Kernel-touch census: input CB id comes from a runtime arg (`DataflowBuffer src_dfb(src_dfb_id)`, kernel `:44`), used only for `get_read_ptr()`, and that *local* address is then read on a *remote* core (`:45, 55`); output CB only `get_write_ptr()` (`:27-28`). No credit calls. All sync-free address sources.

### Semaphores / Tensor accessors
none / none.

### Work split
No `split_work_to_cores`. Every core of the grid builds its own output shard; "Pass 1" (`:258-385`) lists each core's rectangle copies, many from other cores' input shards. Reader gets `ceil(total/2)` of a core's copies, writer the rest.

### Shared kernels
none — the kernel is bound only by this factory.

### Flags
- `physical_cores` table + `shard_core(r, c)` (`:240-256`) assume core `(start_x + gx, start_y + gy)` holds shard `(r, c)` at the same L1 address on every core.
- `calculate_max_tensors_per_concat` (`concat_device_operation.cpp:378-382`) cites the CB-id ceiling for the 16-input cap; device op is off-limits, the cap stays.

## TTNN ProgramFactory
- Concept: `ProgramSpecFactoryConcept` (`create_program_artifacts` → `ProgramArtifacts`). Header switches to `ttnn/metal_v2_artifacts.hpp`.
- Custom hash left intact. No `override_runtime_arguments`, no pybind `create_descriptor`, no op-owned tensors.

## Planned Spec Shape
- KernelSpecs `reader`, `writer` — one source, 1:1 with legacy. Each binds `input_0..input_{N-1}` and `output` and declares `TensorBindingSequence{"inputs", input_0..}`.
- DataflowBufferSpecs: none. SemaphoreSpecs: none.
- TensorParameters: `input_0..input_{N-1}`, `output`.
- WorkUnitSpecs: `main` = `{reader, writer}` over `all_cores`.

## Preserved Multiplicity
| legacy KernelDescriptors | KernelSpecs | WorkUnitSpecs | shared resources |
|---|---|---|---|
| reader, writer of one source, different per-core transfer lists | `reader`, `writer` | `main` | none; both bind every TensorParameter |

## Dropped Plumbing
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:214-226`, kernel `:44-45` | input CB *i* borrowed; `DataflowBuffer(src_dfb_id).get_read_ptr()` read on the remote core | `TensorBinding` per input + sequence `inputs`; kernel `with_input` → `TensorAccessor(std::get<I>(tensor::inputs))` → `ShardView(..).shard(shard_id, noc_id)` |
| factory `:228-238`, CTA 0, kernel `:24-28` | output CB 16, `cb_dst_id` CTA | `TensorBinding{output}`; kernel `LocalTensorAccessor<uint32_t>(tensor::output)` |
| factory `:240-256`, RTAs `src_noc_x/y`, kernel `:48-56` | physical-core table, raw `UnicastEndpoint{x, y, addr}` | per-copy `shard_id = shard_row × shards_per_row + shard_col` |
| RTAs, kernel `:30-42` | `num_transfers` + 9 positional words per transfer | named RTA `num_copies` + 8 per-node runtime **array** args (`input_index, shard_id, src_offset, src_stride, dst_offset, dst_stride, copy_size, num_rows`), each sized to that RISC's max copy count over cores (≥ 1) and zero-padded |

## Applied Patterns
- Tensor binding sequence + `with_input` (runtime → compile-time input index), from the ported interleaved reader.
- Remote shard by id via `ShardView::ResolvedShard` (#62362): one lookup per copy, a 32-bit add per row.
- Dual-instance same-source work split, from `ConcatS2SRMProgramFactory` / S2SMulti.
- Runtime array args with padding (the brief's `reshard_reader` pattern).

## Deferred / Flagged
- Kernel-side whitelist deviation, authorised by the brief: transfer endpoints change (raw coordinates → `ResolvedShard` source, LTA destination).
- **Brief correction:** the brief takes `shards_per_row` from the distribution spec's `tensor_shape_in_pages()[-1] / shard_shape_in_pages()[-1]`. That is wrong for TILE: the spec can merge dimensions (a 2 × 2 grid of one-tile shards is `[4]` pages in 1D), so it gave id 4 for shard (1, 0). Taken from the tensor's 2D physical width over the shard width instead; the temporary core check confirmed it on every reaching test.
- New `TT_FATAL`s: output core order (brief), and "tensor has a distribution spec". The brief's temporary shard-id→core development assert was used during testing and removed before commit.
- RTA budget `TT_FATAL`s re-derived at 8 words per copy: 31 copies per RISC (was 28).
- 16-input cap kept (`concat_device_operation.cpp:232`, `:449`).
