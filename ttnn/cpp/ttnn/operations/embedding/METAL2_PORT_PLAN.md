# Port Plan — `embedding`

Port plan for `ttnn/cpp/ttnn/operations/embedding`, ported from `ProgramDescriptor` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Scope — two of three factories.** `METAL2_PORT_BRIEF.md` is scoped to `EmbeddingsRMProgramFactory`
and `EmbeddingsTilizedIndicesProgramFactory`. `EmbeddingsFusedProgramFactory` is blocked (Type-2
offset-base-pointer wall at `embeddings_tilize.cpp:36`) and is **not touched** by this port: it stays
on `ProgramDescriptorFactoryConcept` inside the same `program_factory_t` variant, which the framework
dispatches per-factory.

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — each factory defines
  `static ProgramDescriptor create_descriptor(const EmbeddingParams&, const EmbeddingInputs&, Tensor&)`.
- Variants: three factories in one device-operation variant
  (`embedding_device_operation.hpp:24-28`), selected by `select_program_factory`
  (`embedding_device_operation.cpp:17-26`): TILE-layout indices → `EmbeddingsTilizedIndicesProgramFactory`;
  else `tilized` attribute → `EmbeddingsFusedProgramFactory` (blocked); else `EmbeddingsRMProgramFactory`.
- Custom `compute_program_hash`: none — default reflection-based hash. No `attribute_values` / `to_hash`
  backdoor either.
- `override_runtime_arguments`: none on any factory.
- Pybound `create_descriptor`: none. `embedding_nanobind.cpp:40-52` exposes only `ttnn::embedding` and
  the `EmbeddingsType` enum, so no pybind line is affected.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN
factory analysis section. Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section
below.)*

Neither in-scope factory declares a compute kernel, so no `ComputeHardwareConfig`, no `unpack_modes`,
and no `O3` opt-level rule applies anywhere in this port.

---

### Variant: `EmbeddingsRMProgramFactory`

Configuration axes that change the spec:

| axis | values | derived at |
|---|---|---|
| `output_sharded` | `false` (interleaved) / `true` (height-sharded) | `embeddings_rm_program_factory.cpp:44` |
| `use_chunked` | `false` / `true` (`!output_sharded && rounded_weight_page_size > 1 MB`) | `:105` |
| `embeddings_type` | `GENERIC` / `PADDED` / `BINARY` | `EmbeddingParams` |
| `embeddings_index_type` | `UINT32` / `BFP16` (from the index dtype) | `:191-196` |

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/embeddings.cpp` | `all_cores` | 0 `out_cb_index`(c_0), 1 `src1_cb_index`(c_1), 2 `src2_cb_index`(c_2), 3 `input_page_size`, 4 `weight_page_size`, 5 `block_height`, 6 `block_height*input_elem_size`, 7 `chunk_size`, 8 `num_chunks`, 9 `last_chunk_size`, 10.. `TensorAccessorArgs(a)` then `TensorAccessorArgs(weights)` (`:177-189`) | none | per core (`:263-274`): 0 `a_buffer` (`Buffer*`), 1 `weights_buffer` (`Buffer*`), 2 batch offset, 3 weights byte offset, 4 `local_num_blocks`, 5 index-in-block, 6 `pad_token` (**PADDED only**) | none | `{<EmbeddingsType>, "1"}`, `{<EmbeddingsIndexType>, "1"}` (`:198-199`) | unset → **O2** (DM default) | `ReaderConfigDescriptor{}` → RISCV_1 / NOC_0 / DM_DEDICATED_NOC |
| writer (chunked; `use_chunked`) | `device/kernels/dataflow/embeddings_rm_writer_chunked.cpp` | `all_cores` | 0 `out_cb_index`, 1 `output_page_size`, 2 `chunk_size`, 3 `num_chunks`, 4 `last_chunk_size`, 5.. `TensorAccessorArgs(output)` (`:213-219`) | none | per core (`:280`): 0 `output_buffer` (`Buffer*`), 1 `local_num_blocks`, 2 `input_offset` | none | none | unset → **O2** | `WriterConfigDescriptor{}` → RISCV_0 / NOC_1 / DM_DEDICATED_NOC |
| writer (stick; `!use_chunked`) | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` **(shared pool)** | `all_cores` | 0 `out_cb_index`, 1 `output_page_size` (**dead** — kernel hardcodes `TensorAccessorArgs<2>()`), 2.. `TensorAccessorArgs(output)` (`:230-232`) | none | per core (`:283`): 0 `output_buffer` (`Buffer*`), 1 `output_page_size`, 2 `local_num_blocks`, 3 `input_offset` | none | none | unset → **O2** | `WriterConfigDescriptor{}` |

No writer kernel at all when `output_sharded` (`:211`).

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) | notes |
|---|---|---|---|---|---|---|
| c_0 | sharded: `output.buffer()->aligned_size_per_bank()`; else `buffering_size * chunk_size` (`buffering_size` 1 or 2) | `all_cores` | `weights_cb_data_format` | `chunk_size` | not set | `.buffer = out_buffer` when `output_sharded` (`:133-135`) — borrowed memory |
| c_1 | `block_height * index_page_size` | `all_cores` | `input_cb_data_format` | same as total | not set | index scratch |
| c_2 | PADDED: `cache_page_size`; BINARY: `2 * cache_page_size`; **not allocated under GENERIC** | `all_cores` | `weights_cb_data_format` | `cache_page_size` | not set | local weight cache (`:151-173`) |

`chunk_size` = `rounded_weight_page_size` unless `use_chunked`, in which case it is the 1 MB-budget
chunk. `cache_page_size` = `round_up_to_mul32(weight_page_size)`.

#### Semaphores

none — the op declares no semaphores of any kind.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `embeddings_rm_program_factory.cpp:188` (args) | `input_tensor_arg` (`a`) | reader slot 0 |
| `embeddings_rm_program_factory.cpp:189` (args) | `weight_arg` | reader slot 1 |
| `embeddings_rm_program_factory.cpp:219` (args, chunked) | output | chunked-writer slot 0 |
| `embeddings_rm_program_factory.cpp:232` (args, stick) | output | stick-writer slot 0 |

Device sites: `embeddings.cpp:39,40`; `embeddings_rm_writer_chunked.cpp:26`;
`writer_unary_stick_layout_interleaved_start_id.cpp:20`.

#### Work split

- Driver (interleaved): `split_work_to_cores(compute_with_storage_grid_size, problem_size)` (`:82-89`),
  `problem_size = num_output_rows`.
- Driver (height-sharded): not a work-split call — `all_cores = shard_spec.grid`,
  `core_group_1 = all_cores`, `num_blocks_per_core_group_1 = shard_spec.shape[0]`,
  `num_blocks_per_core_group_2 = 0`, and `row_major` follows the shard orientation (`:74-80`).
- Per-core block counts reach the kernels as a **runtime** arg (`local_num_blocks`), not a per-group
  CTA, in the legacy factory — so there is no per-group `KernelDescriptor` multiplicity to preserve.

---

### Variant: `EmbeddingsTilizedIndicesProgramFactory`

Configuration axes: `embeddings_type`, `embeddings_index_type`, and
`ONLY_ONE_FACE_COLUMN` (`a.logical_shape()[-1] <= FACE_HEIGHT`, `:157-159`). There is no sharded-output
branch — one output configuration only.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/embedding_ind_tilized.cpp` | `all_cores` | 0 `src0_cb_index`(c_0), 1 `src1_cb_index`(c_1), 2 `src2_cb_index`(c_2), 3 `input_page_size` (**dead** — never read), 4 `weight_page_size`, 5 `a.logical_shape()[-1]`, 6 `FACE_HEIGHT` (**dead** — read into an unused `input_block_size_bytes`), 7.. `TensorAccessorArgs(a)` then `TensorAccessorArgs(weights)` (`:136-145`) | none | per core (`:208-219`): 0 `a_buffer` (`Buffer*`), 1 `weights_buffer` (`Buffer*`), 2 `curr_tile`, 3 `face_offset`, 4 `local_num_blocks`, 5 `col_offset`, 6 `col_offset % FACE_HEIGHT`, 7 `pad_token` (**PADDED only, never read**) | none | `{<EmbeddingsType>, "1"}`, `{<EmbeddingsIndexType>, "1"}`, `{"ONLY_ONE_FACE_COLUMN","1"}` when the row fits one face column | unset → **O2** | `ReaderConfigDescriptor{}` |
| writer | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` **(shared pool)** | `all_cores` | 0 `output_cb_index` (== c_0), 1 `output_page_size` (**dead**), 2.. `TensorAccessorArgs(output)` (`:169-170`) | none | per core (`:223-224`): 0 `output_buffer` (`Buffer*`), 1 `output_page_size`, 2 `local_num_blocks`, 3 `weight_offset` | none | none | unset → **O2** | `WriterConfigDescriptor{}` |

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| c_0 | `2 * rounded_weight_page_size` | `all_cores` | `weights_cb_data_format` | `rounded_weight_page_size` | not set |
| c_1 | `FACE_HEIGHT * index_page_size` | `all_cores` | `input_cb_data_format` | same as total | not set |
| c_2 | PADDED: `cache_page_size`; BINARY: `2 * cache_page_size`; **not allocated under GENERIC** (`:108-130`) | `all_cores` | `weights_cb_data_format` | `cache_page_size` | not set |

`output_cb_index = src0_cb_index` (`:132`) — c_0 is simultaneously the reader's weight-staging buffer
and the writer's output CB, which is what makes it a genuine producer/consumer pair.

#### Semaphores

none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `embeddings_tilized_indices_program_factory.cpp:144` | `input_tensor_arg` (`a`) | reader slot 0 |
| `embeddings_tilized_indices_program_factory.cpp:145` | `weight_arg` | reader slot 1 |
| `embeddings_tilized_indices_program_factory.cpp:170` | output | writer slot 0 |

Device sites: `embedding_ind_tilized.cpp:35,36`;
`writer_unary_stick_layout_interleaved_start_id.cpp:20`.

#### Work split

- Driver: `split_work_to_cores_aligned(compute_with_storage_grid_size, problem_size, FACE_HEIGHT)`
  (`:67`), `problem_size = num_cols * batch_size`.
- `num_cores`, `all_cores`, `core_group_1`, `core_group_2`,
  `num_blocks_per_core_group_1`, `num_blocks_per_core_group_2` from `CoreSplitResult` (`:69-76`).
- Core order for the RTA loop: `grid_to_cores(num_cores, num_cores_x, num_cores_y, false)` (`:183`).
- As with the RM factory the per-core count is an RTA, so there is no per-group CTA multiplicity.

---

### Shared kernels

| kernel source | kind of sharing | other consumers | `_metal2` fork beside it? | rung |
|---|---|---|---|---|
| `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | **borrowed** (shared pool `ttnn/cpp/ttnn/kernel/`) | `data_movement/concat` (`concat_program_factory.cpp:234`, row-major path), `data_movement/copy` (`copy_same_memory_config_program_factory.cpp:39`, row-major interleaved path) | **No** | **2 — create the fork** |
| `device/kernels/dataflow/embeddings_common.hpp` | **intra-op** (included by all three readers, one of which belongs to the blocked factory) | `device/kernels/dataflow/embeddings_tilize.cpp` (blocked `EmbeddingsFusedProgramFactory`) | **No** | **2 — create the fork** |

Census notes:

- `grep -rn writer_unary_stick_layout_interleaved_start_id ttnn/` also hits
  `data_movement/slice/device/kernels/dataflow/slice_writer_unary_stick_layout_interleaved_start_id.cpp`
  and its quasar copy. That is a **different file** with a similar name, not a co-borrower — discarded.
  `ttnn/sources.cmake:170` is the build-file hit — also not a consumer, but see the build note below.
- `embeddings_rm_writer_chunked.cpp` is bound only by `EmbeddingsRMProgramFactory` — **not shared**,
  converted in place.
- **Build-system note.** The Caution's "no build-system change is needed for the new file" holds for
  the op's own kernels (`ttnn/cpp/ttnn/operations/embedding/CMakeLists.txt` has
  `file(GLOB_RECURSE kernels device/kernels/*.cpp device/kernels/*.hpp)`), so
  `embeddings_common_metal2.hpp` is picked up automatically. It does **not** hold for the shared pool
  `ttnn/cpp/ttnn/kernel/`, whose files are enumerated explicitly in `ttnn/sources.cmake`
  (`TTNN_CORE_JIT_API_HEADERS`). The stick-writer fork needs one line added there — the precedent is
  `cpp/ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp` on line 164.

Why `embeddings_common.hpp` must be forked rather than converted in place: the header both
(a) includes `api/dataflow/circular_buffer.h` and constructs a `CircularBuffer` from the CB index it
is handed, and (b) reads the pad token positionally with `get_arg_val<uint32_t>(pad_token_arg_idx)`.
Both must change for a Metal 2.0 caller, and both would break `embeddings_tilize.cpp`, which stays on
the legacy positional-arg API with the blocked factory.

### Flags

- No unreferenced kernel file in the op directory — every kernel under `device/kernels/` is bound by a
  factory.
- Every descriptor type the two in-scope factories use (`KernelDescriptor`, `CBDescriptor`,
  `ReaderConfigDescriptor`, `WriterConfigDescriptor`) maps onto an audit Appendix A entry. No
  descriptor type outside the audit's scan.
- **Dead compile-time / runtime args carried by the legacy factories** (recorded here so the drops
  below are traceable, and repeated in the port report):
  - `embeddings_tilized_indices_program_factory.cpp:140` — CTA slot 3 `input_page_size`, never read by
    `embedding_ind_tilized.cpp` (which derives the value at runtime at `:51`).
  - `embeddings_tilized_indices_program_factory.cpp:143` — CTA slot 6 `FACE_HEIGHT`, read into
    `embedding_ind_tilized.cpp:31`'s `input_block_size_bytes` and never used.
  - `embeddings_tilized_indices_program_factory.cpp:217` — RTA slot 7 `pad_token`, never read; the
    kernel takes its pad token from slot 6 instead (see the *Preserved defect* note below).
  - `embeddings_rm_program_factory.cpp:231` and
    `embeddings_tilized_indices_program_factory.cpp:169` — CTA slot 1 `output_page_size` on the shared
    stick writer, never read (the kernel hardcodes `TensorAccessorArgs<2>()` and gets `stick_size`
    from an RTA instead).
- **Preserved defect — the tilized-indices pad-token slot.** `embedding_ind_tilized.cpp:42` passes
  `pad_token_arg_idx = 6`, but this factory puts `col_offset % FACE_HEIGHT` in slot 6 (`:215`) and the
  real pad token in slot 7 (`:217`). Under `EmbeddingsType::PADDED` with TILE-layout indices the
  kernel therefore treats a per-core face-column index (0-15) as the pad token. The brief directs the
  port to **preserve this**, so the ported kernel reads its pad token from the same value it reads
  today. See *Dropped Plumbing* for how, and the port report for the write-up.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`, for both in-scope factories.
- **Custom `compute_program_hash`**: none — default reflection-based hash. Nothing to preserve.
- **Implementation notes**:
  - Both factory headers change `static ProgramDescriptor create_descriptor(...)` to
    `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`; the parameter
    list is already the concept's `(attributes, tensor_args, tensor_return_value)`.
  - `EmbeddingsFusedProgramFactory` keeps `create_descriptor`, so `program_factory_t` carries a mixed
    set of factory concepts for the duration. No device-op-class edit is forced: no pybound
    `create_descriptor`, no pybind-hook-only factory parameter.
  - `ttnn_op_embedding` is a **unity-build target** (`CMakeLists.txt`:
    `TT_ENABLE_UNITY_BUILD(ttnn_op_embedding)`) with all three factory `.cpp` files in it, so no
    file-scope or anonymous-namespace symbols are introduced: every spec-name constant is declared
    **inside** its `create_program_artifacts` body.

## Planned Spec Shape

### Variant: `EmbeddingsRMProgramFactory`

- **KernelSpecs** (1:1 with legacy):
  - `READER{"reader"}` → `embeddings.cpp`
  - `WRITER{"writer"}` → `embeddings_rm_writer_chunked.cpp` when `use_chunked`, else the
    `_metal2` fork of the shared stick writer. Declared only when `!output_sharded`.
- **DataflowBufferSpecs** (1:1 with legacy `CBDescriptor`s; `num_entries` = legacy
  `total_size / page_size`, which is how the legacy CB derives its own page count):
  - `OUTPUT{"output"}` (legacy c_0) — `entry_size = chunk_size`;
    `num_entries = out_cb_size / chunk_size`. `borrowed_from = OUTPUT_PARAM` when `output_sharded`.
  - `INDEX_SCRATCH{"index_scratch"}` (legacy c_1) — `entry_size = block_height * index_page_size`,
    `num_entries = 1`.
  - `WEIGHT_CACHE{"weight_cache"}` (legacy c_2) — `entry_size = cache_page_size`,
    `num_entries = 1` (PADDED) / `2` (BINARY). **Declared only under PADDED / BINARY** — conditional
    DFB.
  - `data_format_metadata` copied from each legacy CB's `data_format`. No legacy CB sets
    `format_descriptors[i].tile`, so `tile_format_metadata` stays `nullopt` throughout.
- **SemaphoreSpecs**: none — the op declares no semaphores.
- **TensorParameters**: `INPUT{"input"}`, `WEIGHTS{"weights"}`, `OUTPUT_PARAM{"output"}` — one per
  distinct originating tensor. `OUTPUT_PARAM` is declared in every config: bound by the writer in the
  interleaved configs, and named by `OUTPUT`'s `borrowed_from` in the height-sharded one (a
  parameter named by `borrowed_from` counts as used even with no kernel binding).
- **WorkUnitSpecs**: one — `{"main", {READER[, WRITER]}, all_cores}`.
- **Op-owned tensors**: none.

DFB endpoint assignment, re-derived from the kernel-touch census per `(DFB, config)`:

| DFB | config | distinct touchers on a node | assignment |
|---|---|---|---|
| `OUTPUT` | interleaved, chunked | reader locked producer (`embeddings.cpp:71,77`) + chunked writer locked consumer (`embeddings_rm_writer_chunked.cpp:33,43`) | **1P + 1C** — reader PRODUCER, writer CONSUMER |
| `OUTPUT` | interleaved, non-chunked | reader locked producer + stick writer locked consumer (`:32,35`) | **1P + 1C** |
| `OUTPUT` | height-sharded | reader only — no writer kernel is created (`:211`) | **self-loop** — reader bound PRODUCER *and* CONSUMER, plus `borrowed_from` |
| `INDEX_SCRATCH` | all | reader only (`embeddings.cpp:47,48,96`) | **self-loop** |
| `WEIGHT_CACHE` | PADDED / BINARY | reader only, via `prepare_local_cache` | **self-loop** |
| `WEIGHT_CACHE` | GENERIC | not declared | **conditional DFB** |

The chunked writer's `get_read_ptr()` at `embeddings_rm_writer_chunked.cpp:34` is a public peek on its
own consumer binding, not a third toucher. Nothing here reaches ≥3 touchers or two kernels locked to
the same FIFO role, so **`allow_instance_multi_binding` is not set anywhere in this port**, and no DFB
is both self-looped and multi-bound.

### Variant: `EmbeddingsTilizedIndicesProgramFactory`

- **KernelSpecs**: `READER{"reader"}` → `embedding_ind_tilized.cpp`; `WRITER{"writer"}` → the
  `_metal2` fork of the shared stick writer. Both always present.
- **DataflowBufferSpecs**:
  - `OUTPUT{"output"}` (legacy c_0) — `entry_size = rounded_weight_page_size`, `num_entries = 2`.
  - `INDEX_SCRATCH{"index_scratch"}` (legacy c_1) — `entry_size = FACE_HEIGHT * index_page_size`,
    `num_entries = 1`.
  - `WEIGHT_CACHE{"weight_cache"}` (legacy c_2) — as in the RM factory; conditional on
    PADDED / BINARY.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT{"input"}`, `WEIGHTS{"weights"}`, `OUTPUT_PARAM{"output"}`.
- **WorkUnitSpecs**: one — `{"main", {READER, WRITER}, all_cores}`.
- **Op-owned tensors**: none.

| DFB | config | distinct touchers on a node | assignment |
|---|---|---|---|
| `OUTPUT` | all | reader locked producer (`embedding_ind_tilized.cpp:54,59`) + stick writer locked consumer | **1P + 1C** |
| `INDEX_SCRATCH` | all | reader only (`embedding_ind_tilized.cpp:47,48,128`) | **self-loop** |
| `WEIGHT_CACHE` | PADDED / BINARY | reader only | **self-loop** |
| `WEIGHT_CACHE` | GENERIC | not declared | **conditional DFB** |

### Kernel-side accessor names

Accessor names follow the kernel-local variables the legacy CB-index constants fed, so the kernel
diff stays a mechanical `cb_*` → `dfb_*` rename. The one exception is c_2, whose handle is consumed
inside `prepare_local_cache` — there the shared header's own parameter name (`local_cache_cb`) is the
kernel's vocabulary for it.

| DFB | reader accessor | writer accessor |
|---|---|---|
| `OUTPUT` | `in0` (`DataflowBuffer dfb_in0`) | chunked writer: `out0`; stick-writer fork: `out0` |
| `INDEX_SCRATCH` | `in1` (`DataflowBuffer dfb_in1`) | — |
| `WEIGHT_CACHE` | `local_cache` (constructed inside `prepare_local_cache`) | — |

Tensor accessor names: reader `input` / `weights`; writer `dst` (matching the writers' legacy `s0`
accessor built from `dst_addr`, and the name the stick-writer fork fixes for every later consumer).

## Preserved Multiplicity

none — no work-split multiplicity in legacy. Both factories build a **single** `KernelDescriptor` per
kernel over `all_cores` and deliver the per-core block count as a runtime argument
(`local_num_blocks`), so there is no per-group CTA to preserve and no demoting-per-group-CTA hazard.
`core_group_1` / `core_group_2` are used only to pick that runtime value.

## Dropped Plumbing

### `EmbeddingsRMProgramFactory`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader CTA slots 0-2 (`:178-180`) | `out_cb_index`, `src1_cb_index`, `src2_cb_index` (`CBIndex::c_*`) | `DFBBinding`s to `OUTPUT` / `INDEX_SCRATCH` / `WEIGHT_CACHE`; kernel reads `dfb::in0`, `dfb::in1`, `dfb::local_cache` |
| reader CTA slots 10.. (`:188-189`) | `TensorAccessorArgs(*a.buffer()).append_to(...)`, then `TensorAccessorArgs(*weights.buffer()).append_to(...)`; kernel `TensorAccessorArgs<10>()` + `next_compile_time_args_offset()` (`embeddings.cpp:37-38`) | `TensorBinding`s to `INPUT` / `WEIGHTS`; kernel `TensorAccessor(tensor::input)` / `TensorAccessor(tensor::weights)` |
| reader RTA slot 0 (`:264`) | `a_buffer` (`Buffer*`) → `embeddings.cpp:15` | `TensorBinding` to `INPUT` |
| reader RTA slot 1 (`:265`) | `weights_buffer` (`Buffer*`) → `embeddings.cpp:16` | `TensorBinding` to `WEIGHTS` |
| reader CTA slots 3-9 (`:181-187`) | positional | named CTAs `input_page_size`, `weight_stick_size`, `rows_per_block`, `input_block_size_bytes`, `chunk_size`, `num_chunks`, `last_chunk_size` |
| reader RTA slots 2-6 (`:266-272`) | positional | named RTAs `batch_offset`, `weights_offset`, `num_rows`, `index_idx`, `pad_token` (PADDED only) |
| chunked-writer CTA slot 0 (`:214`) | `out_cb_index` | `DFBBinding` to `OUTPUT` (`dfb::out0`) |
| chunked-writer CTA slot 1 (`:215`) | `output_page_size`, whose only use is the `TensorAccessor` third constructor argument (`embeddings_rm_writer_chunked.cpp:26`) | **dropped** — Class 2 (redundant / inert); the binding token supplies the aligned page size |
| chunked-writer CTA slots 5.. (`:219`) | `TensorAccessorArgs(*output.buffer())`; kernel `TensorAccessorArgs<5>()` (`:24`) | `TensorBinding` to `OUTPUT_PARAM`; kernel `TensorAccessor(tensor::dst)` |
| chunked-writer CTA slots 2-4 (`:216-218`) | positional | named CTAs `chunk_size`, `num_chunks`, `last_chunk_size` |
| chunked-writer RTA slot 0 (`:280`) | `output_buffer` (`Buffer*`) → `:15` | `TensorBinding` to `OUTPUT_PARAM` |
| chunked-writer RTA slots 1-2 (`:280`) | positional | named RTAs `num_sticks`, `start_id` |
| stick-writer CTA slot 0 (`:231`) | `out_cb_index` | `DFBBinding` to `OUTPUT` (`dfb::out0`) |
| stick-writer CTA slot 1 (`:231`) | `output_page_size` — **never read** by the kernel, a placeholder forced by its hardcoded `TensorAccessorArgs<2>()` | **dropped** — the hardcoded offset is gone with the binding, so the placeholder has no purpose |
| stick-writer CTA slots 2.. (`:232`) | `TensorAccessorArgs(*output.buffer())`; kernel `TensorAccessorArgs<2>()` (`:18`) | `TensorBinding` to `OUTPUT_PARAM`; fork reads `TensorAccessor(tensor::dst)` |
| stick-writer RTA slot 0 (`:283`) | `output_buffer` (`Buffer*`) → `:12` | `TensorBinding` to `OUTPUT_PARAM` |
| stick-writer RTA slots 1-3 (`:283`) | positional | named RTAs `stick_size`, `num_sticks`, `start_id` |

`stick_size` stays a **runtime** arg rather than being read off the DFB: in the non-chunked RM config
the DFB's `entry_size` is `rounded_weight_page_size` (allocator-aligned) while the write size is the
unaligned `output_page_size`, so the two are not interchangeable.

`stick_size` also carries the same value on every node, which makes it a common-runtime-arg candidate.
That conversion changes dispatch semantics and is **not** port work — noted for a later pass.

### `EmbeddingsTilizedIndicesProgramFactory`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader CTA slots 0-2 (`:137-139`) | `src0_cb_index`, `src1_cb_index`, `src2_cb_index` | `DFBBinding`s to `OUTPUT` / `INDEX_SCRATCH` / `WEIGHT_CACHE` |
| reader CTA slot 3 (`:140`) | `input_page_size` — **never read** | **dropped** (dead CTA; the kernel derives the value from the accessor at `:51`) |
| reader CTA slot 6 (`:143`) | `FACE_HEIGHT`, read into an unused `input_block_size_bytes` (`:31`) | **dropped** (dead CTA; the kernel's unused declaration goes with it) |
| reader CTA slots 7.. (`:144-145`) | `TensorAccessorArgs(a)` then `TensorAccessorArgs(weights)`; kernel `TensorAccessorArgs<7>()` + chaining (`:33-34`) | `TensorBinding`s to `INPUT` / `WEIGHTS` |
| reader RTA slots 0-1 (`:209-210`) | `a_buffer` / `weights_buffer` (`Buffer*`) → `:16-17` | `TensorBinding`s to `INPUT` / `WEIGHTS` |
| reader CTA slots 4-5 (`:141-142`) | positional | named CTAs `weight_stick_size`, `row_length` |
| reader RTA slots 2-6 (`:211-215`) | positional | named RTAs `tile_offset`, `face_offset`, `num_rows`, `curr_col`, `starting_index` |
| reader RTA slot 7 (`:217`) | `pad_token` — **never read**; the kernel takes its pad token from slot 6 | **dropped** (see below) |
| writer CTA slot 0 (`:169`) | `output_cb_index` (== c_0) | `DFBBinding` to `OUTPUT` |
| writer CTA slot 1 (`:169`) | `output_page_size` — never read | **dropped** |
| writer CTA slots 2.. (`:170`) | `TensorAccessorArgs(*output.buffer())` | `TensorBinding` to `OUTPUT_PARAM` |
| writer RTA slot 0 (`:224`) | `output_buffer` (`Buffer*`) | `TensorBinding` to `OUTPUT_PARAM` |
| writer RTA slots 1-3 (`:224`) | positional | named RTAs `stick_size`, `num_sticks`, `start_id` |

**How the pad-token slot mismatch is preserved.** In the ported kernel the pad token reaches
`prepare_local_cache` as a **value** rather than an argument index, so each reader passes what its own
factory actually supplies:

- `embeddings.cpp` passes `get_arg(args::pad_token)` — the real pad token, which is what it reads
  today.
- `embedding_ind_tilized.cpp` passes its `starting_index` runtime arg — the value at legacy slot 6,
  `col_offset % FACE_HEIGHT`, which is what it reads today. The kernel already reads that slot as
  `starting_index` (`:23`), so the ported kernel reads one named argument for both purposes exactly as
  the legacy kernel read slot 6 twice.

Legacy slot 7 (the real pad token) is therefore never read in either version, so it is dropped as dead
plumbing: zero functional change, and a named runtime argument no kernel reads is exactly the
carry-over the port is meant to remove. The defect and its `file:line`s are written up in the port
report so the ops team can rewire it deliberately.

### Both factories — kernel argument retrieval

Every argument in both readers and both writers is read at a constant index as a distinct field, and
every compile-time argument at a constexpr index. **No varargs anywhere in this port** — named
arguments throughout. `prepare_local_cache`'s legacy `pad_token_arg_idx` parameter looked like a
variable index but was a compile-time-constant default supplied per call site; it becomes a value
parameter.

## Applied Patterns

- **[Sync-free and single-ended CBs → self-loop DFB]** — `INDEX_SCRATCH` in both factories (a
  reserve-once index scratchpad with no consumer); `WEIGHT_CACHE` in both factories under
  PADDED / BINARY (`prepare_local_cache` reserves and writes, never `push_back`s, because nothing
  drains it); `OUTPUT` in the RM factory's height-sharded config (no writer kernel exists). Each is a
  **one-toucher** census, so the self-loop is the correct resolution and the multi-binding flag is
  not involved.
- **[Two-toucher DFB → assign 1P+1C]** — the endpoint-assignment procedure was re-derived per
  `(DFB, config)` rather than transcribed; results in the tables above. It agrees with the brief on
  every row.
- **[Conditional / optional DFB bindings]** — `WEIGHT_CACHE` is declared, bound, and referenced only
  under PADDED / BINARY. The host already emits `PADDED` / `BINARY` / `GENERIC` defines
  (`enchantum::to_string(embeddings_type)`), and the binding condition is exactly
  `PADDED || BINARY`, so the kernel gate reuses those existing defines
  (`#if defined PADDED ... #elif defined BINARY`) instead of introducing a fourth define that would
  duplicate them. The `dfb::local_cache` reference sits inside that gate, so it never enters name
  lookup on the GENERIC build.
- **[Porting a shared kernel]** — rung 2 (create the fork) twice: the shared-pool stick writer, and
  the intra-op `embeddings_common.hpp` shared with the blocked factory. Pointer comments go in both
  legacy originals; neither original changes otherwise.
- **[Pass DFB handles directly to LLKs and kernel-lib helpers]** — `dfb::local_cache` is passed
  straight into `prepare_local_cache`, which takes a `DFBBindingToken`. No `.id` extraction, no
  temporary wrapper.
- **[Removing pybound legacy factory entry points]** — **not applicable.** `embedding_nanobind.cpp`
  exposes only `ttnn::embedding` and the `EmbeddingsType` enum, so the port removes no pybind line and
  forces no device-op-class edit.
- **[Unity-build hygiene for anonymous-namespace symbols]** — all three factory `.cpp` files sit in
  one unity-build target, so every spec-name constant is declared inside its
  `create_program_artifacts` body and no new file-scope or anonymous-namespace symbol is added.

## Deferred / Flagged

- **New finding (planning):** `embeddings_common.hpp` cannot be converted in place. The brief flagged
  this as a possible assumption-violation stop; it resolves cleanly as an **intra-op shared-kernel
  fork** (Caution rung 2) inside the op's own directory, so it is not a stop. The forked header
  changes in exactly two ways relative to the original: the CB index parameter becomes a
  `DFBBindingToken` (and the `CircularBuffer` becomes a `DataflowBuffer`), and the positional pad-token
  read becomes a value parameter.
- **New finding (planning):** the Caution's "no build-system change is needed for the new file" does
  not hold for the shared pool `ttnn/cpp/ttnn/kernel/`, which is enumerated in `ttnn/sources.cmake`
  rather than globbed. One line is added there for the stick-writer fork, following the existing
  `generate_bcast_scalar_metal2.hpp` precedent. Recorded as doc friction in the port report.
- `EmbeddingsFusedProgramFactory` and its two kernels
  (`device/kernels/dataflow/embeddings_tilize.cpp`, `device/kernels/compute/tilize_chunked.cpp`) stay
  on the legacy API, untouched, pending the ops-team resolution of the Type-2 offset-base wall.
- Non-gating anomalies the audit recorded (`// Grayskull Device Setup` banners, the unused
  `api/debug/dprint.h` include in `embedding_ind_tilized.cpp`, the shared stick writer's placeholder
  CTA as it affects the *other* co-borrowers) are left alone and routed to the port report.
