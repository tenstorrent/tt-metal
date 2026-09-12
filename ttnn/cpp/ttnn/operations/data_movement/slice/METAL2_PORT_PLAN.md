# Port Plan — `data_movement/slice`

Port plan for `ttnn/cpp/ttnn/operations/data_movement/slice`, ported from the
`ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

All five program factories are in scope. They are ported together rather than one at a time,
because `ccl/mesh_partition` drives the factory variant with a single generic `std::visit` over
`create_descriptor` — a mixed variant would force that call site to branch on which factory it
holds.

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — each factory defines
  `create_descriptor(...)` returning a `tt::tt_metal::ProgramDescriptor`.
- Variants: five, in `SliceDeviceOperation::program_factory_t`
  (`device/slice_device_operation.hpp:36-41`):
  `SliceRmProgramFactory`, `SliceRmShardedProgramFactory`, `SliceRmStrideProgramFactory`,
  `SliceTileProgramFactory`, `SliceTileTensorArgsProgramFactory`.
  Each factory is its own struct with its own `.cpp`; the factory methods are **not** on the
  device-operation struct, so the direct-descriptor exception does not apply.
- Custom `compute_program_hash`: present at `device/slice_device_operation.cpp:348` — **left
  intact**. No backdoor `attribute_values` / `to_hash` anywhere in the op.
- `override_runtime_arguments`: one per factory, each a one-line delegation to the shared free
  function `ttnn::prim::patch_slice_program_addresses`
  (`device/slice_program_factory_rm_sharded.cpp:357-416`).

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's
TTNN factory analysis section. Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory)
section below.)*

---

### Variant: `SliceRmProgramFactory`

Row-major, interleaved input and output. `device/slice_program_factory_rm.cpp`.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` | `all_cores` | `TensorAccessorArgs(*src0_buffer)` only | none | slot 0 `Buffer* src0_buffer`; then `unpadded_stick_size`, `stick_size_offset`, `num_dims`, `misalignment`, `start_id`, `num_sticks_per_core`, `num_sticks_per_core_read`, `num_read_per_barrier`, `chunk_size`, `num_chunks_per_stick`, `last_chunk_size`, `src_offset_bytes`; then three `num_dims`-long blocks `num_unpadded_sticks` / `num_padded_sticks` / `id_per_dim` | none | none | unset → **O2** | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/slice_writer_unary_stick_layout_interleaved_start_id.cpp` | `all_cores` | `src0_cb_index`, then `TensorAccessorArgs(*dst_buffer)` | none | slot 0 `Buffer* dst_buffer`; then `stick_size`, `stick_size_offset`, `num_sticks_per_core`, `num_sticks_per_core_read`, `num_read_per_barrier`, `start_id`, `chunk_size`, `num_chunks_per_stick`, `last_chunk_size` | none | none | unset → **O2** | `WriterConfigDescriptor{}` |

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `src0_cb_index` = 0 | `num_read_per_barrier * 2 * cb_page_size` | `all_cores` | `datatype_to_dataformat_converter(input.dtype())` | `sizing.cb_page_size` | not set |

#### Semaphores

none — the op contains no semaphore of any kind.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `device/slice_program_factory_rm.cpp:365` (`TensorAccessorArgs(*src0_buffer)`) | `tensor_args.input` | reader RTA 0 (`Buffer*` binding, `:406`) |
| `device/slice_program_factory_rm.cpp:362` (`TensorAccessorArgs(*dst_buffer)`) | `output` | writer RTA 0 (`Buffer*` binding, `:414`) |

#### Work split

- Driver: `split_work_to_cores(sub_core_grids | compute_with_storage_grid_size, num_unpadded_sticks)`
  (`device/slice_program_factory_rm.cpp:325-328`)
- `num_cores`, `all_cores`, `core_group_1`, `core_group_2`,
  `num_sticks_per_core_group_1`, `num_sticks_per_core_group_2`.
- **No multi-`KernelDescriptor` split**: one reader descriptor and one writer descriptor over
  `all_cores`; the per-group count reaches the kernel as a per-core runtime argument
  (`num_sticks_per_core`), which is how the legacy factory already had it.

---

### Variant: `SliceRmShardedProgramFactory`

Row-major, height-sharded input and output. `device/slice_program_factory_rm_sharded.cpp`.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp` | `all_cores_unpadded` | `stick_size_unpadded`, `num_sticks_unpadded`, `src_stride_bytes`, `dst_stride_bytes`, `begins_bytes` | none | slot 0 `num_cores_read`; then a per-core-variable block: `(noc_x, noc_y)` per source core, `num_stick_chunks` per source core, `(chunk_start_id, chunk_num_sticks)` per chunk | none | none | unset → **O2** | `ReaderConfigDescriptor{}` |

This factory builds **exactly one** kernel (`desc.kernels` receives only `reader_desc`,
`:348`), so neither CB can have a second toucher.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `src0_cb_index` = 0, `.buffer = input.buffer()` | `shard_height_padded * src_stride_bytes` | `all_cores_unpadded` | from `input.dtype()` | `src_stride_bytes` | not set |
| `c_16`, `.buffer = output.buffer()` | `shard_height_unpadded * dst_stride_bytes` | `all_cores_unpadded` | from `output.dtype()` | `dst_stride_bytes` | not set |

Both are borrowed-memory CBs (legacy `CBDescriptor::buffer` set), at offset zero.

#### Semaphores

none.

#### Tensor accessors

None. Both tensors reach the kernel as borrowed-memory CBs; the kernel addresses them through
`dfb_in.get_write_ptr()` / `dfb_out.get_write_ptr()` and NoC unicast to the source core's
coordinates, never through a `TensorAccessor`.

#### Work split

- Not `split_work_to_cores`: the split follows the output tensor's shard grid.
  `num_cores_unpadded = shard_spec_unpadded.num_cores()`; argument list `i` goes to
  `output_cores[i]` (`:343-346`).

---

### Variant: `SliceRmStrideProgramFactory`

Row-major with a non-unit step. `device/slice_program_factory_rm_stride.cpp`.
**Runtime kernel-source selection** on one axis: tensor rank ≤ 4 selects the `*_4d` pair,
rank > 4 the `*_nd` pair (`:44-54`). Both pairs convert in this change.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader (rank ≤ 4) | `device/kernels/dataflow/reader_multicore_slice_4d.cpp` | `all_cores` | `in_cb`, `element_size`, `TensorAccessorArgs(*input_buffer)` | none | slot 0 `Buffer* input_buffer`; then 24 scalars (`tensor_rank`, four input dims, four output dims, twelve slice start/end/step values, `element_size`, `rows_for_this_core`, `row_start_id`) | none | none | unset → **O2** | `ReaderConfigDescriptor{}` |
| writer (rank ≤ 4) | `device/kernels/dataflow/writer_multicore_slice_4d.cpp` | `all_cores` | `in_cb`, `element_size`, `TensorAccessorArgs(*output_buffer)` | none | slot 0 `Buffer* output_buffer`; then `tensor_rank`, four output dims, `element_size`, `rows_for_this_core`, `row_start_id` | none | none | unset → **O2** | `WriterConfigDescriptor{}` |
| reader (rank > 4) | `device/kernels/dataflow/reader_multicore_slice_nd.cpp` | `all_cores` | `in_cb`, `element_size`, `TensorAccessorArgs(*input_buffer)` | none | slot 0 `Buffer* input_buffer`; then `tensor_rank`, `element_size`, `rows_for_this_core`, `row_start_id`; then five `tensor_rank`-long blocks (`input_dims`, `output_dims`, `slice_starts`, `slice_ends`, `slice_steps`) | none | none | unset → **O2** | `ReaderConfigDescriptor{}` |
| writer (rank > 4) | `device/kernels/dataflow/writer_multicore_slice_nd.cpp` | `all_cores` | `in_cb`, `element_size`, `TensorAccessorArgs(*output_buffer)` | none | slot 0 `Buffer* output_buffer`; then `tensor_rank`, `element_size`, `rows_for_this_core`, `row_start_id`; then one `tensor_rank`-long `output_dims` block | none | none | unset → **O2** | `WriterConfigDescriptor{}` |

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `in_cb` = 0 | `2 * cb_page_size_aligned` | `all_cores` | from `input_tensor.dtype()` | `round_up(input_shape[-1] * element_size, max(src_alignment, dst_alignment))` | not set |

#### Semaphores

none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `device/slice_program_factory_rm_stride.cpp:80` | `tensor_args.input` | reader RTA 0 (`Buffer*`, `:128` / `:147`) |
| `device/slice_program_factory_rm_stride.cpp:83` | `output` | writer RTA 0 (`Buffer*`, `:136` / `:160`) |

#### Work split

- Driver: `split_work_to_cores(..., total_output_rows)` (`:36-39`), but only `num_cores` and
  `all_cores` are used. The per-core row counts are recomputed by hand at `:101-102` and
  `:120-124` (`base_rows_per_core` plus one extra row for the first `extra_rows` cores), **not**
  taken from `core_group_1` / `core_group_2`. That hand-rolled split is preserved verbatim.

---

### Variant: `SliceTileProgramFactory`

Tile layout, interleaved. `device/slice_program_factory_tile.cpp`.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id.cpp` | `all_cores` | `num_dims`, then `TensorAccessorArgs(*src0_buffer)` | `dfb_id_in` = `src0_cb_index` | `start_id`, `num_tiles`, then a `num_dims`-long `id_per_dim` block | slot 0 `Buffer* src0_buffer`; then two `num_dims`-long blocks `num_unpadded_tiles` / `num_padded_tiles` | none | unset → **O2** | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (**slice-owned copy**) | `all_cores` | `TensorAccessorArgs(*dst_buffer)` only | `dfb_id_out` = `src0_cb_index` | slot 0 `Buffer* dst_buffer` (or literal `0u` on a no-op core, `:176`); then `num_tiles`, `start_id` | none | none | unset → **O2** | `WriterConfigDescriptor{}` |

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `src0_cb_index` = 0 | `2 * single_tile_size` | `all_cores` | from `input.dtype()` | `tt::tile_size(cb_data_format)` | not set |

#### Semaphores

none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `device/slice_program_factory_tile.cpp:65` | `tensor_args.input` | reader **CRTA** 0 (`Buffer*`, `:143`) |
| `device/slice_program_factory_tile.cpp:152` | `output` | writer RTA 0 (`Buffer*`, `:180`) |

#### Work split

- Driver: `split_work_to_cores(..., num_unpadded_tiles)` (`:31-34`)
- No-op cores are given an all-zero argument list rather than being left out.

---

### Variant: `SliceTileTensorArgsProgramFactory`

Tile layout, interleaved, with the slice bounds supplied as device tensors.
`device/slice_program_factory_tile_tensor_args.cpp`.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp` | `all_cores` | `src0_cb_index`, `tensor_cb_index`, `num_dims`, `tile_width`, `tile_height`, then `TensorAccessorArgs` for src / start / end | none | `start_id`, `num_tiles`, then a `num_dims`-long `id_per_dim` block | slots 0-2 `Buffer* src_buffer` / `start_buffer` / `end_buffer`; then three `num_dims`-long blocks `num_unpadded_tiles` / `num_padded_tiles` / `input_shape_args` | none | unset → **O2** | `ReaderConfigDescriptor{}` |
| writer | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (**borrowed**) | `all_cores` | `src0_cb_index`, then `TensorAccessorArgs(*dst_buffer)` | none | slot 0 `Buffer* dst_buffer` (on **every** core, no-op ones included, `:151`); then `num_tiles`, `start_id` | none | none | unset → **O2** | `WriterConfigDescriptor{}` |

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `src0_cb_index` = 0 | `2 * single_tile_size` | `all_cores` | from `input_tensor.dtype()` | `tt::tile_size(cb_data_format)` | not set |
| `tensor_cb_index` = 1 | `single_tile_size` | `all_cores` | from `input_tensor.dtype()` | `tt::tile_size(cb_data_format)` | not set |

#### Semaphores

none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `device/slice_program_factory_tile_tensor_args.cpp:82` | `tensor_args.input` | reader CRTA 0 (`Buffer*`, `:182`) |
| `device/slice_program_factory_tile_tensor_args.cpp:83` | `tensor_args.start_tensor` | reader CRTA 1 (`Buffer*`, `:183`) |
| `device/slice_program_factory_tile_tensor_args.cpp:84` | `tensor_args.end_tensor` | reader CRTA 2 (`Buffer*`, `:184`) |
| `device/slice_program_factory_tile_tensor_args.cpp:87` | `output` | writer RTA 0 (`Buffer*`, `:151` / `:168`) |

#### Work split

- Driver: `split_work_to_cores(..., num_unpadded_tiles)` (`:34-37`), same shape as
  `SliceTileProgramFactory`.

---

### Shared kernels

| kernel source | kind | `_metal2` fork beside it? | rung |
|---|---|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | **borrowed** — bound by `SliceTileTensorArgsProgramFactory` (`device/slice_program_factory_tile_tensor_args.cpp:133`) and by thirteen other ops' factories | **yes** — `writer_unary_interleaved_start_id_metal2.cpp`, same directory | **rung 1 — reuse the existing fork.** No new file, no edit to the legacy original. |

**The fork's binding vocabulary is the constraint on this port's `KernelSpec`**, not a free
choice (`writer_unary_interleaved_start_id_metal2.cpp:27-63`):

- DFB accessor name: `out` (kernel constructs `DataflowBuffer dfb(dfb::out)`), CONSUMER.
- Tensor accessor name: `dst` (kernel constructs `TensorAccessor(tensor::dst)`).
- Named runtime args: `num_pages`, `start_id`.
- Preprocessor gates it reads: `OUT_SHARDED`, `BACKWARDS`. Slice sets neither, matching the
  legacy factory, which passed no defines.

Note the **same-basename trap**: `SliceTileProgramFactory` binds slice's *own*
`device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`, a different file that reads its
DFB index through `get_named_compile_time_arg_val("dfb_id_out")`. That copy is slice-owned and
bound by no other op, so it is converted in place. Keying on basename alone would merge the two.

No kernel in this op is *lent* to another op, and no kernel is shared between two of slice's own
factories.

### Flags

- **Two unreferenced kernel files in the op's directory**, instantiated by no factory in the
  repository: `device/kernels/dataflow/strided_slice_reader_rm_interleaved_nd.cpp` and
  `device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp`. Not audited, not ported,
  left exactly as they are. Recorded so the report makes clear what was not covered.
- **A host-side consumer outside the op directory.**
  `ttnn/cpp/ttnn/operations/ccl/mesh_partition/device/mesh_partition_program_factory.cpp` drives
  slice's factories directly rather than going through `ttnn::prim::slice`: it calls
  `SliceOp::validate_on_program_cache_miss` and `SliceOp::select_program_factory` at `:126-127`,
  `Factory::create_descriptor` at `:131`, and `ttnn::prim::patch_slice_program_addresses` at
  `:155`, and it stores a `SliceDeviceOperation::program_factory_t` in its own
  `shared_variables_t` (`mesh_partition_device_operation.hpp:46-50`). Every one of those entry
  points changes in this port. The invoker confirmed MeshPartition is re-wired in the same
  change; see [Deferred / Flagged](#deferred--flagged).
- **A pybound factory entry point.** `slice_nanobind.cpp:167-179` exposes
  `SliceTileProgramFactory.create_descriptor` to Python. Still present on `origin/main` as of
  this port (checked with `git show origin/main:…`), so no removal is in flight and the port
  deletes it. The neighbouring bindings at `:138-166` are not factory entry points and stay.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `CustomProgramSpecFactoryConcept`, on all five factories —
  every one declares an `override_runtime_arguments`, which the port **translates** into a method
  returning a `ProgramRunArgs` rather than deleting.
- **Custom `compute_program_hash`**: present at `device/slice_device_operation.cpp:348` — leave
  intact.
- **Implementation notes**: the five `override_runtime_arguments` methods all delegate to one
  shared free function, `patch_slice_program_addresses`, which `ccl/mesh_partition` also calls.
  The port keeps that single-home shape: the free function is re-expressed as
  `slice_program_run_args(factory, args, tensor_args, output)` returning a `ProgramRunArgs`, each
  factory's `override_runtime_arguments` returns its result, and MeshPartition feeds the same
  result to `UpdateProgramRunArgs`.

## Planned Spec Shape

### Variant: `SliceRmProgramFactory`

- **KernelSpecs**: `rm_reader`, `rm_writer` — one each, 1:1 with the legacy descriptors.
- **DataflowBufferSpecs**: `rm_in` — `entry_size = sizing.dfb_entry_size`,
  `num_entries = sizing.num_read_per_barrier * 2`, `data_format_metadata = dfb_data_format`.
  Reader PRODUCER (`in`), writer CONSUMER (`in`). Plain 1:1.
- **ScratchpadSpecs**: `rm_id_per_dim` — `size_per_node = num_dims * sizeof(uint32_t)`, bound to
  the reader. See [Applied Patterns](#applied-patterns).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `input`, `output`.
- **WorkUnitSpecs**: one, `{rm_reader, rm_writer}` over `all_cores`.
- **Op-owned tensors**: none.

### Variant: `SliceRmShardedProgramFactory`

- **KernelSpecs**: `sharded_reader` — one, 1:1.
- **DataflowBufferSpecs**: `sharded_in` (`borrowed_from = input`,
  `entry_size = src_stride_bytes`, `num_entries = shard_height_padded`) and `sharded_out`
  (`borrowed_from = output`, `entry_size = dst_stride_bytes`,
  `num_entries = shard_height_unpadded`). Both **self-looped** on the reader — it is the only
  toucher of either, and this factory builds only one kernel.
- **ScratchpadSpecs**: none.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `input`, `output` — declared for the borrows. Neither carries a
  `TensorBinding`; a parameter named by a `borrowed_from` counts as used.
- **WorkUnitSpecs**: one, `{sharded_reader}` over `all_cores_unpadded`.
- **Op-owned tensors**: none.

### Variant: `SliceRmStrideProgramFactory`

- **KernelSpecs**: `stride_reader`, `stride_writer` — one each. The **source path** is selected
  at construction from the tensor rank, exactly as legacy; the spec shape is identical on both
  paths, so this is one pair of specs with a computed `source`, not two pairs.
- **DataflowBufferSpecs**: `stride_in` — `entry_size = dfb_entry_size_aligned`, `num_entries = 2`,
  `data_format_metadata = dfb_data_format`. Reader PRODUCER, writer CONSUMER. Plain 1:1.
- **ScratchpadSpecs**: none — neither `*_nd` kernel writes to its vararg blocks, and the `*_4d`
  pair has no varargs at all.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `input`, `output`.
- **WorkUnitSpecs**: one, `{stride_reader, stride_writer}` over `all_cores`.
- **Op-owned tensors**: none.

### Variant: `SliceTileProgramFactory`

- **KernelSpecs**: `tile_reader`, `tile_writer` — one each.
- **DataflowBufferSpecs**: `tile_in` — `entry_size = single_tile_size`, `num_entries = 2`,
  `data_format_metadata = dfb_data_format`. Reader PRODUCER (`in`), writer CONSUMER (`out`).
  Plain 1:1. The two accessor names differ because the legacy kernels read the same CB index
  through two differently-named compile-time args (`dfb_id_in`, `dfb_id_out`).
- **ScratchpadSpecs**: `tile_id_per_dim` — `size_per_node = num_dims * sizeof(uint32_t)`, bound
  to the reader.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `input`, `output`.
- **WorkUnitSpecs**: one, `{tile_reader, tile_writer}` over `all_cores`.
- **Op-owned tensors**: none.

### Variant: `SliceTileTensorArgsProgramFactory`

- **KernelSpecs**: `ta_reader`, `ta_writer` — one each. `ta_writer` binds the **existing
  `_metal2` fork** of the borrowed eltwise/unary writer and inherits its names.
- **DataflowBufferSpecs**: `ta_in` (`entry_size = single_tile_size`, `num_entries = 2`) and
  `ta_tensor` (`entry_size = single_tile_size`, `num_entries = 1`), both
  `data_format_metadata = dfb_data_format`.
  - `ta_in`: reader PRODUCER (`in`), writer CONSUMER (`out` — the fork's accessor name). Plain 1:1.
  - `ta_tensor`: **self-looped** on the reader, which runs the full
    `reserve_back` → `push_back` → `wait_front` → `pop_front` handshake against itself, twice
    (`device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:52-83`).
    One toucher.
- **ScratchpadSpecs**: `ta_id_per_dim` — `size_per_node = num_dims * sizeof(uint32_t)`, bound to
  the reader.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `input`, `start_tensor`, `end_tensor`, `output`.
- **WorkUnitSpecs**: one, `{ta_reader, ta_writer}` over `all_cores`.
- **Op-owned tensors**: none.

## Preserved Multiplicity

none — no work-split multiplicity in legacy. Every factory builds exactly one `KernelDescriptor`
per role over a single `core_ranges`; the per-group counts travel as per-core runtime arguments,
which is how the legacy factories already carried them. No per-group compile-time argument is
demoted, because none exists.

`SliceRmStrideProgramFactory`'s rank-driven choice between the `*_4d` and `*_nd` kernel pairs is
runtime **source selection**, not multiplicity: one pair of `KernelSpec`s is built per program,
with the source path computed the same way legacy computed it.

## Dropped Plumbing

### `SliceRmProgramFactory`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `slice_program_factory_rm.cpp:406` | reader RTA 0 = `Buffer* src0_buffer` | `TensorBinding{input, "src"}` |
| `slice_program_factory_rm.cpp:414` | writer RTA 0 = `Buffer* dst_buffer` | `TensorBinding{output, "dst"}` |
| `slice_program_factory_rm.cpp:365` | `TensorAccessorArgs(*src0_buffer).append_to(reader_cta)` | binding mechanism; reader has no CTAs at all |
| `slice_program_factory_rm.cpp:362` | `TensorAccessorArgs(*dst_buffer).append_to(writer_cta)` | binding mechanism |
| `slice_program_factory_rm.cpp:361` | writer CTA 0 = `src0_cb_index` (magic CB index) | `DFBBinding{rm_in, "out", CONSUMER}` |
| kernel `slice_reader_…_rm_interleaved_start_id.cpp:35` | `TensorAccessorArgs<0>()` | `TensorAccessor(tensor::src)` |
| kernel `slice_reader_…_rm_interleaved_start_id.cpp:42` | `constexpr uint32_t dfb_id_in0 = 0` (hardcoded index) | `DataflowBuffer dfb_in0(dfb::in)` |
| kernel `slice_writer_…_start_id.cpp:26-27` | `get_compile_time_arg_val(0)`, `TensorAccessorArgs<1>()` | `dfb::out`, `TensorAccessor(tensor::dst)` |
| both kernels, all `get_arg_val<uint32_t>(N)` | positional RTAs | named `get_arg(args::<name>)` |

### `SliceRmShardedProgramFactory`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `slice_program_factory_rm_sharded.cpp:299` | `CBDescriptor::buffer = input.buffer()` | `DataflowBufferSpec::borrowed_from = input` |
| `slice_program_factory_rm_sharded.cpp:311` | `CBDescriptor::buffer = output.buffer()` | `DataflowBufferSpec::borrowed_from = output` |
| `slice_program_factory_rm_sharded.cpp:314-319` | five positional CTAs | five named CTAs |
| kernel `slice_reader_…_rm_sharded.cpp:32-33` | `constexpr auto dfb_in0 = tt::CBIndex::c_0` / `c_16` | `dfb::in` / `dfb::out` |
| kernel `slice_reader_…_rm_sharded.cpp:14-19` | `get_compile_time_arg_val(0..4)` | `get_arg(args::<name>)` |
| kernel `slice_reader_…_rm_sharded.cpp:25` | `get_arg_val<uint32_t>(0)` | `get_arg(args::num_cores_read)` |

There is no buffer-address RTA on this factory — the two addresses reach the kernel through the
borrowed DFBs, which is already a binding.

### `SliceRmStrideProgramFactory`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `slice_program_factory_rm_stride.cpp:128`, `:147` | reader RTA 0 = `Buffer* input_buffer` | `TensorBinding{input, "src"}` |
| `slice_program_factory_rm_stride.cpp:136`, `:160` | writer RTA 0 = `Buffer* output_buffer` | `TensorBinding{output, "dst"}` |
| `slice_program_factory_rm_stride.cpp:80`, `:83` | `TensorAccessorArgs(...).append_to(...)` | binding mechanism |
| `slice_program_factory_rm_stride.cpp:79`, `:82` | CTA 0 = `in_cb` (magic CB index) | `DFBBinding{stride_in, …}` |
| `slice_program_factory_rm_stride.cpp:79`, `:82` | CTA 1 = `element_size` | named CTA `element_size` — **kept**, though all four kernels declare it and none reads it (see [Deferred / Flagged](#deferred--flagged)) |
| all four kernels, `get_arg_val<uint32_t>(rt_args_idx++)` | positional RTAs | named `get_arg(args::<name>)` |

### `SliceTileProgramFactory`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `slice_program_factory_tile.cpp:143` | reader CRTA 0 = `Buffer* src0_buffer` | `TensorBinding{input, "src"}` |
| `slice_program_factory_tile.cpp:180` | writer RTA 0 = `Buffer* dst_buffer` (and the literal `0u` at `:176`) | `TensorBinding{output, "dst"}` — one uniform binding per kernel, so the active / no-op-core distinction disappears |
| `slice_program_factory_tile.cpp:65`, `:152` | `TensorAccessorArgs(...).append_to(...)` | binding mechanism |
| `slice_program_factory_tile.cpp:139` | `named_compile_time_args = {{"dfb_id_in", src0_cb_index}}` | `DFBBinding{tile_in, "in", PRODUCER}` — a CB index carried by a *named* CTA still becomes a DFB binding, never a named argument |
| `slice_program_factory_tile.cpp:161` | `named_compile_time_args = {{"dfb_id_out", src0_cb_index}}` | `DFBBinding{tile_in, "out", CONSUMER}` |
| `slice_program_factory_tile.cpp:64` | CTA 0 = `num_dims` | named CTA `num_dims` |

### `SliceTileTensorArgsProgramFactory`

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `slice_program_factory_tile_tensor_args.cpp:182-184` | reader CRTA 0-2 = `Buffer*` src / start / end | `TensorBinding{input, "src"}`, `{start_tensor, "start"}`, `{end_tensor, "end"}` |
| `slice_program_factory_tile_tensor_args.cpp:151`, `:168` | writer RTA 0 = `Buffer* dst_buffer` | `TensorBinding{output, "dst"}` — the fork's accessor name |
| `slice_program_factory_tile_tensor_args.cpp:82-84`, `:87` | four `TensorAccessorArgs(...).append_to(...)` sites | binding mechanism |
| `slice_program_factory_tile_tensor_args.cpp:80-81` | CTA 0-1 = `src0_cb_index`, `tensor_cb_index` | `DFBBinding{ta_in, "in", PRODUCER}`, `DFBBinding{ta_tensor, "tensor", PRODUCER+CONSUMER}` |
| `slice_program_factory_tile_tensor_args.cpp:86` | writer CTA 0 = `src0_cb_index` | `DFBBinding{ta_in, "out", CONSUMER}` |
| `slice_program_factory_tile_tensor_args.cpp:80-81` | CTA 2-4 = `num_dims`, `tile_width`, `tile_height` | named CTAs |

**No page-size third-argument CTA/RTA anywhere in the op** — every `TensorAccessor` in the op
and in the borrowed donor uses the two-argument `(args, addr)` form, so nothing is dropped under
that heading.

## Applied Patterns

- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):
  three DFBs, each with a one-kernel touch census re-derived from the kernel bodies rather than
  transcribed from the brief —
  - `sharded_in` — the reader's only touch is `dfb_in.get_write_ptr()`
    (`slice_reader_unary_unpad_dims_rm_sharded.cpp:41`), a role-free raw peek. One toucher.
  - `sharded_out` — the reader `reserve_back`s, `push_back`s and peeks
    (`…:40,42,89`); a locked producer, but still the only toucher, since the factory builds one
    kernel.
  - `ta_tensor` — the reader runs the full four-call handshake against itself, twice
    (`reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:52-83`). One toucher.

  All three are self-loops, not multi-binding: none has a second toucher, let alone three, and no
  two kernels are locked to the same FIFO role. `allow_instance_multi_binding` is set nowhere in
  this port.

- **Borrowed-memory DFBs** ([migration guide — DataflowBufferSpec](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/migration_guide.md#dataflowbufferspec)):
  `sharded_in` / `sharded_out` set `borrowed_from` to the `input` / `output` `TensorParameter`.
  The backing L1 address resolves from the corresponding `TensorArgument` each dispatch, which is
  what makes the legacy CB-address patch in `patch_slice_program_addresses:365-371` disappear.

- [Caution: Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel),
  **rung 1**: `SliceTileTensorArgsProgramFactory`'s writer binds the existing
  `writer_unary_interleaved_start_id_metal2.cpp` beside the eltwise/unary original. No new file,
  no edit to the original, and slice's `KernelSpec` is built against the fork's names.

- [Caution: Avoid varargs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary):
  six kernels keep a vararg block, each a genuine variable-count collection; every scalar around
  those blocks is named. The two `*_4d` stride kernels look loop-like because of their running
  `rt_args_idx++`, but each read is a distinct field read once — all named, per the recipe's
  explicit non-signal. Retained vararg sites are listed in the port report.

- **`ScratchpadSpec` for kernel-mutable per-dimension scratch** — no catalog entry exists for
  this yet; see [Deferred / Flagged](#deferred--flagged).

## Deferred / Flagged

- **Metal 2.0 has no writable or addressable runtime-vararg accessor, and three slice readers
  need one.** Those kernels take a raw pointer into their runtime-argument buffer and mutate the
  values in place as a per-dimension odometer:

  ```cpp
  tt_l1_ptr uint32_t* id_per_dim = (tt_l1_ptr uint32_t*)(get_arg_addr(2));
  ...
  id_per_dim[j]++;
  if (id_per_dim[j] == num_unpadded_tiles[j]) { id_per_dim[j] = 0; src_tile_id += num_padded_tiles[j]; }
  ```

  `get_vararg(i)` returns a value; there is no address form and no write form
  (`tt_metal/jit_build/genfiles.cpp:535` emits only `get_vararg` / `get_common_vararg`). The
  read-only blocks in the same kernels convert to plain `get_vararg(i)` reads; this one has no
  mechanical translation. Affected: the readers of `SliceRmProgramFactory`,
  `SliceTileProgramFactory` and `SliceTileTensorArgsProgramFactory`.

  **Resolution, agreed with the invoker before any code was written**: declare a
  `ScratchpadSpec` per affected kernel — Metal 2.0's own primitive for "a private, uninitialized
  region of node-local SRAM for a kernel to use as working memory"
  (`scratchpad_spec.hpp:18-21`) — size it from the rank on the host, bind it to the reader, and
  have the kernel seed it from the vararg values at entry and mutate it there. Values,
  arithmetic and results are unchanged, and the scratch stays in L1 exactly as it was. This is
  the reason `id_per_dim` is a vararg *input* in the ported kernels rather than a mutable
  buffer. Reported as friction, since the recipe's kernel-side whitelist has no entry for it.

- **`ccl/mesh_partition` is re-wired in the same change**, on the invoker's instruction. It is
  outside the op directory, and it is a `MeshWorkloadFactoryConcept` op (per-coordinate
  `create_at`), which the port recipe does not cover — so MeshPartition is **not** ported. Its
  `create_at` is adapted to call the new `create_program_artifacts`, build the `Program` with
  `MakeProgramFromSpec` and apply `SetProgramRunArgs`; its `override_runtime_arguments` calls
  `UpdateProgramRunArgs` with the shared helper's result. Recorded as a handoff point.

- **Per-node vararg counts on the sharded reader.** Its argument-list length genuinely differs
  per core (it depends on how many input shards that output shard draws from, and how those rows
  coalesce into chunks), so the uniform `num_runtime_varargs` does not fit. The API's mechanism
  for this is `KernelAdvancedOptions::num_runtime_varargs_per_node`, which is marked
  `[[deprecated]]` ("truly bizarre… will be removed once existing uses are refactored to avoid
  it"). Using it reproduces the legacy per-core layout exactly; the alternative — padding every
  core to the maximum — would change the dispatch footprint. The deprecation warning is
  suppressed tree-wide by `-Wno-deprecated-declarations` (`CMakeLists.txt:211`), so this
  compiles clean. Reported.

- **Dead arguments carried forward unchanged.** The audit recorded these and the port does not
  act on them: `compile_time_element_size` is declared from CTA 1 and never read in all four
  stride kernels; `reader_multicore_slice_4d.cpp:60-62` reads `output_h` / `output_d` /
  `output_n` and never uses them; `writer_multicore_slice_4d.cpp:54-58` reads `tensor_rank` and
  three output dims it never uses. The ported kernels keep every one of these as a named
  argument the host still emits, because removing them would be a functional change. Listed in
  the port report as findings.

- **Two unreferenced kernel files** in the op directory are left untouched (see
  [Flags](#flags)).
