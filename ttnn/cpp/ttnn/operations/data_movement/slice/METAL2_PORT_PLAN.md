# Port Plan — `data_movement/slice`

Port plan for `slice`, ported from `ProgramDescriptor` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**All five factories of `ttnn::prim::SliceDeviceOperation` are ported in this change.** They are not
independent: `ccl/mesh_partition` reaches every alternative of `program_factory_t` through one
`std::visit`, so removing `create_descriptor` from any one of them breaks its build. Converting all
five keeps that consumer on a single uniform path (see [Flags](#flags)).

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` on all five factories.
- Variants: five, in `program_factory_t` on `SliceDeviceOperation`
  ([device/slice_device_operation.hpp:36-41](device/slice_device_operation.hpp#L36-L41)) — a conventional
  `program_factory_t`, so exception 3 (direct-descriptor) does not apply.
  - `SliceRmProgramFactory`
  - `SliceRmShardedProgramFactory`
  - `SliceRmStrideProgramFactory` (runtime kernel-source selection: rank ≤ 4 vs rank > 4)
  - `SliceTileProgramFactory`
  - `SliceTileTensorArgsProgramFactory`
- Custom `compute_program_hash`: present at
  [device/slice_device_operation.cpp:348-432](device/slice_device_operation.cpp#L348-L432) — **left intact**.
- `override_runtime_arguments`: present on all five, each a one-line delegation to
  `patch_slice_program_addresses`, whose implementation sat in
  `slice_program_factory_rm_sharded.cpp` and is deleted by this port.

### Kernels

`unique_id` values below are the `KernelSpecName`s the port assigns. No factory sets `opt_level`, and
every kernel is data movement, so every resolved legacy level is `O2` — which is also Metal 2.0's
default, so no `compiler_options.opt_level` line is needed anywhere in this op. There is no compute
kernel, so no `ComputeHardwareConfig`, no `unpack_modes`, and no `bfp_pack_precision_mode`.

#### Variant: `SliceRmProgramFactory`

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| `reader` | `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` | `all_cores` | `TensorAccessorArgs(input)` only | none | slot 0 `Buffer* input`; 1 `unpadded_stick_size`; 2 `stick_size_offset`; 3 `num_dims`; 4 `misalignment`; 5 `start_id`; 6 `num_sticks_per_core`; 7 `num_sticks_per_core_read`; 8 `num_read_per_barrier`; 9 `chunk_size`; 10 `num_chunks_per_stick`; 11 `last_chunk_size`; 12 `src_offset_bytes`; 13.. three `num_dims`-long blocks | none | none | O2 | `ReaderConfigDescriptor{}` |
| `writer` | `slice_writer_unary_stick_layout_interleaved_start_id.cpp` | `all_cores` | slot 0 `src0_cb_index`; then `TensorAccessorArgs(output)` | none | slot 0 `Buffer* output`; 1 `stick_size`; 2 `stick_size_offset`; 3 `num_sticks_per_core`; 4 `num_sticks_per_core_read`; 5 `num_read_per_barrier`; 6 `start_id`; 7 `chunk_size`; 8 `num_chunks_per_stick`; 9 `last_chunk_size` | none | none | O2 | `WriterConfigDescriptor{}` |

#### Variant: `SliceRmShardedProgramFactory`

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| `reader` | `slice_reader_unary_unpad_dims_rm_sharded.cpp` | `all_cores_unpadded` | 0 `stick_size_unpadded`; 1 `num_sticks_unpadded`; 2 `src_stride_bytes`; 3 `dst_stride_bytes`; 4 `begins_bytes` | none | slot 0 `num_cores_read`; 1.. one variable-length stream (noc x/y pairs, per-core chunk counts, chunk (start,len) pairs) | none | none | O2 | `ReaderConfigDescriptor{}` |

Single-kernel factory: no writer.

#### Variant: `SliceRmStrideProgramFactory`, rank ≤ 4

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| `reader` | `reader_multicore_slice_4d.cpp` | `all_cores` | 0 `in_cb`; 1 `element_size`; then `TensorAccessorArgs(input)` | none | a **fixed** run of 25: slot 0 `Buffer* input`, then `tensor_rank`, `input_w/h/d/n`, `output_w/h/d/n`, `slice_{start,end,step}_{w,h,d,n}`, `element_size`, `num_rows_for_this_core`, `start_row_for_this_core` | none | none | O2 | `ReaderConfigDescriptor{}` |
| `writer` | `writer_multicore_slice_4d.cpp` | `all_cores` | 0 `in_cb`; 1 `element_size`; then `TensorAccessorArgs(output)` | none | a **fixed** run of 9: slot 0 `Buffer* output`, `tensor_rank`, `output_w/h/d/n`, `element_size`, `num_rows_for_this_core`, `start_row_for_this_core` | none | none | O2 | `WriterConfigDescriptor{}` |

#### Variant: `SliceRmStrideProgramFactory`, rank > 4

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| `reader` | `reader_multicore_slice_nd.cpp` | `all_cores` | 0 `in_cb`; 1 `element_size`; then `TensorAccessorArgs(input)` | none | slot 0 `Buffer* input`, `tensor_rank`, `element_size`, `num_rows_for_this_core`, `start_row_for_this_core`, then five `tensor_rank`-long blocks (`input_dims`, `output_dims`, `slice_starts`, `slice_ends`, `slice_steps`) | none | none | O2 | `ReaderConfigDescriptor{}` |
| `writer` | `writer_multicore_slice_nd.cpp` | `all_cores` | 0 `in_cb`; 1 `element_size`; then `TensorAccessorArgs(output)` | none | slot 0 `Buffer* output`, `tensor_rank`, `element_size`, `num_rows_for_this_core`, `start_row_for_this_core`, then one `tensor_rank`-long block (`output_dims`) | none | none | O2 | `WriterConfigDescriptor{}` |

#### Variant: `SliceTileProgramFactory`

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| `reader` | `reader_unary_unpad_dims_interleaved_start_id.cpp` | `all_cores` | 0 `num_dims`; then `TensorAccessorArgs(input)` | `dfb_id_in` = `src0_cb_index` | slot 0 `start_id`; 1 `num_tiles`; 2.. `id_per_dim[num_dims]` | slot 0 `Buffer* input`; 1.. two `num_dims`-long blocks (`num_unpadded_tiles`, `num_padded_tiles`) | none | O2 | `ReaderConfigDescriptor{}` |
| `writer` | `writer_unary_interleaved_start_id.cpp` (slice-owned copy) | `all_cores` | `TensorAccessorArgs(output)` only | `dfb_id_out` = `src0_cb_index` | slot 0 `Buffer* output`; 1 `num_pages`; 2 `start_id` | none | none | O2 | `WriterConfigDescriptor{}` |

#### Variant: `SliceTileTensorArgsProgramFactory`

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| `reader` | `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp` | `all_cores` | 0 `src0_cb_index`; 1 `tensor_cb_index`; 2 `num_dims`; 3 `tile_width`; 4 `tile_height`; then `TensorAccessorArgs` for input, start, end | none | slot 0 `start_id`; 1 `num_tiles`; 2.. `id_per_dim[num_dims]` | slot 0 `Buffer* input`; 1 `Buffer* start`; 2 `Buffer* end`; 3.. three `num_dims`-long blocks (`num_unpadded_tiles`, `num_padded_tiles`, `input_shape`) | none | O2 | `ReaderConfigDescriptor{}` |
| `writer` | **borrowed** `../../eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `all_cores` | 0 `src0_cb_index`; then `TensorAccessorArgs(output)` | none | slot 0 `Buffer* output`; 1 `num_pages`; 2 `start_id` | none | none | O2 | `WriterConfigDescriptor{}` |

Descriptor push order is reader-then-writer in all four two-kernel factories, even where
`SliceTileTensorArgsProgramFactory` *builds* the writer first. Named bindings retire that ordering
dependence, which is what let `patch_slice_program_addresses`'s hardcoded kernel indices go away.

### CBs

| Factory | index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|---|
| `SliceRmProgramFactory` | 0 | `num_read_per_barrier * 2 * cb_page_size` | `all_cores` | `input.dtype()` | `sizing.cb_page_size` | not set |
| `SliceRmShardedProgramFactory` | 0 | `shard_height_padded * src_stride_bytes` | `all_cores_unpadded` | `input.dtype()` | `src_stride_bytes` | not set; **`.buffer = input.buffer()`** |
| `SliceRmShardedProgramFactory` | `c_16` | `shard_height_unpadded * dst_stride_bytes` | `all_cores_unpadded` | `output.dtype()` | `dst_stride_bytes` | not set; **`.buffer = output.buffer()`** |
| `SliceRmStrideProgramFactory` | 0 | `2 * cb_page_size_aligned` | `all_cores` | `input.dtype()` | `cb_page_size_aligned` | not set |
| `SliceTileProgramFactory` | 0 | `2 * single_tile_size` | `all_cores` | `input.dtype()` | `single_tile_size` | not set |
| `SliceTileTensorArgsProgramFactory` | 0 | `2 * single_tile_size` | `all_cores` | `input.dtype()` | `single_tile_size` | not set |
| `SliceTileTensorArgsProgramFactory` | 1 | `single_tile_size` | `all_cores` | `input.dtype()` | `single_tile_size` | not set |

No `GlobalCircularBuffer`. No `address_offset`. No `format_descriptors[i].tile` is ever set, so every
`tile_format_metadata` stays `nullopt`.

### Semaphores

none — the op uses no semaphore of any kind.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| [rm:365](device/slice_program_factory_rm.cpp#L365) | `input` | reader slot 0 |
| [rm:362](device/slice_program_factory_rm.cpp#L362) | `output` | writer slot 0 |
| [rm_sharded:290](device/slice_program_factory_rm_sharded.cpp#L290) | `input` | none — borrowed-memory buffer |
| [rm_sharded:302](device/slice_program_factory_rm_sharded.cpp#L302) | `output` | none — borrowed-memory buffer |
| [rm_stride:80](device/slice_program_factory_rm_stride.cpp#L80) | `input` | reader slot 0 |
| [rm_stride:83](device/slice_program_factory_rm_stride.cpp#L83) | `output` | writer slot 0 |
| [tile:65](device/slice_program_factory_tile.cpp#L65) | `input` | reader **common** slot 0 |
| [tile:152](device/slice_program_factory_tile.cpp#L152) | `output` | writer slot 0 |
| [tile_tensor_args:82](device/slice_program_factory_tile_tensor_args.cpp#L82) | `input` | reader common slot 0 |
| [tile_tensor_args:83](device/slice_program_factory_tile_tensor_args.cpp#L83) | `start_tensor` | reader common slot 1 |
| [tile_tensor_args:84](device/slice_program_factory_tile_tensor_args.cpp#L84) | `end_tensor` | reader common slot 2 |
| [tile_tensor_args:87](device/slice_program_factory_tile_tensor_args.cpp#L87) | `output` | writer slot 0 |

### Work split

- `SliceRmProgramFactory`: `split_work_to_cores(grid, output.physical_volume() / output.padded_shape()[-1])`
  — sticks per core, two groups.
- `SliceRmShardedProgramFactory`: n/a — driven by the output shard spec's grid, not `split_work_to_cores`.
- `SliceRmStrideProgramFactory`: `split_work_to_cores(grid, output_shape.volume() / output_shape[-1])`
  — rows per core. The factory then re-derives its own `base_rows_per_core` / `extra_rows` rather than
  using the returned group counts; the port keeps that verbatim.
- `SliceTileProgramFactory`, `SliceTileTensorArgsProgramFactory`:
  `split_work_to_cores(grid, output.physical_volume() / TILE_HW)` — tiles per core, two groups.

Every factory takes `args.sub_core_grids` as the grid when set, except the sharded one, which logs a
warning and ignores it.

### Shared kernels

| kernel | class | `_metal2` fork beside it? | rung |
|---|---|---|---|
| `../../eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | **borrowed** (bound by `SliceTileTensorArgsProgramFactory`) | **yes** — `writer_unary_interleaved_start_id_metal2.cpp`, same directory | **rung 1: reuse** |

The fork's binding vocabulary, which is this port's constraint rather than a free choice:
`dfb::out`, `tensor::dst`, `args::num_pages`, `args::start_id`. It gates `#ifdef OUT_SHARDED` and
`#ifdef BACKWARDS`; slice defines neither, so the port adds no `compiler_options.defines`. The fork
already has consumers, so it is read-only to this port; no second fork was created and the original
was not touched.

None of slice's own nine kernel files is bound by any other non-quasar op, and no two slice factories
share a kernel source, so every slice-owned kernel converts in place with no fork. In particular the
slice-owned `writer_unary_interleaved_start_id.cpp` is a *different file* from the `eltwise/unary`
namesake and is bound only by `SliceTileProgramFactory`.

### Flags

- **A host-side cross-op consumer forces an out-of-directory edit.** `ccl/mesh_partition` builds its
  MeshWorkload out of slice's factories: `create_at` calls `Factory::create_descriptor` inside a
  `std::visit` over `SliceDeviceOperation::program_factory_t`, and `override_runtime_arguments` calls
  `ttnn::prim::patch_slice_program_addresses`.
  `ProgramDescriptorFactoryConcept` is satisfied by the mere presence of `create_descriptor`
  ([../../../../../../api/ttnn/operation_concepts.hpp:73-74](../../../../../../api/ttnn/operation_concepts.hpp#L73-L74)),
  and both Metal 2.0 concepts require `!ProgramDescriptorFactoryConcept`, so a ported factory cannot
  keep the method as a shim. Because the `std::visit` instantiates its lambda for every alternative,
  there is no subset of slice that ports without touching this peer op. Porting all five keeps the
  edit uniform: both visit sites go through the spec API with no concept branching. The invoker
  authorized the change; it is recorded as a Handoff point in `METAL2_PORT_REPORT.md`.
- **A Python-side consumer of the pybound `create_descriptor`.**
  `models/experimental/ops/descriptors/data_movement/slice.py:54` calls
  `ttnn.SliceTileProgramFactory.create_descriptor`, and four fusion tests import it. The port removes
  the *method* binding and keeps the *class* binding, so `ttnn.SliceTileProgramFactory` still resolves
  and the existing skip mechanism in
  `tests/ttnn/unit_tests/operations/fused/parallel_sequential/conftest.py` can key on the missing
  method, exactly as it already does for the two layernorm factories. See the report.
- **Two unreferenced kernel files** in the op directory, named by no factory:
  `device/kernels/dataflow/strided_slice_reader_rm_interleaved_nd.cpp` and
  `device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp`. Not converted, not audited, not
  deleted — out of port scope.
- **Two host helpers exported from this op's header are called by four other ops.**
  `get_rm_start_offset` and `get_tiled_start_offset`
  ([device/slice_device_operation.hpp:23-25](device/slice_device_operation.hpp#L23-L25)) are pure host-side
  index arithmetic. Both declarations and definitions stay exactly where they are.
- **`slice_tile_dynamic_args` is not the deprecated `get_dynamic_runtime_args` hook** — it is a helper
  the factory's own override called. The port replaces it with `slice_tile_per_core_run_args`, which
  returns a `ProgramRunArgs` instead of a `std::vector<DynamicRuntimeArg>`.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `CustomProgramSpecFactoryConcept` on all five factories — every
  one defines `override_runtime_arguments`, which the port translates rather than deletes.
- **Custom `compute_program_hash`**: present at
  [device/slice_device_operation.cpp:348-432](device/slice_device_operation.cpp#L348-L432) — leave intact.
- **Implementation notes**:
  - The five legacy overrides all delegated to one shared imperative patcher. Since nothing is left on
    the descriptor API, that patcher is deleted outright and each factory's override builds its own
    `ProgramRunArgs`. The two tile factories share the per-core scalar rebuild through
    `slice_tile_per_core_run_args`, which is where the one genuinely shared piece of logic lived.
  - Pybind: the `create_descriptor` `def_static` at
    [slice_nanobind.cpp:168-179](slice_nanobind.cpp#L168-L179) is deleted (device-op-class exception 1).
    The `nb::class_` line is kept, so the type still exists in Python without that method.

## Planned Spec Shape

Common to every factory: `TensorParameter`s are declared from `<tensor>.tensor_spec()` with
`relaxations` left default (strict) — the audit's relaxation value is `none` on all five rows.
Every kernel is data movement, so `hw_config` is `create_reader_datamovement_config(arch)` on each
reader and `create_writer_datamovement_config(arch)` on each writer; every legacy config is the plain
`ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` default, so the resolved triples match the
helpers exactly.

### Variant: `SliceRmProgramFactory`

- KernelSpecs: `reader`, `writer`
- DataflowBufferSpecs: `sticks` (`entry_size = sizing.dfb_page_size`, `num_entries = num_read_per_barrier * 2`)
- SemaphoreSpecs: none
- TensorParameters: `input`, `output`
- WorkUnitSpecs: one, `{reader, writer}` over `all_cores`

### Variant: `SliceRmShardedProgramFactory`

- KernelSpecs: `reader`
- DataflowBufferSpecs: `in` (`borrowed_from = input`, `entry_size = src_stride_bytes`,
  `num_entries = shard_height_padded`); `out` (`borrowed_from = output`,
  `entry_size = dst_stride_bytes`, `num_entries = shard_height_unpadded`)
- SemaphoreSpecs: none
- TensorParameters: `input`, `output` — both exist to back a borrowed buffer. `input` is bound by no
  kernel, which is legal precisely because a `borrowed_from` reference counts as use.
- WorkUnitSpecs: one, `{reader}` over `all_cores_unpadded`

### Variant: `SliceRmStrideProgramFactory` (both rank configs)

- KernelSpecs: `reader`, `writer` — one pair, with `source` and `runtime_arg_schema` selected at
  construction from the rank, exactly as the legacy factory selects its kernel path.
- DataflowBufferSpecs: `row` (`entry_size = dfb_page_size_aligned`, `num_entries = 2`)
- SemaphoreSpecs: none
- TensorParameters: `input`, `output`
- WorkUnitSpecs: one, `{reader, writer}` over `all_cores`

### Variant: `SliceTileProgramFactory`

- KernelSpecs: `reader`, `writer`
- DataflowBufferSpecs: `tiles` (`entry_size = single_tile_size`, `num_entries = 2`)
- SemaphoreSpecs: none
- TensorParameters: `input`, `output`
- WorkUnitSpecs: one, `{reader, writer}` over `all_cores`

### Variant: `SliceTileTensorArgsProgramFactory`

- KernelSpecs: `reader`, `writer` (the writer bound to the existing `_metal2` fork)
- DataflowBufferSpecs: `tiles` (`entry_size = single_tile_size`, `num_entries = 2`);
  `staging` (`entry_size = single_tile_size`, `num_entries = 1`)
- SemaphoreSpecs: none
- TensorParameters: `input`, `start`, `end`, `output`
- WorkUnitSpecs: one, `{reader, writer}` over `all_cores`

### DFB endpoint dispositions

Re-derived from the kernel-touch census rather than transcribed; the result agrees with the brief on
every row.

| Factory | DFB | Touching kernels | Disposition |
|---|---|---|---|
| `SliceRmProgramFactory` | `sticks` | reader (`reserve_back`/`push_back`), writer (`wait_front`/`pop_front`) | 1P + 1C |
| `SliceRmShardedProgramFactory` | `in` | reader only, raw `get_write_ptr()` peek, no FIFO ops | **self-loop** |
| `SliceRmShardedProgramFactory` | `out` | reader only, `reserve_back`/`get_write_ptr`/`push_back`, nothing drains | **self-loop** |
| `SliceRmStrideProgramFactory` | `row` | reader, writer (both configs) | 1P + 1C |
| `SliceTileProgramFactory` | `tiles` | reader, writer | 1P + 1C |
| `SliceTileTensorArgsProgramFactory` | `tiles` | reader, borrowed writer | 1P + 1C |
| `SliceTileTensorArgsProgramFactory` | `staging` | reader only, full reserve→push→wait→pop twice | **self-loop** |

No DFB needs `allow_instance_multi_binding`: none has three or more distinct touchers, and none has
two kernels locked to the same FIFO role. No dead CB. No conditional DFB.

## Preserved Multiplicity

none — no work-split multiplicity in legacy. No factory pushes the same `kernel_source` into two
`KernelDescriptor`s; `SliceRmStrideProgramFactory` selects *between* two sources by rank rather than
instantiating one source twice.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| [rm:406](device/slice_program_factory_rm.cpp#L406) | reader RTA slot 0 `Buffer* src0_buffer` | `TensorBinding` on `input` |
| [rm:414](device/slice_program_factory_rm.cpp#L414) | writer RTA slot 0 `Buffer* dst_buffer` | `TensorBinding` on `output` |
| [rm:362](device/slice_program_factory_rm.cpp#L362), [:365](device/slice_program_factory_rm.cpp#L365) | `TensorAccessorArgs(...).append_to(...)` | binding mechanism end-to-end |
| [rm:361](device/slice_program_factory_rm.cpp#L361) | writer CTA slot 0 `src0_cb_index` | `DFBBinding` on `sticks` |
| [rm reader kernel:35](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp#L35), [:42](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp#L42) | `TensorAccessorArgs<0>()`, `constexpr uint32_t dfb_id_in0 = 0` | `TensorAccessor(tensor::src)`, `dfb::in0` |
| [rm writer kernel:26-27](device/kernels/dataflow/slice_writer_unary_stick_layout_interleaved_start_id.cpp#L26-L27) | `get_compile_time_arg_val(0)`, `TensorAccessorArgs<1>()` | `dfb::out0`, `TensorAccessor(tensor::dst)` |
| [rm_sharded:290](device/slice_program_factory_rm_sharded.cpp#L290), [:302](device/slice_program_factory_rm_sharded.cpp#L302) | `CBDescriptor.buffer = <tensor>.buffer()` | `DataflowBufferSpec::borrowed_from` |
| [rm_sharded kernel:32-33](device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp#L32-L33) | `tt::CBIndex::c_0` / `tt::CBIndex::c_16` literals | `dfb::in` / `dfb::out` |
| [rm_stride:128](device/slice_program_factory_rm_stride.cpp#L128), [:147](device/slice_program_factory_rm_stride.cpp#L147) | reader RTA slot 0 `Buffer* input_buffer` | `TensorBinding` on `input` |
| [rm_stride:136](device/slice_program_factory_rm_stride.cpp#L136), [:160](device/slice_program_factory_rm_stride.cpp#L160) | writer RTA slot 0 `Buffer* output_buffer` | `TensorBinding` on `output` |
| [rm_stride:79](device/slice_program_factory_rm_stride.cpp#L79), [:82](device/slice_program_factory_rm_stride.cpp#L82) | CTA slot 0 `in_cb` | `DFBBinding` on `row` |
| [rm_stride:80](device/slice_program_factory_rm_stride.cpp#L80), [:83](device/slice_program_factory_rm_stride.cpp#L83) | `TensorAccessorArgs(...).append_to(...)` | binding mechanism end-to-end |
| [4d reader:82](device/kernels/dataflow/reader_multicore_slice_4d.cpp#L82), [4d writer:66](device/kernels/dataflow/writer_multicore_slice_4d.cpp#L66), [nd reader:68](device/kernels/dataflow/reader_multicore_slice_nd.cpp#L68), [nd writer:67](device/kernels/dataflow/writer_multicore_slice_nd.cpp#L67) | `TensorAccessorArgs<2>()` | `TensorAccessor(tensor::src)` / `(tensor::dst)` |
| [tile:143](device/slice_program_factory_tile.cpp#L143) | reader CRTA slot 0 `Buffer* src0_buffer` | `TensorBinding` on `input` |
| [tile:180](device/slice_program_factory_tile.cpp#L180) | writer RTA slot 0 `Buffer* dst_buffer` | `TensorBinding` on `output` |
| [tile:139](device/slice_program_factory_tile.cpp#L139), [:161](device/slice_program_factory_tile.cpp#L161) | named CTA `dfb_id_in` / `dfb_id_out` | `DFBBinding` on `tiles` — a named CB index is still a binding, never a named arg |
| [tile:65](device/slice_program_factory_tile.cpp#L65), [:152](device/slice_program_factory_tile.cpp#L152) | `TensorAccessorArgs(...).append_to(...)` | binding mechanism end-to-end |
| [tile_tensor_args:182-184](device/slice_program_factory_tile_tensor_args.cpp#L182-L184) | reader CRTA slots 0/1/2 `Buffer*` × 3 | `TensorBinding` on `input` / `start` / `end` |
| [tile_tensor_args:151](device/slice_program_factory_tile_tensor_args.cpp#L151), [:168](device/slice_program_factory_tile_tensor_args.cpp#L168) | writer RTA slot 0 `Buffer* dst_buffer` | `TensorBinding` on `output` (the fork's `tensor::dst`) |
| [tile_tensor_args:80-81](device/slice_program_factory_tile_tensor_args.cpp#L80-L81) | reader CTA slots 0, 1 `src0_cb_index`, `tensor_cb_index` | `DFBBinding` on `tiles` / `staging` |
| [tile_tensor_args:86](device/slice_program_factory_tile_tensor_args.cpp#L86) | writer CTA slot 0 `src0_cb_index` | `DFBBinding` on the fork's `dfb::out` |
| [tile_tensor_args:82-84](device/slice_program_factory_tile_tensor_args.cpp#L82-L84), [:87](device/slice_program_factory_tile_tensor_args.cpp#L87) | four `TensorAccessorArgs(...).append_to(...)` | binding mechanism end-to-end |
| every positional CTA above | positional `compile_time_args` vector | named `compile_time_args` table |

**Page-size 3rd-argument CTAs/RTAs**: none. All 14 `TensorAccessor` construction sites in this op take
exactly two arguments, so there is no third argument to drop.

**Semaphore-ID RTAs**: none — the op has no semaphore.

### Named vs vararg, per kernel

The audit's vararg census, re-derived. In every vararg case the fixed scalars come first and the
variable-count block last, so no nameable trailing scalar is trapped in the varargs.

| kernel | named RTAs | runtime varargs | common runtime varargs |
|---|---|---|---|
| `rm` reader | `unpadded_stick_size`, `stick_size_offset`, `num_dims`, `misalignment`, `start_id`, `num_sticks_per_core`, `num_sticks_per_core_read`, `num_read_per_barrier`, `chunk_size`, `num_chunks_per_stick`, `last_chunk_size`, `src_offset_bytes` | `3 * num_dims` (`num_unpadded_sticks`, `num_padded_sticks`, `id_per_dim`) | none |
| `rm` writer | all 9 | none | none |
| `rm_sharded` reader | `num_cores_read` | the whole tail; length differs per node, so the longest is declared and shorter nodes are zero-filled | none |
| `4d` reader | all 24 | none | none |
| `4d` writer | all 8 | none | none |
| `nd` reader | `tensor_rank`, `element_size`, `num_rows_for_this_core`, `start_row_for_this_core` | `5 * tensor_rank` | none |
| `nd` writer | `tensor_rank`, `element_size`, `num_rows_for_this_core`, `start_row_for_this_core` | `tensor_rank` | none |
| `tile` reader | `start_id`, `num_tiles` | `num_dims` (`id_per_dim`) | `2 * num_dims` |
| `tile` writer | `num_pages`, `start_id` | none | none |
| `tile_tensor_args` reader | `start_id`, `num_tiles` | `num_dims` (`id_per_dim`) | `3 * num_dims` |
| `tile_tensor_args` writer | `num_pages`, `start_id` (the fork's names) | none | none |

**No compile-time varargs anywhere** — no kernel reads `get_compile_time_arg_val` at a varying index.

### The `id_per_dim` block: a vararg the kernel used to write

Three readers used their runtime-argument region as mutable per-core scratch (`id_per_dim[j]++`,
`id_per_dim[j] = 0`). Metal 2.0's vararg accessor is a value getter with no address form, so the block
cannot be written back. **The port copies the block into a kernel-local array once, at entry, and
advances the local.** This is behaviour-identical: nothing reads the block back from L1 after the
kernel exits, and the host reseeds it on every dispatch either way.

Two shapes, decided by where `num_dims` comes from:

- Both tile readers take it from a **compile-time** argument, so a plain `uint32_t id_per_dim[num_dims]`
  local works.
- The RM reader takes it from a **runtime** argument, so the local is sized by `tensor_accessor::MAX_RANK`
  (8) — the accessor's own rank ceiling, already in scope in these kernels — with a device `ASSERT`.

`data_movement/pad` already established this pattern
([../pad/device/kernels/dataflow/reader_pad_tiled.cpp:21-33](../pad/device/kernels/dataflow/reader_pad_tiled.cpp#L21-L33)).
The underlying framework gap is reported in `METAL2_PORT_REPORT.md`.

## Applied Patterns

- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):
  `SliceRmShardedProgramFactory`'s `in` (sync-free address source) and `out` (single-ended producer),
  and `SliceTileTensorArgsProgramFactory`'s `staging` (one kernel both fills and drains).
- [Self-loop DFB binding](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding):
  the mechanism the three self-loops borrow — PRODUCER and CONSUMER bindings on one kernel sharing one
  `accessor_name`.
- [Caution: Porting a shared kernel](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel),
  rung 1: `SliceTileTensorArgsProgramFactory`'s writer reuses the existing
  `writer_unary_interleaved_start_id_metal2.cpp` fork beside the `eltwise/unary` original.
- [Multi-variant factories](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories):
  `SliceRmStrideProgramFactory` selects its kernel sources by rank inside `create_program_artifacts`.
- [Unity-build hygiene for anonymous-namespace symbols](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols):
  `ttnn_op_data_movement` is a unity build and slice has five factory translation units, so the shared
  spec names live in `device/slice_metal2_names.hpp` as `inline const` and every per-factory buffer
  name carries a factory-specific identifier.
- [Caution: Avoid varargs unless absolutely necessary](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary):
  six kernels keep a genuine variable-count block; the rest are fully named. Reported in the report.
- [Removing pybound legacy factory entry points](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-removing-pybound-legacy-factory-entry-points):
  the `create_descriptor` `def_static` at [slice_nanobind.cpp:168-179](slice_nanobind.cpp#L168-L179).

## Deferred / Flagged

- **New finding during planning: the cross-op host consumer is a scope-boundary breach the recipe has
  no category for.** The audit flagged `ccl/mesh_partition` under its open "anything else" bullet and
  asked the port to "plan for it". Planning established that it is not plannable-around: it forces an
  edit to a peer op's host code, which the recipe's three documented device-op-class exceptions do not
  cover. The invoker authorized the edit before construction began.
- **New finding during construction: removing a pybound entry point has a Python-side tail.** Deleting
  the whole `nb::class_` block (which is what "delete the binding at lines 168-179" reads as) breaks
  `ttnn/ttnn/__init__.py` at import time, because the class name is re-exported there. Keeping the
  class and removing only the `def_static` is the right granularity. Recorded in the report.
- **`SliceRmShardedProgramFactory` declares an `input` `TensorParameter` that no kernel binds.** It
  exists only to back the borrowed `in` buffer. The validator accepts this, but it is the first place
  to look if a "TensorParameter never bound" rejection appears.
- The audit's misc anomalies (dead `end_tensor` read, dead RTAs in the rank ≤ 4 stride kernels, the
  dead `compile_time_element_size` CTA, the dead sharded `writer_kernel_args` vector, `sub_core_grids`
  fed to the hash while the sharded factory ignores it) are all carried forward unchanged.
