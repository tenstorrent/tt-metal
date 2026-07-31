# Port Plan — `ttnn/cpp/ttnn/operations/full`

Port plan for `full` (`FullDeviceOperation`), ported from `ProgramDescriptor` (`ProgramDescriptorFactoryConcept`)
to Metal 2.0 (`ProgramSpecFactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

All three factories are ported in this change. They are independent (one kernel each, no shared kernel
source between them except the interleaved factory's own two instances of `writer_full.cpp`), and the
whole op is small enough that splitting the pass would create more coordination cost than it saves.

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — all three factories declare
  `static ProgramDescriptor create_descriptor(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&)`
  ([interleaved.hpp:13](device/full_program_factory_interleaved.hpp#L13),
  [sharded.hpp:13](device/full_program_factory_sharded.hpp#L13),
  [nd_sharded.hpp:13](device/full_program_factory_nd_sharded.hpp#L13)).
- Variants: three, held in `FullDeviceOperation::program_factory_t`
  ([full_device_operation.hpp:25-26](device/full_device_operation.hpp#L25-L26)) and selected on the output's
  memory config ([full_device_operation.cpp:13-22](device/full_device_operation.cpp#L13-L22)):
  `FullInterleavedProgramFactory`, `FullShardedProgramFactory`, `FullNDShardedProgramFactory`.
- Custom `compute_program_hash`: none — already the default reflection-based hash. Zero grep hits in the
  op directory.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN factory
analysis section. Carried forward in the TTNN ProgramFactory section below.)*

**Structural fact that shapes every variant:** the op takes **no input tensors**. `tensor_args_t` is an empty
struct ([full_device_operation_types.hpp:22](device/full_device_operation_types.hpp#L22)). The output tensor it
creates is the only tensor in play, so each factory has exactly one tensor binding. The fill value is a
scalar on an RTA.

---

### Variant: interleaved (`FullInterleavedProgramFactory`)

Host-side geometry ([full_program_factory_interleaved.cpp:24-33](device/full_program_factory_interleaved.cpp#L24-L33)):
`grid = mesh_device->compute_with_storage_grid_size()`, `num_pages = output.buffer()->num_pages()`,
`page_size = output.buffer()->page_size()`, `elems_per_page = page_size / output.element_size()`,
`data_format = datatype_to_dataformat_converter(dtype)`.

`has_reader = (num_pages > num_cores)` ([:66](device/full_program_factory_interleaved.cpp#L66)) gates the
second kernel **and** the second CB.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| writer | `device/kernels/writer_full.cpp` | `all_cores` | `[0]` = `CBIndex::c_0` (0) · `[1]` = `elems_per_page` · `[2]` = `page_size` · `[3..]` = `TensorAccessorArgs(output.buffer())` ([:52-53](device/full_program_factory_interleaved.cpp#L52-L53)) | none | per core: `{output.buffer(), u.u32, num_pages_per_writer, writer_page_start}` ([:108](device/full_program_factory_interleaved.cpp#L108)); or `{output.buffer(), u.u32, num_pages_per_core, page_offset}` when `!has_reader` ([:110](device/full_program_factory_interleaved.cpp#L110)) | none | exactly one of `OUTPUT_DTYPE_BFLOAT16` / `OUTPUT_DTYPE_INT32` / `OUTPUT_DTYPE_FLOAT32` = `"1"` | field absent → resolves to **O2** (DM) | `WriterConfigDescriptor{}` → `(RISCV_0, NOC_1, DM_DEDICATED_NOC)` |
| reader *(only when `has_reader`)* | `device/kernels/writer_full.cpp` (same source) | `all_cores` (same range) | `[0]` = `CBIndex::c_1` (1) · `[1]` = `elems_per_page` · `[2]` = `page_size` · `[3..]` = `TensorAccessorArgs(output.buffer())` ([:79-80](device/full_program_factory_interleaved.cpp#L79-L80)) | none | per core: `{output.buffer(), u.u32, num_pages_per_reader, reader_page_start}` ([:104](device/full_program_factory_interleaved.cpp#L104)) | none | same set | field absent → **O2** (DM) | `ReaderConfigDescriptor{}` → `(RISCV_1, NOC_0, DM_DEDICATED_NOC)` |

`opt_level` was resolved by `grep -n opt_level` over the whole op directory: **zero hits**, so no kernel sets
one anywhere. Both kernels are DM, whose legacy default is `O2` — the same as Metal 2.0's
`CompilerOptions::opt_level` default, so nothing needs setting.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| 0 (`c_0`) | `page_size` | `all_cores` | `data_format` | `page_size` | not set |
| 1 (`c_1`) *(only when `has_reader`)* | `page_size` | `all_cores` | `data_format` | `page_size` | not set |

Both are single-page (`total_size == page_size`), matching the kernel's `reserve_back(onepage)`.

#### Semaphores

none — the op uses no semaphores in any variant.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| [interleaved:53](device/full_program_factory_interleaved.cpp#L53) (`TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args)`) | `output` (`tensor_return_value`) | slot 0 of the writer's per-core RTAs (`output.buffer()`) |
| [interleaved:80](device/full_program_factory_interleaved.cpp#L80) (same, reader) | `output` (same tensor) | slot 0 of the reader's per-core RTAs |

Kernel side: `constexpr auto dst_args = TensorAccessorArgs<3>()` +
`uint32_t output_addr = get_arg_val<uint32_t>(0)` → `TensorAccessor(dst_args, output_addr)`
([writer_full.cpp:13](device/kernels/writer_full.cpp#L13), [:21](device/kernels/writer_full.cpp#L21),
[:58](device/kernels/writer_full.cpp#L58)).

#### Work split

- Driver: `split_work_to_cores(grid, num_pages)` ([:26-27](device/full_program_factory_interleaved.cpp#L26-L27))
- num_cores: `num_cores`
- core_group_1: `core_group_1`, count_per_core: `num_pages_per_core_group_1`
- core_group_2: `core_group_2`, count_per_core: `num_pages_per_core_group_2`

The per-group page count reaches the kernel as an **RTA** (`num_pages_per_core`), not a CTA, so there is no
per-group CTA and therefore no `KernelSpec` multiplicity forced by the work split. The two
`KernelDescriptor`s of `writer_full.cpp` are a Writer/Reader *config* split over one core range, not a
core-group split — a distinct thing (see Preserved Multiplicity).

---

### Variant: sharded (`FullShardedProgramFactory`)

Host-side geometry ([full_program_factory_sharded.cpp:27-56](device/full_program_factory_sharded.cpp#L27-L56)):
`tensor_width_in_pages = output.buffer()->shard_spec().tensor2d_shape_in_pages[1]`,
`runtime_cores = get_optimal_worker_cores_for_sharded_tensor(output)`,
`compute_core_range = CoreRangeSet(runtime_cores)`,
`aligned_page_size = output.buffer()->aligned_page_size()`, `page_size = output.buffer()->page_size()`,
`elems_per_page = page_size / datum_size(data_format)`.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| writer | `device/kernels/writer_full_sharded.cpp` | `compute_core_range` | `[0]` = `CBIndex::c_24` (24) · `[1]` = `elems_per_page` · `[2]` = `page_size` · `[3]` = `aligned_page_size` **(no kernel reads this slot)** · `[4]` = `tensor_width_in_pages` · `[5..]` = `TensorAccessorArgs(output.buffer())` ([:55-57](device/full_program_factory_sharded.cpp#L55-L57)) | none | per core: `{output.buffer(), u.u32, first_page_id, valid_pages_width, valid_pages_height}` ([:89-90](device/full_program_factory_sharded.cpp#L89-L90)) | none | same `OUTPUT_DTYPE_*` set | field absent → **O2** (DM) | `WriterConfigDescriptor{}` |

Kernel-side RTA names: `output_addr(0)`, `fill_value(1)`, `start_page_id(2)`, `num_pages_per_shard_row(3)`,
`num_pages_per_shard_col(4)` ([writer_full_sharded.cpp:13-17](device/kernels/writer_full_sharded.cpp#L13-L17)) —
so host `valid_pages_width` → kernel `num_pages_per_shard_row` and host `valid_pages_height` → kernel
`num_pages_per_shard_col`.

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| 24 (`c_24`) | `page_size` | `compute_core_range` | `data_format` | `page_size` | not set |

#### Semaphores

none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| [sharded:57](device/full_program_factory_sharded.cpp#L57) | `output` | slot 0 of the writer's per-core RTAs |

Kernel side: `TensorAccessorArgs<5>()` + RTA 0 → `TensorAccessor(dst_args, output_addr)`
([writer_full_sharded.cpp:23](device/kernels/writer_full_sharded.cpp#L23), [:60](device/kernels/writer_full_sharded.cpp#L60)).

#### Work split

n/a — no `split_work_to_cores`. One kernel instance per core in `runtime_cores`, each handed its own shard's
`first_page_id` and valid-page extents by the loop at
[:73-91](device/full_program_factory_sharded.cpp#L73-L91).

---

### Variant: ND-sharded (`FullNDShardedProgramFactory`)

Host-side geometry ([full_program_factory_nd_sharded.cpp:27-56](device/full_program_factory_nd_sharded.cpp#L27-L56)):
`distribution_spec = output.buffer()->buffer_distribution_spec().value()`,
`num_shards = distribution_spec.num_shards()`,
`num_compute_cores = distribution_spec.cores_with_data().size()`,
`ordered_cores_with_data = get_optimal_worker_cores_for_sharded_tensor(output)`,
`compute_core_range = CoreRangeSet(ordered_cores_with_data)`, plus the same `aligned_page_size` /
`page_size` / `elems_per_page`.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| writer | `device/kernels/writer_full_nd_sharded.cpp` | `compute_core_range` | `[0]` = `CBIndex::c_24` (24) · `[1]` = `elems_per_page` · `[2]` = `page_size` · `[3]` = `aligned_page_size` **(no kernel reads this slot)** · `[4]` = `num_shards` · `[5]` = `num_compute_cores` · `[6..]` = `TensorAccessorArgs(output.buffer())` ([:55-57](device/full_program_factory_nd_sharded.cpp#L55-L57)) | none | per core: `{output.buffer(), u.u32, start_shard_id}` ([:69](device/full_program_factory_nd_sharded.cpp#L69)) | none | same `OUTPUT_DTYPE_*` set | field absent → **O2** (DM) | `WriterConfigDescriptor{}` |

Kernel-side names: RTAs `output_addr(0)`, `fill_value(1)`, `start_shard_id(2)`; CTAs `cb_value(0)`,
`elems_per_page(1)`, `page_size(2)`, `num_shards(4)`, `num_cores(5)`
([writer_full_nd_sharded.cpp:13-22](device/kernels/writer_full_nd_sharded.cpp#L13-L22)).

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| 24 (`c_24`) | `page_size` | `compute_core_range` | `data_format` | `page_size` | not set |

#### Semaphores

none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| [nd_sharded:57](device/full_program_factory_nd_sharded.cpp#L57) | `output` | slot 0 of the writer's per-core RTAs |

#### Work split

n/a — one kernel instance per core in `ordered_cores_with_data`; each gets a `start_shard_id` and strides
through the shard list by `num_cores` inside the kernel
([writer_full_nd_sharded.cpp:63](device/kernels/writer_full_nd_sharded.cpp#L63)).

---

### Shared kernels

**none.** Census run per the catalog's procedure:

- `grep -rl writer_full ttnn/cpp/ttnn/operations/` returns hits only inside this op directory. No other op's
  factory binds `writer_full.cpp`, `writer_full_sharded.cpp`, or `writer_full_nd_sharded.cpp` (*not lent*).
- All three `kernel_source` paths point inside `device/kernels/` of this op (*not borrowed*).
- No two factories of this op bind the same source: each variant has its own kernel file. The interleaved
  factory's two `KernelDescriptor`s **do** share `writer_full.cpp`, but both belong to the *same* factory and
  convert in this change, so the intra-op fork case does not arise (*not intra-op* in the
  "won't-all-convert" sense).
- No `_metal2` sibling exists beside any of the three. No fork rung applies.

`device/kernels/full_kernel_common.hpp` is an in-directory header included by all three kernels. It is
op-owned (not a kernel-library file) and all three of its consumers convert together, so it is not a shared
kernel in the Caution sense.

### Flags

- **Dead compile-time arg in both sharded factories.** Slot 3 (`aligned_page_size`) is emitted by the host but
  read by no kernel ([sharded:55-56](device/full_program_factory_sharded.cpp#L55-L56) vs
  [writer_full_sharded.cpp:19-23](device/kernels/writer_full_sharded.cpp#L19-L23);
  [nd_sharded:55-56](device/full_program_factory_nd_sharded.cpp#L55-L56) vs
  [writer_full_nd_sharded.cpp:17-22](device/kernels/writer_full_nd_sharded.cpp#L17-L22)). Recorded as an
  ops-team item in the audit's Misc anomalies, so cleaning it up is **not** port work. The port carries it
  forward as a named CTA; see Dropped Plumbing.
- **No unreferenced kernel files** in the op directory.
- **`elems_per_page` is computed two different ways** across variants (`element_size()` in interleaved,
  `datum_size(data_format)` in both sharded). They agree for the three dtypes the op allows. Left exactly as
  is; carried to the report.
- **No descriptor type outside the audit's scan.** The three factories use only `KernelDescriptor`,
  `CBDescriptor`, and the two DM config tags.

---

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`. Each factory's
  `static ProgramDescriptor create_descriptor(...)` becomes
  `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(...)`.
- **Custom `compute_program_hash`**: none — nothing to delete.
- **Implementation notes**:
  - Nothing outside the three factory bodies (plus their shared `full_program_factory_common.hpp` helper and
    the three kernels) changes. The device-operation class, `full.cpp`, and `full_nanobind.cpp` are untouched:
    the nanobind file binds only the `moreh_full` free function, so no pybind line references a vanishing
    factory entry point.
  - The factories keep querying geometry off the `ttnn::Tensor` (`output.buffer()->page_size()`,
    `->num_pages()`, `->aligned_page_size()`, `->shard_spec()`, `->buffer_distribution_spec()`,
    `element_size()`). `MeshTensor` exposes none of these `Buffer*` accessors, so the migration guide's
    "extract `MeshTensor` and work with it throughout" style does not fit here; `MeshTensor` is used only
    where the API requires it (`output.mesh_tensor()` for the `TensorArgument`). Carried to the report as
    friction.

## Planned Spec Shape

Common to all three variants:

- **TensorParameters**: one — `OUTPUT` (`"output"`), from `output.tensor_spec()`, strict matching (no
  relaxation). Bound with `accessor_name = "output"` so the kernel reads `tensor::output`.
- **SemaphoreSpecs**: none — the op has no semaphores.
- **Op-owned tensors**: none — the legacy factories allocate no device tensors beyond the op's output.

### Variant: interleaved

- **KernelSpecs**: two, both of `writer_full.cpp` — `WRITER` (`"writer"`, writer DM config) and, only when
  `has_reader`, `READER` (`"reader"`, reader DM config). One per legacy `KernelDescriptor`.
- **DataflowBufferSpecs**: `FILL_VALUE_WRITER` (`"fill_value_writer"`, was `c_0`) and, only when
  `has_reader`, `FILL_VALUE_READER` (`"fill_value_reader"`, was `c_1`). Each
  `entry_size = page_size`, `num_entries = 1`, `data_format_metadata = data_format`,
  `tile_format_metadata` left unset (legacy `tile` was unset). No aliasing, no borrowed memory.
  Both bind under `accessor_name = "value"` — the same name on both KernelSpecs, because the two specs
  compile the *same* source, which references `dfb::value`.
- **WorkUnitSpecs**: one — `"main"`, kernels `{WRITER}` or `{WRITER, READER}`, `target_nodes = all_cores`.
- **KernelRunArgs**: one per KernelSpec, RTAs built per node in the existing core loop.

### Variant: sharded

- **KernelSpecs**: one — `WRITER` (`"writer"`) of `writer_full_sharded.cpp`, writer DM config.
- **DataflowBufferSpecs**: one — `FILL_VALUE` (`"fill_value"`, was `c_24`), `entry_size = page_size`,
  `num_entries = 1`, `data_format_metadata = data_format`, `tile_format_metadata` unset. Accessor `"value"`.
- **WorkUnitSpecs**: one — `"main"`, kernels `{WRITER}`, `target_nodes = compute_core_range`.
- **KernelRunArgs**: one, RTAs per node from the existing shard loop.

### Variant: ND-sharded

Identical shape to the sharded variant, with `writer_full_nd_sharded.cpp` as the source and the ND CTA set.

## Preserved Multiplicity

**none — no work-split multiplicity in legacy.** The interleaved factory's two `KernelDescriptor`s of one
source are *not* a core-group split: they cover the **same** `all_cores` range and differ by DM config
(Writer vs Reader RISC/NOC) plus their per-instance page range, which travels as an RTA. Nothing per-group
lives in a CTA, so no CTA-carrying multiplicity has to be preserved and nothing is at risk of the
CTA→RTA demotion anti-pattern.

The two instances are recorded here anyway because they look like the multi-binding shape and are not:

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| `writer_desc` + `reader_desc`, both `writer_full.cpp`, both over `all_cores` ([:55-88](device/full_program_factory_interleaved.cpp#L55-L88)) | `WRITER` + `READER` | one — `"main"` (both kernels, `all_cores`) | **none shared.** Each instance gets its *own* DFB: `WRITER` → `FILL_VALUE_WRITER` (PRODUCER + CONSUMER), `READER` → `FILL_VALUE_READER` (PRODUCER + CONSUMER) |

Endpoint census re-derived from the kernel bodies rather than transcribed from the brief, per the
endpoint-assignment procedure. On any node, each of the five `(CB, config)` pairs has **exactly one**
distinct toucher, and that toucher is both the FIFO producer (`reserve_back`/`push_back`) and the FIFO
consumer (`wait_front`/`pop_front`) — one toucher, so **self-loop**, not 1P+1C and not multi-binding. The
count agrees with the brief.

`cb.get_write_ptr()` ([writer_full.cpp:31](device/kernels/writer_full.cpp#L31) and siblings) is a public peek
by that buffer's own FIFO producer, so it adds no toucher. `zero_buffer` writes into the same buffer from
inside the same kernel instance, so it adds none either. No `evil_set_*`, no `get_local_cb_interface`, no
semaphore anywhere in the op — so there is no hidden co-filler.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| [interleaved:52](device/full_program_factory_interleaved.cpp#L52) CTA slot 0 | `(uint32_t)cb_index` (`CBIndex::c_0`) | `DFBBinding{FILL_VALUE_WRITER, "value", PRODUCER}` + `{…, CONSUMER}` on `WRITER` |
| [interleaved:79](device/full_program_factory_interleaved.cpp#L79) CTA slot 0 | `(uint32_t)cb_index2` (`CBIndex::c_1`) | `DFBBinding{FILL_VALUE_READER, "value", PRODUCER}` + `{…, CONSUMER}` on `READER` |
| [sharded:56](device/full_program_factory_sharded.cpp#L56) CTA slot 0 | `(uint32_t)cb_fill_value_id` (`CBIndex::c_24`) | `DFBBinding{FILL_VALUE, "value", PRODUCER}` + `{…, CONSUMER}` |
| [nd_sharded:56](device/full_program_factory_nd_sharded.cpp#L56) CTA slot 0 | `(uint32_t)cb_fill_value_id` (`CBIndex::c_24`) | `DFBBinding{FILL_VALUE, "value", PRODUCER}` + `{…, CONSUMER}` |
| [interleaved:53](device/full_program_factory_interleaved.cpp#L53), [:80](device/full_program_factory_interleaved.cpp#L80) · [sharded:57](device/full_program_factory_sharded.cpp#L57) · [nd_sharded:57](device/full_program_factory_nd_sharded.cpp#L57) | `TensorAccessorArgs(output.buffer()).append_to(cta)` | `TensorParameter{OUTPUT, output.tensor_spec()}` + `TensorBinding{OUTPUT, "output"}` |
| [interleaved:104](device/full_program_factory_interleaved.cpp#L104), [:108](device/full_program_factory_interleaved.cpp#L108), [:110](device/full_program_factory_interleaved.cpp#L110) · [sharded:89-90](device/full_program_factory_sharded.cpp#L89-L90) · [nd_sharded:69](device/full_program_factory_nd_sharded.cpp#L69) — RTA slot 0 | `output.buffer()` (a `Buffer*` in the RTA list) | `TensorArgument` in `ProgramRunArgs::tensor_args`; the framework injects the base address per dispatch |
| [writer_full.cpp:21](device/kernels/writer_full.cpp#L21) · [writer_full_sharded.cpp:23](device/kernels/writer_full_sharded.cpp#L23) · [writer_full_nd_sharded.cpp:22](device/kernels/writer_full_nd_sharded.cpp#L22) | `constexpr auto dst_args = TensorAccessorArgs<3\|5\|6>()` | dropped; `TensorAccessor(tensor::output)` |
| [writer_full.cpp:13](device/kernels/writer_full.cpp#L13) · [writer_full_sharded.cpp:13](device/kernels/writer_full_sharded.cpp#L13) · [writer_full_nd_sharded.cpp:13](device/kernels/writer_full_nd_sharded.cpp#L13) — RTA 0 | `uint32_t output_addr = get_arg_val<uint32_t>(0)` | dropped with the binding |
| all remaining positional CTAs — interleaved `[1][2]`, sharded `[1][2][3][4]`, nd_sharded `[1][2][3][4][5]` | `get_compile_time_arg_val(N)` | named `compile_time_args`: `elems_per_page`, `page_size`, `aligned_page_size` (sharded variants only), `tensor_width_in_pages` (sharded), `num_shards` / `num_cores` (ND-sharded) — read as `get_arg(args::<name>)` |
| all remaining positional RTAs — interleaved `[1][2][3]`, sharded `[1]..[4]`, nd_sharded `[1][2]` | `get_arg_val<uint32_t>(N)` | named `runtime_arg_schema.runtime_arg_names`, keeping the kernels' own variable names: `fill_value`, `num_pages_per_core`, `start_id` (interleaved); `fill_value`, `start_page_id`, `num_pages_per_shard_row`, `num_pages_per_shard_col` (sharded); `fill_value`, `start_shard_id` (ND-sharded) |
| [full_program_factory_common.hpp:49-56](device/full_program_factory_common.hpp#L49-L56) | `defines_from_map` — bridges a `std::map` into the legacy `KernelDescriptor::Defines` vector | deleted; `KernelSpec::CompilerOptions::Defines` is a `Table<std::string, std::string>` with a range constructor, so `get_writer_defines(dtype)` converts directly |

**Not dropped:**

- **Page-size 3rd-argument CTAs**: none exist. All three `TensorAccessor` constructions are two-argument, and
  the kernels already query the size off the accessor (`get_aligned_page_size()`), which stays. The
  `page_size` CTA that *is* emitted feeds the fill loop and the zero path, not an accessor constructor, so it
  survives as a named CTA.
- **Semaphore-ID RTAs**: none — no semaphores.
- **The dead `aligned_page_size` CTA** (sharded and ND-sharded slot 3). No kernel reads it, but it is an
  ops-team item, not port work, so the port carries it forward as a named CTA rather than deleting it. Named
  CTAs are position-free, so the slot-3 gap that made this a renumbering trap in the legacy code disappears
  on its own; nothing downstream can silently misalign.
- **The `OUTPUT_DTYPE_*` defines**: carried onto `KernelSpec::compiler_options.defines` on every KernelSpec.
  If none reaches a kernel, its fill loop compiles out entirely and the buffer silently holds garbage.

## Applied Patterns

- [Sync-free and single-ended CBs → self-loop DFB](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):
  all five `(buffer, variant)` pairs. Each is a one-toucher whose single kernel is both FIFO producer and
  FIFO consumer, so the touching KernelSpec is bound PRODUCER **and** CONSUMER under one accessor name
  (`"value"`). All five are **DM** self-loops, which is legal on Gen1 and a Quasar-uplift concern only.
- [Self-loop DFB binding](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding):
  the mechanism the above borrows — shared-`accessor_name` form, one `DataflowBuffer` object driving both
  directions.
- [Multi-variant factories](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories):
  applies loosely — the variants are already separate factory *classes* here, so each gets its own
  `create_program_artifacts` and no in-body branch is needed.
- [Unity-build hygiene for anonymous-namespace symbols](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols):
  the three factory `.cpp`s live in one CMake target and each declares its own spec-name constants. Names
  are declared as function-local `const`s inside each `create_program_artifacts` body, so nothing lands in a
  merged anonymous namespace and no prefixing is needed.

**Not applied:** [Two-toucher DFB → assign 1P+1C](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)
— the interleaved factory has the dual-instance *shape* but no co-touched buffer, because each instance
receives its own buffer index and its own `CBDescriptor`. No `allow_instance_multi_binding` anywhere.
[Conditional / optional DFB bindings](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)
— also not applied: `has_reader` gates a whole *KernelSpec* plus its own DFB, not a binding on a kernel that
compiles both ways, so no `#ifdef` gate and no define are needed. Both `dfb::value` and `tensor::output`
exist in every build of `writer_full.cpp`.
[Aliased DFBs](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-aliased-dfbs-legacy-aliased-cbs)
and [Same-FIFO aliasing](https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names)
— every legacy `CBDescriptor` has a single-element `format_descriptors`, and no kernel aliases one buffer
under two names.

## Deferred / Flagged

- **New findings during planning: one, and it is a kernel-helper shape, not a structural problem.**
  `zero_buffer(uint32_t cb_id, uint32_t bytes)`
  ([full_kernel_common.hpp:15](device/kernels/full_kernel_common.hpp#L15)) constructs a `CircularBuffer`
  internally. Whitelist rule 1 makes the CB→DFB transition total across the op directory, so this helper has
  to change. Constructing a `DataflowBuffer` inside it would give the kernel *two* objects for one FIFO,
  which the Same-FIFO-aliasing entry explicitly forbids ("alias the handle, keep one object"). The resolution
  is to hand the helper the object the caller already holds: `zero_buffer(const DataflowBuffer&, uint32_t)`.
  Same two NoC calls, same barrier, one object. All three call sites are in this port's scope.
- No feature gate fired that the audit's Appendix A does not cover. No `GlobalCircularBuffer`, no
  `address_offset`, no `GlobalSemaphore`, no CTA varargs, no `->address()` fold, no 3rd-argument accessor
  site, no Case 2 (raw base pointer) binding.
