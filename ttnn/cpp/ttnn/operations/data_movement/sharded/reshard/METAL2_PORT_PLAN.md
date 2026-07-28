# Port Plan — `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard`

Port plan for `ReshardDeviceOperation`, ported from the `descriptor`
(`ProgramDescriptorFactoryConcept`) API to Metal 2.0 (`MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

Inputs: `METAL2_PORT_BRIEF.md` (porter-facing audit brief, GREEN) and
`METAL2_PREPORT_AUDIT.md` (team-facing full audit record).

**Scope of this pass:** all 8 program-factory variants (5 factory types) are ported together,
because they share one `program_factory_t` variant and — more importantly — the six in-family
shared kernel sources are instantiated only by these factories (see
[Cross-op kernels](#cross-op-kernels)).

---

## Legacy Inventory

### Legacy factory shape

- **Concept:** `ProgramDescriptorFactoryConcept` (`descriptor`) — all 8 variants expose
  `static tt::tt_metal::ProgramDescriptor create_descriptor(const ReshardParams&, const ReshardInputs&, Tensor&)`.
- **Variants** (`reshard_device_operation.hpp:25-33`):
  | # | Factory | Source file |
  |---|---|---|
  | 1 | `ReshardSameWidthFactory<local_is_output=true>` | `device/reshard_program_factory_same_width.cpp` |
  | 2 | `ReshardSameWidthFactory<local_is_output=false>` | same |
  | 3 | `ReshardSameHeightFactory<local_is_output=true>` | `device/reshard_program_factory_same_height.cpp` |
  | 4 | `ReshardSameHeightFactory<local_is_output=false>` | same |
  | 5 | `ReshardGenericFactory` | `device/reshard_program_factory_generic.cpp` |
  | 6 | `NdReshardCopyPagesFactory` | `device/nd_reshard_program_factory_copy_pages.cpp` |
  | 7 | `NdReshardCopyLocalShardFactory<local_is_input=true>` | `device/nd_reshard_program_factory_copy_local.cpp` |
  | 8 | `NdReshardCopyLocalShardFactory<local_is_input=false>` | same |
- **Custom `compute_program_hash`:** none — `ReshardDeviceOperation`
  (`reshard_device_operation.hpp:20-48`) declares only `select_program_factory`,
  `validate_inputs`, `validate_on_program_cache_miss`, `compute_output_specs`,
  `create_output_tensors`, `create_op_performance_model`. Already on the default
  reflection-based hash; nothing to delete.
- **Custom `override_runtime_arguments` / `get_dynamic_runtime_args`:** none.
- **Pybind `create_descriptor`:** none — `reshard_nanobind.cpp` binds only `ttnn::reshard`.
- **Op-owned tensors:** none (a `descriptor` op cannot carry them).

*(Target Metal 2.0 concept was chosen during the audit — `MetalV2FactoryConcept`. Carried
forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

---

### Variant: `NdReshardCopyPagesFactory` (DRAM→DRAM page copy)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| `reader_desc` | `reshard/device/kernels/nd_reshard_copy_pages_reader.cpp` | `grid` = full `compute_with_storage_grid_size()` rect | `TensorAccessorArgs(*input_buffer)…`, then `cb_in0_idx`(=`c_0`), `aligned_page_size` | — | `{start_page, end_page}` per core | `{input_buffer}` (`Buffer*` binding, arg 0) | — | `ReaderConfigDescriptor{}` |
| `writer_desc` | `reshard/device/kernels/nd_reshard_copy_pages_writer.cpp` | same `grid` | `TensorAccessorArgs(*output_buffer)…`, then `cb_in0_idx`, `aligned_page_size` | — | `{start_page, end_page}` per core | `{output_buffer}` (`Buffer*` binding, arg 0) | — | `WriterConfigDescriptor{}` |

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| `c_0` | `aligned_page_size * 1` | `grid` | `datatype_to_dataformat_converter(input.dtype())` | `aligned_page_size` | (unset) |

`cb.buffer = nullptr` — a real FIFO, not borrowed memory.

#### Semaphores
none — the op uses no semaphores anywhere.

#### Tensor accessors

| host site (file:line) | originating Tensor | address surfaces as |
|---|---|---|
| `nd_reshard_program_factory_copy_pages.cpp:49` | input | reader CRTA 0 (`Buffer*`) |
| `nd_reshard_program_factory_copy_pages.cpp:50` | output | writer CRTA 0 (`Buffer*`) |

#### Work split
Manual: `num_dev_pages` split evenly over `cores` with the remainder distributed one page per
core (`:104-116`). Not `split_work_to_cores`. Both kernels get the *same* `[start_page, end_page)`
per core.

---

### Variant: `NdReshardCopyLocalShardFactory<local_is_input>` (L1↔DRAM / L1→L1 shard copy)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| `brisc_desc` | `reshard/device/kernels/nd_reshard_copy_local_shards.cpp` | `CoreRangeSet(local_buffer->…cores_with_data())` | `TensorAccessorArgs(input)…`, `TensorAccessorArgs(output)…`, `aligned_page_size`, `other_aligned_page_size`, `local_is_input`, `logical_width`, `source_width`, `destination_width`, `base_page_size`, `layout==TILE` | — | `{start_shard_id}` | `{input_buffer, output_buffer, num_shards, shard_id_stride}` (args 0,1 are `Buffer*`) | — | `DataMovementConfigDescriptor{RISCV_0, RISCV_0_default(=NOC_0), DM_DEDICATED_NOC}` |
| `ncrisc_desc` | same source | same | same | — | `{start_shard_id + shard_id_stride/2}` | same | — | `DataMovementConfigDescriptor{RISCV_1, RISCV_1_default(=NOC_1), DM_DEDICATED_NOC}` |

#### CBs
**none** — the kernel copies L1↔L1/DRAM directly via `TensorAccessor` + `CoreLocalMem`.

#### Semaphores
none.

#### Tensor accessors

| host site (file:line) | originating Tensor | address surfaces as |
|---|---|---|
| `nd_reshard_program_factory_copy_local.cpp:24` | input | CRTA 0 (`Buffer*`), both kernels |
| `nd_reshard_program_factory_copy_local.cpp:25` | output | CRTA 1 (`Buffer*`), both kernels |

#### Work split
Two same-source kernel instances split the shard list by `shard_id_stride =
num_cores_with_data * 2`; brisc starts at `start_shard_id`, ncrisc at
`start_shard_id + shard_id_stride/2` (`:120-131`).

---

### Variant: `ReshardSameWidthFactory<local_is_output>` (HEIGHT→HEIGHT)

#### Kernels

`kernel_name` = `sharded/device/kernels/dataflow/reshard_same_width_reader.cpp` when
`local_is_output`, else `…/reshard_same_width_writer.cpp`. **One source, two
`KernelDescriptor`s** (`:133-145`).

| unique_id | source | core_ranges | CTAs (positional) | RTAs | config |
|---|---|---|---|---|---|
| `reader_desc` | `kernel_name` | `all_cores` = `CoreRangeSet(get_optimal_worker_cores_for_sharded_tensor(local_tensor))` | `cb_index`(=`c_0`), `interface_with_dram`, `unaligned`, `unit_size`, `local_unit_size_padded`, `remote_unit_size_padded`, `cb_scratch_index`(=`c_1`) | `[remote_buffer(Buffer*), local_offset, num_transfers, (bank_id, offset, units)×n]` | `ReaderConfigDescriptor{}` |
| `writer_desc` | `kernel_name` (same) | `all_cores` | same CTAs | same shape, second work half | `WriterConfigDescriptor{}` |

No named CTAs, no CRTAs, no defines in legacy.

#### CBs

| index | total_size | core_ranges | data_format | page_size | bound buffer |
|---|---|---|---|---|---|
| `c_0` | `local_units_per_shard * local_unit_size_padded` | `all_cores` | local dtype | `local_unit_size_padded` | `local_buffer` (borrowed) |
| `c_1` | `remote_units_per_shard * remote_unit_size_padded` | `all_cores` | local dtype | `remote_unit_size_padded` | `nullptr` — **exists only when `unaligned && local_is_output`** (`:107-121`) |

#### Semaphores
none.

#### Tensor accessors
No `TensorAccessor` in these kernels. The remote tensor is reached by raw
`AllocatorBank<bank_type>` addressing (`{.bank_id, .addr = src_addr + offset}`) — audit
**Case 2**. `src_addr` arrives as RTA slot 0 (`Buffer*` binding).

#### Work split
Per local core, `local_units_per_core` is split in two by
`local_units_per_kernel = div_up(local_units_per_core, 2)`; the loop over
`kernels = {&reader_desc, &writer_desc}` walks a *stateful* cursor over the remote cores
(`remote_core_idx` / `remote_core_units_rem`), so the two kernels' arg lists must be
generated in this exact interleaved order (`:152-199`).

---

### Variant: `ReshardSameHeightFactory<local_is_output>` (row-major WIDTH→WIDTH)

#### Kernels

`kernel_name` = `…/reshard_same_height_reader.cpp` when `local_is_output`, else
`…/reshard_same_height_writer.cpp`. One source, two `KernelDescriptor`s (`:86-98`).

| unique_id | source | core_ranges | CTAs (positional) | RTAs | config |
|---|---|---|---|---|---|
| `reader_desc` | `kernel_name` | `all_cores` (optimal worker cores of local tensor) | `cb_index`(=`c_0`), `interface_with_dram` | `[total_num_sticks_kernel_0, local_stride_bytes, remote_stride_bytes, remote_buffer(Buffer*), num_segments, (write_size, read_offset, bank_id, write_offset)×num_segments]` | `ReaderConfigDescriptor{}` |
| `writer_desc` | same | `all_cores` | same | same, with `total_num_sticks_kernel_1` and offsets advanced by the first half's sticks | `WriterConfigDescriptor{}` |

#### CBs

| index | total_size | core_ranges | data_format | page_size | bound buffer |
|---|---|---|---|---|---|
| `c_0` | `remote_units_per_shard * unit_size` | `all_cores` | local dtype | `unit_size` | `local_buffer` (borrowed) |

#### Semaphores
none.

#### Tensor accessors
None — raw `AllocatorBank` addressing (**Case 2**); `remote_buffer` base arrives as RTA slot 3.

#### Work split
Split along tensor height: `total_num_sticks_kernel_0 = total_num_sticks / 2`,
`_kernel_1 = total_num_sticks - _kernel_0` (`:115-116`); per-segment offsets for kernel 1 are
adjusted by `total_num_sticks_kernel_0 * stride` (`:144-146`). Segment lists come from the
in-family host helper `detail::compute_width_sharding_reshard_segments`
(`sharded/sharded_common.{hpp,cpp}`).

---

### Variant: `ReshardGenericFactory`

#### Kernels

**Runtime-selected source** (`:701-705`):
`input_buffer->page_size() != output_buffer->page_size()`
→ `…/reshard_reader_diff_width.cpp`, else `…/reshard_reader.cpp`.
Both must be converted together. One source, two `KernelDescriptor`s (`:710-722`).

| unique_id | source | core_ranges | CTAs (positional) | RTAs | config |
|---|---|---|---|---|---|
| `kernel_desc_0` | selected source | `all_cores` = optimal worker cores of `output` | `dst_cb_index`(=16), `grid.x`, `grid.y`, `page_size`, `unit_size` | `[physical_core_coords…(grid.x+grid.y), input_buffer(Buffer*), num_output_pages, num_ranges, output_page_offset, payload…]` | `ReaderConfigDescriptor{}` |
| `kernel_desc_1` | same | `all_cores` | same | same, second half of the ranges, `output_page_offset` = kernel 0's `num_output_pages` | `WriterConfigDescriptor{}` |

#### CBs

| index | total_size | core_ranges | data_format | page_size | bound buffer |
|---|---|---|---|---|---|
| 16 | TILE: `(output.shard numel / TILE_HW) * tile_size`; ROW_MAJOR: `shard[0]*shard[1]*element_size` | `all_cores` | input dtype | `output_buffer->page_size()` | `output_buffer` (borrowed) |

Note the CB's `page_size` (`output_buffer->page_size()`) is *not* the same value as the
`page_size`/`unit_size` CTAs (tile size, or the input/output page-size GCD for row-major).

#### Semaphores
none.

#### Tensor accessors
None — the kernel issues raw `UnicastEndpoint` NoC reads with explicit `noc_x`/`noc_y` from the
per-core coordinate table and `.addr = input_shard_addr + addr_offset`
(`reshard_reader.cpp:60-68`) — audit **Case 2**.

#### Work split
Per output core, the range list is split in half by
`div_up(page_stride_vector.size(), 2)`; kernel 1's `output_page_offset` is kernel 0's page
count (`:749`, `:767-769`).

---

### Cross-op kernels

Six kernel sources live **one directory up** from the op, in the in-family shared pool
`ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/`:

- `reshard_reader.cpp`
- `reshard_reader_diff_width.cpp`
- `reshard_same_width_reader.cpp`
- `reshard_same_width_writer.cpp`
- `reshard_same_height_reader.cpp`
- `reshard_same_height_writer.cpp`

**Correction to the brief.** The brief and the audit's Heads-ups call these a "port-together
set" co-borrowed by `ttnn/cpp/ttnn/operations/experimental/quasar/reshard/`, requiring both
consuming ops to be ported as one unit. **That is no longer true.** The Quasar reshard carries
its *own private copies* of all nine kernels under
`experimental/quasar/reshard/device/kernels/` and instantiates only those paths (verified by
grepping every `kernels/` string literal in
`experimental/quasar/reshard/device/*.cpp`). A repo-wide search for consumers of the
`data_movement/sharded/device/kernels/dataflow/reshard_*` paths returns **only** the two
reshard factories being ported here (`reshard_program_factory_same_width.cpp`,
`reshard_program_factory_same_height.cpp`, and `reshard_program_factory_generic.cpp` via its
runtime-selected path string). `tests/tt_metal/.../reshard_hardcoded/kernels/reshard_reader.cpp`
is a separate copy inside the test tree, not this file.

**Decision: modify in place, no fork.** With a single consumer there is no bundled-set to
coordinate and no reason to leave a legacy copy behind. Recorded in the port report under
*Open items for downstream* per the shared-dataflow-kernel Caution.

### Flags

- **Unreferenced kernel files in the op directory:** none. All three op-owned kernels
  (`nd_reshard_copy_local_shards.cpp`, `nd_reshard_copy_pages_reader.cpp`,
  `nd_reshard_copy_pages_writer.cpp`) are referenced.
- **Dead RTA read:** `num_output_pages` is unpacked but never used in
  `reshard_reader.cpp:24` and `reshard_reader_diff_width.cpp:24` (audit "Misc anomalies").
  The port keeps it as a named RTA read — dropping it would be an unrelated cleanup.
- **Live `DPRINT`** in `reshard_same_width_reader.cpp:46` plus commented-out
  `print_bf16_pages` calls (audit "Misc anomalies"). Left exactly as-is.
- **Legacy per-core recomputation:** `ReshardGenericFactory` calls
  `detail::get_core_page_ranges…()` *inside* the per-core loop (`:738-758`), rebuilding the
  whole output-core→page-range map once per core. Preserved verbatim; hoisting it is an
  unrelated perf change.
- **`local_units_per_core` mutated inside the transfer loop** in the same-width factory
  (`:190`) while also bounding `local_units_to_transfer` — preserved verbatim.

---

## TTNN ProgramFactory

- **Concept (inherited from audit):** `MetalV2FactoryConcept` — all 8 variants.
  Each factory's `create_descriptor` becomes
  `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const ReshardParams&, const ReshardInputs&, Tensor&)`.
- **Custom `compute_program_hash`:** none — nothing to delete.
- **Implementation notes:**
  - `program_factory_t` flips wholly to Metal 2.0 in this pass; no mixed-concept variant.
  - Factory headers drop `#include <tt-metalium/program_descriptors.hpp>` and gain
    `#include "ttnn/metal_v2_artifacts.hpp"`.
  - `ProgramArtifacts::op_owned_tensors` is left default-empty everywhere.
  - Every `TensorArgument` references the `MeshTensor` of `tensor_args.input` or of the
    `output_tensor` parameter directly, so the adapter's pointer-identity match succeeds.
    The `local_tensor` / `remote_tensor` aliases in the same-width/same-height factories are
    `const&` bindings to those two, so `.mesh_tensor()` yields the same object.

---

## Planned Spec Shape

### Variant: `NdReshardCopyPagesFactory`

- **KernelSpecs:** `reader` (`nd_reshard_copy_pages_reader.cpp`), `writer`
  (`nd_reshard_copy_pages_writer.cpp`).
- **DataflowBufferSpecs:** `page` — `entry_size = aligned_page_size`, `num_entries = 1`,
  `data_format_metadata = data_format`. Not borrowed.
- **SemaphoreSpecs:** none.
- **TensorParameters:** `input` (bound on `reader`), `output` (bound on `writer`).
- **WorkUnitSpecs:** one, `{reader, writer}` over `grid`.
- **Op-owned tensors:** none.

### Variant: `NdReshardCopyLocalShardFactory<local_is_input>`

- **KernelSpecs:** `brisc`, `ncrisc` — same source `nd_reshard_copy_local_shards.cpp`.
- **DataflowBufferSpecs:** none.
- **SemaphoreSpecs:** none.
- **TensorParameters:** `input`, `output` — both bound on both kernels.
- **WorkUnitSpecs:** one, `{brisc, ncrisc}` over `grid`.

### Variant: `ReshardSameWidthFactory<local_is_output>`

- **KernelSpecs:** `reader`, `writer` — both of the single selected source.
- **DataflowBufferSpecs:**
  - `shard` — `entry_size = local_unit_size_padded`, `num_entries = local_units_per_shard`,
    `data_format_metadata = data_format`, `borrowed_from = "local"`.
  - `scratch` — **only when `unaligned && local_is_output`** —
    `entry_size = remote_unit_size_padded`, `num_entries = remote_units_per_shard`,
    `data_format_metadata = data_format`, not borrowed.
- **SemaphoreSpecs:** none.
- **TensorParameters:** `remote` (bound on both kernels, Case 2), `local`
  (**borrowed-only** — named by `shard.borrowed_from`; no kernel `TensorBinding`, which the
  spec validator explicitly accepts as a use, `program_spec.cpp:533-543`).
- **WorkUnitSpecs:** one, `{reader, writer}` over `all_cores`.

### Variant: `ReshardSameHeightFactory<local_is_output>`

- **KernelSpecs:** `reader`, `writer` — both of the single selected source.
- **DataflowBufferSpecs:** `shard` — `entry_size = unit_size`,
  `num_entries = remote_units_per_shard`, `data_format_metadata = data_format`,
  `borrowed_from = "local"`.
- **SemaphoreSpecs:** none.
- **TensorParameters:** `remote` (both kernels, Case 2), `local` (borrowed-only).
- **WorkUnitSpecs:** one, `{reader, writer}` over `all_cores`.

### Variant: `ReshardGenericFactory`

- **KernelSpecs:** `reader`, `writer` — both of the runtime-selected source.
- **DataflowBufferSpecs:** `output_shard` — `entry_size = output_buffer->page_size()`,
  `num_entries = total_size / entry_size`, `data_format_metadata = data_format`,
  `borrowed_from = "output"`.
  (`entry_size * num_entries == total_size` exactly: the legacy `CircularBufferConfig` already
  required `total_size % page_size == 0`.)
- **SemaphoreSpecs:** none.
- **TensorParameters:** `input` (both kernels, Case 2), `output` (borrowed-only).
- **WorkUnitSpecs:** one, `{reader, writer}` over `all_cores`.

---

## Preserved Multiplicity

Every factory has a **dual-instance work split**: one kernel source instantiated twice
(Reader-config + Writer-config, or BRISC + NCRISC) over one node set, splitting the work by
disjoint index ranges. Both instances live on the same grid, so each shared DFB is a
**two-toucher → 1P+1C assignment** (label is cosmetic on Gen1; no
`allow_instance_multi_binding` anywhere).

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| `reader_desc` + `writer_desc` of `nd_reshard_copy_pages_{reader,writer}.cpp` — *different* sources | `reader`, `writer` | `nd_reshard_copy_pages` | `page`: `reader` PRODUCER (`reserve_back`/`push_back`), `writer` CONSUMER (`wait_front`/`pop_front`) — genuine 1:1 FIFO, no assignment needed |
| `brisc_desc` + `ncrisc_desc` of `nd_reshard_copy_local_shards.cpp` | `brisc`, `ncrisc` | `nd_reshard_copy_local` | none — no DFBs |
| `reader_desc` + `writer_desc` of `reshard_same_width_{reader\|writer}.cpp` | `reader`, `writer` | `reshard_same_width` | `shard`: `reader` PRODUCER, `writer` CONSUMER (both raw-touch via `get_write_ptr`/`get_read_ptr`). `scratch` (conditional): `reader` PRODUCER, `writer` CONSUMER |
| `reader_desc` + `writer_desc` of `reshard_same_height_{reader\|writer}.cpp` | `reader`, `writer` | `reshard_same_height` | `shard`: `reader` PRODUCER, `writer` CONSUMER |
| `kernel_desc_0` + `kernel_desc_1` of `reshard_reader{,_diff_width}.cpp` | `reader`, `writer` | `reshard_generic` | `output_shard`: `reader` PRODUCER, `writer` CONSUMER (both raw-write disjoint page ranges) |

**Re-derived from the kernel-touch census, not transcribed.** The brief's endpoint list agrees
on every entry. Census per CB:

- `nd_reshard_copy_pages` `c_0` — 2 touchers, **locked opposite roles** → plain 1:1.
- same-width `c_0` — 2 touchers (same source, reader-config instance calls `get_write_ptr()`,
  writer-config instance calls `get_read_ptr()`; sync-free raw touches) → 1P+1C.
- same-width `c_1` — 2 touchers (`dfb_scratch.get_write_ptr()` **and** `get_read_ptr()` in the
  *same* kernel body, so both instances touch it) → 1P+1C, not a self-loop: two distinct
  KernelSpecs are available to carry the two roles.
- same-height `c_0` — 2 touchers → 1P+1C.
- generic CB 16 — 2 touchers (both `get_write_ptr()`), output resident, nothing drains →
  1P+1C.

No CB reaches ≥3 distinct touchers and none has ≥2 kernels locked to the same FIFO role, so
**no multi-binding flag is set anywhere**, and no DFB is self-looped.

---

## Dropped Plumbing

### `NdReshardCopyPagesFactory`

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| `copy_pages.cpp:49-50`, `:69`, `:73` | `TensorAccessorArgs(*buffer).get_compile_time_args()` prefix on both CTA lists; kernel `TensorAccessorArgs<0,0>()` + `next_compile_time_args_offset()` chain | `TensorParameter` + `TensorBinding`; kernel `TensorAccessor(tensor::input)` / `(tensor::output)` |
| `copy_pages.cpp:70`, `:74` | `compile_time_args.push_back(cb_in0_idx)` (magic CB index `c_0`) | `DFBBinding{page, "page", PRODUCER\|CONSUMER}`; kernel `DataflowBuffer dfb(dfb::page)` |
| `copy_pages.cpp:71`, `:75` | `push_back(aligned_page_size)` positional CTA | named CTA `page_size` |
| `copy_pages.cpp:97-98` | `emplace_common_runtime_args({input_buffer})` / `({output_buffer})` — `Buffer*` CRTA 0 | `TensorBinding` (implicit base-address CRTA) |
| `copy_pages.cpp:113-114` | positional RTAs `{start_page, end_page}` | named RTAs `start_page`, `end_page` |

### `NdReshardCopyLocalShardFactory`

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| `copy_local.cpp:24-25`, `:81-82` | two `TensorAccessorArgs` CTA blocks; kernel's chained `TensorAccessorArgs<…>()` | two `TensorParameter`s + `TensorBinding`s; `TensorAccessor(tensor::input)` / `(tensor::output)` |
| `copy_local.cpp:83-92` | 8 positional CTAs | named CTAs `src_page_size`, `dst_page_size`, `is_reader`, `logical_width`, `src_width`, `dst_width`, `transfer_size`, `page_to_page` (names taken from the kernel's own locals) |
| `copy_local.cpp:120-121` | `emplace_common_runtime_args({input_buffer, output_buffer, num_shards, shard_id_stride})` — CRTA 0/1 are `Buffer*` | `TensorBinding`s for the two addresses; named CRTAs `num_shards`, `shard_id_stride` |
| `copy_local.cpp:128-129` | positional RTA `{start_shard_id}` | named RTA `first_shard_id` |

### `ReshardSameWidthFactory`

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| `same_width.cpp:124-131` slot 0 | `cb_index` (`c_0`) magic CB index CTA | `DFBBinding{shard, "shard", …}` |
| `same_width.cpp:124-131` slot 6 | `cb_scratch_index` (`c_1`) magic CB index CTA (read but unused on the writer source) | conditional `DFBBinding{scratch, "scratch", …}` |
| `same_width.cpp:124-131` slot 2 | `unaligned` positional CTA | `KernelSpec::compiler_options.defines["UNALIGNED"]`, because the `scratch` binding is conditional on it (whitelist rule 6) |
| `same_width.cpp:124-131` slots 1,3,4,5 | positional CTAs | named CTAs `interface_with_dram`, `unit_size`, `local_unit_size_padded`, `remote_unit_size_padded` |
| `same_width.cpp:164-165` | `kernel_args.emplace_back(remote_buffer)` — RTA slot 0 `Buffer*` | `TensorBinding{remote}` + kernel `TensorAccessor(tensor::remote).get_bank_base_address()` (Case 2 bridge) |
| `same_width.cpp:171`, `:195` | positional RTA slots 1, 2 | named RTAs `write_offset`/`read_offset`, `num_reads`/`num_writes` |
| `same_width.cpp:186-189` | RTA slots 3.. — `(bank_id, offset, units_to_transfer)` × runtime count | runtime **varargs** (`get_vararg`), padded per node to `num_runtime_varargs` |
| `same_width.cpp:104-105`, `:113-120` | `CBDescriptor{… .buffer = local_buffer}` | `DataflowBufferSpec{… .borrowed_from = "local"}` |

### `ReshardSameHeightFactory`

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| `same_height.cpp:91`, `:98` slot 0 | `cb_index` (`c_0`) magic CB index CTA | `DFBBinding{shard, "shard", …}` |
| `same_height.cpp:91`, `:98` slot 1 | `interface_with_dram` positional CTA | named CTA `interface_with_dram` |
| `same_height.cpp:127`, `:134` | `push_back(remote_buffer)` — RTA slot 3 `Buffer*` | `TensorBinding{remote}` + `get_bank_base_address()` (Case 2) |
| `same_height.cpp:124-126`, `:128` | positional RTA slots 0,1,2,4 | named RTAs `total_num_sticks`, `local_stride_bytes`, `remote_stride_bytes`, `num_segments` |
| `same_height.cpp:137-152` | RTA slots 5.. — 4 words × runtime segment count | runtime **varargs**, padded per node |
| `same_height.cpp:78-79` | `CBDescriptor{… .buffer = local_buffer}` | `DataflowBufferSpec{… .borrowed_from = "local"}` |

### `ReshardGenericFactory`

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| `generic.cpp:707-708` slot 0 | `dst_cb_index = 16` magic CB index CTA | `DFBBinding{output_shard, "output_shard", …}` |
| `generic.cpp:707-708` slots 1-4 | positional CTAs | named CTAs `num_x_cores`, `num_y_cores`, `page_size`, `unit_size` |
| `generic.cpp:559`, `:605`, `:779-798` | `input_addr` pushed by the `detail` helpers then **back-patched** to `input_buffer` at arg index `grid.x + grid.y` | `TensorBinding{input}` + kernel `TensorAccessor(tensor::input).get_bank_base_address()` (Case 2). The helpers lose their `input_addr` parameter and no longer emit the slot; the back-patch loop is deleted. |
| `generic.cpp:560-562`, `:606-608` | RTA scalars `num_output_pages`, `num_ranges`/`num_blocks`, `output_page_offset` | named RTAs of the same names |
| `generic.cpp:558`, `:604` | `physical_core_coords` prefix (data-selected by index in the kernel) | runtime **varargs** slots `[0, num_x_cores + num_y_cores)` — kept per-node, exactly as legacy, so the kernel's `get_vararg(start_x_index)` / `get_vararg(y_offset + start_y_index)` indexing is unchanged |
| `generic.cpp:564-592`, `:610-645` | trailing per-range / per-block packed words | runtime **varargs** from index `num_x_cores + num_y_cores` — the kernel's `arg_index` initialiser is unchanged |
| `generic.cpp:692-699` | `CBDescriptor{… .buffer = output_buffer}` | `DataflowBufferSpec{… .borrowed_from = "output"}` |

### Not applicable anywhere
- **Semaphore-ID RTAs** — the op uses no semaphores.
- **Page-size 3rd-argument CTAs/RTAs** — every `TensorAccessor` construction is 2-arg
  (audit: TensorAccessor 3rd arg GREEN, no site fires).
- **`TensorParameter` relaxation** — none (audit).
- **Offset base pointers** — none; every base is clean, offsets are added kernel-side
  (audit gate GREEN).

---

## Applied Patterns

- **Two-toucher DFB → assign 1P+1C (dual-instance work split)** — every DFB in the op except
  `nd_reshard_copy_pages`'s `page` (a genuine locked 1:1 FIFO). Explicitly *not* the
  multi-binding advanced option; explicitly *not* a self-loop, since each two-toucher has two
  distinct KernelSpecs to carry the two roles.
- **Borrowed-memory DFB** (`DataflowBufferSpec::borrowed_from`) — `shard` in same-width and
  same-height, `output_shard` in generic. The backing L1 address refreshes from the
  corresponding `TensorArgument` each enqueue, so no `dfb_run_overrides` entry is needed. The
  borrowed `TensorParameter` gets **no kernel `TensorBinding`** — `program_spec.cpp:533-543`
  counts `borrowed_from` as a use, so a borrowed-only parameter is legal and adds no unused
  accessor to the kernel.
- **Case 2 (raw pointer) tensor binding** — `remote` in same-width/same-height, `input` in
  generic. Bound as a `TensorParameter`; the base is pulled kernel-side with
  `TensorAccessor(tensor::x).get_bank_base_address()`; the existing raw
  `AllocatorBank` / `UnicastEndpoint` walk is left byte-for-byte unchanged (no rewrite to
  `TensorAccessor` page iteration). All three sites are **data-movement** kernels, so the
  bridge is available (no compute-kernel Case 2 anywhere → not blocked).
- **Conditional / optional DFB binding** — same-width `scratch`, bound only when
  `unaligned && local_is_output`. Host binds conditionally, emits
  `compiler_options.defines["UNALIGNED"] = "1"` on both KernelSpecs when `unaligned`, and the
  kernel `#ifdef UNALIGNED`-gates the `DataflowBuffer scratch(dfb::scratch)` construction and
  every expression that reaches it. The `unaligned` CTA disappears in favour of the define.
  The same-width **writer** source uses the same `#ifdef UNALIGNED` gate even though it binds
  no scratch DFB, so the two halves of the templated factory stay symmetric and the flag has
  one mechanism, not two.
- **Runtime-selected kernel source** — `ReshardGenericFactory` picks
  `reshard_reader.cpp` vs `reshard_reader_diff_width.cpp` by page-size equality. Both sources
  convert together; the factory's spec shape is identical for either (same DFB, same tensor
  parameters, same named CTA/RTA sets, same vararg layout).
- **Vararg padding to a uniform per-kernel count** — the legacy per-core RTA tail length varies
  by core (transfer/segment/range count). `KernelAdvancedOptions::num_runtime_varargs` is a
  single per-kernel number and the per-node override is deprecated, so each node's vararg
  vector is zero-padded up to the kernel's maximum. The kernel loops are bounded by the named
  count RTA (`num_reads` / `num_segments` / `num_ranges`), so padding words are never read. The
  peak per-node arg count is unchanged from legacy (the max node already carried the max).
- **`AddRuntimeArgsForNode`** — used to bridge every legacy node-first RTA loop into the
  name-first `runtime_arg_values` table without inverting the loop nesting.

---

## Deferred / Flagged

- **Brief correction (stale cross-op coupling).** The brief's "port-together set" claim — that
  the six shared kernels are co-borrowed by `experimental/quasar/reshard/` and that both ops
  must port as one unit — is **stale**. The Quasar reshard forked its own private kernel copies
  and, further, all five of its factories are *already* on `create_program_artifacts`. Nothing
  outside this op reads the shared paths, so they are modified in place and no Quasar work is
  pulled in. See [Cross-op kernels](#cross-op-kernels).
- **The Quasar reshard is not a usable template.** Beyond the recipe's standing warning, its
  copies are an *older snapshot* of these kernels: `reshard_same_width_writer.cpp` has no
  unaligned path at all (the legacy source here does), its same-width DFB sizing uses
  `unit_size` where the current legacy CB uses `local_unit_size_padded`, and both kernels carry
  leftover `WATCHER_RING_BUFFER_PUSH` debug instrumentation. Values are taken from the current
  legacy sources only.
- **RTA→CRTA candidates (not converted here).** Several named RTAs hold the same value on every
  node and would dispatch more efficiently as CRTAs: same-height `local_stride_bytes`,
  `remote_stride_bytes`, `total_num_sticks`; generic `num_x_cores`-indexed coordinate table
  (a whole-vararg-block candidate for `common_runtime_varargs`). RTA→CRTA changes dispatch
  semantics, so it is left for a separate pass and recorded in the port report.
- **No new structural blockers found during planning.** No GlobalCircularBuffer, no
  `CBDescriptor::address_offset`, no GlobalSemaphore, no CTA varargs, no compute kernels at all
  (every kernel in the op is data-movement), no aliased CBs, no dead CBs.
