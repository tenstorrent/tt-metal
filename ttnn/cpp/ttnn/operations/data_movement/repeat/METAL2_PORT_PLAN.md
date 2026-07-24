# Port Plan — `data_movement/repeat`

Port plan for `ttnn/cpp/ttnn/operations/data_movement/repeat`, ported from the legacy
`ProgramDescriptor` (`create_descriptor`) factory concept to Metal 2.0 (`MetalV2FactoryConcept`,
`create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

Scope: the `ttnn::prim::repeat` / `ttnn::prim::repeat_tile` primitive and its two program factories.
The `ttnn::repeat` host composite in `repeat.cpp` (which calls `view`, `to_layout`, sharded/interleaved
conversions, `zeros`) is out of scope — those are separate ops.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (both factories — each defines `create_descriptor()`
  returning `tt::tt_metal::ProgramDescriptor`).
- Device operation: `RepeatDeviceOperation`, `program_factory_t = std::variant<RepeatProgramFactoryLastDim, RepeatProgramFactoryHigherDim>`.
- Variants (device-op level): two program factories, selected by `select_program_factory` on the
  operation attributes:
  - `m_tile_page_size_bytes > 0` → `RepeatProgramFactoryHigherDim` (tile-native mode)
  - else `m_is_last_dim` → `RepeatProgramFactoryLastDim`
  - else → `RepeatProgramFactoryHigherDim` (row-major mode)
- Custom `compute_program_hash`: none — already the default reflection-based hash (confirmed by grep;
  audit `Custom hash = no`).
- No `override_runtime_arguments`, no `get_dynamic_runtime_args` (confirmed by grep; audit rows `no`).

*(Target Metal 2.0 concept chosen during the audit: `MetalV2FactoryConcept` for both factories, no
op-owned tensors. Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section below.)*

Each factory instantiates a **single** `KernelDescriptor` (a reader that performs the full
read-repeat-write), and selects its kernel **source file** at runtime from the sharding state of the
input/output buffers. So the true port unit per factory is: one reader `KernelSpec` + every kernel
source it can select. All five sources are being converted together.

### Variant: HigherDim factory (`repeat_program_factory_higher_dim.cpp`)

Runtime kernel-source selection (3 sources, one live per program):

| Config (runtime) | Kernel source |
|---|---|
| tile-native (`m_tile_page_size_bytes > 0`) | `device/kernels/repeat_higher_dim_tile.cpp` |
| RM, src or dst sharded | `device/kernels/repeat_higher_dim_rm_sharded.cpp` |
| RM, interleaved (`needs_alignment_cb`) | `device/kernels/repeat_higher_dim_rm_interleaved.cpp` |

#### Kernels
Single `KernelDescriptor` (`reader_desc`), `ReaderConfigDescriptor{}`, `core_ranges = total_core_ranges`
(full compute grid). CTAs shown positionally; the appended `TensorAccessorArgs` (src then dst) supply
the accessor static config (CTA) plus the `RuntimeTensorShape` common runtime args (CRTA).

| source | CTAs (positional) | RTAs (per core, 8) | config |
|---|---|---|---|
| repeat_higher_dim_tile.cpp | `page_size_bytes, src0_cb_index, number_of_lower_pages, number_of_rep_dim_pages` + `TAArgs(src)` + `TAArgs(dst)` | `src_addr, dst_addr, higher_start, higher_end, lower_start, lower_end, repetitions, nop` | ReaderConfigDescriptor |
| repeat_higher_dim_rm_sharded.cpp | same as tile | same | ReaderConfigDescriptor |
| repeat_higher_dim_rm_interleaved.cpp | `page_size_bytes, src0_cb_index, src1_cb_index, number_of_lower_pages, number_of_rep_dim_pages` + `TAArgs(src)` + `TAArgs(dst)` | same | ReaderConfigDescriptor |

Kernel-side CTA names: `original_page_size_bytes`, `dfb_id_in0`/`cb_id_in0` (+ `cb_id_in1` interleaved),
`LOWER_DIMS`, `REP_DIM`. Kernel-side RTA names: `src_addr`, `dst_addr`, `higher_dim_start`,
`higher_dim_end`, `lower_dim_start`, `lower_dim_end`, `repetitions`, `nop`.

RTAs are emitted node-first in a `num_cores_x × num_cores_y` loop. `src_buffer`/`dst_buffer` ride the
`Buffer*` runtime-arg channel (patched on cache hit). Idle cores get all-zero args + `nop=1`. Active
cores split work on the higher dim (`divide_on_higher = number_of_higher_pages > number_of_lower_pages`)
or the lower dim.

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| 0 (src0) | `(READ_ALIGNMENT*2) + page_size_bytes` | total_core_ranges | `datatype_to_dataformat_converter(input.dtype())` | = total_size | (unset) |
| 1 (src1) | `(READ_ALIGNMENT*2) + page_size_bytes` | total_core_ranges | same | = total_size | (unset) |

CB `1` exists **only** in the interleaved-RM config (`needs_alignment_cb = !is_tile_native && !src_sharded && !dst_sharded`), as a write-alignment scratchpad. One page each (`total_size == page_size`).

#### Semaphores
none.

#### Tensor accessors
| host site | originating Tensor | RTA slot (host) |
|---|---|---|
| `TensorAccessor(src_args, src_addr)` in each kernel | input (`tensor_args.input`) | slot 0 (`src_addr`) via `src_buffer` Buffer* |
| `TensorAccessor(dst_args, dst_addr)` in each kernel | output (`tensor_return_value`) | slot 1 (`dst_addr`) via `dst_buffer` Buffer* |

Both consumed through `TensorAccessor` (`s.get_noc_addr`, `noc.async_read(s,…)`, `noc_async_read_sharded(noc, cb_slot, s, …)`) → both **Case 1**. Host builds accessor args with `ArgConfig::RuntimeTensorShape`.

#### Work split
- Driver: hand-rolled loop over the grid; `responsibility_chunk/mod` computed from `number_of_higher_pages` or `number_of_lower_pages` divided by `num_cores_total`.
- num_cores: `num_cores_total = num_cores_x * num_cores_y` (full grid).
- One `KernelDescriptor` over all cores; per-core RTAs carry the page range and an `nop` flag.

### Variant: LastDim factory (`repeat_program_factory_last_dim.cpp`)

Runtime kernel-source selection (2 sources):

| Config (runtime) | Kernel source |
|---|---|
| RM, src or dst sharded | `device/kernels/repeat_last_dim_rm_sharded.cpp` |
| RM, interleaved (`needs_alignment_cb`) | `device/kernels/repeat_last_dim_rm_interleaved.cpp` |

#### Kernels
Single `KernelDescriptor`, `ReaderConfigDescriptor{}`, `core_ranges = total_core_ranges`.

| source | CTAs (positional) | RTAs (per core, 5) | config |
|---|---|---|---|
| repeat_last_dim_rm_sharded.cpp | `source_page_size_bytes, num_repeats, src0_cb_index` + `TAArgs(src)` + `TAArgs(dst)` | `src_addr, dst_addr, page_start, page_end, nop` | ReaderConfigDescriptor |
| repeat_last_dim_rm_interleaved.cpp | `source_page_size_bytes, num_repeats, src0_cb_index, src1_cb_index` + `TAArgs(src)` + `TAArgs(dst)` | same | ReaderConfigDescriptor |

Kernel-side CTA names: `original_page_size_bytes`, `num_repeats`, `dfb_id_in0`/`cb_id_in0` (+ `cb_id_in1` interleaved). Kernel-side RTA names: `src_addr`, `dst_addr`, `page_start`, `page_end`, `nop`.

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| 0 (src0) | `cb_size_bytes` (see note) | total_core_ranges | `datatype_to_dataformat_converter(input.dtype())` | = total_size | (unset) |
| 1 (src1) | `cb_size_bytes` | total_core_ranges | same | = total_size | (unset) |

`cb_size_bytes` is the operator-precedence expression at `repeat_program_factory_last_dim.cpp:53-57`.
The audit's Misc-anomalies section flags it as a pre-existing host-logic defect (routed to the ops
team). **The port carries the expression verbatim into `DataflowBufferSpec::entry_size` — zero
functional change. Do NOT "fix" it here.** CB `1` exists only in the interleaved-RM config.

#### Semaphores
none.

#### Tensor accessors
Same shape as HigherDim: input → Case 1, output → Case 1, both via `TensorAccessor`, host RTA slots 0/1
via `src_buffer`/`dst_buffer` Buffer*.

#### Work split
- Driver: hand-rolled grid loop; `responsibility = ((number_of_pages - 1) / num_cores_total) + 1`, `number_of_pages = input_log_shape[-2]`.
- One `KernelDescriptor` over all cores; per-core RTAs carry `page_start/page_end` and `nop`.

### Cross-op kernels
none. All five kernel `.cpp` sources live in the op's own `device/kernels/` directory.

The kernels `#include "ttnn/operations/data_movement/common/kernels/common.hpp"` (an **in-family** shared
header, same `data_movement` family, already Device 2.0) for `tt_memmove`, `noc_async_read_sharded`,
`noc_async_write_sharded`, `align_address`, and the `MASK_*` / `OFFSET_*` alignment constants. This is a
function-call escape, not a cross-op kernel *file*; the header takes `Noc`, a raw CB-slot address, and a
`TensorAccessor` by value, so the Metal 2.0 tokens pass through without touching the header. Not modified
by this port.

### Flags
- No unreferenced kernel files (all five are live).
- No `GlobalCircularBuffer` / remote-CB idiom, no `CBDescriptor.address_offset`, no aliased CBs
  (single-element `format_descriptors`), no varargs, no semaphores. (Matches audit Appendix-A scan.)

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MetalV2FactoryConcept` (both factories).
- **Custom `compute_program_hash`**: none — already default reflection-based hash. No deletion forced.
- **Pybind `create_descriptor`**: none. `repeat_nanobind.cpp` binds only the `ttnn::repeat` free function
  (no `nb::class_` of the device op, no `create_descriptor` binding). So the disappearance of
  `create_descriptor` forces **no** pybind cleanup — the factory method is renamed
  `create_descriptor` → `create_program_artifacts` in the `.hpp`/`.cpp` only.
- **Implementation notes**:
  - `data_movement` is a **unity build** (`TT_ENABLE_UNITY_BUILD(ttnn_op_data_movement)`), and both
    factory `.cpp` files land in one translation unit. To avoid anonymous-namespace symbol collisions,
    declare all Metal 2.0 named constants (`KernelSpecName` / `DFBSpecName` / `TensorParamName`) as
    **function-local** constants inside each `create_program_artifacts` body (no external linkage). See
    [Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols).
  - The device-operation class (`repeat_device_operation.*`) needs **no** edits: the framework dispatches
    per-factory by concept, so both factories flipping to `create_program_artifacts` is sufficient.

## Planned Spec Shape

Default 1:1 with legacy. Because each factory runtime-selects its kernel source, the single reader
`KernelSpec`'s `.source`, `.compile_time_args`, and `.dfb_bindings` are chosen inside
`create_program_artifacts` by the same config branch the legacy factory used; the rest of the spec is
shared across configs.

### HigherDim
- **KernelSpecs**: one — `READER{"reader"}`. `.source` = the selected kernel path; `.hw_config =
  create_reader_datamovement_config(device->arch())` (legacy `ReaderConfigDescriptor{}` = reader default
  RISCV_1/NOC_0/DM_DEDICATED_NOC); `.tensor_bindings` = {INPUT→"src", OUTPUT→"dst"}; `.dfb_bindings` = SRC0
  self-loop (+ SRC1 self-loop in the interleaved config); named CTAs and RTA schema per config.
- **DataflowBufferSpecs**: `SRC0{"src0"}` (all configs) + `SRC1{"src1"}` (interleaved-RM only). Each:
  `entry_size = cb_size_bytes`, `num_entries = 1`, `data_format_metadata = cb_data_format`,
  `tile_format_metadata` unset (legacy `.tile` unset).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT{"input"}` (`input.tensor_spec()`), `OUTPUT{"output"}` (`output.tensor_spec()`).
- **WorkUnitSpecs**: one — `{name="main", kernels={READER}, target_nodes=total_core_ranges}`.

### LastDim
- **KernelSpecs**: one — `READER{"reader"}`. Same shape; `.source` = the selected last-dim kernel;
  `.dfb_bindings` = SRC0 self-loop (+ SRC1 self-loop in interleaved); CTAs `original_page_size_bytes`,
  `num_repeats`; RTA schema `page_start`, `page_end`, `nop`.
- **DataflowBufferSpecs**: `SRC0` (+ `SRC1` interleaved-RM only). `entry_size = cb_size_bytes`,
  `num_entries = 1`, `data_format_metadata = cb_data_format`, `tile_format_metadata` unset.
- **SemaphoreSpecs**: none.
- **TensorParameters**: INPUT, OUTPUT (as above).
- **WorkUnitSpecs**: one — `{name="main", kernels={READER}, target_nodes=total_core_ranges}`.

## Preserved Multiplicity

none — no work-split multiplicity in legacy. Each factory has a **single** `KernelDescriptor`; the work
split is expressed through per-core runtime args on that one kernel over the full grid, not through
multiple `KernelDescriptor`s of the same source. So one `KernelSpec` per factory, no multi-`KernelSpec`
mapping, no `allow_instance_multi_binding`.

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| both factories, RTA slot 0 (`src_addr`) | `src_buffer` (`Buffer*`) in `emplace_runtime_args` | `TensorParameter INPUT` + `TensorBinding`(accessor "src"); kernel `TensorAccessor(tensor::src)` |
| both factories, RTA slot 1 (`dst_addr`) | `dst_buffer` (`Buffer*`) | `TensorParameter OUTPUT` + `TensorBinding`(accessor "dst"); kernel `TensorAccessor(tensor::dst)` |
| CTA `src0_cb_index` (all sources) | `= 0` positional CTA | `DataflowBufferSpec SRC0` + `DFBBinding`(accessor "in0", self-loop P+C) |
| CTA `src1_cb_index` (interleaved sources) | `= 1` positional CTA | `DataflowBufferSpec SRC1` + `DFBBinding`(accessor "in1", self-loop P+C) |
| both factories, host `TensorAccessorArgs(*src_buffer, RuntimeTensorShape).append_to(cta, crta)` | accessor static CTAs + RuntimeTensorShape CRTAs | subsumed by INPUT `TensorBinding` (both CTA and CRTA halves drop) |
| both factories, host `TensorAccessorArgs(*dst_buffer, RuntimeTensorShape).append_to(cta, crta)` | same | subsumed by OUTPUT `TensorBinding` |
| all kernels, `TensorAccessorArgs<N,M>()` + `src_addr`/`dst_addr` reads | manual offset chaining + buffer-addr RTA | `TensorAccessor(tensor::src)` / `(tensor::dst)` |
| all CTAs | positional `get_compile_time_arg_val(i)` | named `get_arg(args::name)` (`original_page_size_bytes`, `num_repeats`, `LOWER_DIMS`, `REP_DIM`) |
| all RTAs | positional `get_arg_val<uint32_t>(i)` | named `get_arg(args::name)` (per-dim bounds, `repetitions`/`page_*`, `nop`) |

Page-size 3rd-argument CTAs/RTAs: none present (all `TensorAccessor` sites are 2-arg). Semaphore-ID RTAs:
none.

Note on kernel `src_args.is_dram`: the two interleaved kernels read a compile-time `is_dram` off the
dropped `TensorAccessorArgs`. Post-drop, this is queried off the accessor object:
`src_args.is_dram` → `decltype(s)::is_dram` (the `TensorAccessor`'s `static constexpr bool is_dram`).

## Applied Patterns

- **[Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding)** /
  **[Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)**:
  every DFB (SRC0, and SRC1 in the interleaved configs) is a single-toucher scratchpad — the reader
  reserves/`get_write_ptr`/pushes and then uses the slot as local SRAM (no `wait_front`/`pop_front`). Bind
  the reader as both PRODUCER and CONSUMER, one shared accessor name (`in0` / `in1`). Legal on Gen1 for a
  DM kernel (DFB lowers to a plain CB one RISC fills and drains).
- **Runtime kernel-source selection**: each factory chooses its kernel source (and CTA / DFB set) at
  runtime from the sharding state; all selectable sources convert together (the factory does not build
  until every source it can bind is on Metal 2.0).
- **[Multi-variant factories](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories)**:
  the device-op holds two factories (LastDim, HigherDim); both are ported (each builds its own
  `ProgramSpec`/`ProgramRunArgs` in `create_program_artifacts`).
- **[Pass DFB handles / raw pointers to kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)**:
  the sharded kernels pass the DFB write pointer (a raw SRAM address from `get_write_ptr`) into
  `noc_async_read_sharded` / `noc_async_write_sharded`, and the `TensorAccessor` by value — both flow
  through unchanged.
- **[Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)**:
  named constants declared function-local to dodge cross-TU collisions in the unity build.

## Deferred / Flagged

- **`ArgConfig::RuntimeTensorShape` vs. `dynamic_tensor_shape` relaxation.** The legacy host builds its
  accessor args with `tensor_accessor::ArgConfig::RuntimeTensorShape` on both io tensors. The migration
  guide's pre-flight check links this ArgConfig to `TensorParameter::advanced_options.dynamic_tensor_shape
  = true`. The audit, however, records **TensorParameter relaxation = none**, and the recipe/ttnn_factory
  bias is toward the **strict default** during a port (relaxing incorrectly is a wrong-answer bug; not
  relaxing is merely narrower cache equivalence, still correct). The strict default is provably correct
  here: TTNN keys its program cache on the tensor spec (so each distinct spec re-runs the factory and
  bakes the right shape into the kernel's CTAs), and the repeat kernels compute offsets/transfer sizes
  from their own `original_page_size_bytes` CTA rather than the accessor's dynamic page size (they never
  call `get_aligned_page_size()`). Decision: **follow the audit — no relaxation.** Recorded as friction +
  a relaxation candidate for downstream in `METAL2_PORT_REPORT.md`.
- **Operator-precedence defect in `cb_size_bytes`** (`repeat_program_factory_last_dim.cpp:53-57`): carried
  verbatim into `DataflowBufferSpec::entry_size`, zero functional change. Pre-existing host-logic issue
  routed to the ops team by the audit; not touched by the port.
