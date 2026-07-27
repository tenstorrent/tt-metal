# Port Plan — `examples/example` (`ExampleDeviceOperation`)

Port plan for `ttnn/cpp/ttnn/operations/examples/example`, ported from the `descriptor`
(`ProgramDescriptorFactoryConcept`, `create_descriptor()` → `ProgramDescriptor`) API to Metal 2.0
(`MetalV2FactoryConcept`, `create_program_artifacts()` → `ProgramArtifacts`).

Written during the inventory and planning steps; committed alongside the port for review.

> **Outcome: the port CAPITULATED** before any code was written — see `METAL2_PORT_REPORT.md`.
> The blocker is structural (all three bound kernels live outside the op directory and cannot be
> touched or forked under the orchestration scope constraint), not a defect in the spec design.
> This plan is therefore complete and correct as a *design*: it is the actionable blueprint for the
> next porter to finish the factory port in one mechanical pass **once the cross-op shared-kernel
> rewrite lands** (see the Deferred / Flagged section and the report's Handoff points).

## Legacy Inventory

### Legacy factory shape
- Concept: `descriptor` (`ProgramDescriptorFactoryConcept`) — each factory defines
  `static ProgramDescriptor create_descriptor(operation_attributes_t, tensor_args_t, tensor_return_value_t&)`.
- Variants: two — `SingleCore` (`device/single_core_program_factory.cpp`) and
  `MultiCore` (`device/multi_core_program_factory.cpp`). Selected by
  `select_program_factory` on `operation_attributes.attribute` (`device/example_device_operation.cpp:11-17`).
  The two factories are structurally identical (same two CBs, same three kernels, same per-core RTA loop)
  and differ only in the compute grid: `SingleCore` fixes `{1,1}`; `MultiCore` uses
  `device->compute_with_storage_grid_size()`. They share the same kernel sources, so they port together.
- Custom `compute_program_hash`: none — the device-op uses the default reflection-based hash
  (no override in `device/example_device_operation.{hpp,cpp}`). No deletion required.

*(Target concept `MetalV2FactoryConcept` inherited from the audit — see [TTNN ProgramFactory](#ttnn-programfactory).)*

### Kernels

Both factories emit the same three `KernelDescriptor`s (only `compute.core_ranges` differs between variants).
All three `kernel_source` paths are **outside the op directory** (cross-op, borrowed from `eltwise/unary`) — see [Cross-op kernels](#cross-op-kernels).

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | `all_cores` | `TensorAccessorArgs(*src_buffer).append_to(...)` (accessor plumbing at slot 0+) | none | `{src_buffer (Buffer*), num_tiles_per_core, num_tiles_written}` | none | none | `ReaderConfigDescriptor{}` (reader default) |
| writer | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `all_cores` | `{output_cb_index}` (=`CBIndex::c_2`=2) at slot 0, then `TensorAccessorArgs(*dst_buffer).append_to(...)` at slot 1+ | none | `{dst_buffer (Buffer*), num_tiles_per_core, num_tiles_written}` | none | none | `WriterConfigDescriptor{}` (writer default) |
| compute | `eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | `core_group_1` (SingleCore) / `all_cores` (MultiCore) | none | none | `{num_tiles_per_core}` | none | none | `ComputeConfigDescriptor{.math_fidelity = HiFi4, .math_approx_mode = false}` |

CTA slot positions read from the host factory's emission order (authoritative), `single_core_program_factory.cpp:66-97` / `multi_core_program_factory.cpp:64-95`.

Kernel-side arg reads (current, positional — these are the reads that MUST flip to named bindings for a Metal 2.0 factory to drive them):
- reader: `get_arg_val<uint32_t>(0..2)` (src_addr/num_pages/start_id), `TensorAccessorArgs<0>()`, `constexpr uint32_t cb_id_in0 = 0`.
- writer: `get_arg_val<uint32_t>(0..2)` (dst_addr/num_pages/start_id), `get_compile_time_arg_val(0)` (cb_id_out), `TensorAccessorArgs<1>()`.
- compute: `get_arg_val<uint32_t>(0)` (num_tiles), `constexpr auto cb_input = c_0`, `cb_output = c_2`.

### CBs

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` (`src0_cb_index`=0, input) | `2 * single_tile_size` | `all_cores` | `datatype_to_dataformat_converter(input_tensor.dtype())` | `single_tile_size` | not set |
| `c_2` (`output_cb_index`=2, output) | `2 * single_tile_size_output` | `all_cores` | `datatype_to_dataformat_converter(output_tensor.dtype())` | `single_tile_size_output` | not set |

No `GlobalCircularBuffer`, no `.global_circular_buffer`, no `.address_offset` set. Plain CBs only.
Census (from audit, re-verified): `c_0` = reader PRODUCES / compute CONSUMES (1P+1C); `c_2` = compute PRODUCES / writer CONSUMES (1P+1C). Both on every active node in both configs.

### Semaphores
none — the op uses no semaphores.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| reader `TensorAccessor(src_args, src_addr)` (`reader_unary_interleaved_start_id.cpp:25`); host `TensorAccessorArgs(*src_buffer).append_to(reader_cta)` (`single_core_program_factory.cpp:67`, `multi_core_program_factory.cpp:65`) | `input_tensor` (input) | reader RTA slot 0 (`src_buffer`) |
| writer `TensorAccessor(dst_args, dst_addr)` (`writer_unary_interleaved_start_id.cpp:31`); host `TensorAccessorArgs(*dst_buffer).append_to(writer_cta)` (`single_core_program_factory.cpp:79`, `multi_core_program_factory.cpp:77`) | `output_tensor` (output) | writer RTA slot 0 (`dst_buffer`) |

Both **Case 1** (accessed only through `TensorAccessor` — no raw base-pointer arithmetic). No 3rd (page-size) accessor argument at either site. No `TensorParameter` relaxation.

### Work split
- Driver: `split_work_to_cores(compute_with_storage_grid_size, num_tiles)` where `num_tiles = input_tensor.physical_volume() / TILE_HW`.
  - `SingleCore`: grid fixed `{1,1}` (`single_core_program_factory.cpp:31`).
  - `MultiCore`: grid = `device->compute_with_storage_grid_size()` (`multi_core_program_factory.cpp:32`).
- Returns `(num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2)`.
- Per-core loop assigns `num_tiles_per_core` (group 1 or group 2) and a running `num_tiles_written`, emitted as RTAs to all three kernels. **Single `KernelDescriptor` per source** — no per-group CTA multiplicity (work split is expressed purely through per-core RTAs), so no `KernelSpec` multiplicity to preserve.

### Cross-op kernels
**All three bound kernels are cross-op** (live outside the op directory), borrowed by repo-root file path from `eltwise/unary`:
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` — broadly shared (**19** consumer ops).
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — broadly shared (**48** consumer ops).
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` — shared within eltwise/unary (**4** consumer ops).

Each is a [Caution: Modifying a shared dataflow kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-modifying-a-shared-dataflow-kernel) case. **This is the port blocker** — see Deferred / Flagged and the report.

### Flags
- **Unreferenced dead kernel files in the op directory** (not audited, not touched): `device/kernels/compute/eltwise_sfpu.cpp`, `device/kernels/dataflow/{blank,reader_binary_diff_lengths,reader_unary,writer_unary}.cpp`. Both factories point at the `eltwise/unary` paths, never these. Leftover tutorial scaffolding — left as-is (cleanup is out of scope, routed to report).
- **Unused attribute**: `operation_attributes_t::some_other_attribute` (`device/example_device_operation.hpp:22`) is set to `42` (`device/example_device_operation.cpp:42`) but never read. Dead; not hashed. Left as-is (out of scope, routed to report).

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MetalV2FactoryConcept` (op has no op-owned tensors, no GlobalSemaphores, single program per factory).
- **Custom `compute_program_hash`**: none — no deletion required.
- **Implementation notes**: `create_descriptor` → `create_program_artifacts` on both `SingleCore` and `MultiCore`;
  update the `program_factory_t` variant members' method signature accordingly, and drop the
  `<tt-metalium/program_descriptors.hpp>` include in favor of the Metal 2.0 spec headers.
  No pybind entry point to remove (`example_nanobind.cpp` binds only the free function `composite_example`,
  not `create_descriptor`). `select_program_factory` / `validate_*` / `compute_output_specs` /
  `create_output_tensors` are unchanged (out of the factory-body scope).

## Planned Spec Shape

Default 1:1 with legacy. Same shape for both `SingleCore` and `MultiCore` (only grid selection differs).

- **KernelSpecs** (3, 1:1 with the legacy `KernelDescriptor`s):
  - `READER` — source = reader kernel; `dfb_bindings = {INPUT as CONSUMER... }` — **wait**: reader PRODUCES into `c_0`. Binding: `{.dfb_spec_name = INPUT, .accessor_name = "in0", .endpoint_type = PRODUCER}`; `tensor_bindings = {INPUT}`; RTAs `num_pages`, `start_id` (named); `hw_config = create_reader_datamovement_config(device->arch())`.
  - `WRITER` — source = writer kernel; `dfb_bindings = {{.dfb_spec_name = OUTPUT, .accessor_name = "out0", .endpoint_type = CONSUMER}}`; `tensor_bindings = {OUTPUT}`; RTAs `num_pages`, `start_id` (named); `hw_config = create_writer_datamovement_config(device->arch())`.
  - `COMPUTE` — source = compute kernel; `dfb_bindings = {{INPUT, "in", CONSUMER}, {OUTPUT, "out", PRODUCER}}`; RTA `num_tiles` (named); `hw_config` = `ComputeGen1Config` built directly (Style B — see [hw_config](#hardware-configuration) below).
- **DataflowBufferSpecs** (2, 1:1 with legacy CBs):
  - `INPUT` — `entry_size = single_tile_size`, `num_entries = 2`, `data_format_metadata` from input dtype. 1P (reader) + 1C (compute).
  - `OUTPUT` — `entry_size = single_tile_size_output`, `num_entries = 2`, `data_format_metadata` from output dtype. 1P (compute) + 1C (writer).
  - No borrowed-memory, no aliasing, no self-loop, no multi-binding flag, no dead CB. (Endpoint census re-derived and agrees with the brief.)
- **SemaphoreSpecs**: none.
- **TensorParameters** (2): `INPUT` (from `input_tensor.tensor_spec()`), `OUTPUT` (from `output_tensor.tensor_spec()`). Each bound with one `TensorArgument` (`{INPUT, input_tensor}`, `{OUTPUT, output_tensor}`) — direct `MeshTensor` ref, no `std::cref`.
- **WorkUnitSpecs**: `MultiCore` → one (all three kernels on `all_cores`). `SingleCore` → reader/writer target `all_cores`, compute targets `core_group_1`; with the `{1,1}` grid these coincide to the single active node, so effectively one work unit (bind kernels per their `core_ranges`; if `core_group_1 != all_cores` for some tile count, split into per-node-set work units accordingly).
- **Op-owned tensors**: none.

## Preserved Multiplicity
none — no work-split multiplicity in legacy (single `KernelDescriptor` per source; work split expressed through per-core RTAs, not per-group CTAs).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA slot 0 (`single_core_program_factory.cpp:111`, `multi_core_program_factory.cpp:109`) | `src_buffer` (`Buffer*`), consumed kernel-side as `get_arg_val<uint32_t>(0)` → `TensorAccessor(src_args, src_addr)` | `TensorParameter INPUT` + `TensorBinding` in `READER`; kernel `TensorAccessor(tensor::input)` |
| reader CTA (host `TensorAccessorArgs(*src_buffer).append_to(reader_cta)`, `:67`/`:65`; kernel `TensorAccessorArgs<0>()`) | accessor CTA plumbing | dropped — supplied by the binding token |
| writer RTA slot 0 (`:113`/`:111`) | `dst_buffer` (`Buffer*`) → `get_arg_val<uint32_t>(0)` → `TensorAccessor(dst_args, dst_addr)` | `TensorParameter OUTPUT` + `TensorBinding` in `WRITER`; kernel `TensorAccessor(tensor::output)` |
| writer CTA slot 0 (`writer_compile_time_args = {output_cb_index}`, `:78`/`:76`; kernel `get_compile_time_arg_val(0)`) | magic CB index `CBIndex::c_2` (2) | `DFBBinding{OUTPUT, "out0", CONSUMER}` in `WRITER`; kernel `DataflowBuffer dfb(dfb::out)` |
| writer CTA (host `TensorAccessorArgs(*dst_buffer).append_to(...)`, `:79`/`:77`; kernel `TensorAccessorArgs<1>()`) | accessor CTA plumbing | dropped — supplied by the binding token |
| reader/writer RTA slots 1,2; compute RTA slot 0 | positional `num_tiles_per_core`, `num_tiles_written`, `num_tiles` | named RTAs `num_pages`, `start_id` (reader/writer), `num_tiles` (compute), via `AddRuntimeArgsForNode` over the existing per-core loop |
| compute kernel `constexpr auto cb_input=c_0`, `cb_output=c_2` | hardcoded CB indices at LLK call sites (`init_sfpu`, `copy_tile`, `pack_tile`) | `dfb::in` / `dfb::out` handles passed directly |

Page-size 3rd-arg CTAs/RTAs: none. Semaphore-ID RTAs: none.

## Applied Patterns
- [Caution: Modifying a shared dataflow kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-modifying-a-shared-dataflow-kernel) — **all three** bound kernels are cross-op (in `eltwise/unary`). This is the blocking pattern.
- [Pass DFB handles directly to LLKs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers) — compute kernel would pass `dfb::in`/`dfb::out` into `init_sfpu`/`copy_tile`/`pack_tile`.
- Both DFBs are plain 1P+1C — no self-loop, two-toucher, or multi-binding pattern.

## Hardware configuration (planned values — verify against legacy)
- **reader** `ReaderConfigDescriptor{}` → reader default `(RISCV_1, NOC_0, DM_DEDICATED_NOC)` → `create_reader_datamovement_config(device->arch())`.
- **writer** `WriterConfigDescriptor{}` → writer default `(RISCV_0, NOC_1, DM_DEDICATED_NOC)` → `create_writer_datamovement_config(device->arch())`.
- **compute** `ComputeConfigDescriptor{.math_fidelity = HiFi4, .math_approx_mode = false}` — this is **Style B** (Metal `ComputeConfig` set directly; no TTNN `ComputeKernelConfig` feeding it). Build `ComputeGen1Config{ .fpu_math_fidelity = MathFidelity::HiFi4, .sfpu_precision_mode = Precision::Precise /* math_approx_mode=false */ }`. All other fields left at Metal defaults (match legacy). `fp32_dest_acc_en` not set → not Float32-dest → **no forced `unpack_modes` entry**. `bfp_pack_precision_mode` untouched (default). `dst_full_sync_en` unset → `double_buffer_dest` default. No arch branch; Gen1 only.

## Deferred / Flagged

- **PORT BLOCKER (capitulation) — cross-op shared kernels, out of writeable scope.** Both factories bind three kernels that live entirely **outside the op directory**, in `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/`. The port's atomic unit is the factory *plus every kernel entry point it binds*: a `MetalV2FactoryConcept` factory emits only named bindings and can only launch kernels that read `dfb::`/`tensor::`/`args::`. The three shared kernels are Device 2.0 but **not** Metal 2.0 (they still read positional `get_arg_val`/`get_compile_time_arg_val` and use `TensorAccessorArgs<N>()`). Making them Metal 2.0 requires editing them — but:
  - The orchestration constraint restricts the writeable surface to files **under the op directory** and explicitly forbids editing shared kernels outside it.
  - The brief and the orchestration constraint both forbid **forking** copies into the op directory.
  - No already-rewritten (`_metal2`) versions exist on this branch to consume.

  Both recipe-sanctioned paths for a cross-op kernel ([in-place co-migration] / [`_metal2` fork]) are closed by the overriding scope constraint, so there is **no in-scope way to produce Metal 2.0 kernel entry points** for the factory to bind. This is precisely the recipe's most-common stop signal ("reaching past the op's own directory to make kernel changes"). Per [§When the discipline doesn't fit], the port capitulates. See `METAL2_PORT_REPORT.md` → Handoff points. The factory `.cpp` files are left **unmodified** — a half-converted factory that binds unconverted kernels does not build and is not a deliverable.
- No structural issue beyond the blocker; the spec design above is complete and mechanical to apply once the shared-kernel rewrite lands.
