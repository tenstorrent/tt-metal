# Port Plan — clone

Port plan for `data_movement/clone`, ported from the `ProgramDescriptor` (`create_descriptor`) API to Metal 2.0 (`MetalV2FactoryConcept` / `create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `create_descriptor` sits **directly on `CloneOperation`** (`HasDirectDescriptor`; no `program_factory_t` wrapper). Returns `tt::tt_metal::ProgramDescriptor` (`clone_device_operation.hpp:29`).
- Variants: single `create_descriptor` (`clone_program_factory.cpp:18`) that fans out at build time into **four code-paths** — {tilized, row-major} × {interleaved, sharded} — plus an optional compute kernel when `convert_dtype` (input/output data formats differ; only reachable in tile layout).
- Custom `compute_program_hash`: none — already default reflection-based hash (audit confirmed grep-clean).

*(Target concept chosen by the audit: `MetalV2FactoryConcept`. Carried forward below.)*

### Kernels
One `ProgramDescriptor` selects kernel **source files** by config. Reader/Writer are one `KernelDescriptor` each over `all_cores`; the compute kernel (only when `convert_dtype`) is split per core-group.

| unique_id | source (per config) | core_ranges | CTAs (positional) | RTAs (positional) | config |
|---|---|---|---|---|---|
| reader | tilized-int `read_kernel.cpp` / rm-int `read_kernel_rm.cpp` / tilized-shard `read_kernel_sharded.cpp` / rm-shard `read_kernel_rm_sharded.cpp` | `all_cores` | int-tilized `{src_cb_id}`+`TensorAccessorArgs<1>`; int-rm `{src_cb_id, input_unit_size}`+`TensorAccessorArgs<2>` (**CTA idx 1 dead** — see Flags); shard-tilized `{src_cb_id}`; shard-rm `{src_cb_id}` | int-tilized `{in_addr, num_tiles, start_id}`; int-rm `{in_addr, stick_size, num_sticks, start_id}`; shard-tilized `{in_addr, num_tiles}`; shard-rm `{in_addr, stick_size, num_sticks}` | `ReaderConfigDescriptor{}` (reader default: RISCV_1 / NOC_0 / DEDICATED) |
| writer | tilized-int `write_kernel.cpp` / rm-int `write_kernel_rm.cpp` / tilized-shard `write_kernel_sharded.cpp` / rm-shard `write_kernel_rm_sharded.cpp` | `all_cores` | mirror of reader with `dst_cb_id`/`output_unit_size` | mirror of reader with `out_addr`/`output_unit_size` | `WriterConfigDescriptor{}` (writer default: RISCV_0 / NOC_1 / DEDICATED) |
| compute_g1 / compute_g2 (only if `convert_dtype`) | `compute_kernel.cpp` | `core_group_1` / `core_group_2` | `{src_cb_id, dst_cb_id, num_units_per_core_group_N}` | none | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` from `get_compute_kernel_config_args(arch, attrs.compute_kernel_config)` |

Reader/writer use `get_tile_size(cb_id)` for the transfer size (tilized-int & sharded paths). Interleaved paths use `TensorAccessor(args, addr)`; sharded paths use raw `noc.async_read/write(UnicastEndpoint{}, …, {.addr = local_l1_read/write_addr})` with `local_l1_*_addr += tile_size/stick_size` (no `TensorAccessor`).

### CBs
| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| `src_cb` = `c_4` | `2 * aligned_input_unit_size` | `all_cores` | `input_data_format` | `aligned_input_unit_size` | (unset → default 32×32) |
| `dst_cb` = `c_20` (only if `convert_dtype`) | `2 * aligned_output_unit_size` | `all_cores` | `output_data_format` | `aligned_output_unit_size` | (unset) |

When `!convert_dtype`, `dst_cb_id == src_cb_id` (`c_4`) — the reader produces and the writer consumes the single CB. No `GlobalCircularBuffer`, no aliasing, no `address_offset`. `alignment = input.buffer()->alignment()`.

### Semaphores
none.

### Tensor accessors
| host site | originating Tensor | RTA slot (host) |
|---|---|---|
| `read_kernel.cpp:20` / `read_kernel_rm.cpp:21` — `TensorAccessor(args, in_addr)` | `input` (interleaved) | reader RTA slot 0 (`input_buffer`) |
| `write_kernel.cpp:20` / `write_kernel_rm.cpp:21` — `TensorAccessor(args, out_addr)` | `output` (interleaved) | writer RTA slot 0 (`output_buffer`) |
| sharded kernels — no `TensorAccessor`; raw base addr used directly (`.addr`) | `input` / `output` (sharded, **Case 2**) | reader/writer RTA slot 0 |

Delivery is the `Buffer*`-binding form (`emplace_runtime_args(core, {input_buffer, …})`, `clone_program_factory.cpp:231-243`) — the interim `BufferBinding` path, not `->address()`-in-RTA.

### Work split
- Interleaved: `split_work_to_cores(compute_with_storage_grid_size, num_units)` → `(num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2)`.
- Sharded: `all_cores = shard_spec.grid()`, `core_group_1 = all_cores`, `core_group_2 = {}`, `num_units_per_core_group_1 = shard tiles (tilized) / shard_height (rm)`.
- Reader/writer carry `num_units_per_core` as an **RTA** (per-node, differs group_1 vs group_2). Compute carries `num_units_per_core_group_N` as a **CTA** (per-group → preserved multiplicity).

### Cross-op kernels
none — clone owns all nine kernel files under `device/kernels/`. No shared/donor kernels, no out-of-directory `#include`s (all resolve to `tt_metal/hw/inc/api/*`).

### Flags
- **Dead CTA on row-major *interleaved* paths** (audit misc-anomaly): `read_kernel_rm.cpp` / `write_kernel_rm.cpp` declare `TensorAccessorArgs<2>()` but the stick size is read from RTA idx 1, so CTA idx 1 (`input_unit_size` / `output_unit_size`) is never read. It vanishes automatically in the port (the whole positional CTA list is replaced by named bindings; the dead value has no consumer to name). The **RTA** copy (`stick_size`) survives and drives the kernel. Not a separate fix.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: The op currently satisfies `HasDirectDescriptor` (`create_descriptor` on `CloneOperation`, no `program_factory_t`). The port introduces a `CloneProgramFactory` struct with `create_program_artifacts` and adds `using program_factory_t = std::variant<CloneProgramFactory>;` to `CloneOperation`, removing the direct `create_descriptor` declaration. This is the concept-migration wiring the port forces (per `ttnn_factory.md`), not a freelance device-op edit. Everything else on the device-op class (`validate_inputs`, `compute_output_specs`, `create_output_tensors`, perf model, `ttnn::prim::clone`) is untouched.

## Planned Spec Shape

Single factory, build-time branch on `is_sharded` × `tilized` (kernel source) and `convert_dtype` (compute presence + DFB count). One `ProgramSpec` per invocation.

- **KernelSpecs**:
  - `READER` (source per config), `WRITER` (source per config) — always.
  - `COMPUTE_G1` (+ `COMPUTE_G2` if `core_group_2` nonempty) of `compute_kernel.cpp` — only when `convert_dtype`. **Preserved multiplicity** (per-group CTA `num_tiles`).
- **DataflowBufferSpecs**:
  - `SRC` (`entry_size = aligned_input_unit_size`, `num_entries = 2`, `data_format_metadata = input_data_format`) — always.
  - `DST` (`entry_size = aligned_output_unit_size`, `num_entries = 2`, `data_format_metadata = output_data_format`) — only when `convert_dtype`.
  - `tile_format_metadata` unset (legacy `CBFormatDescriptor::tile` unset → default 32×32).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT` (`input.tensor_spec()`, bound by reader), `OUTPUT` (`output.tensor_spec()`, bound by writer).
- **WorkUnitSpecs**:
  - `convert_dtype`: `WU_G1 = {READER, WRITER, COMPUTE_G1}` on `core_group_1`; if `core_group_2` nonempty, `WU_G2 = {READER, WRITER, COMPUTE_G2}` on `core_group_2`. (Reader/writer in both → node coverage = `all_cores`; compute groups node-disjoint. Matches the documented multi-group work-split shape.)
  - `!convert_dtype`: `WU_MAIN = {READER, WRITER}` on `all_cores`.

### DFB bindings (accessor names → generated `dfb::` tokens)
- `READER`: `SRC` PRODUCER, accessor `src` (`dfb::src`).
- `WRITER`: CONSUMER, accessor `dst` (`dfb::dst`) → binds `DST` when `convert_dtype`, else `SRC`. (Writer kernel is identical across both — it always says `dfb::dst`.)
- `COMPUTE_Gn`: `SRC` CONSUMER accessor `src`; `DST` PRODUCER accessor `dst`.

Per-node census (validator): `!convert_dtype` — SRC: reader(P)+writer(C) = 1P+1C. `convert_dtype` — SRC: reader(P)+compute(C); DST: compute(P)+writer(C) = 1P+1C on every node (compute_g1/g2 node-disjoint, so each node sees exactly one compute instance). All legal, no self-loop / multi-binding / dead-CB disposition needed (audit agrees).

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| `compute_desc_1` (core_group_1) / `compute_desc_2` (core_group_2) of `compute_kernel.cpp`, differing on `num_units_per_core_group_N` CTA | `COMPUTE_G1` / `COMPUTE_G2` of `compute_kernel.cpp`, each with its own `num_tiles` CTA | `WU_G1` / `WU_G2` (node-disjoint) | `SRC` (both CONSUMER), `DST` (both PRODUCER) — legal: disjoint node sets, one instance per node |

Reader/writer have **no** work-split multiplicity (single KernelSpec each; per-node work count carried as an RTA, not a per-group CTA). Only the compute kernel splits.

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader/writer CTA slot 0 (all configs) | `src_cb_id` / `dst_cb_id` (magic CB index) | `DFBBinding` (`dfb::src` / `dfb::dst`) |
| reader/writer RTA slot 0 (all configs) | `input_buffer` / `output_buffer` (`BufferBinding`) | `TensorBinding` (`tensor::input` / `tensor::output`); interleaved → `TensorAccessor(tensor::name)`, sharded → Case 2 `get_bank_base_address()` |
| reader/writer `TensorAccessorArgs<N>()` (interleaved only) | host `TensorAccessorArgs(*buffer).append_to(cta)`; kernel `TensorAccessorArgs<1>/<2>()` | binding mechanism (metadata packed by host, base addr auto-injected) |
| reader/writer CTA slot 1 (row-major **interleaved** only) | `input_unit_size` / `output_unit_size` (dead — never read) | dropped (no consumer; RTA `stick_size` copy remains) |
| compute CTA slot 0/1 | `src_cb_id`, `dst_cb_id` | `DFBBinding`s (`dfb::src`, `dfb::dst`) |
| all positional CTAs | `get_compile_time_arg_val(N)` | named `get_arg(args::name)` (`num_tiles` on compute) |
| all positional RTAs | `get_arg_val<uint32_t>(N)` | named `get_arg(args::name)` (`num_tiles` / `stick_size` / `num_sticks` / `start_id`) |
| kernel `get_tile_size(cb_id)` | free fn indexed by CB id | `dfb.get_tile_size()` (DFB member getter, rule 7) |

## Applied Patterns

- [Multi-variant factory](../shared/port_patterns.md#pattern-multi-variant-factories): build-time branching inside `create_program_artifacts` on `is_sharded` × `tilized` (kernel source) and `convert_dtype` (compute presence + DFB set). No class hierarchy.
- [Demoting per-group CTA to RTA — avoided](../shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta): compute preserved as two KernelSpecs (`COMPUTE_G1`/`COMPUTE_G2`) over disjoint WUs, `num_tiles` kept a CTA.
- Case 2 (raw-pointer) tensor bindings on the sharded reader/writer: `get_bank_base_address()` bridge, raw local-L1 walk unchanged (per brief). Data-movement kernels only — compute binds no tensor, so no Case-2-in-compute block.
- `unpack_modes` (FP32): when the compute kernel consumes a `Float32` `SRC` DFB with `enable_32_bit_dest = true`, add the explicit `{SRC, UnpackMode::UnpackToSrc}` entry the Metal 2.0 validator requires (legacy `ComputeConfigDescriptor` left `unpack_to_dest_mode` default → `UnpackToSrc`).

## Deferred / Flagged

- New findings during planning: none. Audit's census, tensor-binding cases, and factory-concept choice all confirmed against source. The dead row-major-interleaved CTA (audit misc-anomaly) resolves for free in the port; recorded under Legacy Inventory → Flags and in the report.
