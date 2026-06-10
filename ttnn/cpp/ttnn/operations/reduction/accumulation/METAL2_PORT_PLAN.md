# Port Plan — accumulation (`AccumulationDeviceOperation`, shared by cumsum / cumprod)

Port plan for the accumulation device-op, ported from the legacy `ProgramDescriptor`
factory (`create_descriptor`) to Metal 2.0's `ProgramSpecFactoryConcept`
(`create_program_spec` → `ProgramArtifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`AccumulationProgramFactory::create_descriptor`
  returns a `tt::tt_metal::ProgramDescriptor`; `accumulation_program_factory.cpp:42`).
- Variants: single (`program_factory_t = std::variant<AccumulationProgramFactory>`).
- Custom `compute_program_hash`: deleted → default (sanctioned exception); was at
  `accumulation_device_operation.hpp:64` (decl) / `accumulation_device_operation.cpp:105` (defn).

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/accumulation_reader.cpp` | `all_cores` | `TensorAccessorArgs(input).append_to(...)` (TA plumbing, slot 0..) | — | `input_base_addr`(0), `num_rows_per_core`(1), `tiles_per_row`(2), `input_tile_offset`(3), `start_id`(4), `low_rank_offset`(5), `high_rank_offset`(6), `flip`(7) | — | — | `ReaderConfigDescriptor` |
| writer | `device/kernels/dataflow/accumulation_writer.cpp` | `all_cores` | `TensorAccessorArgs(output).append_to(...)` | — | `output_base_addr`(0), `num_rows_per_core`(1), `tiles_per_row`(2), `input_tile_offset`(3), `start_id`(4), `low_rank_offset`(5), `high_rank_offset`(6), `flip`(7) | — | — | `WriterConfigDescriptor` |
| compute_1 | `device/kernels/compute/accumulation_compute.cpp` | `core_group_1` | `default_acc_value`(0) | — | `num_rows`(0), `tiles_per_row`(1) | — | `BINARY_OP_INIT`, `BINARY_OP`, `FILL_TILE` | `ComputeConfigDescriptor{ math_fidelity=default_math_fidelity, fp32_dest_acc_en=true, dst_full_sync_en=false, unpack_to_dest_mode=unpack_to_dst, math_approx_mode=false }` |
| compute_2 (optional) | same source | `core_group_2` | `default_acc_value`(0) | — | `num_rows`(0), `tiles_per_row`(1) | — | same defines | same config |

Notes:
- `default_acc_value` is a `std::bit_cast<uint32_t>` of a float (or int bit-pattern 1 for integer cumprod).
- Reader/writer share an identical RTA layout (positions 1–7), differing only in slot 0 (input vs output base addr — both go away under TensorBinding).
- `flip` is constant per dispatch but emitted per-core as an RTA in the legacy code.

### CBs
| index (CBIndex) | name | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|---|
| c_0 | SRC (CB_IN) | `4 * tile_size(input_df)` | `all_cores` | input dtype's df | `tile_size(input_df)` | (default 32x32) |
| c_1 | DST (CB_OUT) | `4 * tile_size(output_df)` | `all_cores` | output dtype's df | `tile_size(output_df)` | (default) |
| c_2 | ACC (CB_ACC) | `1 * tile_size(acc_df)` | `all_cores` | `acc_df` (Float32 unless integer) | `tile_size(acc_df)` | (default) |

`unpack_to_dest_mode` vector (per CB): ACC→UnpackToDestFp32 always; SRC→UnpackToDestFp32 unless input is Float16_b; DST→Default.

### Semaphores
none

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) | CTA offset (kernel) |
|---|---|---|---|
| `accumulation_program_factory.cpp:166` (`TensorAccessorArgs(input_tensor).append_to(reader_compile_time_args)`) | input (reader) | RTA 0 (`input_base_addr`) | `TensorAccessorArgs<0>()` |
| `accumulation_program_factory.cpp:169` (`TensorAccessorArgs(output_tensor).append_to(writer_compile_time_args)`) | output (writer) | RTA 0 (`output_base_addr`) | `TensorAccessorArgs<0>()` |

Both Case 1 (audit): standard interleaved page access; re-express via TensorParameter/TensorBinding, no bridge.

### Work split
- Driver: `split_work_to_cores(grid, num_rows_total)` (`accumulation_program_factory.cpp:78-80`).
- Returns `(num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2)`.
- core_group_1: count_per_core = `num_cols_per_core_group_1`.
- core_group_2: count_per_core = `num_cols_per_core_group_2` (present only if `!core_group_2.ranges().empty()`).

### Cross-op kernels
none — all three kernel `.cpp` are accumulation-owned.

### Flags
- The legacy CB descriptors use a single-element `format_descriptors` each → no aliased CBs.
- No borrowed-memory CBs (`CBDescriptor::buffer` never set).
- `ema/` is a *separate* device-op (`EmaDeviceOperation`) with its own `create_descriptor`, factory,
  and kernels — out of scope for this port. The brief's mention of "ema" sharing the device-op is
  imprecise: ema does not use `AccumulationDeviceOperation`. (Flagged in report.)

## TTNN ProgramFactory
- Concept (inherited from audit): `ProgramSpecFactoryConcept`.
- Custom `compute_program_hash`: delete (was at `accumulation_device_operation.hpp:64` / `.cpp:105`).
- Implementation notes: `create_descriptor` is renamed to `create_program_spec` and changes
  return type to `ttnn::device_operation::ProgramArtifacts`. The concept requires
  `create_descriptor` to NOT exist (`!ProgramDescriptorFactoryConcept`), so the rename is total.

## Planned Spec Shape

- KernelSpecs: `reader` (DM, all_cores), `writer` (DM, all_cores), `compute_g1` (compute, core_group_1),
  and `compute_g2` (compute, core_group_2) when group 2 is non-empty. Preserve compute multiplicity.
- DataflowBufferSpecs: `SRC` (CB_IN), `DST` (CB_OUT), `ACC` (CB_ACC). One per legacy CB; no aliasing.
- SemaphoreSpecs: none.
- TensorParameters: `input` (→ reader TensorBinding), `output` (→ writer TensorBinding).
- WorkUnitSpecs: `wu_g1` = {reader, writer, compute_g1} on core_group_1; `wu_g2` = {reader, writer, compute_g2}
  on core_group_2 (when present). Reader/writer bind into both WUs (single KernelSpec, multi-WU membership).

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (multi-binding) |
|---|---|---|---|
| `compute_desc_1` (`:187`, core_group_1), `cd2`/`compute_desc_2` (`:201-217`, core_group_2) of `accumulation_compute.cpp` | `compute_g1`, `compute_g2` (same source, per-group RTAs) | `wu_g1`, `wu_g2` | SRC (compute CONSUMER), DST (compute PRODUCER), ACC (compute self-loop PRODUCER+CONSUMER) bound on each compute KernelSpec |

Reader and writer are single KernelSpecs over `all_cores`, each a member of both `wu_g1` and `wu_g2`
(or just `wu_g1` when no group 2). Their SRC/DST bindings are shared across the two WUs.

Note: legacy compute differences are RTA-only (`num_rows`, `tiles_per_row`) — the only CTA
(`default_acc_value`) is identical across groups. The multiplicity is preserved structurally
(one KernelSpec per legacy descriptor / WU) per the recipe's non-negotiable, even though there is
no per-group *CTA* to lose here. (See report.)

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `reader.cpp:16` / factory `:166` | `TensorAccessorArgs<0>()` + `input_base_addr = get_arg_val<uint32_t>(0)` | `TensorParameter input` + reader `TensorBinding` → `TensorAccessor(ta::input)` |
| `writer.cpp:13` / factory `:169` | `TensorAccessorArgs<0>()` + `output_base_addr = get_arg_val<uint32_t>(0)` | `TensorParameter output` + writer `TensorBinding` → `TensorAccessor(ta::output)` |
| `reader.cpp:31`, `writer.cpp:33`, `compute.cpp:26-28` | `CircularBuffer cb_*(CB_*)` with magic `CB_IN/CB_OUT/CB_ACC = CBIndex::c_*` | `DataflowBuffer dfb_*(dfb::*)` + DFBBindings |
| compute `:21` CTA slot 0 | `get_compile_time_arg_val(0)` (`default_acc_value`) | named CTA `args::default_acc_value` |
| reader/writer RTA slots 1–7 | positional `get_arg_val<uint32_t>(n)` | named RTAs (`num_rows_per_core`, `tiles_per_row`, `input_tile_offset`, `start_id`, `low_rank_offset`, `high_rank_offset`, `flip`) |
| compute RTA slots 0–1 | positional | named RTAs (`num_rows`, `tiles_per_row`) |

## Applied Patterns

- [Self-loop DFB binding](metal2_port_patterns.md): ACC bound on each compute KernelSpec as both
  PRODUCER and CONSUMER (shared accessor name "acc").
- [Multi-variant / work-split multiplicity](metal2_port_patterns.md): one compute KernelSpec per legacy
  compute descriptor, two WorkUnitSpecs (g1/g2).
- [Pass DFB handles directly to LLKs](metal2_port_patterns.md): compute kernel passes `dfb::in/out/acc`
  to `unary_op_init_common`, `reconfig_data_format`, `pack_tile`, `copy_tile*` via implicit conversion.

## Deferred / Flagged

- Yellow items from audit: none (audit was GREEN).
- New findings during planning:
  - Brief/audit say cumsum/cumprod/**ema** share `AccumulationDeviceOperation`; ema is actually a
    separate device-op. The accumulation port affects cumsum + cumprod only. (No pybind hook for
    `create_descriptor` exists in any of the three consumer nanobind files, so no pybind deletion forced.)
  - `flip` is a per-dispatch-constant emitted as a per-core RTA; could be a CTA or CRTA. Left as a
    named RTA to preserve behavior (no scope creep). Flagged for owner.
  - **Surfaced at verification (not planning):** the legacy `unpack_to_dest_mode` set
    `UnpackToDestFp32` unconditionally on ACC (and on SRC unless Float16_b), including for integer
    DFBs. The Metal 2.0 spec validator rejects `UnpackToDestFp32` on non-Float32 DFBs, so int32
    cumsum/cumprod `TT_FATAL`ed until the entries were gated on `data_format == Float32`. Resolved in
    a behavior-preserving way; full writeup in PORT_REPORT (Friction → Gaps).
