# Port Plan — moreh_abs_pow

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_abs_pow`, ported from the TTNN
`descriptor` concept (`ProgramDescriptor` / `create_descriptor`) to Metal 2.0
(`MetalV2FactoryConcept` / `create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — realized as `HasDirectDescriptor`
  (a `create_descriptor` static directly on `MorehAbsPowOperation`, no `program_factory_t`).
- Variants: single (`MorehAbsPowOperation (single-descriptor)`).
- Custom `compute_program_hash`: none — already default reflection-based hash (grep clean).

Target concept (inherited from audit): `MetalV2FactoryConcept`.

### Kernels
Three kernels, all owned in `device/kernels/`, all file-path instantiated by the factory.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_moreh_abs_pow.cpp` | `all_cores` | `TensorAccessorArgs(*input.buffer())` only | none | `input.buffer()` (arg0), `is_dram(input)`, `decimal`(bitcast), `num_units_per_core`, `Wt`, `tile_offset`, `origin_w` | none | none | `ReaderConfigDescriptor{}` (reader default) |
| writer | `device/kernels/writer_moreh_abs_pow.cpp` | `all_cores` | `TensorAccessorArgs(*output.buffer())` only | none | `output.buffer()` (arg0), `is_dram(output)`, `num_units_per_core`, `Wt`, `tile_offset` | none | none | `WriterConfigDescriptor{}` (writer default) |
| compute_1 | `device/kernels/moreh_abs_pow_kernel.cpp` | `core_group_1` | `{num_units_per_core_group_1}` **(DEAD — kernel never reads any CTA)** | none | `num_units_per_core`, `Wt`, `origin_w`, `floored_p`, `p_is_negative` | none | none | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, math_approx_mode}` |
| compute_2 | (same source) | `core_group_2` (if non-empty) | `{num_units_per_core_group_2}` **(DEAD)** | none | same as compute_1 | none | none | same |

Key inventory finding: the compute kernel reads **no** compile-time args
(`grep` for `get_compile_time_arg_val` in `moreh_abs_pow_kernel.cpp` is empty). The legacy
per-group CTA `{num_units_per_core_group_*}` is therefore **dead** — its value is never
consumed on device. The compute loop count comes from the RTA `num_rows_per_core`
(`moreh_abs_pow_kernel.cpp:10,62`), which is set per-core in the legacy RTA loop.

### CBs (9 `CBDescriptor`s, all plain — no GlobalCircularBuffer, no address_offset)
All `total_size = 1 tile`, `page_size = tile_size`, on `all_cores`, `tile` field unset.

| buffer_index | role | data_format | consumed/produced |
|---|---|---|---|
| c_0 | input (`x`) | `cb_data_format` | reader P → compute C |
| c_1 | one | `cb_data_format` | reader P → compute C |
| c_2 | decimal | `cb_data_format` | reader P → compute C |
| c_3 | mask_w | `cb_data_format` | reader P → compute C (both gated at **runtime** on `do_mask_w = origin_w%32!=0`) |
| c_16 | output (`y`) | `cb_data_format` | compute P → writer C |
| c_24 | xabs `\|x\|` | `intermed_data_format` | compute only (self-loop) |
| c_25 | xpow `\|x\|^p` | `intermed_data_format` | compute only (self-loop) |
| c_26 | logx `log(\|x\|)` | `intermed_data_format` | compute only (self-loop) |
| c_27 | exp_lxmd | `intermed_data_format` | compute only (self-loop) |

`cb_data_format = datatype_to_dataformat_converter(input.dtype())` (BFLOAT16 or INT32 —
never Float32). `intermed_data_format = fp32_dest_acc_en ? Float32 : cb_data_format`.

### Semaphores
None — the op uses no semaphores.

### Tensor accessors
| Tensor | kind | host RTA slot | audit case |
|---|---|---|---|
| input (tensor_args.input) | reader `TensorAccessor(input_args, input_addr)` | reader RTA arg0 (`input.buffer()`) + CTA `TensorAccessorArgs(*input.buffer())` | Case 1 |
| output (tensor_return_value) | writer `TensorAccessor(output_args, output_addr)` | writer RTA arg0 (`output.buffer()`) + CTA `TensorAccessorArgs(*output.buffer())` | Case 1 |

### Work split
`split_work_to_cores(grid, num_units)` →
`(num_cores_to_be_used, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2)`.
Per-core RTA loop assigns `num_units_per_core` = group-1 or group-2 count, and accumulates
`tile_offset += num_units_per_core * Wt`.

### Cross-op kernels
None instantiated cross-op. Function-call escapes into shared headers
`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` (reader) and
`ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` (compute); all helpers this op calls already
take `DataflowBuffer` (Device 2.0 native), so no donor rewrite is forced — out of scope.

### Flags
No unreferenced kernel files. No descriptor type outside the audit scan.

## TTNN ProgramFactory
- Concept (inherited from audit): `MetalV2FactoryConcept`.
- Custom `compute_program_hash`: none.
- Implementation notes: legacy op used `HasDirectDescriptor` (create_descriptor directly on
  the op struct). The MetalV2 adapter has **no** direct-`create_program_artifacts` shortcut
  (`resolve_program_factory` special-cases only `create_descriptor`), so the port introduces a
  nested factory struct `MorehAbsPowProgramFactory` and `using program_factory_t =
  std::variant<MorehAbsPowProgramFactory>` (single alternative → no `select_program_factory`
  needed). `create_descriptor` is removed (its presence would also satisfy
  `ProgramDescriptorFactoryConcept` and break `AllFactoriesValid`).

## Planned Spec Shape
- **KernelSpecs**: `reader`, `writer`, **one** `compute` (see Preserved Multiplicity for why the
  two legacy compute descriptors collapse to one).
- **DataflowBufferSpecs**: one per legacy CBDescriptor — `INPUT`(c_0), `ONE`(c_1), `DECIMAL`(c_2),
  `MASK_W`(c_3), `OUTPUT`(c_16), `XABS`(c_24), `XPOW`(c_25), `LOGX`(c_26), `EXP_LXMD`(c_27).
  `entry_size = tile_size`, `num_entries = 1`, `data_format_metadata` per table above,
  `tile_format_metadata` unset (legacy `.tile` unset).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT` (input.tensor_spec()), `OUTPUT` (output.tensor_spec()).
- **WorkUnitSpecs**: one — `{reader, writer, compute}` over `all_cores`.
- **Op-owned tensors**: none.

## Preserved Multiplicity
**Collapsed to one compute KernelSpec — and this is NOT a CTA→RTA demotion.** The two legacy
compute `KernelDescriptor`s differ *only* by the per-group CTA `{num_units_per_core_group_*}`,
which the compute kernel **never reads** (no `get_compile_time_arg_val` anywhere). The device
loop count is the RTA `num_rows_per_core`, already set per-core in the legacy RTA loop and
identical in value to the dead CTA for each group. Therefore:

- Reproducing the dead CTA as a **named** CTA would create an unreferenced named arg, which the
  build rejects (recipe Build §: "host added a named CTA/RTA without the kernel referencing it →
  reconcile"). So it cannot be carried.
- With the only per-group difference (the dead CTA) gone, the two compute KernelSpecs would be
  byte-identical, so a single KernelSpec over `all_cores` is exact.
- The [Demoting per-group CTA to RTA anti-pattern](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md)
  does **not** apply: its harm is losing *compile-time loop unrolling* on a dimension that was a
  CTA. Here the loop dimension was **already an RTA** in the legacy kernel (`for row_idx <
  num_rows_per_core`, RTA-bound) — nothing is demoted, and there was never compile-time
  unrolling to lose. See the report's Friction section for the full reasoning.

Result: `none — the legacy work-split multiplicity was carried entirely by a dead CTA; no live
per-group CTA exists to preserve.`

## Dropped Plumbing
- **Buffer-address RTAs → `TensorBinding`**:
  - reader RTA arg0 `input.buffer()` (`program_factory.cpp:250`) + kernel `input_addr`
    (`reader:12`) → `TensorParameter INPUT` + `TensorBinding`; kernel `TensorAccessor(tensor::input)`.
  - writer RTA arg0 `output.buffer()` (`program_factory.cpp:260`) + kernel `output_addr`
    (`writer:14`) → `TensorParameter OUTPUT` + `TensorBinding`; kernel `TensorAccessor(tensor::output)`.
- **`TensorAccessorArgs` plumbing → binding mechanism**:
  - `TensorAccessorArgs(*input.buffer()).append_to(reader_ct_args)` (`program_factory.cpp:183`)
    + kernel `TensorAccessorArgs<0>()` (`reader:26`) → dropped.
  - `TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args)` (`program_factory.cpp:194`)
    + kernel `TensorAccessorArgs<0>()` (`writer:23`) → dropped.
  - Both readers/writers end with **empty** `compile_time_args`.
- **Magic CB indices → `DFBBinding`**: the kernels derived CB ids via local counters
  (`reader:20-24`, `writer:20-21`, `compute:16-34`) and passed raw ids to `copy_tile` /
  `binary_op_init_common` / `DataflowBuffer(...)`. All replaced by `dfb::<name>` tokens.
- **Positional CTAs → named**: the only positional CTA was the compute dead CTA — dropped (see
  Preserved Multiplicity), not renamed.
- **Page-size 3rd CTA/RTA**: none (both accessors are 2-arg).
- **Semaphore-ID RTAs**: none.
- **Dead RTAs kept as-is** (per brief; they are *read* by the kernel so not unreferenced):
  `input_is_dram` (`reader:13`), `output_is_dram` (`writer:15`) — unused after read. Ported as
  named RTAs unchanged; not cleaned up.

## Applied Patterns
- **Self-loop DFB binding**: `XABS`/`XPOW`/`LOGX`/`EXP_LXMD` on the compute KernelSpec, each bound
  both PRODUCER and CONSUMER (shared accessor name). Compute-only intermediates.
- **1P+1C ordinary bindings**: `INPUT`/`ONE`/`DECIMAL`/`MASK_W` (reader P → compute C),
  `OUTPUT` (compute P → writer C). `MASK_W` is bound **unconditionally** though its produce/consume
  is gated at *runtime* on `do_mask_w` (derived from the `origin_w` RTA, not a host-time value) —
  so the conditional-binding `#ifdef` pattern does **not** apply; a runtime-unused DFB is harmless.
- **Hardware config — compute (Style A resolve, built directly)**: legacy resolves knobs via
  `get_compute_kernel_config_args` but builds a `ComputeConfigDescriptor` that sets only
  `math_fidelity`, `fp32_dest_acc_en`, `math_approx_mode` — leaving `dst_full_sync_en` at the
  descriptor default. Faithful reproduction builds a `ComputeGen1Config` directly with exactly
  those three fields (leaving `double_buffer_dest` at its default `true`, which equals legacy
  `dst_full_sync_en = false`). The TTNN `to_compute_hardware_config` helper is **not** used because
  it would forward the resolved `dst_full_sync_en`, diverging from the legacy applied config. See
  report Friction.
- **unpack_modes (FP32 required-entry)**: when `fp32_dest_acc_en` is true the intermediate DFBs
  (`XABS`/`XPOW`/`LOGX`/`EXP_LXMD`) are `Float32` and are *consumed* by the compute kernel
  (self-loop), so the Metal 2.0 validator requires an explicit `unpack_modes` entry. Legacy
  `unpack_to_dest_mode` was empty (all `Default`) → map to `UnpackMode::UnpackToSrc`. Added only
  under `fp32_dest_acc_en`. No other DFB is ever Float32 (io/scalar DFBs are BFLOAT16/INT32).
- **Hardware config — DM**: reader = reader default → `create_reader_datamovement_config(arch)`;
  writer = writer default → `create_writer_datamovement_config(arch)`.

## Deferred / Flagged
- New findings from planning:
  - Dead compute CTA `{num_units_per_core_group_*}` — see Preserved Multiplicity / report.
  - Legacy drops `dst_full_sync_en` when building the compute descriptor — reproduced faithfully
    (double_buffer_dest fixed `true`); flagged for owner review in the report.
  - Dead RTAs `input_is_dram` / `output_is_dram` — carried as-is per brief.
