# Port Plan — fast_reduce_nc

Port plan for `ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc`, ported from
the `ProgramDescriptor` (`descriptor`) concept to Metal 2.0 (`MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `FastReduceNCProgramFactory::create_descriptor(...)`
  returns a `tt::tt_metal::ProgramDescriptor` (`device/fast_reduce_nc_program_factory.cpp:54`).
- Variants: single (`FastReduceNCDeviceOperation` → single `FastReduceNCProgramFactory`).
- Custom `compute_program_hash`: none — device op defines only `validate_on_program_cache_miss`
  / `compute_output_specs` / `create_output_tensors`. No deletion needed.

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_reduce_nc.cpp` | all_cores | input_granularity, shard_factor, num_cores_to_be_used, then `TensorAccessorArgs<3>` | input_buffer(Buffer*), num_input_tiles, id_range_length, start_id, dim, reduce_tile_size, inner_tile_size | none | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/writer_reduce_nc.cpp` | all_cores | shard_factor, num_cores_to_be_used, then `TensorAccessorArgs<2>` | output_buffer(Buffer*), id_range_length, start_id | none | `WriterConfigDescriptor{}` |
| compute g1 | `device/kernels/reduce_nc.cpp` | core_group_1 | num_cols_per_core_group_1, num_reduce_input_tile, input_granularity | none | `FP32_DEST_ACC_EN=1` (iff fp32_dest_acc_en) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` |
| compute g2 | `device/kernels/reduce_nc.cpp` | core_group_2 (iff non-empty) | num_cols_per_core_group_2, num_reduce_input_tile, input_granularity | none | same | same |

### CBs
| index | total_size | data_format | page_size | disposition |
|---|---|---|---|---|
| c_0 (in0) | in0_t(=input_granularity*2) * input_tile_size | input dtype | input_tile_size | DFB FRNC_IN0 (reader P, compute C) |
| c_1 (in1/zero) | in1_t(=1) * cb_1_tile_size | BFLOAT16 | cb_1_tile_size | DFB FRNC_IN1 (reader P, compute C) |
| c_24 (intermed0) | 1 * intermed_tile_size | Float32/output | — | **DEAD — dropped** (no kernel touches index 24) |
| c_16 (out0) | out0_t(=2) * output_tile_size | output dtype | output_tile_size | DFB FRNC_OUT (compute P, writer C) |

### Semaphores
none.

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `reader_reduce_nc.cpp:42` `TensorAccessor(tensor_args, input_addr)` | `tensor_args.input` | reader RTA 0 (`Buffer*` input_buffer) — Case 1 |
| `writer_reduce_nc.cpp:30` `TensorAccessor(tensor_args, output_addr)` | `tensor_return_value` | writer RTA 0 (`Buffer*` output_buffer) — Case 1 |

### Work split
- Driver: `split_work_to_cores(grid|sub_core_grids, num_output_tiles, row_wise=true)`, or
  `dspec.core_groups_tuple()` when `divide_by_shards`.
- Yields `(num_cores_to_be_used, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2)`.
- `num_cols_per_core_group_{1,2}` scaled by `shard_factor`.

### Cross-op kernels
none — all three kernels are owned in-directory. Reader `#include`s the shared `kernel_lib`
header `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp` (`dataflow_kernel_lib::prepare_zero_tile`);
this is a `kernel_lib` donor, not a cross-op kernel, and is left unmodified (its
`prepare_zero_tile<uint32_t dfb_id>()` NTTP is satisfied by `dfb::name`'s constexpr cast).

### Flags
- Unused `constexpr uint32_t onetile = 1;` in the reader (`reader_reduce_nc.cpp`) is dead — left
  in place (team-only, not porter work per audit).

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: single-program, no op-owned tensors. The device-op class needs no
  edits (no custom hash, no `override_runtime_arguments`, no pybind `create_descriptor` — the
  nanobind file binds the user-facing function only).

## Planned Spec Shape
- **KernelSpecs**: `reader` (DM), `writer` (DM), `compute_g1` (compute), `compute_g2` (compute,
  only when core_group_2 is non-empty) — 1:1 with the legacy `KernelDescriptor`s, preserving the
  per-group compute multiplicity.
- **DataflowBufferSpecs**: `FRNC_IN0` (c_0), `FRNC_IN1` (c_1), `FRNC_OUT` (c_16). c_24 dropped.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `FRNC_INPUT` (input), `FRNC_OUTPUT` (output).
- **WorkUnitSpecs**: `wu_g1` {reader, writer, compute_g1} on core_group_1; `wu_g2`
  {reader, writer, compute_g2} on core_group_2 (only when present). Reader/writer belong to both
  work units — effective node set is the union (all_cores).

## Preserved Multiplicity
| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| compute g1, g2 (source `reduce_nc.cpp`) | compute_g1, compute_g2 | wu_g1, wu_g2 (disjoint node sets) | FRNC_IN0 (CONSUMER), FRNC_IN1 (CONSUMER), FRNC_OUT (PRODUCER) each |

Disjoint node sets → each node sees one instance = legal single-role binding. **Not** the
`allow_instance_multi_binding` flag. Per-group `num_output_tiles` stays a **CTA** (not demoted to
RTA) — see anti-pattern "Demoting per-group CTA to RTA".

## Dropped Plumbing
| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA slot 0 (`...program_factory.cpp:327`) | `input_buffer` (`Buffer*`) | `TensorBinding(FRNC_INPUT, "src")` |
| reader CTA plumbing (`:201`) | `TensorAccessorArgs(*input.buffer()).append_to(reader_cta)` | binding mechanism |
| reader kernel (`reader_reduce_nc.cpp:21`) | `constexpr auto tensor_args = TensorAccessorArgs<3>();` + `input_addr` RTA read | `TensorAccessor(tensor::src)` |
| writer RTA slot 0 (`:337`) | `output_buffer` (`Buffer*`) | `TensorBinding(FRNC_OUTPUT, "dst")` |
| writer CTA plumbing (`:204`) | `TensorAccessorArgs(*output.buffer()).append_to(writer_cta)` | binding mechanism |
| writer kernel (`writer_reduce_nc.cpp:15`) | `constexpr auto tensor_args = TensorAccessorArgs<2>();` + `output_addr` RTA read | `TensorAccessor(tensor::dst)` |
| all kernels | positional CTAs / magic CB indices (`cb_id_in0=0`, `cb_id_in1=1`, `cb_id_out=16`, `c_0/c_1/c_16`) | named CTAs + `dfb::in0/in1/out0` from `DFBBinding` |
| c_24 CB alloc (`:177-185`) | dead `CBDescriptor` | dropped (no DFB) |

## Applied Patterns
- Anti-pattern avoided: "Demoting per-group CTA to RTA" — two compute KernelSpecs preserved.
- Pattern: "Pass DFB handles directly to LLKs and kernel-lib helpers" — `dfb::in0/in1/out0` into
  `binary_op_init_common` / `add_tiles_init` / `reconfig_data_format` / `add_tiles` /
  `pack_reconfig_data_format` / `pack_tile`, and `prepare_zero_tile<dfb::in1>()`.
- Dead-CB drop: c_24.
- Hardware config Style A (compute): `to_compute_hardware_config(arch, compute_kernel_config)`;
  explicit `unpack_modes` entry added for the Float32 input DFB under `enable_32_bit_dest` (legacy
  Default → `UnpackToSrc`).

## Deferred / Flagged
- New findings during planning: none. Audit matched the code exactly.
