# Port Plan — moreh_norm_backward

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_norm_backward`, ported from the
`descriptor` (`ProgramDescriptorFactoryConcept`, direct `create_descriptor`) API to Metal 2.0
(`MetalV2FactoryConcept`). Written during the inventory and planning steps; committed alongside
the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`, realized as **HasDirectDescriptor** — `create_descriptor`
  lives directly on `MorehNormBackwardOperation` (no `program_factory_t` variant), returning
  `tt::tt_metal::ProgramDescriptor` (`device/moreh_norm_backward_program_factory.cpp:58`).
- Variants: single (one factory, one DeviceOperation). No sharding/config branch; the only split is
  the `core_group_1` / `core_group_2` work-split → two compute `KernelDescriptor`s over disjoint nodes.
- Custom `compute_program_hash`: none — already default reflection-based hash.

*(Target concept `MetalV2FactoryConcept`, inherited from the audit — see the TTNN ProgramFactory section.)*

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_moreh_norm_backward.cpp` | all_cores | `input_grad_rank`, then `TensorAccessorArgs(input)`, `TensorAccessorArgs(output)`, `TensorAccessorArgs(output_grad)` | `input.buffer()`, `output.buffer()`, `output_grad.buffer()`, `decimal`(bitcast), `num_tiles_per_core`, `tile_offset`, then 3× `input_grad_rank`-count blocks (`output_grad_dim`, `input_grad_dim`, `need_bcast_dim`) | none | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/writer_moreh_norm_backward.cpp` | all_cores | `TensorAccessorArgs(input_grad)` | `input_grad.buffer()`, `num_tiles_per_core`, `tile_offset` | none | `WriterConfigDescriptor{}` |
| compute (group 1) | `device/kernels/moreh_norm_backward_kernel.cpp` | core_group_1 | `num_cols_per_core_group_1`, `need_bcast_dim[0]`, `need_bcast_dim[1]` | `num_tiles_per_core`, `floored_p`, `p_is_negative`, `floored_p_minus_one`, `p_minus_one_is_negative` | `FP32_DEST_ACC_EN=1` iff `fp32_dest_acc_en` | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, math_approx_mode}` |
| compute (group 2) | same source | core_group_2 (may be empty) | `num_cols_per_core_group_2`, `need_bcast_dim[0]`, `need_bcast_dim[1]` | same as group 1 | same | same |

Note: compute CTA[0] (`num_output_tiles`) is read into a `constexpr` but is **unused** in the kernel
body (the loop bound is the RTA `num_input_tiles_per_core`). Preserved faithfully as a named CTA
anyway (it is the per-group CTA; do not demote). Flagged in the report as a pre-existing dead read.

### CBs (all `all_cores`, no `.tile` set → default 32×32)
| CB | role | entry_size | num_entries | data_format |
|---|---|---|---|---|
| c_0 | input (x) | `tile_size(cb_data_format)` | 1 | `cb_data_format` |
| c_1 | output (y) | `tile_size(cb_data_format)` | 1 | `cb_data_format` |
| c_2 | output_grad (dy) | `tile_size(cb_data_format)` | 1 | `cb_data_format` |
| c_3 | decimal | `tile_size(cb_data_format)` | 1 | `cb_data_format` |
| c_16 | input_grad (dx) | `tile_size(cb_data_format)` | 1 | `cb_data_format` |
| c_24..c_31 | 8 intermediates (xpow, logx, exp_lxmd, correct_xpow, tmp4, tmp5, recip_ypow, sign) | `tile_size(intermed_data_format)` | 1 | `intermed_data_format` |

`cb_data_format = datatype_to_dataformat_converter(output_grad.dtype())`;
`intermed_data_format = fp32_dest_acc_en ? Float32 : cb_data_format`.

### Semaphores
None.

### Tensor accessors (all Case 1 — fed to `TensorAccessor`, clean bases)
- `input` — reader `:86` `TensorAccessor(input_args, input_addr)`; RTA `program_factory.cpp:255`; CTA `:178`.
- `output` — reader `:89`; RTA `:256`; CTA `:179`.
- `output_grad` — reader `:92`; RTA `:257`; CTA `:180`.
- `input_grad` — writer `:25`; RTA `:268`; CTA `:191`.

### Work split
`split_work_to_cores(grid, num_input_grad_tiles)` →
`(num_cores_to_be_used, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2)`.

### Cross-op kernels
None. All three kernel files are op-owned. Shared-header dependency only:
`ttnn/kernel/dataflow/moreh_common.hpp` (`fill_cb_with_value`) and `ttnn/kernel/compute/moreh_common.hpp`
(compute helper family). Both are already `DataflowBuffer`-based and take DFB objects — **no kernel-side
signature change forced, not edited by this port**.

## TTNN ProgramFactory
- **Concept (inherited from audit):** `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`:** none.
- **Implementation notes:** the legacy op is HasDirectDescriptor (no `program_factory_t`). The framework
  only auto-synthesizes a `DirectDescriptorFactory` wrapper for a direct `create_descriptor`; there is no
  equivalent auto-wrap for a direct `create_program_artifacts`. So the port introduces a
  `program_factory_t = std::variant<MorehNormBackwardProgramFactory>` and a nested factory struct holding
  `create_program_artifacts`, replacing the direct `create_descriptor`. Single-variant, so no
  `select_program_factory` needed. (This mirrors `experimental/quasar/binary_ng`'s `ProgramFactoryMetalV2`.)

## Planned Spec Shape
- **KernelSpecs (4):** READER, WRITER, COMPUTE_1 (core_group_1), COMPUTE_2 (core_group_2, only when non-empty).
  One per legacy `KernelDescriptor` — preserves the two-compute work-split multiplicity.
- **DataflowBufferSpecs (13):** INPUT(c_0), OUTPUT(c_1), OUTPUT_GRAD(c_2), DECIMAL(c_3), DX(c_16),
  and 8 intermediates XPOW/LOGX/EXP_LXMD/CORRECT_XPOW/TMP4/TMP5/RECIP_YPOW/SIGN (c_24..c_31).
  `data_format_metadata` set on all (all bound to compute). No `tile_format_metadata` (legacy `.tile` unset).
- **SemaphoreSpecs:** none.
- **TensorParameters (4):** input, output, output_grad (bound by reader), input_grad (bound by writer). Strict spec match.
- **WorkUnitSpecs:** WU_group1 = {READER, WRITER, COMPUTE_1} on core_group_1; WU_group2 = {READER, WRITER, COMPUTE_2}
  on core_group_2 (only when non-empty). Per-node census (1 producer + 1 consumer per node) holds — verified
  against `program_spec.cpp` `ValidateProgramSpec`.

## Preserved Multiplicity
```
Legacy KernelDescriptors [compute_desc_1, compute_desc_2] of source moreh_norm_backward_kernel.cpp
  → KernelSpecs [COMPUTE_1, COMPUTE_2] of same source
  → in WorkUnitSpecs [WU_group1, WU_group2] (disjoint node sets core_group_1 / core_group_2)
  → shared DFBs: INPUT/OUTPUT/OUTPUT_GRAD/DECIMAL (CONSUMER on each), DX (PRODUCER on each),
    8 intermediates (self-loop PRODUCER+CONSUMER on each).
```
Same-source KernelSpecs over disjoint node sets each bind one role legally — no `allow_instance_multi_binding`,
no self-loop stacking issue. Reader (sole PRODUCER of the 4 read DFBs) and writer (sole CONSUMER of DX) run on
the union of both groups; per node each read DFB has reader(1P)+one-compute(1C) and DX has one-compute(1P)+writer(1C).

## Dropped Plumbing
- **Buffer-address RTAs → `TensorBinding`:** reader `input.buffer()`/`output.buffer()`/`output_grad.buffer()`
  (`program_factory.cpp:255-257`); writer `input_grad.buffer()` (`:268`). All Case 1.
- **`TensorAccessorArgs` CTAs → binding mechanism:** reader `:178-180`; writer `:191`. Kernel-side
  `TensorAccessorArgs<N>()` chains (reader `:37-39`, writer `:14`) replaced by `TensorAccessor(tensor::name)`.
- **Positional CTAs → named CTAs:** reader `input_grad_rank`; compute `num_output_tiles`, `wt_need_bcast`,
  `ht_need_bcast`.
- **Positional RTAs → named RTAs / varargs:** reader named `decimal`, `num_output_tiles`, `start_id`; the three
  `input_grad_rank`-count dim blocks → **runtime varargs** (`num_runtime_varargs = 3 * input_grad_rank`, read
  via `get_vararg`). Writer named `num_input_tiles_per_core`, `tile_offset`. Compute named `num_input_tiles_per_core`,
  `p`, `p_is_negative`, `p_minus_one`, `p_minus_one_is_negative`.
- **Page-size 3rd CTA/RTA:** none (every `TensorAccessor` is 2-arg).
- **Semaphore-ID RTAs:** none.

## Applied Patterns
- **Self-loop DFB binding:** the 8 intermediates (c_24..c_31) — compute is both PRODUCER and CONSUMER. Legal on Gen1.
- **Preserved work-split multiplicity:** two compute KernelSpecs over disjoint node sets (not a two-toucher, no flag).
- **Runtime varargs:** reader's three count-bounded dim blocks (count = `input_grad_rank`, a CTA — a CTA-bounded
  loop still varies per instantiation → vararg per the recipe).
- **Compute config Style B:** the legacy factory builds a Metal `ComputeConfigDescriptor` directly (three fields set;
  `dst_full_sync_en` left at its Metal default `false`), so the port builds a `ComputeGen1Config` directly rather than
  routing through `to_compute_hardware_config` (which would wire the resolved `dst_full_sync_en` the legacy op dropped).
  See report Friction.

## Deferred / Flagged
- Compute CTA `num_output_tiles` is a dead read (see Inventory note) — preserved, not removed (unrelated cleanup).
- Reader dim blocks are identical on every node → could be common runtime varargs; kept as per-node runtime varargs
  to preserve legacy dispatch semantics (RTA→CRTA is a separate pass).
- `decimal_minus_one` unused local in the factory (audit "Misc anomalies") — left untouched.
