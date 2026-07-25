# Port Plan — moreh_sum_backward

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_sum_backward`, ported from the
legacy `ProgramDescriptor` (`descriptor`) factory to Metal 2.0 `MetalV2FactoryConcept`.
Written during the inventory and planning steps; committed alongside the port.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (direct `create_descriptor` on the op struct →
  `HasDirectDescriptor`). Single factory `MorehSumBackwardOperation::create_descriptor`
  (`device/moreh_sum_backward_program_factory.cpp:66`) returning `tt::tt_metal::ProgramDescriptor`.
- Variants: single (interleaved tile I/O; two compute core groups differing only in tile count).
- Custom `compute_program_hash`: none — already default reflection-based hash.

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_moreh_sum_backward.cpp` | all_cores | `{input_grad_rank}` + `TensorAccessorArgs(output_grad)` | `output_grad.buffer()`, num_tiles_per_core, tile_offset, then 3 var-length blocks (output_grad_dim, input_grad_dim, need_bcast_dim; each len=input_grad_rank) | none | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/writer_moreh_sum_backward.cpp` | all_cores | `TensorAccessorArgs(input_grad)` | `input_grad.buffer()`, num_tiles_per_core, tile_offset | none | `WriterConfigDescriptor{}` |
| compute_1 | `device/kernels/moreh_sum_backward.cpp` | core_group_1 | `{num_cols_per_core_group_1, need_bcast_dim[0], need_bcast_dim[1]}` | none | `FP32_DEST_ACC_EN=1` iff fp32_dest_acc_en | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` |
| compute_2 | same source | core_group_2 (if non-empty) | `{num_cols_per_core_group_2, need_bcast_dim[0], need_bcast_dim[1]}` | none | same | same |

### CBs (all plain CircularBuffers; no GlobalCB)
| legacy CB | total_size | page_size | data_format | touch census |
|---|---|---|---|---|
| c_0 (input) | 2·tile | tile | output_grad dtype | PRODUCER=reader (reserve/push), CONSUMER=compute (wait/pop) |
| c_1 (zero)  | 1·tile | tile | output_grad dtype | PRODUCER=reader (`fill_cb_with_value`), CONSUMER=compute (wait/pop) |
| c_16 (out)  | 2·tile | tile | output_grad dtype | PRODUCER=compute (reserve/push+pack), CONSUMER=writer (wait/pop) |

None set `address_offset`. None `.tile`. None a GlobalCircularBuffer.

### Semaphores
None.

### Tensor accessors
- `output_grad` (input) — reader, Case 1 via `TensorAccessor(output_grad_args, output_grad_addr)`
  (`reader...cpp:84`). Host RTA slot 0 = `output_grad.buffer()` (`program_factory.cpp:252`).
- `input_grad` (output) — writer, Case 1 via `TensorAccessor(input_grad_args, input_grad_addr)`
  (`writer...cpp:23`). Host RTA slot 0 = `input_grad.buffer()` (`program_factory.cpp:261`).

### Work split
`split_work_to_cores(grid, num_input_grad_tiles)` →
`(num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2)`.

### Cross-op kernels
None. The op owns all three kernel files. They `#include` shared moreh helper headers
(`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`, `.../compute/moreh_common.hpp`), which are
already Metal-2.0-flavored (`DataflowBuffer` by value) and are out of the porter's writeable scope
— left untouched.

### Flags
No unreferenced kernels. All RTAs/CTAs/defines consumed by the kernels.

## TTNN ProgramFactory
- Concept (inherited from audit): `MetalV2FactoryConcept`.
- Custom `compute_program_hash`: none.
- Implementation notes: `create_program_artifacts` (→ `ttnn::device_operation::ProgramArtifacts`)
  is placed on a **nested `ProgramFactory` struct** referenced by
  `using program_factory_t = std::variant<ProgramFactory>;`. This is mandatory: `MetalV2FactoryConcept`
  is detected only through the `program_factory_t` variant — the `HasDirectDescriptor` direct-method
  shortcut in `operation_concepts.hpp` applies to `create_descriptor` only, so a bare
  `create_program_artifacts` on the op struct builds but is never registered as Metal 2.0 (kernels
  get no generated-header injection). Single-alternative variant → framework auto-selects; no
  `select_program_factory`. No op-owned tensors.

## Planned Spec Shape
- **KernelSpecs**: `reader`, `writer`, `compute_group_1`, `compute_group_2` (the last only if
  core_group_2 is non-empty). Preserves the legacy two-`KernelDescriptor` compute multiplicity.
- **DataflowBufferSpecs**: `c0_in` (entry=tile, entries=2), `c1_zero` (entry=tile, entries=1),
  `c16_out` (entry=tile, entries=2). `data_format_metadata = cb_data_format` on all three
  (all are compute-bound).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `output_grad` (from `output_grad.tensor_spec()`),
  `input_grad` (from `input_grad.tensor_spec()`).
- **WorkUnitSpecs** (per-group co-location, matching proven `moreh_group_norm`):
  - `group1` = {reader, writer, compute_group_1} @ core_group_1
  - `group2` = {reader, writer, compute_group_2} @ core_group_2 (only if non-empty)
  - reader/writer are members of both work units; effective placement = core_group_1 ∪ core_group_2
    = all_cores.
- **Op-owned tensors**: none.

## Preserved Multiplicity
```
Legacy KernelDescriptors [compute_desc_1, compute_desc_2] of source moreh_sum_backward.cpp
  → KernelSpecs [compute_group_1, compute_group_2] of same source
  → in WorkUnitSpecs [group1 @ core_group_1, group2 @ core_group_2]
  → sharing DFBs: c0_in (CONSUMER), c1_zero (CONSUMER), c16_out (PRODUCER)
```
Endpoint roles: compute_group_1 and compute_group_2 are the same source over **disjoint** node sets
(core_group_1 / core_group_2), each binding one endpoint role on the shared DFBs — legal
multi-KernelSpec-on-one-endpoint (no `allow_instance_multi_binding` flag). reader is the single
PRODUCER of c0_in/c1_zero over all_cores; writer the single CONSUMER of c16_out. Per node: 1P+1C.

## Dropped Plumbing
- **Buffer-address RTAs** → `TensorBinding`:
  - reader RTA slot 0 `output_grad.buffer()` (`program_factory.cpp:252`) → `TensorBinding{output_grad}`;
    kernel `TensorAccessor(output_grad_args, output_grad_addr)` (`reader...cpp:84`) → `TensorAccessor(tensor::output_grad)`.
  - writer RTA slot 0 `input_grad.buffer()` (`program_factory.cpp:261`) → `TensorBinding{input_grad}`;
    kernel `TensorAccessor(input_grad_args, input_grad_addr)` (`writer...cpp:23`) → `TensorAccessor(tensor::input_grad)`.
- **`TensorAccessorArgs` plumbing** → binding mechanism:
  - reader `TensorAccessorArgs(output_grad.buffer()).append_to(reader_ct_args)` (`program_factory.cpp:176`)
    + kernel `TensorAccessorArgs<1>()` (`reader...cpp:36`) → dropped.
  - writer `TensorAccessorArgs(input_grad.buffer()).append_to(writer_ct_args)` (`program_factory.cpp:187`)
    + kernel `TensorAccessorArgs<0>()` (`writer...cpp:12`) → dropped.
- **Positional CTAs → named**:
  - reader `{input_grad_rank}` → named CTA `input_grad_rank`.
  - compute `{num_cols, wt, ht}` → named CTAs `num_output_tiles`, `wt_need_bcast`, `ht_need_bcast`.
- **Positional RTAs → named / varargs**:
  - reader `num_tiles_per_core`, `tile_offset` → named RTAs `num_output_tiles`, `start_id`.
    The three variable-length blocks (output_grad_dim, input_grad_dim, need_bcast_dim) → runtime
    **varargs** (`advanced_options.num_runtime_varargs = 3 * input_grad_rank`), read kernel-side
    positionally via `get_vararg(i)` (recipe kernel-side whitelist rule 4 / RTA-varargs shape (a)).
  - writer `num_tiles_per_core`, `tile_offset` → named RTAs `num_tiles`, `start_id`.
- **Page-size 3rd arg CTAs/RTAs**: none (accessors are 2-arg).
- **Semaphore-ID RTAs**: none.

## Applied Patterns
- Multi-KernelSpec work-split (preserve multiplicity): compute_group_1 / compute_group_2 over
  disjoint node sets, each a single endpoint on the shared DFBs — no multi-binding flag.
- Named-CTA / named-RTA / TensorBinding / DFBBinding standard conversions.
- RTA varargs (recipe whitelist rule 4): the three per-dim blocks kept as positional runtime varargs.
- Compute `hw_config` = `ComputeGen1Config` built directly (Style B — legacy set a Metal
  `ComputeConfigDescriptor` directly): `fpu_math_fidelity=math_fidelity`,
  `sfpu_precision_mode = math_approx_mode?Approximate:Precise`, `enable_32_bit_dest=fp32_dest_acc_en`,
  `double_buffer_dest = !dst_full_sync_en`. `unpack_modes`: legacy set none (all default UnpackToSrc);
  the Metal 2.0 validator requires an explicit entry when a Float32 DFB is consumed with
  `enable_32_bit_dest=true`, so add explicit `UnpackToSrc` for the consumed Float32 DFBs (`c0_in`,
  `c1_zero`) only in that case — faithful to the legacy default.
- DM `hw_config`: reader = `create_reader_datamovement_config`, writer = `create_writer_datamovement_config`
  (legacy used default `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}`).

## Deferred / Flagged
- None. No structural issue beyond the audit surfaced during planning.
