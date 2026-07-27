# Port Plan — moreh/moreh_mean_backward

Port plan for `moreh_mean_backward`, ported from the TTNN `descriptor`
(`ProgramDescriptorFactoryConcept`) API to Metal 2.0 (`MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `create_descriptor()` returns
  `tt::tt_metal::ProgramDescriptor` (`device/moreh_mean_backward_program_factory.cpp:58`).
  Detected by the framework as `HasDirectDescriptor` (method sits directly on the
  device-op struct; no `program_factory_t`).
- Variants: single (single-descriptor factory).
- Custom `compute_program_hash`: none — already default reflection-based hash
  (confirmed in `device/moreh_mean_backward_device_operation.hpp/.cpp`; audit `Custom hash = no`).

*(Target concept chosen during audit: `MetalV2FactoryConcept` — carried forward below.)*

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs (order) | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_moreh_mean_backward.cpp` | `all_cores` | `[0]=input_grad_rank`, `[1..]=TensorAccessorArgs(output_grad)` | `output_grad.buffer()`, `num_tiles_per_core`, `tile_offset`, `num_dim`, then `output_grad_dim[rank]`, `input_grad_dim[rank]`, `need_bcast_dim[rank]` | none | `ReaderConfigDescriptor{}` (reader default triple RISCV_1/NOC_0/DEDICATED) |
| writer | `device/kernels/writer_moreh_mean_backward.cpp` | `all_cores` | `[0..]=TensorAccessorArgs(input_grad)` | `input_grad.buffer()`, `num_tiles_per_core`, `tile_offset` | none | `WriterConfigDescriptor{}` (writer default triple RISCV_0/NOC_1/DEDICATED) |
| compute_1 | `device/kernels/moreh_mean_backward.cpp` | `core_group_1` | `[0]=num_cols_per_core_group_1`, `[1]=need_bcast_dim[0]`, `[2]=need_bcast_dim[1]` | none | `FP32_DEST_ACC_EN=1` iff `fp32_dest_acc_en` | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` |
| compute_2 | `device/kernels/moreh_mean_backward.cpp` | `core_group_2` (may be empty) | `[0]=num_cols_per_core_group_2`, `[1]=need_bcast_dim[0]`, `[2]=need_bcast_dim[1]` | none | same as compute_1 | same as compute_1 |

Compute-kernel CTA meaning (from kernel): `[1]` → `wt_need_bcast`, `[2]` → `ht_need_bcast`.

### CBs
| CB | buffer_index | total_size | num_entries | data_format | tile | note |
|---|---|---|---|---|---|---|
| input   | c_0  | 2·tile | 2 | `cb_data_format` | default 32×32 | reader→compute |
| zero    | c_1  | 1·tile | 1 | `cb_data_format` | default | reader(fill)→compute |
| scalar  | c_2  | 1·tile | 1 | `cb_data_format` | default | reader(fill)→compute (bcast-scalar operand) |
| intermed| c_24 | 1·tile | 1 | `cb_data_format` | default | compute self-touch |
| output  | c_16 | 2·tile | 2 | `cb_data_format` | default | compute→writer |

`cb_data_format = datatype_to_dataformat_converter(output_grad.dtype())`; op is BFLOAT16-only
(`check_tensor(..., {DataType::BFLOAT16})`), so the format is `Float16_b` at every CB.
No `.tile`, no `.global_circular_buffer`, no `.address_offset` on any CB.

### Semaphores
None (op uses no semaphores).

### Tensor accessors
| Tensor | role | host RTA surface | kernel construction |
|---|---|---|---|
| `output_grad` (input) | Case 1 | `output_grad.buffer()` (Buffer* overload) at reader RTA slot 0 (`program_factory.cpp:251`) | `TensorAccessor(output_grad_args, output_grad_addr)` (`reader:92`) |
| `input_grad`  (output) | Case 1 | `input_grad.buffer()` at writer RTA slot 0 (`program_factory.cpp:267`) | `TensorAccessor(input_grad_args, input_grad_addr)` (`writer:21`) |

### Work split
`split_work_to_cores(grid, num_input_grad_tiles)` →
`(num_cores_to_be_used, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2)`.
Compute kernel instantiated twice over the two disjoint groups; reader/writer over `all_cores`.

### Cross-op kernels
None. All three kernel `.cpp` files are op-owned under `device/kernels/`. They `#include` the
shared moreh pool headers `ttnn/cpp/ttnn/kernel/{dataflow,compute}/moreh_common.hpp` (already
`DataflowBuffer`-typed, Device 2.0 native) plus `api/...` LLK/HAL headers. The port does **not**
change any helper signature in those shared headers (see Applied Patterns / Watch-for).

### Flags
No unreferenced kernel files. No descriptor types outside the audit's scan.

## TTNN ProgramFactory
- Concept (inherited from audit): `MetalV2FactoryConcept`.
- Custom `compute_program_hash`: none (nothing to delete).
- Implementation notes: legacy op is `HasDirectDescriptor` (method on the op struct). Metal 2.0
  needs a `program_factory_t`, so the port introduces a nested factory struct
  `MorehMeanBackwardProgramFactory` with `create_program_artifacts` and declares
  `using program_factory_t = std::variant<MorehMeanBackwardProgramFactory>;`. Single alternative,
  so no `select_program_factory` needed (framework auto-selects). No pybind `create_descriptor`
  to remove (nanobind binds the host free function).

## Planned Spec Shape
- **KernelSpecs (4)**: `READER`, `WRITER`, `COMPUTE_G1`, `COMPUTE_G2` (G2 only if `core_group_2`
  non-empty). One `KernelSpec` per legacy `KernelDescriptor` — preserves the compute work-split
  multiplicity (per-group `num_output_tiles` CTA reproduced per instance).
- **DataflowBufferSpecs (5)**: `IN`(c_0,2), `ZERO`(c_1,1), `SCALAR`(c_2,1), `INTERMED`(c_24,1),
  `OUT`(c_16,2). `entry_size = tile_size(cb_data_format)`, `data_format_metadata = cb_data_format`.
  No `tile_format_metadata` (legacy `.tile` unset → default 32×32).
- **SemaphoreSpecs**: none.
- **TensorParameters (2)**: `OUTPUT_GRAD` (from `output_grad.tensor_spec()`),
  `INPUT_GRAD` (from `input_grad`/output `tensor_spec()`).
- **WorkUnitSpecs**: `wu_g1`={READER,WRITER,COMPUTE_G1} on `core_group_1`;
  `wu_g2`={READER,WRITER,COMPUTE_G2} on `core_group_2` (only if present). Reader/writer are members
  of both WUs, so their effective node set is `all_cores`.
- **Op-owned tensors**: none.

## Preserved Multiplicity
```
Legacy KernelDescriptors [compute_desc_1, compute_desc_2] of source moreh_mean_backward.cpp
  → KernelSpecs [COMPUTE_G1, COMPUTE_G2] of same source
  → in WorkUnitSpecs [wu_g1, wu_g2]  (disjoint node sets core_group_1 / core_group_2)
  → sharing DFBs: IN (CONSUMER), ZERO (CONSUMER), SCALAR (CONSUMER),
                  INTERMED (PRODUCER+CONSUMER self-loop), OUT (PRODUCER)
```
The two compute instances cover **disjoint** node sets (one per node), so each shared DFB endpoint
bound by both instances contributes exactly one instance per node — legal, **no** multi-binding
flag. Reader/writer are single instances over `all_cores`.

## Dropped Plumbing
- **Buffer-address RTAs**:
  - reader RTA slot 0 `output_grad.buffer()` → `TensorBinding` `OUTPUT_GRAD` (accessor `output_grad`).
  - writer RTA slot 0 `input_grad.buffer()` → `TensorBinding` `INPUT_GRAD` (accessor `input_grad`).
- **`TensorAccessorArgs` plumbing**:
  - reader CTA `TensorAccessorArgs<1>()` (`reader:37`) + host `TensorAccessorArgs(output_grad.buffer()).append_to(reader_ct_args)` (`program_factory.cpp:168`) → binding mechanism.
  - writer CTA `TensorAccessorArgs<0>()` (`writer:11`) + host `TensorAccessorArgs(input_grad.buffer()).append_to(writer_ct_args)` (`program_factory.cpp:179`) → binding mechanism.
- **Magic CB indices**: kernel-side `tt::CBIndex::c_0/c_1/c_2/c_16/c_24` constants (reader:75-77,
  writer:18, compute:19-28) → `dfb::in/zero/scalar/out/intermed` handles. No CB index ever
  appeared in a CTA (CBs were referenced by kernel-side literal), so no CTA slot drops for these.
- **Positional CTAs → named**:
  - reader CTA `[0]` → `{"input_grad_rank", input_grad_rank}`.
  - compute CTA `[0]/[1]/[2]` → `{"num_output_tiles", ...}, {"wt_need_bcast", need_bcast_dim[0]}, {"ht_need_bcast", need_bcast_dim[1]}`.
- **Positional RTAs → named** (reader): `num_output_tiles`, `start_id`, `num_dim`.
  (writer): `num_tiles`, `start_id`.
- **Page-size 3rd CTA/RTA**: none (both accessor sites are 2-arg).
- **Semaphore-ID RTAs**: none.

## Applied Patterns
- **[Self-loop DFB binding]**: `INTERMED` (c_24) on the compute KernelSpec(s) — bound both PRODUCER
  and CONSUMER (compute both fills and drains it, `compute:52/58/63/75`).
- **[Two-toucher / disjoint-node same-source split]**: compute over `core_group_1` / `core_group_2`
  — two KernelSpecs of one source over disjoint node sets. Each shared DFB endpoint legally bound by
  both; **no** `allow_instance_multi_binding`.
- **[RTA varargs]**: reader's three per-dimension blocks `output_grad_dim` / `input_grad_dim` /
  `need_bcast_dim`, each of length `input_grad_rank` (a CTA-bounded count that varies across
  instantiations), ported via the kernel-side vararg mechanism
  (`num_runtime_varargs = 3 * input_grad_rank`), concatenated in that order and read positionally
  with `get_vararg`. The four leading scalars are ordinary named RTAs.
- **[Pass DFB handles directly to LLKs/helpers]**: `dfb::name` passed straight into
  `binary_op_init_common`, `add_tiles_bcast_*`, `copy_tile`, `mul_tiles_bcast`, and to the
  `*_with_dt` moreh_common helpers (which take `DataflowBuffer` objects, constructed from the token).
- **[DFB metadata via object]**: `get_tile_size(cb_id)` (reader:96, writer:25) → `dfb.get_tile_size()`.

## Deferred / Flagged
- None new. Audit's GREEN gate set held. `moreh_common.hpp` shared helpers stay untouched (already
  `DataflowBuffer`-typed; only the token→object construction happens in the op's own kernels).
