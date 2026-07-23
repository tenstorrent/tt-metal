# Metal 2.0 Port Brief — `data_movement/moe_expert_token_remap`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `05554b94288 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

The op has one device operation (`MoeExpertTokenRemapDeviceOperation`) and one program factory (`Multicore`, `device/moe_expert_token_remap_program_factory.cpp`). It instantiates two kernels: the op-owned writer (`device/kernels/dataflow/writer_moe_expert_token_remap.cpp`) and a cross-family borrowed reader (`ccl/all_to_all_combine/.../reader_all_to_all_combine.cpp`).

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (`Multicore::create_descriptor` returns a `ProgramDescriptor`). Per-device specialization rides the `mesh_dispatch_coordinate` adapter — `flat_mesh_idx` is baked into a reader compile-time arg ([factory:150-164](device/moe_expert_token_remap_program_factory.cpp#L150-L164)); this is single-program, not a `WorkloadDescriptor`.
- **Op-owned tensors:** none — outputs are ordinary device tensors from `create_output_tensors`.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind. All `no`.

## Construct — to do

**Tensor bindings** (per binding) — all **Case 1** (base fed into a `TensorAccessor`; all access through the accessor). Each is delivered today via the `Buffer*`-binding form (a `Buffer*` in `emplace_runtime_args`). Express each as a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)` and the RTA `Buffer*` slot plus its `TensorAccessorArgs` compile-time plumbing both disappear.

- `mapping_tensor` — reader, `TensorAccessor(mapping_args, mapping_tensor_addr)` ([reader:82](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L82)).
- `metadata_tensor` — reader, `TensorAccessor(metadata_args, metadata_tensor_addr)` ([reader:81](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L81)).
- `topk_tensor` (data) — reader, `TensorAccessor(data_args, data_tensor_addr)` ([reader:83](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L83)).
- `output_mapping_tensor` — writer, `TensorAccessor(output_mapping_args, output_mapping_base_addr)` ([writer:38](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L38)).
- `output_reduced_tensor` — writer, `TensorAccessor(output_reduced_args, output_reduced_base_addr)` ([writer:39](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L39)).

The host `TensorAccessorArgs(...).append_to(...)` calls ([factory:170-172](device/moe_expert_token_remap_program_factory.cpp#L170-L172), [factory:197-198](device/moe_expert_token_remap_program_factory.cpp#L197-L198)) and the kernel-side `TensorAccessorArgs<N>()` declarations go away with the binding.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every accessor is already 2-arg.

**CB endpoints:** six CBs, single config, all resolvable at port time:

- **Self-loop** (single toucher — bind the one kernel PRODUCER *and* CONSUMER):
  - `c_0` mapping ([factory:66-74](device/moe_expert_token_remap_program_factory.cpp#L66-L74)) — reader-only scratch.
  - `c_4` output_mapping staging ([factory:124-132](device/moe_expert_token_remap_program_factory.cpp#L124-L132)) — writer-only scratch.
  - `c_5` output_reduced staging ([factory:137-145](device/moe_expert_token_remap_program_factory.cpp#L137-L145)) — writer-only scratch.
- **Legal 1:1** (reader PRODUCER + writer CONSUMER — no special action):
  - `c_1` local_experts ([factory:83-91](device/moe_expert_token_remap_program_factory.cpp#L83-L91)).
  - `c_2` metadata ([factory:97-105](device/moe_expert_token_remap_program_factory.cpp#L97-L105)).
  - `c_3` topk/data ([factory:111-119](device/moe_expert_token_remap_program_factory.cpp#L111-L119)).

No multi-binding, no dead CB.

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer and no multi-reader. The three producer CBs are clean single-producer/single-consumer FIFOs.
- **Cross-op / shared kernels:** the reader is `ccl/all_to_all_combine`'s `reader_all_to_all_combine.cpp`, borrowed by file path and **also used by `all_to_all_combine` itself**. Its Metal 2.0 rewrite (CB→DFB, named-token bindings) is a single change that both ops must adopt together — do not migrate this op's use of the reader in isolation, or `all_to_all_combine` breaks. Port the shared kernel as one unit. Both kernels also `#include` `moe_utils.hpp` (only `find_if`) and the in-family `common.hpp` (`tt_memmove`/`fill_with_val`); these cross cleanly.
- **RTA varargs:** none — both kernels read RTAs at fixed constant indices; prefer named RTAs throughout.
