# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/reduction/topk`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓

Scope: one device operation `ttnn::prim::TopKDeviceOperation`, two factories — `TopKSingleCoreProgramFactory` (`device/topk_single_core_program_factory.cpp`) and `TopKMultiCoreProgramFactory` (`device/topk_multi_core_program_factory.cpp`). Nine kernels, all owned in-directory; no donor/borrowed kernels.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `port_op_to_metal2_ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Op-owned tensors:** none (only the two declared output tensors, created in `create_output_tensors`, `device/topk_device_operation.cpp:292-293`).
- **MeshWorkload:** not needed — both factories return a single `ProgramDescriptor`; multi-core coordination is intra-program semaphore-based, not cross-program/cross-device.
- **Pybind `create_descriptor`:** none (`topk_nanobind.cpp` binds only the op function).
- **Other risky pybind:** none.
- **Custom `override_runtime_arguments`:** none.

## Construct — to do

**Tensor bindings** (per binding — all **Case 1**, via `TensorAccessor`): express each as a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(ta::name)` instead, and the existing `TensorAccessorArgs` CTA plumbing + the address-via-RTA both disappear.

Single-core (`TopKSingleCoreProgramFactory`):
- `input` — Case 1. RTA arg 0 = `input_tensor` object (`:262`) → `TensorAccessor` (`reader_create_index_tensor.cpp:40`).
- `indices` (optional input) — Case 1. RTA arg 3 = `indices->mesh_tensor().address()` (`:266`) → `TensorAccessor` (`reader_create_index_tensor.cpp:33`). Binding it as a `TensorParameter` removes the raw `.address()`-via-RTA fast-path-cache stale-address hazard. *(The precomputed-indices read path is currently compiled out — `GENERATE_INDICES` is hardcoded to `1`, GH #36329 — but bind it anyway.)*
- `values` output — Case 1. RTA arg 0 = `value_tensor` (`:272`) → `TensorAccessor` (`writer_binary_interleaved.cpp:30`).
- `indices` output — Case 1. RTA arg 1 = `index_tensor` (`:273`) → `TensorAccessor` (`writer_binary_interleaved.cpp:31`).

Multi-core (`TopKMultiCoreProgramFactory`):
- `input` — Case 1. RTA arg 0 = `input_tensor` object (`:501`) → `TensorAccessor` (`reader_create_index_local_topk.cpp:33`).
- `indices` (optional input) — Case 1. RTA arg 4 = `input_indices_tensor->address()` (`:505`) → `TensorAccessor` (`reader_create_index_local_topk.cpp:44`). Same `.address()`-via-RTA → typed-binding cleanup.
- `values` output — Case 1. RTA arg 0 = `value_tensor` (`:531`) → `TensorAccessor` (`writer_final_topk.cpp:30`).
- `indices` output — Case 1. RTA arg 1 = `index_tensor` (`:532`) → `TensorAccessor` (`writer_final_topk.cpp:31`).

No Case 2 (raw-pointer) bindings, no compute-kernel tensor binding. Compute kernels and the inter-core transfer kernels (`writer_local_topk.cpp`, `reader_final_topk.cpp`) touch CB L1 memory only.

**Custom hash:** none.

## Watch for

- **Notable constructs:** none. The two `SemaphoreDescriptor::initial_value = INVALID` (multi-core, lines 326/332) are *zero* (`INVALID == 0`), so this is a plain zero-init semaphore — not the deprecated non-zero path; translate to a standard `SemaphoreSpec`. The multi-core `gathered_values_cb`/`gathered_indices_cb` (c_4/c_5) are written into by remote cores via raw L1 pointer but are genuine producer/consumer DFBs (`reader_final_topk` pushes, `topk_final` consumes) — declare them as normal `DataflowBufferSpec`s, not fake CBs.
- **Cross-op / shared kernels:** none — all kernels owned in-directory; no port-together coupling.
- **RTA varargs:** none.
