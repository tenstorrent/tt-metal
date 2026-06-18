# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/reduction/topk` (`TopKSingleCoreProgramFactory` subset)

> The op is **RED at op level**; only the **single-core factory** subset clears all gates. This brief covers that subset only. The full record — including why the multi-core factory is blocked — is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared (single-core subset):** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓

**⚠ Scope: `TopKSingleCoreProgramFactory` only** (`device/topk_single_core_program_factory.cpp`). The sibling `TopKMultiCoreProgramFactory` is **RED — do not port**: its many-core→final-core gather writes results across nodes into a common-address CB (`c_4`/`c_5`), expressible in Metal 2.0 only as a `CrossNodeDataflowBuffer`/`GlobalDataflowBuffer`, which is not yet implemented (see audit *Gate detail → cross-node gather*). `select_program_factory` chooses single- vs multi-core by input size, so the single-core path ports independently. Single-core kernels (all owned in-directory, no donor/borrowed kernels): `reader_create_index_tensor.cpp`, `writer_binary_interleaved.cpp`, `topk.cpp`.

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

*(Multi-core bindings omitted — that factory is RED and out of port scope; see the audit.)*

No Case 2 (raw-pointer) bindings, no compute-kernel tensor binding in the single-core factory. `topk.cpp` (compute) touches CB L1 memory only.

**Custom hash:** none.

## Watch for

- **Notable constructs:** none in the single-core factory. (The semaphores and the `c_4`/`c_5` gather CBs all belong to the RED multi-core factory, out of scope here.)
- **Cross-op / shared kernels:** none — all kernels owned in-directory; no port-together coupling.
- **RTA varargs:** none.
