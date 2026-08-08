# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/reduction/topk`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓

## TTNN factory analysis

The factory concept is selected downstream from these facts (→ `port_op_to_metal2_ttnn_factory.md`); the port does not pick it here. Carry these forward:

- **Op-owned tensors:** none
- **MeshWorkload:** not needed
- **Pybind `create_descriptor`:** none
- **Other risky pybind:** none
- **Custom `override_runtime_arguments`:** none

## Construct — to do

**Tensor bindings** (per binding, both factories — SingleCore and MultiCore share the same binding inventory):

- `input` — **Case 1** → re-express via `TensorParameter` / `TensorBinding` (kernel builds `TensorAccessor(ta::name)`). Host side: the `MeshTensor&` currently passed to `emplace_runtime_args` (`input_tensor` arg) becomes a named `TensorBinding`; the `TensorAccessorArgs(input_tensor).append_to(...)` CTA call is replaced by the named-token form.
- `value` (output) — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. Host: `MeshTensor&` in `emplace_runtime_args` → named `TensorBinding`. Sites: `topk_single_core_program_factory.cpp:219`; `topk_multi_core_program_factory.cpp:423`.
- `index` (output) — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. Host: `MeshTensor&` in `emplace_runtime_args` → named `TensorBinding`. Sites: `topk_single_core_program_factory.cpp:220`; `topk_multi_core_program_factory.cpp:424`.
- `indices` (optional input) — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. This binding currently uses a raw `.address()` RTA, which is the **silent-wrong correctness hazard** (stale buffer address on fast-path cache hits with different tensor storage). Priority fix: replace with a named `TensorParameter` / `TensorBinding` so the framework patches it correctly on cache hits. Host-side sites where `.address()` is extracted:
  - `topk_single_core_program_factory.cpp:266` — `static_cast<uint32_t>(tensor_args.indices->mesh_tensor().address())`
  - `topk_multi_core_program_factory.cpp:505` — `static_cast<uint32_t>(input_indices_tensor->address())`
  Kernel-side: the extracted address feeds directly into `TensorAccessor(indices_args, src_indices_addr)` in `reader_create_index_tensor.cpp:33` and `reader_create_index_local_topk.cpp:44`. Re-express as `TensorAccessor(ta::indices_name)`.

**Custom hash:** none

## Watch for

- **Notable constructs:** none
- **Cross-op / shared kernels:** none — all kernels are owned by this op; no borrowed files
- **RTA varargs:** none
- **Misc anomaly (non-blocking, not porter's fix):** `topk_single_core_program_factory.cpp:199` unconditionally sets `GENERATE_INDICES = "1"` regardless of `tensor_args.indices.has_value()`, making the `#if not GENERATE_INDICES` path in `reader_create_index_tensor.cpp` dead code on the single-core factory. The `indices` RTA (arg 3) is still injected but never read. GH issue #36329 tracks this. The multi-core factory conditions the define correctly. Not a port blocker; route to the op owner.
