# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓

---

## TTNN factory analysis

The factory concept is selected downstream from these facts (→ `port_op_to_metal2_ttnn_factory.md`); the port does not pick it here. Carry these forward:

- **Op-owned tensors:** none
- **MeshWorkload:** not needed (no op-owned tensors; no genuine multi-program need)
- **Pybind `create_descriptor`:** none
- **Other risky pybind:** none
- **Custom `override_runtime_arguments`:** none

---

## Construct — to do

### Tensor bindings (per binding)

The two prefill factories pass `Buffer*` objects into `emplace_runtime_args` (the framework's `BufferBinding` auto-registration form). The decode sharded factory binds all tensors via `CBDescriptor::buffer` (borrowed-memory DFBs — already the clean port path).

**`RotaryEmbeddingLlamaMultiCore` (interleaved prefill):**

- `input_tensor` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(ta::input_tensor)`, `src_addr` RTA disappears.
- `cos_cache` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(ta::cos_cache)`, `cos_addr` RTA disappears.
- `sin_cache` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(ta::sin_cache)`, `sin_addr` RTA disappears.
- `trans_mat` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(ta::trans_mat)`, `trans_mat_addr` RTA disappears.
- `output` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(ta::output)`, `dst_addr` RTA disappears.

**`RotaryEmbeddingLlamaMultiCoreSharded` (sharded decode):**

All five working CBs (input/cos/sin/trans_mat/output) bind via `CBDescriptor::buffer` → borrowed-memory DFBs → **clean**. Declare each as `DataflowBufferSpec` with `borrowed_from = <tensor_parameter_name>`. No `TensorAccessor` or RTA changes needed.

**`RotaryEmbeddingLlamaMultiCorePrefillSharded` (prefill with optional sharded cos/sin/trans_mat):**

- `input_tensor` — **Case 1** (always interleaved in prefill) → same as `RotaryEmbeddingLlamaMultiCore`.
- `cos_cache` — split by code path:
  - sharded fast-path (`cos_sin_sharded && !cos_sin_sharded_reload`): `CBDescriptor::buffer = cos_buffer` → borrowed-memory DFB → **clean**.
  - interleaved or sharded-reload path: `Buffer*` in reader RTA → **Case 1** → re-express via `TensorParameter`.
- `sin_cache` — same split as `cos_cache`.
- `trans_mat` — split by code path:
  - global-CB path (`trans_mat_use_global_cb`): `CBDescriptor::buffer = trans_mat_buffer` → borrowed-memory DFB → **clean**.
  - non-global-CB path: `Buffer*` in reader RTA → **Case 1** → re-express via `TensorParameter`.
- `output` — **Case 1** (always interleaved in prefill) → same as `RotaryEmbeddingLlamaMultiCore`.

### Custom hash

**Delete custom `compute_program_hash` → default (sanctioned exception).**

`device/rotary_embedding_llama_device_operation.cpp:228–232` — the override simply calls the framework default `hash_operation<RotaryEmbeddingLlamaDeviceOperation>`. Delete the override declaration in the `.hpp` (line 29) and its definition in the `.cpp` (lines 228–232). The default is correct-by-construction and no custom logic is lost.

---

## Watch for

- **Borrowed-memory DFBs:** Multiple `CBDescriptor::buffer = <ptr>` bindings in `RotaryEmbeddingLlamaMultiCoreSharded` (all five CBs) and conditionally in `RotaryEmbeddingLlamaMultiCorePrefillSharded`. Port these as `DataflowBufferSpec::borrowed_from = <tensor_parameter_name>`. All confirmed real producer+consumer (sharded compute kernel self-serves via `reserve_back` / `push_back` / `wait_front` / `pop_front`). See `METAL2_PREPORT_AUDIT.md` — Heads-ups → Notable LANDED constructs for the full site list.
- **Cross-op / shared kernels:** none — all kernels are op-owned with no file-path borrowing.
- **RTA varargs:** none.
