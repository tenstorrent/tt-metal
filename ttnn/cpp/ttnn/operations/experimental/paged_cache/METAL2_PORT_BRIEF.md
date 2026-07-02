# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/paged_cache`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓

## TTNN factory analysis

The factory concept is selected downstream from these facts (→ `port_op_to_metal2_ttnn_factory.md`); the port does not pick it here. Carry these forward:

- **Op-owned tensors:** none — all three DeviceOperations are in-place; `create_output_tensors` returns the input cache tensor(s) directly.
- **MeshWorkload:** not needed. The `*MeshWorkloadFactory` variants exist only for `mesh_coords` coordinate-filtering (dispatching noop programs to non-target coordinates) — this is a plumbing artifact, not a genuine multi-program or cross-device coordination requirement.
- **Pybind `create_descriptor`:** none.
- **Other risky pybind:** none.
- **Custom `override_runtime_arguments`:** none. (The DeviceOperations use `get_dynamic_runtime_args` for per-dispatch patching; this is a different interface and does not require a port-time deletion.)

## Construct — to do

**Tensor bindings** (per binding):

*PagedFillCacheDeviceOperation:*
- `input_tensor` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. Reader kernel builds `TensorAccessor(ta::input_tensor)`.
- `cache_tensor` (output, in-place) — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. Writer kernel builds `TensorAccessor(ta::cache_tensor)`.
- `page_table` — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. Writer kernel builds `TensorAccessor(ta::page_table)`.
- `batch_idx_tensor` (optional) — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. Writer kernel builds `TensorAccessor(ta::batch_idx_tensor)`.

*PagedUpdateCacheDeviceOperation:*
- `cache_tensor` (output, in-place) — **Case 1** → re-express via `TensorParameter` / `TensorBinding`.
- `input_tensor` — **clean** (borrowed-memory DFB; sharded tensor placed via `CBDescriptor{ .buffer = in1_buffer }`; port uses `DataflowBufferSpec::borrowed_from`).
- `update_idxs_tensor` (optional) — **Case 1** → re-express via `TensorParameter` / `TensorBinding` (when interleaved-DRAM path); when sharded (L1), it is also a borrowed-memory DFB — **clean**.
- `page_table` (optional) — **Case 1** → re-express via `TensorParameter` / `TensorBinding` (when interleaved-DRAM path); when sharded, borrowed-memory DFB — **clean**.

*PagedFusedUpdateCacheDeviceOperation (both tiled and row-major factories):*
- `cache_tensor1` (output, in-place) — **Case 1** → re-express via `TensorParameter` / `TensorBinding`.
- `cache_tensor2` (output, in-place) — **Case 1** → re-express via `TensorParameter` / `TensorBinding`.
- `input_tensor1` — **clean** (borrowed-memory DFB; sharded; port uses `DataflowBufferSpec::borrowed_from`).
- `input_tensor2` — **clean** (borrowed-memory DFB; sharded; port uses `DataflowBufferSpec::borrowed_from`).
- `update_idxs_tensor` (optional) — **Case 1** (interleaved DRAM) or **clean** (sharded L1 borrowed-memory DFB).
- `page_table` (optional) — **Case 1** (interleaved DRAM) or **clean** (sharded L1 borrowed-memory DFB).

**Custom hash:** delete all three custom `compute_program_hash` → default (sanctioned exception):
- `PagedFillCacheDeviceOperation::compute_program_hash` @ `device/fill_cache/paged_fill_cache_device_operation.cpp:182`
- `PagedUpdateCacheDeviceOperation::compute_program_hash` @ `device/update_cache/paged_update_cache_device_operation.cpp:313`
- `PagedFusedUpdateCacheDeviceOperation::compute_program_hash` @ `device/fused_update_cache/paged_fused_update_cache_device_operation.cpp:247`

## Watch for

- **Notable constructs:**
  - **Borrowed-memory DFB** — sharded input tensors and optionally sharded index/page-table tensors use `CBDescriptor{ .buffer = <ptr> }`. Port declares each as `DataflowBufferSpec::borrowed_from = <tensor_parameter_name>`. See `METAL2_PREPORT_AUDIT.md` Heads-ups for the full list of sites across all three DeviceOperations.
  - **Aliased Circular Buffers** — three factories create a single `CBDescriptor` with two `CBFormatDescriptor` entries sharing the same backing memory (`intermed0`/`intermed1` pairs). Port uses `DataflowBufferSpec::advanced_options.alias_with` mutually naming the two DFBs. This is an advanced/ninja feature — both aliased DFBs must have the same `num_entries * entry_size` and be bound to the same kernels. Sites:
    - `paged_update_cache_program_factory.cpp:206-218` — `intermed0_cb_index` (c_24) + `intermed1_cb_index` (c_25)
    - `paged_tiled_fused_update_cache_program_factory.cpp:226-239` — `intermed0_cb_index` (c_24) + `intermed1_cb_index` (c_25)
    - `paged_row_major_fused_update_cache_program_factory.cpp:232-245` — `intermed0_cb_index` (c_5) + `intermed1_cb_index` (c_6)

- **Cross-op / shared kernels:** Compute kernels include `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` and `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` (official shared kernel library). Functions invoked via `uint32_t cb_id` template parameters — the `dfb::name` constexpr cast handles this cleanly; no donor-side changes required.

- **RTA varargs:** none.
