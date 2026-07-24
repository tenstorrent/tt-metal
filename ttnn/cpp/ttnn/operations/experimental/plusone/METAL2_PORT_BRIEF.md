# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/plusone`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓

## TTNN factory analysis

The factory concept is selected downstream from these facts (→ `port_op_to_metal2_ttnn_factory.md`); the port does not pick it here. Carry these forward:

- **Op-owned tensors:** none — `create_output_tensors` returns the input tensor directly (in-place op)
- **MeshWorkload:** not needed — single-program, no cross-device coordination, no op-owned-tensor artifact
- **Pybind `create_descriptor`:** none
- **Other risky pybind:** none
- **Custom `override_runtime_arguments`:** none

## Construct — to do

**Tensor bindings** (per binding):

- `input` — **Case 1** → re-express via `TensorParameter` / `TensorBinding` (kernel builds `TensorAccessor(ta::input)`). The `Buffer*` runtime arg at `plusone_program_factory.cpp:85` and the CTA-packed `TensorAccessorArgs` block at line 70 both disappear; the kernel's `get_arg_val<uint32_t>(0)` at `reader_plusone_interleaved.cpp:16` is replaced by `ta::input`-based construction.

**Custom hash:** delete custom `compute_program_hash` at `device/plusone_device_operation.cpp:26–39` → default (sanctioned exception). No repair; straight deletion.

## Watch for

- **Fake CBs (address-only):** `cb_id_in0` / `CBIndex::c_0` — no producer+consumer FIFO in either path. The kernel at `reader_plusone_interleaved.cpp:31` calls `cb_in0.get_write_ptr()` purely for the scratch/shard address; no `reserve_back` / `push_back` / `wait_front` / `pop_front` anywhere. Port resolves this with the sanctioned fake-CB workaround (see `port_op_to_metal2_recipe.md`). Applies to both the interleaved path (`CBDescriptor::buffer = nullptr`, plain scratch) and the sharded path (`CBDescriptor::buffer = src_buffer`, borrowed address). **Do not** declare this CB as `DataflowBufferSpec::borrowed_from` — the borrowed-memory DFB path requires a genuine FIFO producer+consumer.
- **Sharded path `TensorAccessor` dead code:** in the sharded (`src0_is_dram = false`) path, `s0` and `src_addr` at `reader_plusone_interleaved.cpp:16,25–26` are unused (all `noc.async_read`/`noc.async_write` calls are guarded by `if (src0_is_dram)`). After re-expressing as `TensorParameter`, the kernel's `TensorAccessor(ta::input)` construction is still harmless — but note the dead use in the sharded branch. The port does not need to restructure this.
- **Cross-op / shared kernels:** none — the sole kernel (`reader_plusone_interleaved.cpp`) is owned by this op directory; no out-of-directory coupling.
- **RTA varargs:** none.
