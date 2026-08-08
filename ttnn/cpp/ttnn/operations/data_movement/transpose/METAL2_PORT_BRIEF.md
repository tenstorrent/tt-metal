# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/transpose`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓

---

## TTNN factory analysis

The factory concept is selected downstream from these facts (→ `port_op_to_metal2_ttnn_factory.md`); the port does not pick it here. Carry these forward:

- **Op-owned tensors:** none
- **MeshWorkload:** not needed
- **Pybind `create_descriptor`:** none
- **Other risky pybind:** none
- **Custom `override_runtime_arguments`:** none

---

## Construct — to do

**Tensor bindings** (per binding, per factory):

*TransposeWHProgramFactory — tiled sub-path:*
- `input` (reader `reader_unary_transpose_wh_interleaved_start_id.cpp`) — **Case 1** → re-express via `TensorParameter` / `TensorBinding` (kernel builds `TensorAccessor(ta::input)`; RTA slot 0 `src_addr` disappears).
- `output` (writer `writer_unary_interleaved_start_id.cpp` — borrowed from eltwise/unary) — **Case 1** → re-express via `TensorParameter` / `TensorBinding`.

*TransposeWHProgramFactory — RM sub-path:*
- `input` (reader `reader_unary_transpose_wh_interleaved_start_id_rm.cpp`) — **Case 1** → re-express.
- `output` (writer `writer_unary_transpose_wh_interleaved_start_id_rm.cpp`) — **Case 1** → re-express.

*TransposeWHShardedProgramFactory:*
- `input` (`CBDescriptor::buffer = input_tensor.buffer()`) — **clean** → borrowed-memory DFB (`DataflowBufferSpec::borrowed_from`).
- `output` (`CBDescriptor::buffer = output_tensor.buffer()`) — **clean** → borrowed-memory DFB.

*TransposeWHShardedRMProgramFactory:*
- `input` (`CBDescriptor::buffer = input_tensor.buffer()`) — **clean** → borrowed-memory DFB.
- `output` (`CBDescriptor::buffer = output_tensor.buffer()`) — **clean** → borrowed-memory DFB.

*TransposeHCTiledInterleavedProgramFactory:*
- `input` (reader `reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp`) — **Case 1** → re-express. Note: host currently passes `input_buffer->address()` as per-core RTA[0] (`->address()` form — correctness hazard on cache hits) even though `TensorAccessorArgs(..., RuntimeTensorShape)` supplies accessor config via CRTA. The port drops the raw-address RTA and routes entirely through `TensorParameter`.
- `output` (writer `writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp`) — **Case 1** → re-express. Same `->address()` hazard at `transpose_hc_tiled_interleaved_program_factory.cpp:91`.

*TransposeHCRMProgramFactory:*
- `input` (reader `reader_unary_transpose_hc_interleaved_partitioned_rm.cpp`) — **Case 1** → re-express.
- `output` (writer `writer_unary_transpose_hc_interleaved_start_id_rm.cpp`) — **Case 1** → re-express.

*TransposeHCShardedProgramFactory:*
- `input` (`CBDescriptor::buffer = input_tensor.buffer()`) — **clean** → borrowed-memory DFB.
- `output` (`CBDescriptor::buffer = output_tensor.buffer()`) — **clean** → borrowed-memory DFB.

*TransposeCNProgramFactory:*
- `input` (reader `reader_unary_transpose_cn_interleaved_start_id.cpp`) — **Case 1** → re-express.
- `output` (writer `writer_unary_transpose_cn_interleaved_start_id.cpp`) — **Case 1** → re-express.

**Custom hash:** none — no deletion needed.

---

## Watch for

- **Borrowed-memory DFB:** Three factory pairs use `CBDescriptor::buffer` — `TransposeWHShardedProgramFactory` (`transpose_wh_sharded_program_factory.cpp:63, 76`), `TransposeWHShardedRMProgramFactory` (`transpose_wh_sharded_rm_program_factory.cpp:98, 112`), `TransposeHCShardedProgramFactory` (`transpose_hc_sharded_program_factory.cpp:326, 341`). Port declares each as `DataflowBufferSpec::borrowed_from = <tensor_parameter_name>`.

- **Dynamic TensorAccessor (`ArgConfig::RuntimeTensorShape`):** Eight sites across four active factories:
  - `transpose_wh_program_factory.cpp:229, 245`
  - `transpose_hc_tiled_interleaved_program_factory.cpp:192, 219`
  - `transpose_hc_rm_program_factory.cpp:154, 161`
  - `transpose_cn_program_factory.cpp:72, 78`
  
  Metal 2.0 translation: `TensorParameter::advanced_options.dynamic_tensor_shape = true` on the affected input/output `TensorParameter`s. This option is marked **UNSAFE** in the framework header. Adopting it has structural implications for the factory's interaction with per-dispatch caching. The default remains strict; apply only on explicit user-OK after reviewing the `port_op_to_metal2_ttnn_factory.md` relaxation guidance.

- **Cross-op / shared kernels — port-together sets:**
  1. `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — borrowed by `TransposeWHProgramFactory` (tiled path). Broadly shared across eltwise/unary and other families. A Metal 2.0 rewrite of this kernel must coordinate all co-borrowers in one change — cannot be ported in isolation.
  2. `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` — borrowed by `TransposeWHShardedProgramFactory`. Shared; trivial (no tensor address in kernel).
  3. `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` — borrowed by `TransposeWHShardedProgramFactory`. Shared; trivial.

- **RTA varargs:** none.
