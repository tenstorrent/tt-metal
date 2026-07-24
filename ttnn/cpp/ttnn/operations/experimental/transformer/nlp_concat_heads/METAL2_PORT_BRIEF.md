# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads`

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

**Tensor bindings** (interleaved path — both are Case 1):

- `input` (`in0_buffer`) — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. Kernel builds `TensorAccessor(ta::input)`. The `Buffer*`-as-RTA at `nlp_concat_heads_program_factory.cpp:201` and the `TensorAccessorArgs(*in0_buffer).append_to(...)` CTA injection at line 117 (plus the kernel-side `TensorAccessorArgs<4>()` at `reader_tm_tile_layout_nlp_concat_heads.cpp:27`) both disappear.

- `output` (`out_buffer`) — **Case 1** → re-express via `TensorParameter` / `TensorBinding`. Kernel builds `TensorAccessor(ta::output)`. The `Buffer*`-as-RTA at `nlp_concat_heads_program_factory.cpp:210` and the `TensorAccessorArgs(*out_buffer).append_to(...)` CTA injection at line 119 (plus the kernel-side `TensorAccessorArgs<1>()` at `writer_unary_interleaved_start_id.cpp:16`) both disappear.

**Sharded path:** input and output tensors accessed via fake borrowed-memory CBs — no `TensorParameter` bindings in that path; the fake-CB workaround handles them (see Watch for).

**Custom hash:** none

## Watch for

- **Notable constructs:**
  - Borrowed-memory DFB (sharded path): `CBDescriptor::buffer = in0_buffer` at `nlp_concat_heads_program_factory.cpp:150` (cb_id=0, sharded input) and `CBDescriptor::buffer = out_buffer` at line 163 (cb_id=16, sharded output). Port declares these as `DataflowBufferSpec::borrowed_from` naming the appropriate `TensorParameter`. These are also the fake CBs described below — apply both treatments.

- **Fake CBs (address-only, sharded path):** Two fake CBs in `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp`:
  - `(cb_in0=0, reader)`: `reserve_back` called but no `wait_front`/`pop_front`; used only as address source via `get_read_ptr()` (lines 35,43).
  - `(cb_out0=16, writer)`: `reserve_back` called, `push_back` commented out (`// cb_out0.push_back(block_size);` line 62); used only as address source via `get_write_ptr()` (lines 36,44).
  - Port resolves both with the sanctioned fake-CB workaround (see the porting recipe).

- **Cross-op / shared kernels:** The interleaved path instantiates `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (cross-family, owned by eltwise/unary, broadly shared by many ops). The Metal 2.0 rewrite of this kernel must be a single coordinated change across all co-borrowers. Do not migrate this kernel in isolation; coordinate with the eltwise/unary port and all other families that share it.

- **RTA varargs:** none
