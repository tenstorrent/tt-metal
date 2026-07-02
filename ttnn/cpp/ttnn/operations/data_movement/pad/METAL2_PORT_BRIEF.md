# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/pad`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓

## TTNN factory analysis

The factory concept is selected downstream from these facts (→ `port_op_to_metal2_ttnn_factory.md`); the port does not pick it here. Carry these forward:

- **Op-owned tensors:** Yes — `PadRmReaderWriterMultiCoreProgramFactory::create_workload_descriptor` and `PadRmReaderWriterProgramFactory::create_workload_descriptor` each allocate a pad-value const tensor (32-element bfloat16 L1 tensor) inside the factory and park it on `WorkloadDescriptor::buffers`. These two factories are currently on the MeshWorkload path as a plumbing artifact (see MeshWorkload below).
- **MeshWorkload:** Not a genuine need — op-owned-tensor artifact only. The two `create_workload_descriptor` factories use `WorkloadDescriptor` solely to keep the pad-value const tensor alive across cache hits. The other five factories use plain `ProgramDescriptor`. Downstream concept selection should treat the op as not genuinely requiring MeshWorkload.
- **Pybind `create_descriptor`:** None.
- **Other risky pybind:** None.
- **Custom `override_runtime_arguments`:** None.

## Construct — to do

**Tensor bindings** (per binding):

- `input` (all interleaved and tile factories) — **Case 1** → re-express via `TensorParameter`/`TensorBinding`. Host pushes `src0_buffer`/`input_buffer` as `Buffer*` into `RTArgList`; kernel extracts `get_arg_val<uint32_t>(0)` and constructs `TensorAccessor(src_args, src_addr)`. Replace with `TensorAccessor(ta::input)`.
- `output` (all interleaved and tile factories) — **Case 1** → re-express via `TensorParameter`/`TensorBinding`. Same `Buffer*` pattern; kernel constructs `TensorAccessor(dst_args, dst_addr)`. Replace with `TensorAccessor(ta::output)`.
- `pad_value_const_tensor` (two WorkloadDescriptor factories only) — **Case 1** → re-express via `TensorParameter`/`TensorBinding`. `pad_value_const_buffer` pushed as `Buffer*` at RT-arg slot 13 in both factories; kernel extracts at slot 13 and constructs `TensorAccessor(pad_tensor_args, pad_value_const_buffer_addr)`. Replace with `TensorAccessor(ta::pad_const)`. Requires resolving the op-owned-tensor factory concept first — the binding injection infrastructure must see this tensor's `TensorParameter`.
- `input shard` / `output shard` (sharded factories) — **clean** (borrowed-memory DFBs via `CBDescriptor::buffer`). Port uses `DataflowBufferSpec::borrowed_from` for `cb_src0`, `cb_output`, `cb_input`, `cb_output` across the two sharded factories. Three of these CBs are fake — see Watch for below.

**Custom hash:** None.

## Watch for

- **Dynamic CircularBuffer (borrowed memory):** Four CBs across two sharded factories use `CBDescriptor::buffer`:
  - `PadRmShardedHeightOnlyProgramFactory`: `cb_src0` (c_0) @ `pad_rm_sharded_height_only_program_factory.cpp:289`, `cb_output` (c_16) @ line 304.
  - `PadRmShardedWidthOnlyProgramFactory`: `cb_input` (c_0) @ `pad_rm_sharded_width_only_program_factory.cpp:75`, `cb_output` (c_16) @ line 91.
  - Port uses `DataflowBufferSpec::borrowed_from` for each. Three of these four CBs are **fake** (see below); only `cb_output` (c_16) in `PadRmShardedWidthOnlyProgramFactory` is a real DFB.

- **Fake CBs (address-only) — apply the sanctioned fake-CB workaround from the porting recipe:**
  - `(cb_in0/c_0, height-only sharded factory)` @ `pad_rm_sharded_height_only_program_factory.cpp:289` — kernel reads `cb_in0_exp.get_write_ptr()` as a raw L1 address, no FIFO operations. Fake CB.
  - `(cb_out0/c_16, height-only sharded factory)` @ `pad_rm_sharded_height_only_program_factory.cpp:304` — reader and writer both bulk-reserve and directly address L1; no producer-consumer FIFO. Fake CB.
  - `(cb_input/c_0, width-only sharded factory)` @ `pad_rm_sharded_width_only_program_factory.cpp:75` — kernel reads `cb_input_shard.get_write_ptr()` as raw L1 address, no FIFO operations. Fake CB.

- **Cross-op / shared kernels:** `PadTileCoreProgramFactory` instantiates `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` (cross-family, owned by `eltwise/unary`). Any Metal 2.0 kernel rewrite (DFB tokens, named bindings) must be coordinated with all consumers of that kernel file — this forms a **port-together set** with at least `eltwise/unary`. Do not rewrite this kernel in the pad port without confirming with the eltwise/unary porter.

- **RTA varargs:** None.
