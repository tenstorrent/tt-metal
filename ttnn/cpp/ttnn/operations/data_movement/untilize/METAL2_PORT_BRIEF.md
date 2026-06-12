# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/untilize`

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

**Tensor bindings** (per binding):

Six factories that use interleaved or DRAM-sharded inputs/outputs need binding re-expression:

- `src0_buffer` in `UntilizeSingleCoreProgramFactory` — **Case 1** → re-express via `TensorParameter` / `TensorBinding` (kernel builds `TensorAccessor(ta::name)`). CTA-baked `TensorAccessorArgs` and `Buffer*` RTA slot both disappear.
- `dst_buffer` in `UntilizeSingleCoreProgramFactory` — **Case 1** → same treatment.
- `src0_buffer` in `UntilizeMultiCoreSubCoreGridsProgramFactory` — **Case 1** → re-express.
- `dst_buffer` in `UntilizeMultiCoreSubCoreGridsProgramFactory` — **Case 1** → re-express.
- `src0_buffer` in `UntilizeMultiCoreBlockProgramFactory` — **Case 1** → re-express.
- `dst_buffer` in `UntilizeMultiCoreBlockProgramFactory` — **Case 1** → re-express.
- `src0_buffer` in `UntilizeMultiCoreParallelizeColumnProgramFactory` — **Case 1** → re-express.
- `dst_buffer` in `UntilizeMultiCoreParallelizeColumnProgramFactory` — **Case 1** → re-express.
- `src0_buffer` (interleaved + block-reader sharding paths) in `UntilizeMultiCoreProgramFactory` — **Case 1** → re-express.
- `dst_buffer` in `UntilizeMultiCoreProgramFactory` — **Case 1** → re-express.
- `src0_buffer` / `dst_buffer` in even-sharding path of `UntilizeMultiCoreProgramFactory` — **clean** (borrowed-memory DFB via `CBDescriptor::buffer`; port uses `DataflowBufferSpec::borrowed_from`).
- `src0_buffer` in `UntilizeMultiCoreNDShardInputProgramFactory` — **Case 1** → re-express. Note: this binding appears in both reader and writer RTAs; both slots disappear.
- `dst_buffer` in `UntilizeMultiCoreNDShardInputProgramFactory` — **Case 1** → re-express.
- `src0_buffer` (input CB) in `UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory` — **clean** (borrowed-memory DFB).
- `dst_buffer` (output CB) in `UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory` — **clean** (borrowed-memory DFB).
- `src0_buffer` (input CB) in `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory` — **clean** (borrowed-memory DFB).
- `dst_buffer` (output CB) in `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory` — **clean** (borrowed-memory DFB).

**Custom hash:** none

## Watch for

- **Notable constructs:** Dynamic CircularBuffer (borrowed memory) — present in three factories (`UntilizeMultiCoreProgramFactory` even-sharding path `untilize_multi_core_program_factory.cpp:118–129`, and the two identical-shard-spec factories `untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.cpp:54–79` and `untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.cpp:68–97`). These use `CBDescriptor::buffer = <buffer>`. Port each affected CB as `DataflowBufferSpec::borrowed_from = <tensor_parameter_name>`. All have genuine producer + consumer pairs (not fake CBs).

- **Cross-op / shared kernels:** Six borrowed kernel files. Each borrowed file's Metal 2.0 rewrite is a single rewrite that all co-borrowing ops must adopt together — coordinate with ops that share these files before landing any per-file Metal 2.0 changes. Key files:
  - `eltwise/unary/.../reader_unary_sharded.cpp` — broadly shared across eltwise and data_movement
  - `eltwise/unary/.../reader_unary_interleaved_start_id.cpp` — broadly shared
  - `eltwise/unary/.../reader_unary_interleaved_wh_multicore.cpp` — shared with other data_movement ops
  - `data_movement/sharded/.../writer_unary_sharded.cpp` — shared within data_movement
  - `data_movement/sharded/.../reader_unary_nd_sharded_blocks.cpp` — shared within data_movement
  - `data_movement/untilize_with_unpadding/.../writer_unary_stick_layout_wh_multicore.cpp` — shared with untilize_with_unpadding

- **RTA varargs:** none
