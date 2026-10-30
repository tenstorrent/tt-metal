# rotary_embedding_indexed (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed`
- Source SHA: `d2dc8639a5fa9c21c60a52cfd3f01619ceb1d592`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.deepseek_prefill.*`)
- Forked for: deepseek_v41_flash_d_p P.5
- Used by: deepseek_v41_flash_d_p

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_rotary_embedding_indexed`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### uniform: one program for the mesh
- What: new argument `uniform` (scalar form, default False = the source's one program per mesh coordinate). When it
  is True, `create_mesh_workload` builds one program per coordinate range, at the range's first coordinate. It is in
  `compute_program_hash`. `validate_runtime_args` (miss and hit) refuses it with a metadata tensor, with
  kv_actual_global != 0, or with seq_subshard_axis. Kernels unchanged.
- Why: the stock op builds 32 programs on an 8x4 mesh (`my_sp_coord` is a compile-time arg), and every hit updates
  all 32. With per-chip pre-sliced cos / sin and kv_actual_global 0, update_idxt is 0 on every chip, so the programs
  are identical. Host time on 8x4 goes from 101 to 52 us a call. DeepSeek-V4.1 makes 136 calls a chunk: warm wall
  438.6 -> 399.7 ms (median of 6, same process). Output byte-identical.
- Needed by: deepseek_v41_flash_d_p P.5 (ROPE_UNIFORM_PROGRAM)
- Files: device/rotary_embedding_indexed_device_operation.{hpp,cpp}, rotary_embedding_indexed.{hpp,cpp},
  rotary_embedding_indexed_nanobind.cpp, tests/unit/test_uniform.py, tests/source.yaml, tests/source_baseline.json
