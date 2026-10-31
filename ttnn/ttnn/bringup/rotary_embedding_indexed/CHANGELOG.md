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

### output_tensor: write a caller's output buffer (2026-10-30)
- What: `output_tensor=` (scalar form only, default None = the op allocates its output, unchanged): the op writes that tensor, whose spec must be exactly the output spec (checked on every call, miss and hit), and returns a tensor of its buffer. Not the input. Kernels and programs unchanged; outputs byte-identical to the allocating call.
- Why: each call allocated its output (~35 us of host). The model passes load-time buffers at the call sites whose
  output is read and dropped inside the layer. Measured: rotary_embedding_indexed 69.1 -> 50.8 us of host a call (8x4, 136 calls a chunk).
- Tests: `tests/unit/test_output_tensor.py` passed on the 8x4 (with the fork's uniform tests: 14 passed). Regression: unit 22 passed / 44 skipped, source 9 passed.
- Needed by: deepseek_v41_flash_d_p P.16 (FORK_OUT_BUFFERS, 2026-10-30)
- Files: `rotary_embedding_indexed.{hpp,cpp}`, `rotary_embedding_indexed_nanobind.cpp`, `device/rotary_embedding_indexed_device_operation.{hpp,cpp}`, `tests/unit/test_output_tensor.py`

### max_grid: the op's cores inside a rectangle (deepseek_v41_flash_d_p FIX.15)
- What: option `max_grid` (ttnn.CoreCoord, default None) on `ttnn.bringup.rotary_embedding_indexed`: every program of the op places its cores (and splits its work) inside (0, 0) .. (x - 1, y - 1) of the compute grid. Default: the device's whole grid, the call as before (same program, same hash).
- Why: agent rule 18 (owner 2026-10-30): a Galaxy chassis' power supply trips when a chip's current rises too fast (di/dt); no op may run on more than 110 of a chip's 120 Tensix cores. DeepSeek-V4.1 passes tt/cores.max_grid() = 11 x 10 on a Galaxy, None elsewhere.
- Measured: 8x4 Galaxy, one 5120-token chunk (rotary_embedding_indexed): 120 -> 110 cores a chip (tests/perf/test_core_audit.py in deepseek_v41_flash_d_p); per-op device time in that model's BREADCRUMBS FIX.15.
- Needed by: deepseek_v41_flash_d_p FIX.15
- Files: device/rotary_embedding_indexed_device_operation.cpp, device/rotary_embedding_indexed_device_operation.hpp, rotary_embedding_indexed.cpp, rotary_embedding_indexed.hpp, rotary_embedding_indexed_nanobind.cpp

### chain mode: input_heads, ROW_MAJOR in / out, output_dtype, group_heads (deepseek_v41_flash_d_p P.22)
- What: new scalar-form options `input_heads`, `output_layout`, `output_dtype`, `group_heads`, and a ROW_MAJOR input;
  any of them selects a second device operation (`device/rope_chain_device_operation.{hpp,cpp}`, a ProgramDescriptor
  factory, one program per mesh range, common runtime args only, addresses patched on hits) with its own kernels
  (`device/kernels/chain/`: reader, writer, compute). Heads read in place from a [B, 1, S, H W] TILE input; a
  ROW_MAJOR input tilized in compute; the RoPE region by rotary_embedding_llama's arithmetic step for step; the head's
  tiles assembled in the output format (bfloat8_b with bfp8_pack_precise = ttnn.typecast; bfloat4_b after the
  bfp_round SFPU round to nearest even); ROW_MAJOR output untilized in compute; grouped output pages for wo_a. Default
  (no option, TILE input): the existing program, unchanged.
- Why: DeepSeek-V4.1's three RoPE chains (attention q: nlp_create_q_heads_split -> rope -> untilize; attention output:
  to_layout -> inverse rope -> nlp_concat_heads -> typecast bfp8; indexer q: split -> rope -> bfp_round) were 2-3
  extra launches each over the same data (opportunities report #2, #4: ~7.7 ms device / ~15 ms host a chunk).
- Measured: bit-identical to the unfused chains on 8x4 (every chip, miss and hit); q chain host 259.9 -> 61.4 us a
  call; model warm wall 273.8 -> 250.1 ms a chunk (median of 3, same process, bit-identical output).
- Tests: `tests/unit/test_chain.py` (8 passed on the 8x4, 2026-10-31). Fork regression not run (OV-25: task F.1).
- Needed by: deepseek_v41_flash_d_p P.22 (ROPE_CHAINS)
- Files: `device/rope_chain_device_operation.{hpp,cpp}`, `device/kernels/chain/{reader_rope_chain,writer_rope_chain,
  compute_rope_chain}.cpp`, `device/kernels/chain/chain_share.hpp`, `rotary_embedding_indexed.{hpp,cpp}`,
  `rotary_embedding_indexed_nanobind.cpp`, `CMakeLists.txt`, `tests/unit/test_chain.py`
