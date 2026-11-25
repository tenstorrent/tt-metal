# post_combine_reduce (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce`
- Source SHA: `8c48b4e4af43c481b79608d76e6d6915e2e7a8df`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.deepseek_prefill.*`)
- Forked for: qwen3_6_35b_a3b, ornith_1_5_35b_a3b agentic-models QB2 vLLM
- Used by: qwen3_6_35b_a3b, ornith_1_5_35b_a3b

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_post_combine_reduce`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### Pin compatibility (tt-metal 58afa857)
- What: the source is taken at 8c48b4e4: it carries f29e68e759f (#62565: per-token packed weights,
  `combine_row_index`, `skip_row`) and 1e924097413 (#61552: batched per-token NoC reads), which the pinned
  op lacks. The d2ec50872cf `Tensor::mesh_buffer()` page-size reads are reverted to the pin's `Tensor::buffer()->...`
  API. No behaviour change versus the source.
- Why: the models call it with packed `(1,1,T,8)` weights; the pinned op stays untouched for its other callers.
- Needed by: qwen3_6_35b_a3b, ornith_1_5_35b_a3b (agentic-models QB2 vLLM)
- Files: device/post_combine_reduce_program_factory.cpp

### valid_token_rows (qwen3_6_35b_a3b bring-up 08384c10)
- What: new argument `valid_token_rows` (default 0 = the source program, every row of the chunk). 1..32 computes only
  that prefix of a single 32-row chunk on the dispatch-table path (no `combine_row_index`, expert dim right before the
  embedding, embedding % 1024 == 0) and writes positive zero to the remaining output rows. Op attribute (program cache
  key) and a trailing compile-time arg of all three kernels.
- Why: fast decode runs one padded 32-row chunk with only the live rows meaningful.
- Needed by: qwen3_6_35b_a3b
- Files: post_combine_reduce.cpp, post_combine_reduce.hpp, post_combine_reduce_nanobind.cpp,
  device/post_combine_reduce_types.hpp, device/post_combine_reduce_device_operation.cpp,
  device/post_combine_reduce_device_operation.hpp, device/post_combine_reduce_program_factory.cpp,
  device/kernels/deepseek_moe_post_combine_reduce_{reader,compute,writer}.cpp
