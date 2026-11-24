# unified_routed_expert_ffn (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn`
- Source SHA: `4deb30807d45a6b7f0d853850e76e5247676e989`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.deepseek_prefill.*`)
- Forked for: qwen38_flash_next P.1
- Used by: qwen38_flash_next

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_unified_routed_expert_ffn`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### high_precision: caller's fidelity / fp32 dest, bf16 activations, Float32 partials
- What: new argument `high_precision` (default False = the source program, unchanged). True (ROW_MAJOR bf16
  dispatched buffer only): the compute kernel runs at compute_kernel_config's math_fidelity and fp32_dest_acc_en
  instead of LoFi / bf16 dest; the tilized x, the gate/up intermediates and the activated block are Float16_b instead
  of Bfp8_b; the gate/up and down partials (PACKER_L1_ACC accumulators) are Float32 when fp32 dest is on; the output
  is a fresh BFLOAT16 TILE buffer instead of BFLOAT8_B. The resolved fidelity and fp32 dest are op attributes (program
  cache key). Kernels unchanged: they take every format from their CBs.
- Why: qwen38_flash_next ran its 16 local experts per chip as an extract / ttnn.linear / insert loop over max_tokens
  (5120) rows each at HiFi4 + fp32 dest (136.6 ms, 50% of the warm 4-layer 51200->56320 chunk); the source fused op
  bounds each expert to its occupied rows but forces LoFi and bf16/bfp8 intermediates (known_issues: the fused Silu
  kernel fails the norm-ratio check; bf16 L1 partials). With high_precision at HiFi4 + fp32 dest the three experts
  component tests keep their accuracy (layer 0 vs CPU rel 0.0069, was 0.0068 with the loop).
- Needed by: qwen38_flash_next P.1
- Files: unified_routed_expert_ffn.cpp, unified_routed_expert_ffn.hpp, unified_routed_expert_ffn_nanobind.cpp,
  device/unified_routed_expert_ffn_types.hpp, device/unified_routed_expert_ffn_device_operation.cpp,
  device/unified_routed_expert_ffn_device_operation.hpp, device/unified_routed_expert_ffn_program_factory.cpp
