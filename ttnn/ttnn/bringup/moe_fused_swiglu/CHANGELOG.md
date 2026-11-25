# moe_fused_swiglu (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu`
- Source SHA: `8c48b4e4af43c481b79608d76e6d6915e2e7a8df`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.deepseek_prefill.*`)
- Forked for: qwen3_6_35b_a3b, ornith_1_5_35b_a3b agentic-models QB2 vLLM
- Used by: qwen3_6_35b_a3b, ornith_1_5_35b_a3b

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_moe_fused_swiglu`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### Pin compatibility (tt-metal 58afa857)
- What: the source is taken at 8c48b4e4 (it carries #62503 / 40e63324497: stochastic FPU rounding, perf and
  correctness fixes the pinned op lacks) and built against the pinned tree. `device/kernels/fast_silu_glu_sfpu.h` is
  a copy of `deepseek_prefill/unified_routed_expert_ffn/device/kernels/compute/fast_silu_glu_sfpu.h` @ 8c48b4e4
  (absent at the pin), included by the compute kernel from this folder. The d2ec50872cf `Tensor::mesh_buffer()`
  property reads are reverted to the pin's `Tensor::buffer()->...` API. `RoutedExpertActivation` is the pinned
  `deepseek_prefill::unified_routed_expert_ffn` enum (fully qualified), and the prim launches this fork's device
  operation. `device/kernels/moe_fused_swiglu_compute_helpers.hpp` keeps the source's kernel namespaces
  (`ckernel::sfpu`, `moe_fused_swiglu::compute`): fork_op.py's `::bringup` suffix there left the compute kernel's
  `using namespace` and calls unresolved (JIT compile error on first launch); device kernels are compiled per program,
  so they cannot collide with the pinned op's. No behaviour change versus the source.
- Why: the pinned `ttnn.experimental.deepseek_prefill.moe_fused_swiglu` predates #62503 and cannot serve the models;
  the pinned op stays untouched for its other callers.
- Needed by: qwen3_6_35b_a3b, ornith_1_5_35b_a3b (agentic-models QB2 vLLM)
- Files: device/kernels/fast_silu_glu_sfpu.h, device/kernels/moe_fused_swiglu_compute.cpp,
  device/kernels/moe_fused_swiglu_compute_helpers.hpp, CMakeLists.txt,
  device/moe_fused_swiglu_types.hpp, device/moe_fused_swiglu_device_operation.cpp,
  device/moe_fused_swiglu_program_factory.cpp

### stochastic_rounding (qwen3_6_35b_a3b bring-up 69a03c85)
- What: new argument `stochastic_rounding` (default True = the source's stochastic FPU rounding into bf16 DEST).
  False configures `StochRndType::None`. Emitted as the `MOE_FUSED_SWIGLU_STOCHASTIC_ROUNDING` compute define and
  part of the op attributes (program cache key).
- Why: decode needs output independent of other experts' token counts (the PRNG position depends on the schedule);
  qwen3_6_35b_a3b passes `stochastic_rounding=not batched`.
- Needed by: qwen3_6_35b_a3b
- Files: moe_fused_swiglu.cpp, moe_fused_swiglu.hpp, moe_fused_swiglu_nanobind.cpp, device/moe_fused_swiglu_types.hpp,
  device/moe_fused_swiglu_device_operation.cpp, device/moe_fused_swiglu_device_operation.hpp,
  device/moe_fused_swiglu_program_factory.cpp, device/kernels/moe_fused_swiglu_compute.cpp

### End-of-reader NoC drain (qwen3_6_35b_a3b 79b630e1 + ornith_1_5_35b_a3b 1254c40d)
- What: the reader ends with `noc.async_write_barrier(); noc.async_atomic_barrier();` (unconditional: every caller of
  this fork is a traced vLLM serving model).
- Why: reader-owned metadata / final H multicasts and their acknowledgements must complete before the next kernel
  snapshots its NoC counter baseline (hangs/corruption under traced serving, including the all-experts-empty case).
- Needed by: qwen3_6_35b_a3b, ornith_1_5_35b_a3b
- Files: device/kernels/moe_fused_swiglu_reader.cpp

Not taken: ornith_1_5_35b_a3b ab9905df (separate w_down dtype / DownWeight format key); both models run bfp8 gate/up
and down.
