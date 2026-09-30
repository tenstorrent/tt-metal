# unified_routed_expert_ffn (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn`
- Source SHA: `67ca5f3af4860c04e7ce31e345107f51cb22aa9b`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.deepseek_prefill.*`)
- Forked for: gemma4_a4b_d_p C.sliding.experts
- Used by: gemma4_a4b_d_p, mimo_v2_6_d_p

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_unified_routed_expert_ffn`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

## Changes

### GeluTanh activation, with the requested fidelity and fp32 dest
- What: `RoutedExpertActivation.GeluTanh = 4` (`ROUTED_GELU_TANH` define in the compute kernel):
  down(gelu_tanh(gate) * up). For this variant the factory honours `compute_kernel_config`'s math_fidelity and
  fp32_dest_acc_en (in the program-cache key); the other variants keep the source's LoFi and bf16 dest.
- Why: Gemma-4 experts are GeGLU and the op had no GeLU choice; the forced LoFi / bf16 dest put per-token norms out
  of range (ratio 0.96 at LoFi; HiFi2 + fp32 dest gives 1.011).
- Needed by: gemma4_a4b_d_p C.sliding.experts (originally commit ce75b1f0d7f)
- Files: `device/kernels/compute/fused_swiglu.cpp`, `device/unified_routed_expert_ffn_program_factory.cpp`,
  `device/unified_routed_expert_ffn_types.hpp`, `unified_routed_expert_ffn_nanobind.cpp`

### Opt-in `high_precision` (every activation)
- What: `unified_routed_expert_moe(..., high_precision=True)` (default False = previous behaviour). It honours
  `compute_kernel_config`'s math_fidelity and fp32_dest_acc_en for every activation, and keeps x, the
  intermediates and the output in bf16 (ROW_MAJOR x only; the returned buffer is TILE bf16 instead of bf8_b). The
  flag is in the program-cache key.
- Why: MiMo's Silu experts failed the per-token norm-ratio check at the forced LoFi / bf16 dest, and layer 5's
  outlier channels fail with the bf8 activations the op packs internally.
- Needed by: mimo_v2_6_d_p P.1 (originally commit e3150c04460)
- Files: `unified_routed_expert_ffn.{hpp,cpp}`, `unified_routed_expert_ffn_nanobind.cpp`,
  `device/unified_routed_expert_ffn_device_operation.{hpp,cpp}`, `device/unified_routed_expert_ffn_program_factory.cpp`,
  `device/unified_routed_expert_ffn_types.hpp`

### Tests: fork test suite with a first model case
- What: `tests/` (cases.py, reference.py, test file) with a random-input case for the call mimo_v2_6_d_p makes
  (bringup-fork-tests skill). No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: mimo_v2_6_d_p O.1
- Files: `tests/__init__.py`, `tests/cases.py`, `tests/reference.py`, `tests/test_*.py`

### Tests: gemma4_a4b_d_p case
- What: appended the random-input case for the call gemma4_a4b_d_p makes (1x4 mesh, S 5120, H 2816, 128 experts
  top-8, 32 per chip). No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: gemma4_a4b_d_p O.1
- Files: `tests/cases.py`

### Tests: hy4_preview_d_p case (ClampedSiluGlu)
- What: appended the random-input case for the call hy4_preview_d_p makes (2x2 mesh, H 6144, I 2048, 64 experts per
  chip, bfp8 weights, ClampedSiluGlu, high_precision, HiFi4 + fp32 dest). The reference gains the ClampedSiluGlu
  activation (silu(min(gate, 10)) * clamp(up, -10, 10), `GLU` in tests/reference.py) and the test an optional
  per-case `x_scale` (default 1, so the other cases keep their inputs) that widens x until the clamps are reached.
  No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: hy4_preview_d_p O.1
- Files: `tests/cases.py`, `tests/reference.py`, `tests/test_unified_routed_expert_ffn.py`
