# flat_routed_expert_ttnn (ttnn.bringup.flat_routed_expert, ttnn.bringup.flat_routed_expert_plan)

- Source: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/flat_routed_expert` on `mstaletovic/mimo-v2-dp` @ `b45a923d73c`. New op: it exists on that branch only, not in ttnn/cpp on this tree (no source op to keep in sync,
  no upstream tests to carry: `tests/source.yaml` is empty).
- Python: `ttnn.bringup.*` (was `ttnn.experimental.*` on the source branch)
- Brought in for: glm53_flash_d_p_lb (MoE on the LoudBox, the MiMo-V2 all-gather MoE design: same hidden / expert
  width / top-8 as MiMo-V2.6)
- Used by: glm53_flash_d_p (tt/experts_ag.py, GLM_EXPERTS_MODE=ag; the whole model on the LoudBox via glm53_flash_d_p_lb)

Flat spatially pipelined routed experts (Blackhole): every local expert's SwiGLU FFN in one program that streams each
expert's weights once (DRAM bank readers -> gate/up cores, x relays, down cores writing bfp8 y at each expert's region);
counts and regions read on device, so one cached program serves every routing; bfp8 or bfp4 weights; activations
silu / swigluoai / situ / clamped_silu (DeepSeek-V4 limit 10, GLM-5.3's) / gelu_tanh.
The folder is not named flat_routed_expert: it is also a Python package (a package of the op's own name would shadow
ttnn.bringup.flat_routed_expert). Python side, from the source branch's `models/demos/mimo_v2_d_p/tt/flat_expert.py`:
`flat_expert.py` (`FlatRoutedExpert`: weights in the op's bank layout and the per-device expert tables, the
model-facing wrapper; `FlatExpert`: the Python ProgramDescriptor builder, the parity reference), its research kernels
in `py_kernels/stream_mm` (outside device/kernels, not installed with the op), the three helpers it took from MiMo perf
tests in `_helpers.py`, and the design / measurements log `FLAT_EXPERT_WORKLOG.md` (paths in it are the source branch's).

Mechanical changes (fork_op.py on a temporary checkout of the source folder): namespace `ttnn::operations::bringup`,
CMake target `ttnn_op_bringup_flat_routed_expert_ttnn`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

Tests: `tests/unit/` (the source branch's `models/demos/mimo_v2_d_p/tests/unit/test_flat_*`: op vs Python builder, indexed
shapes, mesh, row-major y).

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### fp32 DEST down projection (`down_fp32`)
- What: `down_fp32=True` (op keyword, `FlatRoutedExpert.__call__`; config attribute, so a separate cached program)
  compiles the down computes (se6_dcompute on the down cores and the reader tails) with `SE_DN_FP32` +
  `SE_DN_FULL_SYNC` and `fp32_dest_acc_en` / `dst_full_sync_en`: the Python builder's `MIMO_FL_DN_ACC=fp32full`, which
  the C++ op did not expose. Default off: the source program, unchanged. No effect on the plan or the weight layout.
- Why: bf16 DEST over K = I inflates y with bfp4 weights (FLAT_EXPERT_WORKLOG.md: "flat (fp32 gate/up, bf16 down)"
  q norm 1.030-1.042). GLM-5.3 at its shape (H 4096, I 2048, 36 experts, bfp4): one chip, N(0, 0.02) weights vs the
  quantized-weight reference, coefficient 1.0337 -> 1.0105 (rel 0.071 -> 0.057); layer 4's golden through the whole
  MoE, 1.048 -> 1.024 (unified 1.003); same kernel time at chunk 2048 (2.34 ms per MoE call). With bfp8 weights
  it overshoots (1.0017 -> 0.9796): the rest of the gain is elsewhere (bfp8 x / h), so it is opt-in.
- Needed by: glm53_flash_d_p (tt/experts_ag.py, GLM_AG_DOWN_FP32=1 default)
- Files: device/flat_routed_expert_plan.hpp, device/flat_routed_expert_program_factory.cpp, flat_routed_expert.cpp /
  .hpp, flat_routed_expert_nanobind.cpp, flat_expert.py
