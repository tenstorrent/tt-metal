# moe_ag (ttnn.bringup.moe_ag_*)

- Source: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_ag` on `mstaletovic/mimo-v2-dp` @ `b45a923d73c`. New op: it exists on that branch only, not in ttnn/cpp on this tree (no source op to keep in sync,
  no upstream tests to carry: `tests/source.yaml` is empty).
- Python: `ttnn.bringup.*` (was `ttnn.experimental.*` on the source branch)
- Brought in for: glm53_flash_d_p_lb (MoE on the LoudBox, the MiMo-V2 all-gather MoE design: same hidden / expert
  width / top-8 as MiMo-V2.6)
- Used by: glm53_flash_d_p (tt/experts_ag.py, GLM_EXPERTS_MODE=ag; the whole model on the LoudBox via glm53_flash_d_p_lb)

The device programs of the all-gather MoE block (MiMo-V2 `tt/moe_ag.py` on the source branch): `moe_ag_route_plan`
(counts / regions of the flat expert space, token_index, y_slot), `moe_ag_local_reduce` (per token the weighted sum of
this chip's experts' outputs, with the two fused send-back phases), `moe_ag_sum_rows_tiled`, `moe_ag_add_rows`,
`moe_ag_untilize_active`, `moe_ag_untilize_x`.

Mechanical changes (fork_op.py on a temporary checkout of the source folder): namespace `ttnn::operations::bringup`,
CMake target `ttnn_op_bringup_moe_ag`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

Tests: `tests/unit/test_moe_ag_ops.py` (the source branch's nightly op test, `tests/ttnn/nightly/.../deepseek_prefill/test_moe_ag_ops.py`).

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->
