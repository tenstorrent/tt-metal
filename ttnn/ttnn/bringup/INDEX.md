# Derived TTNN ops (bring-up forks)

When a model bring-up needs a change to an existing TTNN op, the op is forked here and changed here. The original op
is never edited. Each fork registers as `ttnn.bringup.<op>`, alongside the original.

Before forking, check this table. If the op is already here, reuse the fork, and put any new change behind an option
whose default keeps the fork's current behaviour. Each fork's `CHANGELOG.md` lists its source path and SHA, and every
change made to it (what, why, which model and task).

Python ops (a ProgramDescriptor op in a subfolder, no C++ build) are listed in `PYTHON_OPS` in `__init__.py`;
`ttnn.bringup.<name>` loads and registers them on first use. Their own unit suite lives in `<fork>/tests/unit/`; the
model cases (bring-up task O.1) are the top-level `tests/test_*.py`.

How to fork: `models/demos/common/bringup/skill/bringup-fork-op/SKILL.md`. In short: `python
ttnn/ttnn/bringup/fork_op.py <source op folder> --model <model> --task <task>`, then make the change, fill the
changelog, `./build_metal.sh`, and run the fork's tests (`<fork>/tests/`). Each fork also carries a best-effort selection of its
source op's own tests (`tests/source.yaml`, run in place with `testing/fork_source.py`; baseline in
`tests/source_baseline.json`). Every model that calls a fork adds a
random-input case to those tests for each call it makes: `models/demos/common/bringup/skill/bringup-fork-tests/SKILL.md`
(bring-up task O.1).

A person decides later whether a fork is ported back to the original op. When it is, the fork is deleted and its
users are switched back.

| Fork | Source @ SHA | Changes | Used by |
|---|---|---|---|
| `unified_routed_expert_ffn` | `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn` @ `67ca5f3af48` | GeluTanh activation; opt-in `high_precision` | gemma4_a4b_d_p, mimo_v2_6_d_p, mimo_v2_6_d_p_2x2 |
| `dispatch` | `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch` @ `67ca5f3af48` | 1-device dispatch axis (no fabric) | ernie45_d_p, gemma4_a4b_d_p, mimo_v2_6_d_p, mimo_v2_6_d_p_2x2 |
| `combine` | `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine` @ `67ca5f3af48` | 1-device dispatch axis (no fabric) | ernie45_d_p, gemma4_a4b_d_p, mimo_v2_6_d_p, mimo_v2_6_d_p_2x2 |
| `offset_cumsum` | `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/offset_cumsum` @ `67ca5f3af48` | 1-device dispatch axis (no all_gather) | ernie45_d_p, gemma4_a4b_d_p, mimo_v2_6_d_p, mimo_v2_6_d_p_2x2 |
| `rms_norm_ttnn` (`ttnn.bringup.rms_norm`) | ai-generated (codegen, `dnijemcevic/rms_norm_replacement_run0` @ `31b0b1cbc98`); drop-in for `ttnn.rms_norm` | ai-generated perf optimized version; C++ host side (device op + program factory, program cache), Python builder kept as the parity reference; opt-in `return_residual_sum` (also returns t = x + r, TILE, C++ op only); bug fix: fp32 sum-of-squares carry and fp32 x^2 / normalized CBs at fp32_dest_acc_en | mimo_v2_6_d_p, mimo_v2_6_d_p_2x2 |
| `sdpa` | `ttnn/cpp/ttnn/operations/transformer/sdpa` @ `99f7e834cea` | non-MLA V head dim narrower than K's (a narrower V tensor; output has V's width): scaled_dot_product_attention and chunked_scaled_dot_product_attention | mimo_v2_6_d_p (tt/attention.py, V 128; MIMO_V_PAD=1 keeps ttnn.transformer), mimo_v2_6_d_p_2x2 |
