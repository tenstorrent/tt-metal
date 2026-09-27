# Derived TTNN ops (bring-up forks)

When a model bring-up needs a change to an existing TTNN op, the op is forked here and changed here. The original op
is never edited. Each fork registers as `ttnn.bringup.<op>`, alongside the original.

Before forking, check this table. If the op is already here, reuse the fork, and put any new change behind an option
whose default keeps the fork's current behaviour. Each fork's `CHANGELOG.md` lists its source path and SHA, and every
change made to it (what, why, which model and task).

Forking: `python ttnn/ttnn/bringup/fork_op.py <source op folder> --model <model> --task <task>`, then make the change,
fill the changelog, `./build_metal.sh`, and run the fork's tests (`<fork>/tests/`).

A person decides later whether a fork is ported back to the original op. When it is, the fork is deleted and its
users are switched back.

| Fork | Source @ SHA | Changes | Used by |
|---|---|---|---|
| `unified_routed_expert_ffn` | `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn` @ `67ca5f3af48` | GeluTanh activation; opt-in `high_precision` | gemma4_a4b_d_p, mimo_v2_6_d_p |
| `dispatch` | `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch` @ `67ca5f3af48` | 1-device dispatch axis (no fabric) | ernie45_d_p, gemma4_a4b_d_p, mimo_v2_6_d_p |
| `combine` | `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine` @ `67ca5f3af48` | 1-device dispatch axis (no fabric) | ernie45_d_p, gemma4_a4b_d_p, mimo_v2_6_d_p |
| `offset_cumsum` | `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/offset_cumsum` @ `67ca5f3af48` | 1-device dispatch axis (no all_gather) | ernie45_d_p, gemma4_a4b_d_p, mimo_v2_6_d_p |
