# sdpa (fork)

- Source: `ttnn/cpp/ttnn/operations/transformer/sdpa`
- Source SHA: `473263d9cd7bfa06834955003eff772c38bcf9a3`
- Python: `ttnn.bringup.*` (was `ttnn.transformer.*`)
- Forked for: mimo_v26_flash_rl C.attn_dense.attention
- Used by: mimo_v26_flash_rl

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_sdpa`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

Manual fixes after fork_op.py (the copy did not build): `sdpa_config.hpp` is not forked, so the fork's headers that
include it alias `SDPAProgramConfig` / `PagedCacheGeometryOverride` into `ttnn::operations::bringup`; the double-nested
`detail::bringup`, `sdpa_cb::bringup` and `transformer::bringup::sdpa::bringup` namespaces are back to one level;
`compute_sdpa_ideal_cycles*`, `SparseKVFormat` and the `*_SDPA_*_IDX` constants are qualified with the fork's
namespaces. `tests/source.yaml` carries no source tests yet (`tests: []`).

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### exact_exp: accurate softmax exp in the streaming kernel
- What: new argument `exact_exp` (default False) of `ttnn.bringup.ring_joint_scaled_dot_product_attention`. When True
  the program factory defines `SDPA_EXACT_EXP` (streaming compute only, TT_FATAL otherwise) and
  `compute_streaming.hpp` `sub_exp_block_bcast_cols` uses the accurate bf16 exp (`exp_packthread_tile<false, ...>`,
  ClampToNegative, the scale passed as its bf16 scale factor) instead of the hard-coded Schraudolph approximation
  `exp_packthread_tile<true, ...>`. False: no define, so the source op's program (same kernels, args and CBs). The flag
  is in the program hash.
- Why: MiMo full attention (layer 0, chunk 1, 16 heads x 640 rows per chip, K 192 / V 128, kv_actual_isl metadata
  path, so streaming compute with bf16 dest is mandatory) had 1.3% rel error on the SDPA output against fp32 SDPA on
  the same device q / k / v. A CPU emulation of the Schraudolph exp on those inputs halves the distance to the device
  output, and predicts o_proj output column 3525 at rel 0.070 with it vs 0.006 with an accurate exp (component check
  limit 0.021; device measured 0.063). On the device the SDPA output error vs fp32 SDPA on the same inputs went from
  0.0130 to 0.0064, and the C.attn_dense.attention gate margin (with the model's other changes) to 1.003.
- Needed by: mimo_v26_flash_rl C.attn_dense.attention
- Files: sdpa_nanobind.cpp, sdpa.hpp, sdpa.cpp, device/ring_joint_sdpa_device_operation.{hpp,cpp},
  device/ring_joint_sdpa_device_operation_types.hpp, device/ring_joint_sdpa_program_factory.cpp,
  device/kernels/compute/compute_streaming.hpp
