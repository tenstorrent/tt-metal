# moe_grouped_topk (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk`
- Source SHA: `f3149480fb1ea88ee719599f69600eee49e4069d`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.deepseek_prefill.*`)
- Forked for: deepseek_v41_flash_d_p P.22
- Used by: deepseek_v41_flash_d_p

The source at that SHA already holds Marko Staletovic's fp32_selection commits (71062e76a38 and 570d95b6e7d on
mstaletovic/galaxy-unified-k3-v41-combine; here 5e47659815e, 27c5c2a16ae and the rebase fix 9780c18f0d0).

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_moe_grouped_topk`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### fp32_activation: the fp32 softplus for sqrtsoftplus
- What: new argument `fp32_activation` (default False = the source program, the define absent): the compute kernel
  is built with `INP_FLOAT32`, so `softplus_tile` takes its fp32 branch (degree-8 polynomial on [0, 5] + the exp
  tail), as the stock unary op does for an fp32 input. Only softplus reads the define here (sigmoid, sqrt, the sort
  and the normalization are unchanged). Added to the operation attributes (program-cache key).
- Why: with the bf16 branch (degree 6, 0 below -5) sqrt(softplus(x)) is ~8e-4 relative off on [-3, 9] and up to
  0.08 absolute below; DeepSeek-V4.1's scaled golden (2x2, 8 layers, 2048 rows): 0-13 rows a layer selected other
  experts than the model's op chain (stock ttnn.softplus / ttnn.sqrt / ttnn.topk on fp32), keys max error 8.2e-2 vs
  2.5e-5. With the option: the same experts on every row of every layer, keys bit-identical to stock
  ttnn.sqrt(ttnn.softplus(x)); weights one bf16 rounding from the chain's fp32 weights. The 5 realistic
  sqrtsoftplus nightly failures noted in 71062e76a38 (weights PCC 0.948-0.960) came from the same polynomial
  (they pass at this SHA's tests).
- Needed by: deepseek_v41_flash_d_p P.22 (router through the fused gate, ROUTER_IMPL=fused, ROUTER_SOFTPLUS=fp32)
- Files: moe_grouped_topk.hpp, moe_grouped_topk.cpp, moe_grouped_topk_nanobind.cpp,
  device/moe_grouped_topk_device_operation.hpp, device/moe_grouped_topk_device_operation.cpp,
  device/moe_grouped_topk_program_factory.cpp; tests: tests/unit/test_fp32_activation.py (7 cases; the option off
  fails its 5e-5 activation bar at 8.2e-2), tests/source.yaml + source_baseline.json (203 / 203 both)
