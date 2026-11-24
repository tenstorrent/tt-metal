# sdpa (fork)

- Source: `ttnn/cpp/ttnn/operations/transformer/sdpa`
- Source SHA: `b5ce2601ea32fe06238509093250484a33b108ff`
- Python: `ttnn.bringup.*` (was `ttnn.transformer.*`)
- Forked for: kimi_k27_code C.attn_dense.attention
- Used by: kimi_k27_code

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_sdpa`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### Build fixes after the mechanical fork
- What: put back the types the fork shares with the source op: `ttnn::operations::transformer::SDPAProgramConfig` /
  `PagedCacheGeometryOverride` (sdpa_config.hpp, not forked; `using` declarations in sdpa_nanobind.cpp); folded the
  doubled nestings `prim::bringup::detail::bringup`, `prim::bringup::sdpa_cb::bringup` and
  `transformer::bringup::sdpa::bringup` back to one `bringup` level; qualified the relative references
  `transformer::SparseKVFormat`, `prim::*_SDPA_*_IDX` and `operations::transformer::sdpa::compute_sdpa_ideal_cycles*`
  with the fork's namespaces. No behaviour change.
- Why: the fork did not compile as fork_op.py left it.
- Needed by: kimi_k27_code C.attn_dense.attention
- Files: sdpa_nanobind.cpp, sdpa.cpp, device/*.hpp, device/*.cpp, device/kernels/

### ring_mla (latent V) at fp32 DEST
- What: with `fp32_dest_acc_en=True` in the compute config, latent-V ring attention (`ttnn.bringup.ring_mla`, V read
  from K's prefix) runs the streaming compute path at fp32 DEST, keeping that path's bf16 intermediate CBs
  (sum / qk_im stay bf16). The source refuses it ("Latent-V ring attention is implemented only for streaming compute"
  and "kv_actual_isl requires the ring-joint streaming compute path"). Off (fp32_dest_acc_en False, or tensor-V
  ring joint SDPA at either setting) the program is the source's.
- Why: Kimi-K2.7 layer-0 dense MLA (16 heads / chip, K 576, V 512, 8 SP x 4 TP, bfp8 cache, prefix 5120 + chunk
  5120): stock ring_mla at bf16 DEST has worst (head, row) rel L2 0.165 vs fp32 SDPA on the same inputs (attention out
  rel 0.0111 vs the CPU step: fails the component rel 0.0090 / row 0.03 / ratio limits); at fp32 DEST worst row 0.043,
  attention out rel 0.0081 (same lesson as the hackathon fork's xing40 entry, known_issues "ring_mla at bf16 DEST").
- Needed by: kimi_k27_code C.attn_dense.attention
- Files: device/ring_joint_sdpa_program_factory.cpp

### ring_mla exact_score_exp (fp32-accurate softmax exp on the streaming path)
- What: `ttnn.bringup.ring_mla(..., exact_score_exp=True)` (op attribute `RingJointSDPAParams::exact_score_exp`, define
  `SDPA_EXACT_SCORE_EXP`): `sub_exp_block_bcast_cols` computes exp((s - max) * scale) with the fp32 scale and
  `_sfpu_exp_fp32_accurate_` (inputs clamped at -88.5) instead of the hard-coded approximate exp
  (`exp_packthread_tile<true>`, a bit-trick exp with max rel error 3 % and a bf16-truncated scale). Needs the
  streaming path at fp32_dest_acc_en (TT_FATAL otherwise). Default False: the program is the previous one.
- Why: Kimi-K2.7 layer-1 MLA (scores up to ~35 after scaling): ring_mla output rel 0.0122 vs fp32 SDPA on the same
  device inputs (worst row 0.056); a CPU model of the kernel with the approximate exp reproduces it (0.0136), with an
  exact exp 0.0077. The attention component test failed its rel / row limits (rel 0.0153 > 0.0173 x margin, worst row
  0.041 > 0.037).
- Needed by: kimi_k27_code C.attn_moe.attention
- Files: device/kernels/compute/compute_streaming.hpp, device/ring_joint_sdpa_device_operation_types.hpp,
  device/ring_joint_sdpa_device_operation.{hpp,cpp}, device/ring_joint_sdpa_program_factory.cpp, sdpa.{hpp,cpp},
  sdpa_nanobind.cpp
