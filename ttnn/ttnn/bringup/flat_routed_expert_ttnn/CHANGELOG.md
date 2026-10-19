# flat_routed_expert_ttnn (ttnn.bringup.flat_routed_expert, ttnn.bringup.flat_routed_expert_plan)

- Current source: `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/flat_routed_expert` on `mstaletovic/galaxy-unified-k3-v41-combine` @ `f080c8df021` (imported 2026-10-17 in 9003c0fd2ca for deepseek_v41_flash_d_p P.3), synced to `e4d239c6d81` in dc556fa6726 (with combine_rf); was `mstaletovic/mimo-v2-dp` @ `b45a923d73c` (below).
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

### DeepSeek-V4.1 expert numerics: routing weight on h, fp8 q(h), [N, K] weights, wide x rows
- What: four opt-in arguments of `ttnn.bringup.flat_routed_expert` (and `FlatRoutedExpert`), all off by default (the
  program, its kernels, runtime args and CBs are then the previous ones; the new config fields are in the program key):
  - `routing_weights` (FLOAT32 TILE [rows of x, 32], DRAM; row i column le = dispatched row i's routing weight for local
    expert le): h = act(gate, up) of each row times its weight (fp32 SFPU) before the down projection. The gate/up
    receiver (se5_recv.cpp, `SE_RH_W`) reads the sub-block's 4 KB tile of the tensor and spreads column le (the entry's
    local expert) over faces 0 / 2 of a Float32 tile; the gate/up compute (se3_compute.cpp) copies it straight into the
    fp32 DEST (UnpackToDestFp32, exact) next to the gate/up tiles before the K loop; `se_rh.hpp:rh_wmul` on the pack
    thread multiplies every h tile.
  - `fp8_act_quant`: h (after the weight) quantized / dequantized to fp8 e4m3 per 32 values with a ue8m0 scale, round
    to nearest even, before the down projection (DeepSeek's act_quant(h, 32, 'ue8m0'); `se_rh.hpp:rh_quant`, the math
    of unified_routed_expert_ffn's `calculate_routed_h`). A plan of odd 16-column units (DeepSeek-V4.1 I 2304: 3 units
    a core, 48 cores) splits every third h tile over two cores, and a block's scale needs all 32 values: the lead core
    (first h tile a right half) packs its half as Float32 and its receiver writes it into the tail core's ring (CB 11,
    semaphores 2 / 3); the tail core merges both halves in DEST (`rh_merge`), quantizes the whole tile and sends it to
    the down cores (`SE_RH_X`). Quantizing on the down cores instead would repeat it on every down core.
  - `weights_nk` (+ `FlatRoutedExpert(weights_nk=True)`, `flat_weight_regions(..., nk=True)`): the weights are laid out
    from nn.Linear's [N, K] matrices, every tile the transpose of the x @ W tile at the same place, and every weight
    matmul transposes its in1 tile back (`SE_W_NK`, se_meta.hpp `SE_WT`; se3_compute / se6_dcompute). A bfp8 tile
    then shares its exponents along K: DeepSeek's e2m1 x ue8m0 experts are exact (in x @ W orientation bfp8 rounds
    48 of that run's 2304 matrices, DS41-35). `FlatRoutedExpert(check_exact=True)` / `block_float_exact` assert it.
  - `hidden`: x rows wider than H (x_pages_per_row 1): the relays read each row's first H values in place (a
    dispatch buffer whose rows carry the routing pieces after the token).
  - The plan (`routed` in the plan key; `flat_routed_expert_plan(..., routed=True)`): with either h option, 32-row
    sub-blocks in one gate/up pass in the half-sync fp32 DEST (gate/up width + the weight tile <= 4 tiles), and a
    36 KB h-option region in the gate/up arena. Refused: subgrids, M-groups, bf16 gate/up DEST, h units, two x
    streams (MIMO_FL_X2), token_index, row-major y; routing_weights not FLOAT32 TILE DRAM [rows of x, 32].
- Why: DeepSeek-V4.1's routed expert is w2(q(r * act(w1 x, w3 x))) (q(r h) != r q(h)); P.3 runs it on this op.
- Measured (deepseek_v41_flash_d_p P.3, 2x2 p150, 11 x 10 grid, chunk 2048, 24 experts per chip, layer 0, plan nu 3,
  48 gate/up + 32 down cores, mt 1, bf16 x / h / y, HiFi4, fp32 DEST): the expert op (incl. the routing-rows matmul)
  3.76 ms bfp8 / 3.66 ms bfp4 per layer vs the unified fork 6.37 / 5.99 (tt/perf_experts.py::test_kernels); in the
  model the experts section 149.3 ms (bfp8) / 148.0 ms (bfp4) per chunk vs 176.8 / 173.7 (unified).
- Tests: `tests/unit/test_ds41_h_options.py` (2x2 FABRIC_2D, 7 cases): identity-down with selection weights (exact
  GEMMs) weight + quant / quant only / weight only / a pinned schedule: y == fp8_act_qdq(reference h) bit for bit
  (one near-tie value over 1528 rows; 2848 rows pinned), negative controls (round toward zero, scale / 2, no quant,
  another expert's weight column, per-half scales of the split tiles) all rejected; [N, K] with weights_nk ==
  [K, N] without, bit for bit (bfp8 exact both ways); DeepSeek-V4.1 shapes (H 5120, I 2304, 24 experts, e2m1 x 2^e
  weights, wide x rows) vs the float64 reference: rel 0.0032-0.0053, PCC >= 0.99998, worst row <= 0.019; refusals.
- Regression (options off, 2026-10-18, final tree, build 06:28): `tests/unit/` (the five files carried with the
  import) 19 passed / 13 failed; the import commit 9003c0fd2ca had 18 passed / 14 failed. Every failure is
  pre-existing: 12 TP2 / TP4 subgrid cases (TT_FATAL `p.nsg == 1`, flat_schedule_defines, the same assert at the
  import, program_factory.cpp:91 there) and the MiMo indexed_perf band, which this box runs FASTER than its lower
  bound (1710341 ns vs 1720090; 1700296 at the import). GLM-5.3's case (`tests/test_flat_bench.py`, preset
  glm53flash: 4096 x 2048, 36 experts, clamped_silu, indexed x, default options) passes: 2048 rows 9.558 ms, 5120
  rows 23.721 ms. `tests/source.yaml` carries no source tests (`tests: []`).
- Needed by: deepseek_v41_flash_d_p P.3 (tt/experts.py `_flat`, EXPERTS_KERNEL=flat)
- Files: `flat_routed_expert.{hpp,cpp}`, `flat_routed_expert_nanobind.cpp`, `flat_expert.py`,
  `device/flat_routed_expert_{plan.hpp,plan.cpp,types.hpp,device_operation.cpp,program_factory.hpp,program_factory.cpp}`,
  `device/kernels/{se3_compute.cpp,se5_recv.cpp,se6_dcompute.cpp,se_meta.hpp,se_rh.hpp}`,
  `tests/unit/test_ds41_h_options.py`
