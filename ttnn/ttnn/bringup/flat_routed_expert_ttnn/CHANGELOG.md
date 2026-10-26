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

### Import @ f080c8df021 (2026-10-17, commit 9003c0fd2ca)
- What: the op re-imported from `mstaletovic/galaxy-unified-k3-v41-combine` @ `f080c8df021` (V4.1-tuned), with
  combine_rf (`../combine_rf/`, Python): DRISC readers (`device/flat_drisc.{hpp,cpp}`, `flat_drisc_stop` /
  `flat_drisc_dump`), the uneven gate/up split (DeepSeek-V4.1 I 2304), 16-column units, the combine overlap
  (`flat_routed_expert_with_program`, `device/flat_with_program_device_operation.{hpp,cpp}`), `FLAT_COMBINE_OVERLAP_WORKLOG.md`
  and `tests/unit/test_flat_uneven_split.py`. Copied as is (repo formatters applied); one adaptation: MathFidelity is
  namespaced on this tree (`using tt::tt_metal::MathFidelity;` in the program factory).
- Why: owner decision 2026-10-17 (routed experts only): try the flat routed expert against the fused
  unified_routed_expert (P.2).
- Results (commit 9003c0fd2ca message): builds; unit suite on the 2x2 QuietBox 18 passed / 14 failed, every V4.1 / GLM /
  K3 uneven-split case passing; the failures: the 12 tp2 / tp4 `p.nsg == 1` cases the previous copy (2026-10-13)
  already had, the mimo perf baseline and test_flat_expert_mesh (cq_id 0 out of range).
- Needed by: deepseek_v41_flash_d_p P.3
- Files: the whole folder (CMakeLists.txt, device/, device/kernels/, flat_expert.py, flat_routed_expert.{hpp,cpp},
  flat_routed_expert_nanobind.cpp, the two worklogs, tests/unit/test_flat_uneven_split.py)

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
  `tests/unit/test_ds41_h_options.py`, `tests/test_flat_bench.py` (commit 2bbb41e2bb8, 2026-10-18)

### Sync to e4d239c6d81 (2026-10-19, commit dc556fa6726)
- What: synced with combine_rf to `mstaletovic/galaxy-unified-k3-v41-combine` @ `e4d239c6d81` (22 commits after
  `f080c8df021`): a three-way merge onto the DeepSeek-V4.1 options above, one conflict (flat_expert.py weight packing),
  both sides kept (the source's pack-once replicate + the [N, K] layout and exactness check). Brings the se_yrm
  small-ring hang fix, the stream_linear atomic barrier fix, DRISC-only and uneven-split plans, relay_words, and
  combine_rf's accumulators on port cores.
- Why: keep the fork on the source's fixes.
- Results (commit dc556fa6726 message): build OK. 2x2 QuietBox: unit suite 26 passed / 13 failed (the same 13 as at the
  import: 12 x `p.nsg == 1`, the mimo perf band); test_ds41_h_options 7/7; deepseek_v41 test_c_swa_experts PASS, vs CPU
  rel 0.00224936 (0.00225 before), same plan (nu=3 mt=1 gate/up=48 down=32, no DRISC).
- Needed by: deepseek_v41_flash_d_p
- Files: device/flat_routed_expert_{plan.cpp,program_factory.cpp}, device/flat_with_program_device_operation.{hpp,cpp},
  device/kernels/{se11_xmc.cpp,se_drisc_post.hpp,se_yrm.hpp}, flat_expert.py, flat_routed_expert.{hpp,cpp},
  flat_routed_expert_nanobind.cpp, the two worklogs; ../combine_rf/

### flat_routed_expert_with_program: math_fidelity, weights_nk (2026-10-19, deepseek_v41_flash_d_p P.8)
- What: `flat_routed_expert_with_program(..., math_fidelity=0, weights_nk=False)`: the flat expert's matmul fidelity
  and the [N, K] weight tiles (as for flat_routed_expert); both in the program-cache key. Defaults = the program before
  (LoFi, [K, N] tiles).
- Why: DeepSeek-V4.1's experts run at HiFi4 (owner rule) on bfp8 [N, K] weights (exact e2m1 x ue8m0, DS41-35) with
  combine_rf's weighted combine appended (EXPERTS_COMBINE=weighted).
- Results: see ../combine_rf/CHANGELOG.md (P.8): layer 0, 2x2, the fused call at full precision (HiFi4, bf16 x / h):
  2 passes x ~2.1 ms; experts call 6.1 ms vs separate 8.5 ms (synced wall, one layer).
- Regression (options off, 2026-10-19, after the build): `tests/unit/` 25 passed / 14 failed: the known 13 (12 TP2 /
  TP4 `p.nsg == 1`, the MiMo indexed_perf band) + test_flat_expert_mesh[2x2], which passed in the run's precompile pass
  (min PCC 1.00000) and failed in the real pass on its module-level `_BETWEEN` tensors of the precompile pass's closed
  mesh (K-52); alone with --no-precompile: 1 passed (min PCC 0.99804). So: the baseline's 26 / 13.
- Needed by: deepseek_v41_flash_d_p P.8 (tt/weighted_combine.py)
- Files: flat_routed_expert.{hpp,cpp}, flat_routed_expert_nanobind.cpp

### se5_recv: drain the write command buffer before polling a transaction id (from Marko Staletovic's 318b7ca5403)
- What: the x and h chains wait for `noc_cmd_buf_ready(noc_index, write_cmd_buf)` before
  `ncrisc_noc_nonposted_write_with_transaction_id_flushed`; right after issuing, the last write can still sit in the
  command buffer and its id reads "done" (stale x / h, ~1 in 1e5 calls; his stress: 1.0M calls bit-exact after).
  Only this fix is taken from that commit (not the unicast x snakes: they reuse gate/up semaphores 2/3 that the
  DeepSeek-V4.1 routing_weights / fp8_act_quant path uses). Always on (a bug fix).
- Needed by: deepseek_v41_flash_d_p (2026-10-20, overseer), test_ds41_h_options 7/7.
- Files: device/kernels/se5_recv.cpp

### Removed: the DeepSeek-V4.1 h options `routing_weights`, `fp8_act_quant` and the plan's `routed` (2026-10-22, FIX.9)
- What: the two `flat_routed_expert` / `FlatRoutedExpert.__call__` arguments added in P.3, `flat_routed_expert_plan(...,
  routed=)` / `FlatRoutedExpert(routed=)` and the plan dict's `routed`, and everything only they used: the config's
  `routed_weight` / `fp8_act_quant` (and their program-cache and plan-cache key fields), the `routing_weights` input,
  its validation and `AddrSrc::RoutingWeights`, the plan's h-options branch (mt 1, 4 fp32 DST tiles, no row passes) and
  its 36 KB gate/up arena region (`kRhXd`, `kRhTiles`, `rh_off`), the factory's RH_* CBs / semaphores / runtime args /
  defines and the compute CBs unpacked to fp32 DEST (`cp`'s `fp32_unpack_cbs`), the kernels' `SE_RH_W` / `SE_RH_Q` /
  `SE_RH_X` blocks (se3_compute.cpp: weight tile into DEST, SFPU multiply and fp8 q-dq, split-tile merge; se5_recv.cpp:
  routing-weight tile staging, the partner-half ring), and `device/kernels/se_rh.hpp`. `weights_nk` and `hidden` stay.
  Against the commit before P.3 (`git diff 2bbb41e2bb8^`) the factory and kernels now differ only by weights_nk /
  hidden and the later syncs.
- Why: deepseek_v41_flash_d_p FIX.9 (overseer, precision_debt_report.md J2): the options existed only for DeepSeek's
  official arithmetic (q(r h) before w2), which the model no longer emulates (agent rule 17, F137); its separate
  combine path (the only caller of routing_weights) is deleted, the routed experts run flat_routed_expert_with_program
  + the weighted combine. No other model calls them (grep of every model's code and the forks' tests, 2026-10-22).
  Option-off behaviour unchanged: every removed line was behind `p.routed`, `cfg.routed_weight` / `cfg.fp8_act_quant`
  or an SE_RH define; the weight-cache key never contained `routed` (WEIGHT_LAYOUT_KEYS), so existing caches stay
  valid.
- Tests: `tests/unit/test_ds41_h_options.py` deleted; `tests/unit/test_weights_nk.py` keeps the remaining P.3 options'
  checks without the h options: [N, K] with weights_nk == [K, N] bit for bit, DeepSeek-V4.1 shapes (H 5120, I 2304, 24
  experts, e2m1 x 2^e, wide x rows via hidden) vs float64, block_float_exact.
- Regression (options off, 2026-10-22, after the build, 2x2 QuietBox, --no-precompile per file): the five carried
  `tests/unit/` files 19 passed / 13 failed, the recorded baseline exactly (12 x TT_FATAL `p.nsg == 1` in the TP2 / TP4
  subgrid cases of test_flat_routed_expert_op / _yrm / _indexed_shapes, and the MiMo indexed_perf band, DS41-63);
  test_flat_expert_mesh and test_flat_uneven_split all pass; `tests/unit/test_weights_nk.py` 3 passed (V4.1 shapes rel
  0.0024, PCC 0.9999971 on dev 0 expert 0). GLM-5.3's case (`tests/test_flat_bench.py`, glm53flash): 2048 rows
  9.556 ms, 5120 rows 23.725 ms (9.558 / 23.721 before). deepseek_v41_flash_d_p: its 8 expert / shared-expert
  component tests print the same check values as before the change (bit-identical metrics), L.last per-layer errors
  equal FIX.8's, plain profile device 104.28 ms a chunk (104.31), 7800 programs.
- Needed by: deepseek_v41_flash_d_p FIX.9
- Files: `flat_routed_expert.{hpp,cpp}`, `flat_routed_expert_nanobind.cpp`, `flat_expert.py`,
  `device/flat_routed_expert_{plan.hpp,plan.cpp,types.hpp,device_operation.cpp,program_factory.hpp,program_factory.cpp}`,
  `device/kernels/{se3_compute.cpp,se5_recv.cpp}`, `device/kernels/se_rh.hpp` (deleted),
  `tests/unit/test_ds41_h_options.py` (deleted), `tests/unit/test_weights_nk.py`

### Tests: fork model-case suite with deepseek_v41_flash_d_p cases (O.1, 2026-10-23)
- What: tests/cases.py, reference.py, test_flat_routed_expert_ttnn.py, weighted_case.py: the shared expert's
  flat_routed_expert call (bfp8, I 1280, one always-on expert, token index) and the routed experts'
  flat_routed_expert_with_program call with the weighted combine fused in (bfp4, 24 of 96 experts a chip, top 6, one
  pass of 1024 home tokens, bf16 sums; weighted_case.py ports the model's WeightedCombine wiring onto the combine_rf
  builders). Random weights at exact block-float values (check_exact); limits from noise_limit at the op's formats
  (bfp8 x / h tiles, LoFi operand bits, fp32 sums; reference.ffn(formats=)): shared 4.0e-2 vs 9.9e-2, routed 2.0e-2 vs
  2.95e-2; zero rows exact for unseen home tokens; determinism 5 x A. A halved block of rows fails (checked by hand).
  ttnn/ttnn/bringup/__init__.py: a value repr for MeshProgramDescriptor (`extra`), so the fork-call capture gives one
  signature for the 8 layers' calls instead of one per object address and run. No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: deepseek_v41_flash_d_p O.1
- Files: tests/cases.py, tests/reference.py, tests/test_flat_routed_expert_ttnn.py, tests/weighted_case.py,
  ttnn/ttnn/bringup/__init__.py
- Follow-up (O.1 attempt 4): the routed case's sig is the F179 stable one, 3fc158f72b (the capture now leaves out
  extra_key, an id()-derived program-cache key, DS41-110). Same call as the earlier 284a4a9ead / 31755c8401, which
  differ only in extra_key (checked by re-hashing the captured entry). No op change.

### Intermediate widths with no subgrid split, and a down K of any tile count (2026-10-26, deepseek_v41_flash_d_p FIX.1)
- What: (1) `device/flat_routed_expert_plan.cpp`: an It <= 32 that the 2 / 3 subgrids cannot split (no gate/up split
  for any x ring size on their 16-cell rectangles: at H 5120, Ht 160, two pairs' weight ring alone fills the 1.40 MB
  gate/up budget) now plans on one subgrid, where the uneven pair split covers it: It 18 -> np 1 on 18 gate/up cores,
  It 36 -> np 1 on 36 (Galaxy chip, bfp8 / bf16 x). Before: TT_FATAL "no gate/up split for N tile columns". (2)
  `device/kernels/se6_dcompute.cpp`: the `kt_d % hk == 0` term of the static_assert removed. hk (8) is only
  matmul_block's kt_dim with rt_dim 1, which the LLK never reads (llk_unpack_AB_matmul.h: reuse_a, t = 0), so K 18 / 36
  tiles compile and compute right. se3_compute's same assert stays (it compiles with KT_D 0 there; its in-core down
  path uses hk as the h_all layout).
- Why: owner order (OV-3): never zero-pad a weight around a kernel limit. deepseek_v41_flash_d_p padded its shared
  expert's TP slice to the op's limits: TP 4 576 -> 1280 (2.2x the work), TP 2 1152 -> 1280 (DS41-76, DS41-115).
- Option-off behaviour: no option. Both changes only reach inputs that failed before (the planner TT_FATAL, the kernel
  static_assert); every input that planned / compiled before gets the same plan, kernels and args (the subgrid check
  runs only when the old rule picks 2 / 3 subgrids and keeps them whenever they split; se6_dcompute.cpp changes its
  source hash, not its code).
- Tests: `tests/unit/test_flat_widths.py` (new): H 5120, I 576 / 1152 (one always-on expert, m 640 / 1024; 576 also
  with 3 experts, one empty), bfp8 weights_nk, clamped SwiGLU, glm (LoFi, bfp8 x / h) and full (HiFi2, bf16 x / h),
  vs float64 within noise_limit, plus determinism; the routed width I 2304 (4 experts, bfp4), skipped on a Galaxy
  chip (DS41-119). Galaxy card: rel 0.0400 / 0.0398 / 0.0401 (glm, limits 0.099) and 0.0104 / 0.0105 (full, limits
  0.021); a wrong h (probe MIMO_FL_NO_ACT=1) fails it (rel 1.475).
- Regression (tests/unit/, once, after the build, one Galaxy card, --run-all): 11 passed / 26 failed / 4 skipped. Not
  the 2x2 QuietBox baseline's box: 10 x TT_FATAL `p.nsg == 1` (the TP2 / TP4 subgrid cases of indexed_shapes / yrm:
  as the baseline; their small-H subgrids split, so the plan is unchanged); 4 x test_flat_routed_expert_op's Python
  builder asserting the p150's 11 x 10 grid (12-10 here); 11 first-call hangs on a single Galaxy card (mimo,
  test_flat_uneven_split's six; It > 32, a plan this change never reaches; glm_e12 hangs the same way on the code
  before this change, DS41-119); the k2 indexed perf band, 1573083 ns vs its upper bound 1572413 (+0.04 %, another
  grid). Passed: k2 indexed determinism, test_flat_expert_mesh, test_weights_nk (V4.1 I 2304 shapes on a 2x2 mesh), the
  five new 18 / 36-column cases. Model cases (tests/test_*.py) not run: they need the 2x2 mesh exactly.
- Needed by: deepseek_v41_flash_d_p FIX.1 (tt/shared_expert.py: no padding)
- Files: `device/flat_routed_expert_plan.cpp`, `device/kernels/se6_dcompute.cpp`, `tests/unit/test_flat_widths.py`,
  `README.md`, `../INDEX.md`

### The schedule leader found by its logical coordinate: no hang on Galaxy cards (2026-10-26, deepseek_v41_flash_d_p FIX.2)
- What: `device/kernels/se_dyn.hpp` (SE_DYN_PULL): reader 0's BRISC knows it is the schedule leader by
  `get_absolute_logical_x() / _y()` against the new defines `SE_DYN_LLX / SE_DYN_LLY` (reader 0's logical core), set in
  `device/flat_routed_expert_program_factory.cpp`; `SE_DYN_LPX / SE_DYN_LPY` (its physical NoC0 xy, `flat_worker_noc0`)
  are gone. `flat_expert.py` `_layout` (the Python parity builder): the `rcols == [0, 1, 6, 7] and grid == 11 x 10`
  assert is now the C++ plan's rule (readers in columns 0, 1, E, E + 1, E >= 6, grid >= 11 + E - 6 wide, 10 rows) and the
  east rectangle / east relays move by E - 6 as in flat_routed_expert_plan.cpp. tests/unit/test_flat_widths.py: the I
  2304 skip on a 12-column grid removed.
- Why (cause, from tt-triage of test_flat_uneven_split[glm_e12] on one Galaxy chip, 5 s dispatch timeout): every role
  core sat in se_dyn_load's poll of the leader's semaphore SE_DYN_LSEM (36.9 M completed NoC reads, one in flight;
  se10_fwd / se_reader / se5_recv / se6_drecv / se6_dw in noc_async_read_barrier, the computes in llk_wait_tiles), and
  no core had set it. The leader compared `my_x[0] / my_y[0]` with a physical NoC0 coordinate, but Blackhole firmware
  fills my_x from `NOC_ID_LOGICAL`, the translated coordinate. They agree only when no Tensix column west of reader 0
  (logical (0, 9)) is harvested: true on the 2x2 QuietBox's p150s, false on Galaxy cards with physical column 1
  harvested (triage: logical (0, 5) at physical 2-7; reader 0 physical (2, 11), translated (1, 11)). Probes:
  `MIMO_FL_DYN_PULL=0` passed the same case; `MIMO_FL_DRISC=0` and `MIMO_FL_COLS=11` (the p150's core count) still
  hung. The glm_e12 shape also hung on the full 8x4 FABRIC_2D mesh (so DS41-119's "mesh runs pass" was wrong: those
  runs only used plans without SE_DYN_PULL, i.e. with reader tails or a combine attached, as every deepseek_v41
  call does).
- Option-off behaviour: no option. On a chip where the old check found the leader (physical == translated for reader
  0) the new one selects the same core: same schedule, same kernels apart from the define names. Plans without
  SE_DYN_PULL (reader tails, `flat_routed_expert_with_program`, subgrids) are unchanged.
- Tests (one Galaxy chip, tests/unit/, --run-all, after the build): 24 passed / 14 failed / 3 skipped (the 2x2-mesh
  tests skip on one card); before: 11 first-call hangs. Failures: 12 x TT_FATAL `p.nsg == 1` (TP2 / TP4 subgrid cases;
  the same 12 as the QuietBox baseline, DS41-63) and 2 perf bands calibrated on the QuietBox p150, slow on the Galaxy
  chip: mimo indexed_perf 1843456 / 1838520 ns (upper bound 1826488; it failed fast on the QuietBox, 1710341) and k2
  indexed_rm 1573274 / 1575501 ns (upper 1572413; k2 has reader tails, so this change does not reach it: 1573083 ns in
  FIX.1, before it). test_flat_routed_expert_op mimo / k2 now pass on the 12 x 10 grid (C++ plan == Python layout,
  bit-identical y). The glm_e12 shape (64 tokens an expert) runs on the 8x4 mesh (probe). Not run on a QuietBox (not
  on this box).
- Needed by: deepseek_v41_flash_d_p FIX.2 (overseer: root-cause the flat op's Galaxy hangs, DS41-119)
- Files: `device/flat_routed_expert_program_factory.cpp`, `device/kernels/se_dyn.hpp`, `flat_expert.py`,
  `tests/unit/test_flat_widths.py`, `README.md`
