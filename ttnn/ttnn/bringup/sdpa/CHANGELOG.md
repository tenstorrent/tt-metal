# sdpa (fork)

- Source: `ttnn/cpp/ttnn/operations/transformer/sdpa`
- Source SHA: `99f7e834cea8bc42f0b478b9cc59a7b89f2f9c1a`
- Python: `ttnn.bringup.*` (was `ttnn.transformer.*`)
- Forked for: mimo_v2_6_d_p attention V head dim 128 (non-MLA GQA)
- Used by: mimo_v2_6_d_p (tt/attention.py: full and sliding layers, V 128)
- Note (2026-10-19): the two lines above and "only the first two are carried and tested" below date from the fork.
  Now also used by mimo_v2_6_d_p_2x2, glm53_flash_d_p (sparse_sdpa), xing40_a4b_d_p (ring_mla) and
  deepseek_v41_flash_d_p (sparse_sdpa); sparse_sdpa and ring_mla are changed and tested here too (entries below);
  README.md lists the calls. xing40_a4b_d_p calls `ttnn.bringup.ring_mla` at HiFi2 + fp32 DEST by default
  (`XING_MLA_SDPA_FIDELITY=HiFi2`, the owner's exception for the SDPA; `HiFi4` is the previous setting;
  `tt/attention.py`), and its model case in `tests/cases.py`
  (`xing40_a4b_d_p-4x2-ring-mla-q16x1280-isl51200-n56320-k576-v512-hifi2-fp32-q64k256`) runs HiFi2 + fp32 DEST.

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_sdpa`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

The source is a subfolder of the `ttnn_op_transformer` target, so it had no CMake target of its own; fork_op.py
wrote `CMakeLists.txt` and `sources.cmake` for the fork (every host `.cpp` except the nanobind one; the public headers
`sdpa.hpp`, `sparse_sdpa.hpp`, `sparse_sdpa_msa.hpp`). The whole folder is copied, so every Python name it binds
(scaled_dot_product_attention, chunked_scaled_dot_product_attention, the joint, ring, MLA and sparse variants) is in
`ttnn.bringup.`; only the first two are carried and tested. `sdpa_config.hpp` (SDPAProgramConfig,
PagedCacheGeometryOverride) is shared with the other transformer ops and stays the source's, so the fork takes the same
`ttnn.SDPAProgramConfig`.

Build fixes after fork_op.py (no behaviour change):
- `ttnn::operations::transformer::{SDPAProgramConfig, PagedCacheGeometryOverride}` restored (the namespace rewrite had
  moved them to `ttnn::operations::bringup`); `sdpa_nanobind.cpp` gets `using` declarations for the two.
- Relative qualifications into the nested namespaces: `prim::*_IDX` -> `prim::bringup::*_IDX` (sdpa.cpp),
  `transformer::SparseKVFormat` -> `transformer::bringup::SparseKVFormat`, `operations::transformer::sdpa::` ->
  `operations::bringup::sdpa::`.
- Sub-namespaces nested twice (`ttnn::prim::bringup::detail::bringup`, `...::sdpa_cb::bringup`,
  `ttnn::transformer::bringup::sdpa::bringup`) collapsed to one `bringup`.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### Non-MLA V head dim narrower than K's (GQA)
- What: `scaled_dot_product_attention` (causal, sliding window, attention sink) and
  `chunked_scaled_dot_product_attention` (paged, chunk_start_idx as an int or a tensor) take a V whose last dim is
  smaller than Q/K's; the output then has V's width. No new argument: a narrower V tensor turns it on. V must be a
  multiple of 32 and at most K's head dim (refused otherwise); the paged-cache geometry override and windowed mode
  still require V as wide as K. Host-only:
  - program factory: non-MLA `vDHt` = V's own width (was always DHt), except under a geometry override;
  - reader compile-time arg 23 (`use_mla`) is passed as `use_mla || vDHt != DHt`. In the reader it only selects V's
    addressing width (`v_tile_shape`, the paged `head_dim`); the K/V overlap it also gates needs arg 24
    (`mla_kv_overlap`), which stays false. Chosen over editing the two reader conditions so that the kernels stay the
    source's byte for byte (no kernel hash change, no recompile of any other program);
  - validation: `v_shape[3] <= DH`, `% 32 == 0` (plain and paged); windowed mode refuses a narrower V;
  - `compute_output_specs`: non-MLA output last dim = V's (unless a geometry override is active).
  The kernels already handled vDHt < DHt (the MLA path uses them).
- Default unchanged: with V as wide as K (every existing caller) vDHt == DHt, arg 23 == use_mla and the output spec
  are what they were, so the program (kernels, compile-time and runtime args, CBs) is the source's; the unit tests
  check the fork's output is bit-identical to the source op's (causal and chunked, both compute paths), and the
  source-test check reports 0 regressions (118 tests, warm kernel cache hits). The V shape is in the default
  program-cache key (tensor specs are hashed); a test runs V 192 and V 128 with the same Q/K and gets 2 programs.
- Why: MiMo-V2 attention has Q/K head dim 192 and V 128; the model pads V to 192 with zeros. With the narrow V the
  output equals the padded run's first 128 columns bit for bit, and device time at MiMo's full-attention chunk
  (Q 16x5120x192, paged cache with a 51200-token prefix, q512/k128, HiFi2, approx exp, fp32 dest off = streaming,
  1 chip) drops from 27.15 ms (V 192, source and fork alike) to 22.49 ms (V 128), -17%; chunk 0 (causal S 5120)
  from 1.505 ms to 1.256 ms, -17%.
- Tests: `tests/unit/test_sdpa_narrow_v.py` (20 cases, MiMo shapes: 16 Q heads, 1 or 2 KV heads, K 192, V 128,
  scale 192^-0.5, bf16): causal S 5120 / 2048, sliding window 128 + sink, chunked paged at prefixes 4096 and 49152,
  fp32 dest on (non-streaming) and off (streaming), each against a torch reference and bit-identical to the padded
  source op; V == K bit-identical to the source; the program cache; refusals (V 112, V 224). With arg 23 reverted to
  `use_mla` (the reader addressing V at K's width) all 11 narrow-V cases fail (PCC 0.16-0.83).
- Needed by: mimo_v2_6_d_p (attention V head dim 128; the model calls the fork with V 128 since commit
  "Attention V at its real head dim 128", MIMO_V_PAD=1 restores ttnn.transformer with padded V)
- Files: `device/sdpa_program_factory.cpp`, `device/sdpa_device_operation.cpp`, `tests/unit/`

### Model cases (O.1 test pass, no op change)
- What: `tests/cases.py`, `tests/reference.py`, `tests/test_sdpa.py`: one random-input case per captured call
  (bringup-fork-tests skill), 1x4 mesh, FABRIC_2D. Each case checks the fork's output against a float32 torch
  reference (PCC + relative L2 error, per device) and bit for bit against the source op run on V zero-padded to K's
  width.
- Needed by: mimo_v2_6_d_p O.1 (sigs 239d54bae3 chunked paged at prefix 51200, 3f237ff436 sliding window 128 + sink)
- Files: `tests/cases.py`, `tests/reference.py`, `tests/test_sdpa.py`

### sparse_sdpa: opt-in high_precision (Float32 running state, exact softmax exp)
- What: `ttnn.bringup.sparse_sdpa(..., high_precision=False)`. True (needs fp32_dest_acc_en, bf16 q, kv_format BF16,
  no attention_sink; refused otherwise) makes four changes, all behind the compute define
  `SPARSE_SDPA_HIGH_PRECISION`, which is added only when the option is on:
  - the flash running output and row-sum ping-pong CBs (`cb_out_a/b`, `cb_sum_a/b`) and the 1/sum scratch
    (`cb_recip_scratch`) are Float32 (bf16 in the source). They are L1-accumulated across k_chunks and rescaled by
    the flash correction once per chunk. The kernel switches the pack / unpack formats around every read and write of
    them (PV, the SALAD combine, normalize) and restores the bf16 state afterwards;
  - the row-sum partial comes from the packed bf16 probabilities, the ones PV multiplies (copy_tile, then
    L1-accumulate into the Float32 sum), not from DEST;
  - the softmax exp is exact (`exp_packthread_tile<false, true>`, the scale as a bf16 immediate rounded to nearest;
    exact for a power of two). The source hard-codes the fast approximate exp there, whatever math_approx_mode says.
- Default unchanged: with the option off the CBs, compile-time args and preprocessed kernels are the source's (the
  define is absent). Output is bit-identical to `ttnn.transformer.sparse_sdpa` (GLM shape with fp32 dest on and off,
  and fp8 kv). The option is in the program hash. Source-test check: 118 tests, 0 regressions. The new sparse_sdpa
  entries (92 tests) pass on the original and on the fork alike. MiMo's model cases pass.
- Why: GLM-5.3 DSA attention (64 heads, latent 512 = K_DIM = v_dim, 2176 indices, k_chunk 128 = 17 chunks, scale
  1/16) on the source op fails the component test's per-token norm ratio (0.9928 < 0.994). On its own output vs
  float32 on the same bf16 inputs the source op scores rel L2 0.0066, single head-rows off by up to 3%. A CPU model of
  the kernel puts 0.0070 on the bf16 running output / sum (0.0012 with Float32) and the rest on the approximate exp
  (Schraudolph-like: fp32 state + approx exp 0.0038). With high_precision: 0.0030, head-row ratio [0.987, 1.012]; the
  model's attention output goes from rel 0.0076 / ratio [0.9928, 1.0028] to 0.0044 / [0.9954, 1.0007]. A remaining
  shrink of about 0.1% (coefficient 0.9989) is most likely the FPU reading the Float32 CBs at TF32 in normalize;
  untested.
- Tests: `tests/unit/test_sparse_sdpa_high_precision.py` (8 cases): accuracy vs float32 at GLM (H 64, K 512, 17
  chunks, rows with 1..2176 valid ids) and DeepSeek-like (H 32, K 576, v 512) geometries, and a sharp-softmax GLM case
  (bound by the bf16 scores buffer, which this option leaves as is). Each checks rel L2, per-row ratio and coefficient,
  better than the source op; option off bit-identical to the source (fp32 / bf16 dest, fp8 kv); the program cache;
  the refusals. The output scaled by 1.004 fails the GLM and DeepSeek-like cases. `tests/source.yaml` now also
  carries the source op's sparse_sdpa tests (sanity file and the nightly accuracy cases).
- Needed by: glm53_flash_d_p C.dsa_moe.attention (`tt/mla_attention.py`; `GLM_MLA_SDPA=source` for the source op)
- Files: `sparse_sdpa.hpp`, `sparse_sdpa.cpp`, `sdpa_nanobind.cpp`, `device/sparse_sdpa_device_operation*.{hpp,cpp}`,
  `device/sparse_sdpa_program_factory.cpp`, `device/kernels/compute/sparse_sdpa_compute.cpp`,
  `device/kernels/compute/compute_streaming.hpp` (sub_exp_block_bcast_cols, normalize_row_streaming: guarded by the
  define, which only the sparse compute kernel sets), `tests/unit/test_sparse_sdpa_high_precision.py`,
  `tests/source.yaml`, `tests/source_baseline.json`

### ring_mla (latent V) at fp32 DEST on the streaming path
- What: `ttnn.bringup.ring_mla(..., compute_kernel_config.fp32_dest_acc_en=True)` runs. The source op (and the fork until now)
  refuses this combination (TT_FATAL "Latent-V ring attention is implemented only for streaming compute", and
  "kv_actual_isl requires the ring-joint streaming compute path"): its streaming compute path, the only one with
  latent V and kv_actual_isl, is taken only at bf16 DEST. The fork now takes the streaming path whenever V shares K's
  buffer (`use_streaming_compute = !fp32_dest_acc_en || v_shares_k_buffer`), and on the streaming path keeps the
  intermediate CBs it was written for in bf16 at fp32 DEST as well (`sum_df`, `qk_im_df` Float32 only on the legacy
  path), as sparse_sdpa runs its streaming helpers at fp32 DEST. The only difference from the bf16-DEST program is
  DST_ACCUM_MODE (fp32 accumulation in DEST, dst_size and subblocks from `get_dest_reg_count`). Host-only, no kernel
  edit, no new argument: the switch is the combination the source refuses.
- Default unchanged: every configuration the source accepts keeps its path, CBs and program (a separate K/V ring
  joint at fp32 DEST stays on the legacy path with its Float32 sum / qk CBs; bf16 DEST is untouched).
  `test_ring_mla_bf16_dest_matches_source` checks bf16 DEST bit-identical to `ttnn.transformer.ring_mla`. fp32_dest_acc_en
  is in the program hash (compute kernel config). Source-test check: 210 tests, 0 regressions (the carried tests do not
  reach ring_mla: the source's ring tests open FABRIC_1D / 1D_RING / torus meshes, which the owner's 2D-fabric rule
  forbids here, and its one FABRIC_2D case is full-mesh at bf16 DEST, which this change does not touch). Unit suite 35
  passed. The model cases in `tests/test_sdpa.py` are 1x4 / 2x2 meshes; on the 8-chip 4x2 LoudBox their mesh open times
  out in fabric router sync (environment, not this change).
- Why: Xing4.0 dense MLA (ring_mla over the 4 SP rows, 16 heads, K 576, V 512, scale 0.1447, block-cyclic cache,
  kv_actual_isl). At bf16 DEST the scores accumulate over 18 K tiles in 16-bit DEST; on sharp softmaxes (scores up
  to ~93 on the component test's x2 inputs) that moves single rows by up to 25% in latent space, and the attention output's
  worst row rel L2 is 0.054 (limit 0.045; the bf16 precision model gives 0.029). A CPU model puts the excess on the
  16-bit DEST accumulation (bf16 running state adds ~0). With fp32 DEST: worst row 0.035, golden rel vs CPU 0.0048 ->
  0.0031. On random inputs (tests below) the sharp-case worst row goes 0.70 -> 0.16, rel 0.09 -> 0.019.
- Tests: `tests/unit/test_ring_mla_fp32_dest.py` (7 cases, 4x2 mesh, FABRIC_2D, ring over axis 0, Linear, Xing
  geometry, q32 / k256): accuracy vs float32 torch at chunk 0 and after a 2048 prefix, spread and sharp scores, each
  tighter than the source at bf16 DEST; bf16 DEST bit-identical to the source; the source still refuses fp32 DEST.
  With the output scaled by 1.02 (`RING_MLA_TEST_CORRUPT=1.02`) the spread cases fail.
- Needed by: xing40_a4b_d_p C.dense.attention (`tt/attention.py`; `XING_MLA_SDPA=source` keeps ttnn.transformer.ring_mla
  at bf16 DEST)
- Files: `device/ring_joint_sdpa_program_factory.cpp`, `tests/unit/test_ring_mla_fp32_dest.py`

### ring_mla / ring joint: host kv_actual_isl=0 on a single-chunk cache runs (was TT_FATAL)
- What: `RingJointSDPA` invoke drops a host `kv_actual_isl` of 0 when the input is not chunk-shaped (per-device
  Q.seq == K.seq, causal, not cross), so the op takes the full-prefill causal path instead of failing validate with
  "KV-pad rotation (host kv_actual_isl or metadata) requires chunked-prefill input". kv_actual_isl=0 has no prefix to
  rotate past; the metadata path already treats the non-chunked case the same way (`kv_pad_from_metadata` needs
  `is_chunked`). No new argument.
- Default behaviour: bug-fix exception (skill section 3): only an input the fork (and the source) refused changes;
  every accepted input keeps its attributes, hash and program. Unit suite 38 passed (35 + 3 new); source-test check
  210 tests, 0 regressions; the model cases in `tests/test_sdpa.py` (1x4 / 2x2) still fail to open their mesh on the
  4x2 box (fabric router sync, environment, as recorded above); none of them calls ring_mla.
- Why: xing40_a4b_d_p always passes `kv_actual_isl=start`. test_positions runs one chunk at start 0 with
  max_seq = chunk = 5120, so the latent cache is 1280 rows per SP row, equal to Q, and every layer hit the TT_FATAL.
- Tests: `tests/unit/test_ring_mla_single_chunk.py` (3 cases, 4x2 mesh, FABRIC_2D, ring over axis 0, Xing geometry,
  cache = chunk = 2048): fp32 / bf16 DEST bit-identical to the same call without kv_actual_isl, accuracy vs float32
  torch (rel 0.018 / worst row 0.031 at fp32 DEST, 0.023 / 0.058 at bf16 DEST); the source op still refuses it.
- Needed by: xing40_a4b_d_p X.3 (test_positions, `tt/attention.py:_ring_attend`)
- Files: `device/ring_joint_sdpa_device_operation.cpp`, `tests/unit/test_ring_mla_single_chunk.py`

### Upstream sync to source @ `73027b6e6ff` (2026-10-01): shared CCL halo contract, legacy recip removal, bug fix
- What: a 3-way merge (base = fork_op.py's mechanical copy of the source @ `99f7e834cea`, theirs = the same copy of
  the source with the commits below, ours = this fork; no conflicts) of seven source commits:
  - required to build: #57190 (block-cyclic sliding ring), #55596 (ring MLA chunked Q remainder rotated across grid
    rows), #57453 (multi-hop sliding halo), #57979 (multicast halo, larger packets), #57790 (IDevice -> MeshDevice).
    The fork shares `experimental/ccl/ring_attention_all_gather_async` with the source, and those PRs changed its
    halo runtime-arg layout and `RingAttentionNeighborHaloConfig` (`unicast_hops` -> `distance`/`hop`, writer
    origin field renamed), so ring_joint_sdpa_program_factory.cpp no longer compiled against it; the four ring PRs
    are interlocked in that file, so they come together;
  - required by the kernels: #56292 (legacy sqrt/rsqrt/reciprocal paths dropped from the LLK:
    `calculate_recip_first_column` lost its `legacy_compat` parameter; `recip_tile<false>` is gone), applied to
    compute_common.hpp / compute_streaming.hpp; without it every SDPA kernel fails to JIT-compile;
  - bug fix: #58381 (positive chunk sizes, nonzero K head count validated in every SDPA entry).
- Not taken (new optional features, listed only): #56922 `output_concat_heads` (+ pad-key blanking) for
  scaled_dot_product_attention; #57636 sparse_sdpa_msa rotated causal geometry (multi-turn resume).
- Default behaviour: the fork's own changes are untouched by the merge; new ring options default off.
- Needed by: rebase onto origin/malimpic/llk_helper_library_rebased_0110_2
- Files: 24 merged files under `device/` and `sdpa_nanobind.cpp`; new `device/kernels/chunked_q_mapping.hpp`,
  `device/ring_joint_sdpa_schedule.hpp`

### Upstream sync to source @ `bdc576fa652` (2026-10-12): full sync, including #56922 and #57636
- What: a 3-way merge (base = fork_op.py's mechanical copy of the source @ `73027b6e6ff` without #56922 and #57636,
  the fork's last sync; theirs = the same copy @ `bdc576fa652`; ours = this fork). The fork is now the source @
  `bdc576fa652` plus its own changes. Taken:
  - the two commits not taken at the last sync: #56922 `output_concat_heads` (+ pad-key blanking) for
    scaled_dot_product_attention, #57636 sparse_sdpa_msa rotated causal geometry (multi-turn resume);
  - features (new options, default off): #58032 ring_joint K split and packed latent V, #58223 configurable attention
    matmul fidelity and segmented accumulation, #57998 block-cyclic sliding Q wrapping a slab, #58436 chunked SDPA
    sinks, #59055 windowed mode with causal masking, #58613 split-KV ring MLA with fused full-mesh KV gather;
  - perf: #57878 ring MLA prefill with batched V matmul and compute overlap;
  - bug fixes (they change default behaviour where they apply): #59349 the SDPA reader no longer adds the scalar
    chunk offset to valid_Skt twice, #58436 sliding-window masks, #59060 MiniMax robustness;
  - cleanups: #54162 (docs), #59272 (log level), #58505 (CoreRange aliases), #59097 (alias removal), #59418 (includes).
  Nine conflicts, each resolved as upstream's text with the fork's change on top: the namespace spellings the fork had
  fixed by hand (`ttnn::operations::transformer::PagedCacheGeometryOverride`,
  `ttnn::transformer::bringup::sdpa::streaming_qktv_h`) on lines upstream extended; ring_joint's streaming rule
  (latent V at fp32 DEST streams) next to upstream's new K-stripe assert; a ring_distributed hunk where the fork only
  had formatter whitespace (upstream's kept).
- Default behaviour: the fork's own changes are kept (narrower non-MLA V, sparse_sdpa high_precision, ring_mla at fp32
  DEST, ring joint kv_actual_isl=0 full prefill); the upstream bug fixes above apply to every caller.
- Needed by: rebase onto origin/llk_helper_library @ `bdc576fa652` (2026-10-12)
- Files: 36 merged files under `device/`, `sdpa.{cpp,hpp}`, `sdpa_nanobind.cpp`, `sparse_sdpa*.cpp`; new
  `device/kernels/{ring_joint_ksplit,ring_mla_geometry,ring_mla_packing_plan,windowed_mode}.hpp`
- Tests: the shared `SDPAProgramConfig` now prints three more (default) fields (#58032 `max_k_splits`, #58223
  `matmul_math_fidelity`, `segmented_accumulation`), so every captured call that passes one has a new signature; the
  five such cases in `tests/cases.py` (mimo_v2_6_d_p x2, mimo_v2_6_d_p_2x2 x2, xing40_a4b_d_p ring_mla) and the
  models' `results/fork_calls.json` are re-keyed (same call, recomputed sig; checked against a fresh mimo_v2_6_d_p_2x2
  capture). Unit suite 28 passed / 10 skipped (the 8-device Xing cases), model cases 5 passed / 1 skipped (Xing).

### Tests: glm53_flash_d_p sparse_sdpa case re-captured (O.1, 2026-10-13)
- GLM's replicated-cache sparse_sdpa case replaced by its block-cyclic striped-latent call (kv [1, 1, 81920, 512], block_cyclic_sp_axis 0 / chunk_local 2560 / cache_tp_sharded, natural ids); `tests/test_sdpa.py` lays kv out striped and passes the block_cyclic_* arguments, `tests/reference.py` gains the natural <- physical map. No op change.

### sparse_sdpa: attention_sink under high_precision
- What: `ttnn.bringup.sparse_sdpa(..., high_precision=True, attention_sink=sink)` runs (the fork refused the pair).
  Two changes, both under the `SPARSE_SDPA_HIGH_PRECISION` define (absent when the option is off):
  - `normalize_row_streaming` (compute_streaming.hpp, sink-per-row branch): `reconfig_data_format(max, sink)` before
    the `max - sink` subtraction. Under high_precision the unpacker's srcB is still the Float32 row-sum the
    identity matmul just read; the bf16 max and sink were unpacked with that format (rel 0.17-1.39 vs float32 without
    the reconfig, measured);
  - the sink's exp takes the scale as the scores' exact exp does: bf16, rounded to nearest (was truncated, K-27).
  The device op no longer refuses an attention_sink with high_precision. No new argument: the switch is the
  combination the fork refused.
- Default unchanged: with high_precision off the define is absent, so the kernels, CBs and arguments are the source's
  (the option was already in the program hash). Unit suite 32 passed / 10 skipped (28 + the 4 new sink cases; the
  refusal test no longer expects the sink refusal); source-test check 216 tests, 0 regressions; model cases 5 passed /
  1 skipped (the 8-device Xing case), as recorded.
- Why: DeepSeek-V4.1 window attention (64 heads, K = V = 512, 128 window ids per query, a per-head sink) needs the
  sink, and the source path's bf16 running state / approximate exp (K-31) costs accuracy before two fp8 quantizers.
  The model passes scale 1 (512^-0.5 folded into q, exact) and the sink as is. Component test: pcc 0.99975, rel vs
  the CPU step 0.0212 (limit 0.0292), as a CPU emulation of the device arithmetic predicted (0.0206; most of it the
  bf16 output before the fp8 quantization).
- Tests: `tests/unit/test_sparse_sdpa_high_precision.py::test_high_precision_sink` (4 cases vs a float32 torch
  reference with the sink in the denominator): DeepSeek-V4.1 window geometry at scale 1 and 512^-0.5, a sink that
  dominates short rows, and 4 k_chunks (the sink on the SALAD-combined row-sum); rel L2 0.0023-0.0068, rows within
  [0.985, 1.012]; each case asserts the sink moves its golden by more than 3x the limit (a dropped sink fails).
  Removing the reconfig fails all 4.
- Needed by: deepseek_v41_flash_d_p C.swa.attention (`tt/attention.py:TtWindowAttention`; setting WINDOW_SDPA=source
  keeps ttnn.transformer.sparse_sdpa)
- Files: `device/kernels/compute/compute_streaming.hpp`, `device/sparse_sdpa_device_operation.cpp`, `sparse_sdpa.hpp`,
  `sdpa_nanobind.cpp`, `tests/unit/test_sparse_sdpa_high_precision.py`

### sparse_sdpa: fp32_output under high_precision
- What: `ttnn.bringup.sparse_sdpa(..., high_precision=True, fp32_output=True)` returns the normalized output in
  Float32 (default False: q's dtype, as before). With the option on: the output tensor is FLOAT32; cb_out_im (the
  normalized output) is Float32 and unpacks straight to the fp32 DEST for the untilize (`UnpackToDestFp32`; via srcA
  it would be TF32); cb_out_rm is Float32; and, behind a new `SPARSE_SDPA_FP32_OUTPUT` define, normalize_row_streaming
  reconfigures the packer to the 1/sum scratch explicitly. That last one was the bug to avoid: high_precision leaves
  the packer in the bf16 state before the normalize, and `sdpa_maybe_pack_reconfig_data_format<normalized_out_cb,
  scratch_cb>` skips the change at compile time when both CBs are Float32, so 1/sum was packed as bf16 into a Float32
  CB (output off by 1e2-1e38). Refused without high_precision. In the program hash.
- Default unchanged: with the option off there is no new define, CB format or unpack mode, so the programs are the
  previous ones. Unit suite 37 passed / 10 skipped (32 + 5 new); source-test check 216 tests, 0 regressions; model
  cases 5 passed / 1 skipped, as recorded.
- Why: DeepSeek-V4.1's window attention output goes through an fp8 q-dq; the bf16 rounding of the op's output turned
  into whole e4m3 steps and the S.swa.06 swap check failed on its worst row (0.0522 > 0.05, rel 0.0230). With the fp32
  output: rel 0.0189, worst row 0.0458 (gate PASS). Op level: rel vs float32 0.0022-0.0065 (bf16 output
  0.0028-0.0068); the fp32 output is within 0.5 bf16 steps of the bf16 one (the bf16 output is its RNE rounding).
- Needed by: deepseek_v41_flash_d_p S.swa.06 (`tt/attention.py:TtWindowAttention._sdpa`, setting
  WINDOW_SDPA_OUT=fp32; bf16 keeps the previous output)
- Files: `sparse_sdpa.{hpp,cpp}`, `sdpa_nanobind.cpp`, `device/sparse_sdpa_device_operation{.cpp,.hpp,_types.hpp}`,
  `device/sparse_sdpa_program_factory.cpp`, `device/kernels/compute/compute_streaming.hpp`,
  `tests/unit/test_sparse_sdpa_high_precision.py` (`test_fp32_output` x4: Float32 dtype, off the bf16 / TF32 grids,
  <= 0.5 bf16 steps from the bf16 output, closer to float32; `test_fp32_output_program_and_refusal`). Without the
  pack reconfig all 4 accuracy cases fail (seen during development).

### sparse_sdpa: fp32_scores (Float32 scores / probabilities, TF32 nearest rounding, float32 q)
- What: `ttnn.bringup.sparse_sdpa(..., high_precision=True, fp32_output=True, fp32_scores=True)` (default False: the
  previous program). Behind a new `SPARSE_SDPA_FP32_SCORES` define (absent when off): cb_qk_im (scores, then the
  softmax probabilities, probs@V's operand) is Float32 instead of bf16; a FLOAT32 q is accepted (cb_q_rm / cb_q_in
  Float32); and every Float32 value the FPU reads later is rounded to TF32 to nearest-even in DEST by an SFPU step on
  the pack thread before its pack (the FPU reads Float32 at TF32 by truncation, a coherent bias, K-30): the scores,
  the probabilities, the running output (probs@V), the row-sum (the k_chunk's Skt probability tiles summed in DEST by
  SFPU fp32 adds when Skt <= DEST tiles, else the previous L1-accumulated sum) and 1/sum. Explicit unpack / pack format
  restores where the kernel assumed cb_qk_im was bf16 (per query group, around the bf16 running max, after PV and
  SALAD, in reduce_c_row_group's eltwise max). The running max stays bf16. Refused unless high_precision and
  fp32_output; a float32 q is refused without fp32_scores. In the program hash.
- Default unchanged: option off = no define, the same CB formats and arguments (the program hash adds the flag). Unit
  suite 41 passed / 10 skipped (37 + 4 new); source-test check 216 tests, 0 regressions; model cases 5 passed /
  1 skipped, as recorded.
- Why: deepseek_v41_flash_d_p S.swa.11 (scaled 96-expert router) failed router vs golden 0.9887 < 0.99: the window
  attention's bf16 scores / probabilities and the FPU's truncating TF32 reads (CPU emulation: bf16 scores 0.9910,
  TF32-truncated scores 0.9940, TF32-nearest scores / q / probabilities 0.9953). With the option and q rounded to TF32
  by the model: attention vs the CPU step rel 0.0189 -> 0.0143, bias -0.00075 -> -0.00030, router vs golden 0.9918
  (gate PASS). Op level, vs float32 on a float32 q: rel 0.0026 / 0.0068 / 0.0063 (bf16 scores, bf16 q) -> 0.0009 /
  0.0022 / 0.0017. Cost: the model's attention 26.5 -> 28.6 ms per layer (2x2, chunk 2048).
- Gotcha: the SFPU is one unit shared by the threads; with half-sync DEST the math thread's next tiles overlap the
  pack thread's work on the other half, so an SFPU add issued on the math thread (add_binary_tile) next to the pack
  thread's rounding corrupted the 2-group cases (rel 2-400). All new SFPU work runs on the pack thread.
- Needed by: deepseek_v41_flash_d_p S.swa.11 (`tt/attention.py:TtWindowAttention._sdpa`, setting
  WINDOW_SDPA_SCORES=fp32; bf16 keeps the previous program)
- Files: `sparse_sdpa.{hpp,cpp}`, `sdpa_nanobind.cpp`, `device/sparse_sdpa_device_operation{.cpp,.hpp,_types.hpp}`,
  `device/sparse_sdpa_program_factory.cpp`, `device/kernels/compute/sparse_sdpa_compute.cpp`,
  `device/kernels/compute/compute_streaming.hpp`, `tests/unit/test_sparse_sdpa_high_precision.py`
  (`test_fp32_scores` x3, `test_fp32_scores_program_and_refusal`)

### sparse_sdpa: bf16_probs (DeepSeek's sparse_attn precision: Float32 scores and max, bf16 probabilities)
- What: `ttnn.bringup.sparse_sdpa(..., high_precision=True, fp32_scores=True, bf16_probs=True)` (default False: the
  previous programs). Behind a new `SPARSE_SDPA_BF16_PROBS` define (absent when off): the running max CBs (cb_max_a /
  cb_max_b) are Float32 (the max of the TF32 scores, exact; srcB reconfigured to it for the score - max subtraction
  and both maxima for the flash correction), so the max score's probability is exp(0) = 1; the row-sum is summed from
  the TF32 probabilities as before; then a new in-place pass rounds the probabilities to bf16 to nearest-even in DEST
  (SFPU, pack thread: copy -> round -> pack back, before P@V). With the option, fp32_output may be False (the output
  rounded to bf16 by the packer, the kernel's output dtype); a bf16 q is required. Refused without fp32_scores or with
  a float32 q; fp32_scores without fp32_output is still refused unless bf16_probs. In the program hash.
- Why: deepseek_v41_flash_d_p FIX.2 (overseer: match the reference precision, never above it). The checkpoint's
  inference/kernel.py sparse_attn keeps q / kv bf16, the scores, running max and sum fp32, casts P to bf16 before
  P@V, accumulates fp32 and writes a bf16 output; fp32_scores (TF32 P, fp32 q) was above it, the bf16 scores below it.
- Measured (op level, vs that kernel in float32 torch on the same bf16 inputs; rel L2): DS41 window (1 chunk)
  bf16 out 0.0020 / fp32 out 0.0010, 4 chunks 0.0027 / 0.0021, 2 DST groups 0.0024 / 0.0017; bf16 scores 0.0029 /
  0.0067 / 0.0061. Probabilities read back exactly (one-hot keys, score - max exact): all equal
  rne_bf16(rne_tf32(exp(score))) (with the rounding disabled 0.26). Model: attention 28.8 -> 26.1 ms per layer
  (2x2, chunk 2048; bf16 q and output halve two all_to_all's).
- Limits (kept, documented): the FPU's score - max subtraction is TF32-precise (a difference wider than 11 bits is
  truncated: p rel error up to |s - m| 2^-11); the flash correction exp(m_prev - m_cur) stays bf16 (cb_corr; only
  with more than one k_chunk). Found on the way: with fp32_scores alone the TF32 probabilities reach P@V with ~9 bits
  (0.495 equal rne_tf32(exp); the bf16-grid ones are the truncated values), so that variant is not TF32 in P@V.
- Default unchanged: option off = no define, the same CB formats (max bf16), kernels and arguments. Unit suite 47
  passed / 10 skipped (41 + 6 new); source-test check 216 tests, 0 regressions; model cases 5 passed / 1 skipped, as
  recorded.
- Needed by: deepseek_v41_flash_d_p FIX.2 (`tt/attention.py:TtWindowAttention._sdpa`; settings WINDOW_SDPA_PROBS=bf16
  and WINDOW_SDPA_OUT=bf16, the defaults; WINDOW_SDPA_PROBS=fp32 + WINDOW_SDPA_OUT=fp32 keep the fp32_scores program)
- Files: `sparse_sdpa.{hpp,cpp}`, `sdpa_nanobind.cpp`, `device/sparse_sdpa_device_operation{.cpp,.hpp,_types.hpp}`,
  `device/sparse_sdpa_program_factory.cpp`, `device/kernels/compute/sparse_sdpa_compute.cpp`,
  `device/kernels/compute/compute_streaming.hpp` (comment only), `tests/unit/test_sparse_sdpa_high_precision.py`
  (`test_bf16_probs` x3, `test_bf16_probs_values` x2, `test_bf16_probs_program_and_refusal`)

### Upstream sync to source @ `6b80daf43e3` (2026-10-27): full sync, `max_k_splits` -> `k_splits`, ring_mla race fix
- What: a 3-way merge (base = fork_op.py's mechanical copy of the source @ `bdc576fa652`, the fork's last sync, no
  skipped commits; theirs = the same copy @ `6b80daf43e3`; ours = this fork). The fork is now the source @
  `6b80daf43e3` plus its own changes. 44 files merged clean, 6 had conflicts, 1 new file. Taken (15 commits):
  - features (new options, default off): #57395 `attn_mask_block_map` on `scaled_dot_product_attention` (skip fully
    masked blocks of a non-causal attn_mask; also MLA decode column groups); #60804 `kv_valid_end_tensor` on
    ring_joint / ring_mla (metadata path); #58847 `enable_kv_block_cache` on sparse_sdpa_msa (per-core L1 K/V block
    cache, new `device/kernels/sparse_sdpa_msa_common.hpp`); #59195 ring-joint sliding K split (opt-in);
  - API rename: #61291 `SDPAProgramConfig.max_k_splits` -> `k_splits` (the shared config, so the C++ field, the
    Python keyword and the printed form); a chunked ring-joint call that cannot apply its `k_splits` or
    `segmented_accumulation` now fails with the reason instead of silently dropping them;
  - bug fixes (they change behaviour where they apply): #60337 ring_joint padded multicast K slots now reserve, push
    and pop like real ones (a later multicast overwrote the 3-entry latent-V K buffer compute was still reading:
    wrong ring_mla output on BH), with #60898 (the fp32-DEST legacy-path drain compiled only when padded slots can
    exist); #60804 the ring_joint reader read the logical-length metadata arguments one slot early (4 expected, 5
    sent); #59419 ring_mla converts a 2D topology to 1D (Mesh / Torus no longer fail);
  - perf, on by default, claimed bit-identical: #57395 Blackhole causal / joint K/V forwarding core to core and the
    leftover work on the last cores; #60138 Blackhole streaming compute packs rows in blocks from 2 tiles wide;
    #59338 split-KV ring MLA packed-source schedule compiled only when sources are grouped;
  - numerics: #59195 with a multi-hop sliding halo the local K chunks go first (accumulation order, small change;
    no fork caller uses that path);
  - build: #61991 / #60172 kernel includes (`api/tile_constants.h`, `api/debug/kernel_profiler.hpp`), #61011 /
    #59941 `IDevice` -> `MeshDevice` in ring_fusion and the all-gather params, #58910 a clang `static_assert`.
- Conflicts (6), each resolved as upstream's text with the fork's change re-applied:
  - `device/kernels/compute/sparse_sdpa_msa_compute.cpp`: the fork had only formatter whitespace there; upstream's
    file taken whole (#58847 moved its compile-time args to `sparse_sdpa_msa_common.hpp`).
  - `device/ring_joint_sdpa_program_factory.cpp`: upstream's #61291 K-split block (`k_splits`, `single_q_chunk_blocker`)
    with the fork's spelling `ttnn::transformer::bringup::sdpa::streaming_qktv_h` (fork_op.py writes
    `...::sdpa::bringup::`). The fork's streaming rule (`use_streaming_compute = !fp32_dest_acc_en ||
    v_shares_k_buffer`) and its bf16 `sum_df` / `qk_im_df` on the streaming path merged clean.
  - `device/sdpa_device_operation.{hpp,cpp}`: upstream's new `attn_mask_block_map` parameter, the fork's
    `ttnn::operations::transformer::PagedCacheGeometryOverride` (the shared type stays in the source namespace).
  - `device/sdpa_program_factory.cpp`: the fork's one-line windowed TensorAccessorArgs plus upstream's new
    `attn_mask_block_map` reader accessor.
  - `sdpa.cpp` (ring_mla invoke): upstream's new arguments (`circular_kv_cache`, `logical_n/l_tensor`,
    `kv_valid_end_tensor`) with the fork's `prim::bringup::RING_JOINT_SDPA_*_IDX`.
  After the merge, no spelling the fork fixed by hand came back (`bringup::bringup`, a nested `sdpa::bringup` /
  `sdpa_cb::bringup` / `detail::bringup`, `ttnn::operations::bringup::{SDPAProgramConfig,PagedCacheGeometryOverride}`,
  `operations::transformer::sdpa::`, unqualified `prim::*_IDX`, `transformer::SparseKVFormat`), and no kernel include
  the new base refuses (`tools/profiler/...`, `<tt-metalium/constants.hpp>`, deleted compute init calls).
- Default behaviour: the fork's own changes are kept (narrower non-MLA V, sparse_sdpa high_precision / fp32_output /
  fp32_scores / bf16_probs, ring_mla at fp32 DEST on the streaming path, ring joint kv_actual_isl=0 full prefill);
  the upstream bug fixes and Blackhole perf paths above apply to every caller. Untested combinations to re-run:
  #57395's Blackhole K/V forwarding with a narrow V (the reader and factory size V by vDHt throughout;
  `test_sdpa_narrow_v.py`), #60337 on the fork's fp32-DEST ring_mla (`test_ring_mla_fp32_dest.py`,
  `test_ring_mla_single_chunk.py`), #60138 under high_precision (`test_sparse_sdpa_high_precision.py`).
- Tests: the printed `SDPAProgramConfig` now reads `k_splits=` instead of `max_k_splits=`, so the five captured calls
  that pass one get new signatures (recomputed from the recorded calls, to be confirmed by a fresh capture):
  mimo_v2_6_d_p `3ec3d15624` -> `71d36ab65c`, `3cd893eaad` -> `ed276d5784`; mimo_v2_6_d_p_2x2 `a6a02addce` ->
  `1b59f2c46a`, `9994ef8a85` -> `b0757cea9d`; xing40_a4b_d_p `dfd9514ea7` -> `b3592c11fc`. The carried
  `test_sdpa_prefill.py` gains #57395's 4 block-map tests, so the source baseline count changes. No model passes
  `max_k_splits` (gemma4_d_p's `ring_prefill.py` is upstream's and was renamed by #61291 itself).
- Needed by: rebase onto llk_helper_library @ `6b80daf43e3` (2026-10-27); the fork does not build without #61291
- Files: 50 merged files under `device/`, `sdpa.{cpp,hpp}`, `sdpa_nanobind.cpp`, `sparse_sdpa.cpp`, `sparse_sdpa_msa.{cpp,hpp}`; new
  `device/kernels/sparse_sdpa_msa_common.hpp`; `README.md` (sync SHA)

### sparse_sdpa: opt-in kv_format ROW_BFP4 (bfloat4_b row records + a bf16 window tensor)
- What: `ttnn.bringup.SparseKVFormat.ROW_BFP4` with `kv_window=` (default: none; any other format refuses it). kv is
  `[B, 1, T, 320]` uint8 row records (K_DIM 512: the 32 shared exponents twice, then 256 bytes of nibbles, value 2i in
  the low nibble of byte i; written by `ttnn.bringup.ckv_bfp4`), kv_window `[1, 1, Tw, 512]` bf16. The index row is
  `[window k_chunk | top-k]`, each part with its own sentinel tail (a top-k part with no valid key is allowed).
  - Chunk 0 gathers the window rows and is tilized in bf16 as before.
  - Chunks >= 1: the data-movement RISCs only issue NoC reads, two a row on both NoCs: the exponent copy whose offset
    is congruent with its destination mod 64 -> the staging block's exponent region, the nibbles -> its data region
    (the sentinel tail zeroed by NoC zero writes). The block [k_chunk x 512] is then a row-major bfloat4_b tensor.
  - Compute turns it into the chunk's bfloat4_b K tiles (`cb_k4`) with the unpacker: tile descriptor x = 16,
    y = 32 groups, z = k_chunk rows, one UNPACR per face row (16 values, 1 exponent; z += 1, SrcA row += 1), the MOP
    doing a face (16, the last with SrcA valid, then SrcB valid), the stock datacopy -> DEST and a bfp4 pack (exact:
    the values are on the grid). Both matmuls read the tiles as they are; no tilize, no scale multiply.
  - ISA basis (tt-isa-documentation UNPACR_Regular): tileize mode is undefined for BFP4; outside it a BFP datum's
    exponent is read by datum index from the exponent section; a BFP4 run may start mid 16 B word when it starts a
    new exponent.
  - Afterwards the conversion restores what it changed and the BF16 path's tilize leaves: the unpacker's SrcA Y
    stride, tile x dim and descriptor (saved / written back), its X range, its Y / Z counters (left set they broke the
    next matmul's unpack: rel inf), the base offset registers; the packer as tilize_uninit leaves it (a default pack
    init on the bf16 K CB), L1 accumulation off for its packs.
  - The window chunk gets its own boundary mask (control word WINDOW_VALID).
- Default unchanged: everything is behind the define `SPARSE_SDPA_ROW_BFP4` (reader, writer, compute), added only for
  ROW_BFP4, with its compile-time args (after each block's END) and the kv_window address (after the fixed common args)
  only then. The request and control pages keep their sizes. The compute's matmul / reconfig calls take the chunk's K
  CB as a value equal to `cb_k_in` without the define. The program hash adds kv_window's memory config.
- Why: deepseek_v41_flash_d_p FIX.8, the compressed KV attention table in 4 bits at the reference's width (F208):
  3.2x less table DRAM and gather traffic than bf16 rows (320 vs 1024 B); owner rule 2026-10-28: the data-movement
  RISCs do no byte work (a first version shuffled rows into tiles on them: 517 / 1300 us, withdrawn).
- Tests: `tests/unit/test_sparse_sdpa_row_bfp4.py` (6 cases, one device, K_DIM = v_dim 512, top-k 512 after a 128
  window, sink): a full window equals the BF16 path bit for bit on one bf16 table holding the same values (H 32 / 64,
  fp32 dest off / on, top-k parts of 0 / 1 / 31 / 32 / 200 / 512 keys); a partial window (1 / 17 / 32 / 100 / 127 keys)
  vs float32: rel 0.0249 (the BF16 path with the compacted list: 0.0250); deterministic; the refusals.
  `test_sparse_sdpa_high_precision.py` 27 pass (option off unchanged).
- Time (one p150b, H 64, K_DIM 512, 640 indices all valid, 30 back-to-back calls; BF16 on one bf16 table; probe
  tests/ttnn/unit_tests/operations/ds41_ckv_bfp4/probes/probe_001.py): 160 queries (a Galaxy 8x4 chip at chunk 5120)
  BF16 341 us, ROW_BFP4 353 us (+3.5 %); 512 queries (a 2x2 chip) 1182 vs 1245 us (+5 %). The unpacker conversion
  costs ~7.5 us per 128-row chunk on the compute side (4096 one-row UNPACRs, ~2.5 cycles each) vs ~3.3 us for the bf16
  tilize; with no conversion at all (an ablation, wrong output) it was 251 / 957 us.
- Needed by: deepseek_v41_flash_d_p FIX.8 (CKV_FORMAT=bfp4_rows)
- Files: `sparse_sdpa.hpp`, `sparse_sdpa.cpp`, `sdpa_nanobind.cpp`, `device/sparse_sdpa_device_operation*.{hpp,cpp}`,
  `device/sparse_sdpa_program_factory.cpp`, `device/kernels/sparse_sdpa_common.hpp`,
  `device/kernels/dataflow/sparse_sdpa_{reader,writer,gather}.{cpp,hpp}`, `device/kernels/compute/sparse_sdpa_compute.cpp`,
  `tests/unit/test_sparse_sdpa_row_bfp4.py`

### sparse_sdpa: a bfloat8_b window (kv_window uint8) and kv_format ROW_BFP8
- What: with ROW_BFP4, `kv_window` may be `[1, 1, Tw, 576]` uint8 bfloat8_b row records (`ckv_bfp4(dtype=bfloat8_b)`;
  define `SPARSE_SDPA_WINDOW_BFP8`): the window chunk is gathered like the top-k chunks (two NoC reads a row: the
  congruent exponent copy, the 512 value bytes) into its own staging block (`cb_k8_rm`) and turned into bfloat8_b K tiles
  (`cb_k8`) by the same unpacker conversion; the bf16 window CBs shrink to one page. New `SparseKVFormat.ROW_BFP8` (a
  window-only layer): kv = the records, TOPK == k_chunk_size, no kv_window (the ROW_BFP4 kernels with kv as the window).
  Default unchanged: everything behind the define, added only for a uint8 window; the hash adds the window's dtype.
- Why: deepseek_v41_flash_d_p lowbit, the window KV at the reference's fp8 width (owner 2026-10-29).
- Tests (`tests/unit/test_sparse_sdpa_row_bfp4.py`, 5 new): the bfloat8_b window equals the bf16 window holding the
  same values bit for bit (full and partial windows, top-k parts 0 / 37 / 512, fp32 DEST off / on); ROW_BFP8 equals
  the BF16 path on the same values (H 32 / 64, windows 1..128 keys); both deterministic; the refusals. Regression: 38
  unit + the model case pass.
- Time, one p150b (H 64, K 512): compressed layers (640 keys) 160 q 351 -> 362 us (+3.3 %), 512 q 1250 -> 1286 us
  (+2.9 %); window-only layers (128 keys) 160 q 133 -> 138 us, 512 q 494 -> 567 us (BF16 -> ROW_BFP8); plain fp8 rows
  (FP8_E4M3, no scales) 119 / 484 us.
- Needed by: deepseek_v41_flash_d_p DS41-150 (lowbit phase 1)
- Files: sparse_sdpa.hpp, sdpa_nanobind.cpp, device/sparse_sdpa_device_operation*.{hpp,cpp},
  device/sparse_sdpa_program_factory.cpp, device/kernels/sparse_sdpa_common.hpp,
  device/kernels/dataflow/sparse_sdpa_{reader,writer,gather}.{cpp,hpp}, device/kernels/compute/sparse_sdpa_compute.cpp,
  tests/unit/test_sparse_sdpa_row_bfp4.py
