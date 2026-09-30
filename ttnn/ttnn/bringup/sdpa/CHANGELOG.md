# sdpa (fork)

- Source: `ttnn/cpp/ttnn/operations/transformer/sdpa`
- Source SHA: `99f7e834cea8bc42f0b478b9cc59a7b89f2f9c1a`
- Python: `ttnn.bringup.*` (was `ttnn.transformer.*`)
- Forked for: mimo_v2_6_d_p attention V head dim 128 (non-MLA GQA)
- Used by: mimo_v2_6_d_p (tt/attention.py: full and sliding layers, V 128)

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
