# indexer_score (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/indexer_score`
- Source SHA: `2761111778f18601772ccce3a8e03b8c0fe8ea8d`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.*`)
- Forked for: hy4_preview_d_p C.dense_full.indexer
- Used by: hy4_preview_d_p

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_indexer_score`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### fp32 DEST for DSA scoring (opt-in), with the mask srcA reconfig it needs
- What: `indexer_score_dsa` / `ring_indexer_score_dsa` accept `compute_kernel_config.fp32_dest_acc_en=True` for DSA
  scoring (num_groups 1, block_size 0, learned gates); MSA and block pooling still reject it. The factories already
  sized DEST (4 tiles half-sync) and the qk / accumulator CBs (Float32) for fp32 DEST; the device op only forbade it.
  The compute kernel gains one `reconfig_data_format_srca(cb_qk, cb_mask)` before the causal mask on the non-fused
  path, compiled only under `INDEXER_SCORE_FP32_DEST`, a define the host sets only when fp32 DEST is on. Default
  (fp32_dest_acc_en False) = the source behaviour, the same kernels, defines, compile args and CBs.
- Why: on the Hy4 layer-0 indexer golden (s4096 chunk 1, 32 heads x 128, top-2048 of 4096 keys) the source op's
  logits have rel error 0.024 vs fp32 (every row ~1.7% low: the bf16 DEST MAC truncates; plus ~1.5% per product from
  the blocked custom multiply, see below) and the top-k set overlap is 0.985, under the 0.99 gate. With fp32 DEST
  alone the mask broke: the mul phase leaves srcA in cb_qk's format (Float32 under fp32 DEST) and the bf16 -inf mask
  tiles were unpacked as fp32 (rows 16..31 of full tiles unmasked, diagonal tiles wrong, -inf in causal cells). With
  the reconfig, at k_chunk 32: logits rel 0.0036, overlap 0.99708 (fp32 scores rounded to bf16: 0.99705).
- Note (unchanged, documented): with k_chunk_size > 32 the head reduction uses the blocked custom bcast-col multiply
  (`_llk_math_bcast_cols_reuse_custom_`), which issues one ELWMUL per face with no fidelity phases, so it multiplies
  at LoFi-like precision whatever the requested fidelity (HiFi2 and HiFi4 gave the same logits). k_chunk_size 32
  takes the per-column path, whose `mul_tiles_bcast_cols` honours the fidelity. Random inputs, 32 heads: per-column
  fp32 DEST rel 0.0019, bf16 DEST 0.0118; blocked fp32 DEST 0.029, bf16 DEST 0.023.
- Needed by: hy4_preview_d_p C.dense_full.indexer
- Files: device/indexer_score_device_operation.cpp (validate), device/indexer_score_program_factory.cpp and
  device/ring_indexer_score_dsa_program_factory.cpp (the define), device/kernels/compute_indexer_score.cpp (the
  reconfig), tests/unit/test_fp32_dest.py; tests/source.yaml (source selection; the one expected divergence is
  test_indexer_score_rejects_fp32_dest_acc).

### Build fix after fork_op.py
- What: `ttnn::experimental::bringup::ccl::` back to `ttnn::experimental::ccl::` (FusedOpSignalerMode,
  AllGatherFusedOpSignaler) in device/ring_indexer_score_dsa_program_factory.cpp: fork_op.py nested a sibling op's
  namespace reference.
- Needed by: hy4_preview_d_p C.dense_full.indexer
- Files: device/ring_indexer_score_dsa_program_factory.cpp

### Value repr for IndexerScoreProgramConfig
- What: `IndexerScoreProgramConfig.__repr__` returns
  `IndexerScoreProgramConfig(q_chunk_size=.., k_chunk_size=.., head_group_size=..)` (was the default
  `<... object at 0x...>`). No op change.
- Why: the fork-call capture (models/demos/common/bringup/testing/fork_capture.py) records non-tensor arguments by
  str(), so every captured ring_indexer_score_dsa call had a new signature on every run and no test case could match
  it. With this repr (and a value repr for the global semaphore handle, set in ttnn/ttnn/bringup/__init__.py) the
  model's three per-layer calls collapse to one stable signature.
- Needed by: hy4_preview_d_p O.1
- Files: indexer_score_nanobind.cpp

### Tests: fork test suite with a first model case
- What: `tests/` model-case suite (cases.py, reference.py, test_indexer_score.py) with a random-input case for the
  ring_indexer_score_dsa call hy4_preview_d_p makes (2x2 mesh, SP 2 x TP 2, 4-chip striped block-cyclic key cache,
  q 1280 rows x 32 heads x 128, T 56320, chunk start 51200, HiFi4 + fp32 DEST, q 64 x k 32) (bringup-fork-tests
  skill). No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: hy4_preview_d_p O.1
- Files: `tests/cases.py`, `tests/reference.py`, `tests/test_indexer_score.py`

### Upstream bug fixes ported (source @ `73027b6e6ff`, 2026-10-01)
- What: three source fixes since the fork's SHA, applied unchanged (fork namespaces kept):
  (a) #56992 / #58363 fused full-mesh validation checks per-device sequence EXTENTS (K-local x ring = gathered K;
  Q and weights share Sq) instead of shard placements, which under-counted for activations whose shard dim names the
  axis they were created on (`indexer_score_device_operation.cpp`, both entry points);
  (b) #58363 full-mesh causal geometry on mid-slab chunk starts: queries regroup into (SP rank, TP window) via
  `query_geometry_split` / `query_geometry_ranks` (`indexer_score_host_common.hpp`), and the ring factory hands the
  reader the same split and ranks (`ring_indexer_score_dsa_program_factory.cpp`);
  (c) pooled-output scratch rows strided at the full unit width so each row's NoC write keeps its destination's
  alignment, also for a partial last k-band (`kernels/writer_indexer_score.cpp`).
  Not taken: #58122 (GLM-5.2 -> 5.3 comment rename only).
- Default behaviour: (a)/(b) only change the fused full-mesh mode (hy4's ring call is not full-mesh); (c) changes
  only block-pool output layouts the source wrote misaligned.
- Needed by: rebase onto origin/malimpic/llk_helper_library_rebased_0110_2 (keep the fork's bug fixes current)
- Files: `device/indexer_score_device_operation.cpp`, `device/indexer_score_host_common.hpp`,
  `device/kernels/writer_indexer_score.cpp`, `device/ring_indexer_score_dsa_program_factory.cpp`

### Key stride (pooled keys, pool-causal mask)
- What: new argument `key_stride: int = 1` (R) on `indexer_score_dsa` and `ring_indexer_score_dsa`. R > 1: the K
  sequence is 1/R of the query token sequence; key j pools tokens [R*j, R*j + R) and is visible to query token p iff
  R*j + R - 1 <= p. T, kv_len, the K cache / gathered buffer rows and the block-cyclic key stripe
  (block_cyclic_chunk_local / R keys per chip per chunk) are in KEY units; chunk_start_idx, q rows,
  block_cyclic_chunk_local and the causal geometry stay in TOKEN units. The token diagonal of a q tile maps to key
  tile diag/R with staircase pattern diag%R (`key_diagonal`, kernels/indexer_score_work_split.hpp); the reader
  builds R staircase tiles + the -inf tile (cb_mask R + 1 tiles; -inf iff col >= pattern*32/R + (row+1)/R), compute
  stamps them on the diagonal key tile, the invP remap and the ring's gather extent use the key stripe. Validation in
  key units (chunk_start_idx/R < T and < kv_len; chunk_local % (R*32) == 0; T % (sp*chunk_local/R) == 0), R in
  {1, 2, 4, 8}, DSA only, host-scalar path only (chunk_start_idx_tensor / valid_end_tensor / cache_batch_idx_tensor
  rejected with R > 1); an omitted chunk_start_idx deduces R*T - chunk. key_stride is hashed; the kernel define
  INDEXER_SCORE_KEY_STRIDE is set only for R > 1. Default (1) = the source behaviour: same kernels, defines, compile
  args and CBs (cb_mask 2 tiles, indices 0 = diagonal, 1 = -inf).
- Why: GLM-5.3 Flash's DSA indexer scores pooled keys (4 tokens per key) with a pool-level causal mask; the op could
  only score one key per token, so the model ran it unmasked plus a separate mask op.
- Tests: tests/unit/test_key_stride.py (single device, R 4 and 2, contiguous K, starts 0/32/96/640/1792 tokens,
  runtime kv_len, k_chunk 32/64: -inf placement exact, rel L2 0.0018-0.0019 vs float32 at HiFi4 + fp32 DEST);
  tests/unit/test_key_stride_ring.py (2x4 LoudBox, full-mesh snake ring, R 4, block_cyclic_chunk_local 640, chunk
  starts 5120 / 10240: every chip pcc 0.9999985, rel 0.00186; metadata path refused); the tests/cases.py
  glm53_flash_d_p entry. R = 1: the carried source tests (131 pass, the one expected divergence), test_fp32_dest.py
  and the hy4 ring case unchanged.
- Needed by: glm53_flash_d_p indexer ring
- Files: device/indexer_score_device_operation_types.hpp, device/indexer_score_device_operation.cpp / .hpp,
  device/indexer_score_program_factory.cpp, device/ring_indexer_score_dsa_program_factory.cpp,
  device/kernels/indexer_score_work_split.hpp, device/kernels/indexer_score_cb.hpp,
  device/kernels/indexer_score_common.hpp, device/kernels/compute_indexer_score.cpp,
  device/kernels/reader_indexer_score.cpp, indexer_score_nanobind.cpp, tests/reference.py, tests/cases.py,
  tests/test_indexer_score.py (skips key-stride cases), tests/unit/test_key_stride.py,
  tests/unit/test_key_stride_ring.py
