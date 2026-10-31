# indexer_score (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/indexer_score`
- Source SHA: `2761111778f18601772ccce3a8e03b8c0fe8ea8d`
- Python: `ttnn.bringup.*` (was `ttnn.experimental.*`)
- Forked for: hy4_preview_d_p C.dense_full.indexer
- Used by: hy4_preview_d_p, glm53_flash_d_p

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

### Upstream sync to source @ `bdc576fa652` (2026-10-12): compressed-key indexer, MSA KV dedup
- What: a 3-way merge (base = fork_op.py's mechanical copy of the source @ `73027b6e6ff` without #58122, the fork's
  last sync; theirs = the same copy @ `bdc576fa652`; ours = this fork), no conflicts, of #57194 (compressed-key
  indexer_score and the CSA indexer), #58812 (MSA index_k cache stored once over TP), and #58122 (GLM-5.2 -> 5.3
  comment rename, skipped at the last sync). The fork is now the source @ `bdc576fa652` plus its own changes.
- Default behaviour: the fork's own changes (opt-in fp32 DEST for DSA scoring, k_chunk 32, config `__repr__`) are
  kept; the new source modes are opt-in.
- Needed by: rebase onto origin/llk_helper_library @ `bdc576fa652` (2026-10-12)
- Files: 12 merged files under `device/` and `indexer_score_nanobind.cpp`

### Tests: glm53_flash_d_p ring case (O.1, 2026-10-13)
- GLM's captured ring_indexer_score_dsa call (2x2 full-mesh snake ring, key_compression_ratio 4, chunk start 51200, kv_len 14080) replaces its planned 2x4 sig-None case, with a new `test_ring_indexer_score_dsa_key_stride` model-case test; the 2x4 config moved inline into `tests/unit/test_key_stride_ring.py`. No op change.

### key_compression_ratio 2 and natural_query_blocks (opt-in), 2026-10-16
- What: (a) `key_compression_ratio=2` accepted by `indexer_score_dsa` / `ring_indexer_score_dsa` (the validations
  allowed 1 or 4; the mask, partial-mask index and work split were already generic for any ratio dividing 32: ratio 2
  gives residues {0, 16}). The refusal text keeps its old prefix ("must be 1 or 4, or 2 in the bring-up fork"), so
  the source's and the fork's refusal tests match unchanged. (b) `ring_indexer_score_dsa(..., natural_query_blocks=
  False)`: True = the queries are NATURAL contiguous blocks at any tile-aligned start (device at SP rank d, TP rank c
  holds positions chunk_start_idx + d * block_cyclic_chunk_local + c * Sq + [0, Sq)) while the keys keep the
  block-cyclic cache layout: `device_causal_geometry` returns that linear diagonal with no straddle. New attribute
  `natural_query_blocks` (hashed), refused without a block-cyclic layout on an axis ring and with the metadata
  (`chunk_start_idx_tensor`) path. Default False = the source's rotation-exact geometry; ratio 1 / 4 untouched.
- Why: DeepSeek-V4.1's index keys pool 2 tokens (causal t < (pos + 1) // 2). Its layers compute natural blocks of
  the chunk (row r = positions start + r W) at every 32-aligned start (serving contract), while the served index-key
  cache is block-cyclic by position. The source geometry assumes the server's rotated query order at a mid-slab start:
  at start 32 on 2x2 (S 2048, W 1024, Sq 512) 16384 future columns of one device came out finite
  (tests/unit/test_natural_query_blocks.py with the option off). With it: every start (0, 32, 1088, 2048, 2592, 4064)
  masks exactly, visible columns PCC 0.9999985, rel 0.00186 vs fp32 torch on the same bf16 inputs; the model's
  component gate C.enc_source.indexer err 0.0174 <= 0.0329.
- Regression, option off: tests/unit 17 passed, 1 skipped (the 2x4 case on this 2x2 box); model cases 2 passed;
  fork_source 134 tests, 0 regressions vs baseline (the one failure is the recorded test_indexer_score_rejects_fp32_dest_acc).
- Needed by: deepseek_v41_flash_d_p C.enc_source.indexer
- Files: device/indexer_score_device_operation.cpp (validations, launch, ring entry, hash),
  device/indexer_score_device_operation.hpp, device/indexer_score_device_operation_types.hpp,
  device/indexer_score_host_common.hpp (geometry), indexer_score_nanobind.cpp, tests/unit/test_natural_query_blocks.py

### Block epilogue: block_bias / block_max_out / block_max_pin (opt-in), 2026-10-20
- What: `ring_indexer_score_dsa(..., block_keys=0, block_bias=None, block_max_out=None, block_max_pin=False)`. With
  `block_keys` > 0 (a divisor of 32; exactly when a tensor is given) the WRITER works on each untilized score strip in
  L1 before its rows go out, per fragment of consecutive logical key tiles: `block_bias` [1, 1, Sq, NB] bf16
  TILE (NB block_keys >= T, NB % 32 == 0; a strip reads its row block's bias tile(s), 2 KB each, issued before the
  strip arrives and kept while the next strip needs the same tile): every score of a block (block_keys adjacent
  logical keys) whose bias is nonzero (-inf) is written -inf, 0 keeps it;
  `block_max_out` [1, 1, Sq, NBm] fp32 ROW_MAJOR (NBm block_keys >= T, NBm % 4 == 0, block_keys <= 8 so a tile's run
  is 16-byte aligned), written in place: the max of each block's written bf16 scores, exactly widened to fp32, for
  the blocks of the first kv_len keys (later columns untouched); `block_max_pin`: +inf there at each query's own
  block (its newest visible key (pos + 1) // ratio - 1). New attributes `block_keys`, `block_max_pin` and
  tensor_args `block_bias`, `block_max_out` (presence and widths hashed); the factory adds the writer define
  `INDEXER_SCORE_BLOCK_EPILOGUE`, two scratch CBs after every other CB, appended writer CT args and two common args
  after the shard order, only when the option is on. Off (default): the same kernels' code paths, args and CBs.
- Why: DeepSeek-V4.1's candidate pool (layer 20: max over each block of 8 keys of the [Sq, T] scores, the newest
  block pinned) took 8 strided slices of [512, 55296] bf16 + 7 maximum + the pin ops (~9 ms of a chunk, glue report
  item 3); its re-index layers (24-36) added a 0 / -inf block bias by repeat_interleave x 8 + add (~3.2 ms, item 8).
- Speed (2x2 p150, layer 20's shapes: Sq 512 x 32 heads x 128, T 55296, kv_len 55296, HiFi2 + fp32 DEST, q 64 x k
  32, 30 synced calls): plain 1.68 ms; + block max and pin 1.96 (first scalar version 3.07: per-value branches and
  16-bit loads; now 32-bit loads and Zbb max / min on the fp32 bit patterns); + bias 1.69 (first version with
  ROW_MAJOR bias rows, one 64-byte read per query row per strip: 2.16). In the model (P.16 plain profile) the indexer
  section fell 25.15 -> 12.94 ms a chunk.
- Exactness: tests/unit/test_block_epilogue.py (2x2, ratio 1, natural blocks, block-cyclic keys, T 6144, starts 0 /
  32 / 1088 / 2048 / 2592 / 4064, block_keys 8 and 4): biased scores, block maxes (plain and biased) and the pin equal
  the plain call's scores bit for bit; columns past kv_len / block_keys keep a sentinel; 7 refusals. Negative
  controls by hand (a corrupted max, a bias that skips a key): both fail.
- Needed by: deepseek_v41_flash_d_p P.16
- Files: device/indexer_score_device_operation_types.hpp, device/indexer_score_device_operation.hpp,
  device/indexer_score_device_operation.cpp (launch validation, hash), device/ring_indexer_score_dsa_program_factory.cpp,
  device/kernels/writer_indexer_score.cpp, indexer_score_nanobind.cpp, tests/unit/test_block_epilogue.py

### Tests: deepseek_v41_flash_d_p ring cases (O.1, 2026-10-23)
- What: 3 random-input cases for the ring_indexer_score_dsa calls deepseek_v41_flash_d_p makes (2x2, natural query
  blocks, last chunk of 55296 tokens, HiFi2 + fp32 DEST: ratio 2; ratio 1 with block_bias; ratio 1 with block_max_out
  + pin), run by test_ring_indexer_score_dsa_ds41 (excluded from the key-stride test): -inf / finite exact, visible
  scores within noise_limit with the FPU's operand bits at HiFi2 (reference.dsa_score_at(fidelity=): measured 7.7e-3,
  the model's own 7.7e-3, limit 1.55e-2), block max bit-exact vs the op's own scores, determinism with fresh k / max
  buffers. No op change.
- Why: task O.1, every call a model makes to a fork gets a case.
- Needed by: deepseek_v41_flash_d_p O.1
- Files: tests/cases.py, tests/reference.py, tests/test_indexer_score.py
- Follow-up (O.1 attempt 4): test_ring_indexer_score_dsa_ds41 writes every call's fresh k buffer (garbage) before the
  first call is enqueued. Writing it from the host per call raced: a peer that finished call N starts N+1 and writes
  its shard into this chip's new buffer while this chip still runs N; the host write queued after N then overwrites
  it. Probe (ratio 2 case, 2 cycled semaphore sets): 4 of 299 calls on chip 3 (the most causal work) wrong on all
  13824 remote-shard columns (or a few tiles); pre-written buffers 0 of 299 and 0 of 599. The model's buffer is
  written once at load (tt/indexer.py _kbuf). No op change. The hy4 / glm53 ring tests (_call, _call_key_stride)
  still write per call (latent; they passed). DS41-111.

### Upstream sync to source @ 6b80daf43e3 (rebase onto llk_helper_library, owner: full sync, 2026-10-27)
- Taken: two kernel include changes, the arch read from the device. Clean merge, no behaviour change.

### q / k as bfloat4_b
- What: q and k (and k_local) may be BFLOAT4_B (validation; the CB formats come from the dtypes, not a bf16 / bfp8
  switch). The default fidelity is LoFi when both are block float (bfloat8_b or bfloat4_b), else HiFi2, as before for
  bfloat8_b; a caller config overrides it (DeepSeek-V4.1 passes HiFi2: at LoFi the gate step reads relu(q.k) with 4
  bits).
- Why: deepseek_v41_flash_d_p lowbit, the index keys and query at the reference's fp4 width (owner 2026-10-29). bf16 /
  bfloat8_b programs unchanged (same formats).
- Tests: two ds41 ring cases (2x2, ratio 2 / 1, 55296-token context) with bfloat4_b q / k on the host packer's grid
  (`reference.bfp_grid`): within the noise limit, deterministic. Whole folder: 27 pass, 1 skipped (test_key_stride_ring
  needs 8 devices).
- Time, one p150b (indexer_score_dsa, 32 heads x 128, 512 queries, HiFi2 + fp32 DEST): T 27648 787 -> 774 us, T 55296
  1519 -> 1507 us (-1.7 %; compute-bound). The ring's key traffic drops 3.6x (bfloat4_b tile 576 B vs bf16 2048 B), not
  measured on one card.
- Needed by: deepseek_v41_flash_d_p DS41-150 (lowbit phase 1)
- Files: device/indexer_score_device_operation.cpp, device/indexer_score_program_factory.cpp,
  device/ring_indexer_score_dsa_program_factory.cpp, tests/cases.py, tests/reference.py, tests/test_indexer_score.py

### max_grid: the op's cores inside a rectangle (deepseek_v41_flash_d_p FIX.15)
- What: option `max_grid_x` / `max_grid_y` (int, default 0 = no limit) in `ttnn.bringup.IndexerScoreProgramConfig` (indexer_score_dsa, ring_indexer_score_dsa): every program of the op places its cores (and splits its work) inside (0, 0) .. (x - 1, y - 1) of the compute grid. Default: the device's whole grid, the call as before (same program, same hash).
- Why: agent rule 18 (owner 2026-10-30): a Galaxy chassis' power supply trips when a chip's current rises too fast (di/dt); no op may run on more than 110 of a chip's 120 Tensix cores. DeepSeek-V4.1 passes tt/cores.max_grid() = 11 x 10 on a Galaxy, None elsewhere.
- Measured: 8x4 Galaxy, one 5120-token chunk (ring_indexer_score_dsa): 114 -> 110 cores a chip (tests/perf/test_core_audit.py in deepseek_v41_flash_d_p); per-op device time in that model's BREADCRUMBS FIX.15.
- Needed by: deepseek_v41_flash_d_p FIX.15
- Files: device/indexer_score_device_operation.cpp, device/indexer_score_device_operation_types.hpp, device/indexer_score_program_factory.cpp, device/ring_indexer_score_dsa_program_factory.cpp, indexer_score_nanobind.cpp
