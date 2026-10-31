# ttnn.bringup.scaled_dot_product_attention, ttnn.bringup.chunked_scaled_dot_product_attention, ttnn.bringup.sparse_sdpa, ttnn.bringup.sparse_sdpa_msa, ttnn.bringup.ring_mla, ttnn.bringup.joint_scaled_dot_product_attention, ttnn.bringup.ring_joint_scaled_dot_product_attention, ttnn.bringup.exp_ring_joint_scaled_dot_product_attention, ttnn.bringup.ring_distributed_scaled_dot_product_attention, ttnn.bringup.flash_mla_prefill, ttnn.bringup.chunked_flash_mla_prefill, ttnn.bringup.SparseKVFormat

## What it does
Fork of the `ttnn.transformer` SDPA folder (`ttnn/cpp/ttnn/operations/transformer/sdpa`), synced to source @ `6b80daf43e3`.
Every callable above is bound; the program config is the shared `ttnn.SDPAProgramConfig` (not forked). The fork adds:
- `scaled_dot_product_attention` / `chunked_scaled_dot_product_attention`: a non-MLA V narrower than K (GQA). Pass a
  V whose last dim is smaller than Q/K's (multiple of 32, <= K's); the output has V's width. No new argument.
- `sparse_sdpa`: opt-in `high_precision` (Float32 running output / row-sum / 1/sum, exact softmax exp; also with
  `attention_sink`), `fp32_output`, `fp32_scores` (Float32 scores and probabilities, TF32 round-to-nearest in DEST,
  float32 q), `bf16_probs` (Float32 running max, probabilities rounded to bf16 before P@V: DeepSeek sparse_attn precision).
- `sparse_sdpa`: opt-in `kv_format=SparseKVFormat.ROW_BFP4` + `kv_window=`: K / V from 320 B bfloat4_b row records
  (`ttnn.bringup.ckv_bfp4`) gathered by NoC reads and turned into bfloat4_b tiles by the unpacker; the first k_chunk
  (the window) from a separate bf16 tensor, or (opt-in) from 576 B bfloat8_b row records read the same way; and
  `kv_format=ROW_BFP8` (a window-only layer: kv = the bfloat8_b records). Realistic for a cache the reference stores in 4
  (compressed) / 8 (window) bits.
- `ring_mla` (latent V) runs at `fp32_dest_acc_en=True` on the streaming path (the source refuses it). Host only.
- `ring_mla` / `ring_joint_scaled_dot_product_attention`: host `kv_actual_isl=0` with per-device Q.seq == K.seq
  (non-chunked causal) runs the full-prefill path (the source raises TT_FATAL).
Everything else as `ttnn.transformer.<same name>`.

## Consider it instead of
- `ttnn.transformer.scaled_dot_product_attention` / `chunked_scaled_dot_product_attention` with V zero-padded to K's
  head dim (e.g. K 192 / V 128): the fork skips the padding and its compute; output equals the padded run's first V
  columns bit for bit (CHANGELOG.md).
- `ttnn.transformer.ring_mla` / `ring_joint_scaled_dot_product_attention` when a single-chunk cache is called with
  `kv_actual_isl=0` (the source raises TT_FATAL).
- Not `ttnn.transformer.sparse_sdpa`: the stock op is the realistic choice for sparse attention (DSA / MLA, DeepSeek-V4.1's
  window attention with its sink). The fork's sparse_sdpa options are precision only (its bf16 running state /
  approximate exp / bf16 scores): per-step checks or a measured end-to-end gap (rule 7).
- Precision only, likewise: `ring_mla` at fp32 DEST when bf16 DEST accumulation is too coarse (sharp softmax over
  many K tiles).
For any other SDPA call the fork is the source op; prefer `ttnn.transformer.*`.

## Options
- `max_grid` (ttnn.CoreCoord, default None) on `ttnn.bringup.sparse_sdpa`: the op's cores inside that rectangle of the grid (FIX.15, a Galaxy's 110-core cap: 11 x 10); default the whole grid.
Fork additions only; every other argument as the stock op (see `sdpa_nanobind.cpp` for the full signatures).

Realistic (use by default):
- `scaled_dot_product_attention(q, k, v, ...)`, `chunked_scaled_dot_product_attention(q, k, v, page_table, ...)`: V
  narrower than K is the switch (fork addition, no argument). Refused: V not a multiple of 32, V > K's head dim, with
  `paged_cache_geometry`, in windowed mode (`cu_window_seqlens`).
- `sparse_sdpa(q, kv, indices, v_dim, *, kv_format, scale=None, k_chunk_size=128, compute_kernel_config=None,
  cache_batch_idx=None, block_cyclic_sp_axis=None, block_cyclic_chunk_local=None, block_cyclic_cache_tp_sharded=False,
  attention_sink=None, ...)`: the stock arguments; with the precision options off the program is the source's.
- `ring_mla(..., kv_actual_isl=0)` on a non-chunked causal input accepted (fork bug fix, no new argument).
- `SparseKVFormat`: `BF16`, `FP8_E4M3`, `SCALED_FP8` (the source enum, re-bound under `ttnn.bringup`; pass this one to
  `ttnn.bringup.sparse_sdpa`), plus the fork's `ROW_BFP4`.
- `sparse_sdpa(q, kv, indices, v_dim, kv_format=SparseKVFormat.ROW_BFP4, kv_window=w, ...)` (fork, opt-in): kv
  `[B, 1, T, 320]` uint8 row records, kv_window `[1, 1, Tw, 512]` bf16 DRAM interleaved ROW_MAJOR; the index row is
  `[window k_chunk | top-k]`, each part with its own sentinel tail (no compaction). Needs K_DIM 512, a bf16 q,
  TOPK > k_chunk_size, no block_cyclic. kv_window is given exactly with ROW_BFP4. kv_window may instead be
  `[1, 1, Tw, 576]` uint8 bfloat8_b records (`ttnn.bringup.ckv_bfp4(dtype=ttnn.bfloat8_b)`), read into bfloat8_b tiles.
- `sparse_sdpa(q, kv, indices, v_dim, kv_format=SparseKVFormat.ROW_BFP8, ...)` (fork, opt-in): kv `[1, 1, T, 576]` uint8
  bfloat8_b records, every index a window row (TOPK == k_chunk_size), no kv_window, no cache_batch_idx.

Precision only: per-step checks or a measured end-to-end gap (rule 7):
- `sparse_sdpa(...)` plus:
  - `high_precision=False` (fork, opt-in): Float32 running output / row-sum / 1/sum CBs, exact exp. Needs
    `fp32_dest_acc_en`, bf16 q, `kv_format=SparseKVFormat.BF16`. Takes `attention_sink`.
  - `fp32_output=False` (fork, opt-in): Float32 output. Needs `high_precision`.
  - `fp32_scores=False` (fork, opt-in): Float32 scores / probabilities, TF32 nearest rounding in DEST, accepts a float32 q.
    Needs `high_precision` and `fp32_output` (unless `bf16_probs`).
  - `bf16_probs=False` (fork, opt-in): Float32 running max, probabilities rounded to bf16 before P@V; `fp32_output` may
    then be False. Needs `fp32_scores` and a bf16 q.
- `ring_mla(..., compute_kernel_config)`: `fp32_dest_acc_en=True` accepted (fork, no new argument; the source refuses it).

## Measured
- Narrow V, MiMo full-attention chunk (Q 16x5120x192, paged cache, 51200-token prefix, q512/k128, HiFi2, approx exp,
  bf16 DEST, 1 chip): 27.15 ms (V 192) -> 22.49 ms (V 128), -17%; chunk 0 (causal S 5120) 1.505 -> 1.256 ms
  (CHANGELOG.md, "Non-MLA V head dim narrower").
- Narrow V model cases, 4 chips: chunked paged prefix 51200: pcc 0.99916-0.99921, rel L2 0.090-0.092; sliding 128 +
  sink: pcc 0.99974, rel L2 0.0232-0.0233 (`tests/cases.py`).
- sparse_sdpa `high_precision`, GLM DSA geometry (64 heads, K 512, 2176 ids, 17 k-chunks): rel L2 vs float32 0.0066
  (source) -> 0.0030; model attention output rel 0.0076 -> 0.0044 (CHANGELOG.md). GLM model case (2x2, block-cyclic
  cache): pcc 0.999984, rel L2 0.0057-0.0058 (`tests/cases.py`).
- sparse_sdpa `high_precision` + sink, DeepSeek-V4.1 window: component pcc 0.99975, rel vs CPU 0.0212 (limit 0.0292);
  op rel L2 0.0023-0.0068 (CHANGELOG.md).
- `fp32_output`: op rel vs float32 0.0022-0.0065 (bf16 output 0.0028-0.0068) (CHANGELOG.md).
- `fp32_scores`: op rel vs float32 0.0026 / 0.0068 / 0.0063 -> 0.0009 / 0.0022 / 0.0017; DS41 attention 26.5 -> 28.6 ms
  per layer (2x2, chunk 2048) (CHANGELOG.md).
- `bf16_probs`: rel L2 vs DeepSeek's sparse_attn in float32: 1 chunk 0.0020 (bf16 out) / 0.0010 (fp32 out), 4 chunks
  0.0027 / 0.0021; DS41 attention 28.8 -> 26.1 ms per layer (2x2, chunk 2048) (CHANGELOG.md).
- `ring_mla` fp32 DEST, Xing geometry (16 heads, K 576, V 512): worst row rel L2 0.054 (bf16 DEST) -> 0.035 (CHANGELOG.md).
  Xing model case (4x2, HiFi2 + fp32 DEST): pcc 0.99943, rel L2 0.0339-0.0341 (`tests/cases.py`). HiFi2 vs HiFi4:
  17.9 vs 25.9 ms (`models/demos/xing40_a4b_d_p/bringup/known_issues.md`, XING-3).
- `ring_mla` single-chunk `kv_actual_isl=0`: rel 0.018 / worst row 0.031 at fp32 DEST, 0.023 / 0.058 at bf16 DEST
  (CHANGELOG.md).
- `ROW_BFP4`, one p150b (H 64, 640 indices): 160 q 341 -> 353 us (+3.5 %), 512 q 1182 -> 1245 us (+5 %) vs BF16
  (CHANGELOG.md). In deepseek_v41_flash_d_p on the Galaxy 8x4 (FIX.8, chunk 5120, 64 heads x 160 queries a chip, 38
  compressed layers, real-time profiler, same session): 0.351 -> 0.360 ms a call at position 0 (+2.6 %), 0.432 ->
  0.451 ms at 512k (+4.2 %); the bf16 start < 128 compaction (ttnn.gather, 0.18 ms a chunk) is gone. Table DRAM a chip
  at 1,044,480 tokens: 2570.5 -> 817.4 MiB (BREADCRUMBS FIX.8).
- Box: not stated per number in the CHANGELOG except the Xing tests (4x2 LoudBox, 8x p150b); `tests/source.yaml`
  records 4x Blackhole p150b.

## Limits
- Narrow V: not with `paged_cache_geometry` or windowed mode; V % 32 == 0 and V <= K.
- `high_precision` needs fp32 DEST, bf16 q, `kv_format` BF16 (no fp8 kv). A residual ~0.1% output shrink (coefficient
  0.9989) under `high_precision` is attributed to TF32 reads in normalize, untested (CHANGELOG.md).
- `bf16_probs`: score - max on the FPU is TF32-precise; the flash correction exp(m_prev - m_cur) stays bf16 (CHANGELOG.md).
- `fp32_scores` without `bf16_probs`: TF32 probabilities reach P@V with ~9 bits (CHANGELOG.md; DS41-31 in
  `models/demos/deepseek_v41_flash_d_p/bringup/known_issues.md`).
- Only `scaled_dot_product_attention`, `chunked_scaled_dot_product_attention` and `sparse_sdpa` carry the source op's
  tests (`tests/source.yaml`); the joint / ring / MLA / msa / distributed variants are the source code with no carried
  source tests. The source ring tests open FABRIC_1D / ring / torus meshes and are not run here.
- The 1x4 / 2x2 model cases time out opening their mesh on the 8-chip 4x2 LoudBox (fabric router sync, environment);
  the Xing case (8 devices) is skipped on 4-chip boxes (CHANGELOG.md).
- No model case for deepseek_v41_flash_d_p's sparse_sdpa calls in `tests/cases.py`.

## How to call
```python
o = ttnn.bringup.sparse_sdpa(
    q_rm, kv, idx, 512,
    kv_format=ttnn.bringup.SparseKVFormat.BF16, scale=scale, k_chunk_size=128,
    compute_kernel_config=ckc,  # fp32_dest_acc_en=True
    high_precision=True,
)  # models/demos/glm53_flash_d_p/tt/mla_attention.py
```
Narrow V: `ttnn.bringup.chunked_scaled_dot_product_attention(q, k, v_128, page_table, chunk_start_idx=start, ...)`
(`models/demos/mimo_v2_6_d_p/tt/attention.py`).
Tests: `tests/test_sdpa.py` (model cases, `tests/cases.py`), `tests/unit/test_sdpa_narrow_v.py`,
`tests/unit/test_sparse_sdpa_high_precision.py`, `tests/unit/test_ring_mla_fp32_dest.py`,
`tests/unit/test_ring_mla_single_chunk.py`, source tests via `tests/source.yaml`.

## Used by
- mimo_v2_6_d_p, mimo_v2_6_d_p_2x2: `tt/attention.py`, narrow V 128 (`MIMO_V_PAD=1` keeps `ttnn.transformer` with
  padded V); O.1.
- glm53_flash_d_p: `tt/mla_attention.py`, sparse_sdpa `high_precision` (`GLM_MLA_SDPA=source` keeps the source op);
  C.dsa_moe.attention, O.1.
- xing40_a4b_d_p: `tt/attention.py`, `ring_mla` at HiFi2 + fp32 DEST (`XING_MLA_SDPA_FIDELITY`, default HiFi2; `XING_MLA_SDPA=source` keeps the source op at bf16 DEST);
  C.dense.attention, X.3, O.1.
- deepseek_v41_flash_d_p: sparse_sdpa `ROW_BFP4` + `kv_window` (`tt/attention.py` `_sdpa`, the compressed layers'
  attention; `CKV_FORMAT=bf16` keeps stock ttnn.transformer.sparse_sdpa); FIX.8 (2026-10-28, 2x2 and Galaxy 8x4).
  Earlier: none since FIX.8 (2026-10-21; the model runs stock ttnn.transformer.sparse_sdpa, the
  settings `DS41_WINDOW_SDPA*` are deleted). Before: `tt/attention.py`, `tt/sparse_attention.py`, sparse_sdpa
  `high_precision` + sink + `fp32_scores` + `bf16_probs`; C.swa.attention, S.swa.06, S.swa.11, FIX.2. The options
  `fp32_output` / `fp32_scores` / `bf16_probs` have no user now.

## History
CHANGELOG.md
