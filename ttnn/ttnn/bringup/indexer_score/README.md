# ttnn.bringup.indexer_score_dsa, ttnn.bringup.ring_indexer_score_dsa, ttnn.bringup.indexer_score_msa, ttnn.bringup.IndexerScoreProgramConfig

## What it does
Fork of `ttnn.experimental.indexer_score_dsa` / `ring_indexer_score_dsa` / `indexer_score_msa` and
`ttnn.IndexerScoreProgramConfig` (`ttnn/cpp/ttnn/operations/experimental/indexer_score`), synced to source @
`107f838f379`. The fork adds:
- fp32 DEST for DSA scoring: `compute_kernel_config.fp32_dest_acc_en=True` accepted by `indexer_score_dsa` /
  `ring_indexer_score_dsa` (num_groups 1, block_size 0, learned gates), with the mask srcA reconfig it needs. MSA and
  block pooling still refuse it.
- `key_compression_ratio=2` (source: 1 or 4).
- `ring_indexer_score_dsa(natural_query_blocks=True)`: queries are natural contiguous blocks at any tile-aligned
  start while the keys stay in the block-cyclic cache layout (linear causal diagonal); `natural_query_rows` gives the
  queries' own tokens per SP rank when their chunk is not the one that laid out the keys.
- `IndexerScoreProgramConfig.__repr__` prints its values (stable fork-call capture signatures).
- `ring_indexer_score_dsa` block epilogue (`block_keys`, `block_bias`, `block_max_out`, `block_max_pin`): the writer
  masks blocks of adjacent keys with a 0 / -inf bias and / or writes each block's max (+inf at each query's own block)
  from the untilized scores, before they go out.
Everything else as `ttnn.experimental.<same name>` / `ttnn.IndexerScoreProgramConfig`.

## Consider it instead of
- `ttnn.experimental.ring_indexer_score_dsa` / `indexer_score_dsa` when the index keys pool 2 tokens
  (`key_compression_ratio=2`, DeepSeek-V4.1), or when queries are natural contiguous blocks over a block-cyclic key cache.
- a `ttnn.all_gather` of the keys + `ttnn.experimental.indexer_score_dsa`: the ring op (source and fork) fuses the SP
  gather with the scoring.
- Precision only (rule 7), not a reason to fork: fp32 DEST when the bf16-DEST logits are too coarse for the top-k
  selection (the source scores ~1.7% low per row and fails a 0.99 top-k overlap gate on Hy4's indexer; CHANGELOG.md),
  for per-step checks or a measured end-to-end gap.
- `ttnn.slice` (strided) + `ttnn.maximum` block max-pools of the scores, and `ttnn.repeat_interleave` + `ttnn.add` of a
  per-block bias (DeepSeek-V4.1's candidate pool and re-index restriction): the block epilogue.
With fp32 DEST off, ratio 1 / 4, `natural_query_blocks=False` and no block epilogue the fork is the source program;
prefer the source op then.

## Options
- `max_grid_x` / `max_grid_y` (int, default 0 = no limit) in `ttnn.bringup.IndexerScoreProgramConfig` (indexer_score_dsa, ring_indexer_score_dsa): the op's cores inside that rectangle of the grid (FIX.15, a Galaxy's 110-core cap: 11 x 10); default the whole grid.
Fork additions marked; the rest as the source (`indexer_score_nanobind.cpp`).

Realistic (use by default):
- q / k (k_local) bf16, bfloat8_b or (fork) bfloat4_b tiled; block-float pairs default to LoFi (pass HiFi2 when the gate
  step's relu(q.k) needs more than 4 bits).
- `IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=32, head_group_size=1)`; value `__repr__` (fork).
- `indexer_score_dsa(q, k, weights, *, chunk_start_idx=None, key_compression_ratio=1, program_config=IndexerScoreProgramConfig(),
  compute_kernel_config=None, cache_batch_idx=None, kv_len=None, seq_shard_axes=None, block_cyclic_sp_axis=None,
  block_cyclic_chunk_local=None, block_cyclic_cache_tp_sharded=False)`.
- `ring_indexer_score_dsa(q, k, weights, k_local, ag_multi_device_global_semaphore, *, cluster_axis, topology,
  num_links=1, ag_sub_device_id=None, chunk_start_idx=None, key_compression_ratio=1, program_config=..., compute_kernel_config=None,
  cache_batch_idx=None, kv_len=None, seq_subshard_axis=None, block_cyclic_sp_axis=None, block_cyclic_chunk_local=None,
  block_cyclic_cache_tp_sharded=False, chunk_start_idx_tensor=None, valid_end_tensor=None, cache_batch_idx_tensor=None,
  index_cache_num_layers=1, index_cache_layer_idx=0, natural_query_blocks=False)`.
- `indexer_score_msa(q, k, *, num_groups, chunk_start_idx=None, scale=1.0, block_size=0, program_config=..., compute_kernel_config=None,
  cache_batch_idx=None, kv_len=None, seq_shard_axes=None, block_cyclic_sp_axis=None, block_cyclic_chunk_local=None,
  block_cyclic_cache_tp_sharded=False)`: unchanged by the fork.
- `key_compression_ratio` (fork adds 2): 1 (default), 2, 4.
- `natural_query_blocks=False` (fork, opt-in, ring only): needs `block_cyclic_sp_axis` / `block_cyclic_chunk_local`,
  an axis ring (`cluster_axis` set) and the host-scalar `chunk_start_idx` (refused with `chunk_start_idx_tensor`).
- `natural_query_rows=0` (fork, opt-in, with `natural_query_blocks`; 0 = `block_cyclic_chunk_local`): the query tokens
  per SP rank, tile-aligned, >= Sq. Device (d, c) holds positions chunk_start_idx + d natural_query_rows + c Sq +
  [0, Sq); the keys keep `block_cyclic_chunk_local`'s layout. For a small chunk scoring keys a larger chunk wrote
  (DeepSeek-V4.1's decoder tail: 1024-token queries, keys of 5120-token chunks).

- Block epilogue (fork, opt-in, ring only, batch 1): `block_keys=0` (> 0, a divisor of 32, exactly when a tensor
  below is given: blocks of `block_keys` adjacent logical keys); `block_bias=None` ([1, 1, Sq, NB] bf16 TILE DRAM,
  NB * block_keys >= T, NB % 32 == 0: nonzero (-inf) -> every score of the block written -inf, 0 keeps it);
  `block_max_out=None` ([1, 1, Sq, NBm] fp32 ROW_MAJOR DRAM, NBm * block_keys >= T, NBm % 4 == 0, block_keys <= 8;
  written in place: each block's max of the written bf16 scores as fp32, blocks of the first kv_len keys only);
  `block_max_pin=False` (needs block_max_out: +inf at each query's own block, its newest visible key
  (pos + 1) // ratio - 1). Exact (data selection on the op's own bf16 scores).

Precision only: per-step checks or a measured end-to-end gap (rule 7):
- `compute_kernel_config.fp32_dest_acc_en` (fork, opt-in, default False): fp32 DEST for DSA scoring. Use
  `k_chunk_size=32`: above 32 the head reduction uses a blocked multiply that runs at LoFi-like precision whatever the
  fidelity (CHANGELOG.md).

## Measured
- Hy4 layer-0 indexer golden (32 heads x 128, top-2048 of 4096 keys): source logits rel 0.024, top-k overlap 0.985;
  fork fp32 DEST, k_chunk 32: rel 0.0036, overlap 0.99708 (CHANGELOG.md).
- Random inputs, 32 heads: per-column path fp32 DEST rel 0.0019 vs bf16 DEST 0.0118; blocked path 0.029 vs 0.023
  (CHANGELOG.md; `tests/unit/test_fp32_dest.py`).
- `natural_query_blocks` + ratio 2, 2x2 (S 2048, W 1024, Sq 512): visible columns PCC 0.9999985, rel 0.00186 vs fp32
  torch; DeepSeek-V4.1 component gate C.enc_source.indexer err 0.0174 <= 0.0329 (CHANGELOG.md).
- Model cases, 2x2, 4 chips: hy4 ring SP2 x TP2 and glm53 full-mesh ring ratio 4: pcc 0.9999985, rel 0.00186
  (`tests/cases.py`).
- Block epilogue: exact vs the plain call's scores (`tests/unit/test_block_epilogue.py`). Speed, 2x2 p150, Sq 512 x 32
  heads x 128, T 55296, HiFi2 + fp32 DEST, q 64 x k 32 (probe in CHANGELOG.md): plain 1.68 ms, + block max and pin
  1.96 ms, + bias 1.69 ms; deepseek_v41_flash_d_p P.16: indexer section 25.15 -> 12.94 ms a chunk.
- Speed: not measured for the plain op.
- `natural_query_lane_lo / _hi` (ring op, fork, opt-in, default 0 / 0; PB.3A): with natural_query_blocks, the devices
  whose natural chunk offset lies in [lo, hi) score at chunk_start_idx + (offset - lo); the rest keep today's
  offsets. Runtime values (no new program).
- `lane_rows`, `lane_starts`, `lane_kv_lens`, `lane_buffer`, `k_extra`, `k_local_extra` (ring op, fork, opt-in,
  default 0 / empty; PB.3A): ONE fused ring over the lanes of a packed chunk: each device its lane's start, kv_len and
  key buffer; the all-gather moves every lane buffer's k_local (up to 8 inputs since PB.7B, no partial readiness); kv_len is the
  largest lane's. Starts / kv_lens are runtime values; lane_buffer is hashed (one program a lane -> buffer map).
- `lane_own_extent` (ring op, fork, opt-in, default False; PB.3M): with `lane_rows`, each lane buffer is gathered only
  to its own lanes' largest kv_len instead of the call's kv_len (a mixed packed chunk: 61k / 139k / 308k / 547k moved
  4 x the 547k lane's keys). Same scores. Hashed; the extents are runtime values.
- `lane_ranges` (ring op, fork, opt-in, default empty; PB.7C): ONE fused ring where a device holds query rows of
  several segments (a glued packed chunk): per device (chip-linear r TP + c) 41 words [n, (row_lo, row_hi, buffer,
  start, kv_len) x 8]; each group of q rows (whole q chunks: q_chunk_size 32 for 32-row ranges) looks up its range
  (`kernels/indexer_score_ranges.hpp`) for its key buffers, kv_len and causal geometry (natural blocks, start + row
  offset; the block epilogue's chunk start too). Every value, the range -> buffer map included, is a runtime value:
  the reader binds every buffer's gathered and local address in its common args, the tables are the last common args
  of the reader / writer / compute and re-patched every call; only the presence is hashed. Each buffer is gathered to
  the largest kv_len of the ranges reading it (one slab when none reads it). The key multicast down a core column is
  off in this mode (a column's groups may read different buffers): every core reads its own keys and gates on the
  ring itself. Rows no range covers are not written.

- `keys_gathered` (ring op, fork, opt-in, default False; PB.KR, needs `lane_ranges`): `k` already holds every SP
  shard of the keys each device reads (the caller moved them, e.g. each segment's keys to its own chips only): no
  all-gather kernels, no arrival wait, the own shard read from `k` too, and the all-gather's worker column computes
  (11 x 10 on a capped Galaxy grid instead of 10 x 10). Same scores over [0, kv_len). Hashed. Measured (8x4, 4 x 2048
  rows at 1024000, kv_len 516096): 9.0-9.8 ms a call vs the ranges ring's 10.9-11.2.

## Limits
- fp32 DEST: DSA only (not MSA, not block pooling); `dst_full_sync_en` must stay False.
- `natural_query_blocks`: axis ring only, block-cyclic layout required, no metadata (`chunk_start_idx_tensor`) path.
  `natural_query_rows`: the same; only the query geometry moves (the key stripes and the invP remap are the cache's).
- Block epilogue: `ring_indexer_score_dsa` only, batch 1, one head-summed plane; the bias values are a mask (any
  nonzero bias masks the block; no general additive bias); block_max_out needs block_keys <= 8 (16-byte aligned fp32
  runs per score tile); the writer does the block max on its RISC-V core (scalar, about 1 compare per score).
- Source multi-device tests are not carried (they open FABRIC_1D / torus meshes); the ring op is covered by the
  fork's own 2x2 FABRIC_2D tests. `tests/unit/test_key_stride_ring.py` has a 2x4 case skipped on a 2x2 box.
- Known source-test divergence: `test_indexer_score_rejects_fp32_dest_acc` fails on the fork by design (`tests/source.yaml`).
- `lane_ranges`: fused ring + natural_query_blocks, no lane_rows / natural_query_lane / cache slot / metadata
  tensors; every buffer one capacity (k's / k_local's shape, as the lanes ring: DS41-328); ranges whole q chunks; key
  multicast off (each core reads its keys: ~8 x the K DRAM reads of a 8-row column at 8x4, small next to the gather).

## How to call
```python
logits = ttnn.bringup.ring_indexer_score_dsa(
    q, kbuf, w, k_local, ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=0),
    cluster_axis=0, topology=ttnn.Topology.Linear, num_links=2,
    chunk_start_idx=start, key_compression_ratio=2,
    program_config=ttnn.bringup.IndexerScoreProgramConfig(q_chunk_size=64, k_chunk_size=32, head_group_size=0),
    compute_kernel_config=ckc,  # HiFi4, fp32_dest_acc_en=True
    kv_len=kv_len, seq_subshard_axis=1, block_cyclic_sp_axis=0, block_cyclic_chunk_local=w_rows,
    natural_query_blocks=True,
)  # models/demos/deepseek_v41_flash_d_p/tt/indexer.py
```
Tests: `tests/test_indexer_score.py` (model cases, `tests/cases.py`), `tests/unit/test_fp32_dest.py`,
`tests/unit/test_key_stride.py`, `tests/unit/test_key_stride_ring.py`, `tests/unit/test_natural_query_blocks.py`,
`tests/unit/test_natural_query_rows.py` (BRINGUP_TEST_MESH), source tests via `tests/source.yaml`.

## Used by
- hy4_preview_d_p: `tt/indexer.py`, ring op, HiFi4 + fp32 DEST, q 64 x k 32; C.dense_full.indexer, O.1.
- glm53_flash_d_p: `tt/indexer.py`, ring op over the full-mesh snake, key_compression_ratio 4, and `indexer_score_dsa`
  (`GLM_INDEXER_SCORE`, default `bringup`), HiFi4 + fp32 DEST, q 64 x k 32; O.1.
- deepseek_v41_flash_d_p: `tt/indexer.py`, ring op over axis 0 + TP query sub-shard, key_compression_ratio 2 / 1,
  `natural_query_blocks`; C.enc_source.indexer. P.16 (INDEXER_GLUE=on): layer 20 `block_keys=8, block_max_out,
  block_max_pin=True` (the candidate pool), layers 24-36 `block_keys=8, block_bias` (the candidate restriction).
  P.31 (DECODER_SKIP=replay128): the decoder tail's layers 20-39 at a 1024-token chunk score the 5120-token chunks'
  keys with `natural_query_rows`.
  P.40: INDEX_K_FORMAT=bfp8 scores bfloat8_b keys (k_local and the ring buffer) against the bfloat4_b query.
  PB.3A (PACKED_ATTN_ONECALL=on, `tt/packed_onecall.py`): `natural_query_lane` (a segment's ring call) and the lanes
  ring (one call over a packed chunk); PB.3M (PACKED_LANE_BALANCE=on): `lane_own_extent`. PB.7C
  (PACKED_GLUED_DYNAMIC=on, `tt/packed_glued.GluedCall._index_dynamic`): `lane_ranges` at q_chunk 32, one call a
  layer over a glued chunk's segments (6 key buffers: the unused ones gather one slab).
  PB.KR (PACKED_KEYS_OWNED=on, `tt/owned_keys.py`): `keys_gathered` with every range on buffer 0, the gathered buffer
  each chip's own segment's keys only (sent there by tt/owned_gather.py).

## History
CHANGELOG.md
