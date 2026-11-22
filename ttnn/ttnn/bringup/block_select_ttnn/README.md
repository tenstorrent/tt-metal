# ttnn.bringup.block_gather / block_keys / block_max_bf16 (block_select_ttnn)

## What it does
The data-movement steps of an EXACT block pre-selected top-k (deepseek_v41_flash_d_p PB.BM, INDEXER_BLOCK_SELECT):
per row, the top-K blocks of B adjacent scores by block max, then the top-k keys over only those blocks' scores. The
two top-k's are `ttnn.bringup.topk_large_indices`; these ops do the rest:

- `block_max_bf16(block_max, *, factor=1, n=None, n_tensor=None, output_tensor=None, memory_config=None,
  max_grid=None)`: [.., R, NI] fp32 ROW_MAJOR (bf16 values widened, e.g. `ring_indexer_score_dsa`'s `block_max_out`)
  -> [.., R, NI / factor] bf16 ROW_MAJOR, column b = max of input columns [b factor, (b + 1) factor), exact. Only the
  first `n` output columns (rounded up to 8) are written: `n` is a runtime value (a prepared-program value slot) or
  `n_tensor` ([1] uint32 a device, read on the device). Both RISCs of every core, double-buffered reads.
- `block_gather(scores, ids, *, block, output_tensor=None, memory_config=None, max_grid=None)`: scores [.., R, W]
  bf16 ROW_MAJOR + block ids [.., R, K] uint32 ROW_MAJOR -> [.., R, K block] bf16 ROW_MAJOR, row r =
  scores[r, ids[r, j] block + o] at j block + o; an id >= W / block (0xFFFFFFFF) gives block x -inf. One 64-byte DRAM
  read a block (Blackhole's DRAM read alignment; block 32 reads straight into the output row); work units of K / 4
  blocks of a row (ids at 64-byte aligned offsets) over both RISCs of every core.
- `block_keys(pos, ids, *, block, width, output_tensor=None, memory_config=None, max_grid=None)`: pos [.., R, KP]
  uint32 (positions in block_gather's row, e.g. its top-k) + the ids it read -> key ids ids[pos / block] block +
  pos % block (0xFFFFFFFF where pos is none or its block was).

Every program is built once per (shapes, memory configs, block, grid) and dispatched prepared (ttnn.bringup
.prepare_program / run_prepared: addresses patched; no value enters a cache key). Rows split over the cores by common
runtime args (no per-core args).

## Consider it instead of
- `ttnn.gather` on long ROW_MAJOR rows: its kernel keeps each core's whole input row in L1 (a 1M-key bf16 row is
  2 MiB) and reads it once a core.
- `ttnn.typecast` of a ROW_MAJOR fp32 tensor (on a capped grid: tilize + typecast + untilize) and strided
  `ttnn.slice` + `ttnn.maximum` folds of block maxes.

## Options
`max_grid`: the cores' rectangle (0, 0) .. (max_grid - 1) (a gather-hide region's compute sub-device, the Galaxy's
matmul grid); default the device's grid. Precision only: none (exact data selection).

## Measured
Galaxy 8x4, one chip's shape, 256 rows (DeepSeek-V4.1 chunk 8192), random bf16 scores with causal -inf tails,
pipelined (8 calls between syncs), `tests/perf/test_block_select_probe.py` of the model:
| T keys | plain top-k 512 | block 8 chain | block 32 chain | block_max_bf16 (x4) | block_gather (32) |
|---|---|---|---|---|---|
| 69632 | 0.47 ms | 1.47 | 1.33 | 0.13 | 0.30 |
| 512000 | 3.21 ms | 1.44 | 1.00 | 0.40 | 0.34 |
Sets of the block chain vs the plain top-k: equal up to ties at the 512th value on every row (0 mismatches).
(block_gather was 0.41 / 0.44 ms with one RISC a core and whole rows a unit.) In the model (8x4 ladder, single 8192 at
1024000, 3 index layers): the plain top-k 9.40 ms -> fold 1.17 + top-k's 0.74 + gather 1.11 (one-RISC gather).

## Limits
- ROW_MAJOR DRAM interleaved tensors, one page a row; block a power of two <= 32 dividing W (a multiple of 32).
- block_max_bf16: the inputs must be bf16 values widened to fp32 (it keeps the high halves); NI a multiple of 8 x
  factor; factor <= 16.
- Data-movement RISC-V work: block_gather ~K reads a row, block_max_bf16 one compare a value.

## How to call
```python
bm = ttnn.bringup.block_max_bf16(block_max_fp32, factor=4, n=kv_len // 32)
ids = ttnn.bringup.topk_large_indices(bm, k=512, valid_length=kv_len // 32)
g = ttnn.bringup.block_gather(scores, ids, block=32)
keys = ttnn.bringup.block_keys(ttnn.bringup.topk_large_indices(g, k=512), ids, block=32, width=scores.shape[-1])
```
Tests: `tests/unit/test_block_select.py` (`check_block_select`, also run by the model's probe on the resident mesh).

## Used by
deepseek_v41_flash_d_p PB.BM: `tt/indexer.py` TtIndexer.block_topk (INDEXER_BLOCK_SELECT), single and glued packed
paths.

## History
CHANGELOG.md
