# ttnn.bringup.topk_large_indices

## What it does
Fork of `ttnn.experimental.topk_large_indices` (`ttnn/cpp/ttnn/operations/experimental/topk_large_indices` @
`df92663a019`): the k largest values' indices per row of a bf16 ROW_MAJOR tensor, uint32, descending (Blackhole only).
It carries upstream's split LLK (#62035 / #62488 / #62411, not yet on this branch's base): the row split across MATH
and PACK and across core pairs, fused macros at K' 1024 / 2048. And it adds the K' 512 port of the split segmented
body: k 512 rows longer than one fused segment (32 x 512) took the Classic body; they now take the fused segmented
body split across MATH and PACK, as K' 2048 does. The four top-k LLK headers are copied into `device/kernels/llk/`
(verbatim, includes pointed at the copies, the K' 512 change in the copy only); the stock op and the stock LLK are
untouched.

## Consider it instead of
`ttnn.experimental.topk_large_indices` on Blackhole, any k: the same results (the k largest values, uint32 ids,
-inf -> 0xFFFFFFFF), 1.3-2.4x faster at DeepSeek-V4.1's index top-k shapes.

## Options
As the source (`topk_large_indices_nanobind.cpp`): `topk_large_indices(input, *, k, valid_length=None,
valid_length_tensor=None, valid_length_offset=0, subdevice_id=None, sub_core_grid=None, valid_end_tensor=None,
dram_read_noc=None)`. `dram_read_noc` is upstream's (#62411). No fork-only option: the fork has no other user, so the
changes are its behaviour.

Precision only: none (exact selection).

## Measured
One p150b, device kernel time, stock -> fork (`tests/ttnn/unit_tests/operations/ds41_topk/test_profile_topk.py`):
| call (rows x width, k) | stock us | fork us |
|---|---|---|
| 2x2 index top-k 512 x 27648, 512 | 596 | 337 |
| 2x2 index top-k 512 x 55296, 512 | 1190 | 675 |
| 8x4 index top-k 160 x 262144, 512 | 2256 | 954 |
| 2x2 candidate blocks 512 x 6912, 2048 | 182 | 138 |
| 8x4 candidate blocks 160 x 32768, 2048 | 279 | 166 |
| 512 x 16384, 512 (one segment) | 256 | 196 |
DeepSeek-V4.1 scaled 2x2, 8 layers, chunk 53248 -> 55296 (`test_profile.py`, INDEX_TOPK_OP stock -> fork): device
56.12 -> 54.43 ms a chunk (-3.0 %), the indexers 11.3 -> 9.7 ms; wall unchanged (stock 84.7 / 78.8, fork 82.3 / 79.2 ms,
host-bound). Accuracy: the 4 indexer component tests and the 8 whole-block swap tests identical to the stock op, line
for line; ladder rung last: layer errors within 0.001 of the stock op's (tied bf16 scores pick other ids), s4096 and
s55296 (27 chunks) pass.

## Limits
- As the source: Blackhole only, bf16 ROW_MAJOR input, k 16-2048 (a multiple of 16).
- Source tests (`tests/source.yaml`, the stock nightly file, 154): 150 pass on the fork. The 4 others are upstream's
  own changes in the carried commits: 2 production perf bands (the op is faster; upstream re-baselined them) and 2
  program-cache counts (one program per row-split class now). Upstream's updated file
  (`tests/unit/test_upstream_split.py`, 194) passes.

## How to call
```python
ids = ttnn.bringup.topk_large_indices(scores, k=512, valid_length=kv_len)  # [1, 1, S, T] bf16 RM -> [1, 1, S, 512] uint32
```
Tests: `tests/unit/` (one card), `tests/source.yaml` (`fork_source --fork topk_large_indices`), the model cases
`tests/test_topk_large_indices.py` (`cases.py`, random inputs, determinism).

## Used by
deepseek_v41_flash_d_p (DS41-151): the indexers' top-k (`tt/indexer.py`, setting INDEX_TOPK_OP).

## History
CHANGELOG.md
