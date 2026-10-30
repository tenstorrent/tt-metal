# ttnn.bringup.rotary_embedding_indexed

## What it does
Fork of `ttnn.experimental.deepseek_prefill.rotary_embedding_indexed`. It adds one option, `uniform=True`: ONE
program for the whole mesh range instead of one per mesh coordinate. The stock op bakes each chip's `my_sp_coord` as a
compile-time arg, so a 32-chip mesh gets 32 programs per call and every cache hit re-applies the run args to all 32.
With per-chip pre-sliced cos / sin (`kv_actual_global == 0`, no `seq_subshard_axis`), the reader's cos / sin row offset is
0 on every chip, so the programs are identical and one is enough. The kernels, args and values are the same as the
stock op's. Everything else is unchanged from the stock op.

## Consider it instead of
`ttnn.experimental.deepseek_prefill.rotary_embedding_indexed`, when each chip passes its own cos / sin rows (pre-sliced
per chip) with `kv_actual_global` 0. That is the natural-block layout, where every chip reads from row 0.

## Options
- `uniform` (bool, scalar form only, default `False` = the stock op's per-coordinate programs, unchanged): one program
  per mesh coordinate range. It requires `kv_actual_global == 0` and no `seq_subshard_axis`, checked on every call
  (miss and hit). It is in the program hash.

## Measured
- Galaxy 8x4 (32 BH chips), `tests/unit/test_uniform.py::test_uniform_host_time`, warm host time a call (dispatch
  only), q-with-prefix shape [1, 64, 160, 64] + prefix [1, 64, 160, 448] bf16: stock 101.0 us, uniform 52.0 us.
- DeepSeek-V4.1 all-device model, 136 calls a chunk, profile rung's last chunk [51200, 56320), same process,
  alternating (`deepseek_v41_flash_d_p/tt/perf_rope_uniform.py`): warm wall median 438.6 ms (stock) -> 399.7 ms
  (uniform), n 6 each. Final hidden state bit-identical.
- Output byte-identical to the stock op on every chip: 6 model shapes, a miss and a hit (`test_uniform_matches_stock`).

## Limits
- `uniform=True` is refused with a metadata tensor (the tensor form has no `uniform` argument), with
  `kv_actual_global != 0`, and with `seq_subshard_axis`. Those need per-chip offsets: use the default.
- Tested on 8x4 only (test_uniform has 2x4 / 2x2 cases for those boxes). Source tests carried: the 8x4 and
  single-device cases (18, `tests/source.yaml`). The 2x2 / 2x4 upstream cases are not carried from the Galaxy.

## How to call
```python
out = ttnn.bringup.rotary_embedding_indexed(
    x, cos_rows, sin_rows, trans_mat, 0, 0,  # kv_actual_global 0, cluster_axis 0
    memory_config=ttnn.DRAM_MEMORY_CONFIG, rotary_dim=64, rotary_offset=x.shape[-1] - 64,
    concat_prefix=None, uniform=True)
```
Tests: `tests/unit/test_uniform.py` (uniform vs stock, refusals, host time), `tests/source.yaml` (the source op's tests).

## Used by
deepseek_v41_flash_d_p (P.5, `tt/rope.py:rope_stock`, setting ROPE_UNIFORM_PROGRAM)

## History
CHANGELOG.md
