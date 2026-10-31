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
- `max_grid` (ttnn.CoreCoord, default None) on `ttnn.bringup.rotary_embedding_indexed`: the op's cores inside that rectangle of the grid (FIX.15, a Galaxy's 110-core cap: 11 x 10); default the whole grid.
- `uniform` (bool, scalar form only, default `False` = the stock op's per-coordinate programs, unchanged): one program
  per mesh coordinate range. It requires `kv_actual_global == 0` and no `seq_subshard_axis`, checked on every call
  (miss and hit). It is in the program hash.

- `output_tensor=` (scalar form only, default None = the op allocates its output, unchanged): the op writes that tensor, whose spec must be exactly the output spec (checked on every call, miss and hit), and returns a tensor of its buffer. Not the input. (CHANGELOG: output_tensor, deepseek_v41_flash_d_p P.16.)

- Chain mode (deepseek_v41_flash_d_p P.22, scalar form; selected by any of these options or a ROW_MAJOR input; needs
  kv_actual_global 0, no seq_subshard_axis, no concat_prefix, rotary_dim; device/rope_chain_device_operation.cpp, kernels
  device/kernels/chain/): one program for the mesh, its block split from 2 common runtime args (no per-core args).
  - `input_heads` (int, default 0): H > 0: the input is [B, 1, S, H W] TILE and head h is read from its tile columns
    h W/32 .. (h+1) W/32 - 1 in place (nlp_create_q_heads_split's split, unwritten); `rotary_offset` is within W.
  - a ROW_MAJOR [B, H, S, W] input: tilized in the op (sparse_sdpa's output).
  - `output_layout=ttnn.ROW_MAJOR_LAYOUT`: [B, H, S, W] bf16 rows, untilized in the op (sparse_sdpa's q).
  - `output_dtype` (bfloat16 / bfloat8_b / bfloat4_b, default the input's): bfloat8_b packed with bfp8_pack_precise
    (ttnn.typecast's one rounding, DS41-137); bfloat4_b rounded to nearest even by bfp_round's SFPU rule
    (ckv_bfp4_ttnn/kernels/ckv_round_bfp4.hpp), then packed (bits copied).
  - `group_heads` (int, default 0): G > 1: the output [B, H / G, S, G W] TILE, head h in group h / G at columns
    (h % G) W (nlp_concat_heads of [B H / G, G, S, W]).
  The rotary arithmetic is rotary_embedding_llama's compute step for step (matmul by trans -> bf16, x sin -> bf16,
  x cos -> bf16, sum -> bf16; HiFi4 fp32 DEST by default), so every output equals the unfused chain's bit for bit.

## Measured
- Chain mode (P.22, 8x4 Galaxy, 110 cores, `tests/unit/test_chain.py`): bit-identical on every chip, miss and hit, to
  split -> rope -> to_layout (attention q, [1, 1, 160, 64 x 512] -> [1, 64, 160, 512] RM; indexer-shaped 32 x 128),
  to_layout -> inverse rope -> nlp_concat_heads (8-head groups) -> typecast bfloat8_b (attention output), split -> rope
  -> bfp_round bfloat4_b (indexer q). Warm host a call, q chain: unfused 3 ops 259.9 us, fused 61.4 us. DeepSeek-V4.1
  all-device model, profile rung's last chunk, same process (`deepseek_v41_flash_d_p/tt/perf_rope_chains.py`):
  warm wall median 273.8 ms (off) -> 250.1 ms (on), n 3 each; final hidden state bit-identical.

- Output buffers (P.16, 8x4): rotary_embedding_indexed 69.1 -> 50.8 us of host a call (8x4, 136 calls a chunk).
- Galaxy 8x4 (32 BH chips), `tests/unit/test_uniform.py::test_uniform_host_time`, warm host time a call (dispatch
  only), q-with-prefix shape [1, 64, 160, 64] + prefix [1, 64, 160, 448] bf16: stock 101.0 us, uniform 52.0 us.
- DeepSeek-V4.1 all-device model, 136 calls a chunk, profile rung's last chunk [51200, 56320), same process,
  alternating (`deepseek_v41_flash_d_p/tt/perf_rope_uniform.py`): warm wall median 438.6 ms (stock) -> 399.7 ms
  (uniform), n 6 each. Final hidden state bit-identical.
- Output byte-identical to the stock op on every chip: 6 model shapes, a miss and a hit (`test_uniform_matches_stock`).

## Limits
- Chain mode: bf16 input / cos / sin / trans_mat, interleaved tensors, rows a multiple of 32, tile-aligned head width
  and rotary region; a ROW_MAJOR output is bf16 without group_heads; cos / sin [1, 1, >= S, R] read from row 0
  (per-chip pre-sliced). The bfloat4_b rule is exact for groups whose max is >= 2^-124 (ckv_round_bfp4.hpp).
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
Tests: `tests/unit/test_uniform.py` (uniform vs stock, refusals, host time), `tests/unit/test_chain.py` (chain mode vs the
unfused chains, refusals, host time), `tests/source.yaml` (the source op's tests).

Chain mode (P.22), e.g. the attention q from wq_b's output:
```python
q_rm = ttnn.bringup.rotary_embedding_indexed(
    qb, cos_rows, sin_rows, trans_mat, 0, 0, memory_config=ttnn.DRAM_MEMORY_CONFIG, rotary_dim=64,
    rotary_offset=448, uniform=True, input_heads=64, output_layout=ttnn.ROW_MAJOR_LAYOUT)
```

## Used by
deepseek_v41_flash_d_p (P.5, `tt/rope.py:rope_stock`, setting ROPE_UNIFORM_PROGRAM; P.22 chain mode, `tt/rope.py:rope_chain`
from `tt/attention.py` `_q` / `_out` and `tt/indexer.py` `_query_chain`, setting ROPE_CHAINS)

## History
CHANGELOG.md
