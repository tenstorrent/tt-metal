# ttnn.bringup.grouped_matmul (grouped_matmul_ttnn)

## What it does
New op (Python ProgramDescriptor, no TTNN source). y[g] = x[g] @ w[g] for x [1, G, M, K], w [1, G, K, N] -> [1, G, M, N]
(TILE, DRAM interleaved, any tile dtypes). One grid row a group, GX = min(grid width, N / 32) columns; core (cx, g)
computes all M rows x PN(cx) column tiles of group g:
- x: the row's column-0 core reads each [M x block_w] K block from DRAM and multicasts it along the row (READY / VALID
  semaphores; every wait bounded);
- w: every core reads its own [block_w x PN] weight blocks (no weight tile read twice);
- compute: fp32 DEST accumulates a block's products, the packer adds them to Float32 partials in L1 (an fp32 sum over
  the blocks); one copy + pack to the output dtype at the end; output written by each core.
Always dispatched as a prepared program (ttnn.bringup.prepare_program / run_prepared; x, w, y addresses patched).

## Consider it instead of
- `ttnn.matmul` with one weight per batch at small M (2D multicast, fuse_batch=False): it splits only M over the grid
  rows and loops the batches on every core (55 cores at M = 160 on an 11 x 10 grid).

## Options
- `dtype` (default x's): output dtype. `compute_kernel_config`: math fidelity, fp32 DEST, approx (default HiFi2, fp32).
- `output_tensor`: the caller's buffer (exact spec), returned as is. `memory_config`: of a new output.
- `max_grid` (CoreCoord or (x, y), default the whole grid): cores from (0, 0) .. (max_grid - 1): a Galaxy's 110-core
  matmul cap (agent rule 18).
- `block_w` (default the widest of 32, 16, 8, 4, 2, 1 that fits 1 MB of L1): K tiles a block.
- `depth` (default 2): K blocks of x and of w buffered a core.
- `rt` (default 1): row tiles a DST block (rt x PN tiles; full-sync DEST when it exceeds half of DEST).
- `supported(x_shape, w_shape, x_dtype, w_dtype, out_dtype, device, max_grid=None, block_w=None)`: whether a call has a
  layout (cached).

## Measured
- 8x4 Galaxy, wo_a of deepseek_v41_flash_d_p (8 x [160, 4096] @ [4096, 1024] bfloat8_b in / out, 88 cores), slowest
  chip: stock 2D 0.175 ms; block_w 4, depth 4: 0.133 ms (block_w 8: 0.140, 32: 0.150); bit-identical to the stock call.

## Limits
- G <= the grid's rows; batch shape [1, G]; M, K, N multiples of 32; interleaved DRAM tensors.
- The x sender reads x alone (one core a row); the weight reads go one block in flight a core (a read-ahead version
  measured slower).
- Cores: G x GX (88 on the 8x4).

## How to call
```python
from ttnn.bringup.grouped_matmul_ttnn import supported
if supported(tuple(x.shape), tuple(w.shape), x.dtype, w.dtype, ttnn.bfloat8_b, mesh, grid):
    y = ttnn.bringup.grouped_matmul(x, w, dtype=ttnn.bfloat8_b, compute_kernel_config=ckc, max_grid=grid, block_w=4)
```

## Used by
- deepseek_v41_flash_d_p: tt/attention.py `_out_wo` (ATTN_ROOFLINE, P.81), the grouped wo_a.
