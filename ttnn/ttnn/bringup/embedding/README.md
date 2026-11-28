# ttnn.bringup.embedding

## What it does
Fork of `ttnn.embedding`: the stock op, plus an option that keeps its cores inside a rectangle of the compute grid.

## Consider it instead of
`ttnn.embedding`, when an op may not use the whole grid (a Galaxy: at most 110 of a chip's 120 Tensix cores, agent rule 18).

## Options
- `max_grid` (ttnn.CoreCoord, default None): an interleaved output's work split runs on the rectangle (0, 0) .. (max_grid.x - 1, max_grid.y - 1). None: the whole grid, the stock op's program.

- `server_start`, `server_valid`, `clamp_max` (int, default None; all three or none; deepseek_v41_flash_d_p NC.9): the input is every mesh row's ids in the server's order ([SP, 1, W] uint32, the same on every chip); each chip embeds its natural-order rows [r W, (r + 1) W) of the chunk at server_start (r its mesh row), positions >= server_valid and ids > clamp_max read as clamp_max. server_start / server_valid are runtime values (no program per value). TILE output only.

## Measured
8x4 Galaxy (deepseek_v41_flash_d_p FIX.15): 120 -> 110 cores a chip; per-op device time in that model's BREADCRUMBS FIX.15.
Server order (NC.9, 8x4, 8192 / 5120 chunks): vs the model's IdsReorder kernel + this op, same bytes; host 163-165 -> 133-134 us a call, device 42-48 -> 34-36 us (results/NC.9_layer.md).

## Limits
Server order: GENERIC embeddings, uint32 ROW_MAJOR ids, a TILE interleaved output, the mesh rows as the SP axis. Only the interleaved paths take `max_grid`; sharded inputs / outputs keep their shard grid. Source tests not yet carried (tests/source.yaml: selection and baseline in task F.1; FIX.15 ran no fork suite on the 8x4, OV-25).

## How to call
`ttnn.bringup.embedding`(..., max_grid=ttnn.CoreCoord(11, 10)) — every other argument as the stock op.

## Used by
deepseek_v41_flash_d_p FIX.15: tt/attention.py window-cache server rows (TtWindowCache.write), tt/compressor.py served rows; NC.9: tt/model.embed (server order, tt/page_copy.IdsReorder.embed)

## History
CHANGELOG.md
