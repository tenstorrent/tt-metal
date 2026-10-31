# ttnn.bringup.nlp_create_qkv_heads

## What it does
Fork of `ttnn.experimental.nlp_create_qkv_heads, ttnn.experimental.nlp_create_q_heads_split`: the stock op, plus an option that keeps its cores inside a rectangle of the compute grid.

## Consider it instead of
`ttnn.experimental.nlp_create_qkv_heads, ttnn.experimental.nlp_create_q_heads_split`, when an op may not use the whole grid (a Galaxy: at most 110 of a chip's 120 Tensix cores, agent rule 18).

## Options
- `max_grid` (ttnn.CoreCoord, default None): the program factory's grid is the device grid cut to max_grid. None: the whole grid, the stock op's program.

## Measured
8x4 Galaxy (deepseek_v41_flash_d_p FIX.15): 120 -> 110 cores a chip; per-op device time in that model's BREADCRUMBS FIX.15.

## Limits
Only the interleaved paths take `max_grid`; sharded inputs / outputs keep their shard grid. Source tests not yet carried (tests/source.yaml: selection and baseline in task F.1; FIX.15 ran no fork suite on the 8x4, OV-25).

## How to call
`ttnn.bringup.nlp_create_qkv_heads`(..., max_grid=ttnn.CoreCoord(11, 10)) — every other argument as the stock op.

## Used by
deepseek_v41_flash_d_p FIX.15: tt/attention.py, tt/indexer.py (q heads split into nope / rope)

## History
CHANGELOG.md
