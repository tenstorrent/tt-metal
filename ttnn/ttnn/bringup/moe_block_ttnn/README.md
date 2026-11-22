# ttnn.bringup.moe_block (moe_block_ttnn)

## What it does
Fork of Marko Staletovic's MoE block ops (`ttnn.experimental.moe_block`, `ttnn/cpp/ttnn/operations/experimental/moe_block`,
his split-rows stack + int32 snap): moe_head (x gather over TP, shared expert, router, routing exchange + combine
planner, dispatch prep), sparse_dispatch_2d, routed_experts_combine (flat routed experts + L1-accumulate combine in one
program), moe_tail, moe_block_plan / moe_block_arena_spec, MoEBlockResources. The source's README files in each op
folder describe the ops; this fork adds only the options below. Split rows: the dispatch and combine send a token row in
two packets when the mesh's fabric payload is below a row (+ 32 B trailer) - they read the payload the mesh opened with.

## Consider it instead of
`ttnn.experimental.moe_block` when a model needs one of the options below; `moe_block_main_ttnn` (his older block).

## Options
- `FlatLayoutOptions.weights_nk` (False): routed weights laid out from [N, K] matrices, transposed back in the matmuls.
- `FlatLayoutOptions.x_min_piece_bytes` (0): the flat expert's x snake pieces of at least N bytes.
- `set_max_grid(max_x, max_y)` ((0, 0) = no cap): every program inside the first max_x columns / max_y rows.

## Measured
deepseek_v41_flash_d_p MOE_PORT (2026-11-18): see /data/dnijemcevic/bringup/handoff_20261118/moe_port_report.md.

## Limits
Galaxy 8x4 (Blackhole) with DeepSeek-V4.1 dims only, at the 11 x 10 cap; the source's own tests were not run on the fork.

## How to call
models/demos/deepseek_v41_flash_d_p/tt/moe_block_v2_adapter.py (MoEBlock on these ops, or the ops one by one).

## Used by
deepseek_v41_flash_d_p (MOE_BLOCK_SRC=block, MOE_PORT)

## History
CHANGELOG.md
