# moe_block

The MoE block's prefill ops (`ttnn.experimental.moe_block`): token owners, 2D dispatch, flat experts + combine in
one program. A different algorithm from `deepseek_prefill/` (sparse 2D multi-hop unicast dispatch, L1 accumulate
combine, fused head); the Python `MoEBlock` (models/demos/moe_block) orchestrates them.

- `common/`: `MoEBlockConfig` / `MoEModelDims` / `MoEDtypes`, the shared host math (routes and the fabric's parity tie
  rule, span tables, manifest layout, mesh plan, port LUTs, Ethernet-nearest port placement) and `MoEBlockResources`
  (the persistent words, placement and plans the ops share), `address_patches` (address-only program-cache
  hits for a per-chip mesh workload: a tensor's runtime args are rewritten only when its address moved).
- `fused_head/`, `sparse_dispatch_2d/`, `flat_routed_expert/`, `routed_experts_combine/`, `fused_tail/`: one device op
  each (one program a chip); each has its own README.
- `block/`: `moe_block_arena_spec`, the block's one L1 arena a call (the fused head runs in it, then the routed
  experts + combine; both take `arena=`).

A forward (`models/demos/moe_block/tt/moe_block.py` `MoEBlock`): allocate the arena, `fused_head(..., arena=)`,
`sparse_dispatch_2d`, `routed_experts_combine(..., arena=)`, free the arena, `fused_tail`. `models/demos/moe_block/tests/test_moe_block_smoke.py` runs it on the mock
Galaxy or an 8x4 Galaxy.
