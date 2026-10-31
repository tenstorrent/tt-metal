# moe_block

The MoE block's prefill ops (`ttnn.experimental.moe_block`): token owners, 2D dispatch, flat experts + combine in
one program. A different algorithm from `deepseek_prefill/` (sparse 2D multi-hop unicast dispatch, L1 accumulate
combine, fused head); the Python `MoEBlock` (models/demos/moe_block) orchestrates them.

- `common/`: `MoEBlockConfig` / `CombineTuning` / `MoEModelDims`, the shared host math (routes and the fabric tie rule,
  span tables, manifest layout, mesh plan, port LUTs, Ethernet-nearest port placement) and `MoEBlockResources` (the
  persistent words, semaphores, placement and plans the ops share).
- `fused_head/`, `sparse_dispatch_2d/`, `flat_routed_expert/`, `routed_experts_combine/`, `fused_tail/`: one device op
  each (one program a chip); each has its own README.
