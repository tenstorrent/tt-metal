# combine_rf usage examples (reference only)

Copied unchanged from Marko Staletovic's `mstaletovic/galaxy-unified-k3-v41-combine` @ `e4d239c6d81` (2026-10-19), for
the bring-ups that wire `ttnn.bringup.flat_routed_expert_with_program` and `sparse_combine_rf` (owner exception for the
routed-expert op and its combine). Not collected by pytest and not maintained here; they may import modules that live
only on that branch.

- `scratch_flat_combine_rf.py` (was `models/demos/deepseek_v3_d_p/tests/perf/`): the flat expert with the
  router-forwarded weighted combine appended in one program (`combine_planner_prepare`, `port_luts`,
  `sparse_combine_rf_prepare`, `FlatRoutedExpert`), checked against a torch reference.
- `tt_k3_token_split_moe.py` (was `models/demos/deepseek_v3_d_p/tt/moe/`): Kimi K3's token-split MoE block using it end
  to end (router, routing gather, planner, dispatch, experts + combine, shared expert).
