# moe_block_main_ttnn (new op set, from Marko Staletovic's branch)

- Source: `ttnn/cpp/ttnn/operations/experimental/moe_block` on `mstaletovic/moe-block-main` @ `0c0bbe59d98` ("MoE block
  onto main: moe-block-main's net change ... applied to origin/main as one commit", on origin/main `f93403fa228`,
  2026-11-06). Its tests (`tests/ttnn/unit_tests/operations/experimental/moe_block`) are `tests/` here; its Python block
  (`models/demos/moe_block`, `models/demos/deepseek_v3_d_p/tt/moe_impl.py`, `.../kimi_k3/moe_block_ffn.py`) is
  `models/demos/moe_block_main` (`tt/moe_impl.py`, `tt/k3_moe_block_ffn.py`).
- Python: `ttnn.bringup.moe_block_main.*` (was `ttnn.experimental.moe_block.*`); one nanobind submodule
  (`ttnn._ttnn.operations.bringup.moe_block_main`), merged with the registered ops by ttnn/__init__.py.
- Not a fork of an op on this tree: `ttnn/cpp/.../experimental/moe_block` does not exist here (no source op to keep in
  sync, `tests/source.yaml` empty). Ported for deepseek_v41_flash_d_p (owner 2026-11-06: the MoE block without DRISC,
  to see its host time as it is).

Mechanical changes (a copy script, the fork_op.py conventions): C++ namespace
`ttnn::operations::experimental::moe_block` -> `ttnn::operations::bringup::moe_block_main`, prim functions
`ttnn::prim::moe_block_<op>` -> `ttnn::prim::moe_block_main_<op>`, CMake target `ttnn_op_bringup_moe_block_main_ttnn`
(alias `TTNN::Ops::Bringup::MoeBlockMainTtnn`), kernel paths and fork-internal includes pointing at this folder
(relative includes), Python prefix `ttnn.bringup.moe_block_main.`, the tests' repo-root depths.

API port to this tree (main's API of 2026-11-06 that this base, origin/main `b7856ed3d30` of 2026-11-02, lacks):
- `tt::tt_fabric::append_fabric_connection_rt_args(mesh_device, src, dst, link, ...)`: main added the mesh device
  argument (the fabric context it resolves in); this tree's signature has none. The five calls
  (fused_head head_program.cpp, sparse_dispatch_2d program factory, routed_experts_combine combine_program.cpp,
  fused_tail program factory x2) drop it. No stock code changed.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->
