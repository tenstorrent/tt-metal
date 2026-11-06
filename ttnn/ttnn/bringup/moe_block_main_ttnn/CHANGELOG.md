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

### weights_nk: [N, K] weight tiles, transposed back in the matmuls (2026-11-06, deepseek_v41_flash_d_p, MOE_BLOCK_SRC=main)
- What: `FlatLayoutOptions.weights_nk` (default false = the source's program, plan and key): the gate/up and down
  computes take `SE_W_NK` and pass transpose = true to every `matmul_block_init` / `matmul_block` (in1 = the weight
  tile); `flat_weight_regions(W_l, lay, nk=True)` takes nn.Linear's [N, K] matrices and lays out every tile transposed
  (the x @ W layout's tile at the same place, rows along K). In the plan key / program hashes only when on.
- Why: DeepSeek-V4.1's routed experts are e2m1 values with a ue8m0 scale per 32 K values. A bfp tile shares an
  exponent per 16 values along its rows: in the x @ W orientation a group spans 16 output columns with their own
  scales (bfp8 rounds, bfp4 / bfp4_mse lose the checkpoint's groups); [N, K] tiles keep a group inside one K block.
  The same option the carried `ttnn.bringup.flat_routed_expert` fork has (its CHANGELOG, P.3 / DS41-35), re-applied here.
- Tests: `tests/flat_routed_expert/test_flat_routed_expert_host.py` (`test_flat_weight_regions_nk`: every region tile
  the transpose of the plain layout's for both plans, pairs and 16-column units; `test_flat_layout_options_weights_nk`:
  the key unchanged when off).
- Needed by: deepseek_v41_flash_d_p (DS41_MOE_BLOCK_SRC=main)
- Files: `flat_routed_expert/device/flat_routed_expert_device_operation_types.hpp`, `.../flat_routed_expert_options.cpp`,
  `.../flat_routed_expert_program_factory.cpp`, `.../kernels/{gate_up_compute,down_compute}.cpp`,
  `flat_routed_expert/flat_routed_expert_nanobind.cpp`, `flat_routed_expert/flat_routed_expert_weights.py`,
  `tests/flat_routed_expert/test_flat_routed_expert_host.py`

### set_max_grid: the block's programs inside a capped core grid (2026-11-06, deepseek_v41_flash_d_p FIX.15 re-applied)
- What: `common/core_grid.{hpp,cpp}`: `set_max_grid(max_x, max_y)` / `max_grid()` (bound in the module) and
  `block_grid(full)`, the compute grid cut to the cap. Every planner's grid read goes through it: port placement
  (`all_worker_cores`), the fused head's shape and program (`h.grid`; its matmul columns are then `max_x - 1` and the
  movers the last column), the flat expert's plan (rectangles, relays, down cells), the combine's free cells and pool,
  the fused tail's A2A workers and engine. Not the call's L1 arena or the words' slab (memory on every core, no
  program). The flat plan's 12-column check applies to its fixed rectangles only (a shaped plan's east rectangle is
  sized to the columns left: `bw_cap`), and every rectangle is checked to fit. Default (0, 0): no cap, the source's
  grid and plans. A cap different from the one the first plan saw raises.
- Why: owner rule 18 (FIX.15): on a Blackhole Galaxy no op may run on more than 110 Tensix cores of a chip (the
  chassis' PSU trips on a chip's di/dt); the carried block takes `MoEOptions.max_grid` for the same reason. Unchanged
  at the default.
- Needed by: deepseek_v41_flash_d_p (DS41_MOE_BLOCK_SRC=main; MAX_WORKER_CORES)
- Files: `common/core_grid.{hpp,cpp}`, `common/sources.cmake`, `common/common_nanobind.cpp`, `common/port_placement.cpp`,
  `fused_head/fused_head.cpp`, `fused_head/device/head_program.cpp`, `flat_routed_expert/device/flat_routed_expert_plan.cpp`,
  `routed_experts_combine/routed_experts_combine.cpp`, `routed_experts_combine/device/combine_program.cpp`,
  `fused_tail/device/fused_tail_plan.cpp`
