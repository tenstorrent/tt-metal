# moe_block_ttnn (fork)

- Source: `ttnn/cpp/ttnn/operations/experimental/moe_block`
- Source SHA: `5fac00ddff4353a9dd0ebf279ed44a5c45c00222`
- Source branch: Marko Staletovic's stack as cherry-picked onto this branch (2026-11-18): `mstaletovic/moe-block-65124-split-rows`
  (MoE block [0/6]..[6b/6], MiniMax-M3, the fabric writers' NoC fix, split rows `63f6e28f54a`) + `mstaletovic/moe-block-int32-snap`
  `a7c1a9d78c0` (the combine's int32 snap), on main `8a317f39a8d`.
- Python: `ttnn.bringup.moe_block.*` (was `ttnn.experimental.moe_block.*`): one nanobind submodule, merged by ttnn/__init__.py
  as moe_block_main's. Its Python block stays `models/demos/moe_block/tt/moe_block.py` (MoEBlock), built on these ops by
  the model's adapter (it points `ttnn.experimental.moe_block` at this module while a block is constructed).
- Forked for: deepseek_v41_flash_d_p MOE_PORT (owner 2026-11-18: the model's MoE on Marko's current block, with split rows).
  The older copy of his block, `moe_block_main_ttnn`, stays as the model's MOE_BLOCK_SRC=main fallback.
- Used by: deepseek_v41_flash_d_p

Mechanical fork changes (fork_op.py): namespace `ttnn::operations::bringup`, CMake target `ttnn_op_bringup_moe_block_ttnn`, kernel paths and includes pointing at this folder, Python prefix `ttnn.bringup.`.
By hand after fork_op.py: the nesting had rewritten shared namespaces (`tt::tt_metal::distributed` forward declarations ->
`tt::tt_metal::bringup::distributed::bringup`, `tt::tt_metal`, and the kernels' `ckernel::sfpu`) - restored to the
originals; the ops' registration prefix `"ttnn.experimental.moe_block."` -> `"ttnn.bringup.moe_block."` (it would
clash with the stock ops); flat_routed_expert_weights.py binds `ttnn._ttnn.operations.bringup.moe_block`.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### weights_nk: [N, K] weight tiles, transposed back in the matmuls (2026-11-18, deepseek_v41_flash_d_p MOE_PORT; moe_block_main's entry re-applied)
- What: `FlatLayoutOptions.weights_nk` (default false = the source's program and key): the gate/up and down computes
  get `SE_W_NK` and pass transpose = true to every `matmul_block_init` / `matmul_block` (in1 = the weight tile);
  `flat_weight_regions(W_l, lay, nk=True)` takes nn.Linear's [N, K] matrices and lays every tile out transposed;
  `pack_flat_weights` / `FlatRoutedExpert` pass `layout.weights_nk`. In `FlatLayoutOptions::key()` (plan key and
  program hash) only when on (a tagged word).
- Why: DeepSeek-V4.1's routed experts are e2m1 with a ue8m0 scale per 32 K values; a bfp tile shares an exponent per
  16 values along its rows, so [N, K] tiles keep a group inside one K block (bfp4_mse rounds in the checkpoint's
  groups). Also keeps the model's weight preparation (and, where the plan lays them out the same, its cached files).
- Needed by: deepseek_v41_flash_d_p (DS41_MOE_BLOCK_SRC=block)
- Files: flat_routed_expert/device/{flat_routed_expert_device_operation_types.hpp,flat_routed_expert_options.cpp,
  flat_routed_expert_program_factory.cpp}, flat_routed_expert/device/kernels/{gate_up_compute,down_compute}.cpp,
  flat_routed_expert/{flat_routed_expert_nanobind.cpp,flat_routed_expert_weights.py}

### set_max_grid: the block's programs inside a capped core grid (2026-11-18, deepseek_v41_flash_d_p FIX.15; moe_block_main's entry re-applied)
- What: `common/core_grid.{hpp,cpp}` (moe_block_main's, same code): `set_max_grid(max_x, max_y)` / `max_grid()` (bound
  in the module) and `block_grid(full)`. Every planner's grid read that places programs goes through it: port
  placement (`all_worker_cores`), the head's shape and model (`moe_head.cpp`, `head_model.cpp`), the flat expert's plan,
  the combine's free cores and pool, the tail's workers and engine. Not the words' slab or the call's arena (memory on
  every core, no program). Default (0, 0): no cap. A cap different from the one the first plan saw raises.
- Why: owner rule 18 (FIX.15): the model keeps every op within 110 Tensix cores of a chip on a Blackhole Galaxy.
- Needed by: deepseek_v41_flash_d_p (MAX_WORKER_CORES)
- Files: common/{core_grid.hpp,core_grid.cpp,sources.cmake,common_nanobind.cpp,port_placement.cpp},
  moe_head/{moe_head.cpp,device/head_model.cpp}, flat_routed_expert/device/flat_routed_expert_plan.cpp,
  routed_experts_combine/{routed_experts_combine.cpp,device/combine_program.cpp}, moe_tail/device/moe_tail_plan.cpp

### The flat expert's x snake pieces of at least N bytes (2026-11-18, deepseek_v41_flash_d_p P.59; moe_block_main's entry re-applied)
- What: `FlatLayoutOptions.x_min_piece_bytes` (default 0 = the source's 4 / 2 / 1 aligned pieces): > 0 = the most of
  4 / 2 / 1 pieces of at least that many bytes each (Marko Staletovic's 1e2ab798da1 threshold, 8 KB). In the key only
  when set. The factory's log line prints the piece bytes.
- Why: the forwarder keeps 3 pieces in flight; at 64-row sub-blocks 3 x 4.3 KB starve the gate/up cores (P.59: the flat
  op -17 % at M 256..1024 on the old block, bit-identical).
- Needed by: deepseek_v41_flash_d_p P.59 (EXPERT_X_PIECES=min8k)
- Files: flat_routed_expert/device/{flat_routed_expert_device_operation_types.hpp,flat_routed_expert_options.cpp,
  flat_routed_expert_program_factory.cpp}, flat_routed_expert/{flat_routed_expert_nanobind.cpp,flat_routed_expert_weights.py}

### Not re-applied from moe_block_main (2026-11-18)
- `combine_snap_split` (FIX.23 / P.50): the source's combine now snaps to int32 counts of q (Marko's a7c1a9d78c0, in the
  source SHA above): order-free up to (2^31 - 1) q; the model runs q = 2^-15 (65536 > its largest routed sum, 21376).
- `cached_slots` (P.63): the source's ops keep per-core values in a DRAM args table (common/args_table.hpp); a cache hit
  rewrites only common args, one program for the mesh - the per-core patch loop cached_slots shortened is gone.
- `MoEDtypes.shared_activations` bf8 / `shared_pack_precise` (P.37): the new moe_head is a different engine; not ported
  (the shared expert's x / h stay bf16 - more precise than the reference's fp8, the model's ladder decides).

### shared_scatter_columns: the shared expert reduced by columns without a latent (2026-11-20, deepseek_v41_flash_d_p PB.7)
- What: `MoEBlockConfig.shared_scatter_columns` (default false = the source's rows scatter, same programs and plans):
  true = `HeadShape.scatter_columns` -> `reduce_scatter_dim()` -1 without a latent (the columns scatter Kimi-K3's latent
  path already runs: y [rows, H / TP]), and `ReduceScatterPlan(column_halves)`: the RingBalanced halves split by matmul
  core column (`half_split_column` over the block columns' widths, as the rows scatter splits its columns) instead of
  by core index (= by core row, i.e. by the row again). `row_block_counts` sizes y's pages as x's then. The caller turns
  y into owner rows itself (a TP all-to-all); MoEBlock's tail takes the rows scatter only. With padding rows
  (actual_start / actual_end) the port kernels' skip of an all-padding sub-block (x_gather_block_real) reads a column
  block as every row of the core (named arg `reduce_scatter_columns`), not as a row block.
- Why: DS41-339. The rows scatter adds a row's TP partials in its destination chip's ring order (bf16 relays), so the
  same token row gives other bits (1 bf16 ulp) when it lands on another TP chip: a packed chunk's segment placed off
  a mesh-row boundary differs from the same request alone. By columns every output column's adds run in its column
  block's ring order whatever row it is.
- Measured (8x4, DeepSeek-V4.1 dims, chunk 2048 / 8192, rows rolled 32 inside every SP row): off 1024 of 2048 rows
  differ (0.015625); on 0 rows, determinism 0, PCC vs torch 0.99956 (off 0.99957), one row block, 88 matmul cores
  (11 x 8, as off).
- Needed by: deepseek_v41_flash_d_p PB.7 (Setting MOE_HEAD_ORDER=free)
- Files: common/{moe_block_config.hpp,common_nanobind.cpp}, moe_head/moe_head.cpp, moe_head/device/{head_plan.hpp,
  head_plan.cpp,head_model.cpp,head_sizing.hpp,head_sizing.cpp}, tests/test_shared_scatter_columns.py

### router_canonical_sum: the router's TP logits partials summed in chip order (2026-11-20, deepseek_v41_flash_d_p PB.7)
- What: `MoEBlockConfig.router_canonical_sum` (default false = the source's sum) -> `HeadShape.router_canonical_sum` ->
  router_compute's named arg `router_canonical_sum`: the arrivals are waited for as before (the reader's fixed order) but
  not added as they come; the score pass sums ((chip 0's + chip 1's) + ...) in fp32 DEST, the owner's own partial
  (cb_logits) at its chip's turn, chip c's arrival from slot c < owner ? c : c - 1. Off: the owner's partial first,
  then the arrivals in the owner's predicted landing order (head_model.cpp arrival_order).
- Why: DS41-341. The source's order depends on the chip that owns the token, so a token's fp32 logits differ in the last
  bits with its TP chip; now and then a gate weight's bf16 rounding (or a near-tied pick) flips: deepseek_v41_flash_d_p's
  glued segment differed from alone in 1 of 1600 rows at layer 20 (0.031) after shared_scatter_columns fixed the rest.
- Measured: op test (tests/test_shared_scatter_columns.py, both options): routing rows of rows moved across TP chips
  identical (0 of 2048 / 8192); model probe layer 20: 0 rows differ (was 1). Cost: the adds after the last arrival
  instead of as they come (expert_tiles fp32 adds a row tile).
- Needed by: deepseek_v41_flash_d_p PB.7 (Setting MOE_HEAD_ORDER=free sets both options)
- Files: common/{moe_block_config.hpp,common_nanobind.cpp}, moe_head/moe_head.cpp, moe_head/device/{head_plan.hpp,
  head_router_program.cpp,kernels/head_named_args.hpp,kernels/compute/router_compute.cpp}
