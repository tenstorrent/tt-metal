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

### MoEDtypes.shared_activations bf8 and shared_pack_precise: the fused head's shared expert at the reference's precision (2026-11-06, deepseek_v41_flash_d_p P.37)
- What: `shared_activations = BFLOAT8_B` (the field existed, unused): `HeadShape.x_dtype` / `SeEngineArgs.x_dtype`
  size x own / x in / h own / h all at 1088 B tiles (`SeEngineSizing.xtb / htb`), the engine's CBs take Bfp8_b, the
  kernels' tile bytes come from host defines (`SE_XTB`, `SE_HTB`, `SE_PREP_TB` in se_brisc / se_compute, `XTB` in the
  x gather's reader / sender / walk; default 2048 = the source's), the engine's own-shard load and the gather read the
  new input `xq` (`HeadSrc::SRC_XQ`, `fused_head(..., xq=)`; required exactly when bf8, validated), the router keeps
  bf16 x (SE_RTRX_L1 off when x own is bf8: the router's x from DRAM). `shared_pack_precise` (new):
  `HeadProgramConfig.pack_precise` -> the SE compute's `bfp8_pack_precise` (+ define `SE_PACK_PRECISE`); in the program
  hash and the head's plan / setup keys. `MoEBlock(xq=)` in models/demos/moe_block_main: the caller's xq, else one
  typecast a head call on the capped grid (`to_xq`, `xq_buffer`).
- Why: DeepSeek-V4.1's reference fp8-quantizes the shared expert's x and h (FIX.9, DS41-132; owner rule 7: never above
  the reference) and casts h once (SE_PACK_PRECISE, DS41-137): the carried block's se_engine3 x_dtype / h_dtype /
  pack_precise, ported to the C++ head.
- Tests: `tests/test_fused_head_bf8.py` (8x4: bf8 accuracy and determinism; the packed h bit for bit, precise and
  default). Defaults unchanged (bf16, False): the source's programs.
- Needed by: deepseek_v41_flash_d_p (DS41_MOE_BLOCK_SRC=main: GEMM_INPUTS=bfp8, SE_PACK_PRECISE, NORM_INTO_MOE)
- Files: `common/moe_block_config.hpp`, `common/common_nanobind.cpp`, `fused_head/fused_head.{hpp,cpp}`,
  `fused_head/fused_head_nanobind.cpp`, `fused_head/device/{head_plan,head_program,head_sizing}.{hpp,cpp}`,
  `fused_head/device/fused_head_device_operation{.cpp,_types.hpp}`, `fused_head/device/kernels/{se_defs.hpp,xg_walk.hpp}`,
  `fused_head/device/kernels/dataflow/{se_brisc,xg_reader,xg_sender}.cpp`, `fused_head/device/kernels/compute/se_compute.cpp`,
  `tests/test_fused_head_bf8.py`; models/demos/moe_block_main/tt/moe_block.py

### combine_snap_split: an order-free combine past 2^24 q (DeepSeek-V4.1 FIX.23, 2026-11-12; ported by P.50)
- Source: exp/fix23_snapshot 4e90f611e86 (parent 4c79ce65ece, the replica branch); only the fix's hunks ported onto
  this base (P.50, owner 2026-11-12), no replica work.
- What: `MoEBlockConfig.combine_snap_split` (default false = the single snapped fp32 sum, the program as before). On
  (needs `combine_snap` q): the accumulator splits each weighted term v into hi = v rounded to a multiple of
  q_hi = 2^18 q ((v + C_hi) - C_hi, C_hi = 1.5 x 2^23 q_hi) and lo = v - hi (exact) rounded to a multiple of q, packs
  both onto two fp32 sums a token with packer L1 accumulation (CB_ACC twice as large: the tokens' lo sums, then their
  hi sums), and writes out = lo + hi (one fp32 rounding on the SFPU, then the bf16 pack). Every add of each sum is
  exact while |hi sums| < 2^24 q_hi (2^23 at q = 2^-19) and top_k <= 64 (lo: top_k x 2^17 q < 2^24 q; checked), so
  the output is the same in any landing order. Planner: `acc_cb_bytes` / `combine_l1_layout` / `combine_auto` take
  `acc_sums` (1 or 2) for the accumulators' L1; compile arg 8 of accumulator_compute.cpp is C_hi's bits (0: off).
  The split is one SFPU pass a tile (`combine_split_face`: the same fp32 adds in the same order as add_unary /
  sub_unary / sub_binary).
- Why: the single sum is exact - order-free - only while it stays below 2^24 q (32 at DeepSeek-V4.1's q = 2^-19); the
  model's routed sums reach 21376 (layers 15-39 on a code prompt, the first tokens of any prompt), so its outputs
  depended on the packets' arrival order (code prompt up to 0.258 on the final hidden; DS41-233 / 234 / 235).
- Measured: README "Measured" (combine_snap_split); this base: P.50's A/B.
- Cores: unchanged (the accumulators' cells; their DEST halves hold 2 tiles a pass instead of 4 with the split).
- Needed by: deepseek_v41_flash_d_p FIX.23 / P.50 (setting COMBINE_SPLIT, default off until the owner decides).
- Files: common/{moe_block_config.hpp,moe_block_config.cpp,common_nanobind.cpp},
  routed_experts_combine/device/{combine_host.hpp,combine_host.cpp,combine_program.hpp,combine_program.cpp},
  routed_experts_combine/device/kernels/compute/accumulator_compute.cpp; tests:
  tests/routed_experts_combine/test_experts_combine_checks.py::test_experts_combine_split_determinism,
  models/demos/deepseek_v41_flash_d_p/tests/test_determinism.py.

### The flat expert's x snake pieces of at least N bytes (2026-11-14, deepseek_v41_flash_d_p P.59)
- What: `FlatLayoutOptions.x_min_piece_bytes` (default 0 = the source's choice: 4 cut-through pieces whenever the x
  block splits into 4 16-byte-aligned pieces). > 0: the most of 4 / 2 / 1 pieces of at least that many bytes each.
  Source: Marko Staletovic's WIP commit 1e2ab798da1 (origin/mstaletovic/flat-expert-perf-dw-split, parent 0c0bbe59d98
  = this fork's source), its x-pieces hunk of flat_routed_expert_program_factory.cpp only (his :127-131, the
  threshold 8 KB), taken behind the option instead of replacing the source's line; his DW split (FRE_DW_SPLIT), the
  gate/up hot/cold loop and the TEMP env hooks are not taken. The option is in the plan key and the program hash
  (FlatLayoutOptions::key() appends a tag + the value only when set: 0 keeps the source's key). The factory's log
  line prints the piece bytes.
- Why: the forwarder keeps 3 pieces in flight, so its bytes in flight bound the snake's x rate; at 64-row sub-blocks
  (DeepSeek-V4.1: H 5120, the 110-core cap, mt 2, x block 17,408 B) 3 x 4,352 B starve the gate/up cores; 2 x 8,704 B
  do not. Same math: outputs bit-identical (checked).
- Measured (Galaxy 8x4, flat op alone, 12 experts x M tokens a chip, the block's plan, real-time profiler, slowest
  chip, median of 10; tests/test_expert_x_pieces.py): M 64 0.657 -> 0.661 ms (+0.5 %), 256 1.012 -> 0.822 (-18.8 %),
  512 1.760 -> 1.447 (-17.8 %), 1024 3.243 -> 2.683 (-17.3 %); bit-identical at every M.
- Cores: none added. Host: none (a compile-time arg only).
- Needed by: deepseek_v41_flash_d_p P.59 (EXPERT_X_PIECES=min8k: x_min_piece_bytes 8192, tt/moe_block_main_adapter.py)
- Files: flat_routed_expert/device/{flat_routed_expert_device_operation_types.hpp,flat_routed_expert_options.cpp,
  flat_routed_expert_program_factory.cpp}, flat_routed_expert/{flat_routed_expert_nanobind.cpp,
  flat_routed_expert_weights.py}, README.md; tests: models/demos/deepseek_v41_flash_d_p/tests/test_expert_x_pieces.py

### Cache-hit patching through cached runtime-args records: `cached_slots` (2026-11-14, deepseek_v41_flash_d_p P.63)
- What: `fused_head(..., cached_slots=False)` and `routed_experts_combine(..., cached_slots=False)` (FusedHeadParams /
  RoutedExpertsCombineParams.cached_slots, not hashed). On: a program-cache hit writes each moved address arg through a
  `RuntimeArgsData*` the program's patch state keeps (HeadLastAddresses::slots, FlatProgramHandles::slots,
  CombineHandles::slots; filled on the program's first such hit), and the combine keeps its chip's `Program*`
  (shared_variables_t::program) instead of the workload map lookup a chip. Off (default): GetRuntimeArgs /
  GetCommonRuntimeArgs a patch, as before. Safe because a Kernel allocates its runtime-args records once (Kernel ctor,
  core_to_runtime_args_data_) and only re-points their data; the records live as long as the cached program, the same
  lifetime the existing `last` addresses assume. The same values are written: programs and outputs bit-identical.
- Why: P.63's probe (8x4, 4 layers, C++ timers): the combine's hit patching was ~245-250 us of a ~470 us call (the
  flat expert's 766 address args a chip, ~8 ns each: get_kernel's two hash lookups and a shared_ptr copy), the head's
  3.6 us x 32 chips of a ~340 us call. One mesh program would remove that loop's x32 but keeps the per-device command
  copies (Metal already packs a multi-device program once and copies it per device; DS41-258); the cached records
  remove most of it without a kernel change.
- Needed by: deepseek_v41_flash_d_p P.63 (MESH_PROGRAMS=on).
- Files: fused_head/fused_head.{hpp,cpp}, fused_head/fused_head_nanobind.cpp, fused_head/device/head_program.{hpp,cpp},
  fused_head/device/fused_head_device_operation{.cpp,_types.hpp}, flat_routed_expert/device/flat_program.hpp,
  flat_routed_expert/device/flat_routed_expert_program_factory.cpp, routed_experts_combine/routed_experts_combine.{hpp,
  cpp}, routed_experts_combine/routed_experts_combine_nanobind.cpp, routed_experts_combine/device/combine_program.{hpp,
  cpp}, routed_experts_combine/device/routed_experts_combine_program_factory.{hpp,cpp},
  routed_experts_combine/device/routed_experts_combine_device_operation_types.hpp, README.md.
- Tests: none added to tests/ (host-only, same values: the model's ladder and tests/test_determinism.py check
  outputs; tests/test_mesh_program_probe.py in the model measures the host us a call); fork suites not run on the 8x4
  (OV-25: task F.1).

### Expert replication removed from this fork on model_bringup (2026-11-15, deepseek_v41_flash_d_p FIX.33)
- What: the fork is back to its pre-replica state (model_bringup `6bdf324e716`) plus the two changes that do not
  depend on replication: P.59 `FlatLayoutOptions.x_min_piece_bytes` and P.63 `cached_slots` (both entries above,
  re-applied on the pre-replica code; their conflicts were only the replica fields next to them). Removed with their
  entries (P.52 expert_slot_copy op and the router's extra slot / slot map, P.53 set_combine_fill / REC_FILL, P.55b
  set_combine_fill_exact / REC_FX, P.55c the flat expert's replica-last schedule (SE_FX_SPARE, fx_wait.hpp) and the
  uneven slot split, P.56 / P.60 the exact ring and ordinal, P.62b prev counts, the fused head's HeadTail and
  prev_planner.hpp): `expert_slot_copy/`, `fused_head/device/kernels/prev_planner.hpp`,
  `flat_routed_expert/device/kernels/fx_wait.hpp`, the `RouterSpec` slot fields, `MoEBlockResources`' fill words.
- Why: owner 2026-11-15: model_bringup holds the latest work without expert replication; the replica branch
  (`bringup/deepseek_v41_flash_d_p/moe_replication` @ `b9a45fb0b80`) keeps them and is rebuilt on top later.
- Behaviour: every removed path was off unless a replica option was set; with them unset the programs are the
  pre-replica ones (outputs bit-identical to the model's MOE_REPLICAS=off path: test_determinism, the ladder).
- Cores, host: unchanged for the remaining options.
- Files: CMakeLists.txt, moe_block_nanobind.cpp, common/{common_nanobind.cpp,moe_block_config.{hpp,cpp},
  moe_block_resources.{hpp,cpp}}, expert_slot_copy/ (deleted), flat_routed_expert/device/{flat_program.hpp,
  flat_routed_expert_device_operation_types.hpp,flat_routed_expert_program_factory.cpp,kernels/*}, fused_head/
  (head_plan, head_program, head_sizing, fused_head*, kernels router_*, xg_watcher.cpp, prev_planner.hpp deleted),
  routed_experts_combine/ (combine_program, combine_port_writer.cpp, factory, op, nanobind), README.md, ../INDEX.md.
- Tests: no fork suite on the 8x4 (OV-25: task F.1); the model's gate (ladder, determinism, contract, decoder skip,
  profile) runs the remaining options at their defaults.
