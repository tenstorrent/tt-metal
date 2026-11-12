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

### Replica slots (ported onto model_bringup + P.50 by DeepSeek-V4.1 P.52, 2026-11-12, from ffe570b0b4f; default off): RouterSpec.candidates + the fused head's slot map; gather_tiles (2026-11-11, deepseek_v41_flash_d_p P.46)
- What: `RouterSpec.candidates` (default 0 = the source: the router's candidates are the experts, num_experts x fold).
  n > 0: the router ranks n candidates (its rows) and `MoEModelDims.num_experts` is the SLOT count (> n, whole slots a
  chip); `HeadShape.NE` carries it (EPC() = NE / N), the router's bias region holds one more tile, the layer's slot map
  (`prepare_fused_head_weights(slot_map=)`: 1024 integer-valued fp32 words - [0, E) a candidate's own slot, [E, 2E) its
  copies n | first << 6, then the copies' slots; validated: every slot at most once), read with the bias by the router
  NCRISC. `router_rows.hpp` make() maps each (token, candidate) to one of the candidate's 1 + n slots (`slot_of`: holder
  = hash(chip << 16 | owned row) x (n + 1) >> 16, holder 0 the own slot), the gate weight still from the candidate's
  score; the route words' CHIP_OF covers the N x EPC slots. Every later stage (planner, dispatch, flat experts,
  combine) already works on slot ids. The NCRISC router kernel gets one more runtime arg (its row tile's first owned
  row, before the planner's args) on every path. `gather_tiles(tiles, src, rows=None)` (common/tile_gather.hpp, host
  only): a multi-device host TILE tensor's tiles regrouped across its shards (raw bytes, the input's element type, -1 a
  zero tile, a new second-to-last dim).
- Why: MoE expert replication (owner 2026-11-11): a hot expert copied into a spare slot on other chips, its tokens
  split evenly; the per-layer map is runtime data (the 40 layers share one head program). gather_tiles builds the
  13-slot weights from the 12-slot weight cache at load (bit-identical tiles).
- Measured (8x4, DeepSeek-V4.1 layer 20, 5120 random tokens, tests in models/demos/deepseek_v41_flash_d_p/tests/
  test_moe_replicas.py): 13 slots with an empty map vs the source path bit-identical; with 10 copies bit-identical too
  (the same routing, copies' tiles equal), the split exactly the hash's (e.g. 317 tokens over 4 holders 94/74/69/80).
- Needed by: deepseek_v41_flash_d_p P.46 (MOE_REPLICAS=table)
- Files: common/moe_block_config.{hpp,cpp}, common/common_nanobind.cpp, common/tile_gather.{hpp,cpp},
  common/sources.cmake, fused_head/fused_head.{hpp,cpp}, fused_head/fused_head_nanobind.cpp,
  fused_head/device/head_plan.{hpp,cpp}, fused_head/device/head_sizing.{hpp,cpp}, fused_head/device/head_program.{hpp,cpp},
  fused_head/device/kernels/router_rows.hpp, router_addr.hpp, dataflow/router_brisc.cpp, dataflow/router_ncrisc.cpp,
  compute/router_compute.cpp

### expert_slot_copy (ported by P.52 from 1e4b42b2d98): table-driven chip-to-chip copies of expert weight slots; gather_tiles removed (2026-11-12, deepseek_v41_flash_d_p P.47)
- What: a new op `expert_slot_copy(src_gu, src_d, dst_gu, dst_d, runs, resources, jobs, links=None, depth=8,
  batch=4)` (expert_slot_copy/): per job (src chip, dst chip, run table) chip src reads the table's runs (weight, src
  bank / offset, dst bank / offset, bytes; one page of `runs` a table) from its DRAM bank shards and writes them into
  chip dst's through the 2D fabric (unicast to any chip, through the port cores of the first hop's direction -
  mesh_routing first_hop - one sender core a link of that direction, the chunks split over them); src == dst: a local
  copy split over the chip's port cores. Handshake: the destination's program sends each sender core a ready when it
  starts (every program before it on that chip has finished), the senders write only after it, end with a flushed
  atomic increment of the destination's done word, and the destination's program ends only after every done (persistent
  words "expert_slot_copy/sems", reset by their waiter, no reset between calls). The program hash holds no address and
  no job: buffers bind as Buffer* (address_patches), the jobs are runtime args rewritten each call, so one cached
  program serves every (source, destination) pair. `gather_tiles` (P.46's host regroup) and common/tile_gather.* are
  removed (nothing uses them).
- Why: MoE expert replicas filled at runtime (owner 2026-11-12): P.46 built the 13-slot weights on the host at load
  (model load 188 -> 406 s, ~16 GB host peak a layer). Now the load is the off path's cache hit plus a local device
  re-layout (the same op, local jobs), and the spare slot is filled by a fabric copy between two MoE calls. Stock
  ttnn.point_to_point cannot do it (interleaved, 1D, whole tensors, one core a link).
- Measured (Galaxy 8x4, FABRIC_2D_TORUS_XY, DeepSeek-V4.1 layer 20 weights, one slot 18.98 MiB in 272 runs, chunk
  10880 B, 2 links): see README "Measured" (expert_slot_copy).
- Cores: 8 port cores a chip on the 8x4 torus (4 directions x 2 links), inside the block's port placement (the core
  cap).
- Needed by: deepseek_v41_flash_d_p P.47 (MOE_REPLICAS=runtime)
- Files: expert_slot_copy/{expert_slot_copy.{hpp,cpp},expert_slot_copy_nanobind.{hpp,cpp},sources.cmake},
  expert_slot_copy/device/{expert_slot_copy_device_operation{.hpp,.cpp,_types.hpp},expert_slot_copy_program_factory.{hpp,cpp}},
  expert_slot_copy/device/kernels/dataflow/{slot_copy_common.hpp,slot_copy_reader.cpp,slot_copy_writer.cpp},
  CMakeLists.txt, moe_block_nanobind.cpp, common/{common_nanobind.cpp,sources.cmake}, common/tile_gather.{hpp,cpp}
  (deleted); tests: models/demos/deepseek_v41_flash_d_p/tests/test_expert_slot_copy.py
