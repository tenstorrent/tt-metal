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

### MOE_FILL=piggyback: expert replica copies carried by the combine's port senders (2026-11-13, deepseek_v41_flash_d_p P.53)
- What: `set_combine_fill(resources, enabled, gate_up, down, runs, jobs, cap)` (default: never called = the source's
  program, bit for bit). Enabled, routed_experts_combine's port writer is built with `REC_FILL` (a program of its own
  in the cache: `RoutedExpertsCombineParams.fill` is hashed) and carries jobs [(src, dst, table, p0, p1, last)]: chip
  src's port facing the first hop toward dst (link 0) reads pieces [p0, p1) of expert_slot_copy's run table (each run
  cut in expert_slot_copy_chunk_bytes pieces) from its DRAM into 4 staging slots at the end of its arena (16 KB table +
  4 x 10880 B) and writes them into dst's DRAM through its own fabric connection: a piece only in a main-loop round
  that sent no row and while its channel holds fewer than `cap` packets; the call's rest after the main loop, while
  the credits drain (bounded by the host's p1 - p0: the exit waits on no other core). `last`: a flushed atomic
  increment of word src of the done words (`combine_fill_done`: persistent, the collector cell, [1, chips + 16]; words
  chips + p / chips + 8 + p count port p's pieces sent inside / after the main loop). The jobs are runtime args
  rewritten on every call (create and cache hit). `combine_fill_port(resources, src, dst)`: the carrying port's
  direction. `MoEBlockResources::set_op_input / op_input / clear_op_input`: a generic per-call input slot (the fill's).
- Why: P.47's expert_slot_copy is its own program, serial with the model (2.79 ms for 32 copies); the combine's port
  cores hold every router's one worker channel for the whole ~0.9-1.4 ms program and use ~10 % of the links, so the
  copy can ride there.
- Measured: README "Measured" (combine fill).
- Cores: none added (the 8 port cores' RISCV_0 senders; their reads on NoC1).
- Needed by: deepseek_v41_flash_d_p P.53 (MOE_FILL=piggyback; tt/expert_fill.py schedules the pieces)
- Files: routed_experts_combine/{routed_experts_combine.{hpp,cpp},routed_experts_combine_nanobind.cpp},
  routed_experts_combine/device/{combine_program.{hpp,cpp},routed_experts_combine_device_operation{.cpp,_types.hpp},
  routed_experts_combine_program_factory.{hpp,cpp}}, routed_experts_combine/device/kernels/dataflow/
  combine_port_writer.cpp, common/moe_block_resources.{hpp,cpp}; tests:
  models/demos/deepseek_v41_flash_d_p/tests/test_moe_fill_piggyback.py

### The combine's exact fill: this layer's hot experts streamed in its own call (2026-11-13, deepseek_v41_flash_d_p P.55b)
- What: `set_combine_fill_exact(resources, enabled, gate_up, down, runs, jobs, limit_cycles)` (default: never called =
  the stock port writer; `set_combine_fill` / P.53's REC_FILL path unchanged). Enabled, routed_experts_combine builds
  the port writer with REC_FILL + REC_FX (`RoutedExpertsCombineParams.fill_exact`, hashed): every port core reads its
  jobs from a DRAM page of `jobs` (uint32, per chip COMBINE_FX_PAGES rows of 256 words: row dir x links + link = that
  port, row COMBINE_FX_CHIP_PAGE = the chip's waits), not from runtime args, so a device planner can write them. A job
  sends pieces [p0, p1) of a run table (home slot -> spare slot, or spare -> spare for a relay) to chip dst on this
  port's link, rows first (main loop: only in a round with no row, under `cap` packets in the channel; the rest after
  the main loop); every K pieces a flushed atomic increment of a flag word on dst (a per-block ready flag: the
  receiving router flushes the block's writes first); a gated job (relay) reads piece p only once its own chip's flag
  word counts (p - p0) / gK + 1 blocks. Flag words: 32 a call parity (the port's call counter) in
  `combine_fill_exact_words` (persistent, the collector cell, 448 words); the chip's first fill port zeroes the other
  parity at its start and is the waiter: it stamps each waited word's block arrivals (polled ~1 us), and after its own
  work waits for every waited word's target (bounded by limit_cycles: the job / wait gives up and flags the status,
  never a hang), then every port writes 16 stamp words (t_bar, last row, main-loop end, fill end, status, pieces,
  flags, exit; the waiter's land time and wait status). `combine_fill_links(resources)`; constants COMBINE_FX_*.
- Why: P.55 (owner 2026-11-13): the exact replicas copy this chunk's hot experts in the layer's own program, so the
  receivers compute the replica last; P.53's path copies another layer's slot over many calls.
- Measured: README "Measured" (exact fill).
- Cores: none added (the 8 port cores' RISCV_0 senders; the reads on NoC1); +3328 B of each port's arena end.
- Needed by: deepseek_v41_flash_d_p P.55b (MOE_REPLICAS=exact; tt/expert_exact.py builds the jobs on the host)
- Files: routed_experts_combine/{routed_experts_combine.{hpp,cpp},routed_experts_combine_nanobind.cpp},
  routed_experts_combine/device/{combine_program.{hpp,cpp},routed_experts_combine_device_operation{.cpp,_types.hpp},
  routed_experts_combine_program_factory.cpp}, routed_experts_combine/device/kernels/dataflow/combine_port_writer.cpp;
  tests: models/demos/deepseek_v41_flash_d_p/tests/test_moe_exact_copy.py

### The router's uneven replica split: the planner's share of each holder (2026-11-14, deepseek_v41_flash_d_p P.55c)
- What: `router_rows.hpp slot_of`: a copy word of the slot map may carry a lower bound, `slot | lb << 9` (slot < 512,
  lb 15 bits); when the candidate's first copy word has lb > 0, copy i takes the tokens whose 15-bit hash is in
  [lb_i, lb_{i+1}) and the own slot [0, lb_0). A first copy word below 512 (every map written before) keeps the even
  split bit for bit. Kernel only (JIT); prepare_fused_head_weights' check of the map at load is unchanged (it sees the
  empty map; later maps are host writes into the router bias).
- Why: P.55c (MOE_REPLICAS=exact): the planner sends a hot expert's tokens to its home and replicas in uneven shares
  (the receivers are light chips, each takes what brings it to the mean).
- Needed by: deepseek_v41_flash_d_p P.55c (tt/moe_block_main_adapter.exact_map_words)
- Files: fused_head/device/kernels/router_rows.hpp

### The exact copy's receivers: the replica last, read once its copy landed (2026-11-14, deepseek_v41_flash_d_p P.55c)
- What: with the exact fill (`set_combine_fill_exact`, REC_FX) routed_experts_combine builds the flat expert with
  `FlatRoutedExpertConfig.fx_spare` = experts per chip (1 + the spare slot's local index; default 0 = the source's
  program): every role's schedule (`expert_schedule.hpp`, `SE_FX_SPARE`) puts the spare slot last (after the
  biggest-first order) and never pins it; the gate/up weight reads (`gate_up_weight_read.hpp`, the x forwarder) and the
  down weight reader take `SE_FX_RT` (`fx_wait.hpp`, 6 args the combine writes every call: on, the collector's xy, the
  landed word, the call's token, the limit) and read the spare slot's weights only once the chip's landed word holds
  the call's token (a 4 B NoC read every ~1 us, bounded: past the limit they read anyway). The chip's waiter port
  writes the token (`noc_inline_dw_write`, exact word 448: `COMBINE_FX_WORDS` 448 -> 452) once every waited flag word
  reached its target (polled in the main loop, the drain and the fence too), immediately when nothing comes to the
  chip, and at the wait's limit. The token is a host counter, one a call (`set_fill`), the port's arg 9.
- Why: P.55c: the copy runs in the same program as the receiver's experts; without the wait a receiver computes the
  replica from its spare slot's old bytes.
- Cores: none added. Host: 6 runtime args a weight reader core, written every exact call.
- Needed by: deepseek_v41_flash_d_p P.55c (MOE_REPLICAS=exact, tt/moe_block_main_adapter.set_exact_jobs)
- Files: flat_routed_expert/device/{flat_program.hpp,flat_routed_expert_device_operation_types.hpp,
  flat_routed_expert_program_factory.cpp}, flat_routed_expert/device/kernels/{expert_schedule.hpp,fx_wait.hpp,
  gate_up_weight_read.hpp,gate_up_x_forwarder.cpp,down_weight_reader.cpp}, routed_experts_combine/device/
  {combine_program.{hpp,cpp},routed_experts_combine_program_factory.cpp}, routed_experts_combine/device/kernels/
  dataflow/combine_port_writer.cpp; tests: models/demos/deepseek_v41_flash_d_p/tests/test_moe_exact_replicas.py

### The exact fill's jobs name their own weight buffers (2026-11-14, deepseek_v41_flash_d_p P.56)
- What: an exact-fill job's words 6 / 7 (were 0, unused) are the copied layer's gate/up and down buffer addresses; 0
  keeps the call's (`set_combine_fill_exact`'s `gate_up` / `down`), so every existing jobs table (P.55b / P.55c, words
  6 / 7 zero) runs as before, bit for bit. The port writer's REC_FX `f_load` takes them per job (`f_wj`), its reads
  and remote writes use them.
- Why: MOE_REPLICAS=prev (P.56): a device planner writes the copies of several layers into the job pages of one
  carrier call (P.53's piggyback idea on REC_FX's device-readable pages, since the plan exists only on the device).
- Cores: none added. Host: none.
- Needed by: deepseek_v41_flash_d_p P.56 (tt/expert_prev.py, tt/kernels/moe_prev_planner.cpp)
- Files: routed_experts_combine/device/kernels/dataflow/combine_port_writer.cpp, README.md; tests:
  models/demos/deepseek_v41_flash_d_p/tests/test_moe_prev_replicas.py

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

### The exact fill's job ring read on the device; a cache hit writes no unchanged fill arg (2026-11-14, deepseek_v41_flash_d_p P.60)
- What: `CombineFillCall.exact_ring` / `exact_ordinal` / `exact_wait` (nanobind `set_combine_fill_exact(..., ring=0,
  ordinal=None, wait=True)`; defaults keep the behaviour before). ring > 0: the jobs tensor holds `ring` page sets and
  each REC_FX port reads word 0 of `ordinal` (one 64 B DRAM read, before its jobs page) and takes set `ordinal mod
  ring` (fill args words 10 / 11; the stamps keep the port's own page index). With a ring the factory keeps a
  signature of the fill args per program and a cache hit with the same ones writes nothing. wait False: the flat
  expert's readers get on = 0 and the call has no token (the waiter then writes no landed word). Not in the program
  hash (runtime args only): same program, same device work but the one ordinal read.
- Why: MOE_REPLICAS=prev (P.56) handed every call a different ring tensor and a fresh token from the host, and the
  per-call rewrite of every fill port's and weight reader's args on 32 chips cost ~350-470 us of host a combine call
  (P.57 measured the combine at 731 vs 402 us; P.60's timers put the extra in set_fill, not in the slot choice: a
  fixed ring tensor alone did not help). Also a Metal-trace blocker (a host value per call).
- Measured: see README Measured (P.60). Cores: none added. Host: -0.3 to -0.45 ms a combine call with the options.
- Needed by: deepseek_v41_flash_d_p P.60 (MOE_PREV_HOST=device: tt/expert_prev.py, tt/kernels/moe_prev_planner.cpp)
- Files: routed_experts_combine/device/{combine_program.hpp,combine_program.cpp,
  routed_experts_combine_program_factory.hpp,routed_experts_combine_program_factory.cpp,
  kernels/dataflow/combine_port_writer.cpp}, routed_experts_combine/routed_experts_combine_nanobind.cpp, README.md;
  tests: models/demos/deepseek_v41_flash_d_p/tests/test_moe_prev_replicas.py (legacy vs device, bit-identical, the
  host mirror), tests/test_moe_prev_host_probe.py (host us a call)

### Folded replica planner: count packets from the combine's ports, the planner as the head's watcher tail (2026-11-14, deepseek_v41_flash_d_p P.62b)
- What: (1) routed_experts_combine: `set_combine_fill_exact(..., prev_counts, prev_landing, prev_sem, prev_spc,
  prev_core)` (CombineFillCall fields; fill args 12..17 of the port senders; the fill signature includes them): under
  REC_FX each link-0 port, after its main loop and fill drain, sends the chip's counts row to the chips its first hop
  leads to (fused write + atomic increment, combine_port_writer.cpp), the chip's first port also into its own landing
  row with the tag G[0] + 1; the port keeps G[0] from the ordinal read. (2) fused_head: `tail=HeadTail(ct, l1)`,
  `tail_desc=` (FusedHeadParams.tail hashed, FusedHeadInputs.tail_desc as HeadSrc SRC_TAIL, patched per call; the
  watcher kernel gets W_PREV / W_PREV_A / W_PREV_CT and runs `kernels/prev_planner.hpp` after its work, its L1 the
  arena past the head's bytes; `fused_head_tail_core`); tail programs use their own patch store
  (FusedHeadSetup.patches_tail). (3) `fused_head/device/kernels/prev_planner.hpp`: the replica planner's body (moved
  from models/demos/deepseek_v41_flash_d_p/tt/kernels/moe_prev_planner.cpp ROLE 0, which now includes it), with the
  folded input (counts 0: own row from the landing rows, a plan only when its tag says the call is pending, a bounded
  count wait). Defaults (no prev_counts, no tail): the programs as before (the port's new code is skipped on arg 12 = 0).
- Why: P.62 measured prev's host over off as one 32-chip planner launch a MoE call (~90 us, DS41-263); the owner asked
  for no launch of its own. The head has no free core at the 110-core cap, so the planner rides the watcher's NCRISC.
- Needed by: deepseek_v41_flash_d_p P.62b (MOE_PLANNER=folded).
- Files: routed_experts_combine/device/combine_program.{hpp,cpp}, routed_experts_combine/device/
  routed_experts_combine_program_factory.cpp, routed_experts_combine/device/kernels/dataflow/combine_port_writer.cpp,
  routed_experts_combine/routed_experts_combine_nanobind.cpp, fused_head/fused_head.{hpp,cpp},
  fused_head/fused_head_nanobind.cpp, fused_head/device/head_program.{hpp,cpp},
  fused_head/device/fused_head_device_operation{.cpp,_types.hpp}, fused_head/device/kernels/prev_planner.hpp,
  fused_head/device/kernels/dataflow/xg_watcher.cpp, README.md.
- Tests: none added to tests/ (the head's tail needs a whole MoE block and a planner state: the model's
  tests/test_moe_prev_replicas.py checks it - plans, maps, pages bit-identical to the launch path, host mirror 0
  mismatches, outputs bit-identical); fork suites not run on the 8x4 (OV-25: task F.1).

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
  copies (Metal already packs a multi-device program once and copies it per device; DS41-263); the cached records
  remove most of it without a kernel change.
- Needed by: deepseek_v41_flash_d_p P.63 (MESH_PROGRAMS=on).
- Files: fused_head/fused_head.{hpp,cpp}, fused_head/fused_head_nanobind.cpp, fused_head/device/head_program.{hpp,cpp},
  fused_head/device/fused_head_device_operation{.cpp,_types.hpp}, flat_routed_expert/device/flat_program.hpp,
  flat_routed_expert/device/flat_routed_expert_program_factory.cpp, routed_experts_combine/routed_experts_combine.{hpp,
  cpp}, routed_experts_combine/routed_experts_combine_nanobind.cpp, routed_experts_combine/device/combine_program.{hpp,
  cpp}, routed_experts_combine/device/routed_experts_combine_program_factory.{hpp,cpp},
  routed_experts_combine/device/routed_experts_combine_device_operation_types.hpp, README.md.
- Tests: none added to tests/ (host-only, same values: the model's tests/test_moe_prev_replicas.py and the ladder check
  outputs; tests/test_mesh_program_probe.py in the model measures the host us a call); fork suites not run on the 8x4
  (OV-25: task F.1).
