# moe_head

The MoE block's head as one program for the mesh: the x all-gather over TP straight into the shared expert's
matmul cores, the shared expert's down + reduce-scatter, the router (logits, top-K, the joint routing rows), the
routing exchange + the combine planner (into the exchange's plan tensors), the dispatch prep and, with a latent, the
latent down + all-to-all.

## Pieces

| Piece | Where |
|---|---|
| Layout planner and its sizing | `device/head_plan.*`, `device/head_sizing.*` |
| Block-level plan (`model_checks`, the shape, `plan_moe_head`) cached in `MoEBlockResources` | `moe_head.*` |
| Kernels | `device/kernels/` |
| Host model + the program (per-chip args in DRAM tables) | `device/head_program.*` |
| Device operation (WorkloadDescriptor factory, address-only cache hits), host entry, weights | `device/moe_head_device_operation.*`, `moe_head.*` |

## The device operation

`ttnn.experimental.moe_block.moe_head(x, weights, resources=, dtypes=, accumulators=)` runs one program for the mesh
and returns `MoEHeadOutputs` (shared_expert_output, routing_rows, dispatch_buffer, route_words and the combine
plan: counts, regions, token_index, table, plan - the resources' tensors). `prepare_moe_head_weights` lays out the
shared expert's and the router's weights in the head's layouts (`prepare_head_weights`) from host tensors. The first call plans the head, builds the host model and
creates the persistent words in the resources (`moe_head/...`); later calls only launch.

Design notes:
- the arena is an L1 tensor given to the op (`arena=`; by default one the resources keep, which holds that L1 for
  good - a block passes one arena it shares between its ops), not one allocated each call: the CBs over its regions
  bind to it at their offsets, and the kernels take its base plus the region's offset. No holes or settling;
- the persistent words (mailboxes, flags, counters) are word tensors of the resources;
- every address argument is recorded with its source (a tensor, the arena, a word tensor) and offset. Every kernel's
  runtime and common args then move to per-chip DRAM tables (`head_args_to_dram`, common/args_table.hpp): the table
  holds an address arg's offset, the kernel adds its source's address (a common arg) when it loads its block into the
  arena's args region (`kHeadArgsBytes` past the regions); a compute kernel gets its block as a CB page from its
  core's BRISC. The chips' programs are then alike: one program for the mesh. A cache hit (override_runtime_arguments,
  once) rewrites, for a source whose address moved, only that source's common arg in each kernel (and, for the arena,
  its CBs): no descriptor is rebuilt, nothing per chip or per core is written. The program hash holds no address;
- addresses are common runtime args, read once by each kernel into its args struct (`RouterCommonArgs`, the shared
  expert dataflow kernels' phase words address), never defines;
- fabric connections are main's (`append_fabric_connection_rt_args`; the kernels open them from their block with
  `build_from_eth_channel`); the kernel_lib multicast families use main's
  `Mcast::attach`;
- the routing exchange's and the planner's ports follow the fabric's own first hop between every pair of chips
  (`MoEBlockResources::first_hops`, asked of the fabric's routing tables), and the gather's ring ties follow the
  fabric too: nothing assumes how the fabric routes;
- the routing exchange's Y hello (routing_exchange_y_sender) waits for the hellos that actually land on the chip (one more runtime
  arg, after the fabric connection's: `HeadModel::y_hellos_in`), not for one: on an unwrapped SP axis (a Y line,
  e.g. FABRIC_2D 8x4 or a 4x4 tray half) every row says hello Y+ except the far end, so the near end gets none (a
  wait for one would hang there) and the row before the far end two. A ring or 2 rows: one each.

## Padded chunks

`moe_head(..., actual_start=s, actual_end=e)`: main's chunked-prefill metadata scalars as they are (one-element uint32
DRAM tensors, replicated: the chunk's first real token's absolute position, one past its last). The chunk is in main's
KV-pad-aware rotated layout over the SP rows (`deepseek_prefill/moe_padding_config`): an SP row's real rows are a
prefix of its rows, their count main's closed form (boundary slab / row / offset, two runs on the boundary row); TP chip
`p` owns the row's tokens `p S ..`, so its real tokens are that count less `p S`, clamped to `[0, S]`. With `s` chunk
aligned it is the natural order: the first `e - s` tokens real. The router kernels read both on device every call
(`PADDED_CHUNK`, common args 2..11), so a trace replays with the values of the moment. A chip's padding tokens get no
routes: the planner (`routing_planner::plan`) counts, stages and plans only the real ones (counts, regions, token index, table, plan
hold no padding row) and the route words of the rest are zero (no dispatch). The combine's accumulators write zeros for
a token with no rows. The shared expert still computes every row (its padding rows are discarded downstream). Without
the scalars the programs are the unpadded ones; the padded programs keep their own address patches
(`MoEHeadSetup.patches_padded`).

## Row blocks

A call whose rows do not fit the head's L1 at once runs as k row blocks: one launch whose kernels loop over k blocks of
the call's rows (`device/kernels/row_block.hpp`: every kernel's `kernel_main` runs `row_block::begin` / its work /
`row_block::end` in a plain `for` loop; k is each kernel's named compile-time arg `row_blocks`). The head's planner
picks k (`plan_moe_head`, `HeadFit.row_blocks`): 1 wherever the rows fit, else the count whose plan fits with the least
predicted time, among the counts that give every block whole tile rows and whole pages of every DRAM bank of x and the
shared expert's output (the kernels step a block's stride a bank). Tests can force it
(`MoEBlockConfig.force_head_row_blocks`, test only).

The dispatch, the routed experts and the combine run once on the call's rows whatever k is, so every expert's weights
are read once. The owner-order outputs are in the block layout: block b's owned tokens after block b - 1's (a chip owns
`S` tokens a block - `p S ..` of the block's `S C` rows of its SP row), in the routing rows, route words, dispatch rows
(`b S` on), the combine's rows and the shared expert's output; `MoEHeadOutputs.row_blocks` carries k to the tail, which
maps the owner rows back to the residual in the same layout (`moe_tail(row_blocks=)`).

What moves from one block to the next - x / y (`ROW_BLOCK_X_STRIDE_BYTES` / `ROW_BLOCK_Y_STRIDE_BYTES` bytes a bank),
the dispatch rows (`ROW_BLOCK_DISPATCH_ROW_STRIDE`), the padded span's first row (`ROW_BLOCK_PADDED_ROW_STRIDE`, block
b's first SP row = b S C), the router's owner page and route words (`ROW_BLOCK_ROUTER_PAGE_STRIDE` /
`ROW_BLOCK_ROUTE_WORDS_STRIDE`) - each kernel adds `block x delta` itself. An earlier block's routing planner plans no
token (`ROUTING_PLANNER_IDLE`: the exchange still runs empty); the last block's plans every block's tokens and overlays
the router's dead buffers up to the bias (`router_layout(arr_first)`); the planner picks the earlier blocks' layout or
the last's (`ROUTING_PLANNER_PER_BLOCK`). A padded chunk's planner counts every block's real tokens
(`padded_real_owned`: a prefix of the owned order `b S + t`). Heads with blocks also take planner-only cores
(`dataflow/routing_planner_core.cpp`, up to `kPlannerCores`) so the planner does not share a router core's L1 with the
router's buffers. The host builds block 0's program; a cache miss also builds block 1's and the last block's and checks
that every argument that differs by the block is one of those deltas (`row_block_detail::check_row_block_args`). The
CBs carry on from block to block as from one row chunk to the next, and between two blocks the chip's cores pass a
barrier with public APIs only (`row_block::end`: done words, a coordinator, the program semaphores reset to 0 by each
core's lead).

Splits (`MoEBlockConfig.chunk_splits`, MoEBlock's choice when the chunk does not fit in one call): the whole block runs
the chunk as pieces, each piece one head call (`moe_head(..., split=j)`: x the piece's rows; a padded chunk's real rows
offset by the piece).

`MoEBlockConfig.pad_tokens`: a chunk whose rows a mesh row are not a multiple of 32 x TP runs with each SP row padded
up to it (`MoEBlock` pads x and drops those rows of the output); the padding rows are a padded chunk's padding tokens
(the head always takes actual_start / actual_end then: `(0, chunk)` when the caller has none; common arg 6 is the
caller's SP row, `HeadShape.row_tokens`).

## Host plan

The planner returns a `HeadFit`: `ok`, the planned layouts (best first, each with its arena bytes, region sizes,
predicted us and the cost's parts), `reason` (the requirement that failed, with its numbers and the largest rows
that fit), `unconstrained` (the cost model's pick with unlimited L1) and `note` (what L1 forced; a chunk no layout fits runs as row blocks).
`plan_moe_head(resources, dtypes, A, l1_free)` builds the shape from the block's dims / config / topology / compute
grid / fabric payload and caches the answer in the resources, so a program-cache hit never re-plans.

`l1_free` is the free L1 a core the head plans on: by default measured once at the block's first head (the L1 under
the lowest occupied address, less the planner's margin `kL1Margin`); `kL1Free - kL1Margin` only for host-only plans
(no device).

## Kernels

The kernels, all under `device/kernels/` (one program a chip; a builder a core role builds every kernel's compile-time
args, defines and runtime / common args next to the host model's value behind each: head_engine_program.cpp for the
matmul cores, head_gather_program.cpp for the port cores and the watcher, head_router_program.cpp for the router and
planner cores):

| Kernel | Cores | RISC, NoC | Responsibility |
|---|---|---|---|
| `dataflow/shared_expert_row_dataflow.cpp` | matmul cores | RISCV_0 (BRISC), NoC 0 | loads this core's steps of the chip's x shard from DRAM into x-own L1; multicasts x steps, the router's x (row lead) and h along the row; hands the latent and the reduce-scatter sub-blocks to the port cores |
| `dataflow/shared_expert_column_dataflow.cpp` | matmul cores | RISCV_1 (NCRISC), NoC 1 | reads W_r^T (column lead) and its W blocks from DRAM and multicasts them down the column; sends the partial logits to the router cores / ports; feeds the reduce-scatter arrivals to the compute and writes y; writes the dispatch prep's rows to DRAM |
| `compute/shared_expert_compute.cpp` | matmul cores | compute | gate / up, the GLU, down, the router's partial logits, the dispatch prep's untilize (one kernel a column group) |
| `dataflow/fabric_port_stager.cpp` | port cores (a direction and link each) | RISCV_0 (BRISC), NoC 1 | stages what the port sends into CB 0 from the matmul cores' L1: the chip's own and the relayed x shards, the reduce-scatter sub-blocks (FINB), the latent rows |
| `dataflow/fabric_port_sender.cpp` | port cores | RISCV_1 (NCRISC), NoC 0 | sends CB 0 over the fabric (x into the neighbour's matmul cores, reduce-scatter zones, latent rows), the router's logits partials and the routing exchange's X packets; fabric fences and credits |
| `dataflow/x_gather_watcher.cpp` | the watcher core | RISCV_1 (NCRISC), NoC 1 | counts the fabric-landed shards and raises XREADY on every matmul core; with a second Y direction, that direction's routing exchange send |
| `dataflow/routing_exchange_y_sender.cpp` | the watcher core | RISCV_0 (BRISC), NoC 0 | the routing exchange's first Y direction (fabric writes into the destination chips' DRAM slots) and its Y hello |
| `dataflow/router_input_feeder.cpp` | router cores | RISCV_1 (NCRISC), NoC 0 | reads the bias and expert-index tiles from DRAM; hands the compute the HOLD-ready page and the other chips' logits partials in a fixed order; builds tokens 8 .. 15 of the joint routing rows |
| `compute/router_compute.cpp` | router cores | compute | the logits' sum, the scores, top-K, the route words |
| `dataflow/router_output_writer.cpp` | router cores | RISCV_0 (BRISC), NoC 1 | builds tokens 0 .. 7 of the joint routing rows, writes the routing rows and route words to DRAM, the router iterations' go words; runs the core's routing planner instance |
| `dataflow/routing_planner_core.cpp` | planner-only cores (row blocks) | RISCV_0 (BRISC), NoC 1; RISCV_1 (NCRISC), NoC 0 | waits for every router core's rows, then the routing planner (the NCRISC instance gets no region today) |

In this op the side that reads into a core runs on BRISC (RISCV_0) and the side that sends out of it on NCRISC
(RISCV_1) - the opposite of tt-metal's Reader / Writer config convention - and the NoCs are pinned per kernel: row and
column multicasts go on different NoCs (on one NoC they stall each other), and DRAM that the fabric wrote is read on
NoC 1, the fabric routers' NoC (a read through the other NoC's DRAM endpoint can return stale data). Each
data-movement kernel's header comment states its cores, RISC, NoC and responsibilities.

Shared pieces are headers, one a component:
- `x_gather.hpp`: the x all-gather's compile-time args, its walk, the destination tables and the live row chunks;
- `reduce_scatter.hpp` and `latent_all_to_all.hpp`: the port stager's and the port sender's halves of each;
- `router_fabric_sends.hpp`: the router's logits stream and the routing exchange's send;
- `router_common.hpp`: the router's routing rows, route words and common args;
- `routing_planner.hpp`: the routing tensors' accessors, a padded chunk's metadata read and the planner
  (`routing_planner::read_planner_args` / `routing_planner::plan(args, region_base)`; stubs under
  `ROUTING_PLANNER_NONE`);
- `shared_expert.hpp`: `global_row_tile`; the compute kernels' matmuls name their fidelity at the call site
  (`matmul_block_init_at<fidelity>` / `matmul_block_at<fidelity>`, common/kernels/matmul_fidelity.hpp - the tail's
  engine uses them too);
- `padded_rows.hpp` (a padded forward's real rows) and `row_block.hpp` (every kernel's `kernel_main` runs
  `row_block::begin` / its work / `row_block::end` in a plain `for` loop).

Where the args live (head_workload.cpp `head_args_to_dram`): every kernel's runtime and common args move into this
core's block of a per-chip DRAM table. Each kernel reads them once in `kernel_main`, after `moe_block_args::load()`,
into a named struct of its role (`read_*_args`), at the indices the host laid them out at; a variable-length table
stays in place as a view (`moe_block_args::arg_pointer` / `common_pointer`, common/kernels/moe_block_args.hpp) and the
helpers take the struct, never an index into the arg list. A compute
kernel gets its block as one page of CB 31 from its core's RISCV_0 kernel. What stays a real runtime / common arg: the
converter's table args, the row blocks' table (`ROW_BLOCK_BARRIER_ARG`) and the shared expert BRISC's multicast args
(the kernel_lib multicast families "x0", "x1" and "h", attached with main's `Mcast::attach`).

Defines are plain tokens (numbers, `true` / `false`); the list defines (`ROUTE_WORDS_*_INIT`) are comma-separated.
Every define is structural (no buffer address), so the program hash may include them: addresses
are common runtime args, read into the kernels' args structs. A
value that differs by chip (`ROUTING_EXCHANGE_MY_CHIP`, `ROUTING_PLANNER_MY_CHIP`, `ROUTING_PLANNER_PORTS`,
`ROUTING_PLANNER_NUM_PORTS`, `ROUTER_MY_CHIP`, `ROUTER_TP_POSITION`, `WATCHER_ROUTING_ARGS_OFFSET`) becomes a common
arg of the chip's block (`ChipProgram::chip_values`), so one program serves the mesh.

The kernels name the model's and the layout's values as the host does (`row_tiles_per_core`, `k_step_tiles`,
`row_chunks`, `engine_columns` / `engine_rows`, `gate_up_passes`, `down_passes`, `expert_tiles`, `top_k`,
`router_iterations`, ...), each documented where it is declared. The shared expert's activation is
common/kernels/expert_activation.hpp, shared with the routed experts: the compute kernel's named compile-time arg
`expert_activation` (its `expert_activation::Kind`) and its constants as named compile-time args holding fp32 bit
patterns (`expert_activation_*_bits`; common/moe_block_config.hpp `expert_activation_code` /
`expert_activation_constant_args`). Tile sizes come from `common/kernels/tile_bytes.hpp` (`moe_block_tiles`, on the
host and in the kernels) and `tt::constants`. A compute kernel calls `compute_kernel_hw_startup` once in `kernel_main`;
everything after it reconfigures with the specific `*_init` / `reconfig_data_format` calls.

## Tests

- `tests/ttnn/unit_tests/operations/experimental/moe_block/test_moe_head_lb.py` (a LoudBox, SP 2 x TP 4; GLM-5.3 and Kimi-K3): accuracy vs torch,
  determinism and traced device time.
