# fused_head

The MoE block's head as one program a chip: the x all-gather over TP straight into the shared expert's
matmul cores, the shared expert's down + reduce-scatter, the router (logits, top-K, the joint routing rows), the
routing exchange + the combine planner (into the exchange's plan tensors), the dispatch prep and, with a latent, the
latent down + all-to-all.

## Pieces

| Piece | Where |
|---|---|
| Layout planner and its sizing | `device/head_plan.*`, `device/head_sizing.*` |
| Block-level plan (`model_checks`, the shape, `plan_fused_head`) cached in `MoEBlockResources` | `fused_head.*` |
| Kernels | `device/kernels/` |
| Host model + per-chip programs | `device/head_program.*` |
| Device operation (WorkloadDescriptor factory, address-only cache hits), host entry, weights | `device/fused_head_device_operation.*`, `fused_head.*` |

## The device operation

`ttnn.experimental.moe_block.fused_head(x, weights, resources=, dtypes=, accumulators=)` runs one program a chip and
returns `FusedHeadOutputs` (se, rows, xbuf, route, latent, and the combine plan: counts, regions, token_index, table,
plan - the resources' tensors). `prepare_fused_head_weights` lays out the shared expert's and the router's weights
(SeEngine3.upload, RouterStream.upload) from host tensors. The first call plans the head, builds the host model and
creates the persistent words in the resources (`fused_head/...`); later calls only launch.

Differences from the reference, none in what the kernels compute or in their NoC / fabric traffic:
- the arena is an L1 tensor given to the op (`arena=`; by default one the resources keep, which holds that L1 for
  good - a block passes one arena it shares between its ops), not one allocated each call: the CBs over its regions
  bind to it at their offsets, and the kernels take its base plus the region's offset. No holes or settling;
- the persistent words (mailboxes, flags, counters, the reference's global semaphores) are word tensors of the
  resources;
- every address argument is recorded with its source (a tensor, the arena, a word tensor) and offset. A cache hit
  (override_runtime_arguments) reads one argument a source, and only for a source whose address moved rewrites its
  arguments (and, for the arena, its CBs): no descriptor is rebuilt, nothing is rewritten when nothing moved. The
  program hash holds no address;
- the addresses the reference baked into defines (SE_EW, RT_ENG, RT_MSYNC, RT_PWORD, RT_PART) are common runtime args;
- fabric connections are main's (`append_fabric_connection_rt_args`); the kernel_lib multicast families use main's
  `Mcast::attach`; their semaphore ids differ from the reference's;
- the routing planner splits torus ties by the source's parity (the reference's RAP_YT 2, now the only rule: no
  RAP_YT define) and the gather's ring ties follow the fabric, as main's routing tables (parity since #63501); the
  references for the tests are dumped with the reference tree's default (parity).
- the routing exchange's Y hello (ra_y_send) waits for the hellos that actually land on the chip (one more runtime
  arg, after the fabric connection's: `HeadModel::y_hellos_in`), not for one: on an unwrapped SP axis (a Y line,
  e.g. FABRIC_2D 8x4 or a 4x4 tray half) every row says hello Y+ except the far end, so the near end gets none (the
  reference waits there forever) and the row before the far end two. A ring or 2 rows: one each, as the reference.

## Host plan

The planner returns a `HeadFit`: `ok`, the planned layouts (best first, each with its arena bytes, region sizes,
predicted us and the cost's parts), `why` (the requirement that failed, with its numbers and the largest rows that
fit), `free` (the cost model's pick with unlimited L1) and `note` (what L1 forced, or the DRAM fallback's cost).
`plan_fused_head(resources, dtypes, A, l1_free)` builds the shape from the block's dims / config / topology / compute
grid / fabric payload and caches the answer in the resources, so a program-cache hit never re-plans.

`l1_free` is the free L1 a core the head plans on: by default measured once at the block's first head (the L1 under
the lowest occupied address, less what the block keeps after planning), at most the planner's host figure
(`kL1Free - kL1Margin`).

## Kernels

`device/kernels/head_kernel_contract.hpp` lists every kernel (RISC, NoC, compile-time args, defines and the host
model's value behind each, runtime / common-arg layouts, the kernel_lib multicast families). The kernel files are the
source:

- shared pieces (the x all-gather's compile-time args / walk / latent and reduce-scatter reader and sender, the
  router's rows and L stream, the routing exchange's send, the dispatch prep's route words, the routing planner) are
  headers (`ra_plan.hpp`: `ra::plan(a0, B)`);
- the planner's and the route words' parameters are `RAP_X` / `RW_X` defines; the `GROW()` macro is `se_defs.hpp`;
- addresses (`SE_EW`, `RT_ENG`, `RT_MSYNC`, `RT_PWORD`, `RT_PART`) are common runtime args, so no define carries an
  address.

## Tests

- `test_fused_head.py` (a Galaxy, `MOE_BLOCK_REFS`): bit-exact against saved reference cases, and every output
  bit-identical call to call.
