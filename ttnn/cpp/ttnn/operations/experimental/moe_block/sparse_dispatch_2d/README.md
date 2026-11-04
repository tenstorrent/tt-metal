# sparse_dispatch_2d

`ttnn.experimental.moe_block.sparse_dispatch_2d(xbuf, route, resources, *, group=None, manifest_batch=16, pack=None,
debug=None, spin_limit=0) -> xbuf`

The MoE block's dispatch: every chip's own rows go to the chips that hold one of their experts, and nowhere else.
C++ port of `ttnn/ttnn/bringup/combine_rf/sparse_all_gather_2d.py` (mstaletovic/moe-trim) as
`tt_k3_token_split_moe.dispatch` runs it; the kernels and the wire protocol are the reference's.

## Inputs and result

- `xbuf` `[1, 1, N * S, W]` ROW_MAJOR, DRAM interleaved, per chip: chip `(r, c)`'s own S rows already at slots
  `(r * C + c) * S + t` (the fused head writes them there). Filled in place: slot `home * S + t` lands on every chip
  the row's route word names; other slots are not touched.
- `route` `[1, 1, 1, RW * S]` uint32, per chip: a route word per own row. Up to 16 chips one word
  (`mask | order << 16 | link << 17`), beyond two (`mask`, `order | link << 1`). `order` 0 sends the row along its
  mesh row first (X), 1 along its column first (Y); `link` picks the fabric link of every hop.
- `resources`: `MoEBlockResources`. The op takes from it the mesh plan (neighbours, links, torus wrap), the port
  cores (placement `config.combine_tuning.place`), `plan_key()` and its persistent semaphore words. Call
  `ensure_sparse_dispatch_2d_resources(resources)` before per-call allocations (and before `freeze()`).

## Program

One program per chip. Per direction (X+, X-, Y+, Y-) with a neighbour and per link, one **port core** next to the
link's Ethernet core runs two kernels:

- reader (RISCV_1, NoC0, `sparse_dispatch_2d_reader.hpp`): merges the port's own rows (those whose tree starts along
  its direction) with up to three input manifest rings (0: rows continuing in its direction; 1 / 2: rows turning
  into its axis that arrived moving + / -), reads each row from `xbuf` and hands rows plus 16-byte entries to the
  sender `group` at a time.
- sender (RISCV_0, NoC1, `sparse_dispatch_2d_sender.hpp`): fences with the neighbour's opposite port, writes each
  row one fabric hop into the same slot of the neighbour's `xbuf` (two rows a packet when `pack` is 2), and stages
  for the neighbour's ports the entries they need (continue / turn + / turn -), sent `manifest_batch` at a time
  with a fused write + counter increment.

Protocol (unchanged from the reference): 16-byte entries (word 0: `slot | hops left << 16 | generation << 20 |
order << 24`, word 1: the destination mask). END(g) means "no more rows that had travelled g hops"; a port emits
END(0) after its own rows and END(g) after every input delivered END(g - 1), and is done when END(GMAX) is out and
every input delivered END(GMAX), `GMAX = (C - 1) + (R - 1) - 1`. A link into a chip with no port to continue or
turn into carries a terminal stream (END(GMAX) only). So when the op ends on a chip, every row sent to that chip
has landed: it is a barrier, as the reference was.

The reference's text substitutions are defines (`SA_R SA_C SA_S SA_GMAX SA_RW SA_WRAP_X SA_WRAP_Y SA_XRING
SA_YRING`); everything else is compile-time args.

### State and caching

| What | Where |
|---|---|
| Manifest rings (`manifest_layout(...).total_entries` x 16 B a port core) | a CB of the program (the first one, so at the CB base: the same address on every port core of every chip, which remote senders rely on). The reference allocated an L1 tensor per call. Contents need no zeroing (consumed by count). |
| Ring counters x 3 + fence ready word | `MoEBlockResources.persistent_words("sparse_dispatch_2d/ports/<placement>/sems")`, 16 B apart on every port cell of every chip, zero between calls (each call returns them to zero). |
| Rows, metadata, plan, stage, debug CBs | as the reference (`2 group row_bytes`, `2 group 16`, `plan_bytes`, `3 x 2 x manifest_batch x 16`, 64). |

`compute_program_hash` covers the resources' `plan_key()`, the placement, `group`, `manifest_batch`, `pack`,
`spin_limit` and the tensor specs; no address. Every address in the runtime args is a tensor base (xbuf, route, the
semaphore words, debug); the mesh workload factory records where each sits (`common/address_patches.hpp`), and a
cache hit rewrites a tensor's arguments only when its address moved - with nothing moved, one runtime-arg lookup per
tensor a chip. The mesh plan, port cores and semaphore words are computed once per resources object.

### Differences from the reference

- No zero gate. In the Python block the dispatch also ran the combine arena's zeroing kernels (`l1_zero_kernels`,
  `SA_ZGATE_*`: the senders held their END markers until every zeroing kernel of the chip had counted in, making the
  dispatch the cross-chip barrier between the zeroing and any combine write). With C++ ops the combine arena is a
  CB of the experts + combine program, so no other program can zero it. `routed_experts_combine` zeroes its arena at
  its start on each core and orders it with its own cross-chip start barrier (`bar`), as the reference already did
  with `barrier=dict(tensor=bar)`.
- The manifest is a CB (above) instead of a per-call L1 tensor; the semaphore words are one persistent tensor
  instead of four global semaphores (the same 16 B words, same use).
- Port placement and links come from `MoEBlockResources` (`tt_fabric::get_forwarding_eth_core`), not the Ethernet
  probe program. Placement changes only performance, never the result.

## As a phase of another program

`routed_experts_combine` with `experts_during_dispatch` runs the dispatch as the first phase of its port kernels
(the reference's `dispatch_phase`). The kernel headers are written for that:

- `sa::reader_args(a, man_addr)` / `sa::read_phase<CT0>(args)`: reads its runtime args from `a` (advancing it), its
  compile-time args from `CT0` (`cb_rows, cb_meta, cb_plan, row_bytes, group, plan_bytes, cb_dbg`, then the
  accessors of `xbuf`, `route`, debug). Returns false only on a debug bail-out. The host kernel then counts in on the
  x relays (the reference's `XREADY_N` step).
- `sa::sender_args(a, man_addr)` / `sa::send_phase<CT0>(args, conn)`: on the host kernel's open fabric connection.
  Compile-time args from `CT0`: `cb_rows, cb_meta, cb_stage, row_bytes, group, manifest_batch, pack`, then the
  `xbuf` accessor.
- The host side needs the same defines (`SA_*`), runtime args in the order of the factory here, the semaphore words
  layout (3 counters + ready, 16 B apart, at one address on every port core) and a manifest at one address on every
  port core. `device/sparse_dispatch_2d_plan.hpp` (`wiring`, `cb_sizes`, `port_index`, defaults) is the host math.

## Tests

`tests/ttnn/unit_tests/operations/experimental/moe_block/sparse_dispatch_2d/`:

- `test_sparse_dispatch_2d_host.py` (no device): the wiring's invariants and the group / pack defaults.
- `test_sparse_dispatch_2d.py` (device, LoudBox 2x4 FABRIC_2D or Galaxy 8x4 FABRIC_2D_TORUS_XY): bit-exact against the
  torch model on random rows with uniform routing, a cache hit on new buffers, and
  bit-exact against saved reference cases when `MOE_BLOCK_REFS` is set.
