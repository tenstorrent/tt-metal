# sparse_dispatch_2d

`ttnn.experimental.moe_block.sparse_dispatch_2d(xbuf, route, resources, *, arena=None) -> xbuf`

The MoE block's dispatch: every chip's own rows go to the chips that hold one of their experts, and nowhere else.

## Inputs and result

- `xbuf` `[1, 1, N * S, W]` ROW_MAJOR, DRAM interleaved, per chip: chip `(r, c)`'s own S rows already at slots
  `(r * C + c) * S + t` (the MoE head writes them there). Filled in place: slot `home * S + t` lands on every chip
  the row's route word names; other slots are not touched.
- `route` `[1, 1, 1, RW * S]` uint32, per chip: a route word per own row. Up to 16 chips one word
  (`mask | order << 16 | link << 17`), beyond two (`mask`, `order | link << 1`). `order` 0 sends the row along its
  mesh row first (X), 1 along its column first (Y); `link` picks the fabric link of every hop.
- `resources`: `MoEBlockResources`. The op takes from it the mesh plan (neighbours, links, torus wrap), the port
  cores (the cells nearest their link's Ethernet core on NoC1), `plan_key()` and its persistent semaphore words (made
  at the first call).

## Program

One program for the whole mesh (merged from every chip's descriptor). Per direction (X+, X-, Y+, Y-) with a neighbour and per link, one **port core** next to the
link's Ethernet core runs two kernels:

- reader (RISCV_1, NoC0, `sparse_dispatch_2d_port_reader.cpp`): merges the port's own rows (those whose tree starts along
  its direction) with up to three input manifest rings (0: rows continuing in its direction; 1 / 2: rows turning
  into its axis that arrived moving + / -), reads each row from `xbuf` and hands rows plus 16-byte entries to the
  sender `group` at a time.
- sender (RISCV_0, NoC1, `sparse_dispatch_2d_port_writer.cpp`): fences with the neighbour's opposite port, writes each
  row one fabric hop into the same slot of the neighbour's `xbuf` (two rows a packet when `pack` is 2), and stages
  for the neighbour's ports the entries they need (continue / turn + / turn -), sent `manifest_batch` at a time
  with a fused write + counter increment. `group` is the block's L1 plan's (else 8), `pack` 2 when two rows fit the
  fabric payload (else 1), `manifest_batch` 16.

Protocol: 16-byte entries (word 0: `slot | hops left << 16 | generation << 20 |
order << 24`, word 1: the destination mask). END(g) means "no more rows that had travelled g hops"; a port emits
END(0) after its own rows and END(g) after every input delivered END(g - 1), and is done when END(GMAX) is out and
every input delivered END(GMAX), `GMAX = (C - 1) + (R - 1) - 1`. A link into a chip with no port to continue or
turn into carries a terminal stream (END(GMAX) only). So when the op ends on a chip, every row sent to that chip
has landed: it is a barrier.

The mesh constants are defines (`SPARSE_DISPATCH_R SPARSE_DISPATCH_C SPARSE_DISPATCH_S SPARSE_DISPATCH_GMAX
SPARSE_DISPATCH_RW SPARSE_DISPATCH_WRAP_X SPARSE_DISPATCH_WRAP_Y SPARSE_DISPATCH_XRING SPARSE_DISPATCH_YRING`);
everything else is compile-time args, and the per-core values (addresses, mesh coordinates, ring masks, fabric
connection) are runtime args that the kernels read from a DRAM args table (`common/args_table.hpp`) at their start.

### State and caching

| What | Where |
|---|---|
| Manifest rings (`manifest_layout(...).total_entries` x 16 B a port core) | a CB of the program (the first one, so at the CB base: the same address on every port core of every chip, which remote senders rely on). Contents need no zeroing (consumed by count). |
| Ring counters x 3 + fence ready word | `MoEBlockResources.persistent_words("sparse_dispatch_2d/ports/sems")`, 16 B apart on every port cell of every chip, zero between calls (each call returns them to zero). |
| Rows, metadata, plan, stage CBs | `2 group row_bytes`, `2 group 16`, `plan_bytes`, `3 x 2 x manifest_batch x 16`. |
| Args CB (`CB_ARGS`) | 256 B a port core: the reader's and sender's blocks of values, copied in from the DRAM args table. |

`compute_program_hash` covers the resources' `plan_key()`, `group`, `manifest_batch`, `pack` and the tensor specs; no
address. Every address in the runtime args is a tensor base (xbuf, route, the semaphore words); the blocks hold
them as fixups and the kernels add the addresses from their common args (`common/args_table.hpp`). A cache hit
rewrites those common args and re-points the arena-backed CBs (`write_program_addresses`), every time. The mesh plan,
port cores and semaphore words are computed once per resources object.

### Design notes

- No zero gate. The dispatch does not touch the combine arena: it is a CB of the experts + combine program, so no
  other program can zero it. `routed_experts_combine` zeroes its arena at its start on each core and orders that with
  its own cross-chip start barrier.
- The manifest is a CB (above), not a per-call L1 tensor; the semaphore words are one persistent tensor (the same
  16 B words every call).
- Port placement and links come from `MoEBlockResources` (`tt_fabric::get_forwarding_eth_core`). Placement changes
  only performance, never the result.
- `group` is one of 2, 4, 8 (`kGroups`): at least 2 so the sender can pair rows, and the cap of 8 is an
  empirical choice. `manifest_batch` is 16 (`kManifestBatch`), also empirical.
  Both have a TODO in `sparse_dispatch_2d_plan.hpp`.

## Tests

`tests/ttnn/unit_tests/operations/experimental/moe_block/sparse_dispatch_2d/`:

- `test_sparse_dispatch_2d_host.py` (no device): the wiring's invariants and the group / pack defaults.
- `test_sparse_dispatch_2d.py` (device, LoudBox 2x4 FABRIC_2D or Galaxy 8x4 FABRIC_2D_TORUS_XY): bit-exact against the
  torch model on random rows with uniform routing, and a cache hit on new buffers.
