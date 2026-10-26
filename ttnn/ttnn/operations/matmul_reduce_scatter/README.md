# matmul_reduce_scatter

`matmul_reduce_scatter(A, W, cluster_axis=..., scatter_dim=-2 | -1, topology=Linear | Ring, num_links=None)` computes
`P = sum_g A[g] @ W[g]` over the G devices of each `cluster_axis` group and leaves device p with its block of P
(`scatter_dim=-2`: rows `[p M/G, (p+1) M/G)`, `-1`: columns `[p N/G, (p+1) N/G)`), as one `generic_op` per call. Every
device's partial of block b travels the shorter way to b's owner (ring) or along the line, and lands straight in the L1
of the matmul core that owns the same output tiles on the next chip; no partial touches DRAM.

Inputs: A `(..., M, K)` (leading dims 1, bfloat16 or bfloat8_b), W `(K, N)` (bfloat16, bfloat8_b or bfloat4_b), both
TILE and interleaved in DRAM or L1; M, K, N multiples of 32 and the scattered extent a multiple of `32 G`. Output:
bfloat16, interleaved (default DRAM). `num_links=None` uses every link each hop of the group has. Every cross-device
add runs in fp32 DEST; the fidelity is the caller's (default HiFi2).

Files: `matmul_reduce_scatter.py` (API), `matmul_reduce_scatter_program_descriptor.py` (planner + descriptor),
`_eth_placement.py` (Ethernet-core maps), `kernels/` (five kernels and a shared header).

## Dataflow

Per chip the compute grid is split into a **transport strip** (logical row 0, else the last column) and a
`grid_m x grid_n` **matmul rectangle**. Each matmul core owns a `core_m x core_n` sub-block of every block.

| Core | RISC | Kernel | Role |
|---|---|---|---|
| matmul | NCRISC, NoC0 | `reader` | K-block windows of A and W by multicast; lands arrivals |
| matmul | TRISCs | `compute` | the entry's matmul into a fp32 hold slot, then its finish |
| matmul | BRISC, NoC1 | `writer` | start / relay: signals its port; fin: writes the output |
| port | NCRISC, NoC0 | `port_reader` | gathers finished sub-blocks of its cores into `cb_port_chunks` |
| port | BRISC, NoC1 | `port_sender` | sends each chunk as one fabric packet into the downstream core's zone slot |

Every matmul core walks the G blocks in **entry order** (one entry per block). Per entry:

1. **Load** (reader): `Kt / kb` K-block windows; A is multicast along n-lines (same `gm`), W along m-lines (same
   `gn`), each window split into up to two parts sent by different cores of the line (channels `a`, `a2`, `b`, `b2`,
   staggered senders). The cross-block operand (A for `scatter_dim=-1`, W for `-2`) is **resident** after entry 0 when
   it fits L1: its windows are re-credited without NoC traffic. A sender reads one step ahead while the current
   multicast drains. A receivers ack before the core's own DRAM reads; W receivers ack after A landed, so A and W
   multicasts never cross on the NoC.
2. **Matmul** (compute): subblocks of at most 8 fp32 tiles, accumulated in DEST over a super-block of 2 K-windows and
   packed (packer L1 accumulation) into one fp32 slot of `cb_partial_hold`.
3. **Finish** (compute): copy (chain start), hold + arrival (relay, or fin with one upstream), or hold + fwd + bwd
   arrivals (fin), packed bf16 into `cb_finished`. It is fused into the last super-block when the arrival has landed by
   then (UNPACK checks the CB credit without blocking and broadcasts the answer to MATH and PACK); otherwise the
   `DEST + hold` sum is packed fp32 over the hold and a separate pass adds the arrivals. Both paths run the same FPU
   ops on the same values, so the output never depends on arrival timing.
4. **Handoff** (writer): a start / relay sub-block is announced in its port's mailbox and popped when the port acks; a
   fin sub-block is written to the output row by row as compute packs it.

Ports: port `(d, l)` serves direction d and the matmul cores with `k mod L == l`. It walks **units** (an entry index
and the core halves sending in direction d), reads each core's sub-block in chunks of at most the fabric payload, and
sends every chunk to the same logical core on the downstream chip; the core's last chunk carries a fused increment of
that core's global arrival counter. The reader polls these counters (host thresholds: the arrival's position in the
upstream's send order) and credits the zone slot; it never blocks the loader on the fabric.

Between calls nothing is left armed: readers decrement the arrival counters by what arrived, ports zero their
mailbox slots and consume the ready word of the fence (each port tells its peer's opposite port that this chip has
started before sending into it).

## Planner

`run()` plans each call on the host (`_plan_call`), from the shapes, dtypes, G, links and the free L1:

- **Schedule** (`build_schedule`, `plan_transport`): `line` (Linear, or G = 2), `ring_unbalanced` (odd G) or
  `ring_balanced` (even G >= 4: the opposite block is split between the two core halves). Entries are sorted by
  descending distance to the block's owner, the own block last, so a relay always waits for an arrival produced at an
  earlier entry index upstream (deadlock freedom). On a balanced ring with a resident operand the two halves walk
  mirrored orders and split along the streamed operand's multicast lines, so each entry sends half a block each way.
  An entry whose arrival is produced at the same index upstream finishes behind the next entry's matmul (lag).
- **Grid** (`choose_grid`): strip, orientation and `core_m x core_n` minimizing the busiest core's tile-entries over
  the walk, then subblock area. With `balance_lines` (`scatter_dim=-1`) a block whose columns do not split evenly may
  use every n-line with widths that rotate per block (`balanced_line_columns`); the partition depends only on (line,
  block), so a partial always lands on the core that owns the same tiles.
- **K-block and regime** (`choose_k_block`, `super_block_plan`): resident with the largest `kb | Kt` that fits (>=
  `min(Kt, 8)`), else stream (the cross-block operand re-read every entry); then the 2-window super-block if its deeper
  windows fit. The footprint counts operand windows, hold slots, the staging slice and the zero tile.
- **Output chunks** (`chunk_candidates`, `chunk_cost_us`): when no single-chunk plan fits L1, the block is cut into
  `n_m x n_n` chunks and the whole walk runs once per chunk in the same program. Zone slots are reused across chunks
  (two chunks' worth at once): a port sends chunk c only after the downstream consumed chunk c - 2, which the
  downstream cores report to their port that forwards the credit upstream. The split is the cheapest candidate under a
  fitted cost model.

The global semaphores and the port mailbox are created once per mesh (`prepare_mesh`). The staging tensor (a sharded
L1 tensor with the same address on every chip, backing `cb_finished` and the arrival zones) is resized per call
(`staging="exact"`) or allocated per call and freed once enqueued (`staging="ephemeral"`).

## Circular buffers

| CB | Pages | Notes |
|---|---|---|
| 0 `cb_act_k_window` | resident A: `core_m Kt`; else depth x `core_m kb` | A dtype |
| 1 `cb_weight_k_window` | resident W: `Kt core_n`; else depth x `kb core_n` | W dtype; the weight prefetch target |
| 2 `cb_partial_hold` | `depth_hold P` (`P = core_m core_n`) | fp32; 2 slots only when an entry lags |
| 3 `cb_finished` | `2 P` | bf16, staging offset 0 (ports read it remotely) |
| 4 / 5 `cb_arrivals_fwd / _bwd` | zone slots x `P` | bf16, staging; written by the upstream port |
| 6 `cb_port_chunks` | `2 x 4` chunks | port cores |
| 7 `cb_zero` | 1 | srcB of the fused finish's `DEST += hold + 0` |

CBs 0, 2, 7 and 1 are placed in that order up from the L1 allocator base (`local_cb_layout`), so the W CB's address is
known before the program exists.

## Weight prefetch

`matmul_reduce_scatter_weight_prefetch(A shape, A dtype, W, ...)` plans the call ahead (with an unbounded L1 budget)
and returns a `WeightPrefetch`: per-device jobs that land W's first K-block windows exactly where this op's W CB will
be (`[l1_begin, l1_end)`), for the program just before it to run on an idle data-movement RISC
(`kernel_lib/l1_prefetch_dataflow.hpp`; the KDA gated norm's reader). One job per (window, n-line): a core of the line
reads the window into its slot and multicasts it along the line, one sender per line. The host op's program ends only
after every multicast landed, so no semaphore is needed.

Contract (on `matmul_reduce_scatter_weight_prefetch`): nothing is allocated for the region, so between the host op
and this call no other program may run on the mesh and no L1 buffer may be allocated in the region and written. The
host op checks that the region is free while it runs. This op checks what it can: the prefetch is for its weight
tensor and is used once (else ValueError); its plan is the one the jobs were made for (else it reads the windows
itself); and its reader credits the landed windows only if the W CB is at `l1_begin`. KDA runs the norm, frees L1 and
allocates the final carry in DRAM, then calls this op.

## Limits

- Sharded inputs or output: not supported (the readers address A and W through TensorAccessor).
- `2 L` ports must fit the transport strip; a balanced ring needs at least two matmul cores.
- A shape that misses L1 even split into 256 chunks raises NotImplementedError.
- `balance_lines` applies to `scatter_dim=-1`, unchunked plans, at most 16 n-lines.
