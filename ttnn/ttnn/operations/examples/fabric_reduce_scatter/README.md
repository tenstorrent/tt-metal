# fabric_reduce_scatter — a line / ring reduce-scatter over the fabric, and what sets its speed at each size

**Difficulty:** ⭐⭐⭐ T3  ·  **Concept(s):** balancing a ring's two directions for a reduction · spreading the relay's DRAM reads over cores (workers sharing a link through a fabric mux, final cores per link)
**First profiled on:** `qb2-120-p02t03` · BH (QuietBox 2, 2× p300c, 4 chips) · FABRIC_2D_TORUS_XY · 2026-10-13 · `316bf3de8c4`+

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
In a reduce-scatter each chip holds a full-size partial result and ends with one block of the sum. Every hop adds
one chip's partial to a running sum, so a relaying chip reads two streams from DRAM per chunk (its own partial and
the sum that arrived), adds them and sends the result on. That makes a relay core's DRAM reads, not the link, the
first limit at large sizes; at a few tiles per chip the fixed cost of a call and the number of serial hops dominate.
This example is a complete line / ring reduce-scatter (any mesh axis or a snake over the mesh, one or more links,
one or more workers per link) measured from 4 tiles to 16 MiB per chip.

## What this isolates — and how
- **Concepts:** (1) the ring schedule: which block's partials travel which way, and splitting the far block between
  the two directions; (2) where the relay's reads run: on one core per link, on several workers that share the link
  through a fabric mux, and on how many final cores.
- **Setup:** DRAM → fabric → DRAM, bf16 adds on the Tensix FPU. The input `[1, 1, G·Sb, W]` holds a partial of all G
  blocks; chip p's output `[1, 1, Sb, W]` is block p of the sum.
  - **Port (one per link and direction, or W workers):** its reader (NCRISC, NoC0) reads its own partial and, for a
    relayed block, the arrived sum from a DRAM scratch buffer. Compute adds them; its sender (BRISC, NoC1) writes the
    sum into the same pages of the downstream chip's scratch. A block whose chain starts at this chip goes from the
    input straight to the sender, without an add.
  - **Final cores (F per link, each owning 1/F of the link's DRAM banks):** they read their own partial plus the
    arrivals from both neighbours, add them and write block p.
  - A chunk is a run of up to 7 tiles (one packet at a 14,336 B payload) consecutive in one DRAM bank; links (and
    workers) split the blocks by bank.
- **Schedule:** line: toward p+1 a chip sends blocks G−1, …, p+1 (farthest first), toward p−1 blocks 0, …, p−1.
  Ring: toward p+1 the G/2 blocks p+G/2, …, p+1, toward p−1 the rest; each block is summed along the shorter way to
  its owner.
- **Why it's kernel-level:** the schedule, the split, the number of reading cores per link and how they reach the
  link are all decisions of the kernel and program author.

## The methods being compared
Each is varied alone from the baseline: 4-chip ring, balanced, 1 link, 1 worker per link and direction, 2 final cores
per link, increment every 8 chunks, 112 KiB circular buffer, 14,336 B router payload.

| Setting | What it does | Why it should differ |
|---|---|---|
| `topology`: line *(baseline: ring)* | A line sums toward each end; a ring uses both directions of every link | Line: the busiest link carries G − 1 blocks, and the farthest partial crosses G − 1 hops in series |
| `balance=False` *(baseline: True)* | On an even ring the block opposite each chip goes one way | Busiest link carries G/2 blocks instead of G/2 − ½ |
| `num_links=2` | Each link carries half the DRAM banks | Twice the link bandwidth, half the reads per core |
| `workers` 2 / 4 *(baseline 1)* | W cores per link and direction, each on 1/W of the banks, sharing the link through a fabric mux | W cores read the two relay streams in parallel; the mux adds a hop and a connection per call |
| `finals` 1 / 4 *(baseline 2)* | Final cores per link, each on 1/F of the link's banks | Each reads three streams per chunk; more cores read in parallel |
| `inc_every` 2 / 4 / 16 / 32 *(baseline 8)* | Increment the receiver's arrival counter on every N-th chunk (and each stream's last) | The receiving router stalls until each increment's write has landed; a relay waits up to N chunks behind |
| `cb_bytes` 56 / 224 KiB *(baseline 112)* | 2 / 8 chunks per batch (4 at 112) | Batches amortize the waits; a batch a short block can't fill waits longer |
| router payload 4,352 B *(baseline 14,336)* | Set once for the whole fabric, not per op | 3.3× more packets for the same bytes |

## CLI — measure your own shapes/params
Needs 4 chips.

```bash
python -m ttnn.operations.examples.fabric_reduce_scatter [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--fabric` | comma list | `2d_torus_xy` | `1d`, `1d_ring`, `1d_neighbor_exchange`, `2d`, `2d_torus_x`, `2d_torus_y`, `2d_torus_xy` |
| `--topology` | comma list | all 8 | `2x2_axis0_line`, `2x2_axis1_line`, `2x2_snake_line`, `2x2_snake_ring`, `4x1_line`, `4x1_ring`, `1x4_line`, `1x4_ring` |
| `--links` | comma list | `1,2` | links per hop (up to what the fabric reports) |
| `--workers` | comma list | `1` | cores per link and direction; above 1 they share the link through a fabric mux |
| `--finals` | comma list | `2` | final cores per link (with 1 worker; with W workers there is one per worker) |
| `--balance` | `0` or `1` | `1` | balanced ring |
| `--shape` | `H,W` or `H,W;H,W;…` | `2048,4096` | per-chip input (tile-aligned; H a multiple of 32·G) |
| `--inc-every` | comma list | `8` | arrival increment every N chunks |
| `--cb-kb` | comma list | `112` | circular buffer per core, KiB |
| `--payload` | int (bytes) | `14336` | router max payload, applied when the fabric opens |
| `--trials` | int | `5` | measured calls per case (median) |

Every comma list is crossed with the others. Test environment variables (`test_fabric_reduce_scatter.py`):
`RS_FABRICS`, `RS_TOPOS`, `RS_LINKS`, `RS_WORKERS`, `RS_FINALS`, `RS_BALANCE`, `RS_SHAPE`, `RS_INC_EVERY`, `RS_CB_KB`,
`RS_PAYLOAD`, `RS_TRIALS`, `RS_PROFILER` (`device` or `rt`), `RS_ATOL` (0.25), `RS_LINK_GBPS` (48.5),
`RS_MUX_BUFFERS`, `RS_MUX_NOC`, `RS_WORKER_PLACEMENT`.

Each result row: time, the busiest link direction's rate as a % of the link, and the max abs error against the fp32
sum. A topology the fabric cannot route is reported as unsupported instead of failing.

## Measured result
*Illustrative — see the **First profiled on** stamp; re-run the CLI for your box. Full tables and the shape check in
[`report.md`](report.md).* FABRIC_2D_TORUS_XY, 4 chips, bf16; every cell within 0.0625 of the fp32 sum.

| Tiles per chip | 4 | 16 | 64 | 256 | 1,024 | 4,096 | 8,192 |
|---|---|---|---|---|---|---|---|
| baseline, µs | 12.6 | 14.6 | 18.4 | 26.7 | 40.6 | 130.9 | 243.1 |
| busiest link, % of 48.5 GB/s | 1 | 2 | 5 | 15 | 40 | 50 | 53 |
| line instead of ring | 1.31× | 1.34× | 1.26× | 1.20× | 1.41× | 1.35× | 1.38× |
| balance off | 0.96× | 1.03× | 1.01× | 1.02× | 1.19× | 1.25× | 1.29× |
| 2 links | 1.02× | 0.92× | 0.85× | 0.75× | 0.77× | 0.57× | 0.56× |
| 2 workers (1 link) | 1.27× | 1.05× | 0.99× | 0.99× | 0.99× | 0.79× | 0.77× |
| 2 workers (2 links, vs 2 links) | 1.32× | 1.31× | 1.26× | 1.17× | 1.17× | 0.95× | 0.92× |
| 4 final cores per link | 0.99× | 1.01× | 0.99× | 0.95× | 0.96× | 0.89× | 0.90× |
| increment every 2 chunks | 0.97× | 0.99× | 0.93× | 0.86× | 0.98× | 0.99× | 1.02× |
| 2 chunks per batch (56 KiB) | 0.91× | 0.95× | 0.97× | 0.98× | 1.10× | 1.07× | 1.09× |
| 4,352 B router payload | 1.06× | 1.07× | 1.02× | 1.00× | 1.42× | 1.39× | 1.43× |

Rows below the baseline are time relative to it (or to the 2-link baseline where stated); above 1 is slower.

**Reading of the result:**
- **A fixed cost, then a steady rate.** About 12–16 µs per call. The fixed cost and the transfer are equal near 600
  tiles per chip (~1.2 MiB). Below that the call is latency-bound; at 16 MiB the busiest link runs at 53%, held back
  by the relay's two DRAM reads per chunk. Shape does not matter, only size: tall and wide inputs are within 1% of
  square from 1,024 tiles.
- **Ring over line at every size** (1.2–1.4×): fewer serial hops when small, half the busiest-link load when large.
- **Balance a ring from ~1,024 tiles** (1.19–1.29×); below that it is neutral.
- **Workers are a large-size tool.** At 1 link, 2 workers are 1.27–1.30× faster from 4,096 tiles: two cores read the
  relay streams in parallel. They cost 27% at 4 tiles, where the mux's per-call setup dominates, and are neutral in
  between. At 2 links each core already reads half the banks, so 2 workers gain only 5–8% at size.
- **More final cores help at size** (4 per link: 1.04–1.12× from 256 tiles). Each final core reads three streams per
  chunk.
- **Increment interval and batch depth are size-dependent.** Smaller values (every 2–4 chunks, 2 chunks per batch)
  are a few % faster up to 256 tiles. Larger batches and coarser increments are slower there. The defaults (8, 4 per
  batch) are within 16% of the best setting at every size.
- **The router payload matters at size and is not per op.** At 4,352 B the busiest link reaches 36–37%, 1.4× slower
  from 1,024 tiles; up to 256 tiles it is within 7%.

### The relay and its arrival counters
A relay of a chunk waits until its arrival counter covers it. The upstream sender increments the counter on every
`inc_every`-th chunk of the relayed stream and on its last; a relay reading the r-th chunk of that stream waits for
`r / inc_every + 1` increments. The downstream chip's own block goes to its final cores, each with its own counter
per direction, so every counter has exactly one sender per direction.

Batches: the reader pushes a batch to the next stage when it holds `group` chunks, at the end of every block, and at
each multiple of `group` chunks in the buffer (which holds 2 × `group`), so a reservation never wraps past its end.

### Balancing a ring: split the far block by rows
On an even ring of G chips the block opposite each chip is G/2 hops away both ways. With `balance=True` it is split
into the first and second half of its chunk rounds (every bank in each half); half 0 is summed forward, half 1
backward, so both directions carry G/2 − ½ blocks (1.5 instead of 2 on 4 chips). Every block then travels as two
halves, so what a port sends and what its downstream relays line up entry for entry.

### Workers and the fabric mux
With `workers=W > 1` the core next to each link's Ethernet core runs a fabric mux kernel. It is the only connection
to that link's router; the W workers connect to it as clients and send through it. The mux runs on NoC1: the router
sits one row up, and NoC1 (north, then west) reaches it in one hop. Workers are placed with the fewest NoC1 hops to
their mux plus 4 hops for every other reading core already in their physical column, so the readers' NoC0 read
streams mostly turn south in distinct columns. Each link then has W final cores, final core w owning worker w's
banks, so each final counter still has one sender per direction.

### The ready fence
Each port's sender opens a call by sending a *ready* increment to the peer port that writes into this chip's
scratch, and sends no data until its own peer's ready has arrived: no chip writes into a neighbour's scratch before
that neighbour has started the same call. Each call sends and consumes one ready per sending port.

## Run the predefined sweep (regenerates `report.md`)
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_reduce_scatter.py -s
```
(One device session per fabric config × mesh shape. The size sweep in `report.md` runs the baseline over the ladder
with `RS_SHAPE` and one override per row; the exact command is in the report.)

## Code
`program_descriptor_with_inline_kernels.py`: the port reader / add / sender and final reader / add / writer kernels;
`_entries()` (line and ring schedules, balanced halves), `_with_workers()` (worker and mux placement),
`_with_final_cores()` and the `MeshProgramDescriptor` wiring.
