# fabric_all_gather — a line / ring all-gather at the link's rate, on every fabric config

**Difficulty:** ⭐⭐⭐ T3  ·  **Concept(s):** store-and-forward relay gated by arrival counters (and how often to increment them) · placing each port core below its link's Ethernet core
**First profiled on:** `bh-qb-11-special-mstaletovic-for-reservation-93463` · BH (4× p150a) · 2026-09-29 · `cdf859ad4a7`+

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
An all-gather over more than two chips has to move other chips' shards through the chips in between. Each
intermediate chip receives a chunk, and may forward it only once it has fully landed; the receiving side must be told,
cheaply, what has arrived. On top of that, where each link's worker sits on the chip decides whether several links
can run at full rate at once. This example is a complete line / ring all-gather — any mesh axis, or a snake over the
whole mesh, one to four links, on any fabric config — built from those pieces, and measured on every combination
the 4-chip box can form.

## What this isolates — and how
- **Concepts:** (1) relaying with arrival counters: every chunk's packet can carry an atomic increment of the
  receiving port's counter, and the relay reads a chunk only after the counter covers it; (2) placement: each port
  core sits in the NoC column of its connection's Ethernet core.
- **Setup:** DRAM → fabric → DRAM, no compute. Per chip, per direction and per link there is one port core: its
  reader reads the chip's own shard from the input, then the shards it relays from the output; its sender sends each
  chunk into the same pages of the neighbour's output. Chunks are runs of up to 7 tiles that sit consecutively in one
  DRAM bank. One copy core per link writes the chip's own shard into its own output.
  Line: toward p+1 a chip sends shards p, p−1, …, 0; toward p−1 it sends p, …, G−1. Ring: toward p+1 it sends
  G/2 shards (p, p−1, …), toward p−1 the rest.
  Whatever order a ring travels in, the output holds each chip's shard at its row-major slot (chip (r, c) of the
  group at r × cols + c): a snake over the mesh travels in snake order but writes row-major, as high_bw_all_gather
  does.
- **Placement is found, not configured:** a one-time probe program builds each fabric connection on device and
  reports its router's (translated) coordinates; the host maps them to physical NoC columns with the chip's
  Ethernet harvesting mask, and gives each port the free core with the fewest NoC1 hops (the sender's NoC: -Y,
  then -X, wrapping) to its Ethernet core, ties to the lower row — directly below it when that column has Tensix
  cores. Worker coordinates are translated too, and with Tensix harvesting the live columns are compacted (a column
  right of a harvested one has translated x ≠ physical x), so candidate workers are mapped back to physical columns
  with the chip's Tensix harvesting mask before matching. An Ethernet core can sit above a harvested Tensix column
  (LoudBox p150b: Tensix columns 7 and 10 harvested, the axis-0 links on Ethernet columns 10 / 11 and 6 / 7): nearest
  by column then put both links' ports in one column, sharing its NoC1 segment into the Ethernet row, which cost the
  2-chip line ~60 µs of 263 at 16 MiB (2 links); by NoC1 hops the two land in two columns (205 µs, as the C++ op).
- **Why it's kernel-level:** how often an increment rides on a packet, what the relay waits for, and which core
  serves each link are decisions of the kernel and program author.

## The methods being compared
| Setting | What it does | Why it should differ |
|---|---|---|
| increment on every packet *(baseline)* | Every packet is a fused write + increment. | The receiving router issues the increment only after the write lands, so it stalls on every packet. |
| increment every 8 chunks | Every 8th packet (and the last) carries the increment; a relay needs chunk j, so it waits for j/8 + 1 increments. | The router stalls once per 8 packets. |

The fabric config, the topology and the number of links are swept as parameters.

## CLI — measure your own params
Needs 4 chips.

```bash
python -m ttnn.operations.examples.fabric_all_gather [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--fabric` | comma list | all 7 | `1d`, `1d_ring`, `1d_neighbor_exchange`, `2d`, `2d_torus_x`, `2d_torus_y`, `2d_torus_xy` |
| `--topology` | comma list | all 8 | `2x2_axis0_line`, `2x2_axis1_line`, `2x2_snake_line`, `2x2_snake_ring`, `4x1_line`, `4x1_ring`, `1x4_line`, `1x4_ring` |
| `--links` | comma list | `1,2` | links per hop (up to what the fabric reports) |
| `--shape` | `H,W` | `2048,4096` | per-chip shard (tile-aligned) |
| `--dtype` | `bf16`, `bfp8`, `fp32` | `bf16` | shard dtype |
| `--dim` | `0` or `2` | `0` | gather dim (2 = dim −2 of the `[1, 1, H, W]` shard) |
| `--payload` | int (bytes) | `14336` | router max payload |
| `--trials` | int | `3` | measured launches per case (median) |

Test environment variables (`test_fabric_all_gather.py`): `AG_FABRICS`, `AG_TOPOS`, `AG_LINKS`, `AG_VARIANTS`
(`base`, `b` balanced, `d` desync, `bd`), `AG_SHAPE` (`H,W` or a list `H,W;H,W;...`, one row per size), `AG_DTYPE` (`bf16`, `bfp8`, `fp32`, `fp8`), `AG_LAYOUT` (`tile`, `rm`), `AG_DIM`, `AG_PAYLOAD`, `AG_TRIALS`,
`AG_PROFILER` (`device` or `rt`), `AG_STRICT` (1 = an unroutable topology fails), `AG_REUSE_CALLS`, `AG_LINK_GBPS` (one link direction's rate for the utilization column: 48.5 GB/s by default,
27 on a 32-chip Galaxy, whose links are slower: high_bw_all_gather's 8-rank Galaxy gate implies ≥ 26.9).

Each result row: time, effective receive GB/s per chip (shard bytes × (G − 1) / time), and the busiest hop's rate per
link as a % of that peak. On the QuietBox a 1-link line reaches 98–99 % at 16–64 MiB, a 2-link ring ~80 %.

A topology the fabric cannot route (no direct link for a hop, or fewer links than requested) is reported as
unsupported instead of failing.

## Measured result
*Illustrative — see the **First profiled on** stamp above; re-run the CLI for your box. Full matrix in
[`report.md`](report.md).* Effective receive bandwidth per chip, 16 MiB bf16 per chip, 2 links:

| Fabric | 2-chip line (either axis) | 4-chip line (axis or snake) | 4-chip ring |
|---|---|---|---|
| FABRIC_1D | 88–89 | 88 | **125** |
| FABRIC_1D_RING | 85–88 | 87–88 | 124 |
| FABRIC_1D_NEIGHBOR_EXCHANGE | 72–74 | 80–81 | 114–115 |
| FABRIC_2D | 66–67 | 73 | 104–105 |
| FABRIC_2D_TORUS_X / _Y | 66–74 | 72–82 | 103–105 |
| FABRIC_2D_TORUS_XY | 75–76 | 81–82 | 117–118 |

All 112 combinations (7 fabric configs × 8 topologies × 1 and 2 links) are bit-exact. With 4 links on FABRIC_1D a
4-chip ring reaches **143 GB/s per chip** and a 4-chip line 104.

**Reading of the result:**
- **The relay costs nothing in steady state.** A 4-chip line runs at the same per-chip rate as a 2-chip pair (88 GB/s
  at two links): the busiest link direction carries three shards and runs at ~44 GB/s, near a bare link's rate.
  A ring spreads the same bytes over both directions of every link, hence ~125.
- **Increment on every 8th chunk, not every packet.** A fused increment is issued only after its write lands, which
  stalls the receiving router; on every packet it halves the rate (24.5 vs 46.4 GB/s per chip, 2 chips, one link).
- **The fabric config sets the ceiling.** 1D configs are fastest, NeighborExchange and 2D-torus-XY next, plain 2D
  slowest — the same order as a bare one-hop stream on each config with 14,336 B packets; 2D configs prefer a smaller
  router payload (measure it for your config).
- **More links: diminishing past two.** On FABRIC_1D a ring goes 71 → 125 → 141 → 143 GB/s per chip for 1–4 links,
  a 4-chip line 48 → 88 → 103 → 104.
- **Placement is automatic and matters.** Every port core sits on the NoC1-nearest core to its connection's Ethernet
  core (directly below it where the column is live), so the per-link streams share no NoC links.

### Row-major and fp8
The kernels move whole pages and never look inside them, so any dtype and either layout works: a TILE page is a
32 × 32 tile, a ROW_MAJOR page is one row (1152 B for bf16 × 576, 704 B for fp8_e4m3 × 656 — a chunk is then 12 or
20 rows of one bank). A ROW_MAJOR shard needs no alignment (6875 rows per chip is fine); a TILE shard must be
tile-aligned. Bit-exact on the QuietBox (2 links, balanced, 8192 rows per chip): bf16 ROW_MAJOR ring 151 GB/s on
FABRIC_1D and 143–144 on 2D-torus-XY; fp8 ROW_MAJOR 144–147 / 138–140 — within a few % of TILE.

### Balancing a ring: split the far shard between the two directions
On an even ring of G chips each chip receives G/2 shards over one direction and G/2 − 1 over the other, so one
direction of every link idles for a shard's time. With `balance=True` the shard opposite each receiver is split by
bank set: the forward port sends the first half of its banks, the backward port the second half. Both directions then
carry G/2 − ½ shards. On a 4-chip ring the busiest direction carries 1.5 shards instead of 2:

| FABRIC_1D, 4-chip ring (axis, snake) | 1 link | 2 links | 4 links |
|---|---|---|---|
| base | 71 | 124–126 | 146 |
| `balance=True` | 91 | **153–156** | **166** |

It's bit-exact on all 7 fabric configs (2 links: 1D / 1D_RING 153–156, NeighborExchange 148–151, 2D-torus-XY 146–148,
2D and torus X / Y 130–135). A relay of a half shard waits on its upstream's chunk index in the whole shard's walk, so
the arrival rule stays "chunk / 8 + 1" and the relay stays a prefix of what the upstream sends.

`desync=True` starts the backward port's bank walk halfway round its bank set, so the two directions of a link don't
hit the same DRAM bank at the same time. It measures the same as the default: the banks aren't the bottleneck.

The per-port chunk counts are computed on the host and passed as runtime args. Counting them on device (one walk per
shard entry, before the first read) cost ~10 µs per call, 3–7% of a 16 MiB gather.

### Reusing one output: preallocated, fenced, inside a sub-device
`fabric_all_gather(..., output=out)` writes a preallocated output (gathered shape, TILE, DRAM interleaved) in place,
so a caller can allocate it once and reuse it for every call. Reuse needs a fence: chip A's next call must not write
into chip B's output while B's previous result is still being read (by B's relays, or by whatever B queued after
the gather). Each port's sender therefore opens its connection by sending a *ready* increment to the peer port that
writes into this chip, and sends nothing itself until its own peer's ready has arrived. A chip reaching the
program means everything queued before it on that chip has finished, so a ready says "you may write into my output
now". Each call sends and consumes exactly one ready per sending port, so the counters are back at 0 after every
call. The fence costs nothing measurable (ring: 154 vs 155 GB/s).

`test_fabric_all_gather_output_reuse` checks it: 8 calls into one output, a different input each, no host sync, a
clone after every call, with one chip (a different one each call) kept busy before its clone. Without the fence the
late chip's snapshot already holds its neighbours' next-call data (the test fails); with it every snapshot is exact.

- `sub_core_grid` (a `CoreRangeSet` of logical worker cores) confines the op: port, copy and probe cores are picked
  inside it (each port as close as the grid allows to the column of its Ethernet core).
- `subdevice_id`: the sub-device those cores form. Pass it with its cores as `sub_core_grid` (the Python API can't
  look a sub-device's cores up); the op's one-time semaphore setup then synchronizes only that sub-device.
- `ready_semaphore`, `data_valid_semaphore`: caller-owned global semaphores (initial value 0, both or neither)
  covering every core the op may use, e.g. all of `sub_core_grid`. The op then allocates nothing and never
  synchronizes, so it can overlap with work on other sub-devices. Without them it creates its own once per set of
  port cores, and synchronizes once after creating them.

`test_fabric_all_gather_subdevice` runs the gather in a 4-column strip that is its own sub-device, with its own and
with caller-owned semaphores.

## Four neighbours per chip: two Hamiltonian cycles (emulated Galaxy)
A snake ring over a 2D torus feeds each chip over two of its four neighbours. `scheme="dual_cycles"`
(`cluster_axis=None`, `topology=Ring`, both torus sides ≥ 3) splits the torus into two edge-disjoint Hamiltonian
cycles — found on the host, checked to use every edge exactly once — and runs the ring schedule on each, every cycle
owning half of the DRAM banks. Every chip then uses all four neighbours, and the busiest hop carries half as much.

No board here has four neighbours per chip, so this was run on a **32-chip Blackhole Galaxy (4 × 8 torus) under
tt-emule**: the same kernels, each fabric packet's write / increment applied bit-exactly on the destination chip,
no timing. Busiest hop from the plan (shards; time ≥ busiest × shard / (links × link rate)):

| Topology on the 4 × 8 torus | Neighbours per chip | Busiest hop | Emulated (2D, 2D-torus X / Y / XY; 1 and 2 links) |
|---|---|---|---|
| ring along each column (G = 4) | 2 | 2 shards | bit-exact on all 32 chips |
| ring along each row (G = 8) | 2 | 4 shards | bit-exact on all 32 chips |
| snake ring over all 32 chips | 2 | 16 shards | bit-exact on all 32 chips |
| **two Hamiltonian cycles over all 32 chips** | **4** | **8 shards** | bit-exact on all 32 chips |

Two cycles halve the busiest hop of a whole-mesh gather (2× by the bandwidth bound); the speedup itself is not
measured — it needs a real Galaxy. Under emulation FABRIC_1D is rejected by the control plane (no forwarding
direction on a 2D mesh), and FABRIC_1D_RING aborts in the emulator's 1D routing path (an out-of-range L1 offset in
the sender under emulation; the same kernel is bit-exact on FABRIC_1D_RING on hardware).

Running it under tt-emule: build tt-metal with `-DTT_METAL_USE_EMULE=ON -DTT_EMULE_PATH=<tt-emule-blaze checkout>`
(clang-20 toolchain, into its own build tree) at the tt-metal commit the emulator pins, link `libtt-umd.so*` and the
`_ttnn*.so` files the way a normal build does, then run the test with `TT_METAL_EMULE_MODE=1
TT_METAL_SLOW_DISPATCH_MODE=1 EMULE_FABRIC8=1 TT_METAL_MOCK_CLUSTER_DESC_PATH=<umd>/tests/cluster_descriptor_examples/blackhole_galaxy.yaml`
and `AG_TOPOS=4x8_axis0_ring,4x8_axis1_ring,4x8_snake_ring,4x8_dual_cycles AG_FABRICS=2d_torus_xy AG_TRIALS=0`.

## Run the predefined sweep (regenerates `report.md`)
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather.py -s
```
(One device session per fabric config × mesh shape; run a few fabric configs at a time with `--fabric`.)

## Code
`program_descriptor_with_inline_kernels.py`: the port reader / sender, copy writer and probe kernels; `plan()` (groups,
schedules, probe-based placement) and the `MeshProgramDescriptor` wiring.
