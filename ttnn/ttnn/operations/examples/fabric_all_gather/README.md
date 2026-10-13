# fabric_all_gather — a line / ring all-gather over the fabric, and what sets its speed at each size

**Difficulty:** ⭐⭐⭐ T3  ·  **Concept(s):** relaying a stream with arrival counters (increment interval and batch depth) · balancing a ring's two directions
**First profiled on:** `qb2-120-p02t03` · BH (QuietBox 2, 2× p300c, 4 chips) · FABRIC_2D_TORUS_XY · 2026-10-13 · `316bf3de8c4`+

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
An all-gather over more than two chips moves other chips' shards through the chips in between. An intermediate chip
receives a chunk and may forward it only once it has fully landed, so the receiving side must be told, cheaply, what
has arrived; and a ring has two directions to keep busy. Both decisions change with size: a few tiles per chip is
bound by the fixed cost of a call and by the number of serial hops, megabytes per chip by the busiest link. This
example is a complete line / ring all-gather (any mesh axis or a snake over the mesh, one or more links, any fabric
configuration), measured from 4 tiles to 16 MiB per chip.

## What this isolates — and how
- **Concepts:** (1) relaying with arrival counters: a packet can carry an atomic increment of the receiving port's
  counter, the relay reads a chunk only once the counter covers it, and chunks move in batches through a circular
  buffer; (2) the ring schedule: which shards go which way, and splitting the far shard between the two directions.
- **Setup:** DRAM → fabric → DRAM, no compute. Per chip, per direction and per link there is one **port core**. Its
  reader (NCRISC, NoC0) reads the chip's own shard from the input, then the shards it relays from the output. Its
  sender (BRISC, NoC1) sends every chunk into the same pages of the neighbour's output. A chunk is a run of up to 7
  tiles (one packet at a 14,336 B payload) consecutive in one DRAM bank; links split a shard by bank. One copy core
  per link writes the chip's own shard into its own output. Line: toward p+1 a chip sends shards p, p−1, …, 0; toward
  p−1, p, …, G−1. Ring: toward p+1 G/2 shards (p, p−1, …), toward p−1 the rest.
- **Placement:** each port core sits next to its link's Ethernet core (the free core with the fewest NoC1 hops to
  it), found by a one-time probe program, so the per-link streams share no NoC links.
- **Why it's kernel-level:** the increment interval, the batch depth, the schedule and the split are all decisions of
  the kernel and program author.

## The methods being compared
Each is varied alone from the baseline: 4-chip ring, balanced, 1 link, increment every 8 chunks, 112 KiB circular
buffer, 14,336 B router payload.

| Setting | What it does | Why it should differ |
|---|---|---|
| `topology`: line *(baseline: ring)* | A line sends toward each end; a ring uses both directions of every link | Line: the busiest link carries G − 1 shards, and the farthest shard crosses G − 1 hops in series |
| `balance=False` *(baseline: True)* | On an even ring the shard opposite each receiver goes one way | Busiest link carries G/2 shards instead of G/2 − ½ |
| `num_links=2` | Each link carries half the DRAM banks | Twice the link bandwidth |
| `inc_every` 2 / 4 / 16 / 32 *(baseline 8)* | Increment the receiver's arrival counter on every N-th chunk (and on each shard's last) | The receiving router stalls until each increment's write has landed; a relay waits up to N chunks behind |
| `cb_bytes` 56 / 224 KiB *(baseline 112)* | 2 / 8 chunks per batch (4 at 112): the reader issues a batch of reads before waiting, the sender a batch of packets before flushing | Batches amortize the waits, but a batch a short shard can't fill waits longer |
| router payload 4,352 B *(baseline 14,336)* | Set once for the whole fabric, not per op | 3.3× more packets for the same bytes |

## CLI — measure your own shapes/params
Needs 4 chips.

```bash
python -m ttnn.operations.examples.fabric_all_gather [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--fabric` | comma list | `2d_torus_xy` | `1d`, `1d_ring`, `1d_neighbor_exchange`, `2d`, `2d_torus_x`, `2d_torus_y`, `2d_torus_xy` |
| `--topology` | comma list | all 8 | `2x2_axis0_line`, `2x2_axis1_line`, `2x2_snake_line`, `2x2_snake_ring`, `4x1_line`, `4x1_ring`, `1x4_line`, `1x4_ring` |
| `--variant` | comma list | `base,b` | `base`, `b` = balanced ring |
| `--links` | comma list | `1,2` | links per hop (up to what the fabric reports) |
| `--shape` | `H,W` or `H,W;H,W;…` | `2048,4096` | per-chip shard (tile-aligned), one row per shape |
| `--inc-every` | comma list | `8` | arrival increment every N chunks |
| `--cb-kb` | comma list | `112` | circular buffer per port, KiB |
| `--dtype` | `bf16`, `bfp8`, `fp32` | `bf16` | shard dtype |
| `--dim` | `0` or `2` | `0` | gather dim (2 = dim −2 of the `[1, 1, H, W]` shard) |
| `--payload` | int (bytes) | `14336` | router max payload, applied when the fabric opens |
| `--trials` | int | `3` | measured calls per case (median) |

Every comma list is crossed with the others. Test environment variables (`test_fabric_all_gather.py`): `AG_FABRICS`,
`AG_TOPOS`, `AG_VARIANTS`, `AG_LINKS`, `AG_SHAPE`, `AG_INC_EVERY`, `AG_CB_KB`, `AG_DTYPE`, `AG_LAYOUT` (`tile`,
`rm`), `AG_DIM`, `AG_PAYLOAD`, `AG_TRIALS`, `AG_PROFILER` (`device` or `rt`), `AG_STRICT` (1 = an unroutable topology
fails), `AG_REUSE_CALLS`, `AG_LINK_GBPS` (the link rate for the utilization column; 48.5 GB/s by default).

Each result row: time, receive GB/s per chip (shard bytes × (G − 1) / time), and the busiest link direction's rate as
a % of the link. A topology the fabric cannot route is reported as unsupported instead of failing.

## Measured result
*Illustrative — see the **First profiled on** stamp; re-run the CLI for your box. Full tables, fits and the shape
check in [`report.md`](report.md).* FABRIC_2D_TORUS_XY, 4 chips, bf16, all cells bit-exact. Baseline: balanced ring,
1 link, 14,336 B payload.

| Tiles per chip | 4 | 16 | 64 | 256 | 1,024 | 4,096 | 8,192 |
|---|---|---|---|---|---|---|---|
| baseline, µs | 10.6 | 13.7 | 16.8 | 29.1 | 89.2 | 335.0 | 660.4 |
| busiest link, % of 48.5 GB/s | 2 | 7 | 24 | 56 | 73 | 77 | 79 |
| line instead of ring | 1.29× slower | 1.25× | 1.39× | 1.55× | 1.78× | 1.83× | 1.82× |
| balance off | 1.02× slower | 0.93× | 1.10× | 1.20× | 1.26× | 1.28× | 1.27× |
| 2 links | 0.92× | 0.80× | 0.80× | 0.63× | 0.58× | 0.53× | 0.53× |
| increment every 2 chunks | 1.05× | 1.09× | 1.08× | 1.15× | 1.17× | 1.14× | 1.14× |
| increment every 32 chunks | 1.02× | 1.07× | 1.11× | 1.02× | 0.99× | 0.98× | 0.98× |
| 2 chunks per batch (56 KiB) | 0.95× | 0.96× | 1.01× | 1.11× | 1.16× | 1.16× | 1.18× |
| 8 chunks per batch (224 KiB) | 1.12× | 1.12× | 1.19× | 1.06× | 1.04× | 1.02× | 1.02× |
| 4,352 B router payload | 1.15× | 1.11× | 1.27× | 1.77× | 1.99× | 2.04× | 2.06× |

Rows below the baseline are time relative to it; above 1 is slower.

**Reading of the result:**
- **A fixed cost, then a steady rate.** The baseline fits t = 8.4 µs + bytes received / 77.2 GB/s per chip. Below
  ~100 tiles per chip the call is latency-bound and the link mostly idle; above ~1,000 it runs at 73–79% of the link.
  Shape does not matter, only size: tall (H × 32) and wide (32 × W) shards are within 1% of square from 1,024 tiles.
- **Ring over line at every size.** Small: a ring's farthest shard crosses 2 hops per direction instead of 3 (fixed
  cost 8.4 vs 9.8 µs). Large: the busiest link carries 1.5 shards instead of 3.
- **Balance a ring above ~64 tiles** (1.26–1.28× from 1,024). Below that the split halves are a few chunks each and
  it gains nothing.
- **Increment every 8 chunks, batch 4.** Increments every 2 chunks stall the receiving router (14–17% slower at
  size); every 16–32 gain 1–2% at size but lose 7–11% at 16–64 tiles, where a short shard has few chunks to amortize
  over. 2 chunks per batch starves the pipeline at size; 8 is a batch a short shard can't fill.
- **The router payload matters most, and is not per op.** It is fixed when the fabric opens, for every op in the
  process. At the default 4,352 B the link reaches only 38% of its rate, 2× slower at size; two links at 4,352 B only
  match one at 14,336 B.

### Arrival counters: when a relay may read
A relay of chunk *i* waits until its counter covers that chunk. The sender increments the receiver's counter on every
`inc_every`-th chunk and on the last chunk of every shard it sends, so a relay reading chunk *i* of the upstream's
k-th entry waits for `i / inc_every + 1` increments plus one per earlier entry end that is not a multiple of
`inc_every`.

### Batches
The reader pushes a batch to the sender when it holds `group` chunks, at the end of every shard, and at each multiple
of `group` chunks in the buffer (which holds 2 × `group`), so a reservation never wraps past its end.

### Balancing a ring: split the far shard between the two directions
On an even ring of G chips each chip receives G/2 shards over one direction and G/2 − 1 over the other. With
`balance=True` the shard opposite each receiver is split by bank set: the forward port sends the first half of its
banks, the backward port the second half, so both directions carry G/2 − ½ shards (1.5 instead of 2 on 4 chips).

### Reusing one output: preallocated, fenced, inside a sub-device
`fabric_all_gather(..., output=out)` writes a preallocated output (gathered shape, TILE, DRAM interleaved) in place, so
a caller can allocate it once. Reuse needs a fence: chip A's next call must not write into chip B's output while B's
previous result is still being read. Each port's sender therefore opens by sending a *ready* increment to the peer
port that writes into this chip, and sends nothing until its own peer's ready has arrived. Each call sends and
consumes exactly one ready per sending port, so the counters are back at 0 after every call.

- `sub_core_grid` (a `CoreRangeSet` of logical worker cores) confines the op: port, copy and probe cores are picked
  inside it.
- `subdevice_id`: the sub-device those cores form (pass it with its cores as `sub_core_grid`).
- `ready_semaphore`, `data_valid_semaphore`: caller-owned global semaphores (initial value 0, both or neither)
  covering every core the op may use. With them the op allocates nothing and never synchronizes.

`test_fabric_all_gather_output_reuse` (8 calls into one output, one chip kept busy each call, no host sync) and
`test_fabric_all_gather_subdevice` (inside a 4-column sub-device) check both.

## Run the predefined sweep (regenerates `report.md`)
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather.py -s
```
(One device session per fabric config × mesh shape. The size sweep in `report.md` runs the baseline over the ladder
with `AG_SHAPE` and one override per row; the exact command is in the report.)

## Code
`program_descriptor_with_inline_kernels.py`: the port reader / sender, copy writer and probe kernels; `plan()` (groups,
schedules, probe-based placement) and the `MeshProgramDescriptor` wiring.
