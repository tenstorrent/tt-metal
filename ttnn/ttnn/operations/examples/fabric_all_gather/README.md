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
- **Placement is found, not configured:** a one-time probe program builds each fabric connection on device and
  reports its router's (translated) coordinates; the host maps them to physical NoC columns with the chip's
  Ethernet harvesting mask, and puts each port core directly below.
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
- **Placement is automatic and matters.** Every port core sits directly below its connection's Ethernet core, so the
  per-link streams share no NoC links.

Known limitation: calls are not fenced against each other. A chip may start its next call while a neighbour is
still relaying the previous one; with a different input per call that neighbour could relay newer data. The arrival
counters stay consistent (they are re-armed by exactly what each call consumed).

## Run the predefined sweep (regenerates `report.md`)
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_all_gather.py -s
```
(One device session per fabric config × mesh shape; run a few fabric configs at a time with `--fabric`.)

## Code
`program_descriptor_with_inline_kernels.py`: the port reader / sender, copy writer and probe kernels; `plan()` (groups,
schedules, probe-based placement) and the `MeshProgramDescriptor` wiring.
