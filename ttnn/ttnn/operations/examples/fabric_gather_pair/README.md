# fabric_gather_pair — a two-chip all-gather from DRAM at the link's full rate

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** packing interleaved DRAM pages into fabric packets (bank runs) · keeping the local copy off the link core
**First profiled on:** `bh-qb-11-special-mstaletovic-for-reservation-93463` · BH (4× p150a) · 2026-09-29 · `05d3e4d94b9`+

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
A multi-chip op does not stream from a ready-made L1 buffer: it reads tensor pages from DRAM, sends them to a
neighbour, writes them into the right pages of the neighbour's output, and also places its own pages in its own
output. Each of those steps can take the link below its ceiling. This example is the smallest real collective — a
two-chip all-gather, both directions at once — built up until it matches a bare fabric stream (48.5 GB/s per link
per direction on this box, FABRIC_1D, 14,336 B packets; 97 GB/s for two links).

## What this isolates — and how
- **Concepts:** (1) how DRAM pages are grouped into DRAM reads and fabric packets; (2) where the local copy is made.
- **Isolation setup:** DRAM → fabric → DRAM, no compute. Chips are paired along mesh axis 0 of a 2×2 mesh. Each
  chip's shard is TILE, interleaved in DRAM; the output is `[row-0 shard ; row-1 shard]`. Per link, one link core:
  its reader (RISCV_1, NoC0) reads chunks from DRAM into a CB, a group of reads per barrier; its sender (RISCV_0,
  NoC1) sends each chunk into the same pages of the peer's output, then waits for the peer's last packet (a fused
  write + increment). Links split the DRAM banks and visit them round-robin. Each link core sits in the NoC column of
  its link's Ethernet core.
- **Why it's kernel-level:** the grouping of pages into reads and packets, the bank order, and which core and NoC
  write the local copy are all decisions in the kernels and the core assignment.

## The methods being compared
| Variant | What it does | Why it should differ |
|---|---|---|
| `page_per_packet` *(baseline)* | One tile (2 KiB) per DRAM read and per packet; the link core's sender also writes the local copy. | Every packet pays the router's per-packet cost for 2 KiB. |
| `bank_run` | Up to 7 tiles (14 KiB) per DRAM read and per packet: interleaved tiles p, p+B, p+2B, … (B = DRAM banks) are consecutive in one bank, and so are their places in the output. | Full packets and 7× fewer DRAM reads. |
| `+local_noc0` | The link core writes the local copy on NoC0 instead of its sender's NoC1. | The local copy stops sharing the NoC1 outbound port with the packets. |
| `+copy:copyN` | N separate copy cores make the local copy (each reads its share of the banks and writes it to the local output); the link cores only read and send. | The link core does nothing but feed its link; the shard is read from DRAM twice. |

## CLI — measure your own params
Needs a 2×2 mesh (4 chips).

```bash
python -m ttnn.operations.examples.fabric_gather_pair [options]
```

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--variant` | `all` or comma list | `all` | `page_per_packet`, `bank_run` |
| `--trials` | int | `3` | measured launches per case (median) |
| `--shape` | `H,W` | `8192,4096` | per-chip shard (bf16, tile-aligned; pages must split evenly over the DRAM banks) |
| `--payload` | int (bytes) | `14336` | router max payload |
| `--placements` | `name=x,y;x,y\|...` | see below | link cores (logical), link l on the l-th core |
| `--local-noc` | `same`, `noc0` (comma list) | `same,noc0` | NoC for a local copy made by the link core |
| `--copy-cores` | `name=x,y;...\|...` or `none` | `none\|copy1=0,6\|copy2=0,6;1,6` | copy-core sets to try |
| `--ablate` | `\|`-separated sets | *(none)* | diagnostic: drop `dram_read`, `local_copy`, `fabric` (`+`-joined); output is then not checked |

Default placements: `1link_under_eth=2,0`, `2link_adjacent=0,0;1,0`, `2link_under_eth=2,0;3,0`. "under_eth" puts
each link core in the NoC column of its link's Ethernet core; those columns are board-specific (read them from a NoC
trace).

There is no `--kernel-iters`: one launch moves 64 MiB per chip each way (≥ 0.7 ms).

## Measured result
*Illustrative — see the **First profiled on** stamp above; re-run the CLI for your box. Full matrix in
[`report.md`](report.md).* GB/s per link per direction, link cores under their Ethernet cores, median of 3:

| Local copy made by | `page_per_packet` | `bank_run` |
|---|---|---|
| **1 link:** the link core (NoC1) | 12.9 | 32.6 |
| the link core (NoC0) | 12.7 | 40.2 |
| **1 copy core** | 13.6 | **48.2** |
| **2 links:** the link cores (NoC1) | 12.7 | 28.7 (57 per chip) |
| the link cores (NoC0) | 12.5 | 25.9 |
| 1 copy core | 12.4 | 25.0 (50 per chip) |
| **2 copy cores** | 13.4 | **47.5 (95 per chip)** |

**Reading of the result:**
- **The op reaches the bare link: 48.2 GB/s for one link and 95 GB/s per chip for two** (bare streams: 48.5 and 97)
  with bank runs plus one copy core per link.
- **Bank runs are the first lever: 2.5× over a page loop.** 2 KiB packets cap a link near 13 GB/s whatever else the
  kernel does.
- **Visit banks round-robin.** Walking one bank at a time gives 20.0 GB/s instead of 33.7 (one link): that bank
  serves the reads, the local copy and the peer's incoming writes at once.
- **The local copy must leave the link core.** Ablations (one link): the full op runs at 33.6 GB/s, 48.2 without the
  local copy, 51.7 without the fabric send, 34.5 without the DRAM read — the local copy and the packets fight for
  the core's outbound NoC port. Moving the copy to NoC0 recovers part of it at one link and loses at two; moving it
  to its own core recovers all of it. Reading the shard a second time for the copy costs nothing measurable.
- **One copy core per link, no more.** A copy core copies about 50 GB/s, one link's worth: with two links one copy
  core holds the op at 50 GB/s per chip; two reach 95. Four copy cores give 83 and eight 76 — they finish early
  and meanwhile compete with the link cores for DRAM and NoC.
- **Placement still matters.** With the two link cores side by side instead of under their Ethernet cores, even two
  copy cores reach only 29.5 GB/s per link.

## Run the predefined sweep (regenerates `report.md`)
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_gather_pair.py -s
```

## Code
`program_descriptor_with_inline_kernels.py`: the reader, sender and copy-writer kernels and the
`MeshProgramDescriptor` wiring (one program per chip; the fabric connection args appended last to the sender's
runtime args).
