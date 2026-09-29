# fabric_gather_pair — a two-chip all-gather from DRAM: what the data plumbing costs

**Difficulty:** ⭐⭐ T2  ·  **Concept(s):** packing interleaved DRAM pages into fabric packets (bank runs) · which NoC the local copy uses
**First profiled on:** `bh-qb-11-special-mstaletovic-for-reservation-93463` · BH (4× p150a) · 2026-09-29 · `05d3e4d94b9`+

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, and read the code only if you need to.

## The problem
A multi-chip op does not stream from a ready-made L1 buffer: it reads tensor pages from DRAM, sends them to a
neighbour, and writes them into the right pages of the neighbour's output, while also placing its own pages in
its own output. Each of those steps can take the link from its ceiling. This example is the smallest real
collective — a two-chip all-gather, both directions at once — and measures how close it gets to a bare
fabric stream (48.5 GB/s per link per direction on this box, FABRIC_1D, 14,336 B packets, one link).

## What this isolates — and how
- **Concepts:** (1) how DRAM pages are grouped into DRAM reads and fabric packets; (2) whether the local copy
  shares a NoC with the fabric sends.
- **Isolation setup:** DRAM → fabric → DRAM, no compute. Chips are paired along mesh axis 0 of a 2×2 mesh. Each
  chip's shard is TILE, interleaved in DRAM; the output is `[row-0 shard ; row-1 shard]`. Per link, one core:
  the reader (RISCV_1, NoC0) reads chunks from DRAM into a CB with a group of reads per barrier; the sender
  (RISCV_0) writes each chunk into the local output and sends it into the same pages of the peer's output, then
  waits for the peer's last packet (a fused write + increment). Links split the DRAM banks; each link visits its
  banks round-robin.
- **Why it's kernel-level:** the grouping of pages into reads and packets, the bank visiting order, and the NoC a
  write is issued on are all decisions in the reader and sender kernels.

## The methods being compared
| Variant | What it does | Why it should differ |
|---|---|---|
| `page_per_packet` *(baseline)* | One tile (2 KiB) per DRAM read and per packet. | Every packet pays the router's per-packet cost for 2 KiB. |
| `bank_run` | Up to 7 tiles (14 KiB) per DRAM read and per packet: interleaved tiles p, p+B, p+2B, … (B = DRAM banks) are consecutive in one bank, and so are their places in the output. | Full packets and 7× fewer DRAM reads. |
| `+local_noc0` | The local copy is issued on NoC0 instead of the sender's NoC1. | The local copy and the fabric sends no longer share the core's NoC1 outbound port. |

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
| `--local-noc` | `same`, `noc0` (comma list) | `same,noc0` | NoC for the local copy |
| `--ablate` | `\|`-separated sets | *(none)* | diagnostic: drop `dram_read`, `local_copy`, `fabric` (`+`-joined); output is then not checked |

Default placements: `1link_under_eth=2,0`, `2link_adjacent=0,0;1,0`, `2link_under_eth=2,0;3,0`. "under_eth" puts
each core in the NoC column of its link's Ethernet core; those columns are board-specific (read them from a NoC
trace).

There is no `--kernel-iters`: one launch moves 64 MiB per chip each way (≥ 1.1 ms).

## Measured result
*Illustrative — see the **First profiled on** stamp above; re-run the CLI for your box. Full matrix in
[`report.md`](report.md).* GB/s per link per direction, median of 3:

| Link cores | `page_per_packet` | `bank_run` | `bank_run` + local copy on NoC0 |
|---|---|---|---|
| 1 link, under its Ethernet core | 12.8 | 32.6 | **40.3** |
| 2 links, side by side | 12.2 | 26.2 | 26.3 |
| 2 links, under their Ethernet cores | 12.7 | **28.7** (57.5 per chip) | 25.9 |

**Reading of the result:**
- **Bank runs are the big lever: 2.5× over a page loop** (12.8 → 32.6 GB/s). 2 KiB packets cap a link near
  13 GB/s whatever else the kernel does.
- **Visit banks round-robin.** Walking one bank at a time (every run of bank b before b+1) gives 20.0 GB/s instead
  of 33.7: that bank serves the reads, the local copy and the peer's incoming writes at once.
- **At one link, the local copy is the remaining cost.** Ablations: the full op runs at 33.6 GB/s; without the
  local copy 48.2 (the link ceiling); without the fabric send the core still reads and copies at 51.7; without
  the DRAM read 34.5. The local copy and the packets both leave the core on NoC1, and together they share one
  outbound port. Moving the local copy to NoC0 recovers **+24%** (32.6 → 40.3).
- **Two links do not double it, and moving the local copy to NoC0 then hurts** (28.7 → 25.9 per link). Per
  chip the op moves 57 GB/s each way, well short of the 97 GB/s two bare links reach. The cause is not
  measured yet; with two link cores, DRAM reads, local copies and the routers' incoming DRAM writes all cross
  the NoC toward the same two DRAM columns.

## Run the predefined sweep (regenerates `report.md`)
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_fabric_gather_pair.py -s
```

## Code
`program_descriptor_with_inline_kernels.py`: the reader and sender kernels and the `MeshProgramDescriptor`
wiring (one program per chip; the fabric connection args appended last to the sender's runtime args).
