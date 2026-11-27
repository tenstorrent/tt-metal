# L1 Ledger: g6_zeros

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes: `op_design.md` → Blocking Model (`d0`, `d1`, `ht`, `wt` folded into the page index; `bank`, `bank_slot`, `packet`).

## Inventory

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_zero_src` | `zero_src_depth` = 1 page of `zero_src_bytes = min(ZERO_SRC_BYTES_MAX, ceil(P / cores_used) * page_bytes)`; `ZERO_SRC_BYTES_MAX` = 4096 as built (design Phase 0: 16384) | 1 page: every packet in flight reads the same bytes, so capacity = live set | `{d0: streams → folded, d1: streams → folded, ht: streams → folded, wt: streams → folded, bank: streams → 1 per run, bank_slot: streams → run_slots, packet: spans → packet_bytes <= zero_src_bytes}` | output `dtype`. Metadata only: a dataflow-only payload that never passes DEST or the packer, so the DEST-width audit does not apply | writer | writer | whole kernel: zero-filled once (`make_zero_source`), read-only through every `write_zero_run` | The only CB, so nothing to share with. It is not folded into an output CB because none exists: pages go from this source straight to DRAM. Not replaced by the 512 B hardware zero region (`MEM_ZEROS_BASE`): that would cap packets at 512 B, 32x the packets (regime `zero_region_direct`, rejected) |

Rule 1 (inventory before solve): one buffer, no budget solve, no safety fraction. Rule 3 patterns: pack-into-destination is the design (the output gets no CB at all). Transform-in-place, aliasing and DEST folding have nothing to apply to: there is one buffer and no compute phase.

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `page_bytes` | tile bytes of `dtype` | ∈ {1088, 2048, 4096} | TARGET dtype ∈ {bfloat8_b, bfloat16, float32}, TILE only (`validate()`) |
| `P` | output pages, `d_0*d_1*ceil(d_2/32)*ceil(d_3/32)` | ≥ 1, unbounded above (262648 in INPUTS) | shape validation; appears only under `min(...)`, never as a capacity on its own |
| `cores_used` | `min(writers, P)`, `writers = min(C, WRITER_CORES_PER_BANK * NB)` (8 as built; `C` with the knob at `None`) | ≥ 1 | `C = grid.x * grid.y >= 1` |
| `ZERO_SRC_BYTES_MAX` | host constant (packet-axis knob) | = 4096 as built; design Phase 0 16384 (one Blackhole NoC burst) | single source in the descriptor builder |
| `zero_src_bytes` | zero source size | `<= ZERO_SRC_BYTES_MAX` (4096 as built), multiple of 64 (all `page_bytes` are multiples of 64) | `min(ZERO_SRC_BYTES_MAX, ...)` |
| `packet_bytes` | per-packet transfer | `<= min(zero_src_bytes, NOC_MAX_BURST_SIZE)` | kernel clamp |
| `zero_src_depth` | pages of `cb_zero_src` | = 1 | constant source, never refilled |

## Footprint

Per-core L1 = `zero_src_depth * zero_src_bytes` = `zero_src_bytes` ≤ `ZERO_SRC_BYTES_MAX`. As built (`ZERO_SRC_BYTES_MAX` = 4096, one writer per bank): **4096 B** on every writer for p1/p2/p3/p4, ≈ 0.3 % of the model's 1347776 B (BB-05). The design's Phase 0 value (16384) stays ≤ 1.2 %. Only the `packet`-axis knob scales it. No term scales with `P` or with the shape: the footprint is constant in the tensor size. The CB is allocated on the whole grid rectangle (one range, BB-17), idle cores included. The second-NoC lamp (`writer_riscs = 2`, not built) would add a second identical source per core.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `out` | 1 (write only) | each page lies in exactly one run of exactly one core and is written once, straight from L1; nothing is cleared first and nothing is read back | 0 |
| zero source | 0 | made in each core's own L1 by NoC loopback from the hardware zero region, once per kernel; never read from DRAM (spec rule 4) | 0 (loopback is core-local: `zero_src_bytes` per active core) |

Totals per device per call:

| Case | DRAM written | DRAM read | Cross-core | Core-local loopback (as built: 8 writers x 4096 B) |
|------|--------------|-----------|------------|-------------------------------|
| p1 bfloat8_b | 285,761,024 B | 0 | 0 | 32,768 B |
| p2 float32 | 12,582,912 B | 0 | 0 | 32,768 B |
| p3 float32 | 6,291,456 B | 0 | 0 | 32,768 B |
| p4 bfloat16 | 262,144 B | 0 | 0 | 32,768 B |

On a mesh each device writes its own copy (x4 in production), with no inter-chip traffic.

> Cheapest-traffic split considered: bank-major position split (`bank_run_write`). Every candidate split ties on bytes (output written once, 0 read, 0 cross-core) and this one has the fewest transfers (≤ 2 contiguous runs per core). Implemented: the same split, dealt bank-affine to one writer per bank by default, so each writer holds exactly one run when `P % NB == 0`. It is also the cheapest-traffic split; no deferred alternative exists.
