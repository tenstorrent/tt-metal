# L1 Ledger: g6_zeros

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes: `op_design.md` → Blocking Model (`d0`, `d1`, `ht`, `wt` folded into the page index; `bank`, `bank_slot`, `packet`).

## Inventory

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_zero_src` | `zero_src_depth` = 1 page of `zero_src_bytes = min(ZERO_SRC_BYTES_MAX, ceil(P / cores_used) * page_bytes)` | 1 page: every packet in flight reads the same bytes, so capacity = live set | `{d0: streams → folded, d1: streams → folded, ht: streams → folded, wt: streams → folded, bank: streams → 1 per run, bank_slot: streams → run_slots, packet: spans → packet_bytes <= zero_src_bytes}` | output `dtype`. Metadata only: a dataflow-only payload that never passes DEST or the packer, so the DEST-width audit does not apply | writer | writer | whole kernel: zero-filled once (`make_zero_source`), read-only through every `write_zero_run` | The only CB, so nothing to share with. It is not folded into an output CB because none exists: pages go from this source straight to DRAM. Not replaced by the 512 B hardware zero region (`MEM_ZEROS_BASE`): that would cap packets at 512 B, 32x the packets (regime `zero_region_direct`, rejected) |

Rule 1 (inventory before solve): one buffer, no budget solve, no safety fraction. Rule 3 patterns: pack-into-destination is the design (the output gets no CB at all). Transform-in-place, aliasing and DEST folding have nothing to apply to: there is one buffer and no compute phase.

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `page_bytes` | tile bytes of `dtype` | ∈ {1088, 2048, 4096} | TARGET dtype ∈ {bfloat8_b, bfloat16, float32}, TILE only (`validate()`) |
| `P` | output pages, `d_0*d_1*ceil(d_2/32)*ceil(d_3/32)` | ≥ 1, unbounded above (262648 in INPUTS) | shape validation; appears only under `min(...)`, never as a capacity on its own |
| `cores_used` | `min(C, P)` | ≥ 1 | `C = grid.x * grid.y >= 1` |
| `ZERO_SRC_BYTES_MAX` | host constant, one Blackhole NoC burst | = 16384 | single source in the descriptor builder |
| `zero_src_bytes` | zero source size | `<= 16384`, multiple of 16 (all `page_bytes` are multiples of 64) | `min(ZERO_SRC_BYTES_MAX, ...)` |
| `packet_bytes` | per-packet transfer | `<= min(zero_src_bytes, NOC_MAX_BURST_SIZE)` | kernel clamp |
| `zero_src_depth` | pages of `cb_zero_src` | = 1 | constant source, never refilled |

## Footprint

Per-core L1 = `zero_src_depth * zero_src_bytes` = `zero_src_bytes` ≤ **16384 B** for every shape (p1/p2/p3: 16384; p4: 4096 at C = 110, 2048 at C = 130). That is ≤ 1.2 % of the model's 1347776 B (BB-05). Only the `packet`-axis knob (`ZERO_SRC_BYTES_MAX`) scales it. No term scales with `P` or with the shape: the footprint is constant in the tensor size, so p1's 285,761,024 B output needs the same 16 KiB per core as p4. The second-NoC lamp (`writer_riscs = 2`) adds a second identical 16 KiB source per core (32 KiB total).

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `out` | 1 (write only) | each page lies in exactly one run of exactly one core and is written once, straight from L1; nothing is cleared first and nothing is read back | 0 |
| zero source | 0 | made in each core's own L1 by NoC loopback from the hardware zero region, once per kernel; never read from DRAM (spec rule 4) | 0 (loopback is core-local: `zero_src_bytes` per active core) |

Totals per device per call:

| Case | DRAM written | DRAM read | Cross-core | Core-local loopback (C = 110) |
|------|--------------|-----------|------------|-------------------------------|
| p1 bfloat8_b | 285,761,024 B | 0 | 0 | 110 x 16384 = 1,802,240 B |
| p2 float32 | 12,582,912 B | 0 | 0 | 1,802,240 B |
| p3 float32 | 6,291,456 B | 0 | 0 | 1,802,240 B |
| p4 bfloat16 | 262,144 B | 0 | 0 | 110 x 4096 = 450,560 B |

On a mesh each device writes its own copy (x4 in production), with no inter-chip traffic.

> Cheapest-traffic split considered: bank-major position split (`bank_run_write`). Every candidate split ties on bytes (output written once, 0 read, 0 cross-core) and this one has the fewest transfers (≤ 2 contiguous runs per core). Implemented: the same split. It is also the cheapest-traffic split; no deferred alternative exists.
