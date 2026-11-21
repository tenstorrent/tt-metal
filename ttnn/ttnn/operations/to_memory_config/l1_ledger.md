# L1 Ledger: to_memory_config

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions: `op_design.md` § Blocking Model. Target: `quasar` (4 MiB L1 per cluster; the unreserved query returns 3,963,328 B).

Block axes (from the Blocking Model):

| Axis | Folded into |
|------|-------------|
| `page` | the axis the op is blocked on (extent `block_pages`) |
| leading dims, H | folded into `page` |
| W | folded into `page` (TILE), or inside one entry (ROW_MAJOR) |

## Buffers

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `dfb_pages` | `block_pages * depth`, with `entry_bytes = min(in_aligned_page, out_aligned_page)` | `block_pages * depth`. With depth 2, the reader fills one txn-id slice while the writer drains the other, so both blocks are live at once. With depth 1, the whole block is in flight | `{page: spans → block_pages * depth; leading dims: folded into page; H: folded into page; W: TILE folded into page, RM spans → inside entry_bytes}` | the tensor's dtype (bf16 in Phase 0). It is a dataflow-only payload that never passes through DEST, so audit 2's DEST-width rule does not apply: the bytes are moved, never unpacked | reader | writer | the whole program | Shares with: nothing, because it is the only buffer. The input and output tensors' own L1 storage is not a ring. Backing this ring on the input or output shard (regime R-borrow, zero-copy) is deferred in the regime table: mesh and DRAM traffic are identical, and a borrowed ring's entry count (the shard's pages) breaks the 255-entries-per-txn-id implicit-sync cap. It cannot be split into a producer-only and a consumer-only buffer, because a DM self-loop is forbidden on this part (`dfb.cpp:646-652`) |

## Symbol table

| Symbol | Bound | Predicate that establishes it |
|--------|-------|-------------------------------|
| `entry_bytes` | `page_bytes <= entry_bytes <= 65536` | `validate()` refuses `entry_bytes > 65536` (one NoC packet per implicit transfer) |
| `block_pages` | `1 <= block_pages <= min(255, core_pages_max)`, and `block_pages * depth * entry_bytes <= ring_budget` | derivation in `op_design.md`: `TXN_ENTRIES_CAP = 255`; the single-block or depth-2 branch |
| `depth` | 1 or `BUFFER_DEPTH = 2` | 1 iff `core_pages_max <= min(255, ring_budget // entry_bytes)` |
| `core_pages_max` | max pages any engaged core owns: `ceil(num_pages / cores)` (R-bal), or `pages_per_shard` (R-in / R-out) | the work split. Sizes the ring only through the clamp `block_pages <= core_pages_max` |
| `ring_budget` | `min(RING_BYTES_CAP = 262144, ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes)`, with `l1_reserve_bytes = 0`. On quasar: `min(262144, 3963328) = 262144` | host, at program build. It is a device query, not a literal standing in for one |

## Total per-core footprint

`footprint = block_pages * depth * entry_bytes <= ring_budget (262,144 B on quasar)`.

| Term | Scales with |
|------|-------------|
| `block_pages` | the `page` block knob |
| `depth` | the buffer-depth knob |
| `entry_bytes` | dtype, and the row width in ROW_MAJOR |

Nothing scales with an unbounded op dimension. The largest worked case is a 1-core shard of `[1,1,32,4096]` TILE: 128 × 1 × 2048 = 262,144 B. Typical interleaved cells take 112 B – 33 KB.

## Data-movement budget

The chosen split is owner-aligned page ranges where a side is height-sharded (R-in / R-out), and balanced page ranges over the full grid otherwise (R-bal).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | 1 if in DRAM, else 0. Each page is read exactly once | each page is read into the ring once and never re-read. No page is fetched twice: there is no overlap and no shared operand | R-in: none (local read). R-out / R-bal from L1: one mesh hop per page from the owning core or bank |
| output | 1 if in DRAM, else 0. Each page is written exactly once | each page is written once from the ring | R-out: none (local write). R-in / R-bal into L1: one mesh hop per page to the owning core or bank |

Totals per tier, for a tensor of `B = num_pages * page_bytes` bytes:

| Tier | Bytes |
|------|-------|
| DRAM | `B` read (input in DRAM) + `B` written (output in DRAM) |
| Mesh, R-in / R-out | `B` (0 for equal-config sharded) |
| Mesh, R-bal | `2B` when one side is L1-interleaved, `B` when both are, 0 for DRAM → DRAM, which goes through staging L1 only |
| Core-local | `B` of ring copies in R-in / R-out (the loopback side) |

> Cheapest-traffic split considered: owner-aligned with a borrowed shard ring (R-borrow), plus L1-bank-owner ranges for interleaved-L1 sides (R-bank). Deltas: R-borrow saves `B` of core-local L1 copies, with DRAM and mesh unchanged. R-bank saves `B` of mesh bytes on L1-interleaved cells, with DRAM unchanged. Implemented: R-in / R-out / R-bal. Deferred because R-borrow moves the same mesh and DRAM bytes, and its shard-sized ring exceeds the implicit-sync entries-per-txn-id cap, so it needs a second (explicit) protocol. R-bank needs the allocator's L1 bank → core map, which no Python binding exposes. Both stay reachable: the core ranges are already shard-aligned, the kernels already take `page_stride`, and the ring is one named DFB whose spec can be swapped for a borrowed one.

> Verifier disposition (Phase 0 verification, 2026-11-21). The two deferrals above are not both backed
> by a sanctioned reason. R-borrow's reason ("same mesh and DRAM bytes; needs a second protocol") names
> a cost, not unreachability or coverage by a built regime; it is queued as `op_requirements.md`
> Refinement 1 (native height-shard side). R-bank is reachable without an allocator binding:
> `MemoryView.num_banks` gives the bank count and a kernel can find its own bank from the accessor's NoC
> coordinates. It is queued as Refinement 4 (perf, structural gate). The ring-slicing finding (a
> depth-1 ring whose entry count has no factor in {2, 3, 4} gets one txn id, so reads and writes
> serialize) is folded into Refinement 3 and touches this row's `block_pages` / `depth` only.
