# L1 Ledger: g6_zeros

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes are from `op_design.md` → Blocking Model:
- `device`
- `bank` (block extent 1)
- `slot` (block extent `core_slots`)
- `page`, re-indexed to `(bank, slot)`
- `d0`, `d1`, `tile_row`, `tile_col`, all folded into `page`

**Inventory.** The program has one CB. Two buffers were considered and do not exist:
- **A FIFO of zero tiles between a compute kernel and a writer.** There is no compute stage, and the bytes never change. The source is written once and read by every write.
- **A second region for double buffering.** Nothing refills the source, so a second region has nothing to overlap.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_zero_source` | `zero_source_regions` = 1 page of `zero_source_bytes` = 2048 B. A constant, not an op dimension; bounded by the L1 share. Sized by measurement: the fill runs before the first write (fixed cost on p4), and 2 KiB packets still keep BRISC issue ahead of the bank (1 KiB is issue-bound on p1). | `fill_bytes = min(zero_source_bytes, core_run_bytes)` = 2048 B on every production case (every run is ≥ 32 KiB). The whole filled prefix is live at once, as the source of every in-flight write. | `{device: streams → one chip's program`<br>`bank: streams → 1 bank per block`<br>`slot: streams → the run is walked in zero_source_bytes chunks; the buffer does not scale with core_slots`<br>`page: streams (re-indexed)`<br>`d0, d1, tile_row, tile_col: streams (folded into page)}` | Output `dtype` tag (float32, bfloat16 or bfloat8_b). A dataflow-only payload: no compute kernel exists, nothing goes through DEST, and `fp32_dest_acc_en` is undefined. Audit 2 does not apply. | writer (`fill_zero_source`) | writer (NoC source of `store_bank_run`) | whole kernel invocation: filled first, read by every write, never popped | No other CB exists to share with. Capacity equals the live set on every production case. |

## Symbol table

| Symbol | Meaning | Bound | Predicate that establishes the bound |
|--------|---------|-------|--------------------------------------|
| `zero_source_bytes` | host constant; the single source of the zero-region size and of the write chunk | 64 ≤ `zero_source_bytes` ≤ 1,347,776, and a multiple of 64 | Host asserts against the model's L1 share (BB-05) and `ttnn.get_dram_alignment()`. Phase 0 = 16384; Refinement 2 = 2048 (measured: equal on p1/p2/p3, fastest on p4). |
| `zero_source_regions` | depth of `cb_zero_source` | = 1 | The source never changes, so there is no refill to overlap. |
| `core_slots` | slots in this core's run (runtime arg) | ≤ `ceil(slots_b / writers_per_bank)` ≤ 32,831 in production (`writers_per_bank` = 1) | Host plan. Appears in no capacity expression. |
| `core_run_bytes` | `core_slots · aligned_page_size` | ≤ 35,720,128 B in production | Only bounds `fill_bytes`, through `min(...)`, never the capacity. |
| `aligned_page_size` | DRAM page stride | 4096 / 2048 / 1088 | dtype. Appears in no capacity expression. |
| `N`, `writers_per_bank`, `num_pages` | plan parameters | `N` = allocator DRAM banks. `writers_per_bank ≤ floor(grid_cores / N)`. `num_pages` is unbounded by the contract. | None of them appears in a capacity expression. The footprint does not grow with any of them. |

## Total per-core footprint

```
L1(core) = zero_source_regions · zero_source_bytes = 1 · 2048 B = 2,048 B
```

The total scales only with the `zero_source_bytes` knob. It is independent of `shape`, `dtype`, `num_pages`, grid size and mesh size. Every case of the golden INPUTS fits the 1,347,776 B model share with 1,345,728 B to spare.

## Data-movement budget (chosen split: bank-major `bank_run`, 1 writer per bank on its NoC1-optimal worker, per device)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| output: p1 285,761,024 B, p2 12,582,912 B, p3 6,291,456 B, p4 262,144 B | **1** (write) | Each page is written exactly once, from core-local zeros. There is no clearing pass, and pages are never read back. | 0 |
| input | **0** | The op has no input and reads no DRAM. | 0 |
| zero source (2 KiB per core) | **0** | Each core synthesizes it from the firmware zero region by L1 loopback reads: the core-local tier. | 0 |

Totals per tier, per device:

| Case | DRAM write | DRAM read | Cross-core NoC | Core-local L1 loopback | NoC write packets |
|------|------------|-----------|----------------|------------------------|-------------------|
| p1 | 285,761,024 B | 0 | 0 | 8 × 2048 = 16,384 B | 139,536 (vs 262,648 page-at-a-time) |
| p2 | 12,582,912 B | 0 | 0 | 16,384 B | 6,144 |
| p3 | 6,291,456 B | 0 | 0 | 16,384 B | 3,072 |
| p4 | 262,144 B | 0 | 0 | 16,384 B | 128 |

The packet count is not the binding cost: p1 writes at 511 GB/s (DRAM peak 512 GB/s) with 2 KiB packets, the same as with 16 KiB packets. Phase 0's 104 writers sent 17,472 16 KiB packets on p1 and reached only 257 GB/s, because 13 NoC1 paths converged on each bank endpoint.

> Cheapest-traffic split considered: bank-major `bank_run`. All candidate splits tie on the DRAM minimum (output written once, 0 B read, 0 B cross-core). Bank-major moves the fewest NoC transactions (17,472 against 262,648 for page-at-a-time on p1). Implemented: bank-major `bank_run`. It is also the cheapest-traffic split, so no traffic-cheaper alternative is deferred.
