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
| `cb_zero_source` | `zero_source_regions` = 1 page of `zero_source_bytes` = 16384 B. A constant, not an op dimension; bounded by the L1 share and sized to one max NoC packet. | `fill_bytes = min(zero_source_bytes, core_run_bytes)`: 16384 B for p1/p2/p3, 2048–4096 B for p4. The whole filled prefix is live at once, as the source of every in-flight write. | `{device: streams → one chip's program`<br>`bank: streams → 1 bank per block`<br>`slot: streams → the run is walked in zero_source_bytes chunks; the buffer does not scale with core_slots`<br>`page: streams (re-indexed)`<br>`d0, d1, tile_row, tile_col: streams (folded into page)}` | Output `dtype` tag (Float32 in Phase 0). A dataflow-only payload: no compute kernel exists, nothing goes through DEST, and `fp32_dest_acc_en` is undefined. Audit 2 does not apply. | writer (`fill_zero_source`) | writer (NoC source of `store_bank_run`) | whole kernel invocation: filled first, read by every write, never popped | No other CB exists to share with. Capacity exceeds the live set only on p4 (2–4 KiB runs). One CB size serves the whole core range, and the fill writes only the live prefix, so the extra L1 costs no time. |

## Symbol table

| Symbol | Meaning | Bound | Predicate that establishes the bound |
|--------|---------|-------|--------------------------------------|
| `zero_source_bytes` | host constant; the single source of the zero-region size and of the write chunk | 64 ≤ `zero_source_bytes` ≤ 1,347,776, and a multiple of 64 | Host asserts against the model's L1 share (BB-05) and `ttnn.get_dram_alignment()`. Phase 0 = 16384 = Blackhole `NOC_MAX_BURST_SIZE`. |
| `zero_source_regions` | depth of `cb_zero_source` | = 1 | The source never changes, so there is no refill to overlap. |
| `core_slots` | slots in this core's run (runtime arg) | ≤ `ceil(slots_b / writers_per_bank)` ≤ 2526 in production | Host plan. Appears in no capacity expression. |
| `core_run_bytes` | `core_slots · aligned_page_size` | ≤ 2,748,288 B in production | Only bounds `fill_bytes`, through `min(...)`, never the capacity. |
| `aligned_page_size` | DRAM page stride | 4096 / 2048 / 1088 | dtype. Appears in no capacity expression. |
| `N`, `writers_per_bank`, `num_pages` | plan parameters | `N` = allocator DRAM banks. `writers_per_bank ≤ floor(grid_cores / N)`. `num_pages` is unbounded by the contract. | None of them appears in a capacity expression. The footprint does not grow with any of them. |

## Total per-core footprint

```
L1(core) = zero_source_regions · zero_source_bytes = 1 · 16384 B = 16,384 B
```

The total scales only with the `zero_source_bytes` knob. It is independent of `shape`, `dtype`, `num_pages`, grid size and mesh size. Every case of the golden INPUTS fits the 1,347,776 B model share with 1,331,392 B to spare.

## Data-movement budget (chosen split: bank-major `bank_run`, per device)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| output: p1 285,761,024 B, p2 12,582,912 B, p3 6,291,456 B, p4 262,144 B | **1** (write) | Each page is written exactly once, from core-local zeros. There is no clearing pass, and pages are never read back. | 0 |
| input | **0** | The op has no input and reads no DRAM. | 0 |
| zero source (16 KiB per core) | **0** | Each core synthesizes it from the firmware zero region by L1 loopback reads: the core-local tier. | 0 |

Totals per tier, per device:

| Case | DRAM write | DRAM read | Cross-core NoC | Core-local L1 loopback | NoC write packets |
|------|------------|-----------|----------------|------------------------|-------------------|
| p1 | 285,761,024 B | 0 | 0 | 104 × 16,384 = 1,703,936 B | 17,472 (vs 262,648 page-at-a-time) |
| p2 | 12,582,912 B | 0 | 0 | 1,703,936 B | 832 |
| p3 | 6,291,456 B | 0 | 0 | 1,703,936 B | 416 |
| p4 | 262,144 B | 0 | 0 | 262,144 B | 104 |

The p4 loopback figure is 24 × 4096 + 80 × 2048 B of fill (8 banks × 3 two-slot and 10 one-slot writers). It equals the output size because each p4 fill equals its run.

> Cheapest-traffic split considered: bank-major `bank_run`. All candidate splits tie on the DRAM minimum (output written once, 0 B read, 0 B cross-core). Bank-major moves the fewest NoC transactions (17,472 against 262,648 for page-at-a-time on p1). Implemented: bank-major `bank_run`. It is also the cheapest-traffic split, so no traffic-cheaper alternative is deferred.
