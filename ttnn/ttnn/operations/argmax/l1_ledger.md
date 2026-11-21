# L1 Ledger: argmax

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes are named as in `op_design.md` → Blocking Model: `rows`, `w`, `partials`, `scan_threads`. The op has no DataflowBuffers and no compute kernel, so its per-cluster L1 is three DM-only scratchpads and one semaphore. Worked numbers are for `quasar` (32 clusters, `SCAN_THREADS = 6`, S_total = 192, budget 3,897,792 B).

**Inventory, before any solve.** Each scan thread needs (a) somewhere for the NoC to land the bytes it scans, (b) somewhere to stage the uint32 indices a NoC write sources, and (c) the gather needs a landing zone. Reuse options that were considered:
- *Pack into destination*: the output lives in DRAM and a DM hart cannot NoC-write from registers, so staging (b) is required.
- *Transform in place*: the indices are 4 B per row and the block is `row_pitch` bytes per row. Writing a row's index over its first bytes in `sp_blocks` would remove `sp_out_stage`. It is rejected because the next `load_block` (depth 2) or the pending read reuses the slot while the index write is still in flight, and the slot geometry (pitch) would scatter indices so that a run is not contiguous. The staging is ≤ 1/4 of the block's bytes.
- *Alias disjoint lifetimes*: `sp_partials` with `sp_blocks` (the records arrive only after the finalizer's own scan) is rejected. Remote records land at any time during the finalizer's scan, because other scanners finish first, so the lifetimes overlap.
- *Fold into an accumulator*: the running (key, idx) lives in registers, never in L1.

The budget solve sets only the **cap** of `block_bytes` (the coarsest block that fits). In Phase 0 every PHASE0 shape is far below it, and the block is the whole assignment.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `sp_blocks` | `SCAN_THREADS × in_depth` slots of `block_bytes` B, plus one `line` of base-alignment pad. `in_depth = min(IN_DEPTH, max_blocks_per_scanner)`; `block_bytes = min(need, cap)`, a multiple of `line` | `SCAN_THREADS × in_depth` slots: thread t's block being scanned plus, when `in_depth = 2`, its next block in flight | `{rows: spans → block_rows, w: spans → block_cols, partials: streams (none held), scan_threads: spans → SCAN_THREADS slices}` | none (raw bf16; DM-only, so no `data_format_metadata`) | thread t's NoC reads (trid `block_idx % in_depth`) | thread t's `scan_block` | whole program; slot reused per block | **not shared**. It is live at the same time as `sp_out_stage` (a block's indices are written while the next block scans) and as `sp_partials` (remote records land during the finalizer's own scan). Capacity = live set: `in_depth` collapses to 1 when every scanner has one block |
| `sp_out_stage` | `SCAN_THREADS × OUT_DEPTH` slots of `stage_bytes = align_up(out_stage_rows · 4, line)` B, plus one `line` pad. `out_stage_rows = block_rows` (ROWS) or `tensor_rows` (SEGMENTED) | ROWS: 2 slots per thread (one write in flight, one being filled). SEGMENTED: 1 slot of the finalizer thread; the other threads' slots are unused | `{rows: spans → block_rows (ROWS) / tensor_rows (SEGMENTED, ≤ S_total/2), w: streams (one index per row, independent of block_cols), partials: streams, scan_threads: spans → SCAN_THREADS slices}` | none (uint32 indices; uint32 is not a buffer format on Quasar, DM-only) | thread t's stores, flushed by `scoped_lock` release | thread t's NoC writes (trid 2 + k%2) | per block (ROWS); end of program (finalizer) | **not shared**. It is live with `sp_blocks` (a write of block k overlaps the scan of k+1). In SEGMENTED the unused non-finalizer slots are the price of one uniform allocation on every cluster: 6 × 2 × ≤ 384 B. Aliasing them into the dead `sp_blocks` would make the layout regime-dependent for < 5 KB |
| `sp_partials` | landing zone `align_up(S_total · RECORD_BYTES, line)` (one 16-B record per unit) + `SCAN_THREADS` send slots of one `line` each (`send_stride`), plus one `line` pad | finalizer cluster: U ≤ S_total landing slots + 1 send slot. Other clusters: ≤ T send slots (1 per thread, live from its flush to its write barrier) | `{rows: streams (one record per unit, not per block), w: spans → spr segments per row, inside U ≤ S_total, partials: spans → U, scan_threads: spans → T send slots}` | none (uint32 fields) | landing slot u: the remote scanner owning unit u (NoC write, trid 4); send slot t: thread t | landing: the finalizer thread after `wait(U)`; send slot t: thread t's NoC write | SEGMENTED; allocated in both regimes | **not shared**. The landing zone is written remotely while the finalizer's `sp_blocks` and other threads' slots are live (overlapping lifetimes). Capacity S_total ≥ live U: the zone is sized by the grid, not the shape, so it never depends on an op dimension (3,456 B on quasar) |
| `records_arrived` (semaphore) | 1 word per engaged cluster (semaphore region, not this budget) | 1 word | `{rows: streams, w: streams, partials: spans → counts U, scan_threads: streams}` | — | every scanner's remote `up(…, 1)` | finalizer `wait(U)`, `set(0)` | SEGMENTED | **not shared**: a semaphore is its own resource (16 per cluster) |

## Symbol table

| Symbol | Bound | Predicate / source establishing it | Value on `quasar` |
|--------|-------|-----------------------------------|-------------------|
| `SCAN_THREADS` (T) | 1 ≤ T ≤ 6 user DM harts | host constant; `kernel_spec.cpp:124-130` validation | 6 |
| `IN_DEPTH`, `in_depth` | `in_depth ≤ IN_DEPTH = 2`; `IN_DEPTH + OUT_DEPTH + 1 ≤ 8` user trids | host constants, asserted | 2 / 1 for all PHASE0 shapes |
| `OUT_DEPTH` | 2 | host constant | 2 |
| `NC` (clusters) | ≤ grid size | `device.compute_with_storage_grid_size()` or `sub_core_grids` (query) | 32 |
| `S_total` | `NC · T` | derived | 192 |
| `U` (records) | ≤ `S_total` | SEGMENTED predicate `spr ≤ S_total // tensor_rows` | ≤ 192 |
| `tensor_rows` in `sp_out_stage` (SEGMENTED) | ≤ `S_total / 2` | SEGMENTED predicate `spr ≥ 2` | ≤ 96 |
| `budget` | `ttnn.get_max_worker_l1_unreserved_size() − L1_RESERVE_BYTES` | device query | 3,963,328 − 65,536 = 3,897,792 B |
| `line` | `max(l1_align, 64)`: the DM cache line | `line_bytes()`, the single source for host sizes and the kernel's `line_bytes` CT arg | 64 B |
| `cap` | `align_down((budget − partials_bytes − 3·line) / T / (IN_DEPTH + OUT_DEPTH/4), line)`, `partials_bytes = align_up(S_total·16, line) + T·line` | derived from the budget (`derive_blocks`) | 259,584 B (partials_bytes 3,456 B) |
| `block_bytes` | `≤ cap` | `min(need, cap)`, `need` aligned up to `line` | (1,1,2,151936): 3,200; (4,8,32,256): 3,072; (1,1,2048,256): 5,632; (1,1,32,4096): 2,048 |
| `row_pitch` | `≥ l1_align = 16` | `align_up(block_cols·elem_bytes, l1_align)` | 512 for W=256; 96 for W=47; 16 for W=1 |
| `block_rows` | `≤ block_bytes / 16` | `floor(block_bytes / row_pitch)` clamped to the scanner's rows | 6, 11, 1… |
| `tensor_w`, `tensor_rows` (elsewhere) | < 2^32 | `validate()` refuses larger (uint32 index); they never size a buffer except through the predicates above | — |

## Footprint

```
L1(per cluster) = T · in_depth · block_bytes                        # scales with block_bytes (block_rows × block_cols) and in_depth
                + T · OUT_DEPTH · align_up(out_stage_rows · 4, line) # scales with block_rows (ROWS) or tensor_rows ≤ S_total/2 (SEGMENTED)
                + align_up(S_total · 16, line) + T · line            # sp_partials: scales with the grid only
                + 3 · line                                           # one base-alignment pad per scratchpad
                ≤ budget          (by construction of cap: out_stage_rows·4 ≤ block_bytes/4; asserted in derive_blocks)
```

Worked on `quasar` (line = 64 B; figures as allocated, verifier-checked): (1,1,2,151936) 6·3,200 + 6·2·64 + 3,456 + 192 = **23,616 B**. (4,8,32,256) 6·3,072 + 6·2·64 + 3,456 + 192 = **22,848 B**. (1,1,2048,256) 6·5,632 + 6·2·64 + 3,456 + 192 = **38,208 B**. At the cap (a huge input, `in_depth = 2`, rows of ≤ 8 elements): 6·2·259,584 + 6·2·64,896 + 3,456 + 192 = 3,897,408 B ≤ 3,897,792 B.

# Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 1 | single-pass running (key, idx) scan: every element is read once into its scanner's slot and never re-read, in both regimes and at every block size | none |
| `output` | 1 | each index is written once, by its row's scanner (ROWS) or by the finalizer (SEGMENTED) | none |
| partial records (SEGMENTED) | 0 (never touch DRAM) | the gather is L1 → L1 | U × 16 B NoC writes + U remote semaphore increments, fan-in U → 1 cluster. (1,1,2,151936): 3,072 B + 192 atomics. (1,1,32,4096): 2,048 B + 128 |

Totals per tier. DRAM: `tensor_rows·tensor_w·elem_bytes + tensor_rows·4` B (the minimum); (1,1,2,151936): 607,744 B + 8 B; (4,8,32,256): 524,288 B + 4,096 B. NoC cluster-to-cluster: 0 (ROWS) or `16·U` B ≤ 3,072 B (SEGMENTED). Core-local: each element is loaded once by a DM hart from L1 (through its cache), and each index is stored once.

> Cheapest-traffic split considered: `rows across scanners (ROWS)` and `rows to clusters + w within a cluster (INTRA-CLUSTER-W)`, both at the minimum with zero cross-core bytes: SEGMENTED's delta is +16·U B NoC (≤ 3 KB) and +U atomics. Implemented: `ROWS` wherever `spr < 2`, which is also the cheapest-traffic split there. `SEGMENTED` is used only where ROWS would engage fewer than half of the 192 scanners. INTRA-CLUSTER-W is not implemented: it saves those ≤ 3 KB but caps the scanners at `rows × 6` (12 of 192 for the 2-row model call), and SEGMENTED supersedes it (regime table, `rejected`).

## Implementation deltas (ttnn-implementer, phase 0)

- **Cache-line spacing.** A DM hart reaches L1 through a write-back cache of 64-B lines (`internal/tt-2xx/quasar/cache.h`; `L2_CACHE_LINE_SIZE` in `overlay_addresses.h`), and `scoped_lock` release flushes whole lines. A slot one thread writes with the CPU (`sp_out_stage` slots, `sp_partials` send slots) that shares a line with another thread's slot could have that thread's bytes overwritten by a flush. So every per-thread slot is spaced by and aligned to `line = max(l1_align, 64)`: `block_bytes` and `stage_bytes` are rounded up to a line, the send slots start at `align_up(S_total·16, line)` with a stride of one line, and each scratchpad is allocated one line larger so the kernel can align its base up. `row_pitch` (inside one thread's slot) stays 16-B aligned.
- Effect on the worked footprints: `sp_partials` = `align_up(S_total·16, 64) + T·64 (+64 pad)` = 3,456 B (was 3,168); (1,1,2,151936) block 3,200 B (was 3,168); `stage_bytes` ≥ 64 B (was ≥ 16). Every PHASE0 footprint stays below 40 KB per cluster; the cap is derived with the pads subtracted, so the audit `L1 ≤ budget` still holds by construction (asserted in `derive_blocks`).
- `block_bytes_cap` (test/measurement only, default None) lowers the cap so a scanner walks several blocks at depth `IN_DEPTH`; it does not change the inventory.

## Verifier audit (Phase 0)

- Ledger currency: the table, symbol table and footprint above were brought in line with the allocation (the cache-line spacing of the implementation deltas had not been folded into the size expressions, the cap or the worked totals). The cache-line size is now one host function, `line_bytes()`, carried in `Blocks.line_bytes` to both the scratchpad pads and the kernel's CT arg (it was computed three times).
- Capacity vs live set: `sp_out_stage` in SEGMENTED allocates `T × OUT_DEPTH` slots of which the finalizer uses one (≤ 6·2·64 B = 768 B); justified above (one uniform layout). `sp_partials` is allocated in ROWS too (3,456 B); justified (grid-sized, never shape-sized). No other over-capacity row; no CB spans an axis without scaling with it.
- Page format vs DEST width: not applicable (no compute kernel, no DEST; scratchpads carry no format).
- Disjoint lifetimes: every pair has a written reason (above).
- Bounds: every non-block symbol has a bound and its predicate; total closed-form.
