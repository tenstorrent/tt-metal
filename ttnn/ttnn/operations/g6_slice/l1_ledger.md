# L1 Ledger: g6_slice

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions: `op_design.md` → *Blocking Model*.
Block axes: one named block axis, `out_page`; `lead0`, `lead1`, `h`, `w` are folded into it (fold justified in `op_design.md`).
No compute kernel exists, so no CB passes through DEST: every page format is a dataflow-only payload in `x`'s own dtype.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_slice_pages` | `cb_depth × block_pages`, `cb_depth = min(CB_DEPTH, num_blocks_max)` | `cb_depth × block_pages` — slot k being stored and slot k+1 being fetched are resident at once; a core with one block has only one slot live, so its ring is one block deep | `{out_page: spans → block_pages × cb_depth; lead0: folded into out_page; lead1: folded into out_page; h: folded into out_page; w: folded into out_page}` | `x.dtype` (bf16 in Phase 0), page = `x.buffer_page_size()`; dataflow-only payload, never in DEST | reader | writer | whole program | Shares with nothing, and needs no partner: it **is** the destination — the reader's DMA lands bytes in the page the writer sends (Rule 3 pattern 1), so there is no input/output CB pair to merge. The device-form `starts` read (refinement) aliases slot 0 before the first fetch (disjoint lifetime, Rule 3 pattern 3) instead of adding a CB. Cannot alias `cb_right_col_tile`: both are live during `realign_columns_block` (destination and second source). |
| `cb_right_col_tile` | `block_pages` when `W_SHIFT` (regime `col_realign`); not allocated otherwise | `block_pages` — one right-column input tile per page of the block between its fetch and its realign | `{out_page: spans → block_pages (one block); lead0: folded into out_page; lead1: folded into out_page; h: folded into out_page; w: folded into out_page}` | `x.dtype`; dataflow-only payload | reader | reader (private scratch, no handshake) | `col_realign` programs; per block, from `fetch_block` to the end of `realign_columns_block` | Cannot share with `cb_slice_pages` (concurrently live, see above). The left input column is fetched straight into the destination slot and realigned in place (Rule 3 pattern 2) — that is why one scratch CB exists, not two. Single block deep: the same thread consumes it before the next fetch, so a second slot would never be live. |

Capacity equals live set in both rows (audit 1), including single-block cores (depth clamped to 1). No 16/32-bit DEST width question applies (audit 2: dataflow-only payloads). The only CB pair has overlapping lifetimes (audit 3).

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `page_bytes` | bf16 2048; ≤ 4096 over TARGET (fp32/int32/uint32 4096, bf8_b 1088, bf4_b 576) | dtype ∈ TARGET |
| `cb_depth` | `min(CB_DEPTH, num_blocks_max)` ∈ {1, 2} | `CB_DEPTH` = 2; 1 when every core holds a single block (the second slot would never be live) |
| `slots_per_page` | `cb_depth` (+1 when `W_SHIFT`); the L1 bound on `block_pages` is solved with `CB_DEPTH`, before `cb_depth` is clamped | regime |
| `block_pages` | `≤ min(core_pages_max, block_pages_cap)`; default 32 | `_plan()` |
| `block_pages_cap` | `min(BLOCK_PAGES_CAP, floor((L1_budget − fixed_bytes) / ((CB_DEPTH + W_SHIFT) × page_bytes)))`; L1 bound bf16: 329 (`page_copy`/`row_runs`), 219 (`col_realign`); `BLOCK_PAGES_CAP` = 32 (overlap lamp). Measured, same-session paired A/B at 1350 MHz, 3 reps each: 32 beats 329 by 1.5 % on 703 (2669.6 vs 2710.4 µs), 1.4 % on 795, 2.1 % on 1112, 1.2 % on 179, tie on 730; spreads < 0.5 %. The edge is consistent but under the campaign's 2–3 % win threshold, so 32 is kept as a tie that costs 10× less L1 (131 KB vs 1.35 MB per core), not as a speedup | BB-05 + overlap lamp |
| `L1_budget` | `min(1347776, ttnn.device.get_max_worker_l1_unreserved_size())` | BB-05 |
| `fixed_bytes` | `out_l1_bytes_per_core + 64 × num_cbs` (CB allocation alignment) | — |
| `out_l1_bytes_per_core` | 0 for DRAM `out` (Phase 0); L1 interleaved (refinement): `ceil(total_out_pages / num_l1_banks) × aligned_page_bytes`; width-sharded: one output shard | `memory_config` |
| `core_pages_max` | `ceil(total_out_pages / num_cores)` | `split_work_to_cores` |
| `total_out_pages` | `M4[0]·M4[1]·ceil(M4[2]/32)·ceil(M4[3]/32)`; ≤ 278528 over the golden INPUTS (case 703) | golden INPUTS |
| `num_cores` | `≤ grid.x · grid.y` (≤ 130 on Blackhole) | `MAX_CORES` / grid |
| `num_cbs` | 1, or 2 when `W_SHIFT` | regime |

## Total per-core footprint

```
L1_op = slots_per_page × block_pages × page_bytes  +  ≤ 64 × num_cbs (CB alignment)   ≤  L1_budget − out_l1_bytes_per_core
```

- `cb_slice_pages` scales with `block_pages` and `cb_depth`; `cb_right_col_tile` with `block_pages` only.
- Nothing scales with `N` or `M` except through `block_pages ≤ core_pages_max`, which the `block_pages_cap` clamp bounds — no golden shape (including `N_{r-2} = 8448` and `N_{r-1} = 62080`) can push the footprint past the budget.
- Phase 0 default (bf16, `BLOCK_PAGES_CAP = 32`): `2 × 32 × 2048 = 131,072 B` (`page_copy`/`row_runs`), `3 × 32 × 2048 = 196,608 B` (`col_realign`). Single-block assignments (`core_pages ≤ 32`, e.g. 587: 6 pages per core) take `1 × block_pages × page_bytes` (587: 12,288 B). With `BLOCK_PAGES_CAP = None` the L1 bound takes over: `2 × 329 × 2048 = 1,347,584 B ≤ 1,347,712 B`; `col_realign`: `3 × 219 × 2048 = 1,345,536 B`.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `x`, `page_copy` | 1 × each selected page | each output page has exactly one source page; nothing is re-read, so no residency decision exists | none |
| `x`, `row_runs` | 1 × each selected row-slice (`rv × 64 B` per page for bf16) | only rows that land in logical output rows are read; a shifted output page reads its two complementary bands, never a whole tile | none |
| `x`, `col_realign` | ≤ 2 × the selected face halves | the left and right input tile columns are fetched per output tile; reusing the right page as the next left page is `col_realign_sliding` (deferred) | none |
| `out`, `page_copy` | 1 × each output page | written once, whole | none |
| `out`, `row_runs` / `col_realign` | 1 × each logical row-slice | padding rows are never written | none |
| `starts` (device form, refinement) | 64 B × `num_cores` per call | each core reads it itself; a broadcast would add a rendezvous to save 64 B per core | none |

Totals per tier:

| Case | DRAM read | DRAM write | Cross-core |
|------|-----------|------------|------------|
| p1, 703 `(16,4096,8704)[..., 4352:]` | 570,425,344 B | 570,425,344 B | 0 B |
| 587 `(16,4096,1280)[:, 4095:]` | 640 × 64 = 40,960 B | 40,960 B | 0 B |
| 1175 `(1,1,16,62080)[..., 15:16, :]` | 1940 × 64 = 124,160 B | 124,160 B | 0 B |

> Cheapest-traffic split considered: contiguous output-page ranges — every candidate split (lead-only, tile rows, tile columns, 2-D rectangles, input-page ownership) moves the same bytes per tier, delta 0 B DRAM / 0 B NoC. Implemented: contiguous output-page ranges. The implemented split is the cheapest; it wins the tie on occupancy (full grid down to one-tile-row outputs) and balance (±1 page).
