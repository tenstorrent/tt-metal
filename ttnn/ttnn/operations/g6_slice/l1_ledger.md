# L1 Ledger: g6_slice

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md` → Blocking Model: `out_page` (the fold of `D0, D1, H, W`; extent `block_pages`) and the `stencil` footprint factor (`sources_per_output = (1 + needs_right_any)(1 + needs_below_any)`; `H` contributes `1 + needs_below_any`, `W` contributes `1 + needs_right_any`).

No compute kernel runs, so no page passes through DEST: every page format is `x`'s own storage format (a dataflow-only payload — audit 2 does not apply; `fp32_dest_acc_en` is not set).

## Inventory

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_staged_tiles` | `buffer_depth · sources_per_output · block_pages` × `in_page_bytes` | `buffer_depth · sources_per_output · block_pages`: at depth 2 the reader fills block k+1 while the writer still holds block k; at depth 1 one block. Slots of stencil pages a given output page does not need are reserved but unread (nominal push count; the needed set varies per page). In row_shift a slot carries only the covering byte range, but stays a full page so the read lands at the source's own in-page offset (64-B read congruence) | `{out_page: spans → block_pages; H: spans → 1 + needs_below_any; W: spans → 1 + needs_right_any; D0: folded into out_page; D1: folded into out_page}` | `x.dtype` (bfloat16 in Phase 0) | reader | writer | whole launch, every regime | none. It is the only CB of page_copy and row_shift (the staged page *is* the output page, or the source of its write segments — pattern 1, no separate output CB). In element_shift it cannot alias `cb_assembled_tiles`: the writer reads it while writing that one (concurrent lifetime) |
| `cb_assembled_tiles` | `(1 + needs_below_any) · block_pages` × `out_page_bytes` | the same: the writer composes the whole block, issues every write, flushes once, pops | `{out_page: spans → block_pages; H: spans → 1 + needs_below_any; W: streams (the right stencil page is folded into each assembled page by the column copy); D0: folded into out_page; D1: folded into out_page}` | `x.dtype` | writer | writer | element_shift launches only (allocated only then) | none: concurrent with `cb_staged_tiles` (above). A 2-page ring with a flush per page would drop it to 2 pages at the cost of one flush per page instead of one per block — rejected: the stated frequency is one flush per block |
| `cb_begin_index` (refinement) | 1 × `round_up(4·r, DRAM_ALIGNMENT)` (64 B) | 1 — constant (one `starts` stick) | `{out_page: streams (read once per launch, independent of the block); H, W, D0, D1: not spanned}` | uint32 | reader | reader | reader prologue (device form only) | could alias slot 0 of `cb_staged_tiles` (disjoint lifetime: consumed before the first `reserve`); not done — it saves 64 B and would add an aliasing contract |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `block_pages` | `1 ≤ block_pages ≤ block_pages_cap` | `plan_blocks()`: `block_pages = ceil(pages_per_core / ceil(pages_per_core / block_pages_cap))` |
| `block_pages_cap` | `floor((L1_OP_BUDGET − fixed_bytes) / (MAX_BUFFER_DEPTH · sources_per_output · in_page_bytes + assembled_per_output · out_page_bytes))` | BB-05 |
| `L1_OP_BUDGET` | 1347776 B | contract (model leaves this per core) |
| `MAX_BUFFER_DEPTH` | 2 | host constant |
| `buffer_depth` | ∈ {1, 2} | `1` iff `ceil(pages_per_core / block_pages_cap) == 1` |
| `sources_per_output` | ∈ {1, 2, 4} | regime: page_copy 1; row_shift `1 + needs_below_any`; element_shift `(1 + needs_right_any)(1 + needs_below_any)` |
| `assembled_per_output` | element_shift: `1 + needs_below_any` ∈ {1, 2}; else 0 | regime |
| `in_page_bytes`, `out_page_bytes` | 2048 (bfloat16, Phase 0); 4096 (float32/int32/uint32); 1088 (bfloat8_b); 576 (bfloat4_b) | dtype (`x.buffer_page_size()`; equal for `x` and `out`) |
| `fixed_bytes` | Phase 0: 0. Device form: + 64. L1-interleaved `out`: + `ceil(total_out_pages / num_l1_banks) · out_page_bytes`. Width-sharded `out`: + output shard bytes on the core | placement / form |
| `pages_per_core` | `ceil(total_out_pages / num_cores)` ≤ 2533 for every INPUTS shape at ≥ 110 cores | `split_work_to_cores` |
| `r` | 2 ≤ r ≤ 4 | `rank` axis |

## Total per-core footprint

```
L1(core) = buffer_depth · sources_per_output · block_pages · in_page_bytes        # scales with block_pages, depth, stencil
         + assembled_per_output · block_pages · out_page_bytes                    # element_shift only; scales with block_pages
         + fixed_bytes                                                            # begin stick, L1 out share (refinement)
         ≤ L1_OP_BUDGET = 1347776
```

Phase 0 maxima (bfloat16, DRAM, `fixed_bytes = 0`):

| Regime | `sources_per_output` | `block_pages_cap` | Max L1 per core |
|--------|----------------------|-------------------|-----------------|
| page_copy | 1 | 329 | 2·1·329·2048 = 1,347,584 B |
| row_shift | 1 | 329 | 1,347,584 B |
| row_shift | 2 | 164 | 2·2·164·2048 = 1,343,488 B |
| element_shift | 2 (no below) | 131 | 2·2·131·2048 + 1·131·2048 = 1,341,440 B |
| element_shift | 4 | 65 | 2·4·65·2048 + 2·65·2048 = 1,331,200 B |

Flagged p1 (703) at 110 cores: `pages_per_core = 2533` → 8 blocks of 317 → 2·317·2048 = 1,298,432 B. A single-block core (851 of 868 Phase 0 cases at 110 cores) holds `pages_per_core · sources_per_output · 2048` B at depth 1.

## Data-movement budget

Chosen split: contiguous ranges of the flat output-page index (S1). Every term counts the most expensive tier first.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `x` — page_copy | each selected page once; unselected pages 0 | the gather reads `A(p)` once per output page and pages map 1:1 | 0 |
| `x` — row_shift | one covering sub-page range per needed source page per output page; production (`valid_rows = 1`): 544 B per output page (bf16) | sub-page read at the source's own offset; `B` read only when it holds a selected row (no production case) | 0 |
| `x` — element_shift | `sources_per_output` whole pages per output page (≤ 4) | per-output staging without in-block reuse (R3r deferred) | 0 |
| `out` — page_copy, element_shift (`row_offset = 0`) | each page once | whole-page writes | 0 |
| `out` — row_shift, element_shift (`row_offset > 0`) | valid rows only: `valid_rows · 2 · FR` B per page | segment writes; padding rows not written | 0 |
| `starts` (refinement) | `num_cores` reads of one 64-B stick | each core reads the begin itself (`begin_broadcast` rejected) | 0 |

Totals per tier for the flagged cases (all page_copy, bf16):

| Case | `x` read (DRAM) | `out` written (DRAM) | Cross-core |
|------|-----------------|----------------------|------------|
| 703 / 704 (p1/p2) `(16,4096,8704)` half | 278528 pages = 570.4 MB | 570.4 MB | 0 |
| 733 (p3) `(16,4096,4160)` cols 0..2560 | 163840 pages = 335.5 MB | 335.5 MB | 0 |
| 732 (p3) cols 2560..4096 | 98304 pages = 201.3 MB | 201.3 MB | 0 |
| 730 / 731 (p3) 12 columns | 2048 pages = 4.2 MB | 4.2 MB | 0 |
| 795 / 796 (p4) `(8,4096,8704)` half | 139264 pages = 285.2 MB | 285.2 MB | 0 |
| 587 (row 4095, row_shift) | 640 × 544 B = 348 KB | 640 × 64 B = 41 KB | 0 |

> Cheapest-traffic split considered: S1 (flat output-page ranges) — it reads each selected page once and writes each output page once, 0 cross-core bytes; every alternative either ties on bytes (S2–S4, worse balance) or adds a full extra read of `x` (S5, +570.4 MB on p1). Implemented: S1. The implemented split is also the cheapest. The only traffic above the minimum inside S1 is the stencil re-read of row_shift (multi-row selections) and element_shift, which no production case incurs; their removal is the `row_shift_window` / `element_shift_reuse` deferred rows, reachable because the reader already walks `(o0, o1, i, j)` and only the traversal order and slot retention change.
