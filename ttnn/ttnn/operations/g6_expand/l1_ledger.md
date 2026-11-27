# L1 Ledger: g6_expand

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes: `src_col` (x pages; `image` and `tile_col` folded), `out_row` (out tile rows per image), `row_in_tile` (whole tile, never split). Regime: `column_group_broadcast` (`op_design.md`, Regimes).

## CB ledger (per core, every core of the rectangle)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_expanded_tiles` | `src_block_depth · block_src_col_tiles` (page = `page_bytes`) | `block_src_col_tiles` pages: the whole block is live from `load_source_block` / `receive_block` until `store_block`'s last write is flushed. Capacity equals the live set at `src_block_depth = 1` (every production shape). At `src_block_depth = 2` the second block is the double-buffer window for the next block's read and multicast. | `{src_col: spans → block_src_col_tiles; out_row: streams → 0 (each E page is re-read from L1 for every out row; no out-row footprint); row_in_tile: spans → whole tile (page granularity); image: folded into src_col}` | input dtype (bf16, `Float16_b`). A dataflow-only payload that never passes through DEST, so the DEST-width audit has nothing to compare: the page is the byte image of x. | reader (sender: DRAM read + in-place expansion; receiver: multicast landing, credit owned by its reader) | writer | whole call (one block in production) | Only CB of the op, so nothing to alias. x pages land in it directly (Rule 3.1, no x staging CB) and are expanded in place (Rule 3.2, no separate E CB). The in-place step runs before `cb_push_back` on the producer thread, so the consumer never sees pre-expansion bytes. |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `page_bytes` | 2048 | SUPPORTED dtype = bfloat16 and validate step 8 (32×32 tile) |
| `core_src_col_tiles` | `ceil(S / min(S, grid_y))`, with `S = d_0·d_1·ceil(W/32)` | `select_plan`; production ≤ 2 |
| `block_src_col_tiles` | `≤ min(core_src_col_tiles, floor(1347776 / (src_block_depth · page_bytes)))` = ≤ 658 at depth 1, ≤ 329 at depth 2 | `select_plan` clamp (BB-05) |
| `src_block_depth` | ∈ {1, 2} | `select_plan`: 1 when `core_src_col_tiles · page_bytes ≤ 1347776`, else 2 |
| `block_out_row_tiles` | `ceil(Hc / min(Hc, grid_x))`; appears in no capacity expression | `select_plan` |
| semaphores (Mcast) | 2 per core (consumer-ready, data-ready), allocated by `Mcast.attach` in the semaphore region | fixed by the channel, independent of every knob |

## Total per-core footprint

`L1_op(core) = src_block_depth · block_src_col_tiles · page_bytes` (+ the channel's 2 semaphores, which live outside the CB region).

- It scales with `block_src_col_tiles` (the `src_col` extent knob) and `src_block_depth`; it does not scale with `out_row`, `num_row_groups`, `num_col_groups` or W beyond the clamp.
- Production: W=512 → 1·2·2048 = **4096 B**; W=256, 196, 207, 185 → 1·1·2048 = **2048 B**; bound 1347776 B (BB-05) holds with 0.3 % used.
- The clamp makes every shape fit: past it the core streams over `src_col` (more blocks), never refuses.

## Data-movement budget (`column_group_broadcast`, Phase 0 plan)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x (`input_tensor`) | 1 — each x page read once, by its column group's sender (W=512: 16 pages, 32 KB; W=256: 8 pages, 16 KB; W=196/207: 7 pages, 14 KB; W=185: 6 pages, 12 KB) | the expanded tiles stay resident in the group's CBs and are delivered by multicast instead of re-read; a core writing several out rows re-reads E from L1, not DRAM | per group per block: one multicast of `block_src_col_tiles · 2 KB` to `num_row_groups − 1` receivers, `num_row_groups − 1` ready increments, 1 flag multicast. W=512: 8 × 4 KB to 7 receivers each = 224 KB delivered (32 KB injected); W=256: 8 × 2 KB × 7 = 112 KB; W=196/207: 7 × 2 KB × 6 = 84 KB; W=185: 6 × 2 KB × 5 = 60 KB |
| out | 1 — each out page written once, by exactly one core, as a whole 2 KB page (W=512: 256 pages, 512 KB; W=256: 64 pages, 128 KB; W=196/207: 49 pages, 98 KB; W=185: 36 pages, 72 KB) | out is never read; no intermediate tensor touches DRAM | none |
| E (intermediate, L1 only) | 0 | built in place inside `cb_expanded_tiles` | sender-local NoC self-reads ≈ 1.9 KB per E tile (fan-out 896 B at `seed_rows = 2` + lower faces 1024 B) |

Totals per call, W=512 (the PERF FOCUS case): DRAM 32 KB read + 512 KB written = 544 KB, which is the minimum (each tensor crosses once); NoC 32 KB multicast-injected (224 KB delivered) + ≈ 30 KB of sender-local self-reads.

> Cheapest-traffic split considered: `src_col` only (`num_row_groups = 1`, column owners) — same DRAM bytes (x once, out once), 0 cross-core bytes (−224 KB delivered / −32 KB injected at W=512, −56 ready increments, −8 flags). Implemented: `column_group_broadcast`; its `num_row_groups = 1` setting **is** that split (same kernels, channel absent), so the cheapest split is not deferred but reachable by the knob alone. Phase 0 defaults to the full-grid `num_row_groups` because the spec requires the core count to be the blocking decision's choice rather than the `S`-column limit (6–16 cores), and the op is DRAM-write-bound with out ≈ 16× x. Lamp L1 (`op_design.md`) measures `num_row_groups ∈ {1, 2, 4, full}`.
