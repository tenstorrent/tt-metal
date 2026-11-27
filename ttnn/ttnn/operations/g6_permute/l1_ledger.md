# L1 Ledger: g6_permute

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions: `op_design.md` → Blocking Model. Built regime: `rows_tile_scatter` (Phase 0).

## CB ledger

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `min(input_depth, max_blocks_per_core) · block_in_tiles` pages of `in_page_bytes` (production: 1 page) | `min(input_depth, max_blocks_per_core) · block_in_tiles`: with ≥ 2 blocks per core the reader fills block `k+1` while the writer scatters block `k`, so both are live; capacity = live set | `{in_tile: spans → block_in_tiles · min(input_depth, max_blocks_per_core); r: spans → 32 (whole tile slot; the prefix fills rows_live(bt) of them); h: spans → 2; n, a, bt, wt: folded into in_tile}` | Float16_b — a dataflow-only payload; no compute kernel exists, so there is no DEST width to match (audit 2 does not apply) | reader | writer | whole program | Nothing to share with: it is the op's only buffer. No output CB exists — rows go from this slot straight to their output pages in DRAM (pack-into-destination, Rule 3.1); no second lifetime exists to alias |

## Symbol table

| Symbol | Bound | Established by |
|--------|-------|----------------|
| `in_page_bytes` | `input_tensor.buffer_page_size()` = 2048 | Phase 0 SUPPORTED dtype bfloat16, 32×32 tiles (validate) |
| `input_depth` | 2 | host constant |
| `max_core_in_tiles` | `ceil(tensor_in_tiles / num_cores)` = `tiles_g1` of `split_work_to_cores` | work split |
| `block_in_tiles` | `min(max_core_in_tiles, block_cap)`, ≥ 1 | Work Distribution formula (single source) |
| `max_blocks_per_core` | `ceil(max_core_in_tiles / block_in_tiles)` | derived |
| `block_cap` | `floor((L1_OP_BUDGET_BYTES − out_l1_bytes_per_core) / (input_depth · in_page_bytes))` = 329 in Phase 0 | mechanism cap (BB-05) |
| `L1_OP_BUDGET_BYTES` | 1347776 | model environment: L1 left per core, output's own L1 pages included |
| `out_l1_bytes_per_core` | 0 in Phase 0 (output in DRAM); `ceil(tensor_out_tiles / num_l1_banks) · out_page_bytes` for buffer `"l1"` | output placement |

## Total per-core footprint

`min(input_depth, max_blocks_per_core) · block_in_tiles · in_page_bytes + out_l1_bytes_per_core ≤ input_depth · block_cap · in_page_bytes + out_l1_bytes_per_core ≤ L1_OP_BUDGET_BYTES`.

The first term scales with `block_in_tiles` (block knob) and `input_depth` (depth knob); the second with the output placement only. Every production shape: `1 · 1 · 2048 = 2048` B per core.

## Data-movement budget (`rows_tile_scatter`)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `x` | 1 — the live prefix of each tile, one transaction per tile | each input tile belongs to exactly one core and is read once; its rows go straight to their destinations, so nothing is re-read | 0 |
| `out` | 1 — exactly the logical bytes, 32 B per transaction | each half-row is written once, from the slot that holds it; output padding rows are never written | 0 |

General totals per call: DRAM read `Σ_tiles prefix_bytes(rows_live(bt))`; DRAM write `tensor_n · tensor_a · tensor_b · tensor_w · 2` B in `2 · tensor_n · tensor_a · tensor_b · tensor_wt` writes; NoC cross-core 0; core-local copies 0.

| Case | Input `[N, A, B, W]` | Input tiles (= cores) | DRAM read (B) | DRAM write (B) | Write transactions |
|------|----------------------|-----------------------|---------------|----------------|--------------------|
| 301 | `[1, 1, 16, 64]` | 2 | 2048 | 2048 | 64 |
| 302 | `[1, 16, 1, 64]` | 32 | 17408 | 2048 | 64 |
| 303 | `[1, 6, 16, 64]` | 12 | 12288 | 12288 | 384 |
| 304 | `[1, 16, 6, 64]` | 32 | 22528 | 12288 | 384 |
| 305 | `[1, 1, 8, 64]` | 2 | 1536 | 1024 | 32 |
| 306 | `[1, 8, 1, 64]` | 16 | 8704 | 1024 | 32 |
| 307 | `[1, 6, 8, 64]` | 12 | 9216 | 6144 | 192 |
| 308 | `[1, 8, 6, 64]` | 16 | 11264 | 6144 | 192 |

Reads exceed the logical bytes only by padding rows inside a prefix (`(16 − B) · 32` B per tile for `B < 16`); writes equal them.

> Cheapest-traffic split considered: input tiles with rows scattered straight to the output (`rows_tile_scatter`) — x once (live prefix), out once (logical bytes), 0 cross-core. Implemented: the same split. The alternatives cost more on some tier: `rows_tile_group_assemble` writes whole output tiles (303: 65536 B against 12288 B), `rows_tile_gather` reads 2× through 64 B bounces, `rows_tile_reread` re-reads x up to 32×, `rows_tile_mcast` adds 32× cross-core bytes.
