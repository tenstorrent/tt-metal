# L1 Ledger: g6_concat

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions:
`op_design.md` → Blocking Model. Block axes: `out_page` (fold of `lead`, `join`, `trail`, `tile_row`,
`tile_col`), `row_in_tile`, `source`.

## CB table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_concat_tiles` (index 0, page `tile_bytes`) | `depth_eff · block_tiles`, `depth_eff = min(cb_depth, ceil(max_core_tiles / block_tiles))` | `depth_eff · block_tiles` pages: block `i` held by the writer (place + store) while the reader fills block `i+1`; `block_tiles` when the core has one block | `{out_page: spans → block_tiles; row_in_tile: spans → 32 rows (a page is one tile; only rows [0, m) carry data); source: streams — every source of a tile lands in that tile's one page; lead, join, trail, tile_row, tile_col: folded into out_page}` | input dtype (`Float16_b` in Phase 0). Dataflow-only payload, never through DEST — no compute kernel, `fp32_dest_acc_en` does not exist | reader | writer | whole program, both regimes | Is the destination: direct reads land in it, `place_rows_block` transforms it in place, `store_block` writes from it (reuse patterns 1 + 2). Cannot alias `cb_row_staging`: concurrent lifetime — rows are copied from staging into it while both are held by the writer. Fold-into-DEST does not apply (nothing computed) |
| `cb_row_staging` (index 1, page `tile_bytes`; **row_gather only**) | `depth_eff · block_tiles · S` | ≤ `depth_eff · block_tiles · S` pages; a page holds only the rows staged for one (shifted input, output tile). Slots beyond the shifted-input count of a tile row stay unused | `{out_page: spans → block_tiles; source: spans → S; row_in_tile: spans → 32 rows (staged rows sit at their source intra-tile offsets); lead, join, trail, tile_row, tile_col: folded into out_page}` | input dtype; dataflow-only | reader | writer | whole program, row_gather only (not allocated in page_run) | Cannot alias `cb_concat_tiles` (concurrent, above). Capacity exceeds the bytes staged (Qwen 79: 64 B of each 2 048 B page): **mechanism reason** — Blackhole DRAM reads require L1 and DRAM offsets equal mod 64, so a row is staged at its source offset in a tile-sized slot. The fixed per-tile stride `S` keeps slot addressing static and every push nominal for the ring invariant. Packing rows of several sources into one slot at matching residues is the compaction alternative; not built because Qwen 79's total (589 824 B) fits the budget |

## Symbol table

| Symbol | Meaning | Bound | Predicate / source establishing it |
|--------|---------|-------|------------------------------------|
| `block_tiles` | output pages per block (knob) | `1 ≤ block_tiles ≤ min(max_core_tiles, floor(1347776 / (cb_depth · tile_bytes · (1 + S))))` | host plan; the cap clamp in Mechanism caps |
| `cb_depth` | buffer-depth knob | `= 2` (Phase 0) | host plan constant |
| `depth_eff` | depth actually allocated | `min(cb_depth, num_blocks_max)` ∈ {1, 2} | derived from `block_tiles` |
| `S` | `staging_slots_per_tile` | `0` in page_run; `1 ≤ S ≤ min(K, 32)` in row_gather | each source contributes ≥ 1 of 32 rows; regime predicate |
| `tile_bytes` | page size | 2 048 (bf16, Phase 0); 4 096 (fp32); 1 088 (bfp8) | dtype |
| `max_core_tiles` | pages of a group-1 core | `ceil(total_out_pages / num_cores)` | `split_work_to_cores` |
| `total_out_pages` | `output.buffer_num_pages()` | unbounded — appears only inside `max_core_tiles`, which `block_tiles` is clamped against | — |
| `num_cores` | cores used | `min(G, total_out_pages)`, `G = 110` here | grid query |
| `K` | inputs | `1 ≤ K ≤ 160` | RT-arg cap (host `RuntimeError`) |
| `partial_rows_max` | transfer-shape threshold (does not size any CB) | 16 (0 for bfloat8_b) | host plan constant |
| L1 budget | op's L1 per core | 1 347 776 B | BB-05 |

## Footprint

`L1_per_core = depth_eff · block_tiles · tile_bytes · (1 + S)` ≤ 1 347 776 B by construction.

Scaling: linear in `block_tiles` (knob) and `depth_eff` (knob); `(1 + S)` scales with the number of
shifted inputs meeting one output tile row (shape-derived, ≤ 33); `tile_bytes` with dtype. No term
scales with an unbounded op dimension.

| Case (G = 110) | Regime | `max_core_tiles` | `S` | `block_tiles` | `depth_eff` | L1 per core |
|----------------|--------|------------------|-----|---------------|-------------|-------------|
| Qwen 17 `16×(1,6,4096,256)` d0 bf16 (p2) | page_run | 894 | 0 | 329 (cap) | 2 | 1 347 584 B |
| Qwen 18 `(108,4096,128)+(84,4096,128)` d0 fp32 (p1, refinement) | page_run | 894 | 0 | 164 (cap) | 2 | 1 343 488 B |
| FLUX 29 `(1,8192,3072)+(1,256,3072)` d1 | page_run | 231 | 0 | 231 | 1 | 473 088 B |
| Qwen 81 `16×(1,1,128,256)` d0 | page_run | 5 | 0 | 5 | 1 | 10 240 B |
| Qwen 79 `16×(1,1,1,62080)` d2 | row_gather | 18 | 15 | 18 | 1 | 589 824 B |
| Qwen 143 `8×(1,1,1,62080)` d2 | row_gather | 18 | 7 | 18 | 1 | 294 912 B |
| Qwen 89 `2×(1,1,1,62080)` d2 | row_gather | 18 | 1 | 18 | 1 | 73 728 B |
| FLUX 23 `2×(1,1,1,128)` d-1 | page_run | 1 | 0 | 1 | 1 | 2 048 B |

## Data-movement budget

The footprint and the traffic are one decision: nothing is resident across blocks because nothing
is re-read — every output page has its own sources.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| each input `x_k` | 1 (each logical row read once) | the flat `out_page` split is a bijection between output pages and input pages (page_run) or between output rows and input rows (row_gather); no page or row has two readers. Tiles with `m > partial_rows_max` carry their padding rows along; tiles with `m ≤ 16` move only logical rows | none |
| output | 1 (each logical row written once) | each output page belongs to exactly one core and one block; written once from `cb_concat_tiles` | none |
| staged rows (row_gather) | 0 | core-local L1→L1 self-copies; they never leave the core | none (self-aimed NoC reads stay on the core) |

| Case | DRAM read | DRAM written | Core-local copies | Note |
|------|-----------|--------------|-------------------|------|
| Qwen 18 (p1, fp32) | 402 653 184 B | 402 653 184 B | 0 | = LOOSE_CASES minimum 805 306 368 B |
| Qwen 17 (p2) | 201 326 592 B | 201 326 592 B | 0 | = stated minimum 402 653 184 B |
| Qwen 79 | 1 986 560 B (62 080 × 32 B) | 1 986 560 B (1 940 × 1 024 B) | 1 862 400 B | whole-tile gather would read 63 569 920 B |
| Qwen 160 (H = 1, w) | 124 160 B | 124 160 B | 0 | whole tiles would move 3 973 120 B each way |
| Qwen 109 (H = 31, w) | 98 304 B | 98 304 B | 0 | `m = 31 > 16`: whole tiles, 1 padding row per tile moved |

Totals per tier for the chosen split: DRAM = Σ logical input rows + Σ logical output rows (plus the
padding rows of tiles with `16 < m < 32`); inter-core NoC = 0; core-local = the shifted rows of
row_gather once.

> Cheapest-traffic split considered: flat `out_page` ranges over the full grid — delta 0 B at
> every tier (it is the floor: one DRAM crossing per logical row, no inter-core traffic).
> Implemented: flat `out_page` ranges (`page_run` / `row_gather`). The implemented split is the
> cheapest; no deferral is needed for traffic.
