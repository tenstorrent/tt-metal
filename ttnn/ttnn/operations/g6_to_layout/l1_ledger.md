# L1 Ledger: g6_to_layout

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md` → Blocking Model: `tile_row` (extent `block_rows_tiles`) and `tile_col` (extent `block_cols_tiles`); `stage_row` / `stage_col_bytes` are their images in the row stage and are accounted with them.

R1 `staged_reorder` and R2 `face_direct` are different programs: their CBs never coexist on a core.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_tile_window` (R1, idx 0) | `tile_window_depth · read_batch_tiles` pages × 2048 B (Phase 0: 16 pages, 32768 B) | `tile_window_depth · read_batch_tiles`: one batch landing from DRAM while the previous one is reordered | `{tile_row: streams → read_batch_tiles window, tile_col: streams → read_batch_tiles window}` | Float16_b, raw bytes — a dataflow-only payload that never reaches DEST | reader | reader | whole R1 reader kernel | none: live at the same time as `cb_row_block` (the reorder reads the window while writing the row block), so no alias; reader-private (no handshake) |
| `cb_row_block` (R1, idx 1) | `row_depth` pages × `block_rows_tiles · tile_row_stage_rows · block_cols_tiles · 64` B | `row_depth` pages: the reader reorders into one while the writer drains the other | `{tile_row: spans → block_rows_tiles (× tile_row_stage_rows rows), tile_col: spans → block_cols_tiles (× 64 B)}` | Float16_b, raw bytes | reader | writer | whole R1 kernels; one page per block | none: its two pages are concurrently live by the explicit double-buffering decision, and it is concurrent with `cb_tile_window`. Capacity exceeds the live bytes only on ragged blocks (partial head/tail span, short last block, last tile-row of an image with `live_rows < 32`): they use a prefix of the nominal page so every push/pop stays 1 page (ring-wrap invariant) |
| `cb_left_faces` (R2, idx 2) | `row_depth` pages × `block_rows_tiles · 1024` B | `row_depth` pages: reader loads one, writer drains the other | `{tile_row: spans → block_rows_tiles, tile_col: spans → 1 (pinned by the R2 predicate tensor_cols_tiles == 1)}` | Float16_b, raw bytes | reader | writer | whole R2 kernels; one page per block | none: the only CB of R2 (no window — the writer reads face rows in place). Faces 1/3 are not allocated (all padding when `W ≤ 16`); the face-2 half of a slot is unused only for an image's last tile-row with `live_rows ≤ 16` — the transfer quantum is the fixed 1024-B slot |

## Symbol table

| Symbol | Value / definition | Bound | Predicate that establishes it |
|--------|--------------------|-------|-------------------------------|
| `block_rows_tiles` | see `op_design.md` Axes | `1 ≤ · ≤ floor(max_block_tiles / block_cols_tiles)` | host clamp in `_choose_blocking()` |
| `block_cols_tiles` | `min(tensor_cols_tiles, core_tiles_max, max_block_tiles)` | `1 ≤ · ≤ max_block_tiles` | host clamp |
| `read_batch_tiles` | 8 | constant | host constant (single source) |
| `tile_window_depth` | 2 | `≤ 14` | transaction ids 1…15 (`noc_parameters.h:15`), one reserved for the reorder copies |
| `row_depth` | 2 | constant | host constant |
| `tile_row_stage_rows` | `min(32, H)` | `≤ 32` | definition (for `H < 32` every tile-row has exactly `H` live rows) |
| `core_tiles_max` | `ceil(tensor_tiles / num_cores)` (`n1` of `split_work_to_cores`) | `≤ tensor_tiles` | work split |
| `out_l1_share` | L1 output: `ceil(out_pages / num_l1_banks) · out_aligned_page_bytes · banks_per_core`; DRAM output: 0 | `< 1347776 − minimal footprint` | host refusal (BB-05) otherwise; production max ≈ one 5120-B page (case 2038) |
| `l1_budget` | 1347776 B | model contract | spec, Model environment |
| `max_block_tiles` | R1: `floor((l1_budget − out_l1_share − 2048·tile_window_depth·read_batch_tiles) / (row_depth · tile_row_stage_rows · 64))`; R2: `floor((l1_budget − out_l1_share) / (row_depth · 1024))` | `≥ 1` | host refusal otherwise (unreachable for golden INPUTS) |

## Total per-core footprint

| Regime | Expression | Scales with | Phase 0 max over golden INPUTS (G = 110) |
|--------|-----------|-------------|-------------------------------------------|
| R1 | `2048 · tile_window_depth · read_batch_tiles + row_depth · block_rows_tiles · tile_row_stage_rows · block_cols_tiles · 64 + out_l1_share ≤ 1347776` | window term: depth knobs only; row term: both block extents and `row_depth`; output term: L1 output only | 688128 B (case 1978) |
| R2 | `row_depth · block_rows_tiles · 1024 + out_l1_share ≤ 1347776` | `block_rows_tiles`, `row_depth` | 153600 B (case 308) |

Audit notes: no Float32 page (no DEST, audit 2 n/a); no disjoint-lifetime pair (audit 3); every capacity symbol is bounded above (audit 4).

## Data-movement budget (chosen split: S1, flattened row-major tile range)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 1, live face-row prefixes only | each tile is read by exactly one core into its window and reordered from L1; nothing is re-read, padding faces/rows are skipped | none. On-core only (R1): 2 self-aimed 32-B L1→L1 copies per live tile row (64 per full tile) |
| `output` | 1, live bytes of each page | the row stage holds a block's whole row segment, so each (row, core) pair is one write; R2 writes each row straight from its face row | none |

Per flagged case (bytes across the input/output memory boundary):

| Case | Input read | Output written (logical) | Write transactions | On-core copies |
|------|------------|--------------------------|--------------------|----------------|
| 308 `[1,2048,2048,3]` (R2) | 131072 tiles × 1024 B = 128 MiB (whole tiles would be 256 MiB; logical 24 MiB) | 4194304 × 6 B = 24 MiB, one per 64-B DRAM page slot | 4194304 | 0 |
| 304 `[1,1024,2048,128]` (R1) | 262144 × 2048 B = 512 MiB | 2097152 × 256 B = 512 MiB | 2097152 + ≤ 64/core | 16777216 × 32 B |
| 1978 `[16,4096,2560]` (R1) | 163840 × 2048 B = 320 MiB | 65536 × 5120 B = 320 MiB | 65536 + ≤ 64/core | 10485760 × 32 B |
| 286 `[1,1024,1024,3]` (R2) | 32768 × 1024 B = 32 MiB | 1048576 × 6 B = 6 MiB | 1048576 | 0 |

Totals per tier, any shape: DRAM (or L1 bank) = live input face rows + logical output bytes, each once; cross-core NoC = 0; core-local L1 = one extra pass over the input's live bytes (R1 only).

> Cheapest-traffic split considered: every candidate split (S1 tile range, S2 height, S3 width, S4 2-D) moves identical bytes per tier — 1 input crossing, 1 output crossing, 0 cross-core — so the delta is 0 B per tier; they differ only in write-transaction count (S2 ≤ S1 + 64/core ≤ S4 ≤ S3). Implemented: S1, which is also the cheapest in bytes and the only split that fills the grid for every golden shape. Nothing is deferred on traffic grounds; the one byte-level alternative, `same_bank_slot_writes` (more output bytes, fewer write transactions), is a `deferred` regime row in `op_design.md` with its delta stated.
