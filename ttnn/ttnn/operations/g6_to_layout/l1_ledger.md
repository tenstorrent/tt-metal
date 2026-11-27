# L1 Ledger: g6_to_layout

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md` → Blocking Model: `tile_row` (extent `block_rows_tiles`) and `tile_col` (extent `block_cols_tiles`); `stage_row` / `stage_col_bytes` are their images in the row stage and are accounted with them.

R1 `staged_reorder` and R2 `face_direct` are different programs: their CBs never coexist on a core.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_tile_window` (R1, idx 0) | `tile_window_depth · read_batch_tiles` pages × 2048 B (Phase 0: 16 pages, 32768 B) | `tile_window_depth · read_batch_tiles`: one batch landing from DRAM while the previous one is reordered | `{tile_row: streams → read_batch_tiles window, tile_col: streams → read_batch_tiles window}` | Float16_b, raw bytes — a dataflow-only payload that never reaches DEST | reader | reader | whole R1 reader kernel | none: live at the same time as `cb_row_block` (the reorder reads the window while writing the row block), so no alias; reader-private (no handshake). Slot = a whole 2048-B tile so every face lands at its source offset (DRAM reads need `dst % 64 == src % 64`) and adjacent faces merge into one read; for `H ≤ 16` only face-row prefixes of faces 0/1 land, so most of the 32 KB is unused — a fixed window, never scaled by an op dimension, kept for the uniform read path |
| `cb_row_block` (R1, idx 1) | `row_depth` pages × `block_rows_tiles · tile_row_stage_rows · block_cols_tiles · 64` B | `row_depth` pages: the reader reorders into one while the writer drains the other | `{tile_row: spans → block_rows_tiles (× tile_row_stage_rows rows), tile_col: spans → block_cols_tiles (× 64 B)}` | Float16_b, raw bytes | reader | writer | whole R1 kernels; one page per block | none: its two pages are concurrently live by the explicit double-buffering decision, and it is concurrent with `cb_tile_window`. Capacity exceeds the live bytes only on ragged blocks (partial head/tail span, short last block, last tile-row of an image with `live_rows < 32`): they use a prefix of the nominal page so every push/pop stays 1 page (ring-wrap invariant) |
| `cb_left_faces` (R2, idx 2) | `row_depth` pages × `block_rows_tiles · 1024` B | `row_depth` pages: reader loads one, writer drains the other | `{tile_row: spans → block_rows_tiles, tile_col: spans → 1 (pinned by the R2 predicate tensor_cols_tiles == 1)}` | Float16_b, raw bytes | reader | writer | whole R2 kernels; one page per block | none: the only CB of R2 (no window — the writer reads face rows in place). Faces 1/3 are not allocated (all padding when `W ≤ 16`); the face-2 half of a slot is unused only for an image's last tile-row with `live_rows ≤ 16` — the transfer quantum is the fixed 1024-B slot |

## Symbol table

| Symbol | Value / definition | Bound | Predicate that establishes it |
|--------|--------------------|-------|-------------------------------|
| `block_rows_tiles` | see `op_design.md` Axes | `1 ≤ · ≤ floor(max_block_tiles / block_cols_tiles)` | host clamp in `_choose_blocking()` |
| `block_cols_tiles` | `min(tensor_cols_tiles, core_tiles_max, max_block_tiles)` | `1 ≤ · ≤ max_block_tiles` | host clamp |
| `read_batch_tiles` | 8 | `≤ 127` | host constant (single source); ≤ 2 DRAM reads per tile per batch id, so `2·read_batch_tiles ≤ 255` (kernel `static_assert`) |
| `tile_window_depth` | 2 | `≤ 13` | transaction ids 1…15 (`noc_parameters.h:15`): ids 1…`tile_window_depth` tag DRAM batches, two more alternate per tile over the reorder copies so no id holds more than one tile's 64 copies (per-id outstanding counter ≤ 255, `NOC_MAX_TRANSACTION_ID_COUNT`); kernel `static_assert` |
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

## Block-size default: departure and its measurement

The interleaved default is the coarsest block that fits (`block_rows_tiles` at `floor(max_block_tiles / block_cols_tiles)`). Phase 0 departs from it on full-row blocks: `block_rows_tiles = ceil(core_full_rows_max / pipeline_blocks_per_core)` with `pipeline_blocks_per_core = 16`. Column-chunked blocks (`block_cols_tiles < tensor_cols_tiles`) are one tile-row tall and unaffected. Device kernel time, 2 runs each (verifier sweep, `probes/test_probe_verifier_pbpc.py`, BH, 110 cores):

| Case | `pipeline_blocks_per_core = 1` (L1 cap) | 4 | **16 (shipped)** | 64 |
|------|------------------------------------------|---|------------------|----|
| 304 `[1,1024,2048,128]` (R1) | 3083 / 3081 µs (80 rows, 1343488 B) | 3093 / 3084 µs | **2949 / 2933 µs** (38 rows, 655360 B) | 2925 / 2926 µs |
| 1978 `[16,4096,2560]` (R1) | 1906 / 1906 µs (4 rows, 1343488 B) | 1902 / 1904 µs | **1789 / 1779 µs** (2 rows, 688128 B) | 1898 / 1899 µs |
| 308 `[1,2048,2048,3]` (R2) | 3217 / 3215 µs (658 rows, 1347584 B) | 3522 / 3472 µs | **3407 / 3275 µs** (75 rows, 153600 B) | 3408 / 3313 µs |

16 is the measured best (or tied, 304 at 64) on R1, 5–6% ahead of the L1-cap block, at half its L1. On R2 the spread between repeats of one setting (4%) is as large as the spread between settings: 308 is bound by DRAM write transactions (writes alone 2167 µs, reads alone 517 µs, verifier ablation), not by the block.

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
