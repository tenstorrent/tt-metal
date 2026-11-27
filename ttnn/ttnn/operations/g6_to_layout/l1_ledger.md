# L1 Ledger: g6_to_layout

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md` → Blocking Model: `tile_row` (extent `block_rows_tiles`) and `tile_col` (extent `block_cols_tiles`); `stage_row` / `stage_col_bytes` are their images in the row stage and are accounted with them.

R1 `staged_reorder`, R2 `face_direct`, R3 `packer_untilize`, R4 `sharded_in_place`, R5 `row_major_to_tile` and R6 `packer_tilize` are different programs: their CBs never coexist on a core. R3/R4 reuse indices 0 and 1 with different page formats; R5/R6 use indices 4 and 5.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_tile_window` (R1, idx 0) | `tile_window_depth · read_batch_tiles` pages × 2048 B (Phase 0: 16 pages, 32768 B) | `tile_window_depth · read_batch_tiles`: one batch landing from DRAM while the previous one is reordered | `{tile_row: streams → read_batch_tiles window, tile_col: streams → read_batch_tiles window}` | Float16_b, raw bytes — a dataflow-only payload that never reaches DEST | reader | reader | whole R1 reader kernel | none: live at the same time as `cb_row_block` (the reorder reads the window while writing the row block), so no alias; reader-private (no handshake). Slot = a whole 2048-B tile so every face lands at its source offset (DRAM reads need `dst % 64 == src % 64`) and adjacent faces merge into one read; for `H ≤ 16` only face-row prefixes of faces 0/1 land, so most of the 32 KB is unused — a fixed window, never scaled by an op dimension, kept for the uniform read path |
| `cb_row_block` (R1, idx 1) | `row_depth` pages × `block_rows_tiles · tile_row_stage_rows · block_cols_tiles · 64` B | `row_depth` pages: the reader reorders into one while the writer drains the other | `{tile_row: spans → block_rows_tiles (× tile_row_stage_rows rows), tile_col: spans → block_cols_tiles (× 64 B)}` | Float16_b, raw bytes | reader | writer | whole R1 kernels; one page per block | none: its two pages are concurrently live by the explicit double-buffering decision, and it is concurrent with `cb_tile_window`. Capacity exceeds the live bytes only on ragged blocks (partial head/tail span, short last block, last tile-row of an image with `live_rows < 32`): they use a prefix of the nominal page so every push/pop stays 1 page (ring-wrap invariant) |
| `cb_left_faces` (R2, idx 2) | `row_depth` pages × `block_rows_tiles · 1024` B | `row_depth` pages: reader loads one, writer drains the other | `{tile_row: spans → block_rows_tiles, tile_col: spans → 1 (pinned by the R2 predicate tensor_cols_tiles == 1)}` | Float16_b, raw bytes | reader | writer | whole R2 kernels; one page per block | none: the only CB of R2 (no window — the writer reads face rows in place). Faces 1/3 are not allocated (all padding when `W ≤ 16`); the face-2 half of a slot is unused only for an image's last tile-row with `live_rows ≤ 16` — the transfer quantum is the fixed 1024-B slot |
| `cb_tile_window` (R3, idx 0) | `tile_block_depth · block_rows_tiles · block_cols_tiles` pages × `in_tile_bytes` (1088 B bfp8) | `tile_block_depth` blocks: the reader lands block `k+1` while compute untilizes block `k` | `{tile_row: spans → block_rows_tiles, tile_col: spans → block_cols_tiles}` | input dtype (Bfp8_b), unpacked | reader | compute | whole R3 kernels; the nominal block's pages per block | none: concurrently live with `cb_row_block` (compute reads one while packing the other) and with `cb_nan_marks` (the reader scans pushed tiles). Ragged blocks use a prefix of the nominal page group so every block moves the same page count (ring-wrap) |
| `cb_tile_window` (R4, idx 0) | the shard: `shard_rows_tiles · shard_cols_tiles` pages × `in_tile_bytes`, **globally allocated on the input buffer** (`cb_descriptor_from_sharded_tensor`) | the shard's live tile pages, published once | `{tile_row: spans → shard_rows_tiles (live rows only), tile_col: spans → shard_cols_tiles}` | input dtype | reader (publishes) | compute | whole R4 kernels | none, and it costs the op nothing: it is the caller's resident shard (zero-copy), outside the 1347776-B op budget |
| `cb_row_block` (R3/R4, idx 1) | R3: `row_depth · block_rows_tiles · block_cols_tiles` pages × 2048 B; R4: `min(row_depth, blocks_per_shard)` blocks | R3: 2 blocks (compute packs one, writer drains the other); R4 production: 1 block (the shard is one block, so a second page buys no overlap) | `{tile_row: spans → block_rows_tiles (× 32 rows), tile_col: spans → block_cols_tiles (× 64 B)}` | Float16_b | compute | writer | whole kernels; `block_rows_tiles · block_cols_tiles` pages per block (R4: live tile-rows only) | none: concurrent with `cb_tile_window` (see above); the writer patches NaNs in place in it (transform in place, pattern 2) rather than in a scratch copy |
| `cb_row_stage` (R5, idx 4) | `row_window_depth` pages × `tile_row_stage_rows · block_cols_tiles · 64` B (one tile-row slot) | `row_window_depth` slots: one tile-row's row loads land while the previous one is copied into tiles | `{tile_row: streams → 1 tile-row per slot, tile_col: spans → block_cols_tiles (× 64 B)}` | Float16_b, raw bytes — never unpacked | reader | writer (`split_reorder`: pops a slot after its bottom-face copies); reader-private when the split is off | whole R5 kernels; one page per tile-row | none: concurrently live with `cb_tile_block` (copies read one while writing the other). Sized by tile-row, not by block: a tile-row is the copy unit, so a block of many tile-rows never needs them all staged at once |
| `cb_tile_block` (R5/R6, idx 5) | `row_depth · block_rows_tiles · block_cols_tiles` pages × `out_tile_bytes` (2048 bf16, 1088 bfp8) | `row_depth` blocks: one being filled (R5 copies / R6 packer) while the writer drains the other | `{tile_row: spans → block_rows_tiles, tile_col: spans → block_cols_tiles}` | output dtype | R5: reader (top faces; the writer adds the bottom faces of the block at the CB front before its wait) / R6: compute | writer | whole kernels; the nominal block's pages per block | none: the writer writes tiles out of it directly (whole-tile writes), so no further stage exists to fold it into. Ragged blocks use a prefix of the nominal page group (ring-wrap) |
| `cb_row_stage` (R6, idx 4) | `tile_block_depth · block_rows_tiles · block_cols_tiles` pages × 2048 B (bf16 tile page = 32 rows × 32 values of one tile column's span) | `tile_block_depth` blocks: the reader lands and conditions block `k+1` while compute tilizes block `k` | `{tile_row: spans → block_rows_tiles (× 32 rows), tile_col: spans → block_cols_tiles}` | Float16_b (the tilize input) | reader | compute | whole R6 kernels; the nominal block's pages per block | none: the conditioning pass transforms it in place (pattern 2), so no scratch copy exists; concurrently live with `cb_tile_block` |
| `cb_nan_marks` (R3/R4, idx 3) | `nan_marks_depth` pages (R3: `tile_block_depth + row_depth` = 4; R4: `min(4, blocks_per_shard)`) × `round_up(136 · block_tiles, 16)` B | one page per block between the reader's scan and the writer's patch; the reader runs up to `tile_block_depth + row_depth` blocks ahead | `{tile_row: spans → block_rows_tiles, tile_col: spans → block_cols_tiles}` (136 B per tile) | raw bytes, never unpacked | reader | writer | whole kernels; 1 page per block | none: it is the only reader→writer channel and outlives the tiles it describes (compute may pop them before the writer patches). Could fold into `cb_row_block` only if compute wrote the marks, and compute never sees NaN bits (the unpacker already turned them to ±inf) |

## Symbol table

| Symbol | Value / definition | Bound | Predicate that establishes it |
|--------|--------------------|-------|-------------------------------|
| `block_rows_tiles` | see `op_design.md` Axes | `1 ≤ · ≤ floor(max_block_tiles / block_cols_tiles)` | host clamp in `_choose_blocking()` |
| `block_cols_tiles` | `min(tensor_cols_tiles, core_tiles_max, max_block_tiles)` | `1 ≤ · ≤ max_block_tiles` | host clamp |
| `read_batch_tiles` | 8 | `≤ 127` | host constant (single source); ≤ 2 DRAM reads per tile per batch id, so `2·read_batch_tiles ≤ 255` (kernel `static_assert`) |
| `tile_window_depth` | 2 | `≤ 13` | transaction ids 1…15 (`noc_parameters.h:15`): ids 1…`tile_window_depth` tag DRAM batches, two more alternate per tile over the reorder copies so no id holds more than one tile's 64 copies (per-id outstanding counter ≤ 255, `NOC_MAX_TRANSACTION_ID_COUNT`); kernel `static_assert` |
| `row_depth` | 2 | constant | host constant (R4: `min(row_depth, blocks_per_shard)`) |
| `row_window_depth` | 2 | `2 ≤ · ≤ 11` | trids `1…row_window_depth` tag R5 tile-row loads, `+1, +2` the reader's copies, `+3, +4` the writer's (`split_reorder`); a pushed-ahead slot needs a second page; host `assert` + kernel `static_assert` |
| `split_reorder` | `True` | bool | host constant; R5 only |
| `out_tile_bytes` | `ttnn.tile_size(output dtype)`: 2048 bf16, 1088 bfp8 | — | output dtype (TILE output) |
| `tile_block_depth` | 2 | constant | host constant (R3) |
| `nan_marks_depth` | `tile_block_depth + row_depth` = 4 | constant | derived from the two depth knobs (R4: `min(·, blocks_per_shard)`) |
| `in_tile_bytes` | `ttnn.tile_size(input dtype)`: 1088 B for bfloat8_b | — | dtype |
| `blocks_per_shard` | `ceil(shard_rows_tiles / block_rows_tiles)`; 1 for every production shard | `≥ 1` | R4 host clamp `block_rows_tiles = min(shard_rows_tiles, max_block_tiles / shard_cols_tiles)` |
| `tile_row_stage_rows` | `min(32, H)` | `≤ 32` | definition (for `H < 32` every tile-row has exactly `H` live rows) |
| `core_tiles_max` | `ceil(tensor_tiles / num_cores)` (`n1` of `split_work_to_cores`) | `≤ tensor_tiles` | work split |
| `out_l1_share` | L1 output: `ceil(out_pages / num_l1_banks) · out_aligned_page_bytes · banks_per_core`; DRAM output: 0 | `< 1347776 − minimal footprint` | host refusal (BB-05) otherwise; production max ≈ one 5120-B page (case 2038) |
| `l1_budget` | 1347776 B | model contract | spec, Model environment |
| `max_block_tiles` | R1: `floor((l1_budget − out_l1_share − 2048·tile_window_depth·read_batch_tiles) / (row_depth · tile_row_stage_rows · 64))`; R2: `floor((l1_budget − out_l1_share) / (row_depth · 1024))`; R3: `floor((l1_budget − out_l1_share − 16·nan_marks_depth) / (tile_block_depth·in_tile_bytes + row_depth·2048 + nan_marks_depth·136))`; R4: R3 without the `tile_block_depth·in_tile_bytes` term; R5: `floor((l1_budget − out_l1_share) / (row_window_depth · tile_row_stage_rows · 64 + row_depth · 2048))`; R6: `floor((l1_budget − out_l1_share) / (tile_block_depth · 2048 + row_depth · out_tile_bytes))` | `≥ 1` | host refusal otherwise (unreachable for golden INPUTS) |

## Total per-core footprint

| Regime | Expression | Scales with | Phase 0 max over golden INPUTS (G = 110) |
|--------|-----------|-------------|-------------------------------------------|
| R1 | `2048 · tile_window_depth · read_batch_tiles + row_depth · block_rows_tiles · tile_row_stage_rows · block_cols_tiles · 64 + out_l1_share ≤ 1347776` | window term: depth knobs only; row term: both block extents and `row_depth`; output term: L1 output only | 688128 B (case 1978) |
| R2 | `row_depth · block_rows_tiles · 1024 + out_l1_share ≤ 1347776` | `block_rows_tiles`, `row_depth` | 153600 B (case 308) |
| R3 | `tile_block_depth · block_tiles · in_tile_bytes + row_depth · block_tiles · 2048 + nan_marks_depth · round_up(136 · block_tiles, 16) + out_l1_share ≤ 1347776` | both block extents, the depth knobs | 43840 B (case 29: 5-tile blocks, out_l1_share 9728); case 43: 25920 B |
| R5 | `row_window_depth · tile_row_stage_rows · block_cols_tiles · 64 + row_depth · block_tiles · 2048 + out_l1_share ≤ 1347776` | window: `block_cols_tiles`, `row_window_depth`; tile block: both block extents, `row_depth` | 219136 B (case 38: 1×8-tile blocks, 65536 B of CBs + 153600 B L1 output share); case 49: 110592 B; case 301 (DRAM): 65536 B |
| R6 | `tile_block_depth · block_tiles · 2048 + row_depth · block_tiles · out_tile_bytes + out_l1_share ≤ 1347776` | both block extents, the depth knobs | 6272 B (cases 16 and Q3: one tile per core) |
| R4 | `min(row_depth, blocks_per_shard) · block_tiles · 2048 + min(nan_marks_depth, blocks_per_shard) · round_up(136 · block_tiles, 16) + out_l1_share ≤ 1347776` (the shard itself is the caller's) | shard live tiles | 90816 B (case 37: 24-tile shard, out_l1_share 38400); case 48: 45408 B; case 20: 2704 B |

Audit notes: no Float32 page (R3/R4 DEST is 16-bit, `fp32_dest_acc_en = False`; audit 2 n/a); no disjoint-lifetime pair (audit 3); every capacity symbol is bounded above (audit 4).

## Block-size default: departure and its measurement

The interleaved default is the coarsest block that fits (`block_rows_tiles` at `floor(max_block_tiles / block_cols_tiles)`). Phase 0 departs from it on full-row blocks: `block_rows_tiles = ceil(core_full_rows_max / pipeline_blocks_per_core)` with `pipeline_blocks_per_core = 16`. Column-chunked blocks (`block_cols_tiles < tensor_cols_tiles`) are one tile-row tall and unaffected. Device kernel time, 2 runs each (verifier sweep, `probes/test_probe_verifier_pbpc.py`, BH, 110 cores):

| Case | `pipeline_blocks_per_core = 1` (L1 cap) | 4 | **16 (shipped)** | 64 |
|------|------------------------------------------|---|------------------|----|
| 304 `[1,1024,2048,128]` (R1) | 3083 / 3081 µs (80 rows, 1343488 B) | 3093 / 3084 µs | **2949 / 2933 µs** (38 rows, 655360 B) | 2925 / 2926 µs |
| 1978 `[16,4096,2560]` (R1) | 1906 / 1906 µs (4 rows, 1343488 B) | 1902 / 1904 µs | **1789 / 1779 µs** (2 rows, 688128 B) | 1898 / 1899 µs |
| 308 `[1,2048,2048,3]` (R2) | 3217 / 3215 µs (658 rows, 1347584 B) | 3522 / 3472 µs | **3407 / 3275 µs** (75 rows, 153600 B) | 3408 / 3313 µs |

R5/R6 inherit `pipeline_blocks_per_core = 16` from the shared `_choose_blocking` without a sweep of their own: every production RM-input cell holds ≤ 10 tile-rows per core, so its blocks are one tile-row tall either way, and a block's fixed cost there is one reserve/push plus one wait/flush/pop.

16 is the measured best (or tied, 304 at 64) on R1, 5–6% ahead of the L1-cap block, at half its L1. On R2 the spread between repeats of one setting (4%) is as large as the spread between settings: 308 is bound by DRAM write transactions (writes alone 2167 µs, reads alone 517 µs, verifier ablation), not by the block.

## Data-movement budget (chosen split: S1, flattened row-major tile range)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 1, live face-row prefixes only | each tile is read by exactly one core into its window and reordered from L1; nothing is re-read, padding faces/rows are skipped | none. On-core only (R1): 2 self-aimed 32-B L1→L1 copies per live tile row (64 per full tile) |
| `output` | 1, live bytes of each page | the row stage holds a block's whole row segment, so each (row, core) pair is one write; R2 writes each row straight from its face row | none |
| `input_tensor`, bf8b interleaved (R3) | 1, whole bfp8 tiles (1088 B) | the unpacker consumes whole tiles; each tile is read by exactly one core, once | none |
| `input_tensor`, bf8b sharded (R4) | 0 | consumed in place from the resident shard (zero-copy CB); the reader issues no NoC read | none |
| `input_tensor`, RM bf16 (R5/R6) | 1, live bytes of each row segment | each output tile is owned by one core, which reads the live bytes of its rows once; padding columns of the last tile are never read | none. On-core only (R5): 2 self-aimed 32-B copies per live tile row (64 per full tile), half on each DM RISC |
| `output`, TILE (R5/R6) | 1, whole tiles | one write per tile (2048 B bf16, 1088 B bfp8), including the tile's padding — a tile page is the transfer unit (catalog `tile_reorder`) | none |

Per flagged case (bytes across the input/output memory boundary):

| Case | Input read | Output written (logical) | Write transactions | On-core copies |
|------|------------|--------------------------|--------------------|----------------|
| 308 `[1,2048,2048,3]` (R2) | 131072 tiles × 1024 B = 128 MiB (whole tiles would be 256 MiB; logical 24 MiB) | 4194304 × 6 B = 24 MiB, one per 64-B DRAM page slot | 4194304 | 0 |
| 304 `[1,1024,2048,128]` (R1) | 262144 × 2048 B = 512 MiB | 2097152 × 256 B = 512 MiB | 2097152 + ≤ 64/core | 16777216 × 32 B |
| 1978 `[16,4096,2560]` (R1) | 163840 × 2048 B = 320 MiB | 65536 × 5120 B = 320 MiB | 65536 + ≤ 64/core | 10485760 × 32 B |
| 286 `[1,1024,1024,3]` (R2) | 32768 × 1024 B = 32 MiB | 1048576 × 6 B = 6 MiB | 1048576 | 0 |
| 29 `[1,32,64,256]` bf8b (R3, L1 → L1) | 512 × 1088 B = 544 KiB | 2048 × 512 B = 1 MiB | 2048 rows × 2–3 segments (5-tile column-chunk blocks on 8-tile rows) | 0 |
| 43 `[1,128,256,32]` bf8b (R3, L1 → L1) | 1024 × 1088 B = 1.06 MiB | 32768 × 64 B = 2 MiB | 32768 | 0 |
| 37 `[1,64,128,256]` bf8b HEIGHT (R4) | 0 (resident shard) | 8192 × 512 B = 4 MiB | 8192 | 0 |
| 48 `[1,64,128,128]` bf8b HEIGHT (R4) | 0 (resident shard) | 8192 × 256 B = 2 MiB | 8192 | 0 |
| 301 `[1,128,256,256]` RM → TILE (R5, DRAM) | 32768 rows × 512 B = 16 MiB | 8192 tiles × 2048 B = 16 MiB | 8192 | 524288 × 32 B (half per DM RISC) |
| 38 `[1,128,256,256]` RM → TILE (R5, L1 → L1) | 16 MiB | 16 MiB | 8192 | 524288 × 32 B |
| 1989 `[16,1,1536]` RM → TILE (R5) | 16 rows × 3072 B = 48 KiB | 768 tiles × 2048 B = 1.5 MiB (1 live row per tile) | 768 | 1536 × 32 B (reader only: no bottom faces at H = 1) |
| 16 `[1,1,1,256]` RM → TILE bf8b (R6) | 1 row × 512 B, as 8 segments of 64 B | 8 tiles × 1088 B | 8 | 0 (packer) |
| 20 `[1,1,1,256]` bf8b BLOCK (R4) | 0 (resident shard) | 1 page × 512 B, as 8 segments of 64 B | 8 | 0 |

Totals per tier, any shape: DRAM (or L1 bank) = live input face rows + logical output bytes, each once; cross-core NoC = 0; core-local L1 = one extra pass over the input's live bytes (R1 only).

> Cheapest-traffic split considered: every candidate split (S1 tile range, S2 height, S3 width, S4 2-D) moves identical bytes per tier — 1 input crossing, 1 output crossing, 0 cross-core — so the delta is 0 B per tier; they differ only in write-transaction count (S2 ≤ S1 + 64/core ≤ S4 ≤ S3). Implemented: S1, which is also the cheapest in bytes and the only split that fills the grid for every golden shape. Nothing is deferred on traffic grounds; the one byte-level alternative, `same_bank_slot_writes` (more output bytes, fewer write transactions), is a `deferred` regime row in `op_design.md` with its delta stated.
