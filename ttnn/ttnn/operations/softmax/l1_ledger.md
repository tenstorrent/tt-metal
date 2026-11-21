# L1 ledger: softmax (Phase 0, regime `column_resident`)

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Design: `op_design.md` (same directory).
Target: Quasar (ttsim), 32 clusters, budget `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES` = 3,963,328 B per cluster.
Block axes (from the Blocking Model): `cols` (folded column-unit axis `nc·Wt + wt`, extent `block_cols`) and `rows` (tile-row axis, extent `block_rows = Ht`).
Every capacity is per NEO × `C` (STRIDED rings slice per compute thread, `C = THREAD_PLAN[1] = 4`); `fp32_dest_acc_en = True`.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `in_depth · block_cols · Ht · C` | `block_cols · Ht · C` (the whole block's columns stay resident through three ops); `2 · block_cols · Ht · C` when `in_depth = 2` (next block landing) | `{cols: spans → block_cols, rows: spans → Ht}` | bf16 (input dtype; dataflow payload unpacked to fp32 DEST, never repacked) | reader | compute | `load_block` → end of `normalize_block`, per block | Nothing: it is the only reader-filled ring, live across all three compute ops; output cannot pack into it (one producer per ring: the reader). Capacity above live set only by `in_depth` (overlap of the next block's read, multi-block shapes only) |
| `cb_h_mask` | `1 · C`, allocated only when `h_valid_last < 32` | `1 · C` | `{cols: streams → 1 (one mask serves every column), rows: streams → 1 (applies to the last tile-row only)}` | bf16 (same format as the input it is added to: no unpack reconfig between them; `NEG_BIG = 0xFF7F` is exact in bf16 and fp32) | reader (explicit sync) | compute | whole kernel | Constant; cannot alias the stat rings (concurrent for the whole kernel). Could be generated in compute only if an SFPU row-select fill existed — none does, so the reader builds it from local stores (no DRAM) |
| `cb_col_max` | `block_cols · C` | `block_cols · C` (every column's `m` is read again by `normalize_block` after all of `column_expsum_block`) | `{cols: spans → block_cols, rows: streams → 1 tile per column (the row axis is collapsed)}` | Float32 (fp32 DEST; a 16-bit page would truncate `m` before `exp`) | compute | compute | `column_max_block` → end of `normalize_block` | Not with `cb_col_recip`: both live during `normalize_block`. Not in place over the max (the max is still needed after `1/s` exists). Not folded into DEST: three ops with pack/commit between them cannot share a DEST section (perf lamp: a single-section path for `Ht ≤ DEST_AUTO_LIMIT − 2` would drop it) |
| `cb_col_recip` | `block_cols · C` | `block_cols · C` | `{cols: spans → block_cols, rows: streams → 1 tile per column}` | Float32 | compute | compute | `column_expsum_block` → end of `normalize_block` | Not with `cb_col_max` (concurrent in `normalize_block`); not with `cb_input_tiles` (different producer and format). An `exp` intermediate ring is deliberately absent: `normalize_block` recomputes `exp` in DEST instead (Rule 3.1/3.4; perf lamp records the alternative) |
| `cb_output_tiles` | `OUT_DEPTH · out_chunk_tiles · C`, `out_chunk_tiles = min(block_cols · Ht, OUT_CHUNK_MAX_TILES)` | `out_chunk_tiles · C` (one DEST section's packed tiles) | `{cols: streams → window, rows: streams → window}` | bf16 (output dtype; the ABI of the output tensor) | compute | writer | `normalize_block` → `store_block` | Cannot be `cb_input_tiles` (reader owns that ring's production; the writer would be a second consumer). Capacity is 2× live set by `OUT_DEPTH` (writer drains one chunk while compute packs the next) |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `C` | 4 (`THREAD_PLAN = (2, 4, 2)`) | host constant; target page: C ∈ {1, 2, 4} |
| `Ht` | `1 ≤ Ht ≤ floor(DFB_RING_MAX_BYTES / (tile_in · C))` = 127 for bf16 at C = 4 | `resident_fits` (validate refuses otherwise); INPUTS max = 64 |
| `block_cols` | `1 ≤ block_cols ≤ min(max_rounds, BLOCK_COLS_CAP, floor(DFB_RING_MAX_BYTES / (4096 · C)) = 63, ring and budget bounds)` | `select_blocking()` |
| `in_depth` | `{1, 2}` | `select_blocking()` (1 when the share is one block) |
| `OUT_DEPTH` | 2 | host constant |
| `OUT_CHUNK_MAX_TILES` | 8 | host constant: the largest DEST section (fp32, full sync) |
| `tile_in` | 2,048 B (bf16 tile) | `input.buffer_page_size()` |
| `tile_f32` | 4,096 B | Float32 tile |
| `has_mask` | 0 / 1 | `h_valid_last = H − (Ht − 1)·32 < 32` |
| `DFB_RING_MAX_BYTES` | 1,048,560 B per ring | target page, cluster model (no run-time query exists) |
| `budget` | `ttnn.get_max_worker_l1_unreserved_size() − L1_RESERVE_BYTES` (`L1_RESERVE_BYTES = 0`) = 3,963,328 B on Quasar/ttsim | run-time query |

## Footprint per cluster

```
footprint = C · tile_in · (in_depth · block_cols · Ht            # cb_input_tiles  — scales with block_cols × Ht × in_depth
                         + OUT_DEPTH · min(block_cols · Ht, 8)    # cb_output_tiles — bounded window
                         + has_mask)                              # cb_h_mask       — constant
          + C · tile_f32 · 2 · block_cols                         # cb_col_max + cb_col_recip — scale with block_cols only
```

Bounds: every ring ≤ `DFB_RING_MAX_BYTES`; total ≤ `budget`. Upper bound of the expression under the ring caps: ≈ 1,040,384 (input) + 2 × 1,032,192 (stats) + 131,072 (output) + 8,192 (mask) = 3,244,032 B < 3,963,328 B, so on Quasar the per-ring cap binds before the budget. Worked values: `[1,4,16,8400]` 507,904 B; `[1,4,16,400]` 65,536 B; `[1,1,2048,256]` 688,128 B; `[4,8,32,256]` 114,688 B.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input (`NC · Ht · Wt` tiles) | 1 | the whole column is resident in `cb_input_tiles` across `column_max_block`, `column_expsum_block`, `normalize_block` | none |
| output (same tiles) | 1 (+ at most `C − 1` recomputed columns' tiles in the last round of the op, identical bytes) | written once from `cb_output_tiles`; the padded round re-stores NEO 0's column | none |
| H mask | 0 | built in L1 by the reader from local stores | none |
| stats `m`, `1/s` | 0 | produced and consumed in the owning NEO's L1 | none |

Total per call: DRAM (or the input's L1 bank) read `NC·Ht·Wt · tile_in` bytes once and written the same once; cross-core 0 bytes; core-local L1 re-reads: each input tile is unpacked 3× (one per compute op), each stat tile `Ht`-proportionally (per section of `column_expsum_block` and `normalize_block`).

> Cheapest-traffic split considered: column-units across clusters and NEOs with the column resident — 1× input, 1× output, 0 cross-core. Implemented: the same split; it is the cheapest. The H-split with a cross-cluster combine (`h_split_combine`) moves the same DRAM bytes plus 16 KB of stat tiles per column per slice across the NoC; it is deferred because the built regime already serves every shape it would serve, and it becomes the cheapest-traffic option only for columns too tall for one cluster's ring (`Ht > 127`), which Phase 0 refuses.
