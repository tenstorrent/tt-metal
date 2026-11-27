# L1 Ledger: g6_concat

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes (from `op_design.md`): `out_page` (page_copy), `out_tile_row`, `tile_col` and `in_tile_row` (row_gather; `in_tile_row` in both).

Inventory: two buffers, the minimum this data path admits.
- `cb_moved_pages` is the destination: page_copy reads straight into it (pattern 1, pack into the destination), and row_gather's on-parity rows land in it directly.
- `cb_parity_stage` exists only for rows whose DRAM source offset cannot be congruent with their destination offset (bf16 on BH). The design uses no other scratch, no intermediate and no solve.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_moved_pages` | `buffer_depth · block_pages` pages of `page_bytes` | `buffer_depth · block_pages`. At peak the reader fills one block slot while the writer drains the other (depth 2). | page_copy: `{out_page: spans → block_pages (streams over blocks, window buffer_depth), in_tile_row: spans → 32 (whole page)}`. row_gather: `{out_tile_row: streams → 1 (a block never spans two tile rows), tile_col: spans → block_pages, in_tile_row: spans → 32}` | Input dtype, raw bytes (`Float16_b` for bf16 Phase 0). Dataflow-only payload that never passes through DEST, so audit 2 does not apply. | reader | writer | Whole program, both regimes | **None.** It is the only buffer in page_copy. In row_gather it is live concurrently with `cb_parity_stage`, because phase 2 reads the stage and writes a reserved slot at the same time, so they cannot alias. Capacity equals the live set; `buffer_depth` is the double-buffering mechanism reason. |
| `cb_parity_stage` | `(block_pages · page_bytes + dram_read_align) / dram_read_align` pages of `dram_read_align` bytes. Present only when `regime == row_gather and face_row_bytes < dram_read_align` (bf16 on BH). Absent for fp32, for bf16 on WH, and in page_copy. | `block_pages · page_bytes + dram_read_align` bytes. The block's off-parity rows sit at their destination in-tile offsets, one tile image per column, plus the class-offset slack (base used = CB base + class). | `{out_tile_row: streams → 1, tile_col: spans → block_pages, in_tile_row: spans → 32 (destination row positions; the face spans copied out are contiguous in this layout only)}` | Input dtype, raw bytes. Never pushed or popped. | reader | reader (private scratch) | row_gather, phases 1–2 of each block. It is emptied by the phase-2 barrier before the next block's phase 1. | **Cannot alias `cb_moved_pages`.** Phase 2 copies stage → slot, so both are live at once. The *other* `cb_moved_pages` slot cannot host it either, because the writer owns that slot while the block is staged. A compact (source-row-packed) stage would save L1 but break the one-span-per-face copy, turning 2 copies per tile into one per run (rejected regime `per_run_source_staging`). Capacity exceeds the live rows by the slack only. |

## Symbol table

| Symbol | Bound | Predicate / source establishing it |
|--------|-------|------------------------------------|
| `block_pages` | `1 ≤ block_pages ≤ min(ceil(core_out_pages_max / buffer_depth), cap)`. page_copy: `cap = floor(L1_OP_BUDGET / (buffer_depth · page_bytes))`. row_gather: `cap = floor((L1_OP_BUDGET − s·dram_read_align) / ((buffer_depth + s) · page_bytes))`, also `≤ out_tile_cols`. | Host plan (`select_plan`), regime predicate |
| `buffer_depth` | `= 2` (Phase 0) | Host constant `BUFFER_DEPTH` |
| `s` | `1` if `row_gather and face_row_bytes < dram_read_align`, else `0` | Regime predicate + dtype + HAL |
| `page_bytes` | 2048 (bf16, Phase 0); 4096 (fp32); 1088 (bf8b); `buffer_aligned_page_size()` per stick (RM) | Dtype / layout axis |
| `face_row_bytes` | 32 (bf16), 64 (fp32) | Dtype |
| `dram_read_align` | 64 (Blackhole), 32 (Wormhole) | `ttnn.get_dram_alignment()` / `NOC_DRAM_READ_ALIGNMENT_BYTES` |
| `L1_OP_BUDGET` | `1347776` B per core. TARGET `output_buffer: l1` subtracts `ceil(P / num_l1_banks) · aligned_page_bytes`. | Spec (model leaves this per core); BB-05 |
| `core_out_pages_max` | `≤ ceil(P / num_cores)` | `split_work_to_cores` |
| `P` (output pages) | Unbounded by contract. It never sizes a buffer; it only sets the per-core assignment. | Shape |

## Total per-core footprint

```
footprint = (buffer_depth + s) · block_pages · page_bytes + s · dram_read_align   ≤ L1_OP_BUDGET (by construction of cap)
```

- Every term scales linearly with `block_pages`. `buffer_depth` multiplies the streaming term only. No term scales with K, `P`, or any whole-op dimension.
- Phase 0 worst case, bf16 page_copy with the cap binding: `2 · 329 · 2048 = 1,347,584 B ≤ 1,347,776`.
- Qwen 79 (row_gather, ~130 cores, 15 pages per core, `block_pages = 8`): `3 · 8 · 2048 + 64 = 49,216 B`.
- Qwen 151 (page_copy, 8 pages per core, `block_pages = 4`): `2 · 4 · 2048 = 16,384 B`.
- The plan never depends on free L1, so a production shape gets the same footprint inside the model.

## Data-movement budget

Chosen split: contiguous `out_page` ranges over the full grid, with valid-row-prefix transfers. A full tile is one transfer.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| each input `x_k` | 1. page_copy moves each source page's valid-row prefix once. row_gather reads each logical row exactly once, as disjoint row ranges even when two output tile rows (or two cores) share a source tile. | Every source page (or row range) belongs to exactly one block on one core. No pass re-reads it, and nothing is staged through DRAM. | 0 |
| output | 1, each output page's valid-row prefix written once | Each output page belongs to exactly one block | 0 |
| (row_gather only) off-parity rows | 0 extra DRAM. One extra **core-local** L1→L1 crossing (stage → slot, one span per face per tile). | NoC DRAM-read congruence (64 B on BH) vs. a 32 B bf16 face row | 0 |

Totals per tier for representative production cells:

| Cell | Regime | DRAM read | DRAM write | Core-local L1→L1 | Cross-core |
|------|--------|-----------|------------|------------------|------------|
| Qwen 17 (16 × [1,6,4096,256], d0, bf16) | page_copy | 201,326,592 B | 201,326,592 B | 0 | 0 |
| Qwen 19 (16 × [1,4096,1536], d0, bf16) | page_copy | 201,326,592 B | 201,326,592 B | 0 | 0 |
| Qwen 151 ([1,1,4096,64]+[1,1,4096,192], d−1) | page_copy | 2,097,152 B | 2,097,152 B | 0 | 0 |
| Qwen 109 ([1,6,31,64]+[1,6,31,192], d−1) | page_copy (vr = 31) | 95,232 B | 95,232 B | 0 | 0 |
| Qwen 160 (3 × [1,1,1,16384]+[1,1,1,12928], d−1) | page_copy (vr = 1) | 124,160 B | 124,160 B | 0 | 0 |
| Qwen 80 (same with 16 rows) | page_copy (vr = 16) | 1,986,560 B | 1,986,560 B | 0 | 0 |
| Qwen 79 (16 × [1,1,1,62080], d2) | row_gather | 1,986,560 B | 1,986,560 B (vr = 16) | 1,862,400 B | 0 |
| Qwen 18 (TARGET fp32, [108,4096,128]+[84,4096,128], d0) | page_copy | 402,653,184 B | 402,653,184 B | 0 | 0 |

Comparison points for the rejected alternatives:
- Whole-page transfers would move 3,973,120 B each way on Qwen 160 (32×) and 3,973,120 B on Qwen 80 (2×).
- Whole-tile staging for row_gather would read 63,569,920 B on Qwen 79 (32×).

> Cheapest-traffic split considered: contiguous `out_page` ranges with valid-row-prefix transfers. Every logical row segment crosses DRAM once in and once out, with zero cross-core traffic. That is the minimum against the named boundary; the only above-minimum term is row_gather's core-local parity copy, which no DRAM-tier split removes. Implemented: the same split. **The implemented split is the cheapest.** Bank-affine page sets (lamp L5) move the same bytes in fewer transactions and are a measurement, not a traffic delta.
