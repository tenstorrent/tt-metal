# L1 Ledger: g6_pad

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Axes are those of `op_design.md` § Blocking Model. Each regime is its own program, so an axis of the other regime is tagged `n/a (other program)`.

Axis list: `slice`, `band_tile_col`, `band_unit` (extension_band); `tile_row`, `row_in_tile` (both); `tile_col`, `out_page`, `col_in_tile` (new_storage_tiles).

Inventory decisions, made before any sizing:
- No compute kernel, so no compute-side CBs.
- `new_storage_tiles` has one staging CB, not an input CB plus an output CB. The reader packs each input page into the slot the writer stores from (Rule 3 pattern 1). Masking is in place in that slot while the reader still owns it, between `cb_reserve_back` and `cb_push_back`. That gives real CB credit and no second thread on the slot (pattern 2).
- Fill-only pages get a constant buffer, not slots. One build per core instead of one full-tile store loop per fill page. It also lets the writer issue them before the first hand-off (the stall shadow in `op_design.md`).

## CB table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_fill_source` (extension_band, idx 0) | `1` (page = `page_bytes`) | `fill_source_bytes = max(band_segments bytes) ≤ page_bytes`, so 1 page | `{slice: streams, band_tile_col: streams, band_unit: streams (every band tile sources the same bytes), tile_row: streams (fixed extent 1), row_in_tile: spans → fill_source_bytes, tile_col: n/a (other program), out_page: n/a (other program), col_in_tile: n/a (other program)}` | input dtype bit pattern. A dataflow-only payload that never passes through DEST, so the DEST-width audit does not apply | writer (`build_fill_source`) | writer (NoC write source) | whole kernel | Only buffer of its program, so nothing to share. Capacity exceeds the live set by up to one page minus the longest segment. The reason is page granularity: a fixed one-page CB covers every padding, and `fill_source_bytes` is set per call by the segment table |
| `cb_assembled_tiles` (new_storage_tiles, idx 0) | `block_tiles · eff_depth` | `block_tiles · eff_depth`. With `eff_depth = 2` (≥ 2 blocks per core), block `k+1` loads while block `k` drains. With 1 block per core (every production shape) `eff_depth = 1` | `{out_page: spans → block_tiles × eff_depth, slice: spans via out_page (folded), tile_row: spans via out_page (folded), tile_col: spans via out_page (folded), row_in_tile: spans (whole page, masked in place), col_in_tile: spans (whole page, masked in place), band_tile_col: n/a (other program), band_unit: n/a (other program)}` | input dtype, raw tile page (never unpacked) | reader | writer | per block, whole kernel | Cannot alias `cb_fill_page`: the lifetimes are concurrent, because the writer sources fill pages while the reader loads blocks. It already serves as both the input staging and the output staging buffer |
| `cb_fill_page` (new_storage_tiles, idx 1) | `1` | 1 page on cores with `core_fill_pages > 0`, 0 elsewhere. It is allocated on every core of the range because a CB config covers the core range, at a cost ≤ 4 KB | `{out_page: streams (sourced by every fill-only page), slice: streams, tile_row: streams, tile_col: streams, row_in_tile: spans (whole page of c), col_in_tile: spans (whole page of c), band_tile_col: n/a (other program), band_unit: n/a (other program)}` | input dtype bit pattern; dataflow-only payload | writer (`build_fill_page`) | writer (NoC write source) | whole kernel | Cannot alias `cb_assembled_tiles` (concurrent). Folding it into slots was rejected: that costs a full-tile store loop per fill page instead of one per core, and it serializes fill pages behind the reader hand-off |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `page_bytes` | 2048 (bfloat16) or 4096 (float32/int32/uint32) | TILE layout with 32×32 tiles (host asserts `tile_shape == (32, 32)`); dtype ∈ TARGET |
| `fill_source_bytes` | `≤ page_bytes − face_row_bytes` (rows `[row_rem, row_mask_end)` with `row_rem ≥ 1`) | `row_segments(a, b)` ranges lie inside one tile; regime predicate `extension_band` (`row_rem > 0`) |
| `block_tiles` | `1 ≤ block_tiles ≤ l1_block_cap = floor((1 347 776 − page_bytes) / (page_bytes · buffer_depth))` = 328 (bf16) / 164 (4-byte dtypes) | `make_plan()` clamp; `block_tiles = min(max(1, max_core_input_pages), l1_block_cap)` |
| `max_core_input_pages` | `≤ ceil(out_pages / num_cores)` | `split_work_to_cores` contiguous runs |
| `eff_depth` | `∈ {1, 2}` | `eff_depth = min(buffer_depth = 2, max_num_blocks_per_core)` |
| `buffer_depth` | 2 | module constant `BUFFER_DEPTH` |
| `L1_OP_BUDGET_BYTES` | 1 347 776 | BB-05 (the model leaves this much L1 per core) |

## Per-core footprint

| Regime | Total (bytes) | Scales with | Production values |
|--------|---------------|-------------|-------------------|
| extension_band | `page_bytes` | nothing (constant) | 2048 (bf16) |
| new_storage_tiles | `page_bytes · (block_tiles · eff_depth + 1)` ≤ `L1_OP_BUDGET_BYTES` by construction of `l1_block_cap` | `block_tiles` (block knob) and `eff_depth` (depth knob, only with ≥ 2 blocks); the `+1` fill page is constant | Phase 0 shapes: `2048 · (1·1 + 1)` = 4096. p1 (refinement): `4096 · (14·1 + 1)` = 61 440 |
| new_storage_rows (deferred) | `stick_bytes_out · (block_rows · eff_depth + 1)` | `block_rows`, `eff_depth` | — |

No shape is ever refused for L1: `block_tiles` is clamped, so the coarsest-fitting block always exists. No budget solve or safety fraction exists beyond that clamp, and the inventory (≤ 2 CBs) is minimal.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input (`extension_band`) | 0 | the band is written from an in-core constant; rows `< row_rem` are never read or written | 0 |
| output = input's buffer (`extension_band`) | the new rows once: `band_units · Σ band_segments` B. p2 3 724 800; p3 993 280; p4 2 979 840; `[1,3844,1536]` 86 016 | each new logical row is written once; no other tile row is addressed | 0 |
| input (`new_storage_tiles`) | 1× (every input page is input-backed) | each input page feeds exactly one output page and is read once into its slot | 0 |
| output (`new_storage_tiles`) | 1× every page | assembled in L1 and written once; fill-only pages come from `cb_fill_page`, never from DRAM | 0 |
| fill value `c` | 0 | 4-byte common RT arg, expanded in L1 | 0 |

Totals per tier:

| Regime | DRAM read | DRAM write | Cross-core NoC | Core-local L1 work |
|--------|-----------|------------|----------------|--------------------|
| extension_band | 0 | `band_units · Σ band_segments` | 0 | `fill_source_bytes` of stores once per core |
| new_storage_tiles | `in_pages · page_bytes` (`[1,1,196,196]`: 100 352; p1: 3 932 160) | `out_pages · page_bytes` (`[1,1,196,196]`: 131 072; p1: 6 291 456) | 0 | mask stores on boundary pages; one fill page per core with fill pages |

> Cheapest-traffic split considered: contiguous runs of `band_unit` (extension_band) and of `out_page` (new_storage_tiles). Every candidate split moves identical bytes per tier (no reuse-shared operand, no overlapping footprint, no dependent axis), so the delta is 0 B at every tier. Implemented: the same splits. The implemented split is the cheapest.
