# L1 Ledger: g6_pad

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md` → Blocking Model:
extension_band: `band` (flat slice × tile-column band index, extent `band_block_tiles`), `band_rows` (rows `[band_row_begin, band_row_end)` inside a band tile, not split).
new_storage_tiles: `out_tile` (flat slice × tile-row × tile-column output index, extent `block_tiles`).
No compute kernel, no DEST: every page holds raw input-dtype bytes (`fp32_dest_acc_en` does not apply; audit 2: dataflow-only payload).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_fill_tile` | 1 (constant: one tile of `c`; every band segment / fill-only tile is a sub-range or the whole of it) | 1 | `{band: streams (every band tile re-sources the same page), band_rows: streams (segments are sub-ranges of the page), out_tile: streams (every fill-only tile re-sources it)}` | input dtype bits (bf16 / fp32 / int32 / uint32; no DEST) | writer (`band_writer` in extension_band) | writer (NoC write source) | whole kernel | Cannot share with `cb_input_tiles`: concurrent lifetime (it is the source of fill-only writes issued while `load_block` fills `cb_input_tiles`). Not dropped in favor of a hardware zero region: `c` is per-call data and is non-zero in production (`-2^127`, `-1`). |
| `cb_input_tiles` (new_storage_tiles only; not allocated in extension_band) | `input_depth_blocks · block_tiles`, `input_depth_blocks = min(2, num_blocks_max)` | `block_tiles` (one block assembled at a time when one block per core; `2 · block_tiles` when the L1 cap splits a run and reads of block `i+1` overlap writes of block `i`) | `{out_tile: spans → block_tiles (slot k ↔ k-th output tile of the block), band: n/a (other regime), band_rows: n/a}` | input dtype bits | reader | writer (patches edge slots in place while it owns them, then sends) | per block | Is itself the assembly buffer (pack-into-destination: patch happens in place, no separate output CB — rule 3 patterns 1+2). Fill-only slots are reserved but unwritten: capacity over the source-tile live set is the price of the nominal push count that closes the ring cycle (blocking-model §2); bounded by `block_tiles`. |
| `cb_fill_tile_lane1` (extension_band only, index 2; allocated when `band_lanes = 2`, the default) | 1 | 1 | `{band: streams (the RISCV_1 / NOC0 lane's half of the per-core band range re-sources it), band_rows: streams}` | input dtype bits | band writer lane 1 (RISCV_1) | same kernel (NoC write source) | whole kernel | Cannot share with `cb_fill_tile`: each lane's kernel owns its own fill page (single producer / single consumer per CB; the design's `band writes on one RISC` lamp). Measured win of the second lane: p2 14.0 -> 11.9 us, p4 12.0 -> 9.8 us. |
| `cb_input_tiles` in new_storage_rows (same index 1, same kernels, `rows_mode` CT flag) | `input_depth_blocks · block_tiles` slots of `slot_bytes = align64(row_chunk_bytes)` | `block_tiles` units | `{out_row: spans → rows of the block, row_chunk: spans → chunk wt of each row (slot k ↔ k-th (row, chunk) unit)}` | input dtype bits | reader (input part of each source chunk, at slot start) | writer (fills the row tail `[d_w, D_w)` in place: RISC head to 16 B, local NoC copy from the fill page for the rest; then sends) | per block | Is itself the assembly buffer (pack-into-destination); fill-only (row, chunk) units are reserved-but-unwritten slots, as in the tiles regime. `block_tiles_cap = (1347776 − slot_bytes) // (depth · slot_bytes)`; `row_chunk_bytes ≤ 1347776 // (1 + depth)` so one unit always fits. |
| `cb_fill_tile` in new_storage_rows | 1 page of `slot_bytes` | 1 | `{row_chunk: streams (every fill-only chunk / tail re-sources it)}` | input dtype bits | writer (RISC-seeded `fill_seed_bytes` = 256 B, grown by local NoC doubling copies) | writer (NoC write source) | whole kernel | as in the tiles regime (concurrent with `cb_input_tiles`). |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `tile_bytes` | 2048 (bf16), 4096 (fp32/int32/uint32) | dtype ∈ TARGET |
| `block_tiles` | `min(core_max_tiles, block_tiles_cap)`, `block_tiles_cap = floor((1347776 − tile_bytes) / (max(1, input_depth_blocks knob) · tile_bytes))` → ≤ 328 (bf16), ≤ 164 (32-bit) at the default `input_depth_blocks = 2` (the cap divides by the depth knob, `make_plan`) | L1 budget BB-05 |
| `core_max_tiles` | `ceil(N_out / ncores)`; INPUTS: 1 for the bf16 `[1,1,256,256]` and `[1,256,128]` outputs (64 and 32 tiles), p1 `ceil(1536 / ncores)` ≈ 12–14 | `split_work_to_cores` |
| `num_blocks_max` | `ceil(core_max_tiles / block_tiles)`: 1 when `core_max_tiles < pipeline_min_units` (8); otherwise `block_tiles = 2` (Refinement 2) → p1 `[5,12,128,128]` fp32: 14 tiles/core → 7 blocks, `cb_input_tiles` = 2·2 = 4 tiles = 16 KB (was 14 tiles = 56 KB) | `KNOBS` block_tiles / pipeline_min_units |
| `input_depth_blocks` | `min(2, num_blocks_max)` ∈ {1, 2} | above |

## Footprint

`L1(core) = tile_bytes · (1 + input_depth_blocks · block_tiles)` (new_storage_tiles); `band_lanes · tile_bytes` (extension_band; 4 KB bf16 at the default 2 lanes).
Scales with: `block_tiles` and `input_depth_blocks` (`cb_input_tiles`); nothing else. Max over INPUTS: p1 fp32 ≈ 4096 · (1 + 14) ≈ 61 KB; bf16 `[1,1,256,256]` 64 tiles / 64 cores → 4 KB; extension 2 KB. Hard bound by construction: ≤ 1347776 B.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input (extension_band) | 0 | the band is write-only: the logical rows of the last tile row are never touched, so nothing needs reading | none |
| output band (extension_band) | 1 × band bytes = `in_slices · in_tiles_w · 2 · (band_row_end − band_row_begin) · row_bytes` | output = view of the input storage; only rows that must become `c` are written | none |
| input (new_storage_tiles) | 1 | each input tile maps to exactly one output tile; one block per core holds it until sent | none |
| output (new_storage_tiles) | 1 | each output page assembled once in L1 (patch in place) or sourced from the fill page | none |
| input (new_storage_rows) | 1 | each input row's bytes belong to exactly one (row, chunk) unit; fill-only rows / chunks are never read | none |
| output (new_storage_rows) | 1 | each (row, chunk) assembled once in L1 (tail filled in place) or sourced from the fill page | none |
| fill value `c` | 0 | materialized per core from one RT arg | none |

Totals: extension_band — DRAM read 0 B, DRAM write = band bytes (p4 `[1,1,8,62080]→32`: 1940 · 2 · 24 · 32 B = 2.98 MB; p2: 1940 · 2 · 30 · 32 B = 3.72 MB; p3: 1940 · 2 · 8 · 32 B = 0.99 MB). new_storage_tiles — DRAM read = input tile bytes, DRAM write = output tile bytes (p1: 3.93 MB / 6.29 MB). NoC cross-core: 0.

> Cheapest-traffic split considered: flat band-tile range (extension) / flat output-tile range (new_storage) — delta 0 B per tier against the minimum. Implemented: the same splits. Deferred: none on traffic grounds (the op has no dependent or reuse-shared axis; `new_storage_rows` was built in Refinement 1 with the same minimum: input 1×, output 1×).

**Refinement 2 (pipeline knobs).** `block_tiles = 2` at `input_depth_blocks = 2` once a core holds ≥ `pipeline_min_units` (8) units; below that, one block per core (unchanged). The CB shrinks for p1 (56 KB → 16 KB). `cb_fill_tile`'s first `fill_seed_bytes` (256 B) are RISC stores, and the rest of the page is grown by local NoC doubling copies. The CB inventory is unchanged and no buffer was added. DRAM crossings are unchanged: each source tile is read once and each output page is written once.
