# L1 Ledger: g6_permute (Phase 0, regime R1 `rows_tile`)

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking terms (`block_a`, `block_b`,
`block_wt`, cells, row runs): `op_design.md`, Blocking Model.

Inventory: two buffers, minimal for the scheme. The input tiles of a block must be resident because each
one feeds `live_b` output tiles; the output tiles of a block must be assembled somewhere because each one
gathers `live_a` rows from different input tiles. At `block_a = 1` the second buffer disappears: the
input slot is the source of the DRAM write (pattern "pack into the destination": the destination is
DRAM, and the input slot already holds each row contiguously). No compute kernel, no DEST, no scaler,
no intermediate between phases beyond the staging tiles.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` (idx 0, page `tile_bytes`) | `depth_in · block_a · block_wt` | `block_a · block_wt` tiles of the block being regrouped or stored, plus the same count landing for the next block when `depth_in = 2`; per tile only `live_b ≤ block_b` rows are meaningful | `{n: streams → 1 image per block, a: spans → block_a slots, b: spans → block_b rows inside each 32-row slot, wt: spans → block_wt slots, c: spans → whole tile row (2 halves)}` | Float16_b (dataflow-only payload, never in DEST) | reader | writer | `load_block` push → pop after the regroup read barrier (regroup path) or after the store flush (direct path) | Cannot alias `cb_output_staging`: the regroup reads these slots while writing staging slots (concurrent). Capacity above the live rows (a whole 32-row slot for `block_b` rows): mechanism — a row run must land at its source's tile-relative offset to keep the BH mod-64 DRAM read match (`noc_parameters.h:379`), so slots stay whole tiles. Capacity above one block when `depth_in = 2`: double buffering (reader prefetch) |
| `cb_output_staging` (idx 1, page `tile_bytes`) | `depth_staging · block_b · block_wt`; `depth_staging = 0` (not allocated) when `block_a = 1` | `block_b · block_wt` tiles being filled by the regroup, plus the previous block's tiles still draining to DRAM when `depth_staging = 2`; per tile only `live_a ≤ block_a` rows are meaningful | `{n: streams → 1 image per block, a: spans → block_a rows inside each 32-row slot, b: spans → block_b slots, wt: spans → block_wt slots, c: spans → whole tile row}` | Float16_b | writer | writer (private scratch: no CB handshake; slot reuse ordered by `noc_async_writes_flushed`) | `regroup_block` → `store_block` of the same block | Cannot alias `cb_input_tiles` (concurrent, above). Capacity above the live rows: a closed block (`live_a = la`) writes the slot as one whole tile, which needs the rows at their output positions in a full tile; an a-split block writes runs from the same positions, so one slot layout serves both. Capacity above one block when `depth_staging = 2`: overlap of block k+1's regroup with block k's write drain. Not needed at `block_a = 1`: considered and dropped there (direct writes) |

### Symbol table

| Symbol | Bound | Predicate / source |
|--------|-------|--------------------|
| `tile_bytes` | 2048 | bfloat16 32×32 tile; host asserts `tile_shape == [32, 32]` and `face_shape == [16, 16]` |
| `block_a`, `block_b` | ∈ {1, 2, 4, 8, 16, 32} | `select_plan` walk (`op_design.md`, selection rule) |
| `block_wt` | 1 | host constant `BLOCK_WT` |
| `depth_in` | ∈ {1, 2} | `min(MAX_DEPTH_IN = 2, max_blocks_per_core)` |
| `depth_staging` | ∈ {0, 1, 2} | `0` if `block_a = 1`, else `min(MAX_DEPTH_STAGING = 2, max_blocks_per_core)` |
| `max_blocks_per_core` | `ceil(num_blocks / num_cores)` | `split_work_to_cores` |
| `live_a`, `live_b` | ≤ `block_a`, ≤ `block_b` | ragged last group of a tile row; never sizes a buffer |

No whole-op dimension (`A`, `B`, `W`, `N`) appears in any capacity.

### Total per-core footprint

`L1_op = tile_bytes · block_wt · (depth_in · block_a + depth_staging · block_b)`

- `cb_input_tiles` scales with `block_a · block_wt · depth_in`; `cb_output_staging` with `block_b · block_wt · depth_staging`.
- Upper bound over every legal knob value: `2048 · 1 · (2·32 + 2·32) = 262144 B`, 19.5 % of the 1347776 B the model leaves. Phase 0 output is in DRAM, so no output page counts against it.
- Production (Qwen) plans: `block_a = 1`, `depth_in = 1`, no staging → `2048 B` per core.
- No budget predicate, safety fraction or blocking search exists: the walk is the occupancy step, not an L1 solve.

## Data-movement budget (chosen split: R1, occupancy walk)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x | 1 | every live input row is read by exactly one block: the a-groups partition the input tiles, the b-groups partition their rows; a tile is read whole (padding rows included) when its block's b-span covers all its live rows, else as 2 row runs of live rows only | 0 |
| out | 1 | every live output row is written by exactly one block: whole tile from staging when the a-span covers the tile's live rows (padding rows written with stale bytes), else 2 row runs; at `block_a = 1` two 32-B runs per output tile straight from the input slot | 0 |

Core-local L1 tier: each live row is copied once (input slot → staging) when `block_a > 1`; zero copies when `block_a = 1`.

Totals per tier (DRAM read / DRAM write / L1 local / cross-core):

| Case | Plan (G = 110) | DRAM read | DRAM write | L1 local | Cross-core |
|------|----------------|-----------|------------|----------|------------|
| Qwen 304 `(1,16,6,64)` | `(1, 2)`, 96 blocks | 12288 B in 192 runs of 64 B (= live bytes) | 12288 B in 384 runs of 32 B | 0 | 0 |
| Qwen 303 `(1,6,16,64)` | `(1, 2)`, 96 blocks | 12288 B in 192 runs | 12288 B in 384 runs | 0 | 0 |
| Qwen 302 `(1,16,1,64)` | `(1, 1)`, 32 blocks | 65536 B in 32 whole tiles (2048 B live) | 2048 B in 64 runs | 0 | 0 |
| Qwen 301 `(1,1,16,64)` | `(1, 1)`, 32 blocks | 2048 B in 64 runs | 2048 B in 64 runs | 0 | 0 |
| `(3,40,50,512)` closed | `(32, 32)`, 192 blocks | 7864320 B in 3840 tiles (6144000 B live) | 9830400 B in 4800 tiles (6144000 B live) | 6144000 B in 192000 copies of 32 B | 0 |

> Cheapest-traffic split considered: R1 with row-run transfers for every partially live tile (never a
> whole tile unless all 32 rows are live) — saves the padding rows of ragged tile rows: Qwen 302
> 63488 of 65536 read bytes, `(3,40,50,512)` 1720320 read and 3686400 write bytes; equal DRAM
> crossings (each live row once), no cross-core change, 2–4 transactions instead of 1 on those tiles
> (4 when the live rows cross the row-16 face band, e.g. `lb = 18`).
> Implemented: R1 with a whole-tile transfer when a block covers all live rows of a tile. Not taken
> because the crossing count is equal and, at these sizes, the per-transaction cost sets the time,
> not the bytes (`tile_reorder` catalog entry: whole-page transfers beat sub-tile ones); the run
> form is the same row-run transfer code under a threshold, so it stays one comparison away —
> recorded as the "whole-tile vs live-row reads" perf lamp in `op_design.md`. Among splits with
> different crossing counts, the implemented one is the cheapest (X1 re-reads x up to 32×, X5 adds a
> cross-core copy of all live bytes).
