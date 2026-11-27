# L1 Ledger: g6_untilize

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes: `rows` (with `batch`
folded in) and `tile_cols` (`op_design.md`, Blocking Model).

## Inventory

There is one buffer. The op has no compute phase, so no phase boundary needs storage. The reader packs each
face-row piece straight into its final row-major position (Rule 3 pattern 1: pack into the destination). The
writer drains that same buffer to DRAM. No scratch, staging or tile-order copy exists: the
`tile_stage_local_reorder` regime that would need one is rejected in the regime table. No budget solve, safety
fraction or blocking search was introduced. `block_tile_cols` has a fit check only so that the bound below
holds for every shape, and at Phase 0 it never binds.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_row_block` | `block_tile_cols · buffer_depth` tile pages of `tile_bytes`. Phase 0: `4 · 1 = 4` pages = 16384 B | one block per slot: `block_rows` rows × `chunk_row_bytes`, at stride `padded_chunk_row_bytes` inside the slot's `block_tile_cols` pages. Phase 0: 1 row = 512 B, inside a 16384-B slot | `{rows: spans → block_rows ≤ TILE_HEIGHT (capacity sized at the cap: one tile page column holds 32 rows), tile_cols: spans → block_tile_cols, batch: streams (folded into rows), blocks-per-core: streams → buffer_depth window}` | UInt32. Dataflow-only payload: NoC in, NoC out, never unpacked and never in DEST, so the DEST-width audit does not apply | reader | writer | whole kernel; each slot lives from `cb_reserve_back` of a block to `cb_pop_front` after that block's write barrier | Nothing to share with: it is the only buffer. **Capacity > live set at Phase 0** (16384 B allocated, 512 B live) for a mechanism reason. The store helper's wait/pop quantum is `width_in_tiles` tile pages per ≤ 32-row block (`tilize_helpers_dataflow.inl:209,229,239`), so a slot is sized for a full 32-row block whatever `core_rows` is. Shrinking it would mean replacing the helper; the slack is 16 KB of a 1347776-B budget |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing the bound |
|--------|---------|-------|----------------------------------|
| `tile_bytes` | one 32×32 uint32 tile | = 4096 | dtype = uint32 (SUPPORTED) |
| `TILE_HEIGHT` | rows per tile page column, rows-per-wait cap of the store helper | = 32 | CB tile shape 32×32 (`tile_constants.h:18`) |
| `tensor_tile_cols` | `ceil(W/32)` | any ≥ 1 (production 4) | input shape |
| `block_tile_cols` | tile columns per block | `1 ≤ block_tile_cols ≤ tensor_tile_cols`; divides `tensor_tile_cols`; `buffer_depth · block_tile_cols · tile_bytes ≤ L1_OP_BUDGET_BYTES` | `derive_block_tile_cols` (largest divisor that fits); production 4 |
| `buffer_depth` | slots in flight | `1 ≤ buffer_depth ≤ ROW_BLOCK_DEPTH = 2` | `min(ROW_BLOCK_DEPTH, max_blocks_per_core)`; production 1 |
| `block_rows` | rows in one block | `1 ≤ block_rows ≤ TILE_HEIGHT` | `min(core_rows − 32k, TILE_HEIGHT)`; production 1 |
| `padded_chunk_row_bytes` | row stride inside a slot | `= block_tile_cols · 128` (a multiple of 64) | helper's `round_up(row_bytes, tile_row_bytes)` (`tilize_helpers_dataflow.inl:208`) |
| `L1_OP_BUDGET_BYTES` | model's free L1 per core (BB-05) | = 1347776 | spec, Model environment |

## Per-core footprint

`L1_op = buffer_depth · block_tile_cols · tile_bytes`

- scales with `block_tile_cols` (linearly) and `buffer_depth` (linearly). It is constant in `block_rows`,
  because the tile-page quantum already provides 32 row slots.
- Phase 0: `1 · 4 · 4096 = 16384 B` per core on each of 32 cores, against the 1347776-B bound.
- Worst case for any shape: `2 · block_tile_cols · 4096 ≤ L1_OP_BUDGET_BYTES` by the `block_tile_cols`
  derivation. With `block_tile_cols ≥ 1` this needs only 8192 B, so no shape needs a host refusal.

## Data-movement budget

Chosen split: `rows` only, regime `row_gather`, Phase 0 shape `(1, 1, 32, 128)`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x (input, TILE) | 1 read: 16384 B as 256 reads × 64 B | each face-row piece is read once, by the core that owns its output row, directly into its final L1 position. No piece is resident twice or re-read | 0 |
| out (output, ROW_MAJOR) | 1 write: 16384 B as 32 writes × 512 B | each row page is written whole, once, by its owning core from the gathered block | 0 |

Totals per call: DRAM 16384 B read + 16384 B written, in 288 transactions; cross-core NoC 0 B; core-local
L1 copies 0 B (no byte is moved within L1).

> Cheapest-traffic split considered: `rows` only. It is the implemented split, at minimum bytes per tier
> (each tensor crosses DRAM exactly once, nothing crosses between cores) and with the fewest DRAM
> transactions of any minimum-byte split (`rows × tile_cols` and `tile_cols` only: +96 partial-page writes).
> The one split with an equal transaction count, the mirror `tile_scatter` (32 × 512-B reads + 256 × 64-B
> writes), moves the same bytes. It is a `deferred` regime row because `row_gather` covers every INPUTS
> shape with identical traffic and writes every output page whole from one core. The shared rectangle, CB
> and row ranges keep it reachable.
