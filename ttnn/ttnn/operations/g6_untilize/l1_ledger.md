# L1 Ledger: g6_untilize

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes and knobs: `op_design.md`, Blocking Model.

Block axes: `batch` (folded into the flattened row axis), `rows` (`block_rows`), `tile_cols` (`block_tile_cols`).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_row_staging` | `block_rows * scratch_slots` pages of `block_row_bytes = block_tile_cols * 32 * 4` B; production `1 * 1` page of 512 B | `block_rows * scratch_slots` pages. Every slot in the ring can be live at once (one being gathered, one whose writes have not departed), and each slot holds `block_rows` rows simultaneously because one read barrier covers the whole block. Production: 1 page. | `{batch: streams -> folded into rows, never resident beyond block_rows; rows: spans -> block_rows; tile_cols: spans -> block_tile_cols (page width)}` | UInt32: a dataflow-only payload that never enters DEST, so the width is the data's, not a DEST width; `fp32_dest_acc_en` does not apply (no compute kernel) | `untilize_dm` (NoC gather reads) | `untilize_dm` (NoC page writes), the same single thread | whole kernel; slot `s` from its gather issue until its writes depart (`noc_async_writes_flushed`) | **No other CB exists, so nothing can share an allocation.** Rule 3 patterns: (1) pack into destination: the destination is DRAM, and the NoC cannot move DRAM->DRAM without an L1 hop, so one staging hop is the minimum; (2) transform in place: the source is DRAM, not an L1 buffer; (3) alias: no second buffer to alias with; (4) fold into accumulator: no compute. Capacity equals the live set; `scratch_slots = 1` in production because each core runs one block. |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing the bound |
|--------|---------|-------|----------------------------------|
| `block_rows` | rows per block (block knob) | `1 <= block_rows <= core_rows = ceil(tensor_rows * col_blocks / num_cores)`, and `block_rows * block_row_bytes * scratch_slots <= 1347776` | host plan (`rows_fit_cap`); production 1 |
| `block_tile_cols` | tile columns per block (block knob) | `1 <= block_tile_cols <= Wt`, and `block_tile_cols * 128 * scratch_slots <= 1347776` | host plan (`tile_cols_fit_cap`); production 4 = `Wt` |
| `scratch_slots` | slot-ring depth (depth knob) | `1 <= scratch_slots <= min(SCRATCH_DEPTH = 2, max_blocks_per_core)` | host plan; production 1 |
| `32 * 4` | tile width in values x uint32 element bytes | constant 128 B per tile column per row | TILE layout, dtype uint32 (`SUPPORTED`) |
| `Wt = ceil(W/32)` | op dimension, appears only through `block_tile_cols <= Wt` | bounded by the clamp above, never used unclamped in a capacity | `tile_cols_fit_cap` |
| `1347776` | per-core L1 the model leaves to the op (BB-05) | constant | model environment |

## Total per-core footprint

`L1_op = block_rows * block_tile_cols * 128 * scratch_slots` bytes.

- Scales with `block_rows` (rows knob), `block_tile_cols` (tile_cols knob) and `scratch_slots` (depth knob); nothing else.
- Production: `1 * 4 * 128 * 1 = 512 B`, 0.04 % of the 1347776 B bound.
- No budget solve, safety fraction or blocking search exists. The fit caps only clamp shapes far outside INPUTS.

## Data-movement budget (chosen split: `rows`, 32 cores x 1 row)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x (input, 16384 B) | 1 read | each 64 B face-row piece belongs to exactly one output row, and each row to exactly one core; the gather lands it at its final row position, so it is never re-read | none |
| out (output, 16384 B) | 1 write | each page is assembled whole in one core's slot and written once | none |

Totals per call: DRAM read 16384 B (256 transactions of 64 B); DRAM write 16384 B (32 transactions of 512 B); cross-core NoC 0 B; core-local L1 copies 0 B (the staging slot is written by the NoC read and read by the NoC write, with no L1->L1 pass).

> Cheapest-traffic split considered: `rows` — every candidate split (rows, rows x tile_cols, tile_cols, whole tile row) moves the same 16384 B across DRAM once per tensor with no cross-core traffic; `rows` has the fewest write transactions (32 whole pages) of every multi-core split. Implemented: `rows`. The implemented split is the cheapest; nothing is deferred on traffic grounds. The `rows x tile_cols` variants (more cores, fewer reads per core, 2x-4x the write transactions) are a perf lamp, reachable by turning `block_tile_cols`.
