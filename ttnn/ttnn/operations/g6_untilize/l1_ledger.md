# L1 Ledger: g6_untilize

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions: `op_design.md` → *Blocking Model*.

## Inventory

There is one buffer. Rule 3 was applied before any sizing:

- **Pack into the destination.** The gather lands every face-row piece directly at its final row-major position in the buffer the store reads. No tile-staging buffer exists between "read tiles" and "reorder".
- **Transform in place.** Not needed: the reorder happens in transit (NoC read addressing), not in L1.
- **Alias** and **fold into DEST** do not apply: there is no second buffer and no compute kernel.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_row_staging` | `staging_depth · block_rows` pages of `staged_row_bytes = block_tile_cols · tile_w · elem` bytes. Phase 0: `1 · 1` page of `4 · 32 · 4 = 512` B. | `staging_depth · block_rows` pages. Each slot's rows are all resident between the read barrier and the store issue; at depth 2 one slot fills while the other drains. Capacity = live set. | `{row: spans → block_rows; tile_col: spans → block_tile_cols (page width); batch: folded into row}` | `UInt32` (`ttnn.uint32`): a dataflow-only payload of raw bytes that never enters DEST. There is no compute kernel, so `fp32_dest_acc_en` does not apply (audit 2: mechanism reason "dataflow-only payload"). | `untilize_rows` (NoC reads) | `untilize_rows` (NoC write) | whole kernel | No other CB exists to share with. The input tiles are never staged: the gather writes into this buffer directly (Rule 3, pattern 1). |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing the bound |
|--------|---------|-------|----------------------------------|
| `elem` | bytes per element | `= 4` | dtype ∈ SUPPORTED = {uint32} (validate) |
| `tile_w` | tile width in elements | `= 32` | input tile shape 32×32 (TILE layout of the producer) |
| `Wt` | `ceil(W / 32)` | production 4; unbounded in general | bounds the capacity only through `block_tile_cols ≤ Wt` and the L1 clamp below |
| `block_tile_cols` | tile columns per block | `1 ≤ block_tile_cols ≤ Wt`; Phase 0 `= Wt = 4` | `plan()` L1 clamp |
| `block_rows` | rows per block | `1 ≤ block_rows ≤ max_i n_i`; Phase 0 `= 1` | `plan()` L1 clamp; `n_i` = rows owned by core `i` |
| `staging_depth` | slots | `∈ {1, 2}`; Phase 0 `= 1` | `plan()`: `1` iff every core owns at most one block |
| `L1_BUDGET` | per-core L1 the op may use | `1347776` B | model environment (BB-05) |

Clamp in `plan()`: `staging_depth · block_rows · block_tile_cols · tile_w · elem ≤ L1_BUDGET` (mechanism caps in `op_design.md`).

## Total per-core footprint

`staging_depth · block_rows · block_tile_cols · tile_w · elem` bytes = `staging_depth · block_rows · block_tile_cols · 128` B for uint32.

- Phase 0 (production `(1, 1, 32, 128)` on 8×4 cores): `1 · 1 · 4 · 128 = 512` B per core, 0.04 % of `L1_BUDGET`.
- Scales linearly with `block_rows` (the row knob), `block_tile_cols` (the tile-column knob) and `staging_depth`. No term scales with an unbounded op dimension: the clamp bounds all three.

## Audits

| Audit | Result |
|-------|--------|
| 1. Capacity vs live set | Equal (no over- or under-allocation). The buffer spans both block axes, and its capacity scales with both. |
| 2. Page format vs DEST width | No compute kernel and no DEST; UInt32 is the raw payload. |
| 3. Disjoint lifetimes without justification | None: there is one buffer. |
| 4. Bounds and closed form | Every symbol is bounded above; the total is closed-form. |

## Data-movement budget

Split: `row` across a `cx × cy` rectangle (production 8×4), gather-on-read (regime R1).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `x` (input, 16384 B) | 1 | Each face-row piece is fetched exactly once, by the one core whose row needs it, straight to its row-major L1 position. Tiles are never staged or re-read. Shape: `B·H·2·Wt` reads of 64 B (production 256). | 0 |
| `out` (output, 16384 B) | 1 | Each page is written once, whole, by its one owning core from the staged row. Shape: `B·H` writes of `W·4` B (production 32 × 512 B). | 0 |

Totals per device: DRAM 16384 B read + 16384 B written. Cross-core NoC 0 B. Core-local L1: 16384 B written by the NoC reads and read once by the NoC writes.

> Cheapest-traffic split considered: `row` across cores with gather-on-read (R1). Every legal split moves the same bytes per tier: delta 0 B DRAM and 0 B cross-core for R2 (scatter-on-write), R3 (tile read + L1 reorder) and the tile-column split. R2 also ties on transaction count (288). Implemented: R1, which is also the cheapest split (tied with R2). R2 is deferred because R1 already serves every shape R2 would. R2 reuses R1's skeleton (plan → one gather → one store per block) with the 64 B granularity moved from the read side to the write side.
