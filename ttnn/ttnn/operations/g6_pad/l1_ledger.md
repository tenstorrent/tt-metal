# L1 Ledger: g6_pad

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes (from `op_design.md`):
`page`, `lead`, `tile_row`, `tile_col` (the last three folded into `page`), `tile_interior`.
No compute kernel and no DEST: every page is a dataflow-only payload in the tensor's own dtype.

## CB ledger

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_staged_tiles` (R2 only) | `staging_depth * block_pages`, with `block_pages = min(r_max, block_pages_cap)` and `staging_depth = 1` if `r_max * page_bytes <= staging_budget`, else 2 | `staging_depth * block_pages`. With depth 1 the whole block is resident between the read barrier and the writer's flush; with depth 2 block k is being written while block k+1 lands | `{page: spans → block_pages; lead: streams (folded into page); tile_row: streams (folded into page); tile_col: streams (folded into page); tile_interior: spans → whole tile (page_bytes)}` | input dtype (bf16 Phase 0; fp32/int32/uint32 refinement). Never passes through DEST, so audit 2 does not apply | reader | writer | R2: whole kernel | Cannot share with `cb_fill_page_writer`: concurrent lifetime (fill writes run while staged pages land; that overlap is the stall shadow). Cannot share with `cb_fill_page_reader`: concurrent lifetime (mask copies read the fill page into this CB's slots) |
| `cb_fill_page_reader` (R2 only) | `1` — constant live set (one page of c) | 1 | `{page: streams (same page sources every edge page); lead: streams; tile_row: streams; tile_col: streams; tile_interior: spans → whole tile (mask runs start at any in-tile offset, and each copy sources the same offset)}` | input dtype (bit pattern `c_bits`) | reader | reader | R2: built in block 0's read shadow, read until the last mask barrier | Cannot share with `cb_fill_page_writer`: same content, but one CB read by two kernels breaks the ownership invariant (`ttnn-cb-memory-fundamentals.md`, CB Ownership Invariant). The duplicate costs one page. Could be dropped by masking with RISC stores instead of NoC copies; kept because a NoC copy costs a fixed issue per run while stores cost per byte on the critical path after the read barrier (`noc-alignment-theory.md` §5.2). Mask-mechanism perf lamp |
| `cb_fill_page_writer` (R1 and R2) | `1` — constant live set (one page of c) | 1 | `{page: streams (same page sources every band run / fill page); lead: streams; tile_row: streams; tile_col: streams; tile_interior: spans → whole tile (band runs and whole fill pages at the same in-page offsets)}` | input dtype (bit pattern `c_bits`) | writer | writer | R1/R2: built once at kernel start, read until the final write barrier | Cannot share with `cb_staged_tiles`: concurrent lifetime. Cannot share with `cb_fill_page_reader`: ownership invariant (above). R1 has no other CB |

## Symbol table

| Symbol | Bound | Predicate / source that establishes it |
|--------|-------|----------------------------------------|
| `es` | 2 or 4 | dtype: bf16 → 2; float32, int32, uint32 → 4 |
| `page_bytes` | `1024 * es` ≤ 4096 | TILE layout (R1, R2). R3 (deferred) bounds stick bytes separately |
| `L1_BUDGET_BYTES` | 1347776 | BB-05, host constant |
| `num_fill_pages` | R1: 1; R2: 2 | regime |
| `staging_budget` | `L1_BUDGET_BYTES - num_fill_pages * page_bytes` ≥ 1339584 | derived |
| `r_max` | `ceil(R / num_cores)`, unbounded in principle | `R = S_in * Ht_in * Wt_in`, `num_cores = min(grid_cores, ceil(max(R, F, 1) / MIN_PAGES_PER_CORE))` |
| `staging_depth` | 1 or 2 | 1 iff `r_max * page_bytes <= staging_budget` |
| `block_pages_cap` | `floor(staging_budget / (staging_depth * page_bytes))` ≥ 163 | derived |
| `block_pages` | `min(r_max, block_pages_cap)` | clamped, so `staging_depth * block_pages * page_bytes <= staging_budget` for every shape |
| `MIN_PAGES_PER_CORE` | ≥ 1 (Phase 0: 1) | host constant (core-count knob) |

## Per-core footprint

- R1: `page_bytes` (one fill page). Constant in every knob: 2048 B for bf16.
- R2: `(staging_depth * block_pages + 2) * page_bytes <= L1_BUDGET_BYTES` by construction. Only `cb_staged_tiles` scales with a knob (`block_pages`, `staging_depth`). The two fill pages are constant.
- Phase 0 INPUTS (bf16): R2 cases 64/65/68/69/71/72 have `r_max = 1` → `(1 + 2) * 2048 = 6144 B`. p1 (refinement, fp32): `r_max = ceil(960 / grid_cores)` ≈ 8–9 → `(9 + 2) * 4096 ≈ 45 KB`. Every INPUTS shape runs at `staging_depth = 1`, with one block per core.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input (R1) | 0 reads; band bytes written once: `(b - a) * 2 * row_bytes` per last-row tile, `U = S_in * Wt_in` tiles | the output is a view of the input's buffer; c comes from L1, so nothing is read; only the new logical rows (and the output-padding columns inside the same runs) change | none |
| output (R1) | same bytes as above (one buffer) | — | none |
| input (R2) | 1 read of each of its `R` pages | each staged page stays resident in `cb_staged_tiles` from its read until its write; it is never re-read | none (edge masks are L1→L1 copies on the reader's own core) |
| output (R2) | 1 write of each of its `P` pages; the `F` fill pages are written from L1 and never read | `cb_fill_page_writer` stays resident for the whole kernel | none |

Totals per tier for the named shapes (DRAM; NoC cross-core = 0 B in every case):

| Case | Regime | DRAM read | DRAM written |
|------|--------|-----------|--------------|
| p2 `[1,1,2,62080]` → 32 rows | R1 | 0 B | 1940 × 1920 B = 3,724,800 B |
| p3 `[1,1,8,62080]` → 16 rows | R1 | 0 B | 1940 × 512 B = 993,280 B |
| p4 `[1,1,8,62080]` → 32 rows | R1 | 0 B | 1940 × 1536 B = 2,979,840 B |
| 247 `[1,3844,1536]` → 3872 | R1 | 0 B | 48 × 1792 B = 86,016 B |
| 65 `[1,1,196,196]` → 256² | R2 | 49 × 2048 B = 100,352 B | 64 × 2048 B = 131,072 B |
| p1 `[5,12,128,128]` fp32 → dim 0 = 8 (refinement) | R2 | 960 × 4096 B = 3,932,160 B | 1536 × 4096 B = 6,291,456 B |

> Cheapest-traffic split considered: flattened page split (R1 band units; R2 staged and fill pages). Delta 0 B per tier: it already moves the minimum (each input page at most once, each changed output byte once, no fill page read, no cross-core bytes). Implemented: the same split. Nothing is deferred on traffic grounds. Every other split (by slice, tile row, tile column) moves the same bytes and only loses occupancy; the op has no dependent axis to combine and no shared operand to broadcast.
