# L1 ledger: rms_norm (regime `row_resident`, target `quasar`)

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`, Blocking Model): `tile_rows` (extent `rows_per_block` = 1), `w_tiles` (extent `block_wt` = Wt), intra-tile rows (fixed 32, inside every tile), the weight operand (`gamma_block_tiles`). Every capacity is the ring's total `num_entries`: per-NEO capacity × `C_eff` (STRIDED rings are sliced per NEO, also for compute-owned rings). `fp32_dest_acc_en = True` in Phase 0.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `IN_DEPTH_ROWS · rows_per_block · block_wt · C_eff` | `rows_per_block · block_wt` per NEO (the whole row is waited cumulatively in pass 1 and held through pass 2) | `{tile_rows: spans → rows_per_block, w_tiles: spans → block_wt, intra-tile: spans → 32 (in the tile), weight: — }` | Float16_b (input dtype; a dataflow payload that has not passed through DEST) | reader | compute | whole kernel, one row per round per NEO | none: it is live while every other ring is live (the row is held across all four compute operations); capacity = live set (depth 1, no double buffer: the per-tile pops of pass 2 let the reader refill early instead) |
| `cb_gamma_tiles` (ring `gamma_tiles`) | `GAMMA_DEPTH_BLOCKS · gamma_block_tiles · C_eff` | `gamma_block_tiles` per NEO (one block being consumed) | `{tile_rows: streams, w_tiles: streams → window gamma_block_tiles, intra-tile: row 0 only meaningful, weight: spans → gamma_block_tiles}` | Float16_b (mechanism: Quasar's `UnaryBcast` refuses 32-bit pages, `bcast.h:113`; the weight is bf16) | reader | compute | whole kernel (pass 2 of each row) | not with `cb_input_tiles` (concurrent); not aliased with `cb_output_tiles` (concurrent, different producer); capacity is 2× the live set for double buffering (assembly of block j+1 while block j is consumed) |
| `cb_scaler` | `1 · C_eff` | 1 per NEO | `{tile_rows: streams (constant), w_tiles: streams (constant), intra-tile: row 0 of each face, weight: —}` (constant, a bare 1 is legal) | Float16_b (reduce scaler format; 1.0 exact) | reader | compute | whole kernel (popped after the last round) | not shareable: held for the whole kernel while every other ring cycles; a constant tile of a different producer |
| `cb_row_stat` | `rows_per_block · C_eff` | `rows_per_block` per NEO (one tile per tile-row of the unit) | `{tile_rows: spans → rows_per_block, w_tiles: streams (Σ over the row collapses into one tile), intra-tile: spans → 32, weight: —}` | Float32 (follows DEST width; the Σx² and inv_rms values come straight from fp32 DEST) | compute | compute | per row: Σx² (`sum_squares_block` → `reduce_row_block`), then inv_rms (`inv_rms_block` → end of `normalize_block`) | **already shared**: Σx² and inv_rms are two disjoint lifetimes in one ring (Rule 3, pattern 3). Not with `cb_row_sum`: `inv_rms_block` reads `cb_row_sum` while it writes `cb_row_stat`, and a helper call whose input and output are one ring is avoided (`reduce<>` and the chain would hold an input and reserve an output on one 1-entry slice) |
| `cb_row_sum` | `rows_per_block · C_eff` | `rows_per_block` per NEO | `{tile_rows: spans → rows_per_block, w_tiles: streams, intra-tile: column 0 meaningful, weight: —}` | Float32 (follows DEST width) | compute | compute | per row: `reduce_row_block` → `inv_rms_block` | not with `cb_row_stat` (concurrent within `reduce_row_block` and `inv_rms_block`, above). It exists only because the post-op is kept out of `reduce<>`; perf lamp 2 removes it |
| `cb_output_tiles` | `OUT_DEPTH_TILES · C_eff` | 1 per NEO (one packed tile in flight) | `{tile_rows: streams, w_tiles: streams → window OUT_DEPTH_TILES, intra-tile: spans → 32, weight: —}` | Float16_b (output dtype: the ABI requires it; the pack narrows fp32 DEST once) | compute | writer | whole kernel (pass 2) | not with `cb_input_tiles`: different producer and consumer, and the input slot is freed only after its tile is unpacked, so packing in place into the input ring would hand the writer a compute-produced entry on a reader-produced ring (two producers); capacity 2× the live set for double buffering |

## Symbol table

| Symbol | Bound | Predicate establishing it | Value on this target |
|--------|-------|---------------------------|----------------------|
| `tile_bytes` | 2,048 (bf16 tile) | Phase 0 `dtype = bfloat16`; `input.buffer_page_size()` | 2,048 |
| `Wt` = `block_wt` | `Wt · C_eff · tile_bytes ≤ DFB_EXTENT_CAP_BYTES` | regime predicate `c_fit ≥ 1` (`row_resident`) | ≤ 511 at C_eff = 1; ≤ 255 at 2; ≤ 127 at 4. Phase 0 INPUTS: Wt ∈ {2, 4, 8, 16, 64, 80, 128, 256} |
| `C_eff` | `min(THREAD_PLAN.C = 4, c_fit, pow2floor(num_tile_rows))` ∈ {1, 2, 4} | host derivation | 1, 2 or 4 |
| `rows_per_block` | 1 | host constant `ROWS_PER_BLOCK` | 1 |
| `IN_DEPTH_ROWS` | 1 | host constant | 1 |
| `gamma_block_tiles` | largest divisor of Wt ≤ `GAMMA_BLOCK_MAX_TILES` (16) | host derivation | 2 … 16 |
| `GAMMA_DEPTH_BLOCKS` | 2 | host constant | 2 |
| `OUT_DEPTH_TILES` | 2 | host constant | 2 |
| `DFB_EXTENT_CAP_BYTES` | 1,048,560 per ring (uint16 extent in 16-B units) | target page *Cluster model*; `eval/targets/quasar.yaml:89` | 1,048,560 |
| `budget` | `ttnn.get_max_worker_l1_unreserved_size() − L1_RESERVE_BYTES` | run-time query; `L1_RESERVE_BYTES` = 524,288 leaves room for L1-resident input, output and weight banks (≤ 64 KiB per bank across INPUTS), which the query does not subtract | 3,963,328 − 524,288 = 3,439,040 |

## Total per-core footprint

```
footprint = C_eff · [ IN_DEPTH_ROWS · rows_per_block · block_wt · tile_bytes          (input: scales with block_wt, rows_per_block, IN_DEPTH_ROWS, C_eff)
                    + GAMMA_DEPTH_BLOCKS · gamma_block_tiles · tile_size(GAMMA_FORMAT) (gamma: scales with gamma_block_tiles, depth, C_eff)
                    + 2,048                                                             (scaler: constant per NEO)
                    + 2 · rows_per_block · stat_bytes                                   (row_stat + row_sum: stat_bytes = 4,096 with fp32 DEST, 2,048 without)
                    + OUT_DEPTH_TILES · tile_bytes ]                                    (output: scales with depth)
predicate: every ring ≤ DFB_EXTENT_CAP_BYTES  and  footprint ≤ budget
```

Worst cases on this target (bf16):

| Case | Input ring | Gamma | Scaler | Row stats | Output | Total |
|------|-----------:|------:|-------:|----------:|-------:|------:|
| C_eff = 4, Wt = 127 (largest at 4; gamma block 1) | 1,040,384 | 16,384 | 8,192 | 32,768 | 16,384 | 1,114,112 |
| C_eff = 4, Wt = 112 (gamma block 16) | 917,504 | 262,144 | 8,192 | 32,768 | 16,384 | 1,236,992 |
| C_eff = 1, Wt = 256 ((1,1,32,8192); at C = 2 the ring would be 1,048,576 B, 16 B over the cap) | 524,288 | 65,536 | 2,048 | 8,192 | 4,096 | 604,160 |
| C_eff = 2, Wt = 128 ((1,1,128,4096)) | 524,288 | 131,072 | 4,096 | 16,384 | 8,192 | 684,032 |

Every case is under 1.3 MB against a 3,439,040-B budget. On this target the ring extent cap binds and the budget does not. The budget predicate stays in the host check for parts with less L1.

## Data-movement budget (chosen split: `tile_rows` across clusters and NEOs, row resident)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | 1 (+ at most `C_eff − 1` padded tile-rows in the whole op) | the tile-row is resident in `cb_input_tiles` across both passes (`block_wt = Wt`) | none |
| weight (gamma) | twice per tile-row: `2 · num_tile_rows · W · 2 B` = 1/16 of the input bytes (implemented; the design had 1/32) | streamed per row per NEO (`gamma_resident` deferred). **Implementation change:** each gamma entry is assembled from two whole-page (64 B) NoC reads of the same weight row, to entry + 0 and entry + 480, instead of one read plus a 32-B RISC copy (+32 → +512): Quasar DM cores have write-back L1 D-caches (`reduce_helpers_dataflow.inl`, the `flush_l2_cache_range` note), so a RISC copy of NoC-written bytes would need an invalidate before the read and a flush after the write. The transaction count is the same (2 per gamma tile); the DRAM bytes double, still 1/16 of the input | none |
| reduce scaler | 0 | generated in L1 by the reader | none |
| output | 1 (+ the same padded duplicates) | written once from `cb_output_tiles` | none |

Total per tier, for an input of B bytes: DRAM reads ≈ B + B/16 (gamma read twice per entry, above), DRAM writes ≈ B; cross-core NoC 0; core-local L1 reads: the input row is unpacked twice from L1 (pass 1 and pass 2), gamma once, row stats a few tiles per row.

> Cheapest-traffic split considered: `tile_rows` split with the row resident (`row_resident`): it is the minimum on input and output, and every other split moves more (`w_split` adds G·C × 4 KiB of partials plus one 4 KiB multicast per row; `row_streaming` reads the input twice). Implemented: `row_resident`. **The implemented split is the cheapest.** The deferred `w_split` is not a traffic win; it is the occupancy refinement for shapes with fewer tile-rows than NEO slots (the model calls), and Phase 0's structure keeps it reachable (op_design.md, Regimes).

## Implementation notes (ttnn-implementer, Phase 0)

- Ring names in the code: `input_tiles`, `gamma_tiles`, `scaler`, `row_stat`, `row_sum`, `output_tiles` (the `cb_` prefix dropped). Capacities exactly as the table, all derived in `rms_norm_program_artifacts.py` from `ROWS_PER_BLOCK`, `IN_DEPTH_ROWS`, `GAMMA_BLOCK_MAX_TILES`, `GAMMA_DEPTH_BLOCKS`, `OUT_DEPTH_TILES` and `C_eff`.
- A reserved gamma block's entry i is at `i · get_stride_size()` from the write pointer, not `i · entry_size`: a STRIDED ring interleaves the NEOs' entries in L1.
- `rows_per_block` is live: the kernels walk `grid(rows_per_block, Wt)` (DEST accumulation `PerRow`, which equals `WholeShape` on one row), the reduce and finalize take `rows_per_block` tiles, inv_rms is indexed by row. Tested at 2 (`test_rows_per_block_knob`), including a ragged last unit (a row past the tensor maps to its unit's first row and is stored again, same bytes).

## Verifier audit (Phase 0)

- Currency: the six rows match the six `DataflowBufferSpec`s of `rms_norm_program_artifacts.py`. The `row_stat` / `row_sum` capacities were stated as `1 · C_eff`; the code allocates `ROWS_PER_BLOCK · C_eff` (equal at the Phase 0 value 1, different once the knob turns) — corrected above, with the footprint line.
- The gamma and scaler ring formats are now one host constant each (`GAMMA_FORMAT`, `SCALER_FORMAT`); the reader receives the gamma entry geometry (`gamma_page_bytes`, `gamma_face_bytes`) as CT args instead of restating bf16 literals.
- Capacity vs live set, page format vs DEST width, disjoint lifetimes, bounds: no finding (see `verification_report.md`).
