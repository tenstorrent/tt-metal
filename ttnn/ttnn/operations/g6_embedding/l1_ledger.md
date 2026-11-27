# L1 Ledger: g6_embedding

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes and knobs:
`op_design.md` → Blocking Model. Named block axes: `tile_row` (extent `BLOCK_TILE_ROWS = 1`),
`col_tile` (extent `block_col_tiles`), `image` (folded into `tile_row`), `table_row` (gathered, no
extent; ≤ 32 rows per block = tile height). DEST is 16-bit (`fp32_dest_acc_en = False`).

## Inventory (Rule 1, before any solve)

| Phase boundary | Storage? | Why |
|----------------|----------|-----|
| idx DRAM → address formation | `cb_index_values` (256 B) | the reader must hold index values in L1 to load them into registers; a NoC read cannot target registers |
| gather → tilize | `cb_gathered_rows` | tilize reads row-major data from L1 through the unpacker; rows cannot be placed into tile order by the NoC at useful granularity (rejected `dm_face_permute`) |
| tilize → store | `cb_output_tiles` | the packer writes to L1; the writer moves pages from L1 to DRAM. The tilize helper cannot run in place (`tilize_helpers.inl:89-90`) and input and output have different byte orders |
| store → DRAM | none | pages go straight from `cb_output_tiles` |

Three buffers is the minimum for an engine relayout between a DRAM gather and a DRAM write. No
budget predicate, safety fraction or blocking search exists; the only L1-driven choice is the
`q_l1` clamp on `block_col_tiles`, which never binds for a production shape.

## Ledger

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_index_values` (128 B pages) | `INDEX_DEPTH · BLOCK_TILE_ROWS` = 2 (one slot = one block-row of indices) | 2 slots: indices of the tile-row being gathered + the prefetched next tile-row (in column_blocked: 1 slot used) | `{tile_row: streams → window INDEX_DEPTH, col_tile: not spanned (idx does not vary along D; one slot serves every column block of its tile-row), image: streams (folded), table_row: not spanned}` | UInt32 — dataflow-only payload, never enters DEST (audit 2 exemption) | reader | reader (RISC loads; ordered by the reader's read barrier) | whole reader kernel | cannot share `cb_gathered_rows`: concurrent (the prefetched slot is written while the current block's rows land) and a different page size/format |
| `cb_gathered_rows` (2048 B pages) | `ROWS_DEPTH · BLOCK_TILE_ROWS · block_col_tiles` = `2·block_col_tiles` | `2·block_col_tiles`: block k being tilized + block k+1 being gathered | `{tile_row: streams → window ROWS_DEPTH·BLOCK_TILE_ROWS, col_tile: spans → block_col_tiles, image: streams (folded), table_row: spans ≤ 32 gathered rows (the tile height, fixed)}` | UInt16 (bf16 bits) — 16-bit page under 16-bit DEST; UInt16 rather than Float16_b is the bit-exactness mechanism | reader | compute | whole kernel, block-pipelined | cannot alias `cb_output_tiles`: lifetimes overlap by design (tilize of block k runs while block k+1 is gathered and block k−1 is written); tilize is not in-place (`tilize_helpers.inl:89-90`). Capacity = live set (depth 2 is the overlap mechanism) |
| `cb_output_tiles` (2048 B pages) | `OUT_DEPTH · BLOCK_TILE_ROWS · block_col_tiles` = `2·block_col_tiles` | `2·block_col_tiles`: block k being packed + block k−1 being written | `{tile_row: streams → window OUT_DEPTH·BLOCK_TILE_ROWS, col_tile: spans → block_col_tiles, image: streams (folded), table_row: not present (rows already placed)}` | UInt16 — 16-bit page under 16-bit DEST; bytes are the output tensor's bf16 tiles | compute | writer | whole kernel, block-pipelined | cannot alias `cb_gathered_rows` (concurrent, above); it is not a resident shard, so packing into the output tensor directly is not available (output is DRAM interleaved) |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `block_col_tiles` | `1 ≤ block_col_tiles ≤ min(tensor_col_tiles, col_tile_cap)`; production max 40 | `select_plan`: `q_l1` clamp; `block_col_tiles = tensor_col_tiles / col_blocks_per_row` |
| `col_tile_cap` | 164 at Phase 0 depths | `floor((1_347_776 − INDEX_DEPTH·BLOCK_TILE_ROWS·128) / ((ROWS_DEPTH + OUT_DEPTH)·BLOCK_TILE_ROWS·2048))` |
| `INDEX_DEPTH`, `ROWS_DEPTH`, `OUT_DEPTH` | 2 each (constants) | module constants |
| `BLOCK_TILE_ROWS` | 1 (constant) | module constant; reason in `op_design.md` Axes |
| `tensor_col_tiles = D/32` | unbounded by the contract | appears in no capacity directly; only through the clamped `block_col_tiles` |
| `tensor_tile_rows`, `S`, `B`, `N`, `V` | unbounded | appear in no capacity (tokens stream; table rows are gathered) |

## Total per-core footprint

```
L1(core) = INDEX_DEPTH·BLOCK_TILE_ROWS·128 + (ROWS_DEPTH + OUT_DEPTH)·BLOCK_TILE_ROWS·block_col_tiles·2048
         = 256 + 8192·block_col_tiles            (Phase 0)
```

| Term | Scales with |
|------|-------------|
| `INDEX_DEPTH·BLOCK_TILE_ROWS·128` | `INDEX_DEPTH`, `BLOCK_TILE_ROWS` (256 B at Phase 0) |
| `ROWS_DEPTH·…·2048` | `ROWS_DEPTH`, `BLOCK_TILE_ROWS`, `block_col_tiles` |
| `OUT_DEPTH·…·2048` | `OUT_DEPTH`, `BLOCK_TILE_ROWS`, `block_col_tiles` |

Production maximum: `block_col_tiles = 40` → 327 936 B (24.3 % of 1 347 776). Clamp ceiling:
`block_col_tiles = 164` → 1 343 744 B. Floor (`block_col_tiles = 1`) → 8 448 B, so every `D` fits
and the BB-05 refusal in `validate()` is unreachable. All CB sizes are multiples of 64 B, so the
allocator's DRAM-alignment rounding adds nothing.

## Data-movement budget

Chosen split for the perf focus p1 (`[16,4096]` × `[248320,1280]`, regime `whole_row`):

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| idx (262 144 B) | 1 | each core reads only the 128-B segments of its own tile-rows; whole-row blocks need no second read | none |
| W (65 536 rows read, 167 772 160 B) | 1 per index (65 536 reads of 2 560 B) | whole rows: one read per token, the transaction floor; repeated indices are read again (dedup deferred) | none |
| out (81 920 tiles, 167 772 160 B) | 1 | each tile written once from `cb_output_tiles` | none |

Total p1: DRAM read 168 034 304 B, DRAM write 167 772 160 B, cross-core 0 B, core-local L1 →
L1: 0 (the relayout is unpack/pack inside the core).

Representative `column_blocked` call (`[1,77]` × `[248320,1280]`, 60 cores, `block_col_tiles = 2`):

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| idx (308 B) | 20 (each tile-row's segment read by its 20 column blocks) | idx does not vary along D; no multicast in Phase 0 | none |
| W (77 rows, 197 120 B) | 1 in bytes; 20 reads of 128 B per row (1 540 reads) | column blocks of one tile-row read disjoint segments of the same rows | none |
| out (120 tiles, 245 760 B) | 1 | — | none |

Total: DRAM read 203 280 B, write 245 760 B, cross-core 0 B.

> Cheapest-traffic split considered: `tile_row` only (`whole_row`) — it is the minimum on every
> tier for every shape (idx once, one read per index, out once, no cross-core bytes). Implemented:
> `whole_row` wherever `tensor_tile_rows` fills the grid (all of p1–p4), and `column_blocked` where it
> does not; `column_blocked` moves the same table and output bytes and adds only
> `(col_blocks_per_row − 1)·4·N` index bytes plus `(col_blocks_per_row − 1)·N` extra row
> transactions. Deferred: `index_page_multicast` (removes the duplicate idx reads: ≤ C requests of
> ≤ 128 B per call, for ≤ 16 KB of multicast per idx page) and `row_injector_scatter` (removes the
> extra row transactions for `2·D·N·(col_blocks_per_row−1)/col_blocks_per_row` B of core-to-core
> traffic), because both put a cross-core rendezvous on the serial chain of one-wave calls; the
> reasons and reachability are in the regime table of `op_design.md`.

## Implementation notes (deviations from the planner's inventory)

- `cb_index_values` holds `INDEX_DEPTH` slots of one **block-row** (`BLOCK_TILE_ROWS` tile-rows) each,
  so `BLOCK_TILE_ROWS > 1` stays a live knob (a block may then straddle images; each tile-row keeps
  its own image, `valid_rows` and index segment). Identical to the planner's 2 × 128 B at Phase 0.
- No CB was added, resized or merged. The two perf knobs below change NoC usage, not L1:
  - `WRITER_BANK_ROTATION` (on): `store_block` starts each core's tile-row at tile
    `core ordinal mod block_col_tiles`.
  - `read_noc_split` (plan field; `READ_NOC_SPLIT_MAX_SEGMENT_BYTES = 256`, `READ_NOC_SPLIT_MIN_CORES = 16`,
    token extent > 1): odd rows of each tile-row are read on NoC1, with both data-movement kernels in
    `DM_DYNAMIC_NOC` mode. Narrow-segment gathers are bound by the request rate of the DRAM endpoints
    per NoC (halving the grid barely moves them), so the second NoC adds request capacity. Measured on
    Blackhole, 110 cores: `[16,4096]`×64 146 → 101 µs, `[1,4096]`×64 11.7 → 8.9 µs, `[1,128]`×1280
    8.6 → 5.9 µs; on 2560-B segments it costs ~8% (p1 752 → 810 µs), hence the gate.
- Data-movement budget above is unchanged: same bytes and transactions per tensor; only the NoC a row
  read travels on differs.
