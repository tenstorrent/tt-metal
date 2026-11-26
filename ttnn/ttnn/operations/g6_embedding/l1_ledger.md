# L1 Ledger: g6_embedding

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`
→ Blocking Model): **tokens** (knob `block_tile_rows`, tile rows of 32 tokens; images folded in),
**columns** (knob `col_block_tiles`), **table rows V** (not blocked: addressed by index value).
Depth knobs: `index_depth = rows_depth = tiles_depth = 2`. `fp32_dest_acc_en = False` (16-bit DEST).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_index` (128-B pages) | `index_depth · block_tile_rows` | `2 · block_tile_rows` pages: block `k`'s indices (read while its row reads are issued) and block `k+1`'s prefetch in flight | `{tokens: spans → block_tile_rows (one 32-index page per tile row); columns: streams (indices do not vary along columns; one chunk serves all col_block_tiles of the unit); V: none (values are page ids, not data)}` | UInt32, dataflow-only (never unpacked; no DEST relation) | reader | reader (private scratch: write pointer as a 2-slot ring, no push/pop) | whole reader kernel | **none** — live concurrently with `cb_rows` (the prefetch overlaps the gather by design) and page size differs (128 B vs 2048 B). Capacity = live set (depth 2 is the stall-shadow prefetch, not slack) |
| `cb_rows` (2048-B pages) | `rows_depth · block_tile_rows · col_block_tiles` | `2 · block_tile_rows · col_block_tiles` pages: one block landing from DRAM, one block being tilized | `{tokens: spans → 32 · block_tile_rows rows; columns: spans → col_block_tiles · 32 values per row; V: streams (one row slice per token, no row resident past its block)}` | **UInt16** — carries bf16 bits; Float16_b is forbidden (probe_001: engine tilize rewrites -0.0, subnormals, NaN). 16-bit page under 16-bit DEST: width decided, no audit-2 finding | reader | compute | whole kernel | **none** — cannot alias `cb_tiles`: tilize is out of place (`static_assert(input_dfb != output_dfb)`, `tilize_helpers.inl:89-90`) and both are live at once under cross-block pipelining. Capacity = live set; depth 2 is the reader/compute overlap |
| `cb_tiles` (2048-B pages) | `tiles_depth · col_block_tiles` | `2 · col_block_tiles` pages: one unit being packed, one unit being written | `{tokens: streams → 1 tile row per unit (does not scale with block_tile_rows); columns: spans → col_block_tiles; V: none}` | **UInt16** (same bits as the bf16 output tile; page size = bf16 tile page 2048 B) | compute | writer | whole kernel | **none** — concurrent with `cb_rows` (pipelining) and with itself across units; output pages must persist until `noc_async_writes_flushed()`. Capacity = live set; depth 2 is the compute/writer overlap |

Audit notes:

| Audit | Result |
|-------|--------|
| 1. capacity vs live set | every capacity equals its live set; the factor 2 in each is a stated overlap depth. Under-scaling: `cb_rows` spans tokens and columns and scales with both knobs; `cb_tiles` spans columns only and scales with `col_block_tiles` only; `cb_index` spans tokens only |
| 2. page format vs DEST width | DEST 16-bit; all compute-path pages are 16-bit (UInt16); `cb_index` is dataflow-only |
| 3. disjoint lifetimes | no pair has disjoint lifetimes: all three are live for the whole kernel and overlap block to block by design |
| 4. bounds | every capacity symbol is bounded below |
| Inventory | 3 CBs, the minimum for a pipelined gather → tilize → write: a staging buffer for rows (tilize cannot read DRAM), an output buffer for tiles (tilize is out of place), an index scratch (row addresses come from L1). No intermediate exists between phases; the tilize packs straight into the buffer the writer drains |

### Symbol table

| Symbol | Bound | Predicate / source that establishes it |
|--------|-------|----------------------------------------|
| `col_block_tiles` | `1 ≤ col_block_tiles ≤ min(Dt, MAX_COL_BLOCK_TILES = 64)`, divides `Dt` | host plan: `ct_cap` selection (`op_design.md` → Regimes); production max 40 |
| `block_tile_rows` | `1 ≤ block_tile_rows ≤ bt_l1 = floor((L1_BUDGET − 2048·col_block_tiles·tiles_depth) / (2048·col_block_tiles·rows_depth + 128·index_depth))` and `≤ ceil(core_units_max / min_blocks_per_core)` | host plan; `min_blocks_per_core = 4` |
| `index_depth`, `rows_depth`, `tiles_depth` | 2 each (Phase 0 constants) | host constants |
| `L1_BUDGET` | 1,347,776 B | BB-05 (free L1 per core when the call starts) |
| `Dt`, `T`, `N`, `B`, `S`, `V` | unbounded by this contract (`N` has no upper limit) | appear in **no** capacity expression; `Dt` enters only through `min(Dt, 64)` |

### Total per-core footprint

```
footprint = 128 · index_depth · block_tile_rows                                   (cb_index: scales with block_tile_rows)
          + 2048 · rows_depth  · block_tile_rows · col_block_tiles                (cb_rows:  scales with both knobs)
          + 2048 · tiles_depth · col_block_tiles                                  (cb_tiles: scales with col_block_tiles)
          ≤ L1_BUDGET = 1,347,776 B   (checked by the host plan; ValueError naming the bound otherwise)
```

| Case (C = 110) | regime | `col_block_tiles` | `block_tile_rows` | footprint (B) |
|----------------|--------|-------------------|-------------------|---------------|
| p1 `[16,4096]` × `[248320,1280]` | token_split | 40 | 5 | 984,320 (the INPUTS maximum) |
| p2 `[8,4096]` × `[248320,1280]` | token_split | 40 | 3 | 656,128 |
| p3 `[16,1024]` × `[248320,1280]` | token_split | 40 | 2 | 492,032 |
| p4 `[1,3844]` × `[248320,1280]` | token_split | 40 | 1 | 327,936 |
| `[16,4096]` × `[262144,64]` | token_split | 2 | 5 | 50,432 |
| `[16,128]` × `[248320,1280]` | token_column_split | 20 | 1 | 164,096 |
| FLUX `[1,512]` × `[131072,2560]` | token_column_split | 10 | 1 | 82,176 |
| FLUX `[1,1]` × `[32,5120]` | token_column_split | 1 | 1 | 8,448 |

The smallest legal plan (`col_block_tiles = block_tile_rows = 1`) is 8,448 B, so no `D % 32 == 0` shape can exceed the bound; the ValueError path exists for the contract, not for a reachable shape.

## Data-movement budget (chosen split)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` (idx) | `P` (= 1 under `token_split`): each tile row's ≤ 128-B chunk read once per column piece | the chunk lives in `cb_index` only for its block; under `token_column_split` each of the `P` cores holding a column piece reads it | none |
| `weight` (W) | the named rows only: `N` reads of `2·D` B (`token_split`) or `N·P` reads of `2·D/P` B (`token_column_split`) — `2·N·D` B either way; repeated indices re-read; **never a table pass** | a row slice is resident only until its block is tilized; dedup is rejected (≈ 0.03% of reads at p1) | none |
| output | 1: every tile written once as a whole 2048-B page | tiles leave `cb_tiles` straight to DRAM | none |

| Case | idx bytes (reads) | W bytes (reads) | out bytes (writes) | Total DRAM bytes |
|------|-------------------|-----------------|--------------------|------------------|
| p1 `[16,4096]` × D=1280, `token_split` | 262,144 (≤ 2,048) | 167,772,160 (65,536) | 167,772,160 (81,920) | 335,806,464 |
| p4 `[1,3844]` × D=1280, `token_split` | 15,376 (121, one bank) | 9,840,640 (3,844) | 9,912,320 (4,840) | 19,768,336 |
| `[1,1]` × D=1280, `token_column_split` (`P = 40`) | 160 (40 × 4 B) | 2,560 (40 × 64 B) | 81,920 (40) | 84,640 |

Per tier: DRAM as above; cross-core 0 B; core-local L1: each row byte written once by the NoC into `cb_rows`, read once by the unpacker, written once by the packer into `cb_tiles`, read once by the NoC.

> Cheapest-traffic split considered: tokens only (`P = 1`) — minimum bytes and minimum transactions
> (`N` row reads, idx read once). Implemented: tokens only whenever `T ≥ C`; tokens × columns when
> `T < C`, which adds `(P−1)·N` row transactions (no extra row bytes) and `≤ (P−1)·128` B of
> index reads per tile row — a deliberate trade for occupancy (spec rule 6), not a traffic win.
> The one remaining traffic alternative, `index_broadcast` (multicast the duplicated index chunks),
> is a `deferred` regime row: it saves only the duplicate 128-B chunk reads and the queueing on the
> single DRAM bank of a `B = 1` index page, and it slots into `cb_index` without changing the gather,
> tilize or store operations.
