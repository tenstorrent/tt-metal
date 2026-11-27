# Operation Design: g6_embedding

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (index-driven row gather + row-major → tile layout change, no arithmetic) |
| Goal | Gather one table row per index on the device and write the rows as TILE pages of the result, every value bit-identical, in one device program. |
| Math | `out[i, d] = W[idx[i], d]`, `out.shape = idx.shape + [D]`, `out.dtype = W.dtype`; a `[1, V, D]` weight is the table `W[0]` |
| References | spec (task statement); `docs/explorations/g6_embedding/{DOSSIER,math,tensix,risks,lessons}.md`; `eval/golden_tests/g6_embedding/{feature_spec,helpers}.py`; probe `tests/ttnn/unit_tests/operations/g6_embedding/probes/probe_006.py` (tilize bit-exactness, all 65536 patterns); catalog `ttnn/ttnn/operations/examples/master.md` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes (positional) | uint32, rank 1 `[N]` or 2 `[B, S]`, DRAM interleaved, ROW_MAJOR (Phase 0) | — | buffer address: common RT arg; shape: host plan |
| `weight` | `ttnn.Tensor` | yes (positional) | bfloat16, ROW_MAJOR, `[V, D]` or `[1, V, D]`, `D % 32 == 0`, DRAM interleaved | — | buffer address: common RT arg; `D`: host plan |
| `layout` | `ttnn.Layout` | no (kw) | `ttnn.TILE_LAYOUT` only (axis `output_layout`) | `ttnn.TILE_LAYOUT` | host only |
| `memory_config` | `ttnn.MemoryConfig` | no (kw) | `None` or DRAM interleaved | `None` (= DRAM interleaved) | host only |
| `runtime_rows` | `bool` | no (kw) | `False` only (axis `runtime_rows`; `True` → `UnsupportedAxisValue`) | `False` | host only |

No `compute_kernel_config`: the op does no arithmetic. The compute kernel's config is fixed by this design (Key Risks, row 1).

## Tensors

### Input

| Property | `input_tensor` (idx) | `weight` (W) |
|----------|----------------------|--------------|
| Shape | `[B, S]` or `[N]` (token extent `S` / `N` ≥ 1, unbounded) | `[V, D]` or `[1, V, D]` (read as is: page id = row index, no reshape — BB-02) |
| Dtype | uint32 | bfloat16 |
| Layout | ROW_MAJOR (Phase 0): one page per row of `B`, `4·S` bytes (rank 1: one page of `4·N` bytes). TILE: refinement | ROW_MAJOR: one page per table row, `2·D` bytes |
| Memory | DRAM interleaved | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `idx.shape + [D]`: `[B, S, D]` or `[N, D]`; padded per image to `ceil(S/32)·32` rows |
| Dtype | dtype of `weight` (bfloat16) |
| Layout | TILE |
| Memory | `memory_config`, `None` → DRAM interleaved (the only placement of this contract) |

Rows of a partial last tile row of each image are padding: they hold stale L1 bytes (unspecified by contract), and nothing reads them.

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Everything below this section is its realization.

**The work unit** is one *(token tile row, column piece)* pair: 32 token slots of one image × `block_width_tiles` tile columns. **A block** is `block_height_tiles` consecutive units of one core's run (an ND rectangle `[block_height_tiles × 32 token slots] × [block_width_tiles × 32 columns]` of the output tile index space, folded along the unit order). Units are numbered **piece-major**: `unit = piece · tensor_tile_rows + tile_row`, so consecutive units of a core are consecutive tile rows of one piece (a run may cross into the next piece; every unit carries its own `(piece, tile_row)`).

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| token tile rows `tile_row ∈ [0, tensor_tile_rows)`, `tensor_tile_rows = B·ceil(S/32)` (B folded with the per-image S tiles; rank 1: `B = 1`, `S = N`) | **independent** — each output row reads only its own index and the row it names (math.md F1); tile rows of two images never share a tile | `block_height_tiles` (units per block) | `ceil(core_units_max / ceil(core_units_max / block_height_l1_cap))` — the fewest blocks the L1 bound allows, balanced (p1: 7; p2: 10; p3: 5 = whole assignment; D=64 `[16,4096]`: 19 = whole assignment) | `select_plan()` (host), one value per program | spread over the grid: `split_work_to_cores(grid, num_units, row_wise=True)` → contiguous unit runs; every used core runs the same kernels | knob-turn (block size, depth); remainder-balanced split is a `deferred` regime |
| token slots within a tile row (32 per tile row; valid `= min(32, S − 32·tile_in_image)`) | **independent** — tile-internal axis | fixed by the tile height (32) — not a knob: the unpacker tilizes 32 rows as one tile | 32 (valid count runtime-derived per unit) | tile geometry | never split: a tile is produced on one core | none |
| columns `d`, in tile columns `[0, tensor_d_tiles)`, `tensor_d_tiles = D/32` | **independent** — column `d` of the result copies column `d` of the table (math.md F2) | `block_width_tiles = tensor_d_tiles / piece_count` | from `select_plan()`: p1 40 (whole D), p2 20, p4 8, `[1,1]·D1280` 1, FLUX43 2 | `select_plan()` → `piece_count` (single source; `block_width_tiles` derived) | `piece_count` pieces, each unit owns one piece; pieces are spread across cores with the tile rows | knob-turn |
| table rows `v ∈ [0, V)` | **not an output axis — data-addressed** (`v = idx[i]`, values unknown until read on the device) | none — the rows a block touches are its ≤ `32·block_height_tiles` index values; no extent can be chosen on host (spec rule 1: no table pass, no host read of indices) | — | — | not assigned: each core reads exactly the rows its units' indices name | none (a V split is `rejected`, below) |
| (dependent axes) | **none** — no result spans any axis; no reduction, scan or combine | — | — | — | — | — |

Reuse-shared by construction of the split (operand-reuse check, per operand × chosen split):

| Operand | Varies along the token split? | Varies along the column split? | Consequence |
|---------|-------------------------------|--------------------------------|-------------|
| `input_tensor` | yes | **no** — every piece of a tile row needs the same 32 index values | at `piece_count > 1` the `piece_count` cores of one tile row re-read the same 128-B index slice → `deferred` regime "index broadcast" |
| `weight` | yes (data-dependent rows) | yes (each piece reads its own 64·`block_width_tiles` B slice of a row) | no duplication by the split; repeats of an index re-read a row → `deferred` regime "repeat-merged gather" |
| output | yes | yes | written once |

### Axes of stage data

| Stage data | Its axes | Blocked by | Core-assignment |
|------------|----------|------------|-----------------|
| staged rows (`cb_row_staging`) | token slots (32·`block_height_tiles`) × piece bytes (64·`block_width_tiles`) | the block (both axes spanned) | produced and consumed on the core that owns the units — no hand-off, no collecting core |
| tiles (`cb_out_tiles`) | tile rows (streamed, window 2) × `block_width_tiles` | one tile row at a time (the tilize helper's push quantum) | same core |
| index slice (`cb_index_slots`) | token slots (32·`block_height_tiles`) | the block | same core; no core computes for another |

No stage computes on one core while others wait: there is no coordinator, collector or combine stage.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_row_staging` | `ROWS_DEPTH_BLOCKS` | 2 (blocks) | reader gathers block k+1 while compute tilizes block k |
| `cb_out_tiles` | `TILES_DEPTH_ROWS` | 2 (tile rows) | compute packs tile row r+1 while the writer drains tile row r |
| `cb_index_slots` | `INDEX_SLOTS` | 1 (block) | none needed: block k's values are consumed (all row reads issued) before block k+1's index is fetched into the same slot |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Blackhole DRAM read: `local_addr % 64 == noc_addr % 64` (`.claude/references/noc-alignment-in-tt-metal.md:43`; `NOC_DRAM_READ_ALIGNMENT_BYTES 64`, `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`) | column slice offset/width; index slice offset; staging slot stride | slices are whole tile columns: offset `64·piece·block_width_tiles`, width `64·block_width_tiles`; staging slot stride `64·block_width_tiles`; index slice offset `128·tile_in_image`, slot stride 128; CB bases are DRAM-aligned by the allocator (`tt_metal/impl/program/program.cpp:2505-2509`) | silent data corruption (watcher: NoC sanitize error) — a sub-tile-column slice (32 B face row) is never issued |
| BB-05 L1 bound, 1,347,776 B per core | `block_width_tiles`, `block_height_tiles` | `block_width_tiles ≤ bw_cap = floor((1347776 − 128) / 8192) = 164` (candidate pieces only); `block_height_tiles ≤ block_height_l1_cap = floor((1347776 − 4096·bw) / (4096·bw + 128))` | refused launch inside the model (spec R10) |
| `compute_kernel_lib::tilize` `block_width_tiles` is a template (CT) parameter (`ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:194-209`) | `block_width_tiles` uniform per kernel group | `piece_count` must divide `tensor_d_tiles` (no ragged piece) | a ragged piece would need a second compute kernel |
| UInt16 CB format on the tilize path (bit-opaque; probe_006) | none on extent — on **format** | `cb_row_staging` and `cb_out_tiles` are declared `ttnn.uint16` (2048-B tile pages), never `bfloat16`; `ReconfigureRegisterDatatypeMode::NoReconfigure` | with `Float16_b` the tilize path flushes all 255 subnormal/−0.0 patterns and rewrites all 254 NaN patterns (probe_006: 509 of 65536 patterns changed) — wrong bits, random data never shows it |
| tile height 32; per-image tile rows | token slots of a unit | `valid = min(32, S − 32·tile_in_image)` row reads; slots ≥ `valid` are not read and never address the table | rows of the next image or index padding used as table addresses (spec rule 3, risk R2) |
| index format uint32 | V | page id = index value; contract guarantees `idx ∈ [0, V)`; no device clamp (D6) | out-of-contract input only |

### Regimes

Boundary for "minimum": DRAM. Minimum = `input_tensor` crosses once (`4N` B), each index names one row read once (`N` reads of `2D` B; never a table pass), the result crosses once as tile pages.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 gather → UInt16 tilize**, units = (tile row × column piece), `piece_count` from `select_plan()` | **built** | every valid call (the only built regime; `piece_count`, `block_height_tiles` are knob values chosen by `select_plan()`, below) | `block_height_tiles` units × `block_width_tiles` tile columns | `piece_count = 1`: minimum, except the result carries the tile padding rows (structural: TILE result; ≤ 31 rows per image, e.g. `[1,1]·D64` writes 4096 B for 128 logical B). `piece_count = P > 1`: index read P× (`+4N·(P−1)` B, P small reads per tile row on the index page's bank); table bytes unchanged in `P·N` transactions of `64·block_width_tiles` B; result unchanged. Cross-core: none | per block: one read-barrier drain of the core's NoC read stream, one `cb_row_staging` reserve+push, one compute block wait, ≤ one index read per index page touched. Fixed per kernel regardless of block: tilize init/uninit, hw startup, final write barrier. Per tile row regardless of block: `cb_out_tiles` handshake + one `writes_flushed` |
| R2 index broadcast per tile-row group (one core reads the 128-B slice, multicasts to the other `piece_count − 1` pieces) | **deferred** — the duplicated traffic it removes is ≤ 1/32 of a unit's transactions and ≤ 6.25 % of its bytes (at `block_width_tiles = 1`: 128 B index vs 2048 B rows), while it adds a semaphore round trip per block on the critical path of every receiver; spec rule 8 (RB-010) is advisory. Reachable: `fetch_index_block` is a separate block op writing `cb_index_slots`; R2 swaps its source (DRAM read → `ttnn.Mcast`/`McastArgs` receive) and adds a sender role, no other op changes | `piece_count > 1` | as R1 | index crosses DRAM once (`4N` B) instead of P×; adds `(P−1)·128` B NoC multicast per tile row | as R1 + one mcast handshake per block |
| R3 remainder-balanced split (whole-width tile rows for `floor(T/G)·G` units, the tail tile rows split by columns) | **deferred** — R1's cost model keeps the busiest core within 1.09× of perfect balance on every INPUTS shape whose result exceeds 1 MB (worst: p4 `[1,3844]·D1280`, 48 vs 44 tile-column·rows); R3 needs two unit widths per core (two CT `block_width_tiles`). Reachable: units are self-describing `(piece, tile_row)`; R3 is a second unit list per core | `tensor_tile_rows mod num_grid_cores ≠ 0` | two block widths per core | fewer row transactions than R1's P>1 (bulk stays whole-width); same bytes | as R1 |
| R4 repeat-merged gather (one DRAM read per distinct index in a block; repeats copied L1→L1) | **deferred** — the saving is data-dependent and zero on the suite's uniform indices (V ≥ 131072); costs a compare per token in the reader issue loop. Reachable: only `gather_rows_block`'s issue loop changes | a block holds repeated indices (device-side, data-dependent) | as R1 | table reads drop to distinct indices per block | as R1 |
| R5 table pass / table residency (tilize, convert or stage the whole table) | **rejected** — forbidden (spec rule 1: up to 671 MB per call for ≤ 168 MB needed); superseded by R1 | — | — | `2·V·D` B vs `2·N·D` B | — |
| R6 split V (table rows) across cores | **rejected** — V is data-addressed: every core would have to read every index (`G·4N` B) to find its rows, and no core's V range (≥ 6 MB at FLUX) fits L1, so nothing becomes resident; superseded by R1 (each core reads only its rows) | — | — | index ×G | — |
| R7 data-movement tilize (NoC L1→L1 copies of 32-B face rows, no compute kernel) | **rejected** — 64 local NoC transactions per output tile (5.24 M at p1) against zero for the unpacker; superseded by R1 once probe_006 showed the UInt16 tilize path bit-exact on all 65536 patterns | — | — | same DRAM bytes, +5.24 M local transactions (p1) | — |
| R8 direct DRAM → face-row reads (32-B row segments landing in tile position) | **rejected** — `src % 64 ≠ dst % 64` for every odd face row (BH DRAM read alignment, risk R4) and `2D/32` DRAM transactions per token; superseded by R1 | — | — | ×(D/32) row transactions | — |
| dependent-axis split + cross-core combine | **rejected** — no dependent axis exists | — | — | — | — |

### Traffic ranking

Qualitative, DRAM crossings per tensor first, then cross-core, then core-local.

| Rank | Candidate split | `input_tensor` | `weight` rows | result | Cross-core | Notes |
|------|-----------------|----------------|---------------|--------|------------|-------|
| 1 | token tile rows only (`piece_count = 1`) | 1× | 1× (N reads of 2D B) | 1× | none | **cheapest**; fills the grid only when `tensor_tile_rows ≥ num_grid_cores`; ragged when `tensor_tile_rows mod G` is small (p4: 121 rows on 110 cores → busiest core 2× average) |
| 2 | tile rows × column pieces (`piece_count = P`) | P× (128 B per unit) | 1× bytes, P× transactions | 1× | none | the available parallelism for one-tile-row and ragged shapes; R2 would remove the index duplication |
| 3 | column pieces only (`piece_count = tensor_d_tiles`, each core all tile rows of a piece) | `tensor_d_tiles`× | 1× bytes, `D/32`× transactions of 64 B | 1× | none | dominated by 2 |
| 4 | V split | G× | 1× | 1× | none | R6, rejected |
| — | dependent split + combine | — | — | — | — | does not exist |

Chosen: rank 1 whenever it does not lengthen the busiest core's critical path, rank 2 otherwise — the single host function `select_plan()` (Work Distribution) picks `piece_count`, with ties going to the smaller `piece_count` (fewer transactions, fewer index re-reads). Occupancy is the tiebreaker inside the cost model, not an override of the traffic order.

**Stall-shadow check.** The stage that waits is the reader's row-read barrier. Work independent of the awaited rows: the next block's index fetch. Licence: tokens are independent (math.md F1), so block k+1's index read commutes with block k's row reads; no floating-point order exists (no arithmetic), so no precision gate. Realized in `gather_rows_block` (issue block k+1's index read after the last row read of block k, before the barrier). The compute's wait (rows of block k) and the writer's wait (tile row r) have nothing independent to fill them.

### Block schedule

```cpp
// per core: units [unit_start, unit_start + core_units), num_blocks_this_core = ceil(core_units / block_height_tiles)
fetch_index_block(0);                                   // reader: the small reads that gate everything go first (RB-012)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    gather_rows_block(block_idx);    // reader: rows named by the block's indices → cb_row_staging; prefetches index of block_idx+1
    tilize_block(block_idx);         // compute: UInt16 tilize of block_height_tiles tile rows → cb_out_tiles
    store_tiles_block(block_idx);    // writer: tile rows → result tile pages in DRAM
}
// writer: one final write barrier (BB-03)
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `fetch_index_block(k)` | `block_height_tiles` × 128-B index slices (≤ 32 valid uint32 each) | `cb_index_slots` (1 slot) | ≤ one DRAM read per index page touched by the block (adjacent units of one image share a page → one read for the run); one barrier only for block 0 (later blocks ride block k−1's row barrier) |
| `gather_rows_block(k)` | `block_height_tiles` units × 32 slots × `64·block_width_tiles` B | the index slot of block k (consumed by the issue loop), the reserved staging block | one `cb_reserve_back(block_height_tiles·block_width_tiles)`; ≤ `32·block_height_tiles` row-slice reads (only valid slots of valid units); one `noc_async_read_barrier`; one `cb_push_back` of the **nominal** `block_height_tiles·block_width_tiles` pages, also for a ragged last block (unread units = phantom tile rows of stale bytes) |
| `tilize_block(k)` | `block_height_tiles` tile rows × `block_width_tiles` tiles | the staged block until each tile row is popped | one wait for the block; tilize LLK init once per kernel (`InitOnly`/`Neither`/`UninitOnly`, or `InitAndUninit` for a single block); no data-format reconfig (`NoReconfigure`); one output reserve/push and one input pop per tile row (helper quantum) |
| `store_tiles_block(k)` | `block_height_tiles` tile rows × `block_width_tiles` tiles | — | per tile row: one wait/pop of `block_width_tiles` pages, `block_width_tiles` tile writes (skipped for phantom tile rows), one `noc_async_writes_flushed`; one `noc_async_write_barrier` per kernel |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap | `block_height_tiles` is the coarsest the L1 bound allows: p3 and the D=64 `[16,4096]` case run their whole assignment as one block, so the reader's whole read precedes the first tilize; p1 uses 1.31 MB of L1 for 3 blocks | `block_height_tiles` at ~64–160 KiB per barrier (p1: 1–2 tile rows; D=64: 8–16) vs the L1 maximum, on p1–p4 |
| Grid synchronization | `UNIT_OVERHEAD_TILES = 1` spreads one-tile-row calls over up to 80 cores with 64-B row slices (`[1,1]·D1280`: 40 cores × 1 tile; FLUX43: 80 × 2), where launch and first-read latency may dominate | `UNIT_OVERHEAD_TILES ∈ {4, 8}` (fewer cores, wider slices) on `[1,1]`, `[1,8]`, `[1,16]` × D1280 and FLUX43, judged on blocking trace-replay host time (RB-053), not profiled kernel time |
| Barrier drain (trid) | one drain-everything barrier per block empties the core's read stream at each block boundary | transaction-id ring (`trid_double_issue`: `set_read_trid`, `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:147`) with blocks k and k+1 in flight, on p1 |
| Write bank stagger | Phase 0 rotates each core's tile-write start column by `core_linear_idx mod block_width_tiles` (`STAGGER_WRITES = True`); tile rows start at tile id `tile_row·tensor_d_tiles`, bank-aligned on BH for D1280 | `STAGGER_WRITES = False` (`bank_stagger`) on p1, p2 |
| Reader issue | D=64 tables are 128-B rows: 596 row reads per core at `[16,4096]`; one RISC issues them all | split row-read issue across NCRISC and BRISC (`split_reader`) — only if profiling shows the reader issue-bound |
| Wide one-tile-row index hotspot | B = 1 calls put every core's index read on one DRAM bank at kernel start (RB-010) | R2 (deferred row) — measured on `[1,4096]·D1280` (P = 5) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| index slice DRAM → L1 (`cb_index_slots`) | uint32, 128 B per unit (only `4·valid` B read) | `TensorAccessor::get_noc_addr(page = image b, offset = 128·tile_in_image)` + `noc_async_read` on NoC0 | rank 1: page 0. Reader-private scratch (no push/pop) |
| index values → row addresses | registers | RISC loads after the barrier (`noc_async_read_barrier` invalidates the L1 cache, `dataflow_api.h:1921-1934`) | only `valid` values are read; none past the logical shape addresses the table |
| table row slice DRAM → L1 (`cb_row_staging`) | bf16 bits (CB labelled UInt16), `64·block_width_tiles` B per token | `get_noc_addr(page = idx value, offset = 64·piece·block_width_tiles)` → slot `(unit_in_block·32 + slot)·64·block_width_tiles` | `[1, V, D]` weight: same page numbering (leading extent 1), no reshape |
| rows → tiles | UInt16 RM rows → UInt16 tiles (bit-opaque) | `compute_kernel_lib::tilize` (symmetric pages, standard path) | unpacker → SrcA as Integer16 → MOVA2D with the Src zero flag disabled → 16-bit DEST → packer; probe_006: 0 of 65536 patterns changed |
| tiles L1 → DRAM | tile pages, 2048 B | `noc_async_write` on NoC1, tile id `tile_row·tensor_d_tiles + piece·block_width_tiles + c` | whole-page writes (`tile_reorder`); write start column rotated (`STAGGER_WRITES`) |

No Tensix-to-Tensix traffic in the built regime. R2's contract, kept reachable: per tile-row group of `piece_count` cores, one sender reads the 128-B slice and multicasts it into the receivers' `cb_index_slots` at the same slot offset, one semaphore per block, via `ttnn.Mcast` (host) / `McastArgs` (`ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp`); no other block op changes.

TARGET axes seen against the scheme (`feature_spec.py` names):

| TARGET axis value | Phase 0 | Class | What changes |
|-------------------|---------|-------|--------------|
| `dtype` bfloat16 | built | — | — |
| `layout` ROW_MAJOR (index) | built | — | — |
| `layout` TILE (index) | refinement | **knob-turn** | `fetch_index_block` address math only: unit `(b, tile_in_image)` reads index tile `(b/32)·ceil(S/32) + tile_in_image` (rank 1: `b = 0`), two 64-B segments — face `2·((b%32)/16)` and the next face, at offset `(b%16)·64` in each 1024-B face — into the same 128-B slot. Both 64-B aligned. The `valid` clamp keeps every padding position (the golden poisoned-padding case) away from the table |
| `alignment` `tile_aligned`, `h_non_aligned` | built (both) | — | `valid = min(32, S − 32·tile_in_image)` per unit |
| `output_layout` TILE | built | — | — |
| `runtime_rows` False | built | — | `True` is outside TARGET: refused |

No tensor of this contract is sharded; no placement axis exists. **Structural impossibilities:** none beyond `INVALID = []` (agreed).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | `(tile_row, piece)`; `num_units = tensor_tile_rows · piece_count` |
| Grid | `device.compute_with_storage_grid_size()` (110 cores on the campaign boards: 11 × 10) — part of the cache key, never a literal |
| Per-core work | `ttnn.split_work_to_cores(grid, num_units, row_wise=True)` → `core_units ∈ {units_g1, units_g2}` contiguous in piece-major unit order; `num_blocks_this_core = ceil(core_units / block_height_tiles)` |
| Remainder | `tiles_per_image = ceil(S/32)` per image, `tensor_tile_rows = B·tiles_per_image`; ragged units within a tile row via `valid`; ragged last block via nominal push + phantom tile rows (never written) |

**`select_plan(index_shape, weight_shape, grid_size)` — pinned, host-only, deterministic (BB-08), and the single source of every knob:**

```
S = index_shape[-1];  B = index_shape[0] if len(index_shape) == 2 else 1
D = weight_shape[-1]; tensor_d_tiles = D // 32
tiles_per_image = ceil(S / 32);  tensor_tile_rows = B * tiles_per_image
G = grid_size.x * grid_size.y
bw_cap = (L1_BOUND_BYTES - INDEX_ROW_BYTES) // (2 * ROWS_DEPTH_BLOCKS * TILE_BYTES)        # 164; equals the bh = 1 fit
candidates = [P for P in divisors(tensor_d_tiles) if tensor_d_tiles // P <= bw_cap]
cost(P) = ceil(tensor_tile_rows * P / G) * (tensor_d_tiles // P + UNIT_OVERHEAD_TILES)
piece_count = min(candidates, key=lambda P: (cost(P), P))
block_width_tiles = tensor_d_tiles // piece_count
num_units = tensor_tile_rows * piece_count
num_cores, all_cores, g1, g2, units_g1, units_g2 = ttnn.split_work_to_cores(grid, num_units, row_wise=True)
core_units_max = max(units_g1, units_g2)
block_height_l1_cap = (L1_BOUND_BYTES - TILES_DEPTH_ROWS * block_width_tiles * TILE_BYTES) \
                      // (ROWS_DEPTH_BLOCKS * block_width_tiles * TILE_BYTES + INDEX_SLOTS * INDEX_ROW_BYTES)
num_blocks_max = ceil(core_units_max / block_height_l1_cap)
block_height_tiles = ceil(core_units_max / num_blocks_max)
l1_bytes_per_core = TILE_BYTES * block_width_tiles * (ROWS_DEPTH_BLOCKS * block_height_tiles + TILES_DEPTH_ROWS) \
                    + INDEX_SLOTS * INDEX_ROW_BYTES * block_height_tiles
```

Host constants (one definition each): `L1_BOUND_BYTES = 1347776`, `TILE_BYTES = 2048`, `INDEX_ROW_BYTES = 128`, `ROWS_DEPTH_BLOCKS = 2`, `TILES_DEPTH_ROWS = 2`, `INDEX_SLOTS = 1`, `UNIT_OVERHEAD_TILES = 1`, `STAGGER_WRITES = True`. The 2048 in `TILE_BYTES` is `ttnn.tile_size(ttnn.uint16)`.

Resulting plans (grid 110; computed with the function above for every distinct INPUTS pair):

| Case | `tensor_tile_rows` × `tensor_d_tiles` | `piece_count` / `block_width_tiles` | cores | units/core | `block_height_tiles` × blocks | L1 B/core | busiest / ideal |
|------|------|------|------|------|------|------|------|
| p1 `[16,4096]·D1280` | 2048 × 40 | 1 / 40 | 110 | 19 | 7 × 3 | 1,311,616 | 1.02 |
| p2 `[8,4096]·D1280` | 1024 × 40 | 2 / 20 | 110 | 19 | 10 × 2 | 902,400 | 1.02 |
| p3 `[16,1024]·D1280` | 512 × 40 | 1 / 40 | 110 | 5 | 5 × 1 | 983,680 | 1.07 |
| p4 `[1,3844]·D1280` | 121 × 40 | 5 / 8 | 110 | 6 | 6 × 1 | 230,144 | 1.09 |
| `[1,4096]·D1280` (112 calls) | 128 × 40 | 5 / 8 | 110 | 6 | 6 × 1 | 230,144 | 1.03 |
| `[16,4096]·D64` | 2048 × 2 | 1 / 2 | 110 | 19 | 19 × 1 | 166,272 | 1.02 |
| `[1,1]·D1280`, `[1,8]`, `[1,16]`, `[1,31]` | 1 × 40 | 40 / 1 | 40 | 1 | 1 × 1 | 8,320 | 1.0 |
| FLUX41 `[1,512]·D2560` | 16 × 80 | 20 / 4 | 110 | 3 | 3 × 1 | 65,920 | 1.03 |
| FLUX43 `[1,1]·[32,5120]` | 1 × 160 | 80 / 2 | 80 | 1 | 1 × 1 | 16,512 | 1.0 |
| FLUX44 `[1,1]·[1,256,128]` | 1 × 4 | 4 / 1 | 4 | 1 | 1 × 1 | 8,320 | 1.0 |

Core ranges: `split_work_to_cores(..., row_wise=True)` yields full grid rows plus at most one partial row → ≤ 2 rectangles per kernel group (BB-17); every used core runs the same three kernels (one kernel group per RISC).

Only one compute regime is built, so there is no regime-selection predicate; `select_plan()` is still a plan choice by shape and grid (BB-08): the entry point calls exactly this function, it is exported, and the acceptance test pins its invariants (L1 bound, `piece_count · block_width_tiles = tensor_d_tiles`, full grid when `tensor_tile_rows ≥ G`, more than one core for wide one-tile-row results).

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_row_staging` | 0 | 2048 B (`TILE_BYTES`) | `ROWS_DEPTH_BLOCKS · block_height_tiles · block_width_tiles` | spans token tile rows (`block_height_tiles`, ×2 depth) and columns (`block_width_tiles`); streams over table rows | `ttnn.uint16` (bit-opaque bf16) | reader | compute | whole kernel; one block per reserve/push |
| `cb_index_slots` | 1 | 128 B (`INDEX_ROW_BYTES`: 32 × uint32) | `INDEX_SLOTS · block_height_tiles` | spans token tile rows of one block; columns: not spanned | `ttnn.uint32` | reader | reader (private scratch: no push/pop) | whole kernel; reused per block |
| `cb_out_tiles` | 2 | 2048 B (`TILE_BYTES`) | `TILES_DEPTH_ROWS · block_width_tiles` | spans columns; streams over token tile rows (window 2) | `ttnn.uint16` (bit-opaque bf16) | compute | writer | whole kernel; one tile row per push/pop |

Full ledger: `l1_ledger.md`. CB sync: `cb_row_staging` push `block_height_tiles·block_width_tiles` per block = compute wait per block, compute pops `block_width_tiles` per tile row (`block_height_tiles` pops per block); `cb_out_tiles` push `block_width_tiles` per tile row = writer wait/pop `block_width_tiles` per tile row. Ring closure: each capacity is a multiple of its push quantum.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `fetch_index_block(k)` | `block_height_tiles` × 128 B | no — raw `TensorAccessor` + `noc_async_read` (see API Mapping) | — (DRAM `input_tensor`) | `cb_index_slots`, `block_height_tiles` pages (written in place, no push) | slot holds block k's values after block k−1's row barrier (block 0: after its own barrier) |
| 2 | `gather_rows_block(k)` | `block_height_tiles` × 32 × `64·block_width_tiles` B | no — index-driven reads (see API Mapping) | `cb_index_slots` (read by RISC loads) + DRAM `weight` | `cb_row_staging`, `block_height_tiles·block_width_tiles` pages (nominal) | pushed; index slot refilled with block k+1 |
| 3 | `tilize_block(k)` | `block_height_tiles` tile rows × `block_width_tiles` | yes — `compute_kernel_lib::tilize` | `cb_row_staging`, waits `block_height_tiles·block_width_tiles`, pops `block_width_tiles` per tile row | `cb_out_tiles`, `block_width_tiles` per tile row | staged block fully popped |
| 4 | `store_tiles_block(k)` | `block_height_tiles` tile rows × `block_width_tiles` | no — raw `noc_async_write` (see API Mapping) | `cb_out_tiles`, `block_width_tiles` per tile row | — (DRAM result) | popped; writes flushed per tile row; final barrier |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| startup | helper prerequisite | `compute_kernel_hw_startup(icb, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:113` | `(cb_row_staging, cb_out_tiles)` — establishes the UInt16 formats; the math default zero-flag rule disables MOVA2D's denormal flush for UInt16 sources (`tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_defs.h:40-43`, `.../common/inc/cmath_common.h:99-107`, applied at `.../llk_lib/llk_math_common.h:59`) | — | — | none |
| `tilize_block` | helper | `compute_kernel_lib::tilize<block_width_tiles, cb_row_staging, cb_out_tiles, InitUninitMode, WaitMode::NoWait, ReconfigureRegisterDatatypeMode::NoReconfigure>(in, out, block_height_tiles)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:194-209`; loop `tilize_helpers.inl:214-239`; fast path excluded for UInt16 at `tilize_helpers.inl:66-69` (standard `tilize_block`, `:230`) | symmetric pages (no `total_input_pages`); `InitUninitMode` `InitOnly`/`Neither`/`UninitOnly` across blocks (`InitAndUninit` when one block); explicit zoned `cb_wait_front(cb_row_staging, block_height_tiles·block_width_tiles)` before the call | `cb_row_staging` | `cb_out_tiles` | `block_width_tiles` (CT template), `num_blocks = block_height_tiles` (RT) |
| `fetch_index_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)`; `noc_async_read`; `noc_async_read_barrier` (block 0) | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165`; `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547`, `:1921` | page = image `b`, offset `128·tile_in_image`, size `4·valid` (merge adjacent units of one page into one read) | — | `cb_index_slots` | `block_height_tiles` (slices per block) |
| `gather_rows_block` | raw_api | `cb_reserve_back` / `get_write_ptr` / `cb_push_back`; `TensorAccessor::get_noc_addr(page_id, offset)`; `noc_async_read`; `noc_async_read_barrier` | `tensor_accessor.h:165`; `dataflow_api.h:547`, `:1921` | page = index value, offset `64·piece·block_width_tiles`, size `64·block_width_tiles`; one barrier per block | `cb_index_slots` | `cb_row_staging` | `block_height_tiles`, `block_width_tiles` |
| `store_tiles_block` | raw_api | `cb_wait_front` / `cb_pop_front`; `TensorAccessor::get_noc_addr(tile_id)`; `noc_async_write`; `noc_async_writes_flushed`; `noc_async_write_barrier` | `tensor_accessor.h:165`; `dataflow_api.h:876`, `:1980`, `:1951` | tile id `tile_row·tensor_d_tiles + piece·block_width_tiles + c`, `c` rotated by `core_linear_idx` when `STAGGER_WRITES` | `cb_out_tiles` | — | `block_width_tiles` |
| zones (BB-21) | raw_api | `DeviceZoneScopedN(name)` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012` (no-op build: `:1047`) | names in Host Contract | — | — | none |

**Helpers considered and rejected**

| Raw entry | Candidate helper | File:line of the mismatch | Concrete reason |
|-----------|------------------|---------------------------|-----------------|
| `gather_rows_block` | `dataflow_kernel_lib::read_sticks_for_tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.inl:121` (`accessor.get_noc_addr(start_page + block_row + row, …)`) | reads consecutive pages `start_page + row`; the gather needs the page id of every row taken from a device-read index value — the helper has no page-list or per-row page parameter. Missing block op: "read sticks for tilize from a page-id list", the same loop with `page_ids[row]` |
| `fetch_index_block` | `read_sticks_for_tilize` | same line | reads whole sticks into tile-sized CB pages and pushes them; the index slice is a 128-B sub-range of a stick into reader-private scratch with no push |
| `fetch_index_block` | `generate_index_tile` (`index_tile_dataflow.hpp`) | `ttnn/cpp/ttnn/kernel_lib/index_tile_dataflow.hpp` (generator for sort ops) | generates position tiles; it reads no tensor |
| `store_tiles_block` | `dataflow_kernel_lib::write_sticks_after_untilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:129-135` | writes row-major sticks after untilize; the result here is whole tile pages |
| (rows → tiles with `Float16_b` CBs) | `compute_kernel_lib::tilize` with bf16 formats | `tilize_helpers.inl:66-69` selects fast tilize for `Float16_b`; MOVA2D flushes datums with zero exponent unless the Src zero flag is disabled (`ckernel_defs.h:19-31`), which the default rule does only for UInt16 | probe_006: bf16 formats change 509 of 65536 patterns (subnormals, −0.0 → +0, every NaN); the helper is kept, the formats are UInt16 |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Special values through the tilize path (spec rule 2, R3) | bf16 CB formats flush subnormals, −0.0 and rewrite NaN (probe_006: 509/65536) — random data never shows it | `cb_row_staging`, `cb_out_tiles` are `ttnn.uint16`; `compute_kernel_hw_startup` with those CBs; `NoReconfigure`; compute config `fp32_dest_acc_en = False`, `dst_full_sync_en = False`, unpack-to-dest mode default (matches the probe's `ttnn.tilize` uint16 config, `tilize_multi_core_default_program_factory.cpp:36-38,146-149`). Golden special-rows cases and the acceptance test grade all patterns |
| Probe width coverage | probe_006 tilized `block_width_tiles = 32`; plans use 1–80 | acceptance special-values cases at `block_width_tiles` 1, 2 and 40 (`[1,128]·D64`, `[1,128]·D1280`, `[16,1024]·D1280`) |
| Index positions past the logical shape (rule 3, R2) | the row of a partial tile row of an image continues into the next image's indices or into page padding | `valid = min(32, S − 32·tile_in_image)` reads only; slots ≥ `valid` keep stale bytes and are never dereferenced; same clamp in the TILE-index refinement |
| Ragged last block | units per core is rarely a multiple of `block_height_tiles` | nominal push/wait counts (`block_height_tiles·block_width_tiles`); compute tilizes phantom tile rows of stale bytes; the writer pops them without writing (blocking-model §2, ragged tails) |
| Per-image tile geometry | `B > 1` with `S % 32 ≠ 0` (legal, not in production) | `tile_row = b·tiles_per_image + tile_in_image`, `tiles_per_image = ceil(S/32)`; never `floor(B·S/32)` |
| Index values read too early | RISC loads of `cb_index_slots` must follow the read that filled it | block 0: own barrier; block k+1: its read is issued before block k's barrier and completes with it; the barrier invalidates the L1 cache (`dataflow_api.h:1932`) |
| Prefetch overwrites live index values | `INDEX_SLOTS = 1` | block k+1's index read is issued only after block k's last row read has been issued (all values already loaded into registers) |
| DRAM read alignment (R4) | every row/index read lands at an address ≡ its source mod 64 only by construction | whole-tile-column slices and 128-B index slots (Mechanism caps); kernel `ASSERT((get_write_ptr(cb_row_staging) & 63) == 0)` under `--dev` |
| Stale index on replay (rule 1, R7, BB-16) | the call is replayed from a trace with new index values | no index value, row id or index-derived address is a kernel arg; the plan depends on shapes only; indices are read on the device every call |
| L1 bound (BB-05, R10) | p1 and p2 use 1.31 MB / 0.90 MB of the 1,347,776 B | `select_plan()` clamps both extents; `l1_bytes_per_core > L1_BOUND_BYTES` → `ValueError` naming the bound (unreachable for valid inputs, kept as the contract's refusal) |
| Unfinished NoC traffic at exit (BB-03) | another program starts right after | reader: every block ends in a read barrier; writer: final `noc_async_write_barrier` |
| Output padding rows | hold stale L1 bytes, can differ between runs | outside the logical result (contract: unspecified); BB-06 holds on the logical values |

## Host Contract

| Item | Decision |
|------|----------|
| Registry | `INPUT_TAGGERS = {"alignment": tag_alignment, "layout": tag_layout}`; `tag_alignment` reads `inputs[0]["index_shape"][-1] % 32` → `"tile_aligned"` / `"h_non_aligned"`; `tag_layout` maps `inputs[0]["index_layout"]` `"ROW_MAJOR"`/`"TILE"` → `ttnn.ROW_MAJOR_LAYOUT`/`ttnn.TILE_LAYOUT`. `validate()` builds `(scenario,)` with keys `index_shape`, `weight_shape`, `index_layout`, `memory_config` (`"dram_interleaved"` or `None`, as `axes.py` does) and axes `dtype = weight.dtype`, `output_layout = layout`, `runtime_rows = bool(runtime_rows)`; Phase 0 SUPPORTED per the spec (`layout` ROW_MAJOR only), `EXCLUSIONS = []` |
| Validation order | `validate()` first line (SUPPORTED → EXCLUSIONS, typed refusals); then `ValueError` for: index dtype ≠ uint32, index rank ∉ {1, 2}, weight rank ∉ {2, 3}, rank-3 leading extent ≠ 1, weight not ROW_MAJOR, `D % 32 ≠ 0`, a tensor not on device / not DRAM interleaved, `memory_config` not DRAM interleaved, `plan.l1_bytes_per_core > 1347776` (message names the bound). Each refusal message names what it rejects, and the acceptance test matches these fragments: `runtime_rows` (axis refusal), `weight` (weight rank / leading extent / layout), `32` (`D % 32`), `uint32` (index dtype), `rank` (index or weight rank), `memory_config` (placement), `1347776` (L1 bound). No index value is read on host |
| Dispatch | output `ttnn.allocate_tensor_on_device(idx.shape + [D], weight.dtype, TILE, device, DRAM interleaved)`; `ttnn.generic_op([input_tensor, weight, output], descriptor)` — one program per call, own kernels only |
| Program caching (BB-07) | module-level cache keyed by `(index shape, index layout, index dtype, weight shape, weight dtype, weight layout, input/weight/output memory configs, output layout, runtime_rows, device id, grid size)` → `(plan, ProgramDescriptor)`. A hit only re-points the three buffer addresses (index, weight, output), carried as **common runtime args** (`KernelDescriptor.common_runtime_args`, `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:775`); per-core args (`unit_start`, `core_units`, `num_blocks_this_core`, `core_linear_idx`) are plan-static. The device program cache keys on the descriptor hash, which excludes runtime-arg values (`ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:78-101`) |
| Plan report (BB-08) | `select_plan` and its frozen result `G6EmbeddingPlan` are exported from `ttnn.operations.g6_embedding`; fields at least `piece_count`, `block_width_tiles`, `block_height_tiles`, `num_cores`, `core_ranges`, `units_per_core` (g1, g2), `l1_bytes_per_core`, `regime` (`"gather_tilize"`), `roles` (core → `"gather"`). `select_plan(index_shape, weight_shape, grid_size)` accepts shape tuples and a grid `CoreCoord` or `(x, y)` |
| Zones (BB-21) | reader: `IDX-FETCH`, `IDX-WAIT` (block-0 barrier), `STAGE-RESERVE`, `ROW-ISSUE`, `ROW-WAIT` (barrier + push); compute: `STAGE-WAIT`, `TILIZE` (the helper; its per-tile-row output reserve wait falls inside it); writer: `TILE-WAIT`, `TILE-WRITE`, `WRITE-FLUSH`, `WRITE-BARRIER`. Every core's role comes from `plan.roles`. `DeviceZoneScopedN` compiles to nothing in the plain build |
| Kernel size (BB-04, BB-18) | default optimization level; no per-token division (units walked incrementally, `(piece, tile_row, b, tile_in_image)` advanced by counters) |

## Catalog entries consulted

| Entry | Knob it informed |
|-------|------------------|
| `width_split` | column pieces for one-tile-row results (`piece_count`) |
| `double_buffer` | issue a block of reads, one barrier; `ROWS_DEPTH_BLOCKS = TILES_DEPTH_ROWS = 2` |
| `trid_double_issue` | "batch barriers first" → `block_height_tiles` at the L1 maximum; trid ring kept as a lamp |
| `noc_placement` | `row_wise=True` core order; reader NoC0, writer NoC1 (defaults) |
| `bank_stagger` | `STAGGER_WRITES` rotation of the writer's start column |
| `tile_reorder` | whole 2-KB tile writes, no face scatter (R7, R8 rejected) |
| `compute_block_size` | tilize init once per kernel, `NoReconfigure` (formats never change) |
| `split_reader` | lamp for D=64 tables only |
