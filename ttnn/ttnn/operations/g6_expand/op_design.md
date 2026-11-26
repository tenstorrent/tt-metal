# Operation Design: g6_expand

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no arithmetic) |
| Goal | Materialize `x` (d0, d1, 1, W) broadcast along dim 2 into a new tensor `out` of shape `sizes` (-1 → d of x), bit-exact, in **one** device program (`ttnn.generic_op` with one ProgramDescriptor). Production: x (1,1,1,W) → out (1,1,W,W), W ∈ {185, 196, 207, 256, 512}, bfloat16, TILE, DRAM interleaved. |
| Math | `out[n0, n1, h, w] = x[n0, n1, 0, w]` (dims 0, 1, 3 keep their size; dim 2 of x is 1). In tile terms: out tile (n, h, w) = **B(n, w)**, where B(n, w) is x tile (n, w) with its row 0 copied into all 32 rows. Every out tile row of an image is the same row of tiles. |
| References | `docs/explorations/g6_expand/{DOSSIER,math,tensix,risks}.md`; `eval/golden_tests/g6_expand/{feature_spec,helpers}.py`; `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp`; catalog `ttnn/ttnn/operations/examples/{noc_placement,width_split,tile_reorder,double_buffer,split_reader,mcast_topology}` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 4, bfloat16, TILE, DRAM interleaved, on device, `shape[2] == 1` | — | buffer address = RT |
| `sizes` | list / tuple of 4 ints | yes (positional) | `len == 4`; each entry `-1` or `> 0`; `sizes[d] == shape[d]` or `-1` for d ∈ {0, 1, 3}; `sizes[2] ≥ 1` or `-1` | — | host only (sets out shape → geometry args below) |

No `memory_config`, no output dtype, no `compute_kernel_config`: out takes dtype, layout and memory config of `input_tensor`.

### Validation (Python, in `validate()` — first line of `g6_expand`)

| Check | Raises |
|-------|--------|
| `input_tensor` not on device (`ttnn.is_tensor_storage_on_device` false) | `ValueError` |
| rank ≠ 4, or `len(sizes) ≠ rank` | `ValueError` |
| any `sizes[d]` == 0, or `< 0` and `≠ -1` | `ValueError` |
| `sizes[d] ∉ {-1, shape[d]}` where `shape[d] ≠ 1` | `ValueError` |
| `shape[d] == 1` and `sizes[d] ∉ {-1, 1}` for d ∈ {0, 1, 3} (expansion of a dim other than 2) | `ValueError` |
| memory config not DRAM interleaved | `ValueError` |
| SUPPORTED per axis (`dtype`, `layout`, `alignment` via `tag_alignment`) | `UnsupportedAxisValue` |
| EXCLUSIONS (empty in Phase 0) | `ExcludedCell` |

Order: structural `ValueError` checks first (they make the tagger's inputs well-defined), then the SUPPORTED / EXCLUSIONS gate. `tag_alignment` reads the **output** shape (sizes with -1 replaced): `"w_non_aligned"` if `W_out % 32` (also when H_out is not aligned), else `"h_non_aligned"` if `H_out % 32`, else `"tile_aligned"`. Its `inputs` argument is `(scenario,)` with `scenario = {"shape": list(input_tensor.shape), "sizes": list(sizes)}`.

Phase 0 SUPPORTED: `dtype: [bfloat16]`, `layout: [TILE_LAYOUT]`, `alignment: ["tile_aligned", "w_non_aligned"]` (= TARGET). EXCLUSIONS: none. Structural impossibilities: none beyond feature_spec's `INVALID = []`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | (d0, d1, 1, W); production (1, 1, 1, W) |
| Dtype | bfloat16 |
| Layout | TILE (32×32, four 16×16 faces; row 0 = row 0 of faces 0 and 1) |
| Memory | DRAM interleaved |
| Padding | rows 1..31 of every tile and columns ≥ W of the last tile are **arbitrary** (never read into a logical element of out) |

### Output

| Property | Value |
|----------|-------|
| Shape | `sizes` with each -1 replaced by `input_tensor.shape[d]`; production (1, 1, W, W) |
| Dtype | dtype of input (bfloat16) |
| Layout | layout of input (TILE) |
| Memory | memory config of input (DRAM interleaved); a freshly allocated tensor (`ttnn.allocate_tensor_on_device`), never a view |
| Padding | rows ≥ H_out of the last tile row and columns ≥ W of the last tile column hold whatever B carries (x's padding columns, replicated row 0) — allowed, padding of out is not part of the result |

### Geometry (host, alignment-aware, per image)

| Symbol | Formula | W=512 | 256 | 207 | 196 | 185 |
|--------|---------|-------|-----|-----|-----|-----|
| `tensor_images` | `d0 · d1` | 1 | 1 | 1 | 1 | 1 |
| `tensor_tile_cols` | `ceil(W / 32)` | 16 | 8 | 7 | 7 | 6 |
| `tensor_out_tile_rows` | `ceil(H_out / 32)` (per image) | 16 | 8 | 7 | 7 | 6 |
| x page of (n, w) | `n · tensor_tile_cols + w` (x has exactly one tile row per image: `ceil(1/32) = 1`) | | | | | |
| out page of (n, h, w) | `(n · tensor_out_tile_rows + h) · tensor_tile_cols + w` | | | | | |
| `tile_bytes` | `input_tensor.buffer_page_size()` (2048) | | | | | |

Never `floor(N·H/32)` and never `W // 32`.

## Blocking Model

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| image `n` (dims 0·1 of out, folded) | independent — each image reads only its own x row, no result spans images | `block_images` | 1 | host constant `BLOCK_IMAGES = 1` in program-descriptor module | images are an outer factor of the block-id space; blocks of different images go to different cores | knob-turn |
| out tile row `h` (dim 2, the expanded dim) | independent **and** x is reuse-shared along it — no result spans rows; every tile row of an image needs the same B(n, w) | `block_tile_rows` | `ceil(tensor_out_tile_rows / row_groups)` (production: 3 / 2 / 2 / 2 / 2 on a 110-core grid, see Work Distribution) | host-derived from the knob `min_block_tile_rows` (host constant, Phase 0 = 2) and the grid size; passed to kernels as RT arg | `row_groups` row-groups per image, one block each | knob-turn (`min_block_tile_rows`); reuse via mcast = scheme-change (deferred regime R2) |
| out tile column `w` (dim 3) | independent — out column w reads only x tile column w | `block_tile_cols` | `min(ceil(tensor_tile_cols / num_cores), max_block_tile_cols)` = 1 in every production case | host-derived; `max_block_tile_cols` (L1 clamp) is a host constant expression (Mechanism caps); passed as RT arg | `col_groups = ceil(tensor_tile_cols / block_tile_cols)` column groups, assigned **first** (it is the split that adds no x re-read) | knob-turn |
| x row (dim 2 of x, size 1) | reuse-shared — the only source of every out row | none (extent 1 by the tensor: x has one logical row, one tile row) | 1 | tensor shape | not split (nothing to split) | — |
| row inside a tile (32 rows, faces) | not a block axis — the replication target inside one B tile | none (a tile is the atomic page) | 32 (whole tile) | tile shape | not split | — |

Why `block_tile_cols = 1` in production: occupancy first — `tensor_tile_cols` ≤ 16 is far below the grid, so every column gets its own column group; a core's assignment along w *is* one column, and the block equals the whole assignment (coarsest that fits). It grows automatically when `tensor_tile_cols > num_cores`.

Why `block_images = 1`: every image needs a different x row and therefore different B tiles; production has one image. Folding several images into one block would only matter when `tensor_images · col_groups · row_groups > num_cores`, which the multi-block-per-core loop already covers.

**One block = the core's whole assignment** in every production case (`num_blocks_this_core ≤ 1`). The block-id walk `for (block_idx = 0; block_idx < num_blocks_this_core; ++block_idx)` exists for grids smaller than the block count (multi-image or very wide inputs).

### Re-run on the intermediate (B tiles)

| Stage data | Axes | Assignment |
|------------|------|------------|
| B(n, w) tiles built in L1 | n, w (one tile per (n, w)) | built by **every core that writes column w** of image n — no designated builder core, no core waits on another core. There is no coordinating core in Phase 0. |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_bcast_row_tiles` | `bcast_buffer_depth` (host constant) | 2 (blocks) | for cores with `num_blocks_this_core > 1`: the reader reads/replicates block k+1 while the writer drains block k. In production (one block per core) it buys nothing and costs one extra block (2 KB); kept so the multi-block path is overlapped without a code change. Within a block, overlap comes from pushing **one B tile at a time** (writer starts on column 0 while column 1 is being replicated). |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DRAM read alignment (`ttnn.get_dram_alignment()`: 64 B BH, 32 B WH) on the partial x-tile read | size of the x read per B tile, `x_row_read_bytes` | `x_row_read_bytes = round_up(face_bytes + face_row_bytes, dram_alignment)` = `round_up(512 + 32, 64)` = 576 B on BH (544 on WH); read starts at page offset 0 into the CB page start (same alignment relation as a whole-page read); must be ≤ `tile_bytes`, else read the whole page (2048) | a sub-aligned read size or offset silently corrupts bytes or traps under the NoC sanitizer |
| `set_read_state<transfer_size>` single-packet state (`local_copy_helpers_dataflow.hpp:74-75`) | one replication transfer = one face row, `face_row_bytes` = 32 B (16 bf16) | static `transfer_size = 32` ≤ `NOC_MAX_BURST_SIZE` (512 on WH, `wormhole/noc/noc_parameters.h:219`) | static_assert |
| L1 budget 1 347 776 B (BB-05) | `block_tile_cols` (only extent the CB spans) | `max_block_tile_cols = floor(L1_BUDGET_FOR_OP / (bcast_buffer_depth · tile_bytes))` with `L1_BUDGET_FOR_OP = 1347776 − reserved_margin`; cores that would exceed it get more blocks (`num_blocks_this_core > 1`), never a refusal | OOM / launch refusal at very large W |
| Block-id partition must tile (n, h, w) exactly | ragged last row group / column group | runtime extents `this_block_tile_rows = min(block_tile_rows, tensor_out_tile_rows − rg·block_tile_rows)` (same for cols); nominal push/pop of **1 page per B tile** | out pages written twice or not at all (violates "every page written once") |

No accumulator, no DEST, no fixed-width index cap (page ids fit `uint32_t` for any L1-legal tensor).

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 — 2D grid split, per-core row-0 read + local replication** | **built** (stepping stone; successor R2) | always (single regime; selection is the knob derivation in Work Distribution) | `block_images × block_tile_rows × block_tile_cols` tiles of out; resident: `block_tile_cols` B tiles | Boundary DRAM. Minimum: x row read once (Wt × 576 B partial reads), out written once (`tensor_images·Ht·Wt` pages). Actual: out written exactly once (minimum); x: each x tile read once **per row group** = `row_groups` times (6 / 4 / 4 / 4 / 3 per tile at W = 512 / 256 / 207 / 196 / 185 on 110 cores), each a single 576 B request — 28 % of a page. Above minimum by `(row_groups − 1)·Wt·576 B` (46 KB at W=512 against 512 KB written). | Per block: one DRAM read round trip per column, one replication round (62 self-aimed 32 B reads + one read barrier) per column, one `cb_reserve`/`push` per column, one writes-flushed + `cb_pop` per column. A bigger `block_tile_rows` pays all of these once for more written tiles and lowers the x re-read count (`row_groups`); a bigger `block_tile_cols` batches the x reads under one barrier. Intended frequency: setup once per kernel; read barrier once per block; flush once per B tile; write barrier once per kernel (BB-03). |
| R2 — read-once + multicast B(n, w) down each column group | deferred — positive reason: in every production shape a core's x read is **one** 576 B request issued at kernel start, concurrent with every other core's; R2 puts the sender's DRAM read + replication + a semaphore round trip **in front of every receiver's first write** and moves 2048 B per receiver over the NoC to save 576 B of DRAM read per receiver, so it lengthens the critical path of the writes, which are the bound (DRAM write traffic). It wins only where duplicated small reads measurably queue on x's banks behind out's writes (spec rule 6 / RB-010) — reachable by measurement, see Perf lamps | `row_groups > 1` | same block; receivers skip `read_x_row_block` + `replicate_row_block` | x read once per call (minimum); adds `(row_groups − 1)·Wt·2048 B` cross-core mcast + one semaphore handshake per column group | amortizes the handshake over `block_tile_rows · block_tile_cols` written tiles per receiver |
| R3 — flat split of out tiles (`split_work_to_cores` over all `Ht·Wt` tiles), x tile read per out tile | rejected — dead end: reads x once per output tile (256 page reads at W=512 for 16 pages of data), exactly what spec rule 6 forbids preferring; superseded by R1, whose `min_block_tile_rows = 1` already reaches its occupancy with fewer reads | — | 1 tile | x: `Ht` reads per tile | — |
| R4 — column-only split (`row_groups = 1`), x read exactly once | built **as a knob value of R1** (`min_block_tile_rows = tensor_out_tile_rows` → `row_groups = 1`), not a separate algorithm; not the default because it leaves `num_cores − Wt` cores idle (16 of 110 busy at W=512, 16 writes = 32 KB serialized per core) | `min_block_tile_rows ≥ tensor_out_tile_rows` | `1 × Ht × block_tile_cols` | x once (minimum), out once (minimum) — the cheapest-traffic split | — (see Perf lamps: grid-synchronization) |
| R5 — build B through the compute engine (`unary_bcast<BroadcastType::ROW>`, `tt_metal/hw/inc/api/compute/bcast.h:83`) | rejected — bit identity of bfloat16 NaN payloads / denormals through SrcB (19-bit, `bcast.h:86`), DEST and the packer is not established (risk R2); it adds a compute kernel, a second CB and an unpack/pack per tile. Superseded by the dataflow replication of R1, which moves bytes and is bit-exact by construction | — | — | same DRAM traffic as R1, plus 2 CB passes per B tile | — |
| R6 — split a dependent axis + cross-core combine | not applicable — the op has no dependent axis (no result spans any axis; math.md axes) | — | — | — | — |
| R7 — physical sharded placements of x / out | not applicable — TARGET holds no `memory_layout` axis; production is DRAM interleaved (would be a placement knob-turn: every axis is independent) | — | — | — | — |

### Traffic ranking

Counts per call, W=512 on a 110-core grid (C = 110 is illustrative; the host reads the real grid).

| Rank | Split | DRAM out | DRAM x | Cross-core | Occupancy | Verdict |
|------|-------|----------|--------|------------|-----------|---------|
| 1 | columns only (R4: `row_groups = 1`) | 256 × 2 KB once | 16 × 576 B (once) | 0 | 16 cores | cheapest traffic; reachable as a knob value of R1 |
| 1 | columns + rows, read-once + mcast (R2) | once | 16 × 576 B | 80 × 2 KB mcast + 16 handshakes | 96 | same DRAM as rank 1, adds NoC traffic and a rendezvous; deferred |
| 3 | **columns first, then rows, per-core read (R1, built)** | once | 96 × 576 B = 55 KB | 0 | 96 (W=512), 32 (256), 28 (207/196), 18 (185) | +46 KB DRAM read in 80 single-request reads vs rank 1; chosen — see below |
| 4 | rows first, then columns | once | ≥ R1 (rows-first leaves more column re-reads per core) | 0 | same | dominated by R1 |
| 5 | flat over tiles (R3) | once | 256 × 576 B–2 KB | 0 | 110 | rejected |
| — | dependent-axis split | — | — | — | — | no dependent axis |

Chosen R1 with the column split first (the split that adds no x re-read) and the row split second, bounded below by `min_block_tile_rows = 2` so that each x tile is read **at most once per two output tiles of its column** (spec rule 6: never once per output tile). Rank 1 is not the default because the op is DRAM-**write**-bound (524 KB written vs 9 KB of x values) and the extra reads are single 576 B requests issued concurrently at kernel start; spreading the 256 writes over 96 injection points instead of 16 is what the grid-fill step buys. That trade (occupancy of the write stream vs 46 KB of duplicated small reads) is stated, not assumed: it is the grid-synchronization perf lamp, settled by one sweep of `min_block_tile_rows`.

### Block schedule

```cpp
// per core; num_blocks_this_core, block ids and extents are RT args
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    read_x_row_block(block_idx);      // reader: row 0 (+face-1 row 0) of block_tile_cols x tiles -> cb_bcast_row_tiles
    replicate_row_block(block_idx);   // reader: row 0 -> rows 1..31 of each tile, in place, push 1 page per tile
    write_out_block(block_idx);       // writer: each B tile -> block_tile_rows out pages, pop 1 page per tile
}
```

| Block operation | Kernel | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|--------|-------------|--------------------|-----------------------------------|
| `read_x_row_block` | reader (NCRISC, NoC0) | `block_tile_cols` x tiles of image n, columns `[cg·block_tile_cols, +this_block_tile_cols)`; each a single `x_row_read_bytes` (576 B) read from page offset 0 into the start of a reserved CB page | the reserved `block_tile_cols` pages | one `cb_reserve_back(block_tile_cols)` per block (nominal count, also for a ragged block); all reads of the block issued back to back, **one** `noc_async_read_barrier` per block (it also invalidates the RISC-V L1 cache, `dataflow_api.h:1932`, before replication reads the landed row) |
| `replicate_row_block` | reader | per B tile: source A = face-0 row 0 (page + 0) → face-0 rows 1..15 and face-2 rows 0..15 (31 × 32 B); source B = face-1 row 0 (page + 512) → face-1 rows 1..15 and face-3 rows 0..15 (31 × 32 B). Destinations of face f row r: `page + f·512 + r·32` | same pages | `set_read_state<32>` once per source row (2 per tile); 62 `read_with_state` per tile, all independent (sources are never overwritten); **one read barrier per tile**, then `cb_push_back(1)` for that tile so the writer starts while the next tile replicates. Rows 1..15 of faces 0/1 that arrived with the 576 B read are overwritten — every one of the 32 rows of every face pair is a destination or the source. |
| `write_out_block` | writer (BRISC, NoC1) | per B tile (column w): `this_block_tile_rows` out pages `(n·Ht + h)·Wt + w`, h ∈ row-group range, each a whole 2 KB page from the same L1 page | the B tile until its writes are flushed | `cb_wait_front(1)` per tile; `this_block_tile_rows` `noc_async_write_page` from one source; one `noc_async_writes_flushed` before `cb_pop_front(1)`; one `noc_async_write_barrier` at kernel end (BB-03) |

Stall shadow: the writer waits on the DRAM read latency + replication of the first B tile; every out page depends on a B tile, so no out work is independent of that wait. Within a block with `block_tile_cols > 1` the reader issues all column reads up front (one barrier) so later columns' read latency hides behind earlier columns' replication and writes; across blocks, depth 2 lets block k+1's reads overlap block k's writes. In production (one column per core) nothing is independent of the wait — stated, not hidden.

Ragged blocks: the last row group has `this_block_tile_rows < block_tile_rows` (W=512, 110 cores: 16 = 5·3 + 1); the last column group may have `this_block_tile_cols < block_tile_cols` (only when `block_tile_cols > 1`). CB accounting stays **nominal per block**: the reader reserves `block_tile_cols` pages per block (so every block starts at a multiple of `block_tile_cols` in a ring of `block_tile_cols · bcast_buffer_depth` pages and its pages are contiguous), pushes 1 page per real B tile and, for a ragged block, pushes the `block_tile_cols − this_block_tile_cols` unused pages at block end; the writer waits/pops 1 page per real tile and waits/pops the unused remainder at block end without writing it. Push total = pop total = `block_tile_cols` per block; the ring closes after `bcast_buffer_depth` blocks.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization / occupancy vs re-read | default fills the grid (96 cores at W=512) at the cost of `row_groups` reads of each x tile; with 1–3 writes per core, per-core launch and the per-core DRAM read latency may dominate | sweep `min_block_tile_rows` ∈ {1, 2, 4, Ht} (Ht = R4, x read once); measure device kernel time on LOOSE_CASES[0] (W=512) and [1] (W=196) |
| Read-once + mcast (R2) | duplicated small reads may queue behind out's writes on x's banks (RB-010) | R2 (deferred regime); measure only if the sweep above shows read queueing |
| Replication realization | 62 single-packet self-reads per tile may cost more issue time than fewer, larger dependent copies | (a) log-doubling self-reads (32→64→128→256 B within faces 0/1, then one 1024 B copy faces 0-1 → 2-3: 9 transfers, 5 barrier rounds); (b) RISC-V word stores of the two 32 B rows (≈ 496 stores per tile) |
| Writer split across RISCs | one BRISC issues all writes of a core; at 3 writes per core issue time is small, at Ht writes (R4) it may bound | split each block's out pages between BRISC (NoC1) and NCRISC (NoC0) (catalog `split_reader`, mirrored) |
| Core placement / bank stagger | blocks are laid row-wise over the grid (catalog `noc_placement`: row lines beat column lines ~2.9×); all cores of a column group write the same bank sequence | `row_wise=False` placement; start each core's write walk at a staggered row (catalog `bank_stagger`) |
| Single dataflow kernel | in production each core does read → replicate → write strictly in sequence; two kernels add one binary to the kernel-config region (BB-18) and one CB handshake | fuse reader and writer into one dataflow kernel (no CB) and compare launch-to-end time |
| Buffer depth | depth 2 is unused when `num_blocks_this_core == 1` | depth 1 (saves 2 KB; L1 is not under pressure — measure only for BB-18 / L1 neighbours) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x DRAM → L1 | bfloat16 tile bytes (first 576 B: face-0 rows 0..15 and face-1 row 0..) | `noc_async_read(x_accessor.get_noc_addr(x_page), l1_page, x_row_read_bytes)` on NoC0 | only row 0 of faces 0 and 1 is used; padding rows of x never reach a destination row (risk R1). Partial last tile: columns ≥ W of row 0 are x padding and land only in out padding columns |
| L1 replication (B tile) | raw bytes | self-aimed single-packet NoC reads (`dataflow_kernel_lib::set_read_state` / `read_with_state`) | byte moves only: bit-exact for NaN payloads, ±0, denormals, ±inf (risk R2). No compute kernel, no unpack/pack |
| reader → writer | one B tile per CB page | `cb_bcast_row_tiles`, push/pop 1 page per tile | the only CB |
| L1 → out DRAM | whole 2 KB tile pages | `noc_async_write_page(out_page, out_accessor, l1_page)` on NoC1, `this_block_tile_rows` writes per B tile from the same source | every out page written exactly once (blocks partition (n, h, w)); no intermediate tensor in DRAM |
| Deferred R2 contract | B tile 2 KB | `ttnn.Mcast` + `McastArgs` (`mcast/kernel/mcast_args.hpp`): sender = the row-group-0 core of a column group, receivers = the other cores of that column group, which the block→core mapping must place in one rectangle (re-key the host mapping cg-major); receivers' reader skips read/replicate and receives into the same CB page | kernels' block loop unchanged; reader gains a role CT arg |

Sharded placements: not in TARGET; if added, every axis is independent, so any shard flavour of out is a placement knob-turn (blocks = the shard, B tiles still built per core).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block (n, rg, cg): out tiles `n × [rg·block_tile_rows, +this_block_tile_rows) × [cg·block_tile_cols, +this_block_tile_cols)` |
| Grid | `device.compute_with_storage_grid_size()` → `num_cores = gx · gy` (runtime, never a literal) |
| Knob derivation (host, single place) | `block_tile_cols = clamp(ceil(Wt / num_cores), 1, max_block_tile_cols)`; `col_groups = ceil(Wt / block_tile_cols)`; `row_groups_fill = max(1, floor(num_cores / (tensor_images · col_groups)))`; `row_groups_cap = ceil(Ht / min_block_tile_rows)`; `row_groups = clamp(min(row_groups_fill, row_groups_cap), 1, Ht)`; `block_tile_rows = ceil(Ht / row_groups)`; `row_groups = ceil(Ht / block_tile_rows)`; `num_blocks = tensor_images · row_groups · col_groups`; block id `b = (n · row_groups + rg) · col_groups + cg` |
| Per-core work | if `num_blocks ≤ num_cores`: core k (row-wise order over the grid) owns block k, `num_blocks_this_core = 1`, `num_cores_used = num_blocks`; else `ttnn.split_work_to_cores(grid, num_blocks, row_wise=True)` → contiguous block-id ranges (two groups differing by 1 block) |
| Remainder | ragged last row group / column group via runtime extents (`min(...)`), alignment-aware `ceil` everywhere; per-image tile rows |
| Core ranges (BB-17) | used cores = the first `num_cores_used` cores in row-wise order = ≤ 2 rectangles (full grid rows + one partial row); one reader and one writer kernel over that range set |
| Plan report (BB-08) | host exposes `g6_expand.describe_plan(input_tensor, sizes)` → dict {grid, num_cores_used, row_groups, col_groups, block_tile_rows, block_tile_cols, num_blocks, x_row_read_bytes} and per-core role/block range (BB-21 role dump) |

Production plan on a 110-core grid (illustrative; recomputed from the device grid):

| W | Ht = Wt | col_groups | row_groups | block_tile_rows (last) | cores used | writes / core | x reads / tile |
|---|---------|-----------|------------|------------------------|-----------|---------------|----------------|
| 512 | 16 | 16 | 6 | 3 (1) | 96 | ≤ 3 | 6 |
| 256 | 8 | 8 | 4 | 2 | 32 | 2 | 4 |
| 207 | 7 | 7 | 4 | 2 (1) | 28 | ≤ 2 | 4 |
| 196 | 7 | 7 | 4 | 2 (1) | 28 | ≤ 2 | 4 |
| 185 | 6 | 6 | 3 | 2 | 18 | 2 | 3 |

Single regime (R1); no regime-pinned tests needed beyond the production widths, which already exercise ragged (512, 207, 196) and even (256, 185) row groups.

### Program cache (BB-07, BB-16)

The ProgramDescriptor is built once per key and cached in a module dict. Key = (input logical shape, out shape, dtype, layout, input memory config, device id, grid size, `min_block_tile_rows`, `bcast_buffer_depth`). Buffer addresses are **not** in the key: on a hit only the address RT args (x and out base) of each core are rewritten. Compile-time args are only tensor-accessor args and `x_row_read_bytes`/`face_row_bytes`/CB index; every W-dependent quantity (Wt, Ht, block extents, block ids, n) is an RT arg, so all widths share binaries. Output allocated per call with `ttnn.allocate_tensor_on_device` (no allocation inside kernels; traced replay sees one allocation inside the captured call).

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_bcast_row_tiles` | 0 | `tile_bytes` (2048) | `block_tile_cols · bcast_buffer_depth` | spans `w` (block_tile_cols B tiles live per block); streams over `h` (one B tile serves all `this_block_tile_rows` writes) and over blocks (depth) | bfloat16 (input dtype; raw bytes, never unpacked) | reader | writer | whole kernel; each page: from reserve (read landing) through replication, push, all its writes flushed, pop |

Inventory: one CB. The x landing area **is** the B tile (transform in place, Rule 3 pattern 2); out pages are written straight from it (pack-into-destination analogue: no output staging CB). Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `read_x_row_block` | `this_block_tile_cols` partial x tiles (576 B each) | raw NoC read | — (DRAM x) | `cb_bcast_row_tiles`, `block_tile_cols` reserved pages (nominal) | reserved, not pushed; row 0 of faces 0/1 landed |
| 2 | `replicate_row_block` | per tile: 2 sources × 31 destinations of 32 B | `local_copy_helpers_dataflow` (`set_read_state` / `read_with_state`) | `cb_bcast_row_tiles` reserved pages (writer cannot see them yet) | `cb_bcast_row_tiles`, +1 pushed page per tile | each tile pushed as soon as its replication barrier returns |
| 3 | `write_out_block` | per B tile: `this_block_tile_rows` out pages | raw NoC page write (no helper writes interleaved tile pages from one source) | `cb_bcast_row_tiles`, wait 1 page | — (DRAM out) | page popped after `noc_async_writes_flushed` |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| all | raw_api | `TensorAccessor` / `TensorAccessorArgs` (`get_noc_addr(page_id, offset)`) | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:71`, `:165` | CT accessor args at the end of CT args (x for reader, out for writer) | — | — | — |
| `read_x_row_block` | raw_api | `cb_reserve_back` / `get_write_ptr` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:406`, `:325` | `(cb_bcast_row_tiles, block_tile_cols)` (nominal) | — | `cb_bcast_row_tiles` | page count = `block_tile_cols` |
| `read_x_row_block` | raw_api | `noc_async_read` | `dataflow_api.h:547` | `(x.get_noc_addr(n·Wt + w), page_l1, x_row_read_bytes)` | — | `cb_bcast_row_tiles` | `x_row_read_bytes` (mechanism cap) |
| `read_x_row_block`, `replicate_row_block` | raw_api | `noc_async_read_barrier` | `dataflow_api.h:1921` | once per block (reads), once per tile (replication); invalidates L1 cache (`:1932`) | — | — | — |
| `replicate_row_block` | helper | `dataflow_kernel_lib::set_read_state<32>(noc, src)` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:74-75` | `transfer_size = face_row_bytes = 32`; src = page+0, then page+512 | `cb_bcast_row_tiles` | `cb_bcast_row_tiles` | — |
| `replicate_row_block` | helper | `dataflow_kernel_lib::read_with_state<32>(noc, dst, src)` | `local_copy_helpers_dataflow.hpp:94-95` | 31 destinations per source (see Block schedule) | `cb_bcast_row_tiles` | `cb_bcast_row_tiles` | — |
| `replicate_row_block` | raw_api | `cb_push_back` | `dataflow_api.h:208` | `(cb_bcast_row_tiles, 1)` per tile | — | `cb_bcast_row_tiles` | — |
| `write_out_block` | raw_api | `cb_wait_front` / `get_read_ptr` | `dataflow_api.h:479`, `:346` | `(cb_bcast_row_tiles, 1)` | `cb_bcast_row_tiles` | — | — |
| `write_out_block` | raw_api | `noc_async_write_page` | `dataflow_api.h:1418` | `(out_page, out_accessor, l1_page)` × `this_block_tile_rows` | `cb_bcast_row_tiles` | — | trip count = `this_block_tile_rows` |
| `write_out_block` | raw_api | `noc_async_writes_flushed` / `cb_pop_front` | `dataflow_api.h:1980`, `:260` | before each pop | `cb_bcast_row_tiles` | — | — |
| kernel end (both) | raw_api | `noc_async_write_barrier` (writer), read barrier already final (reader) | `dataflow_api.h:1951` | BB-03: no transaction in flight at return | — | — | — |

Helpers considered and rejected (every raw_api row):

| Raw API use | Helper considered | Mismatch (file:line) | Reason |
|-------------|-------------------|----------------------|--------|
| x read / out write | `tilize_helpers_dataflow.hpp`, `reduce_helpers_dataflow.hpp`, `tile_column_dataflow.hpp` | `tile_column_dataflow.hpp:7-18` (builds a tile *column* by RISC-V stores, for column-broadcast operands) | no kernel_lib helper moves interleaved DRAM tile pages or a partial page; the page APIs are the primitive |
| B-tile build | `tile_column_dataflow.hpp` | `:10-14` addresses a column (stride FACE_WIDTH over faces 0/2 or 1/3), not a row | this op needs a row replicated; the self-aimed copy helper is used instead |
| B-tile build | compute `unary_bcast<ROW>` / `eltwise` broadcast helpers (`kernel_lib/eltwise/broadcast/bcast.hpp`) | `tt_metal/hw/inc/api/compute/bcast.h:86` (SrcB 19 bits) | bit identity unproven (R2); rejected regime R5 |
| multicast | `ttnn.Mcast` / `McastArgs` | — | only in deferred R2 |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Padding rows of x reach out (R1) | a whole-tile copy puts x rows 1..31 into out; the 576 B read lands rows 0..15 of face 0 | every row of every face is either the source row 0 or a replication destination; the poisoned-padding test (padding = 1000.0) catches a missed row |
| Bit pattern changed (R2) | gate is bit-exact incl. NaN payloads, denormals | dataflow byte moves only; no compute kernel; special-values test |
| Replication reads stale/unlanded data | self-reads issued before the DRAM read landed copy garbage; on Blackhole a RISC-V-side copy must also see L1 fresh | read barrier after `read_x_row_block` before any replication; barrier invalidates the L1 cache (`dataflow_api.h:1932`) |
| Push before replication lands | writer would write a half-built tile | read barrier after each tile's 62 self-reads, then push |
| Pop before writes read L1 | the next block overwrites a page still being sent | `noc_async_writes_flushed` before every pop; final write barrier (BB-03) |
| Out pages written twice / missed | ragged last row group (W=512: 16 = 5·3+1; W=207/196: 7 = 3·2+1) | runtime extents `min(...)`; blocks partition (n, h, w) exactly |
| Self-aimed read coordinates | NoC0 and NoC1 coordinate spaces differ (`local_copy_helpers_dataflow.hpp:47-52`) | use the reader's own NoC id for `local_addr`/state |
| Program cache keyed on address | a fresh output tensor misses the cache (BB-07) | key excludes addresses; addresses patched into RT args |
| Grid differs by device | cores used and row_groups change with the grid | everything derived from `compute_with_storage_grid_size()`; plan exposed (BB-08) |
| Profiling zones (BB-21) | un-zoned waits read as work | zones `read_x_row`, `replicate_row`, `wait_bcast_tile` (writer `cb_wait_front`), `write_out`, `drain_writes`; compiled only in the zone build |
