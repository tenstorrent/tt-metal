# Operation Design: g6_expand

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel; pure byte movement) |
| Goal | Materialize `x.expand(*sizes).contiguous()` for a 4-D tiled bf16 tensor whose dim 2 has size 1, as **one** device program: every output tile is the input tile of its column with **row 0 replicated into all 32 rows**, written once to DRAM. Bit-exact. |
| Math | `out[i0, i1, h, w] = x[i0, i1, 0, w]` (production: `out[0,0,h,w] = x[0,0,0,w]`, `h,w ∈ [0, W)`) |
| Tile form | `out_tile[img, ht, wt] = bcast_row0(x_tile[img, 0, wt])` for every `ht ∈ [0, Ht_out)`; `bcast_row0` copies the 32 B row 0 of face 0 into every row of faces 0 and 2, and row 0 of face 1 into every row of faces 1 and 3. |
| References | `eval/golden_tests/g6_expand/feature_spec.py`, `eval/golden_tests/g6_expand/helpers.py` (oracle `torch.expand(*sizes).contiguous()`), `eval/op_template.py`, `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `ttnn/ttnn/operations/examples/{noc_placement,width_split,tile_reorder,double_buffer,bank_stagger}/README.md`, `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` (`get_noc_addr(page_id, offset)`) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 4, on device, DRAM interleaved, bf16, TILE, `shape[2] == 1` | — | buffer address: RT |
| `sizes` | list / tuple of 4 ints | yes (positional) | each `-1` or `≥ 1`; `sizes[d] ∈ {-1, shape[d]}` for `d ∈ {0,1,3}`; `sizes[2] ∈ {-1} ∪ [1, ∞)` | — | host only (derives `Ht_out`, the work split; part of the program cache key) |

No `memory_config`, no output dtype, no `compute_kernel_config`: out takes dtype, layout and placement of `input_tensor`.

### Entry-point contract (validate() runs first)

| Check (order) | Raises |
|---|---|
| `validate()` — SUPPORTED per-axis (`dtype`, `layout`, `alignment`), then EXCLUSIONS (none) | `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract` |
| input on device; memory config DRAM INTERLEAVED | `ValueError` |
| rank 4; `len(sizes) == 4` | `ValueError` |
| every size is `-1` or `≥ 1` (0 and other negatives refused) | `ValueError` |
| `sizes[d] ∉ {-1, shape[d]}` for a dim with `shape[d] != 1` | `ValueError` |
| expansion of any dim other than 2 (`shape[d] == 1`, `sizes[d] ∉ {-1, 1}`, `d ≠ 2`) | `ValueError` |
| `shape[2] != 1` (Phase 0 input requirement: dim 2 is the expanded dim) | `ValueError` |

`tag_alignment(inputs, axes)`: `out = [d if s == -1 else s for d, s in zip(scenario["shape"], scenario["sizes"])]`; `"w_non_aligned"` if `out[-1] % 32` (also when `out[-2] % 32`), else `"h_non_aligned"` if `out[-2] % 32`, else `"tile_aligned"`. The entry point builds the same scenario dict `{"shape": list(x.shape), "sizes": list(sizes)}` for `validate()`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d0, d1, 1, W)`; production `(1, 1, 1, W)`, `W ∈ {185, 196, 207, 256, 512}` |
| Dtype | bfloat16 |
| Layout | TILE (32×32, 4 faces of 16×16, 2048 B page) |
| Memory | DRAM interleaved |
| Padding | rows 1..31 of every tile and columns ≥ W are **padding with arbitrary content**; no logical output element may depend on them |

### Output

| Property | Value |
|----------|-------|
| Shape | `sizes` with each `-1` replaced by `shape[d]`; production `(1, 1, W, W)` |
| Dtype | dtype of input (bfloat16) |
| Layout | layout of input (TILE) |
| Memory | memory config of input (DRAM interleaved); new allocation, never a view |
| Padding | rows ≥ H_out of the last tile row and columns ≥ W of the last tile column hold whatever the replicated tile holds; not part of the result |

## Blocking Model

Geometry (alignment-aware, per image, `ceil` everywhere):

| Symbol | Definition | Production W=512 / 256 / 207 / 196 / 185 |
|---|---|---|
| `N_img` | `out[0] * out[1]` | 1 |
| `Wt` | `ceil(W / 32)` | 16 / 8 / 7 / 7 / 6 |
| `Ht` | `ceil(H_out / 32)` (output tile rows **per image**; input has 1 tile row per image) | 16 / 8 / 7 / 7 / 6 |
| `num_cols` | `N_img * Wt` — one "column" = one input page = one broadcast tile | 16 / 8 / 7 / 7 / 6 |
| `total_tiles` | `num_cols * Ht` | 256 / 64 / 49 / 49 / 36 |

**Work order (single source of the split):** output tiles are enumerated **column-major**: linear index `t ∈ [0, total_tiles)`, `col = t / Ht` (= input page id = `img*Wt + wt`), `ht = t % Ht`, output page id `= (img*Ht + ht)*Wt + wt` with `img = col / Wt`, `wt = col % Wt`. Each core owns one contiguous range `[start_t, start_t + n_t)` of this order. A contiguous range in column-major order is a few **column segments**: one broadcast tile per segment is built once and written to every output tile of the segment.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| image (`N_img = d0*d1`) | independent — each image's output reads only its own input row | folded into the linear column index `col`; extent per core = whatever the range covers | whole per-core range (`n_t` tiles) | host: `split_work_to_cores(grid, total_tiles, row_wise=True)` | spread over the grid via the linear order (Phase 0) | knob-turn |
| output tile column `wt` (W) | independent — no value crosses a tile column; **x varies along it** (each column needs its own input page) | `col_block` — broadcast tiles resident at once per core (one fetch batch, one read barrier) | `min(max_cols_per_core, COL_BLOCK_CAP)` = every column the core touches (≤ 3 in production) | host constant computed once from the split, passed as one CT arg | spread over the grid via the linear order (Phase 0) | knob-turn |
| output tile row `ht` (H_out) | independent for the output; **x is reuse-shared along it** — the same input page feeds all `Ht` tile rows of its column | per-core run length along `ht` inside a column segment = whole segment (`≤ n_t`); no sub-chunking | whole segment (1–3 tiles at W=512 on a 110–130-core grid; `Ht` max) | derived from `start_t`, `n_t` (RT args) and `Ht` (CT arg) — no second literal | spread over the grid via the linear order (Phase 0) | knob-turn (core count); scheme-change for the reuse (column-group multicast, deferred) |
| tile-internal row `r ∈ [0,32)` | reuse-shared — the broadcast itself: row 0 of the input tile feeds all 32 rows | the whole tile (one 2048 B page is the transfer unit) | 32 rows (whole tile) | fixed by the tile format | not split (sub-tile split would fragment 2 KB writes into faces — `tile_reorder` shows whole-page writes win) | none |
| tile-internal column (32 within a tile) | independent | whole tile | 32 | fixed by the tile format | not split (same reason) | none |
| core grid | — (parameter, not an axis) | `grid` | `device.compute_with_storage_grid_size()` | host, one place; flows into `split_work_to_cores` | — | knob-turn (fewer cores = grid-sync lamp) |

There is **no dependent axis**: nothing is reduced, scanned or combined. Every stage's own data (the broadcast tile) has the axes of one column — it is produced and consumed on the same core; no core collects or hands out work.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_bcast_tiles` | `bcast_depth` (blocks of `col_block` tiles in flight) | `1` when `num_col_blocks_max == 1` (all production shapes: the core's whole column set is one block), else `2` | depth 2 lets the reader fetch + replicate block k+1 while the writer drains block k; with one block per core there is nothing to overlap at block granularity — the per-tile push (see schedule) already lets writes of column j overlap replication of column j+1 |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DRAM NoC read alignment (32 B WH, 64 B BH — `ttnn.get_dram_alignment()`) | `row_fetch_bytes`, the size of each row-0 fetch | `row_fetch_bytes = max(32, dram_alignment)`; fetch offsets 0 and 512 within the page are multiples of every DRAM alignment; destination offsets 0 and 512 inside a CB tile slot keep the slot's alignment | a 32 B read on a 64 B-aligned DRAM tier returns wrong bytes / NoC error |
| face geometry (bf16 face = 16×16×2 = 512 B) | `row_fetch_bytes ≤ 512` | `row_fetch_bytes` ≤ 512 (assert on host) | fetch would cross into the next face — harmless bytes but breaks the "one row-0 segment per fetch" model |
| L1 budget (1 347 776 B per core left by the model, BB-05) | `col_block * bcast_depth` | `COL_BLOCK_CAP = 64` tiles (128 KiB); a core whose range touches more columns loops `num_col_blocks = ceil(cols_this_core / col_block)` blocks | an unbounded `col_block` grows with `N_img*Wt / num_cores` → OOM on huge W·N (W has no upper limit in the contract) |
| `split_work_to_cores` remainder | `n_t` differs by 1 between `core_group_1` and `core_group_2` | `n_t` is an RT arg per core | — |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 — column-major linear split, per-core row-0 fetch + in-L1 replicate, whole-tile writes** | **built** | every valid call (only regime) | per core: `col_block` broadcast tiles (all columns its range touches), each written to its whole column segment | **out crosses DRAM exactly once, as writes** (= minimum, `total_tiles × 2048 B`; 512 KiB at W=512). x: each (core, column) pair fetches `2 × row_fetch_bytes` (128 B on BH) — total `Σ_cores cols_touched × 128 B` ≤ `(num_cores + num_cols) × 128 B` ≈ 18 KiB at W=512 = 3.5 % of write bytes; minimum (each input page's row 0 once) is `num_cols × 128 B` = 2 KiB | per block: one read barrier (all row-0 fetches of the core in flight together), one CB reserve; per column segment: one replicate, one push/wait/pop, one write flush. Bigger `col_block` = fewer read barriers (one per core in production). Bigger column segment (fewer cores) = fewer duplicate fetches — that is the grid-sync lamp, not a block change |
| R2 — column-group multicast of the broadcast tile (one injector per column group fetches + replicates, mcasts 2 KiB to the group via `ttnn.Mcast`/`McastArgs`) | deferred — the duplicate fetches R2 removes are ≈16 KiB of 128 B requests spread over all DRAM banks against 512 KiB of writes, while R2 puts a semaphore handshake + multicast on the critical path of a ~1.5 µs op; adopt only if a profile shows fetch latency/queuing (RB-010) on the critical path | `cols_touched` of cores sharing a column > 1 and profile-justified | same blocks as R1; the broadcast tile arrives by mcast instead of fetch+replicate | x crosses DRAM once per column (minimum); adds `(group_size−1) × 2 KiB` NoC multicast per column | reachable: R1's column-major order makes the cores of one column a **consecutive run** in `split_work_to_cores(row_wise=True)` order; R2 groups those runs, the first core is the injector, and the writer side is unchanged (it still waits on `cb_bcast_tiles`) |
| R3 — row-major linear split (same `split_work_to_cores`, order `t = ht*Wt + col`) | rejected — a core touches `min(n_t, Wt)` columns instead of `ceil(n_t/Ht)+1`: equal fetches at `n_t ≤ 2`, up to `Wt`× more fetch requests as `n_t` grows (fewer cores, larger N_img); never fewer; same writes. Dominated; superseded by R1 | — | — | — | — |
| R4 — fetch the whole 2 KiB input page per (core, column) | rejected — 16× the read bytes of R1 for the same request count (1 vs 2 requests is not worth 2 KiB vs 128 B against a write-bound op); superseded by R1's row-0 fetch | — | — | — | — |
| R5 — compute-kernel broadcast (unpack row-broadcast → DEST → pack, e.g. `copy`/`unary_bcast<ROW>`) | rejected — values traverse unpacker/DEST/packer: format conversion can flush denormals / canonicalize NaN payloads, violating the bit-exact contract; adds three TRISCs of work for zero arithmetic; superseded by R1's byte copy | — | — | — | — |
| R6 — one core per column (`num_cores = num_cols`) | rejected — 16 / 8 / 7 / 7 / 6 writers for a DRAM-write-bound op; occupancy superseded by R1 at equal fetch minimum per core | — | — | — | — |
| R7 — split a dependent axis + combine | n/a — the op has no dependent axis (no reduction, no scan) | — | — | — | — |
| R8 — sharded placements of out / x | n/a — TARGET holds only DRAM interleaved; if added, a height/width/block shard of out is a knob-turn (every output tile is self-contained; the core-assignment becomes the shard's) | — | — | — | — |

Single built regime: R1 covers every valid input because nothing in the op's math changes with shape — the only shape dependence is the per-core range, which `split_work_to_cores` handles for any `total_tiles ≥ 1` (a single-tile output runs on one core; that is the split's choice, not a cap).

### Traffic ranking

| Candidate split | DRAM bytes (x) | DRAM bytes (out) | Cross-core | Rank |
|---|---|---|---|---|
| R1 column-major linear, row-0 fetch | `≈ (num_cores + num_cols) × 128 B` (≤ 18 KiB @512) | `total_tiles × 2 KiB`, once | none | **1 (chosen)** — near-minimal bytes, full grid, no synchronization |
| R2 column-group mcast | `num_cols × 128 B` (2 KiB @512) | same | `≈ num_cores × 2 KiB` mcast + semaphores | 2 — fewer DRAM bytes, but adds NoC traffic and a handshake on the critical path; deferred |
| R3 row-major linear | `Σ_cores min(n_t, Wt) × 128 B` — equal to R1 at 2 tiles/core, up to `Wt`× R1 with more tiles per core | same | none | 4 — never better than R1 |
| R4 full-page fetch | `≈ (num_cores + num_cols) × 2 KiB` (≈ 290 KiB @512) | same | none | 5 |
| R6 one core per column | `num_cols × 128 B` | same | none | 3 on bytes, rejected on write parallelism |

Output writes dominate every candidate (fixed at the minimum, once); the ranking is decided by x's fetch bytes/requests and by whether a handshake enters the critical path. Occupancy (R6) is accounted separately and loses.

### Stall-shadow check

The writer's first write waits on the reader's first fetch (one DRAM round trip). Nothing in out is independent of x, so no output work fits in that window. What does fit: the reader issues **all** row-0 fetches of the block before its single barrier, and pushes each column's broadcast tile as soon as it is replicated, so the writes of column j overlap the replication of column j+1. No reorder changes values (pure copies), so no precision gate is involved.

### Block schedule

```cpp
// reader (NCRISC, NoC0)                         // writer (BRISC, NoC1)
for (blk = 0; blk < num_col_blocks_this_core; ++blk) {
    fetch_row0_block(blk);        // all cols of blk in flight, 1 barrier
    for (c in blk) {
        replicate_row0(c);        // in-place in the tile slot
        publish_bcast_tile(c);    //  ───────────►   for (seg = 0; seg < num_segments_this_core; ++seg) {
    }                                                  await_bcast_tile(seg);
}                                                      write_column_segment(seg); // seg_len tiles, same L1 src
                                                       retire_bcast_tile(seg);    // flush, then pop
                                                   }
                                                   drain_writes();               // write barrier before return (BB-03)
```

| Block operation | Kernel | Block shape | Resident across it | Intended frequency of inits / barriers |
|---|---|---|---|---|
| `fetch_row0_block` | reader | `col_block` columns × 2 fetches of `row_fetch_bytes` (offsets 0 and 512 of input page `col` → offsets 0 and 512 of the column's tile slot in `cb_bcast_tiles`) | reserved `col_block` slots of `cb_bcast_tiles` | one `cb_reserve_back(col_block)` and **one** `noc_async_read_barrier` per block (= once per core in production) |
| `replicate_row0` | reader | one 2048 B tile slot: face0 row0 (32 B) → face0 rows 1..15 and face2 rows 0..15; face1 row0 → face1 rows 1..15 and face3 rows 0..15 (overwrites the fetched padding row 1 too) | the column's tile slot | once per column; RISC-V word stores in L1, no NoC |
| `publish_bcast_tile` | reader | 1 page | — | one `cb_push_back(1)` per column |
| `await_bcast_tile` | writer | 1 page | — | one `cb_wait_front(1)` per column segment |
| `write_column_segment` | writer | `seg_len` whole-tile writes (2048 B each) from the same L1 page to output pages `(img*Ht + ht)*Wt + wt`, `ht` over the segment | broadcast tile at the CB front | no barrier between writes |
| `retire_bcast_tile` | writer | 1 page | — | one `noc_async_writes_flushed()` then `cb_pop_front(1)` per segment (source must be read out before the slot is reused) |
| `drain_writes` | writer | — | — | one `noc_async_write_barrier()` at kernel end |

Segments and blocks per core are derived on device from `start_t`, `n_t` (RT) and `Ht`, `col_block` (CT): `first_col = start_t / Ht`, `last_col = (start_t + n_t − 1) / Ht`, `cols_this_core = last_col − first_col + 1`; segment `c` covers `ht ∈ [max(start_t, c*Ht) − c*Ht, min(start_t+n_t, (c+1)*Ht) − c*Ht)`. Reader and writer derive the same column list from the same RT args — the CB sync count is `cols_this_core` on both sides.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| grid synchronization | at W=512 the full grid gives ~2 tiles per core; launch + one DRAM read round trip per core may dominate 2 × 2 KiB of writes | `grid` restricted so each core owns ≥ 4 / 8 tiles (e.g. 64 / 32 cores) |
| write issue on one RISC | all 2 KiB writes are issued by BRISC; the reader RISC is idle after its fetches | split each segment's writes between writer (NoC1) and reader (NoC0) — `split_reader` pattern mirrored for writes |
| core placement / NoC | `row_wise=True` per `noc_placement`; Blackhole may prefer diagonal | `row_wise=False`, diagonal core order |
| bank stagger | consecutive cores write column-major pages `ht*Wt + wt` with stride `Wt` → with `Wt % 8 == 0` (W=256, 512) every core's segment starts on the same bank subset | rotate each core's segment start (`bank_stagger` `stagger_rows`) |
| replication mechanism | RISC-V word stores (~500 per tile) vs. local NoC L1→L1 doubling copies | local `noc_async_read` from `get_noc_addr(my_x, my_y, slot)` doubling 32 B → 1 KiB per face pair |
| row-fetch request count | 2 small requests per (core, column) | 1 request of the first 1024 B (faces 0+1) per (core, column) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x DRAM → L1 slot | bf16 tile bytes (Float16_b), 2 × `row_fetch_bytes` | `noc_async_read(accessor.get_noc_addr(col, 0 / 512), slot + 0 / 512, row_fetch_bytes)` on NoC0 (reader) | fetches only row 0 (+ alignment pad) of faces 0 and 1 — the padding rows are fetched only as alignment filler and are overwritten |
| replicate in L1 | same bytes | RISC-V `uint32_t` copies in the reader | bit-exact: no format conversion anywhere |
| reader → writer | 1 page per column | `cb_bcast_tiles` push/wait | ownership: reader produces, writer consumes |
| L1 → out DRAM | 2048 B page writes | `noc_async_write(slot, out_accessor.get_noc_addr(page), 2048)` on NoC1 | each output page written exactly once by exactly one core |
| deferred R2 contract | — | injector = first core of the consecutive run of cores touching a column; it fetches + replicates as in R1 and multicasts the 2 KiB tile into the same slot of `cb_bcast_tiles` of its group (`McastArgs` sender `send()`, receivers `receive()` replace `fetch_row0_block`+`replicate_row0`); writer unchanged | groups are consecutive core runs → rectangular / few ranges (BB-17) with `row_wise=True` ordering |

No compute kernel. No DRAM intermediate. No other device op is dispatched; one `ttnn.generic_op` call per invocation.

Placement axis: TARGET holds only DRAM interleaved; no `*_SHARDED` value is in scope (R8).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one output tile in column-major order (`t → (col, ht)`); per core a contiguous range = 1..few column segments |
| Grid | `device.compute_with_storage_grid_size()` (the `grid` knob) |
| Split | `ttnn.split_work_to_cores(grid, total_tiles, row_wise=True)` → `core_group_1` (`n_t = per_core_g1`), `core_group_2` (`n_t = per_core_g2`); ranges assigned in the returned core order, `start_t` = running sum |
| Per-core work | `n_t` output tiles; `cols_this_core ≤ ceil(n_t / Ht) + 1` broadcast tiles |
| Remainder | handled by the two groups (sizes differ by 1); `Wt`, `Ht` use `ceil` per image, so partial last tile row/column are whole tiles in the order |
| `col_block` | host: `max_cols_per_core = max over cores of cols_this_core` (computed from the actual ranges); `col_block = min(max_cols_per_core, COL_BLOCK_CAP)`; `num_col_blocks_max = ceil(max_cols_per_core / col_block)`; `bcast_depth = 1 if num_col_blocks_max == 1 else 2` |
| Core groups | `core_group_1` / `core_group_2` from `split_work_to_cores` are rectangular ranges (BB-17); the same two kernels run on both, only RT args differ |
| Regime selection | single regime (R1); report it as `"R1-colmajor-row0fetch"` together with `num_cores`, `col_block` (BB-08) |

| Production | total_tiles | cores (130-core grid; 110 if harvested) | tiles/core | cols/core |
|---|---|---|---|---|
| W=512 | 256 | 128 (110) | 2 (2–3) | ≤ 2 |
| W=256 | 64 | 64 | 1 | 1 |
| W=207, 196 | 49 | 49 | 1 | 1 |
| W=185 | 36 | 36 | 1 | 1 |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_bcast_tiles` | 0 | 2048 (`input_tensor.buffer_page_size()`) | `bcast_depth * col_block` | spans the core's `wt`/`col` block (`col_block`); streams over `ht` (one page serves the whole segment); spans the tile's 32 rows | bfloat16 (Float16_b), never through DEST | reader | writer | whole program |

That is the whole inventory: the fetched row 0 is landed **in the destination tile slot** and replicated **in place**, so there is no staging CB; the write source is that same slot, so there is no output CB.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `fetch_row0_block` | `col_block` columns × 2 × `row_fetch_bytes` | no (raw NoC read) | — (x in DRAM) | `cb_bcast_tiles`, reserve `min(col_block, remaining cols)` | reserved, row 0 of faces 0/1 landed per slot |
| 2 | `replicate_row0` | 1 tile slot | no | `cb_bcast_tiles` reserved slot (write pointer + `j*2048`) | same slot | full broadcast tile |
| 3 | `publish_bcast_tile` | 1 page | no | — | `cb_bcast_tiles`, push 1 | visible to writer |
| 4 | `await_bcast_tile` | 1 page | no | `cb_bcast_tiles`, wait 1 | — | front page readable |
| 5 | `write_column_segment` | `seg_len` tiles | no (raw NoC write) | `cb_bcast_tiles` front page | out DRAM pages | unchanged |
| 6 | `retire_bcast_tile` | 1 page | no | `cb_bcast_tiles`, pop 1 after `noc_async_writes_flushed()` | — | slot free |
| 7 | `drain_writes` | — | no | — | — | all writes landed (BB-03) |

Push/wait count on `cb_bcast_tiles`: `cols_this_core` on both sides. The reader reserves per block but pushes per column — legal because it pushes exactly the pages it reserved, in order, before reserving the next block.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `fetch_row0_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` + `noc_async_read` + `noc_async_read_barrier` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | `page_id = col`, `offset ∈ {0, 512}`, size `row_fetch_bytes` | — | `cb_bcast_tiles` | `col_block` (fetches per barrier) |
| `replicate_row0` | raw_api | volatile `uint32_t*` copies in L1 (`get_write_ptr(cb_bcast_tiles)`) | — (plain C++) | 8 words per face row | `cb_bcast_tiles` | `cb_bcast_tiles` | — |
| `publish/await/retire` | raw_api | `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` | dataflow API | 1 page per column; reserve `col_block` per block | `cb_bcast_tiles` | `cb_bcast_tiles` | `col_block`, `bcast_depth` |
| `write_column_segment` | raw_api | `TensorAccessor::get_noc_addr(page_id)` + `noc_async_write` (+ `noc_async_writes_flushed`, `noc_async_write_barrier`) | `tensor_accessor.h:165` | 2048 B per page | `cb_bcast_tiles` | — | segment length (from RT `start_t`, `n_t`) |
| host split | raw_api | `ttnn.split_work_to_cores(grid, total_tiles, row_wise=True)` | `.claude/references/ttnn-python-utility-bindings.md:255` | | | | `grid` |

**Helpers considered and rejected** (every phase is dataflow-only; there is no compute phase):

| Candidate | Mismatch | Reason |
|---|---|---|
| `eltwise_convenience.hpp` `copy` / `unary` with `BroadcastDim::Row` (compute) | routes through unpack → DEST → pack | contract forbids a representation that may change bits (denormals / NaN payloads); compute TRISCs add cost for zero arithmetic (regime R5) |
| `tilize_helpers.hpp` / `untilize_helpers.hpp` | layout conversion | input and output are both TILE; no conversion exists in the op |
| `mcast/kernel/mcast_args.hpp` / `ttnn.Mcast` | multicast of a shared operand | the matching use is deferred regime R2; Phase 0 fetches per core (R1) |
| `reduce_helpers_dataflow.hpp` scaler prep | — | no reduction |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Dependence on x's tile padding | combining or copying rows 1..31 of x leaks padding into logical output (poisoned-padding golden case fills 1000.0) | only row 0 of faces 0/1 is used; the fetched alignment filler (row 1) is overwritten by `replicate_row0`; columns ≥ W of the last input tile land only in output padding columns |
| Bit-exactness | any compute path may flush denormals / alter NaN | no compute kernel; CB format irrelevant to bytes; byte copies only |
| Slot reuse before writes read it | popping `cb_bcast_tiles` while writes still read the slot corrupts later columns (BB-06 nondeterminism) | `noc_async_writes_flushed()` before every pop; `noc_async_write_barrier()` at kernel end (BB-03) |
| DRAM alignment on Blackhole (64 B) | a 32 B read of row 0 is below BH's DRAM alignment | `row_fetch_bytes = max(32, ttnn.get_dram_alignment())` passed as one CT arg |
| Reader / writer disagree on column count | hang on `cb_wait_front` / `cb_reserve_back` | both derive `cols_this_core` from identical RT args (`start_t`, `n_t`) and CT `Ht` |
| Reserve per block, push per column | reserving `col_block` pages when fewer columns remain in the last block over-reserves | reserve `min(col_block, cols remaining)` |
| Partial tiles (W = 185, 196, 207) | `floor` geometry drops the last tile row/column | `Wt = ceil(W/32)`, `Ht = ceil(H_out/32)` per image |
| Program cache (BB-07) / trace (BB-16) | rebuilding the descriptor every call, or a key missing `sizes`, replays a wrong program | descriptor cached on `(shape, out_shape, dtype, layout, memory_config, grid)`; only buffer-address RT args change per call; output allocated by the entry point (no allocation inside kernels) |
| Profiling (BB-21) | time outside zones reads as waiting | zones `G6E_FETCH`, `G6E_REPLICATE` (reader), `G6E_WAIT_BCAST`, `G6E_WRITE`, `G6E_DRAIN` (writer), compiled only in the zone build; host can dump per-core role (all cores: reader+writer, same role) |
| Single-tile outputs | `total_tiles = 1` → one core | correct by construction; the split decides, no shape cap |

### Structural impossibilities

None beyond `feature_spec.py`'s `INVALID = []`.
