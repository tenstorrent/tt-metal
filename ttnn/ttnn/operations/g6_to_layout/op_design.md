# Operation Design: g6_to_layout

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no arithmetic). The bf16 regimes R1/R2 have **no compute kernel** (reader + writer only); the bfloat8_b regimes R3/R4 (Refinement 1) add an untilize compute kernel, because bfp8 → bf16 is a value conversion |
| Goal | Convert a TILE bfloat16 interleaved tensor to ROW_MAJOR bfloat16 interleaved (Phase 0) in one program: every live input byte read once from where it lies, every output byte written once to where the caller asked, bit-identical for every bf16 pattern. Grows to RM→TILE, bfloat8_b in/out and sharded inputs through the deferred regimes below. |
| Math | `y[n, h, w] = x[n, h, w]` for `n < N, h < H, w < W`; `N = prod(shape[:-2])` (1 for rank 2). Input storage: tile page `t = (n·image_rows_tiles + h//32)·tensor_cols_tiles + w//32`, face `f = 2·((h%32)//16) + (w%32)//16`, byte in tile `f·512 + (h%16)·32 + (w%16)·2`. Output storage: page `n·H + h`, byte `2·w` (page = one row, `2·W` B, slot = `round_up(2·W, buffer alignment)`). |
| References | task spec (contract); `eval/golden_tests/g6_to_layout/feature_spec.py`, `helpers.py`; `.claude/references/blocking-model.md`, `l1-footprint-discipline.md`, `noc-alignment-in-tt-metal.md`, `ttnn-cb-memory-fundamentals.md`; `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp`; catalog `ttnn/ttnn/operations/examples/master.md`: `noc_placement`, `double_buffer`, `trid_double_issue`, `tile_reorder`, `width_split`, `bank_stagger`, `split_reader` |

Names used throughout (scope prefix per `blocking-model.md` §3):

| Name | Definition |
|------|-----------|
| `image_rows_tiles` | `ceil(H / 32)` — tile-rows of one image (each image is tile-padded independently) |
| `tensor_rows_tiles` | `N · image_rows_tiles` — the folded tile-row axis |
| `tensor_cols_tiles` | `ceil(W / 32)` |
| `tensor_tiles` | `tensor_rows_tiles · tensor_cols_tiles` = input tile pages |
| `live_rows(g)` | `min(32, H − 32·(g mod image_rows_tiles))` — live rows of tile-row `g` |
| `live_cols(c)` | `min(32, W − 32·c)` — live columns of tile column `c` |
| `out_page(g, r)` | `(g div image_rows_tiles)·H + 32·(g mod image_rows_tiles) + r` — output page of row `r < live_rows(g)` of tile-row `g` |
| `G` | cores of `device.compute_with_storage_grid_size()` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 2–4; Phase 0: bfloat16, TILE, INTERLEAVED in DRAM or L1 | — | buffer address RT; geometry selects the program |
| `layout` (positional) | `ttnn.Layout` | yes | `TILE_LAYOUT` / `ROW_MAJOR_LAYOUT`, ≠ input layout; Phase 0: `ROW_MAJOR_LAYOUT` | — | host |
| `dtype` (kw-only) | `ttnn.DataType` | no | Phase 0: `None` or `bfloat16` | `None` → input dtype (bfloat8_b → ROW_MAJOR gives bfloat16) | host |
| `memory_config` (kw-only) | `ttnn.MemoryConfig` | no | INTERLEAVED, DRAM or L1 | `None` → input's buffer type, INTERLEAVED | host |

No `compute_kernel_config`: nothing is computed. No index axis (no `dim`), so no sign canonicalization.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d0, …, H, W)`, rank 2–4, any `H, W ≥ 1` (production: `H = 1`, `W ∈ {3, 185, 196, 207}`, `W = 131072`) |
| Dtype | bfloat16 (Phase 0) |
| Layout | TILE (Phase 0) |
| Memory | INTERLEAVED, DRAM or L1 (Phase 0); padding content of edge tiles is arbitrary and never read as a value |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` |
| Dtype | `dtype` if given, else input dtype; bfloat8_b input → ROW_MAJOR gives bfloat16 |
| Layout | `layout` (Phase 0: ROW_MAJOR) |
| Memory | INTERLEAVED; buffer type of `memory_config`, else of the input. Allocated once per call by `ttnn.allocate_tensor_on_device`; no other device tensor is ever allocated |

## Blocking Model

Everything below this section realizes these decisions. Semantics: `.claude/references/blocking-model.md`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `tile_row` (folded `n` × tile-rows of `H`, length `tensor_rows_tiles`) | **independent** — output row `h` of image `n` depends only on input row `h` of image `n`. Folding is honest: walking `g` in order enumerates input tile pages in order and output pages in order (`out_page` is contiguous across image boundaries) | `block_rows_tiles` | if `block_cols_tiles == tensor_cols_tiles`: `clamp(ceil(core_full_rows_max / pipeline_blocks_per_core), 1, floor(max_block_tiles / block_cols_tiles))`, where `core_full_rows_max = floor(core_tiles_max / tensor_cols_tiles)`; else `1` (a row wider than one block is cut into column blocks, one tile-row tall) | host `_choose_blocking()` in the program descriptor → one CT value; every CB size and loop bound derived from it | contiguous row-major **tile range** per core (`split_work_to_cores(grid, tensor_tiles, row_wise=True)`): a core owns a run of tile-rows plus a partial head and tail row | knob-turn |
| `tile_col` (tile columns of `W`, length `tensor_cols_tiles`) | **independent** — no value crosses a column; the output page spans the axis but is assembled from disjoint byte ranges | `block_cols_tiles` | `min(tensor_cols_tiles, core_tiles_max, max_block_tiles)` — the whole row when it fits, else the core's whole column span | host `_choose_blocking()` → one CT value | same tile range: when `core_tiles_max < tensor_cols_tiles` a core's range is a column span of one tile-row — the width split of catalog `width_split` falls out of the flattened split | knob-turn |
| `stage_row` (rows of `cb_row_block`, produced by `reorder_block`) | **independent** (inherits `tile_row`) — intermediate data of the reorder stage | `block_rows_tiles · tile_row_stage_rows`, `tile_row_stage_rows = min(32, H)` | as `tile_row` | derived; never restated | the core that read the tiles reorders and writes them — no collector core, no stage runs on one core while others wait | knob-turn |
| `stage_col_bytes` (bytes of a `cb_row_block` row) | **independent** (inherits `tile_col`) | `block_cols_tiles · 64` B | as `tile_col` | derived | as above | knob-turn |

No axis is **dependent** (no reduction, scan or combine), so no split needs a cross-core combine. No axis is **reuse-shared**. **Operand-reuse check** against the chosen split (tile range per core): the single operand `input_tensor` varies along both `tile_row` and `tile_col`, so every core reads disjoint bytes — no broadcast row exists.

Within-tile geometry (faces, face rows) is not a block axis: the tile is the atom of the tile index space. Partial tiles (`live_rows < 32`, `live_cols < 32`) are a *partial-tile* concern handled by the live-footprint rule in `load_block`, not a partial *block*.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_tile_window` (R1) | `tile_window_depth` (batches in flight) × `read_batch_tiles` (tiles per batch) | `2` × `8` | batch `b+1`'s DRAM reads stay in flight while batch `b` is reordered (catalog `double_buffer`: 4–8 reads per barrier, double-buffered; `trid_double_issue`: a trid-scoped wait keeps the next batch on the wire) |
| `cb_row_block` (R1) | `row_depth` (blocks) | `2` | reader reorders block `k+1` while the writer drains block `k` |
| `cb_left_faces` (R2) | `row_depth` (blocks) | `2` | reader loads block `k+1` while the writer drains block `k` |
| `cb_tile_window` (R3) | `tile_block_depth` (blocks) | `2` | reader loads block `k+1`'s bfp8 tiles while compute untilizes block `k` |
| `cb_row_block` (R3/R4) | `row_depth` (blocks); R4: `min(row_depth, blocks per shard)` | `2` (R4 production: 1) | compute untilizes block `k+1` while the writer drains block `k`; a shard that is one block gains nothing from a second page |
| `cb_nan_marks` (R3/R4) | `nan_marks_depth = tile_block_depth + row_depth`; R4: `min(·, blocks per shard)` | `4` (R4 production: 1) | the reader may run `tile_block_depth + row_depth` blocks ahead of the writer (its marks are produced with its tiles), so it never stalls on a marks page before compute does on the tile window |
| — | `pipeline_blocks_per_core` (sets `block_rows_tiles`, see Perf lamps) | `16` | bounds pipeline fill/drain to ~1/17 of a core's run — the exposed cost when the writer is issue-bound (R2: one write per 6-B row) |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| L1 budget (model contract): CBs + L1-output pages on a core ≤ 1347776 B | `block_rows_tiles · block_cols_tiles` | R1: `max_block_tiles = floor((1347776 − out_l1_share − 2048·tile_window_depth·read_batch_tiles) / (row_depth · tile_row_stage_rows · 64))`; R2: `max_block_tiles = floor((1347776 − out_l1_share) / (row_depth · 1024))`; R3: `floor((1347776 − out_l1_share − 16·nan_marks_depth) / (tile_block_depth·in_tile_bytes + row_depth·2048 + nan_marks_depth·136))`; R4: as R3 without the tile-window term (the shard is the caller's L1, outside the op's budget); `out_l1_share = ceil(out_pages / num_l1_banks) · out_aligned_page_bytes · banks_per_core` for an L1 output, else 0. Host refuses (BB-05) only if `max_block_tiles < 1` — unreachable for every golden INPUT | launch failure or overwrite of the model's L1 tensors (golden guard pages) |
| BH NoC alignment, DRAM reads compare `local % 64 == src % 64` (`noc_parameters.h:379`, `noc-alignment-in-tt-metal.md` §1) | tile-slot offsets in `cb_tile_window` / `cb_left_faces` | slot strides 2048 B (R1) / 1024 B (R2); face bases at multiples of 512 B in both source and slot; CB bases are 64-aligned (`program.cpp:2505-2509` allocates CBs at DRAM alignment) | silent corruption (watcher stop under `--dev`) |
| NoC writes / L1 traffic compare mod 16 (`noc_parameters.h:375-380`) | `cb_row_block` row stride, segment start | row stride `block_cols_tiles · 64`, segment start `c_lo · 64`, face rows at 32-B multiples — all ≡ 0 mod 16 | silent corruption |
| Stateful single-packet local copy: `transfer_size ≤ NOC_MAX_BURST_SIZE` (`local_copy_helpers_dataflow.inl:66`; BH 16 KB, `noc_parameters.h:289-290`) | reorder copy quantum | fixed 32 B (one face row) | compile error |
| Row-segment writes may exceed one NoC packet (16 KB) | `block_cols_tiles · 64` B per write | use multi-packet `noc_async_write` (`dataflow_api.h:876`) for segments, not the one-packet stateful form | truncated rows |
| Transaction ids 1…`NOC_MAX_TRANSACTION_ID` = 15 (`noc_parameters.h:15`); per-id outstanding counter ≤ `NOC_MAX_TRANSACTION_ID_COUNT` = 255 | `tile_window_depth`; reorder copies in flight per id | `tile_window_depth ≤ 13`: two ids alternate per tile over the reorder copies (64 per full tile; one id per batch would hold 512), each waited before reuse; `2·read_batch_tiles ≤ 255` DRAM reads per batch id | trid aliasing → a wait retires the wrong batch → data read before it lands (BB-06); counter overflow → a wait returns early |
| CB ring wrap (`ttnn-cb-memory-fundamentals.md`, Ring-Wrap Invariant) | push/pop quantum of `cb_row_block`, `cb_left_faces` | page = one whole nominal block; every block (ragged head/tail/last included) pushes and pops exactly 1 page | pointer walks past `fifo_limit` into neighbouring L1 |

### Regimes

Named boundary for "minimum": **each live input byte crosses the input's memory boundary (DRAM or L1 bank) once at 32-B face-row granularity, each output byte crosses the output's boundary once.** Face rows are the read quantum: a 32-B face row holding any live value is read whole.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 `staged_reorder`** | **built** (Phase 0) | Phase-0 class (TILE bf16 → RM bf16, interleaved in/out) ∧ ¬(`tensor_cols_tiles == 1` ∧ `W ≤ 16`) | `block_rows_tiles × block_cols_tiles` tiles; staged as `block_rows_tiles · tile_row_stage_rows` rows × `block_cols_tiles · 64` B | **minimum** at the boundary: only live face-row prefixes are read (faces 1/3 skipped when `live_cols ≤ 16`, face rows ≥ `live_rows` skipped); one write per (output row, core) carrying the core's live bytes of that row. Added on-core traffic (not a boundary crossing): 2 self-aimed 32-B L1→L1 copies per live tile row (64 per full tile) | per block: reader `cb_reserve_back`/`cb_push_back`, writer `cb_wait_front` + `noc_async_writes_flushed` + `cb_pop_front`. `block_cols_tiles` also sets the write size (one write per page when a block spans the full row). Intended: accessor + local-copy NoC state once per kernel; trid switch + one trid wait per batch; one handshake per block |
| **R2 `face_direct`** | **built** (Phase 0) | Phase-0 class ∧ `tensor_cols_tiles == 1` ∧ `W ≤ 16` (every output row is a prefix of one face row) | `block_rows_tiles × 1` tiles, each staged as a 1024-B left-half slot (face 0 at +0, face 2 at +512) | **minimum** in bytes: faces 0 and 2 live prefixes only (faces 1/3 are all padding and never read); one write per output page — the minimum transaction count for a page-per-row output. No on-core copies. Output DRAM cost is set by page slots, not logical bytes: 1 write per 64-B slot holding `2·W` B | per block: one reserve/push + one read barrier (reader), one wait/flush/pop (writer). Intended: accessor once per kernel; one read barrier per block |
| **R3 `packer_untilize`** | **built** (Refinement 1) | `dtype == bfloat8_b` ∧ input interleaved (DRAM or L1). bf16 stays on R1/R2: this path crosses MOVA2D, which zero-substitutes exponent-0 bf16 datums (-0.0 → +0, subnormals → +0; `ckernel_defs.h:20-26`, default Src zero flag `cmath_common.h:99-104`) | the interleaved tile rectangle of R1 (same block walk and `_choose_blocking`, same full-grid tile-range split); `cb_tile_window` holds the block's whole bfp8 tiles at slot `i·block_cols_tiles + j`; compute packs each tile-row as 32 rows of `block_cols_tiles·64` B into `cb_row_block` tile pages, the row stage R1's writer already reads | input: whole bfp8 tiles (1088 B; the unpacker consumes whole tiles), one read per tile, one read barrier per block — above the face-row minimum only on edge tiles. Output: minimum (one write per live row segment, R1's `store_block`). The reorder moves off NCRISC onto the packer. **NaN rendering:** the host decodes a bfp8 datum to NaN (group exponent 0xFF, mantissa bit 6 and a lower bit set) and its fp32 → bf16 cast renders every NaN as 0xFFFF; unpack → SrcA → pack renders those datums ±inf, and no compute routing emits 0xFFFF. So the reader marks each block's NaN datums on `cb_nan_marks` (a per-tile bitmap of 0xFF groups + one 16-bit datum mask per group; tiles without a 0xFF exponent byte are skipped by one unrolled 16-word test) and the writer stores 0xFFFF over them in the row stage before writing. Every other bfp8 datum (±inf, zeros, flushed values) is bit-exact through the packer (loose c29 specials: exact) | per block: one reader reserve/barrier/push + one marks page, one untilize call (init/uninit hoisted once per kernel via `InitUninitMode::Neither`), one writer wait/patch/flush/pop. A bigger block amortizes those handshakes; `pipeline_blocks_per_core` sizes it as in R1 |
| `same_bank_slot_writes` | **deferred** — structurally more DRAM bytes than R2, so the byte-cheaper R2 is taken: writes `out_aligned_page_bytes` (64 B) per page instead of `2·W` (6 B at W=3, 10.7×), in exchange for `block_rows_tiles·32 / num_dram_banks` pages per write instead of 1. The trade (write transactions vs DRAM bytes) is the open experiment on case 308 | Phase-0 class ∧ `2·W < out_aligned_page_bytes` | as R1/R2 plus a per-bank slot image: pages `p, p+B, p+2B…` (`B = num_dram_banks`) are consecutive aligned slots of bank `p mod B` | output above minimum by `out_aligned_page_bytes / (2·W)`; write transactions ÷ (`32·block_rows_tiles / B`) | larger blocks → longer same-bank runs. Reachable: R2's writer already walks rows in page order; adds a reader-side 6-B copy per row into the slot image (16-B aligned) |
| **R4 `sharded_in_place`** | **built** (Refinement 1) | `memory_layout ∈ {HEIGHT_SHARDED, BLOCK_SHARDED}` ∧ `dtype == bfloat8_b` (bf16 × sharded is in EXCLUSIONS: no production cell, and the packer path is not bit-exact for bf16 -0.0/subnormals) | the whole resident shard's **live** rectangle: `block_cols_tiles = shard_cols_tiles`, `block_rows_tiles = min(shard_rows_tiles, max_block_tiles / block_cols_tiles)` — one pass per shard for every production case; a last shard with fewer live tile-rows (32 of 96 rows in cases 37/48) publishes and untilizes only those, and a shard wholly beyond the tensor runs nothing | input crosses no NoC: `cb_tile_window` is backed on the shard (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy, rebound per call) and the reader only publishes its live tile pages (one `cb_push_back`) and marks NaNs; compute and writer are R3's. Output once to interleaved pages (BLOCK case 20: 8 cores each write the 64-B segment of their tile column of the one page) | one block per shard = one pass; core assignment = the caller's shard grid (HEIGHT: shard k on the k-th core in orientation order; BLOCK: shard (i, j) on core (x0+j, y0+i) for ROW_MAJOR) |
| `row_major_to_tile` | **deferred** — outside Phase 0 SUPPORTED (`layout`) | input `ROW_MAJOR` | mirror of R1: `load_rows_block` reads each live row segment (`block_cols_tiles·64` B, 64-aligned in the stage) into a row stage; `reorder_block` applies the inverse 32-B permutation into tile slots; `store_tiles_block` writes whole tiles | minimum (one read per row segment, one write per tile); bf8b output needs the packer (bfp8 shared exponent per 16-value face row) with **zero-filled padding inside live groups** | as R1. Reachable: same block walk and CB topology with read/write transfer shapes swapped |
| `direct_face_row_writes` | **rejected** — superseded by R1 everywhere a row needs >1 face-row piece, and by R2 where it needs exactly one | — | — | same bytes, but 2 writes of 32 B per live tile row (64 per tile) into 64-B DRAM slots: half-burst writes and a transaction-bound writer (catalog `tile_reorder`: move whole pages, don't scatter faces) | — |
| `single_core` | **rejected** — forbidden; superseded by R1/R2 on the full grid | — | — | — | — |

**Selection function (host, exact, grid-independent):** `"sharded_in_place"` if the input is sharded, else `"packer_untilize"` if `dtype == bfloat8_b`, else `"face_direct"` if `tensor_cols_tiles == 1 and W <= 16`, else `"staged_reorder"` (`select_regime()` in the program descriptor; every other class is refused by `validate()`). Regime-pinned tests: case 286 `[1,1024,1024,3]` → `face_direct`; case 273 `[1,64,128,32]` (`tensor_cols_tiles == 1`, `W = 32`) → `staged_reorder` — the boundary pair.

### Traffic ranking

All candidate splits move **identical bytes** across every tier: the only operand is read once per tile by exactly one core, each output byte is written once, and no split adds cross-core traffic. The ranking is therefore decided by write-transaction count, with occupancy as the stated tiebreak (`blocking-model.md` §4 keeps them separate; this is the trade, made explicit).

| Split | DRAM crossings (in / out) | Write transactions | Occupancy | Verdict |
|-------|---------------------------|--------------------|-----------|---------|
| **S1 flattened row-major tile range** (`split_work_to_cores` over `tensor_tiles`) | 1 / 1 | Σ live rows + at most 2·32 extra segment rows per core (partial head and tail tile-row) | `min(G, tensor_tiles)` cores, balanced to one tile | **chosen** |
| S2 height (whole tile-rows per core) | 1 / 1 | exactly one per output row | `min(G, tensor_rows_tiles)` cores; `[1,131072]`, `[32,5120]`, `[1,32,2560]` run on 1 core; imbalance up to `tensor_cols_tiles` tiles | rejected: within 64 writes/core of S1 on tall tensors, strands the grid on short ones |
| S3 width (column ranges, all rows) | 1 / 1 | rows × column splits (64-B segments at 1 tile per core) | fills only when `tensor_cols_tiles ≥ G` | rejected |
| S4 2-D rectangles | 1 / 1 | rows × column splits | fills only when the factors divide `G` | rejected: dominated by S1 |
| Split a dependent axis + combine | — | — | — | not applicable: no dependent axis |
| Broadcast a reuse-shared operand | — | — | — | not applicable: no reuse-shared operand |

**Stall-shadow check.** (1) The reader waits on a DRAM batch: the reorder of the previous batch fills that wait (batch `b+1` issued before batch `b`'s trid wait). Legal unconditionally — the stages touch disjoint L1 and move bits, no algebra involved. (2) The reader waits in `cb_reserve_back` for a free `cb_row_block` page: the next batch's DRAM reads are already in flight; nothing else is independent. (3) The writer waits in `cb_wait_front`: every write depends on the block it waits for — nothing independent. (4) R2 reader waits on the block's read barrier: the writer, the bottleneck in R2, drains the other page.

### Block schedule

R1 `staged_reorder` (reader realizes the first two, writer the third):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);      // live face prefixes of the block's tiles -> cb_tile_window, batch by batch
    reorder_block(block_idx);   // cb_tile_window -> cb_row_block page (32-B self-aimed copies), batch-interleaved with load_block
    store_block(block_idx);     // cb_row_block page -> one write per live row segment
}
```

R2 `face_direct`:

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_left_faces_block(block_idx);   // faces 0/2 live prefixes -> cb_left_faces page
    store_face_rows_block(block_idx);   // one write of 2·W bytes per live row, straight from its face row
}
```

R3 `packer_untilize` / R4 `sharded_in_place` (reader, compute, writer; the walk is R1's for R3 and the shard's live rectangle for R4):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);       // R3: whole bfp8 tiles -> cb_tile_window (one barrier); R4: the resident shard, one push
    mark_block_nans(block_idx);  // reader: bfp8 NaN datums of the block -> one cb_nan_marks page
    untilize_block(block_idx);   // compute: unpack bfp8 + pack-untilize -> cb_row_block tile pages (bf16 rows)
    store_block(block_idx);      // writer: 0xFFFF over the marked datums, then R1's one write per live row segment
}
```

**Block walk** (identical in reader and writer — implement once in a shared kernel header): the core's tile range `[t0, t0 + n)` splits into row spans `(g, c_lo, c_hi)`. If `block_cols_tiles == tensor_cols_tiles`, a block is a run of ≤ `block_rows_tiles` consecutive **full** spans, and a partial (head/tail) span is its own one-row block. Otherwise every span is cut into blocks of ≤ `block_cols_tiles` columns, one tile-row tall. Every block is a rectangle `[g, g+rows) × [c_lo, c_lo+cols)`; ragged extents are runtime values passed into the same block operation.

| Block operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|-----------------|------------------------|--------------------|-----------------------------------|
| `load_block` | the block's tiles, in row-major order, in batches of `read_batch_tiles` (batches never straddle blocks; the last batch of a block may be short) | `tile_window_depth` batches in `cb_tile_window` | TensorAccessor once per kernel; one trid set + one trid wait per batch; per tile one read per needed face prefix, adjacent full faces merged (interior tile = one 2048-B read; `live_cols ≤ 16` → faces 0,2; `live_rows ≤ 16` → face-row prefixes of faces 0(,1)) |
| `reorder_block` | same tiles → the block's page of `cb_row_block`: face row `fr` of face `f` of tile `(i, j)` goes to stage row `i·tile_row_stage_rows + (f/2)·16 + fr`, byte `j·64 + (f%2)·32`; only face rows of live rows / live halves are copied | the reserved `cb_row_block` page (reserved before the block's first batch is reordered) | local-copy NoC state (self coordinates, 32 B) once per batch (the batch's DRAM reads reprogram the read command buffer); one reorder-trid wait per tile (alternating ids, see Mechanism caps) plus two per batch; one `cb_reserve_back` + `cb_push_back` per block (push only after the block's copies are retired) |
| `store_block` | the block's page: for tile-row `i` of the block, rows `r < live_rows(g+i)`, each one write of `2·(min(W, 32·(c_lo+cols)) − 32·c_lo)` bytes from stage row `i·tile_row_stage_rows + r` to `out_page(g+i, r)` at byte offset `64·c_lo` | the waited page | one `cb_wait_front`, one `noc_async_writes_flushed` (source reuse only — not a full barrier), one `cb_pop_front` per block; one `noc_async_write_barrier` per kernel |
| `load_left_faces_block` | `block_rows_tiles` tiles of column 0: face 0 rows `[0, min(16, live_rows))` → slot+0, face 2 rows `[0, live_rows−16)` → slot+512 | the reserved `cb_left_faces` page | TensorAccessor once per kernel; one reserve, one read barrier, one push per block |
| `store_face_rows_block` | for each tile-row `i`, rows `r < live_rows`: one write of `2·W` bytes from `slot_i + (r/16)·512 + (r%16)·32` to `out_page(g+i, r)` | the waited page | one wait/flush/pop per block; one write barrier per kernel |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Overlap (fill/drain)** — `block_rows_tiles` is sized by `pipeline_blocks_per_core = 16`, not at its L1 cap | the per-block fixed cost here is one handshake + one flush (cheap), while fill/drain grows with the block and is exposed whenever the writer is the issue-bound stage (R2 at W=3: 32 writes per tile; at the L1 cap case 308 gets ~2 blocks per core and the writer idles for half a block at start) | `pipeline_blocks_per_core ∈ {4, 8, 32}` and the L1-cap coarsest `block_rows_tiles`, on cases 308, 304, 1978 |
| **Grid synchronization** — S1 spreads even a few-tile tensor over `min(G, tensor_tiles)` cores | `[1,196]` (7 tiles), `[1,512]` (16), `[1,32,2560]` (80) give each core 1 tile; launch and per-core start cost may dominate | a floor `min_tiles_per_core ∈ {4, 16}` (fewer cores, more tiles each) on cases 290, 269, 2003 |
| **Read batch / window depth** | 8-tile batches × 2 follow catalog `double_buffer`; with the reorder sharing NCRISC the sweet spot may move | `read_batch_tiles ∈ {4, 16}`, `tile_window_depth ∈ {3, 4}` (catalog `trid_double_issue`) on cases 1978, 304 |
| **Bank stagger** | every core starts at a tile-row whose first input page and first output page sit on the same DRAM bank (`32·k mod 8 == 0`, catalog `bank_stagger`) | rotate each core's starting tile-row / page by `core_idx mod num_dram_banks` on cases 1978, 308 |
| **Reorder issue on NCRISC** (R1) | 64 self-aimed copies per tile share NCRISC with the DRAM reads; on wide tensors the reader may become issue-bound | split the reorder copies across both DM RISCs (catalog `split_reader`; the handoff needs its own CB under the ownership rule), or issue copies on NoC1 from the reader, on cases 1978, 304 |
| **Write issue in R2** | 4194304 writes of 6 B (case 308): per-write TensorAccessor bank division and full command setup dominate the writer | incremental page walk (bank `+1`, slot `+aligned_page` every `num_dram_banks` pages) with `noc_async_write_one_packet_set_state`/`_with_state` (`dataflow_api.h:982/1008`), and splitting rows across both DM RISCs, on case 308 |
| **NoC placement** | defaults: cores row-wise (`row_wise=True`), reader NoC0, writer NoC1 (catalog `noc_placement`: 2.9× row vs column, 2.5–4.8× for reads-NoC0/writes-NoC1) | column placement and swapped NoCs only to confirm the default on this op's write-heavy narrow-row cases |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| input tiles in DRAM / L1 banks | bf16 tile pages (2048 B, faces 0–3) | — | page id = flattened tile index `t` (row-major per image, images consecutive) |
| R1 `load_block` | tile-ordered bytes in `cb_tile_window` slots (2048-B slots) | reader (NCRISC, NoC0): `noc_async_read` of live face prefixes via TensorAccessor; batch tagged with trid `1 + (batch mod tile_window_depth)` | padding faces / rows never read; edge-tile padding columns inside a live face row are read but never written out |
| R1 `reorder_block` | row-major rows in a `cb_row_block` page | reader: self-aimed 32-B NoC reads (`local_copy_helpers_dataflow.hpp`), trid `tile_window_depth + 1` | bits move as bytes: bit identity by construction (signed zeros, subnormals, inf, NaN payloads) |
| R1 `store_block` | output RM pages | writer (BRISC, NoC1): one `noc_async_write` per live row segment | segment = the block's live bytes of the row; output padding/alignment gap untouched |
| R2 `load_left_faces_block` | faces 0/2 in 1024-B slots of a `cb_left_faces` page | reader: `noc_async_read` of face prefixes, one barrier per block | — |
| R2 `store_face_rows_block` | output RM pages | writer: one write of `2·W` B per row from its face row | no stage copy |
| Cross-core | — | none | no Tensix-to-Tensix traffic in any built or deferred regime |

### TARGET axes as regimes

| TARGET value | Phase 0 | Cost class | Realization |
|--------------|---------|------------|-------------|
| `dtype` bfloat16 / `output_dtype` bfloat16 / `layout` TILE / `memory_layout` INTERLEAVED / every `alignment` / every `rank` | SUPPORTED | — | R1 + R2 |
| `dtype` bfloat8_b (TILE input → RM bf16) | SUPPORTED (Refinement 1) | scheme-change (adds the compute stage) | `packer_untilize`: unpack bf8b → pack-untilize bf16 between the same two CBs (value conversion, not a byte copy) |
| `layout` ROW_MAJOR (input) | refused | same scheme, mirrored transfer shapes | `row_major_to_tile` |
| `output_dtype` bfloat8_b (RM bf16 → TILE bf8b) | refused | scheme-change (packer conversion) | `row_major_to_tile` + packer bfp8 with zero-filled padding in groups holding live values |
| `memory_layout` HEIGHT_SHARDED | SUPPORTED with bf8b (Refinement 1) | **knob-turn on placement** — the shard cuts `tile_row` (independent): each core's work is self-contained | `sharded_in_place`; production pairs it with bf8b → also `packer_untilize` |
| `memory_layout` BLOCK_SHARDED | SUPPORTED with bf8b (Refinement 1) | **knob-turn on placement** — the shard cuts `tile_row` × `tile_col` (both independent); `[1,1,1,256]` on 8 cores = one tile each, each core writes a 64-B segment of the one output page | `sharded_in_place` (+ `packer_untilize`) |

### Structural impossibilities

None beyond `feature_spec.py`'s INVALID: every TARGET combination is projected from a production scenario, so no impossible cell is generated.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one tile of the flattened `(tile_row, tile_col)` space; blocks are rectangles cut from a core's run (Block walk) |
| Grid | `device.compute_with_storage_grid_size()`; `split_work_to_cores(grid, tensor_tiles, row_wise=True)` → `(num_cores, all_cores, core_group_1, core_group_2, n1, n2)`; cores ordered by `ttnn.grid_to_cores(num_cores, grid.x, grid.y, row_wise=True)` — ≤ 2 rectangular ranges (BB-17) |
| Per-core work | core `k` (in that order) gets `n_k ∈ {n1, n2}` tiles by group membership, `t0_k = Σ_{j<k} n_j`; RT args carry `t0_k`, `n_k` and buffer addresses; everything else is CT and identical on all cores (one binary per role) |
| Remainder | alignment-aware geometry: `image_rows_tiles = ceil(H/32)` **per image**, `tensor_cols_tiles = ceil(W/32)`; never `floor(N·H/32)`; ragged head/tail spans and short last blocks are runtime extents of the same block operation |
| Sharded input (R4) | core assignment = the caller's shard grid (`ShardPlacement.core_rects`); RT args `(tile_row, rows, col, cols)` of the core's live shard rectangle; no `split_work_to_cores` |
| Kernels | reader `ttnn.ReaderConfigDescriptor()` (NCRISC, NoC0), writer `ttnn.WriterConfigDescriptor()` (BRISC, NoC1); R3/R4 add the untilize compute kernel (config hard-coded: HiFi4, `fp32_dest_acc_en = False`, no approx — the signature has no `compute_kernel_config`, and a 16-bit DEST holds every bf16 value exactly) |
| Regime selection | `sharded_in_place` iff sharded; `packer_untilize` iff bf8b; else `face_direct` iff `tensor_cols_tiles == 1 and W <= 16`, else `staged_reorder` — no grid-dependent regime; regime-pinned tests in the acceptance tests |

Phase 0 values for the flagged / extreme cases (illustrated at `G = 110`; the code derives them from the live grid):

| Case | Shape | Regime | `tensor_tiles` | `core_tiles_max` | `block_cols_tiles` | `block_rows_tiles` | CB bytes / core |
|------|-------|--------|----------------|------------------|--------------------|--------------------|-----------------|
| 308 (perf focus) | `[1,2048,2048,3]` | R2 | 131072 | 1192 | 1 | 75 | 153600 |
| 304 | `[1,1024,2048,128]` | R1 | 262144 | 2384 | 4 | 38 | 655360 |
| 1978 | `[16,4096,2560]` | R1 | 163840 | 1490 | 80 | 2 | 688128 |
| 294 | `[1,131072]` | R1 | 4096 | 38 | 38 | 1 | 37632 (`tile_row_stage_rows = 1`) |
| 286 | `[1,1024,1024,3]` | R2 | 32768 | 298 | 1 | 19 | 38912 |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_tile_window` | 0 | 2048 B (one tile slot) | `tile_window_depth · read_batch_tiles` | streams over `tile_row` and `tile_col` (a window, not the block); live = all slots | Float16_b (raw bytes, never unpacked) | reader | reader | R1 reader kernel. Reader-private scratch: addressed from `get_write_ptr` once, no push/pop |
| `cb_row_block` | 1 | `block_rows_tiles · tile_row_stage_rows · block_cols_tiles · 64` B (one whole nominal block) | `row_depth` | spans `tile_row` (`block_rows_tiles`) and `tile_col` (`block_cols_tiles`) | Float16_b (raw) | reader | writer | R1, whole kernel; 1 page per block |
| `cb_left_faces` | 2 | `block_rows_tiles · 1024` B | `row_depth` | spans `tile_row`; `tile_col` pinned to 1 by the R2 predicate | Float16_b (raw) | reader | writer | R2, whole kernel; 1 page per block |

| `cb_tile_window` (R3) | 0 | `in_tile_bytes` (1088 B for bfp8: one whole tile) | `tile_block_depth · block_rows_tiles · block_cols_tiles` | one nominal block per page group, slot `i·block_cols_tiles + j` | input dtype (Bfp8_b), unpacked by compute | reader | compute | R3, whole kernel; every block pushes/pops the nominal block's pages (ring-wrap) |
| `cb_tile_window` (R4) | 0 | `in_tile_bytes` | the shard (globally allocated on the input buffer) | the resident shard; only its live tile pages are pushed | input dtype | reader (publishes) | compute | R4; zero-copy, rebound to the input buffer each call |
| `cb_row_block` (R3/R4) | 1 | 2048 B (one bf16 tile page as the untilize packs it) | `row_depth · block_rows_tiles · block_cols_tiles` (R4: `min(row_depth, blocks per shard)` blocks) | spans `tile_row` (× 32 rows) and `tile_col` (× 64 B); same byte layout as R1's whole-block page | Float16_b | compute | writer | R3/R4; `block_rows_tiles · block_cols_tiles` pages per block (R4: live tile-rows only) |
| `cb_nan_marks` (R3/R4) | 3 | `round_up(block_tiles · 136, 16)` B: per tile an 8-B group bitmap + 64 × 2-B datum masks | `nan_marks_depth` (R4: `min(·, blocks per shard)`) | one page per block | raw bytes (never unpacked) | reader | writer | R3/R4; 1 page per block |

R1 allocates indices 0 and 1, R2 only index 2, R3/R4 indices 0, 1 and 3 — the regimes are different programs. Full ledger, sharing decisions and the data-movement budget: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` (R1) | `rows × cols` tiles in batches of `read_batch_tiles` | trid helpers (`local_copy_helpers_dataflow.hpp`) + raw reads | input tensor (TensorAccessor) | `cb_tile_window`, batch slots `(batch mod tile_window_depth)·read_batch_tiles + k` | slots of the next batch in flight while the current one is reordered |
| 2 | `reorder_block` (R1) | same tiles → `rows·tile_row_stage_rows` stage rows × `cols·64` B | helper `set_read_state` / `read_with_state` / trid wait | `cb_tile_window` (private, landed batch) | `cb_row_block`, 1 page (reserved per block, pushed after the block's last copy retires) | page handed to writer |
| 3 | `store_block` (R1) | stage rows of live tile rows × live segment bytes | raw (helper gap, see API Mapping) | `cb_row_block`, wait 1 | output tensor (TensorAccessor) | page popped after `noc_async_writes_flushed` |
| 4 | `load_left_faces_block` (R2) | `rows × 1` tiles | raw | input tensor | `cb_left_faces`, 1 page | pushed after one read barrier |
| 5 | `store_face_rows_block` (R2) | live rows of `rows` tile-rows, `2·W` B each | raw | `cb_left_faces`, wait 1 | output tensor | popped after flush |
| 6 | `load_block` (R3) / shard publish (R4) | the block's whole bfp8 tiles | raw reads (R3); none (R4) | input tensor (R3) / resident shard (R4) | `cb_tile_window`, block pages | pushed after one read barrier (R3) / once for the shard's live pages (R4) |
| 7 | `mark_block_nans` (R3/R4) | the block's bfp8 exponent bytes (+ datums of 0xFF groups) | raw RISC loads (`g6_to_layout_bfp8_nan_marks.hpp`) | `cb_tile_window` pages already pushed (read-only; their slots are refilled only by this reader, after the scan) | `cb_nan_marks`, 1 page | page handed to writer |
| 8 | `untilize_block` (R3/R4) | `moved_tile_rows × block_cols_tiles` tiles → 32 rows each | **helper** `compute_kernel_lib::untilize<block_cols_tiles, cb_tile_window, cb_row_block, InitUninitMode::Neither, WaitMode::WaitBlock>` | `cb_tile_window`, waited per tile-row | `cb_row_block`, `block_cols_tiles` pages per tile-row | input popped, output pushed by the helper |
| 9 | `store_block` (R3/R4) | R1's `store_block` on tile pages | raw | `cb_row_block` (wait the block's pages) + `cb_nan_marks` (wait 1) | output tensor | both popped after the patch / flush |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_block`, `load_left_faces_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | `page_id = t`, `offset = face_base` | — | — | none |
| `load_block`, `load_left_faces_block` | raw_api | `noc_async_read(src_noc, dst_l1, bytes)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | bytes = live prefix of a face (merged when contiguous) | input | `cb_tile_window` / `cb_left_faces` | — |
| `load_block` | helper | `dataflow_kernel_lib::set_read_trid(noc, trid)` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:147` (impl `.inl:110`) | trid `1 + (batch mod tile_window_depth)`; reset to 0 at kernel end | — | — | `tile_window_depth` |
| `load_block`, `reorder_block` | helper | `dataflow_kernel_lib::async_read_barrier_with_trid(noc, trid)` | `local_copy_helpers_dataflow.hpp:158` (impl `.inl:112-114`) | wait the batch about to be reordered; wait the reorder trid before push | — | — | — |
| `reorder_block` | helper | `dataflow_kernel_lib::set_read_state<32>(noc, src_addr)` | `local_copy_helpers_dataflow.hpp:74-75` (impl `.inl:62-70`) | once per kernel (self coordinates + 32-B size) | — | — | none (32 B is the face-row quantum) |
| `reorder_block` | helper | `dataflow_kernel_lib::read_with_state(noc, dst_addr, src_addr)` | `local_copy_helpers_dataflow.hpp:94-95` (impl `.inl:72-77` → `noc_async_read_one_packet_with_state`, `dataflow_api.h:599`, which takes per-call src and dst) | one per live face row | `cb_tile_window` | `cb_row_block` | — |
| `reorder_block`, `load_left_faces_block` | raw_api | `cb_reserve_back` / `cb_push_back` / `get_write_ptr` | `dataflow_api.h:406` / `:208` / `:325` | 1 page per block | — | `cb_row_block` / `cb_left_faces` | page size = block |
| `load_left_faces_block` | raw_api | `noc_async_read_barrier` | `dataflow_api.h:1921` | once per block | — | — | — |
| `store_block`, `store_face_rows_block` | raw_api | `cb_wait_front` / `get_read_ptr` / `cb_pop_front` | `dataflow_api.h:479` / `:346` / `:260` | 1 page per block | `cb_row_block` / `cb_left_faces` | — | page size = block |
| `store_block`, `store_face_rows_block` | raw_api | `noc_async_write(src_l1, dst_noc, bytes)` + `TensorAccessor::get_noc_addr(out_page, 64·c_lo)` | `dataflow_api.h:876`; `tensor_accessor.h:165` | one per live row segment (multi-packet allowed) | — | output | `block_cols_tiles` (segment length) |
| `store_block`, `store_face_rows_block` | raw_api | `noc_async_writes_flushed` / `noc_async_write_barrier` | `dataflow_api.h:1980` / `:1951` | flushed before each pop; barrier once at kernel end (BB-03) | — | — | — |

**Helpers considered and rejected**

| Raw-API operation | Candidate helper | File:line of the mismatch | Concrete reason | What would close the gap |
|-------------------|------------------|---------------------------|-----------------|--------------------------|
| the tile→row reorder (`reorder_block`) | `compute_kernel_lib::untilize<block_width_tiles, in, out>` | `untilize_helpers.hpp:152-163` → `pack_untilize.h:54-62` / `llk_math_fast_untilize.h:60-73`; flush rule `ckernel_defs.h:20-26`, default flag `cmath_common.h:99-104`, `llk_math_common.h:57-59`; preserve set only by `tile_move_copy.h:69` | the value path crosses MOVA2D with the Src zero flag at its Float16_b default, which zero-substitutes exponent-0 datums: -0.0 and subnormals do not survive — the spec grades bit identity on exactly these patterns. Rejected for bf16 only; R3/R4 use it for bfp8 (a value conversion anyway) | an untilize init that pins `_configure_preserve_zero_flag_state_()` plus an all-65536-pattern device probe |
| `store_block` | `dataflow_kernel_lib::write_sticks_after_untilize<cb_id>` | `tilize_helpers_dataflow.inl:219-220` (waits `width_in_tiles` tile pages per 32-row tile-row), `:223-232` (pages `start_page + block_row + row`: contiguous across tile-rows), `:236` (`noc_async_write_barrier` per tile-row) | handshake and a **full write barrier per tile-row** instead of per block (on case 304, 596 barriers per core on the writer's critical path); assumes pages contiguous across tile-rows, false inside a block that crosses an image boundary when `H % 32 ≠ 0` (folded `tile_row`); cannot read R2's face-row sources | a block-granular variant: one wait/pop per block, `writes_flushed` instead of a barrier, a per-tile-row page base |
| `load_block` | `dataflow_kernel_lib::read_sticks_for_tilize` | `tilize_helpers_dataflow.hpp:88` | reads row-major sticks; the Phase-0 input is tiles. Candidate for `row_major_to_tile`'s `load_rows_block` | — |
| `reorder_block` (alternative) | `tile_column_dataflow.hpp` | `tile_column_dataflow.hpp:7-25` | writes one tile *column* by RISC-V stores; the reorder moves face *rows* and RISC stores block NoC issue (`noc-alignment-in-tt-metal.md` §5 strategy B) | — |
| — | `compute_kernel_hw_startup` | n/a | no compute kernel in R1/R2 | called once at the top of the R3/R4 untilize kernel |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Bit identity on special patterns | the golden suite injects -0.0, subnormals, ±inf, NaN 0x7FC1 into production shapes and grades by bits | bf16: no value ever enters the Tensix coprocessor (R1/R2 move bytes with the NoC only), and bf16 × sharded is EXCLUDED rather than routed through the packer. bf8b: the packer's ±inf for a NaN datum is overwritten with the host's 0xFFFF from the reader's marks; all other bfp8 datums decode bit-exactly |
| Stale RISC data-cache lines (BH) | the shard is written by the host and the marks page by another RISC; the writer's patch stores must reach L1 before the NoC reads the stage | `invalidate_l1_cache()` before the reader scans a shard, before the writer reads a marks page, and after the patch stores |
| Ragged last shard (cases 37/48: 32 live rows of 96) | untilizing the shard's dead tile-rows would touch L1 beyond the live rows | R4 moves `block.rows` (live) tile-rows through `cb_tile_window` / `cb_row_block`, not the nominal block (`moved_tile_rows<sharded>`) |
| Poisoned input padding | the model's producer leaves anything in tile padding; host-made tensors hold zeros, so a leak passes randn tests | reads skip padding faces/rows; segments end at the last live byte (`2·W`), so padding columns copied with a live face row are never written; acceptance test poisons padding with 1000.0 and NaN |
| Image boundaries inside a block (`H % 32 ≠ 0`, rank 3/4) | after an image's last tile-row the next tile-row starts `live_rows(g)` pages later, not 32 | every tile-row computes `live_rows(g)` and `out_page(g, r)` from `g`; never `page_base + 32·i` |
| `H < 32` stage size | `tile_row_stage_rows = min(32, H)`; using 32 there over-allocates and using `live_rows` per tile-row mis-strides | one CT value, used by both reorder (row index) and writer (row stride) |
| Ring wrap on ragged blocks | head/tail spans and short last blocks | page = whole nominal block; push/pop exactly 1 page every block |
| Trid hazard | a reorder-trid wait must not drain the next batch, and a window slot must not be overwritten while its reorder copies are in flight | per batch: issue `b+1` reads → wait trid of `b` → issue reorder of `b` → wait reorder trid; slots of `b+1` were freed by the previous iteration's reorder wait |
| Writer pops before data leaves L1 | popping lets the reader overwrite the page | `noc_async_writes_flushed()` before every `cb_pop_front`; write barrier at kernel end (BB-03) |
| L1 output beside the model's tensors | case 2038 keeps its L1 placement; the budget includes the output's pages | `max_block_tiles` subtracts `out_l1_share`; acceptance test squeezes L1 to 1347776 B with guard pages |
| Partial last column `live_cols ≤ 16` in R1 | faces 1/3 not read; their stage bytes are stale | segment length ends at `2·W`; stale bytes never leave L1 |
| Very wide row (`[1,131072]`) | 8 MB per tile-row | S1 splits it into ~38-tile column spans per core; `block_cols_tiles ≤ core_tiles_max` |

## Host Contract

| Item | Specification |
|------|---------------|
| `validate(input_tensor, layout, *, dtype=None, memory_config=None)` — first line of `g6_to_layout` | (1) argument errors → `ValueError`: `layout` not TILE/ROW_MAJOR or equal to the input layout; rank ∉ {2,3,4}; `dtype == bfloat8_b` with `layout == ROW_MAJOR`. (2) build the scenario dict exactly as `feature_spec.py` / `eval/golden_tests/g6_to_layout/axes.py:classify_call` do (keys `shape, dtype, layout, memory_layout, buffer, shard_spec, target_layout, out_dtype, memory_config`) and run `INPUT_TAGGERS` on `(scenario,)`. (3) SUPPORTED per axis → `UnsupportedAxisValue`. (4) EXCLUSIONS (bf16 × HEIGHT_SHARDED, bf16 × BLOCK_SHARDED) → `ExcludedCell`. (5) placement errors → `ValueError`: `memory_config` sharded; `memory_config is None` while the input is sharded. Every message raised by `validate()` contains `g6_to_layout` (the acceptance test matches on it) |
| `INPUT_TAGGERS` (declaration order) | `dtype`, `layout`, `output_dtype`, `memory_layout`, `alignment`, `rank`, each `tag_<axis>(inputs, axes)` reading `inputs[0]` per the spec: `alignment` = `"w_non_aligned"` if `W % 32`, else `"h_non_aligned"` if `H % 32`, else `"tile_aligned"` |
| SUPPORTED / EXCLUSIONS | Phase 0 + Refinement 1: dtype {bf16, bf8b}, memory_layout {INTERLEAVED, HEIGHT_SHARDED, BLOCK_SHARDED}; EXCLUSIONS = bf16 × sharded. A sharded input must be resident in L1 and needs an interleaved `memory_config` for the output |
| Plan report (BB-08) | `select_plan(input_tensor, layout, *, dtype=None, memory_config=None) -> dict` exported from `ttnn.operations.g6_to_layout`, keys at least `regime` (`"staged_reorder"` / `"face_direct"` / `"packer_untilize"` / `"sharded_in_place"`), `shard` (memory layout, grid, shard shape in tiles, orientation; `None` when interleaved), `num_cores`, `block_rows_tiles`, `block_cols_tiles`, `read_batch_tiles`, `tile_window_depth`, `row_depth`, `l1_bytes_per_core`, and per-core roles (reader/writer on every core of `all_cores`). Calls `validate()` and dispatches nothing |
| Program build cache (BB-07) | memoize the descriptor on (shape, input dtype/layout/memory config incl. the shard spec's grid, shard shape and orientation, target layout, output dtype, output memory config, grid); a hit only rebinds buffer addresses in RT args and, for a sharded input, the shard-backed `cb_tile_window` |
| Dispatch (BB-01/02) | `ttnn.allocate_tensor_on_device(shape, out_dtype, layout, device, out_mem_cfg)` then exactly one `ttnn.generic_op([input, output], descriptor)`; no other ttnn op, host transform, or device tensor |
| Zones (BB-21) | `DeviceZoneScopedN` around every named block operation and every blocking wait (`cb_reserve_back`, `cb_wait_front`, trid waits, flush, barriers) in the zone build only |
