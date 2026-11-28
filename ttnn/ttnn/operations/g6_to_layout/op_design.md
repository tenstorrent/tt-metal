# Operation Design: g6_to_layout

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no arithmetic). Perf 1: every TILE ↔ ROW_MAJOR reorder of more than one tile column runs on the Tensix — R1 (bf16) and R3/R4 (bfp8) through an untilize compute kernel, R5 (bf16) and R6 (bfp8 out) through a tilize compute kernel. A bf16 tensor crosses the Tensix through **UInt16 CBs** (the same bytes as 16-bit integers), the only datapath that keeps all 65536 bf16 patterns. Only the one-tile-column regimes R2/R7 stay reader + writer only |
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
| `cb_tile_window` (R1) | `tile_block_depth` (blocks) | `2` | Perf 1: R1 runs R3's pipeline — the reader loads block `k+1`'s live face prefixes while compute untilizes block `k` |
| `cb_row_block` (R1) | `row_depth` (blocks) | `2` | reader reorders block `k+1` while the writer drains block `k` |
| `cb_left_faces` (R2) | `row_depth` (blocks) | `2` | reader loads block `k+1` while the writer drains block `k` |
| `cb_tile_window` (R3) | `tile_block_depth` (blocks) | `2` | reader loads block `k+1`'s bfp8 tiles while compute untilizes block `k` |
| `cb_row_block` (R3/R4) | `row_depth` (blocks); R4: `min(row_depth, blocks per shard)` | `2` (R4 production: 1) | compute untilizes block `k+1` while the writer drains block `k`; a shard that is one block gains nothing from a second page |
| `cb_nan_marks` (R3/R4) | `nan_marks_depth = tile_block_depth + row_depth`; R4: `min(·, blocks per shard)` | `4` (R4 production: 1) | the reader may run `tile_block_depth + row_depth` blocks ahead of the writer (its marks are produced with its tiles), so it never stalls on a marks page before compute does on the tile window |
| `cb_row_stage` (R5) | `tile_block_depth` (blocks) | `2` | Perf 1: R5 runs R6's pipeline (rows land as 32-row tile-rows, compute tilizes) — the reader lands block `k+1` while compute tilizes block `k`. R6's barrier per block beat a trid ring of 2/3/4 tile-rows (c38 32.2 vs 41.2 µs, c30 10.3 vs 13.9) |
| `cb_tile_block` (R5/R6) | `row_depth` (blocks) | `2` | the reorder (R5) or tilize (R6) fills block `k+1` while the writer drains block `k` |
| `cb_row_stage` (R6) | `tile_block_depth` (blocks) | `2` | the reader lands block `k+1`'s rows while compute tilizes block `k` |
| — | `NOC_CHECKERBOARD` (reader/writer NoCs swapped on cores with odd logical x + y; R3/R4/R5/R6, and R1 at ≤ `STAGED_CHECKERBOARD_MAX_CORE_TILES = 4` tiles per core) | on | Perf 1: spreads the grid's all-to-all row/tile traffic over both NoCs (c37 6.50 → 4.77 µs, c296 58.0 → 38.3, c301 102 → 90, c2006 3.45 → 3.15). R1 above 4 tiles per core keeps NoC0/NoC1 (c283 362 → 412, c1998 7.54 → 7.82 with it); R2/R7 keep it (Refinement 3: NoC0 run writes 1497 vs 866 µs); x-only, y-only and writer-only swaps measured slower |
| — | `pipeline_blocks_per_core` (sets `block_rows_tiles`, see Perf lamps) | `16` | bounds pipeline fill/drain to ~1/17 of a core's run — the exposed cost when the writer is issue-bound (R2: one write per 6-B row) |
| `cb_left_faces` (R7) | `slot_faces_depth` (blocks) = `slot_read_depth_max + 2` | `3` | besides the blocks in flight, the reader builds its rows of block `k` while the writer still holds block `k-1`'s faces page; at one page less the two halves of consecutive blocks serialize (Refinement 3: 308 1267 → 1359 µs) |
| — | `slot_read_depth` (R7: blocks whose face reads stay on the wire, one trid each, across the reader's build and its wait on the oldest) | `1` | Refinement 4, 308 (µs): 1 → 1095, 2 → 1144, 3 → 1152, 4 → 1151 (bit-exact each). More reads queued per core raise the fast cores' share and slow the starved top rows further |
| — | `slot_read_depth_max` (R7: adaptive read-ahead — after its build, the reader issues more blocks only while the oldest has not landed) | `1` (= `slot_read_depth`: off) | Refinement 4, 308: max 2 → 1147, max 4 → 1173 µs. Deeper queues on the starved cores alone do not raise their share either |
| — | `slot_read_stagger` (R7 `bank_stagger`: each block's first read goes to input bank `(x + y) % banks` of the core) | `False` | Refinement 4, 308: 1110 vs 1109 µs. Each core's tile range already opens on its own bank (1191.6 tiles per core is not a multiple of 8) |
| `cb_slot_image` (R7) | `slot_image_depth` (blocks) | `3` | block `k` is built while block `k-1`'s runs leave L1 and block `k+1`'s reader half starts (at `2`: 308 1391 µs) |
| — | `reader_slot_eighths` (R7: share of each block's rows the reader builds) | `5` | the reader's RISC copies run while its next block's reads are in flight; the writer's share runs as NoC copies. Measured on 308 at 128 blocks/core: 3 → 1190, 4 → 1148, **5 → 1113**, 6 → 1197 µs |
| — | `writer_slot_noc_copies` (R7) | `True` | the writer's share as self-aimed NoC reads, which do not contend with the reader's RISC loads/stores for L1 the way RISC copies on both RISCs do |
| — | `slot_pipeline_blocks_per_core` (R7's `pipeline_blocks_per_core`) | `128` | see Perf lamps: 308 / 286 µs at 16 → 1242 / 289, 64 → 1144 / 275, **128 → 1126 / 269**, 256 → 1132 / 273, 600 → 1121 / 318 |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| L1 budget (model contract): CBs + L1-output pages on a core ≤ 1347776 B | `block_rows_tiles · block_cols_tiles` | R1: `max_block_tiles = floor((1347776 − out_l1_share − 2048·tile_window_depth·read_batch_tiles) / (row_depth · tile_row_stage_rows · 64))`; R2: `max_block_tiles = floor((1347776 − out_l1_share) / (row_depth · 1024))`; R7: `floor((1347776 − out_l1_share) / (slot_faces_depth · 1024 + slot_image_depth · tile_row_stage_rows · out_slot_bytes))`; R3: `floor((1347776 − out_l1_share − 16·nan_marks_depth) / (tile_block_depth·in_tile_bytes + row_depth·2048 + nan_marks_depth·136))`; R4: as R3 without the tile-window term (the shard is the caller's L1, outside the op's budget); R5: `floor((1347776 − out_l1_share) / (row_window_depth · tile_row_stage_rows · 64 + row_depth · 2048))` (the row window scales with `block_cols_tiles` only, so charging it per tile bounds it for every `block_rows_tiles ≥ 1`); R6: `floor((1347776 − out_l1_share) / (tile_block_depth · 2048 + row_depth · out_tile_bytes))`; `out_l1_share = ceil(out_pages / num_l1_banks) · out_aligned_page_bytes · banks_per_core` for an L1 output, else 0 (a TILE output: `out_pages = tensor_tiles`, page = `out_tile_bytes`). Host refuses (BB-05) only if `max_block_tiles < 1` — unreachable for every golden INPUT | launch failure or overwrite of the model's L1 tensors (golden guard pages) |
| BH NoC alignment, DRAM reads compare `local % 64 == src % 64` (`noc_parameters.h:379`, `noc-alignment-in-tt-metal.md` §1) | tile-slot offsets in `cb_tile_window` / `cb_left_faces` | slot strides 2048 B (R1) / 1024 B (R2); face bases at multiples of 512 B in both source and slot; CB bases are 64-aligned (`program.cpp:2505-2509` allocates CBs at DRAM alignment) | silent corruption (watcher stop under `--dev`) |
| NoC writes / L1 traffic compare mod 16 (`noc_parameters.h:375-380`) | `cb_row_block` row stride, segment start | row stride `block_cols_tiles · 64`, segment start `c_lo · 64`, face rows at 32-B multiples — all ≡ 0 mod 16 | silent corruption |
| Stateful single-packet local copy: `transfer_size ≤ NOC_MAX_BURST_SIZE` (`local_copy_helpers_dataflow.inl:66`; BH 16 KB, `noc_parameters.h:289-290`) | reorder copy quantum | fixed 32 B (one face row) | compile error |
| Row-segment writes may exceed one NoC packet (16 KB) | `block_cols_tiles · 64` B per write | use multi-packet `noc_async_write` (`dataflow_api.h:876`) for segments, not the one-packet stateful form | truncated rows |
| Transaction ids 1…`NOC_MAX_TRANSACTION_ID` = 15 (`noc_parameters.h:15`); per-id outstanding counter ≤ `NOC_MAX_TRANSACTION_ID_COUNT` = 255 | `tile_window_depth`; reorder copies in flight per id | `tile_window_depth ≤ 13`: two ids alternate per tile over the reorder copies (64 per full tile; one id per batch would hold 512), each waited before reuse; `2·read_batch_tiles ≤ 255` DRAM reads per batch id. R5: ids `1…row_window_depth` tag the tile-row loads (≤ 32 row reads each), the reader's copies alternate `row_window_depth + 1, + 2`, the writer's (split) `row_window_depth + 3, + 4`; host `assert` | trid aliasing → a wait retires the wrong batch → data read before it lands (BB-06); counter overflow → a wait returns early |
| CB ring wrap (`ttnn-cb-memory-fundamentals.md`, Ring-Wrap Invariant) | push/pop quantum of `cb_row_block`, `cb_left_faces` | page = one whole nominal block; every block (ragged head/tail/last included) pushes and pops exactly 1 page | pointer walks past `fifo_limit` into neighbouring L1 |

### Regimes

Named boundary for "minimum": **each live input byte crosses the input's memory boundary (DRAM or L1 bank) once at 32-B face-row granularity, each output byte crosses the output's boundary once.** Face rows are the read quantum: a 32-B face row holding any live value is read whole.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 `staged_reorder`** | **built** (Phase 0; Tensix since Perf 1) | Phase-0 class (TILE bf16 → RM bf16, interleaved in/out) ∧ ¬(`tensor_cols_tiles == 1` ∧ `W ≤ 16`) | R3's tile rectangle (`block_cols_tiles` capped at `SHORT_IMAGE_BLOCK_COLS = 8` when `H < 32`: c294 20.7 → 18.8 µs); `cb_tile_window` / `cb_row_block` hold tile pages as **UInt16** | **minimum** at the boundary: only live face-row prefixes are read into the tile slots (adjacent whole faces merged); one write per (output row, core) carrying the core's live bytes of that row. The reorder is the packer's (unpack → MOVA2D → pack-untilize on UInt16: bit-exact on all 65536 patterns; a Float16_b view alters 509 even with the Src zero flag preserved) — no on-core NoC copies (Phase 0–Refinement 4 issued 64 self-aimed 32-B copies per tile on NCRISC: c2036 46.1 → 26.5 µs) | per block: reader reserve / reads / one barrier / push; one `untilize` call; writer waits per tile-row (cumulative), issues its rows through the bank-cursor stateful issue (`g6_to_layout_row_issue.hpp`), one flush + pop per block |
| **R2 `face_direct`** | **built** (Phase 0) | Phase-0 class ∧ `tensor_cols_tiles == 1` ∧ `W ≤ 16` (every output row is a prefix of one face row) | `block_rows_tiles × 1` tiles, each staged as a 1024-B left-half slot (face 0 at +0, face 2 at +512) | **minimum** in bytes: faces 0 and 2 live prefixes only (faces 1/3 are all padding and never read); one write per output page — the minimum transaction count for a page-per-row output. No on-core copies. Output DRAM cost is set by page slots, not logical bytes: 1 write per 64-B slot holding `2·W` B | per block: one reserve/push + one read barrier (reader), one wait/flush/pop (writer). Intended: accessor once per kernel; one read barrier per block |
| **R3 `packer_untilize`** | **built** (Refinement 1) | `dtype == bfloat8_b` ∧ input interleaved (DRAM or L1). bf16 stays on R1/R2: this path crosses MOVA2D, which zero-substitutes exponent-0 bf16 datums (-0.0 → +0, subnormals → +0; `ckernel_defs.h:20-26`, default Src zero flag `cmath_common.h:99-104`) | the interleaved tile rectangle of R1 (same block walk and `_choose_blocking`, same full-grid tile-range split); `cb_tile_window` holds the block's whole bfp8 tiles at slot `i·block_cols_tiles + j`; compute packs each tile-row as 32 rows of `block_cols_tiles·64` B into `cb_row_block` tile pages, the row stage R1's writer already reads | input: whole bfp8 tiles (1088 B; the unpacker consumes whole tiles), one read per tile, one read barrier per block — above the face-row minimum only on edge tiles. Output: minimum (one write per live row segment, R1's `store_block`). The reorder moves off NCRISC onto the packer. **NaN rendering:** the host decodes a bfp8 datum to NaN (group exponent 0xFF, mantissa bit 6 and a lower bit set) and its fp32 → bf16 cast renders every NaN as 0xFFFF; unpack → SrcA → pack renders those datums ±inf, and no compute routing emits 0xFFFF. So the reader marks each block's NaN datums on `cb_nan_marks` (a per-tile bitmap of 0xFF groups + one 16-bit datum mask per group; tiles without a 0xFF exponent byte are skipped by one unrolled 16-word test) and the writer stores 0xFFFF over them in the row stage before writing. Every other bfp8 datum (±inf, zeros, flushed values) is bit-exact through the packer (loose c29 specials: exact) | per block: one reader reserve/barrier/push + one marks page, one untilize call (init/uninit hoisted once per kernel via `InitUninitMode::Neither`), one writer wait/patch/flush/pop. A bigger block amortizes those handshakes; `pipeline_blocks_per_core` sizes it as in R1 |
| **R7 `same_bank_slot_writes`** | **built** (Refinement 3) | R2's predicate ∧ `2·W < out_slot_bytes` (`out_slot_bytes = round_up(2·W, output buffer alignment)`: 64 B in DRAM, so every DRAM R2 shape; R2 keeps L1 outputs with `2·W` a multiple of 16) | R2's tile-column block, `block_rows_tiles` sized by `slot_pipeline_blocks_per_core = 128` (308: 10 tiles = 320 rows); the block's rows are output pages `[p, p + rows)`, gathered into a **run-major slot image**: run `j` = pages `p + j + k·B` (`B = contiguous_page_stride()`, 8 DRAM banks) = consecutive aligned slots of one bank, `q + 1` slots for `j < m`, `q` after (`q = rows / B`, `m = rows % B`) | input: R2's (faces 0/2 prefixes). Output: **one NoC write per bank run per block** (308: 105 600 writes of 2560 B instead of 4 194 304 of 6 B), writing whole slots, so output bytes rise to `out_slot_bytes` per page (308: 256 MiB for 24 MiB logical; the gap is the page's alignment slack — the output buffer's own memory, never a logical value). On-core: one copy of each row's live prefix into its slot, `reader_slot_eighths` of the rows as RISC word loads/stores on the reader (while its next block's reads are in flight), the rest as self-aimed NoC reads on the writer | per block: reader reserve + read issue + one trid wait (the oldest block in flight) on `cb_left_faces`, reserve/push on `cb_slot_image`; writer waits both, pops the faces page after its copies and the previous image after one `noc_async_writes_flushed`. A bigger block lengthens runs but bursts the writes and lengthens fill/drain: smaller measured faster down to ~3 tiles |
| **R4 `sharded_in_place`** | **built** (Refinement 1) | `memory_layout ∈ {HEIGHT_SHARDED, BLOCK_SHARDED}` ∧ `dtype == bfloat8_b` (bf16 × sharded is in EXCLUSIONS: no production cell, and the packer path is not bit-exact for bf16 -0.0/subnormals) | the whole resident shard's **live** rectangle: `block_cols_tiles = shard_cols_tiles`, `block_rows_tiles = min(shard_rows_tiles, max_block_tiles / block_cols_tiles)` — one pass per shard for every production case; a last shard with fewer live tile-rows (32 of 96 rows in cases 37/48) publishes and untilizes only those, and a shard wholly beyond the tensor runs nothing | input crosses no NoC: `cb_tile_window` is backed on the shard (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy, rebound per call) and the reader only publishes its live tile pages (one `cb_push_back`) and marks NaNs; compute and writer are R3's. Output once to interleaved pages (BLOCK case 20: 8 cores each write the 64-B segment of their tile column of the one page) | one block per shard = one pass; core assignment = the caller's shard grid (HEIGHT: shard k on the k-th core in orientation order; BLOCK: shard (i, j) on core (x0+j, y0+i) for ROW_MAJOR) |
| **R5 `row_major_to_tile`** | **built** (Refinement 2; Tensix since Perf 1) | input `ROW_MAJOR` bf16 ∧ output bf16 (interleaved DRAM/L1 in and out) | R6's walk and CBs (`cb_row_stage` 32-row tile-rows, `cb_tile_block` tile pages), both declared **UInt16** | minimum: one read per live row segment (64-aligned in the stage), one whole-tile write per tile. The reorder is the unpacker's tilize (unpack-tilize → MOVA2D → pack on UInt16: bit-exact on all 65536 patterns; a Float16_b view alters 522) — the face-half NoC copies on both DM RISCs are gone (c49 39.2 → 15.5 µs before the checkerboard). Row reads from an L1 source are token-bucket paced (`L1_ROW_READ_GRID_BYTES_PER_CYCLE = 780`): at the fast issue rate full-grid 512-B L1 reads collapse (c38 28.6 → 37.3 µs) | per block: reader reserve / row reads / one barrier / push (no conditioning: that is R6's bfp8 pass); one `tilize` call per core; writer wait / tile writes / flush / pop |
| **R6 `packer_tilize`** | **built** (Refinement 2) | input `ROW_MAJOR` bf16 ∧ output bfloat8_b | R5's walk; `cb_row_stage` holds the block as 32-row tile-rows (`block_cols_tiles` bf16 tile pages each), the input of `compute_kernel_lib::tilize` | input: one read per live row segment. Output: one whole bfp8 tile write (1088 B) per tile. Before the push the reader **conditions** the block for the packer: (a) every bf16 datum with an all-ones mantissa (0x7F) is lowered to the next value toward zero — the packer rounds such a datum, when it holds its group's exponent, into the next binade and requantizes the whole group one exponent higher (measured: −64.5 → −66 under a step of 1), where the host saturates the mantissa and keeps the exponent (`tech_reports/data_formats/data_formats.md`, Mantissa Rounding); a lowered sub-max datum moves by ≤ ¼ step, so the device stays within one host step; (b) when `W % 16 ≠ 0`, the bytes after a row's last value up to the end of its 16-value group are zeroed, so only live values set a group's exponent | per block: one reader reserve/barrier/condition/push, one writer wait/flush/pop; the compute is one `tilize` call per core (nominal tile-rows of every block), init/uninit once |
| `direct_face_row_writes` | **rejected** — superseded by R1 everywhere a row needs >1 face-row piece, and by R2 where it needs exactly one | — | — | same bytes, but 2 writes of 32 B per live tile row (64 per tile) into 64-B DRAM slots: half-burst writes and a transaction-bound writer (catalog `tile_reorder`: move whole pages, don't scatter faces) | — |
| `single_core` | **rejected** — forbidden; superseded by R1/R2 on the full grid | — | — | — | — |

**Selection function (host, exact, grid-independent):** for a ROW_MAJOR input (TILE output), `"packer_tilize"` if the output dtype is bfloat8_b, else `"row_major_to_tile"`; for a TILE input, `"sharded_in_place"` if the input is sharded, else `"packer_untilize"` if `dtype == bfloat8_b`, else, if `tensor_cols_tiles == 1 and W <= 16`, `"same_bank_slot_writes"` when `2·W < out_slot_bytes` and `"face_direct"` otherwise, else `"staged_reorder"` (`select_regime()` in the program descriptor; every other class is refused by `validate()`). Regime-pinned tests: cases 286 `[1,1024,1024,3]` and 308 → `same_bank_slot_writes` (`face_direct` only for an L1 output with `2·W` a multiple of 16, `test_g6_to_layout_slot_writes.py`); case 273 `[1,64,128,32]` (`tensor_cols_tiles == 1`, `W = 32`) → `staged_reorder` — the boundary pair.

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

R7 `same_bank_slot_writes` (reader: loads `k+1 … k+slot_read_depth` in flight, one trid each, while building its rows of `k`; writer: the other rows of `k`, then the runs):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_left_faces_block(block_idx + slot_read_depth);  // reader: issued before its share of build_slot_image(block_idx); landing waits on block_idx + 1's trid only
    build_slot_image(block_idx);           // rows -> their slots in the run-major image (reader: RISC copies, writer: NoC copies)
    write_bank_runs(block_idx);            // writer: one write per bank run; the previous image is popped after one flush
}
```

R5 `row_major_to_tile` (reader: load + top-face reorder; writer: bottom-face reorder + store) and R6 `packer_tilize` (reader: load + condition; compute: tilize; writer: store):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_rows_block(block_idx);    // one read per live row segment -> cb_row_stage (R5: per tile-row slot, trid-pipelined)
    reorder_block(block_idx);      // R5: 32-B self-aimed copies into cb_tile_block tile slots (reader top faces, writer bottom faces)
    // R6 instead: condition_block_for_bfp8 (reader) + tilize_block (compute: unpack-tilize, pack bfp8 -> cb_tile_block)
    store_tiles_block(block_idx);  // writer: one whole-tile write per tile of the block
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
| `load_left_faces_block` | `block_rows_tiles` tiles of column 0: face 0 rows `[0, min(16, live_rows))` → slot+0, face 2 rows `[0, live_rows−16)` → slot+512 | the reserved `cb_left_faces` page (block `n` at page `n % depth` of the ring) | TensorAccessor once per kernel; one reserve, one trid wait, one push per block (R2: depth-1 ring; R7: `slot_read_depth` blocks in flight) |
| `store_face_rows_block` | for each tile-row `i`, rows `r < live_rows`: one write of `2·W` bytes from `slot_i + (r/16)·512 + (r%16)·32` to `out_page(g+i, r)` | the waited page | one wait/flush/pop per block; one write barrier per kernel |
| `build_slot_image` (R7) | the block's rows in page order; row `i`'s face row (at `32·i` of the faces page when every tile-row is full) → slot `k = i div B` of run `j = i mod B` | the faces page and the image page | reader: one image reserve/push and one RISC fence pair per block; writer: one NoC-copy state set and one read barrier per block |
| `write_bank_runs` (R7) | `min(rows, B)` runs of `q` or `q + 1` slots | the image page | one `noc_async_writes_flushed` (the previous block's runs) and one image pop per block; one write barrier per kernel |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Overlap (fill/drain)** — `block_rows_tiles` is sized by `pipeline_blocks_per_core = 16`, not at its L1 cap | the per-block fixed cost here is one handshake + one flush (cheap), while fill/drain grows with the block and is exposed whenever the writer is the issue-bound stage (R2 at W=3: 32 writes per tile; at the L1 cap case 308 gets ~2 blocks per core and the writer idles for half a block at start) | `pipeline_blocks_per_core ∈ {4, 8, 32}` and the L1-cap coarsest `block_rows_tiles`, on cases 308, 304, 1978 |
| **Grid synchronization** — S1 spreads even a few-tile tensor over `min(G, tensor_tiles)` cores | `[1,196]` (7 tiles), `[1,512]` (16), `[1,32,2560]` (80) give each core 1 tile; launch and per-core start cost may dominate | a floor `min_tiles_per_core ∈ {4, 16}` (fewer cores, more tiles each) on cases 290, 269, 2003 |
| **Read batch / window depth** | 8-tile batches × 2 follow catalog `double_buffer`; with the reorder sharing NCRISC the sweet spot may move | `read_batch_tiles ∈ {4, 16}`, `tile_window_depth ∈ {3, 4}` (catalog `trid_double_issue`) on cases 1978, 304 |
| **Bank stagger** | every core starts at a tile-row whose first input page and first output page sit on the same DRAM bank (`32·k mod 8 == 0`, catalog `bank_stagger`) | rotate each core's starting tile-row / page by `core_idx mod num_dram_banks` on cases 1978, 308 |
| **Reorder issue on NCRISC** (R1) | resolved in Perf 1: R1 and R5 reorder on the Tensix through UInt16 CBs, so no DM RISC issues reorder copies | — |
| **Small-row reads in R5** | c296 `[1,128,256,32]`: 64-B rows, 64.4 µs with per-core reader issue 20 µs mean / 52 µs max — skew, not depth (`row_window_depth = 4` flat) | `bank_stagger` of each core's first row page; batching two tile-rows per load |
| **Write transactions in R2** | 4194304 writes of 6 B (case 308) | **resolved by R7** (Refinement 3), same-session device time: 308 3394.7 → 1119.8 µs, 286 797.1 → 273.0 µs. Floors of R7 on 308 (ablations): reads only 520 µs, slot-run writes only 866 µs (≈310 GB/s; on NoC0 1497 µs, alternating NoCs 854 µs), both without the copies 1034 µs, full 1113 µs. RISC copies on one RISC cost ≈14 cycles/row (store-bound, 5.7 cycles/store; the data cache off changes nothing); split between both RISCs they contend for L1 and gain nothing, so the writer's share runs as NoC copies. Rotating each core's run order: noise (1108–1113 µs) |
| **Read congestion in R7** | per-core read time grades by core row (308: kernel 400 µs on core row y = 11 to 1120 µs on y = 2, with equal work per core); the top rows wait up to 725 µs for read responses, and those rows bound the kernel | **measured in Refinement 4, no lever moves it**: deeper or adaptive trid read rings, `bank_stagger` and the block co-tune are flat or worse (see the knob rows). A probe-only split weighted by measured row speed gives 1075 µs: the top rows still finish at ~1066 µs with 40% less work, so the gradient is NoC0 arbitration order, not idle capacity. 308 moves 402.6 MB of DRAM traffic at ≈368 GB/s, the same combined read + write rate c304 (363 GB/s) and c1978 (375 GB/s) reach through other kernels: the op sits at this board's mixed-traffic DRAM ceiling. Only fewer DRAM bytes would move it, and the 64-B slot fixes the write bytes of a run write |
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
| `layout` ROW_MAJOR (input) | SUPPORTED (Refinement 2) | same scheme, mirrored transfer shapes | R5 `row_major_to_tile` |
| `output_dtype` bfloat8_b (RM bf16 → TILE bf8b) | SUPPORTED (Refinement 2) | scheme-change (packer conversion) | R6 `packer_tilize`: tilize compute + reader-side conditioning (all-ones mantissas lowered, zero-filled padding in groups holding live values) |
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
| Kernels | reader `ttnn.ReaderConfigDescriptor()` (NCRISC, NoC0), writer `ttnn.WriterConfigDescriptor()` (BRISC, NoC1); R3/R4 add the untilize compute kernel, R6 the tilize compute kernel (config hard-coded: HiFi4, `fp32_dest_acc_en = False`, no approx — the signature has no `compute_kernel_config`, and a 16-bit DEST holds every bf16 value exactly) |
| Regime selection | ROW_MAJOR input: `packer_tilize` iff the output is bf8b, else `row_major_to_tile`; TILE input: `sharded_in_place` iff sharded; `packer_untilize` iff bf8b; else `face_direct` iff `tensor_cols_tiles == 1 and W <= 16`, else `staged_reorder` — no grid-dependent regime; regime-pinned tests in the acceptance tests |

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
| `cb_tile_window` (R1) | 0 | 2048 B (one tile page) | `tile_block_depth · block_rows_tiles · block_cols_tiles` | one nominal block per page group, slot `i·block_cols_tiles + j` | **UInt16** (bf16 bytes as integers) | reader | compute | R1 since Perf 1 (R3's CB, bf16 bytes) |
| `cb_row_block` (R1) | 1 | 2048 B (one tile page as the untilize packs it) | `row_depth · block_rows_tiles · block_cols_tiles` | as R3 | **UInt16** | compute | writer | R1 since Perf 1 |
| `cb_left_faces` | 2 | `block_rows_tiles · 1024` B | `row_depth` | spans `tile_row`; `tile_col` pinned to 1 by the R2 predicate | Float16_b (raw) | reader | writer | R2, whole kernel; 1 page per block |

| `cb_tile_window` (R3) | 0 | `in_tile_bytes` (1088 B for bfp8: one whole tile) | `tile_block_depth · block_rows_tiles · block_cols_tiles` | one nominal block per page group, slot `i·block_cols_tiles + j` | input dtype (Bfp8_b), unpacked by compute | reader | compute | R3, whole kernel; every block pushes/pops the nominal block's pages (ring-wrap) |
| `cb_tile_window` (R4) | 0 | `in_tile_bytes` | the shard (globally allocated on the input buffer) | the resident shard; only its live tile pages are pushed | input dtype | reader (publishes) | compute | R4; zero-copy, rebound to the input buffer each call |
| `cb_row_block` (R3/R4) | 1 | 2048 B (one bf16 tile page as the untilize packs it) | `row_depth · block_rows_tiles · block_cols_tiles` (R4: `min(row_depth, blocks per shard)` blocks) | spans `tile_row` (× 32 rows) and `tile_col` (× 64 B); same byte layout as R1's whole-block page | Float16_b | compute | writer | R3/R4; `block_rows_tiles · block_cols_tiles` pages per block (R4: live tile-rows only) |
| `cb_nan_marks` (R3/R4) | 3 | `round_up(block_tiles · 136, 16)` B: per tile an 8-B group bitmap + 64 × 2-B datum masks | `nan_marks_depth` (R4: `min(·, blocks per shard)`) | one page per block | raw bytes (never unpacked) | reader | writer | R3/R4; 1 page per block |

| `cb_row_stage` (R5) | 4 | 2048 B (tile page: 32 rows × `block_cols_tiles` tiles per tile-row) | `tile_block_depth · block_rows_tiles · block_cols_tiles` | as R6 | **UInt16** | reader | compute | R5 since Perf 1 (R6's CB); `cb_tile_block` (5) is UInt16 as well |
| `cb_row_stage` (R6) | 4 | 2048 B (bf16 tile page: 32 rows × `block_cols_tiles` tiles per tile-row) | `tile_block_depth · block_rows_tiles · block_cols_tiles` | spans `tile_row` (× 32 rows) and `tile_col` | Float16_b | reader | compute | R6; the nominal block's pages per block (ring-wrap) |
| `cb_tile_block` (R5/R6) | 5 | `out_tile_bytes` (2048 bf16, 1088 bfp8) | `row_depth · block_rows_tiles · block_cols_tiles` | spans `tile_row` and `tile_col`, slot `i·block_cols_tiles + j` | output dtype | reader (+ writer's bottom faces, R5 split) / compute (R6) | writer | R5/R6; the nominal block's pages per block |

| `cb_left_faces` (R7) | 2 | `block_rows_tiles · 1024` B | `slot_faces_depth` | as R2 | Float16_b (raw) | reader | writer (the reader also reads its own pushed page for its share of the build, before it reserves that page again) | R7; 1 page per block |
| `cb_slot_image` (R7) | 6 | `block_rows_tiles · tile_row_stage_rows · out_slot_bytes` B (one block's run-major image) | `slot_image_depth` | spans `tile_row` (× rows × slot) | Float16_b (raw bytes) | reader (its rows; reserve/push) | writer (adds its rows to the waited page, writes the runs, pops after the next block's flush) | R7; 1 page per block |

R1 allocates indices 0 and 1, R2 only index 2, R3/R4 indices 0, 1 and 3, R5/R6 indices 4 and 5, R7 indices 2 and 6 — the regimes are different programs. Full ledger, sharing decisions and the data-movement budget: `l1_ledger.md`.

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
| 10 | `load_rows_block` (R5/R6) | live row segments of the block's tile-rows | raw reads (TensorAccessor, `get_noc_addr(page, 64·c_lo)`) | input tensor | `cb_row_stage` (R5: one slot per tile-row; R6: the block's pages) | R5: slot pushed to the writer after its loads land; R6: pushed after barrier + conditioning |
| 11 | `reorder_block` (R5) | tile-row slot → `block.cols` tile slots | helper `set_read_state` / `read_with_state` / trid wait (`g6_to_layout_face_copies.hpp`, shared by reader and writer) | `cb_row_stage` slot | `cb_tile_block`, the block's pages | reader pushes the block after its top-face copies retire; the writer finishes the bottom faces before its wait |
| 12 | `tilize_block` (R6) | `block_rows_tiles` tile-rows × `block_cols_tiles` | **helper** `compute_kernel_lib::tilize<block_cols_tiles, cb_row_stage, cb_tile_block>(total tile-rows)` | `cb_row_stage`, waited per tile-row | `cb_tile_block`, `block_cols_tiles` pages per tile-row | input popped, output pushed by the helper |
| 13 | `store_tiles_block` (R5/R6) | the block's live tiles | raw | `cb_tile_block`, wait the block's pages | output tensor (one whole-tile write each) | popped after flush |
| 14 | `build_slot_image` (R7) | the block's rows, split by `reader_slot_eighths` | reader: raw RISC loads/stores; writer: helper `set_read_state` / `read_with_state` (`g6_to_layout_slot_image.hpp`, shared) | `cb_left_faces` page (pushed) | `cb_slot_image` page | reader pushes the image after its rows are fenced; the writer pops the faces page after its copies land |
| 15 | `write_bank_runs` (R7) | `min(rows, B)` bank runs | raw (`TensorAccessor::get_noc_addr` + `noc_async_write`; run lengths match `num_contiguous_pages`, asserted under watcher) | `cb_slot_image`, waited | output tensor | image popped after the next block's flush |

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
| `load_left_faces_block` | helper | `set_read_trid` / `async_read_barrier_with_trid` (`kernel_lib/local_copy_helpers_dataflow.hpp`); adaptive read-ahead polls `ncrisc_noc_read_with_transaction_id_flushed` (no helper has a non-blocking trid check) | — | once per block | — | — | — |
| `store_block`, `store_face_rows_block` | raw_api | `cb_wait_front` / `get_read_ptr` / `cb_pop_front` | `dataflow_api.h:479` / `:346` / `:260` | 1 page per block | `cb_row_block` / `cb_left_faces` | — | page size = block |
| `store_block`, `store_face_rows_block` | raw_api | `noc_async_write(src_l1, dst_noc, bytes)` + `TensorAccessor::get_noc_addr(out_page, 64·c_lo)` | `dataflow_api.h:876`; `tensor_accessor.h:165` | one per live row segment (multi-packet allowed) | — | output | `block_cols_tiles` (segment length) |
| `store_block`, `store_face_rows_block` | raw_api | `noc_async_writes_flushed` / `noc_async_write_barrier` | `dataflow_api.h:1980` / `:1951` | flushed before each pop; barrier once at kernel end (BB-03) | — | — | — |

**Helpers considered and rejected**

| Raw-API operation | Candidate helper | File:line of the mismatch | Concrete reason | What would close the gap |
|-------------------|------------------|---------------------------|-----------------|--------------------------|
| the tile→row reorder (`reorder_block`) | `compute_kernel_lib::untilize<block_width_tiles, in, out>` | — | **adopted in Perf 1 for bf16** (and `compute_kernel_lib::tilize` for R5). The Src zero flag was not the whole loss: with Float16_b CBs, unpack/pack themselves turn -0.0 and subnormals into +0 and NaN into ±inf whatever the flag (509 / 522 of 65536 patterns). Declaring the CBs UInt16 makes the Tensix move integers; `compute_kernel_hw_startup` keeps the flag disabled for UInt16 by itself. Gate: `tests/.../test_g6_to_layout_all_bf16_patterns.py` | — |
| `store_block` | `dataflow_kernel_lib::write_sticks_after_untilize<cb_id>` | `tilize_helpers_dataflow.inl:219-220` (waits `width_in_tiles` tile pages per 32-row tile-row), `:223-232` (pages `start_page + block_row + row`: contiguous across tile-rows), `:236` (`noc_async_write_barrier` per tile-row) | handshake and a **full write barrier per tile-row** instead of per block (on case 304, 596 barriers per core on the writer's critical path); assumes pages contiguous across tile-rows, false inside a block that crosses an image boundary when `H % 32 ≠ 0` (folded `tile_row`); cannot read R2's face-row sources | a block-granular variant: one wait/pop per block, `writes_flushed` instead of a barrier, a per-tile-row page base |
| `load_block` | `dataflow_kernel_lib::read_sticks_for_tilize` | `tilize_helpers_dataflow.hpp:88` | reads row-major sticks; the Phase-0 input is tiles. Candidate for `row_major_to_tile`'s `load_rows_block` | — |
| `reorder_block` (alternative) | `tile_column_dataflow.hpp` | `tile_column_dataflow.hpp:7-25` | writes one tile *column* by RISC-V stores; the reorder moves face *rows* and RISC stores block NoC issue (`noc-alignment-in-tt-metal.md` §5 strategy B) | — |
| — | `compute_kernel_hw_startup` | n/a | no compute kernel in R1/R2 | called once at the top of the R3/R4 untilize kernel and the R6 tilize kernel |

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
| bfp8 packer rounding overflow (R6) | a group whose largest datum has an all-ones bf16 mantissa is requantized one exponent higher by the packer; the host saturates instead, so the device lands up to 1.5 host steps away (Q3: 4 values; `[1,1024,512]`: 316) | reader conditioning lowers all-ones mantissas before the push (R6 row); BH has no packer saturation control (`ALU_ROUNDING_MODE` holds only stochastic-rounding enables) |
| L1 output beside the model's tensors | case 2038 keeps its L1 placement; the budget includes the output's pages | `max_block_tiles` subtracts `out_l1_share`; acceptance test squeezes L1 to 1347776 B with guard pages |
| Partial last column `live_cols ≤ 16` in R1 | faces 1/3 not read; their stage bytes are stale | segment length ends at `2·W`; stale bytes never leave L1 |
| Very wide row (`[1,131072]`) | 8 MB per tile-row | S1 splits it into ~38-tile column spans per core; `block_cols_tiles ≤ core_tiles_max` |

## Host Contract

| Item | Specification |
|------|---------------|
| `validate(input_tensor, layout, *, dtype=None, memory_config=None)` — first line of `g6_to_layout` | (1) argument errors → `ValueError`: `layout` not TILE/ROW_MAJOR or equal to the input layout; rank ∉ {2,3,4}; `dtype == bfloat8_b` with `layout == ROW_MAJOR`. (2) build the scenario dict exactly as `feature_spec.py` / `eval/golden_tests/g6_to_layout/axes.py:classify_call` do (keys `shape, dtype, layout, memory_layout, buffer, shard_spec, target_layout, out_dtype, memory_config`) and run `INPUT_TAGGERS` on `(scenario,)`. (3) SUPPORTED per axis → `UnsupportedAxisValue`. (4) EXCLUSIONS (bf16 × HEIGHT_SHARDED, bf16 × BLOCK_SHARDED) → `ExcludedCell`. (5) placement errors → `ValueError`: `memory_config` sharded; `memory_config is None` while the input is sharded. Every message raised by `validate()` contains `g6_to_layout` (the acceptance test matches on it) |
| `INPUT_TAGGERS` (declaration order) | `dtype`, `layout`, `output_dtype`, `memory_layout`, `alignment`, `rank`, each `tag_<axis>(inputs, axes)` reading `inputs[0]` per the spec: `alignment` = `"w_non_aligned"` if `W % 32`, else `"h_non_aligned"` if `H % 32`, else `"tile_aligned"` |
| SUPPORTED / EXCLUSIONS | Phase 0 + Refinements 1–2: dtype {bf16, bf8b}, layout {TILE, ROW_MAJOR}, output_dtype {bf16, bf8b}, memory_layout {INTERLEAVED, HEIGHT_SHARDED, BLOCK_SHARDED}; EXCLUSIONS = bf16 × sharded (which also covers ROW_MAJOR × sharded: a ROW_MAJOR input is always bf16). A sharded input must be resident in L1 and needs an interleaved `memory_config` for the output |
| Plan report (BB-08) | `select_plan(input_tensor, layout, *, dtype=None, memory_config=None) -> dict` exported from `ttnn.operations.g6_to_layout`, keys at least `regime` (`"staged_reorder"` / `"face_direct"` / `"same_bank_slot_writes"` / `"packer_untilize"` / `"sharded_in_place"` / `"row_major_to_tile"` / `"packer_tilize"`), `shard` (memory layout, grid, shard shape in tiles, orientation; `None` when interleaved), `num_cores`, `block_rows_tiles`, `block_cols_tiles`, `read_batch_tiles`, `tile_window_depth`, `row_depth`, `l1_bytes_per_core`, and per-core roles (reader/writer on every core of `all_cores`). Calls `validate()` and dispatches nothing |
| Program build cache (BB-07) | memoize the descriptor on (shape, input dtype/layout/memory config incl. the shard spec's grid, shard shape and orientation, target layout, output dtype, output memory config, grid); a hit only rebinds buffer addresses in RT args and, for a sharded input, the shard-backed `cb_tile_window` |
| Dispatch (BB-01/02) | `ttnn.allocate_tensor_on_device(shape, out_dtype, layout, device, out_mem_cfg)` then exactly one `ttnn.generic_op([input, output], descriptor)`; no other ttnn op, host transform, or device tensor |
| Zones (BB-21) | `DeviceZoneScopedN` around every named block operation and every blocking wait (`cb_reserve_back`, `cb_wait_front`, trid waits, flush, barriers) in the zone build only |
