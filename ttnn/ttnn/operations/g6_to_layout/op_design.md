# Operation Design: g6_to_layout

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (layout conversion; the only compute stage is a bit-identity untilize) |
| Goal | Return `input_tensor`'s logical values in the other layout (Phase 0: TILE → ROW_MAJOR, bfloat16, interleaved DRAM/L1 in and out) in **one device program** that reads each live input burst once and writes each output row once. No staging tensor, no second DRAM pass, no host round trip. |
| Math | `y[n, h, w] = x[n, h, w]` for `n < N, h < H, w < W`; `shape(y) = shape(x)`; bfloat16 output is bit-identical (signed zeros, subnormals, inf, NaN payloads). |
| Storage orders | TILE input page `(n·Ht + ht)·Wt + wt`, face `2·((h%32)//16) + (w%32)//16`, row `h%16`, col `w%16` inside the face. ROW_MAJOR output page `n·H + h` holding `W` values. |
| References | `eval/golden_tests/g6_to_layout/{feature_spec.py, helpers.py}`, `eval/op_template.py`, `.claude/references/{blocking-model.md, l1-footprint-discipline.md, noc-alignment-in-tt-metal.md}`, `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.{hpp,inl}`, `tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_defs.h:19-43` (MOVA2D denormal flush), catalog `ttnn/ttnn/operations/examples/{double_buffer, width_split, noc_placement, tile_reorder, bank_stagger, trid_double_issue, split_reader, compute_block_size}` |

Catalog entries and the knob each one informed:

| Catalog entry | Knob it informed |
|---------------|------------------|
| `width_split` | core-assignment: column groups (`num_col_groups`) when tile-rows under-fill the grid |
| `double_buffer` | buffer depth 2 on both CBs; all reads of a block issued before one barrier |
| `tile_reorder` | a full (interior) tile is one 2 KB read, never four face reads |
| `noc_placement` | `split_work_to_cores(..., row_wise=True)`; reader NoC0, writer NoC1 |
| `compute_block_size` | one `untilize` helper call per kernel (init/uninit once), not per block |
| `bank_stagger`, `trid_double_issue`, `split_reader` | perf lamps L4, L9, L5 |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 2–4; Phase 0: bfloat16, TILE, INTERLEAVED in DRAM or L1 | — | buffer address RT |
| `layout` (positional) | `ttnn.Layout` | yes | `ROW_MAJOR_LAYOUT` or `TILE_LAYOUT`, ≠ `input_tensor.layout`; Phase 0: `ROW_MAJOR_LAYOUT` | — | host only (selects kernels) |
| `dtype` (kw-only) | `ttnn.DataType \| None` | no | `bfloat16`, `bfloat8_b` (bf8b only with `TILE_LAYOUT` target); Phase 0: `bfloat16` or None | None → input dtype, except bf8b → ROW_MAJOR gives bf16 | host only |
| `memory_config` (kw-only) | `ttnn.MemoryConfig \| None` | no | INTERLEAVED, DRAM or L1 | None → input's memory config | host only (output placement; output accessor CT args) |

### Registry contract (op file)

| Item | Decision |
|------|----------|
| Scenario dict | `validate()` builds `{"shape", "dtype", "layout", "memory_layout", "buffer", "shard_spec", "target_layout", "out_dtype", "memory_config"}` from `input_tensor` and the keywords exactly as `eval/golden_tests/g6_to_layout/axes.py:classify_call` does, then applies `INPUT_TAGGERS` to `(scenario,)`. |
| `tag_dtype` | `scenario["dtype"]` |
| `tag_layout` | `scenario["layout"]` |
| `tag_output_dtype` | `out_dtype` if given; else bf16 when dtype is bf8b and target is ROW_MAJOR; else dtype |
| `tag_memory_layout` | `scenario["memory_layout"]` |
| `tag_alignment` | `W % 32 != 0` → `"w_non_aligned"`; else `H % 32 != 0` → `"h_non_aligned"`; else `"tile_aligned"` (H = shape[-2], W = shape[-1]) |
| `tag_rank` | `len(shape)` |
| `SUPPORTED` (Phase 0) | dtype `[bf16]`, layout `[TILE]`, output_dtype `[bf16]`, memory_layout `[INTERLEAVED]`, alignment all three, rank `[2, 3, 4]` (implementer writes the block; listed here so the plan and validate agree) |
| `EXCLUSIONS` | `[]` — no production cell is ever excluded |
| `validate()` order | (1) structural `ValueError`s: `layout` not RM/TILE or equal to input layout; rank ∉ {2,3,4}; `dtype == bf8b` with RM target. (2) `SUPPORTED` per-axis → `UnsupportedAxisValue`, then `EXCLUSIONS` → `ExcludedCell`. (3) placement `ValueError`s: `memory_config` sharded; `memory_config is None` while input is sharded. |
| Error messages | each `ValueError` names what it refuses — the word `layout`, `bfloat8_b` (or `dtype`), `shard`, or `rank` (the acceptance test matches them case-insensitively) |
| Entry point | `validate()` is its first line; then output dtype/placement resolution, `plan()`, output allocation, one `ttnn.generic_op`. No other ttnn op runs on device. |
| Structural impossibilities | None beyond the two INVALID entries already in `feature_spec.py`. |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d0, …, H, W)`, rank 2–4; `N = ∏ leading dims` (1 for rank 2); any H ≥ 1, W ≥ 1 |
| Dtype | bfloat16 (Phase 0) |
| Layout | TILE (Phase 0); tile padding content arbitrary — never reaches the output |
| Memory | INTERLEAVED, DRAM or L1 (Phase 0) |

### Output

| Property | Value |
|----------|-------|
| Shape | `shape(input_tensor)` |
| Dtype | `dtype` kw, else input dtype (bf8b → RM gives bf16) |
| Layout | `layout` (Phase 0: ROW_MAJOR; page = one row of `W·2` B, stride `aligned_page_size = round_up(W·2, buffer alignment)`) |
| Memory | INTERLEAVED, buffer type of `memory_config`, else of `input_tensor` |

## Blocking Model

Pure data movement. No output value depends on more than one input value, so **no axis is dependent**, and the op has one input, so **no operand is reuse-shared**. Every axis is independent; the design question is entirely transfer shape and fan-out.

Scopes: `tensor_tile_rows = N·ceil(H/32)` (per-image ceil), `tensor_tile_cols = ceil(W/32)`; core scope: the core's unit range `[unit_begin, unit_end)`; block scope: `block_height_tiles × block_width_tiles`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `n` (folded leading dims) | independent — each image's values move alone | folded into `tile_row`; extent via `block_height_tiles` | folded (a block may span images; each tile-row carries its own `n`) | `plan()` | with `tile_row` | knob-turn |
| `ht` (tile-rows in an image, `ceil(H/32)`) | independent — rows move alone | folded into `tile_row`; extent via `block_height_tiles` | folded | `plan()` | with `tile_row` | knob-turn |
| `tile_row` (= `n·Ht + ht`, contiguous in input page order) | independent | `block_height_tiles` | `ceil(core_rows_max / ceil(core_rows_max / floor(block_tiles_cap / block_width_tiles)))` — coarsest equal split of the core's run under the L1 cap | `plan()` → CT arg to reader, compute, writer | contiguous unit runs: `split_work_to_cores(grid, tensor_tile_rows·num_col_groups, row_wise=True)`, units ordered column-group-major | knob-turn |
| `wt` (tile columns, `ceil(W/32)`) | independent — no reduction along W | `block_width_tiles` (column sub-block) and `num_col_groups` / `col_group_width_tiles` (cross-core cut) | `block_width_tiles = ceil(cgw / ceil(cgw / block_tiles_cap))` with `cgw = col_group_width_tiles`; `num_col_groups` by the occupancy rule in Work Distribution | `plan()` → CT args | column groups across cores only when tile-rows alone do not balance the grid within `COL_SPLIT_TOLERANCE` | knob-turn |
| `row_in_tile` (32 rows of a tile-row; live ≤ 32 on an image's last `ht`) | independent | whole tile height (32) — not split | 32 | TILE format | not assigned across cores: the tile is the input page; cutting it would turn one read into 32-B face-row reads for no byte saving | none needed |
| `col_in_tile` (32 columns of a tile; live ≤ 32 in the last `wt`) | independent | whole tile width (32) — not split | 32 | TILE format | not assigned across cores: a column cut inside a tile breaks 64-B alignment of output segments (Mechanism caps) | none needed |

**Live-extent narrowing (not a block knob, a read policy of `load_live_tiles`).** For tile `(tile_row, wt)` with `live_rows = min(32, H − 32·ht)` and `live_cols = min(32, W − 32·wt)`: faces 0/1 hold rows 0–15, faces 2/3 rows 16–31, faces 0/2 cols 0–15, faces 1/3 cols 16–31. The reader reads only live faces, and of each live face only its live-row prefix (`rows × 32 B`, contiguous at the face start); a full tile (`live_rows = live_cols = 32`) is one 2 KB read; adjacent full faces merge into one read. Unread positions of `cb_tiles_in` hold stale bytes; they map only to padding positions of `cb_rows_out`, which the writer never writes.

**Intermediate data (re-run of the table).** The only intermediate is the untilized block in `cb_rows_out` (32·`block_height_tiles` rows × `block_width_tiles`·32 values). Its axes are the block's own (`tile_row`, `row_in_tile`, `wt`): already assigned to the core that produced them, no stage collects to a designated core, and there is no stage that computes on one core while others wait.

**Stall-shadow check.** Stages that wait: compute on `cb_tiles_in`, writer on `cb_rows_out`, reader on its read barrier, writer on its write flush. No arithmetic exists, so there is no algebraic reorder to make; the shadows are filled by the three-kernel pipeline at depth 2 (the reader fetches block k+1 while compute/writer drain block k). Deferring a barrier across blocks (trids, lazy write flush) is perf lamp L9.

**Every knob is a parameter.** `TILES_IN_DEPTH`, `ROWS_OUT_DEPTH`, `L1_OP_BUDGET_BYTES = 1347776`, `COL_SPLIT_TOLERANCE` are module constants in the program-descriptor file; the grid is `device.compute_with_storage_grid_size()` passed into `plan()`; `num_col_groups`, `col_group_width_tiles`, `block_height_tiles`, `block_width_tiles`, per-core unit ranges and block counts are outputs of `plan()`, and every CB size, CT arg and RT arg is derived from them. No kernel contains a literal standing in for any of them.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_tiles_in` | `TILES_IN_DEPTH` | 2 | reader issues block k+1's reads while compute untilizes block k |
| `cb_rows_out` | `ROWS_OUT_DEPTH` | 2 | compute untilizes block k+1 while the writer's NoC writes for block k drain |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| L1 budget (BB-05): CBs + this core's L1-output pages ≤ `L1_OP_BUDGET_BYTES` | `block_height_tiles · block_width_tiles ≤ block_tiles_cap = floor((L1_OP_BUDGET_BYTES − out_l1_bytes_per_bank) / ((TILES_IN_DEPTH + ROWS_OUT_DEPTH) · tile_bytes))`, `tile_bytes = 2048` | `plan()` derives both extents under the cap; if `block_tiles_cap < 1` the host refuses with `RuntimeError` (only possible for an L1 output of > ~1.3 MB/bank — no golden shape) | launch fails with CB/L1 clash, only when the model holds its L1 |
| Core's own assignment | `block_height_tiles ≤ core_rows_max`, block rows ≤ the run left in the column group; `block_width_tiles ≤ col_group_width_tiles` | runtime extent `min(block_height_tiles, tensor_tile_rows − tile_row, unit_end − unit)`; `min(block_width_tiles, col_group_end − col)` | a block reads another core's tiles / another column group |
| NoC alignment: DRAM reads match mod 64, writes mod 16 (`noc-alignment-in-tt-metal.md` §1) | column cuts (column groups, column sub-blocks) | only at whole-tile boundaries (64 B of bf16 per tile row); face-segment reads start at face starts (`k·512` B into the tile) | silent data corruption on DRAM reads / writes |
| Output page stride `round_up(W·2, alignment)` | output row addressing | always `TensorAccessor::get_noc_addr(page, byte_offset)` (`tensor_accessor.h:165`), never `page·W·2` | rows land in the wrong slots for W·2 not a multiple of 64 (W = 3, 185, 196, 207) |
| `pack_untilize` sub-block ≤ DEST limit (8 tiles, 16-bit half-sync) | none on `block_width_tiles` — the helper takes the largest divisor ≤ `DEST_AUTO_LIMIT` (`untilize_helpers.inl:62-87`) | none needed for correctness | widths with no divisor in [2, 8] (prime > 8) run 1-tile sub-blocks — slower, correct (lamp L7) |
| Single-packet stateful NoC write ≤ `NOC_MAX_BURST_SIZE` | per-row write size if the writer uses `noc_async_write_one_packet_*` | rows wider than one burst use `noc_async_write` | wrong size programmed into the write state |
| bf16 through SrcA: MOVA2D flushes any datum with zero exponent (`ckernel_defs.h:19-31`) | not an extent cap — a format cap | both CBs are declared **UInt16** (opaque 16-bit: `cunpack_common.h:476-482`, `cpack_common.h:269-270`); the flush is disabled for UInt16 (`requires_disabled_src_zero_flag`, applied at hw configure `llk_math_common.h:57-59`) | subnormals and −0.0 become +0.0; random inputs pass, the specials loose cases fail |

### Regimes

The minimum is stated against DRAM/L1-bank boundary of the interleaved tensors: **each live input burst (32-B face row) crosses once, each output row's live bytes cross once.**

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| R1 `tilerow_colgroup_untilize` | **built** | `layout == TILE ∧ memory_layout == INTERLEAVED ∧ dtype == output_dtype == bf16` (all of Phase 0: 56 of 80 production cases, every flagged perf case) | `block_height_tiles × block_width_tiles` | **minimum** on the input (live face-row prefixes only; dead faces and dead rows never read; padding columns inside a live face row are read because a face row is one 32-B burst). Output: each live row written once **per column group** — `num_col_groups` NoC writes per row, same bytes; `num_col_groups = 1` whenever the tile-rows balance the grid within `COL_SPLIT_TOLERANCE`. Cross-core: none. | Per block: one reader barrier, one CB reserve/push (reader) and wait/pop (writer), one writer flush. Untilize init/uninit is **once per kernel**, not per block. Pipeline fill/drain once per kernel. |
| R2 `dm_untilize` (writer gathers face rows, no compute kernel) | deferred — R1 is exact by ISA (UInt16 is opaque at unpack, MOVA2D and pack: citations in Mechanism caps) and moves the 64 face-row copies per tile off the DM RISCs, which already carry the NoC issue load (the writer is the bottleneck on narrow rows). R2 becomes the built regime only if the Phase 0 precision gate (Key Risks #1) refutes the ISA claim. | same as R1 | same as R1 | same as R1 | same as R1; plus the DM-side copy cost per tile |
| R3 `single_core` | rejected — Phase 0 is multi-core by construction; superseded by R1 at every shape | — | — | minimum bytes, 1/C occupancy | — |
| R4 `row_split_only` (`num_col_groups ≡ 1`) | rejected — it is R1 at one knob value; forced, it strands one-tile-row shapes (`[1,131072]`, `[32,5120]`, `[1,32,2560]`) on one core | — | — | minimum | — |
| R5 `column_split_only` (`num_col_groups ≡ tensor_tile_cols`) | rejected — superseded by R1's occupancy rule; forced, it writes every row as `tensor_tile_cols` 64-B segments | — | — | ×`Wt` write transactions | — |
| R6 dependent-axis split + cross-core combine | rejected — there is no dependent axis; nothing to combine | — | — | — | — |
| R7 operand broadcast (read once + mcast) | rejected — single input; no core needs bytes another core reads | — | — | — | — |
| R8 `tilize_rm_to_tile` | deferred — `layout = ROW_MAJOR` input is outside Phase 0 SUPPORTED (refinement). Reachable: same `plan()`, same unit order and block walk; reader reads RM row segments into `cb_rows_in` (row slot stride `block_width_tiles·64` B), compute `tilize` helper, writer writes live tiles. bf8b output: compute packs Bfp8_b and the reader zero-fills each live row's tail `[W, 32·tensor_tile_cols)` of the last tile column (a face-row group of 16 shares one exponent). | `layout == ROW_MAJOR ∧ memory_layout == INTERLEAVED` | `block_height_tiles × block_width_tiles` | minimum (each input row segment once, each output tile once) | same terms as R1 |
| R9 `sharded_in_place_untilize` | deferred — HEIGHT/BLOCK_SHARDED inputs (all three are bf8b) are outside Phase 0 SUPPORTED. Reachable: core assignment and per-core extent are the caller's shard; the block is the core's whole **live** shard (live tile-rows × shard tile columns; the last HEIGHT shard has 1 live tile-row of 3); `cb_tiles_in` is backed on the shard buffer (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy, no reader NoC traffic); compute and writer are R1's (the writer's page/offset math takes the shard's row and column origin). | `memory_layout ∈ {HEIGHT_SHARDED, BLOCK_SHARDED}` | one block = live shard (`live_shard_tile_rows × shard_tile_cols`) | input: zero NoC bytes (resident); output: minimum | one block per core: one wait/pop, one flush |

**Selection.** Phase 0 builds R1 only; `validate()` refuses every cell outside R1's predicate before a plan is made. When R8/R9 are built, the predicate column is the selection function, evaluated on host in `plan()`, and each regime gets pinned tests (case ids 1989/277 for R8, 20/37/48 for R9).

### Traffic ranking

Counted per tensor in crossings of the DRAM (or L1-bank) boundary, then NoC transactions; no core-to-core traffic exists in any candidate.

| Candidate split | Input crossings | Output crossings | Transfer shape | Occupancy | Verdict |
|-----------------|-----------------|------------------|----------------|-----------|---------|
| (a) `tile_row` only | 1× live bursts | 1×; one write per row | whole rows (`W·2` B) | fails when `tensor_tile_rows < C` (29 of the 49 Phase 0 INPUTS entries at `C = 130`) | cheapest bytes and fewest writes; incomplete |
| (b) `wt` only | 1× | 1×; `Wt` writes per row | 64-B segments | fills on wide shapes only | same bytes, most write transactions; rejected (R5) |
| (c) `tile_row × column groups`, `num_col_groups` by the occupancy rule (R1) | 1× | 1×; `num_col_groups` writes per row | `col_group_width_tiles·64` B segments | fills the grid on every shape with ≥ C tiles | **chosen** — equals (a) whenever (a) balances; fragments rows only as far as occupancy requires |
| (d) cut `row_in_tile` across cores | 1× bytes, ×16 read transactions (face-row reads) | 1× | 32-B reads | no gain over (c) | rejected |
| (e) full-tile reads (no live-extent narrowing) under (c) | up to 2× on W ≤ 16 shapes (`[1,2048,2048,3]`: 256 MiB vs 128 MiB), 32× on H = 1 shapes (`[1,131072]`: 8 MiB vs 256 KiB) | 1× | 2 KB reads | same | rejected — live-extent reads are part of `load_live_tiles` |
| (f) dependent split + combine / broadcast | n/a — no dependent axis, no shared operand | | | | n/a |

Cheapest-traffic split is (a); (c) is (a) wherever (a) fills the grid, so the implemented split moves the minimum bytes on every shape and adds write transactions only on the few-tile-row shapes, where it is the only way to occupy the grid.

### Block schedule

Logical schedule; reader, compute and writer realize their parts asynchronously, pipelined at depth 2. `block_idx` walks the core's unit range: a row-chunk is `min(block_height_tiles, tensor_tile_rows − tile_row, unit_end − unit)` tile-rows of one column group; each row-chunk is cut into column sub-blocks of `block_width_tiles`.

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_live_tiles(block_idx);    // reader
    untilize_block(block_idx);     // compute
    store_live_rows(block_idx);    // writer
}
```

| Block operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|-----------------|------------------------|--------------------|-----------------------------------|
| `load_live_tiles` | `block_rows × block_cols` live tiles of a nominal `block_height_tiles × block_width_tiles` slot; per tile only live face-row prefixes; a full tile is one read | nothing (streams) | **one** `cb_reserve_back` / read barrier / `cb_push_back` of `block_height_tiles·block_width_tiles` pages per block; reads of the whole block in flight together; TensorAccessor built once per kernel |
| `untilize_block` | `block_height_tiles` tile-rows × `block_width_tiles` tiles (nominal; stale pages of a ragged block are untilized and never written) | DEST state only | untilize init/uninit **once per kernel** (one helper call over `num_blocks_this_core · block_height_tiles` tile-rows); the helper's internal wait/pop/push quantum is one tile-row (`block_width_tiles` pages) |
| `store_live_rows` | for each live tile-row of the block and each of its `live_rows`: one NoC write of the block's live column bytes `min(block_cols·32, W − 32·col)·2` B from cb row `(32·t + r)·block_width_tiles·64` to `get_noc_addr(n·H + 32·ht + r, 64·col)` | nothing (streams) | **one** `cb_wait_front` / `noc_async_writes_flushed` / `cb_pop_front` of `block_height_tiles·block_width_tiles` pages per block; `noc_async_write_barrier` once at kernel end (BB-03) |

No multicast, no semaphores, no cross-core barrier.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 overlap | `[1,4096,2560]` (most calls) gives each core one 80-tile block: read → untilize → write serialize on that core | cut the core's row into 2–4 column sub-blocks (`block_width_tiles = 20…40`) so the pipeline overlaps |
| L2 grid synchronization | one-tile-row shapes (`[1,32,2560]`, `[1,185]`, `[32,5120]`, `[1,64,1536]`) give 1–2 tiles per core at maximum participation; launch/arg fixed cost may dominate | cap participation so each core gets ≥ 4–8 tiles |
| L3 column-group fragmentation | `COL_SPLIT_TOLERANCE = 0.10` trades balance against row fragmentation without pricing a write transaction | tolerance 0.0 / 0.25 on `[16,128,2560]`, `[8,128,2560]`, `[1,3872,2560]` |
| L4 bank stagger | `Wt = 80` is a multiple of the DRAM bank count, so every core's block starts on the same bank | rotate each core's per-block issue order by `core_idx % num_banks` (catalog `bank_stagger`) |
| L5 narrow-row write issue | `W·2 ≤ 64` B (`W = 3`: `[1,2048,2048,3]`, `[1,1024,1024,3]`) means one NoC write per 6-B row; the writer is issue-bound (~32 writes per tile) | (i) bank-grouped stateful writes (`noc_async_write_one_packet_set_state` per bank, `_with_state` per row, `dataflow_api.h:982-1008`), posted writes; (ii) split the row writes across both DM RISCs (catalog `split_reader`) — needs a second output CB so each CB keeps one consumer |
| L6 compute stage on `live_cols ≤ 16` | on `W ≤ 16` every output row is one contiguous span of a face row already in `cb_tiles_in`; untilize adds nothing | writer sources rows straight from face rows of `cb_tiles_in`; drop `cb_rows_out` for those shapes (frees L1 for a larger block) |
| L7 helper sub-block | `block_width_tiles` with no divisor in [2, 8] (e.g. 35 on `[1,131072]`) runs 1-tile pack sub-blocks | round `block_width_tiles` to a multiple of 8 (or 4) within the cap |
| L8 placement | `row_wise=True` (catalog `noc_placement`) is the default; BH grids are asymmetric | column-wise / diagonal placement |
| L9 barrier granularity | one drain-all read barrier per block and one write flush per block idle the NoC at block edges when blocks are small | trid-scoped read barriers (catalog `trid_double_issue`); flush block k only after issuing block k+1's writes |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM/L1 → `cb_tiles_in` | TILE, bf16 bytes declared UInt16 | reader (NCRISC, NoC0): `TensorAccessor` page `tile_row·Wt + wt`, `noc_async_read` of live face-row prefixes; one barrier per block | input page = 2048 B, page-aligned in its bank |
| `cb_tiles_in` → `cb_rows_out` | UInt16 → UInt16 (bit identity) | compute: `compute_kernel_lib::untilize<block_width_tiles, cb_tiles_in, cb_rows_out>` (standard `pack_untilize`; the BH fast path is Float16_b-only, `untilize_helpers.inl:23-50`) | `fp32_dest_acc_en = false` (16-bit DEST) |
| `cb_rows_out` → DRAM/L1 | ROW_MAJOR rows | writer (BRISC, NoC1): one `noc_async_write` per live row per block, `get_noc_addr(page, 64·col)` | output page stride handled by the accessor |

TARGET axes as physical realizations of the logical split (none crosses a dependent axis, so none is a scheme-change):

| TARGET value | What it cuts / changes | Class |
|--------------|------------------------|-------|
| `memory_layout = HEIGHT_SHARDED` | caller pins the `tile_row` cut (independent) and pre-places it in L1 | knob-turn of placement: `cb_tiles_in` backed on the shard, reader removed (R9) |
| `memory_layout = BLOCK_SHARDED` | caller pins `tile_row` and `wt` cuts (both independent) | knob-turn of placement (R9); writer takes a column origin |
| `layout = ROW_MAJOR` (RM → TILE) | reverses the data path; same split, same block walk | new reader/compute/writer on the same blocking (R8), not a scheme-change |
| `dtype = bfloat8_b` (TILE in) | `cb_tiles_in` Bfp8_b (tile page 1088 B), `cb_rows_out` Float16_b; unpack converts | format change; precision gate in Key Risks #10 |
| `output_dtype = bfloat8_b` (RM in) | packer converts Float16_b → Bfp8_b in R8; zero-filled row tail | format change |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | `(column group j, tile_row)`; unit id `u = j·tensor_tile_rows + tile_row` (column-group-major, so a core's run is a vertical strip and stacks into `block_height_tiles`-tall blocks) |
| Grid | `grid = device.compute_with_storage_grid_size()`, `C = grid.x·grid.y`; cores from `ttnn.split_work_to_cores(grid, num_units, row_wise=True)` → ≤ 2 rectangular core ranges (BB-17), one kernel per role on `all_cores` |
| Column groups | `LB = ceil(tensor_tile_rows·tensor_tile_cols / C)`. For `g = 1, 2, …, tensor_tile_cols`: `cgw = ceil(tensor_tile_cols / g)`, `g_eff = ceil(tensor_tile_cols / cgw)`, `cost = ceil(tensor_tile_rows·g_eff / C)·cgw`. Take the **smallest** `g` with `cost ≤ (1 + COL_SPLIT_TOLERANCE)·LB` (always reached at `g = tensor_tile_cols`). `num_col_groups = g_eff`, `col_group_width_tiles = cgw`; the last group is `tensor_tile_cols − (g_eff − 1)·cgw` wide. |
| Per-core work | `num_units = tensor_tile_rows·num_col_groups`; group 1 gets `ceil(num_units / num_cores)` units, group 2 one fewer; `core_rows_max = min(ceil(num_units / num_cores), tensor_tile_rows)` |
| Block extents | `block_tiles_cap` (Mechanism caps); `block_width_tiles = ceil(cgw / ceil(cgw / block_tiles_cap))`; `cap_rows = floor(block_tiles_cap / block_width_tiles)`; `block_height_tiles = ceil(core_rows_max / ceil(core_rows_max / cap_rows))` |
| Per-core block count | host walks each core's unit range with the device's walk (row-chunks per column group × column sub-blocks) → `num_blocks_this_core` (compute RT arg); reader/writer get `unit_begin, unit_end` |
| Remainder | ragged in rows (end of a column group's run, end of the core's range) and in columns (last column group, last sub-block): runtime extents shrink, CB push/pop counts stay nominal. Tile geometry per image: `Ht = ceil(H/32)`, `live_rows = min(32, H − 32·ht)`, `Wt = ceil(W/32)`, `live_cols = min(32, W − 32·wt)`; never `floor(N·H/32)` |
| Plan report (BB-08) | `plan(input_tensor, layout, *, dtype=None, memory_config=None, grid=None)` in the op module returns `{regime, num_cores, num_col_groups, col_group_width_tiles, block_height_tiles, block_width_tiles, units_per_core (g1, g2)}`; the entry point dispatches from exactly this dict |
| Program cache (BB-07) | `plan()` and the program descriptor are memoized on `(shape, dtype, layout, input memory config, output dtype, output memory config, grid)`; a repeat call patches only buffer addresses |

Illustration at `C = 130`, DRAM output (`block_tiles_cap = 164`); `plan()` recomputes for the device's grid:

| Case | Shape | tile-rows × Wt | `num_col_groups` × cgw | cores | block (h × w) | blocks/core |
|------|-------|----------------|------------------------|-------|---------------|-------------|
| 1978 | [16,4096,2560] | 2048 × 80 | 1 × 80 | 130 | 2 × 80 | 8 |
| 2031 | [1,4096,2560] | 128 × 80 | 1 × 80 | 128 | 1 × 80 | 1 (lamp L1) |
| 308 (p1) | [1,2048,2048,3] | 131072 × 1 | 1 × 1 | 130 | 145 × 1 | 7 |
| 304 (p2/p3) | [1,1024,2048,128] | 65536 × 4 | 1 × 4 | 130 | 39 × 4 | 13 |
| 286 (p4) | [1,1024,1024,3] | 32768 × 1 | 1 × 1 | 130 | 127 × 1 | 2 |
| 1995 | [16,128,2560] | 64 × 80 | 2 × 40 | 128 | 1 × 40 | 1 |
| 1993 | [16,1,2560] | 16 × 80 | 8 × 10 | 128 | 1 × 10 | 1 |
| 293 | [1,1,131072] | 1 × 4096 | 118 × 35 | 118 | 1 × 35 | 1 |
| 292 | [32,5120] | 1 × 160 | 80 × 2 | 80 | 1 × 2 | 1 |
| 2003 | [1,32,2560] | 1 × 80 | 80 × 1 | 80 | 1 × 1 | 1 (lamp L2) |

Single built regime in Phase 0; no regime-selection function is needed until R8/R9 land (then: Regimes → Selection).

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_tiles_in` | 0 | `tile_bytes` = 2048 | `TILES_IN_DEPTH · block_height_tiles · block_width_tiles` | spans `tile_row` (`block_height_tiles`) and `wt` (`block_width_tiles`) of one block × depth; streams over blocks | UInt16 (bf16 bytes, opaque) | reader | compute | whole kernel |
| `cb_rows_out` | 1 | `tile_bytes` = 2048 (helper emits tile-sized pages, `untilize_helpers.hpp:130-131`) | `ROWS_OUT_DEPTH · block_height_tiles · block_width_tiles` | same spans; row `(32·t + r)` of block-local tile-row `t` at byte `(32·t + r)·block_width_tiles·64` | UInt16 | compute | writer | whole kernel |

CB sync: reader pushes `block_height_tiles·block_width_tiles` per block; compute waits/pops `block_width_tiles` per tile-row (`block_height_tiles` times per block) and pushes `block_width_tiles` per tile-row; writer waits/pops `block_height_tiles·block_width_tiles` per block. Totals match per block; every quantum divides the capacity, so the ring closes. A ragged block keeps the nominal counts.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_live_tiles` | `block_rows × block_cols` live tiles in a `block_height_tiles × block_width_tiles` slot | no (raw NoC; see API Mapping) | input tensor via `TensorAccessor` | `cb_tiles_in`, `block_height_tiles·block_width_tiles` pushed | stale bytes in dead faces/rows/tiles of the slot |
| 2 | `untilize_block` | `block_height_tiles` tile-rows × `block_width_tiles` | yes — `compute_kernel_lib::untilize` | `cb_tiles_in`, `block_width_tiles` per tile-row, waited and popped | `cb_rows_out`, `block_width_tiles` per tile-row | `cb_tiles_in` slot freed |
| 3 | `store_live_rows` | live rows of live tile-rows × live column bytes | no (raw NoC; see API Mapping) | `cb_rows_out`, `block_height_tiles·block_width_tiles` waited | output tensor via `TensorAccessor` | popped after `noc_async_writes_flushed` |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| (compute boot) | helper | `compute_kernel_hw_startup(icb, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:113` | `(cb_tiles_in, cb_rows_out)`; configures UInt16 formats → MOVA2D zero flag disabled (`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_common.h:57-59`) | — | — | — |
| `untilize_block` | helper | `compute_kernel_lib::untilize<block_width_tiles, cb_tiles_in, cb_rows_out, InitAndUninit, WaitBlock>(in, out, num_blocks)` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:152-163` (dispatch `untilize_helpers.inl:79-87`) | `num_blocks = num_blocks_this_core · block_height_tiles`, one call per kernel | `cb_tiles_in` | `cb_rows_out` | `block_width_tiles` (CT template param); `block_height_tiles` enters via `num_blocks` |
| `load_live_tiles` | raw_api | `TensorAccessor::get_noc_addr(page, offset)`, `noc_async_read`, `noc_async_read_barrier`, `cb_reserve_back` / `cb_push_back` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165`; `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547`, `:1921` | per live tile: one read (full tile) or one read per live face-row prefix | — | `cb_tiles_in` | `block_height_tiles`, `block_width_tiles` (push quantum, walk) |
| `store_live_rows` | raw_api | `TensorAccessor::get_noc_addr(page, offset)`, `noc_async_write`, `noc_async_writes_flushed`, `noc_async_write_barrier` | `tensor_accessor.h:165`; `dataflow_api.h:876`, `:1980`, `:1951` | per live row: `(src = cb row, dst = get_noc_addr(n·H + 32·ht + r, 64·col), size = live bytes)` | `cb_rows_out` | — | `block_height_tiles`, `block_width_tiles` (wait quantum, cb row stride) |

**Helpers considered and rejected — `load_live_tiles`:**

| Candidate | File:Line | Why it cannot express the shape |
|-----------|-----------|---------------------------------|
| `dataflow_kernel_lib::read_sticks_for_tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:87-94` | reads ROW_MAJOR sticks (one page per row) for tilize; the input here is TILE pages with face-level live extents |
| `local_copy_helpers_dataflow.hpp` (`local_addr`, `set_read_state`) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:7-35, 59-64` | self-aimed L1→L1 reads only; the source here is a DRAM/L1 bank of an interleaved tensor |

No kernel_lib helper reads interleaved TILE pages; gap to close: a "read live tiles of a block rectangle" dataflow helper (face-prefix aware).

**Helpers considered and rejected — `store_live_rows`:**

| Candidate | File:Line | Why it cannot express the shape |
|-----------|-----------|---------------------------------|
| `dataflow_kernel_lib::write_sticks_after_untilize` | decl `tilize_helpers_dataflow.hpp:129-135`; impl `tilize_helpers_dataflow.inl:185-245` | (1) the cb row stride is derived from the written bytes, `padded_row_bytes = round_up(row_bytes, tile_row_bytes)` (`inl:209`), so a ragged column block (live cols < `block_width_tiles`, cb rows strided at `block_width_tiles·64` B) reads the wrong rows; (2) its wait/pop count is `width_in_tiles` derived from `row_bytes` (`inl:210, 228, 238`), so a ragged block pops fewer pages than compute pushed and desyncs the CB; (3) it groups `total_num_rows` contiguous pages 32 per cb tile-row (`inl:221-233`), which is wrong when a block spans images with `H % 32 ≠ 0` (`[16,1,2560]`: 16 rows from 16 different tile-rows); (4) it pays a full `noc_async_write_barrier` per tile-row (`inl:236`) where a per-block flush suffices. |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| 1. Bit-exactness through the FPU path | MOVA2D flushes zero-exponent Src data for float formats (`ckernel_defs.h:19-31`): a Float16_b untilize turns subnormals and −0.0 into +0.0; random inputs never show it | both CBs UInt16. **Phase 0 gate, before any perf work:** run the specials and poisoned-padding loose cases on cases 308 and 304 and the acceptance test's special-pattern test. If any bit differs, switch to R2 (same reader, CB `cb_tiles_in`, block walk; delete compute and `cb_rows_out`; writer gathers face rows). |
| 2. Stale bytes in unread faces / ragged slots | the reader skips dead faces, dead rows and ragged tiles; the untilize carries those stale bytes into padding positions of `cb_rows_out` | the writer writes only `live_rows` rows and `min(block_cols·32, W − 32·col)·2` bytes per row; poisoned-padding cases (1000.0, NaN) gate it |
| 3. Output page stride ≠ row bytes | RM pages are strided at `round_up(W·2, 64)` in DRAM (16 in L1); W = 3, 185, 196, 207 are production widths | address rows only through `get_noc_addr(page, offset)` |
| 4. Image boundaries inside a block | a block's tile-rows can span images; the last `ht` of each image has `live_rows < 32` when `H % 32 ≠ 0` (`[16,1,2560]`: every tile-row has 1 live row) | per tile-row: `n = tile_row / Ht`, `ht = tile_row % Ht` (incremental), output row base `n·H + 32·ht` |
| 5. Column cuts and alignment | DRAM reads compare addresses mod 64, writes mod 16 | column groups and sub-blocks cut only at tile boundaries; face reads start at face starts |
| 6. L1 budget at the cap | default blocks fill ~1.2–1.3 MB of CBs; an L1 output's pages share the same 1347776 B | `block_tiles_cap` subtracts `out_l1_bytes_per_bank`; acceptance test squeezes L1 to 1347776 B free with a guard page |
| 7. Narrow rows are write-issue-bound | `[1,2048,2048,3]`: 4194304 output rows, one NoC write each; each 6-B write still occupies a DRAM burst | Phase 0 writes exactly the live bytes; lamp L5 carries the issue-rate alternatives |
| 8. Plan/descriptor rebuilt per call | the `num_col_groups` search is O(`tensor_tile_cols`) on host | memoize `plan()` + descriptor (BB-07); trace replay allocates only the output (BB-16) |
| 9. Ragged block CB accounting | ragged rows/cols occur mid-core (column-group-major runs cross group boundaries) | nominal push/wait/pop counts on every block; actual extents only in the NoC transfers and the writer's row loop |
| 10. bf8b refinement precision | Bfp8_b → Float16_b unpack lands in a float Src format, so the MOVA2D flush applies (−0.0 → +0.0); case 29 with `bf16_specials` gates it | refinement must disable the Src zero flag for the untilize datacopy (or prove the flush cannot fire) before claiming bf8b |
| 11. Completion and profiling | another program starts right after the call (BB-03); BB-21 needs every wait inside a named zone | reader barrier per block, writer `noc_async_write_barrier` at kernel end; `DeviceZoneScopedN` around each block operation and each blocking wait in the zone build only |
