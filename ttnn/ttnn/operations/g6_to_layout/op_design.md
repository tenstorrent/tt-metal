# Operation Design: g6_to_layout

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (layout conversion; one format conversion bfloat16 <-> bfloat8_b in refinements) |
| Goal | Return the same logical values in the other layout (TILE <-> ROW_MAJOR), in the requested dtype and interleaved placement, as ONE device program built from this op's own kernels: one read of the input, one write of the output. Phase 0: bfloat16 TILE -> ROW_MAJOR, interleaved DRAM or L1, every alignment, ranks 2-4. |
| Math | `y[n, h, w] = x[n, h, w]` for n < N, h < H, w < W; N = product of the leading dims (1 for rank 2). bfloat16 output is bit-identical (signed zeros, subnormals, inf, NaN payloads). |
| References | `docs/explorations/g6_to_layout/` (DOSSIER.md, tensix.md, risks.md, oracle/pytorch_g6_to_layout.py); `eval/golden_tests/g6_to_layout/feature_spec.py`, `helpers.py`; `.claude/references/blocking-model.md`, `l1-footprint-discipline.md`; catalog `ttnn/ttnn/operations/examples/master.md` entries `double_buffer`, `trid_double_issue`, `tile_reorder`, `width_split`, `noc_placement`, `split_reader`, `bank_stagger`, `compute_block_size` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes (positional) | rank 2-4; Phase 0: bfloat16, TILE, INTERLEAVED in DRAM or L1 | - | buffer address: RT |
| `layout` | `ttnn.Layout` | yes (positional) | `ROW_MAJOR_LAYOUT` / `TILE_LAYOUT`, must differ from the input's | - | host only (selects the program) |
| `dtype` | `ttnn.DataType` | no (keyword) | bfloat16, bfloat8_b (bfloat8_b only with `TILE_LAYOUT`) | None -> input dtype; bfloat8_b input to ROW_MAJOR -> bfloat16 | host only (sets CB formats: CT via CB descriptor) |
| `memory_config` | `ttnn.MemoryConfig` | no (keyword) | interleaved DRAM or L1 only | None -> the input's placement | host only (output allocation; output address RT) |

There is no `compute_kernel_config`: the op performs no arithmetic. `fp32_dest_acc_en` is fixed **off** (16-bit DEST).

### Entry point and validation order

`g6_to_layout(input_tensor, layout, *, dtype=None, memory_config=None)` calls `validate(...)` as its first line. `validate` builds the scenario dict (`shape`, `dtype`, `layout`, `memory_layout`, `buffer`, `shard_spec`, `target_layout`, `out_dtype`, `memory_config`) exactly as `eval/golden_tests/g6_to_layout/axes.py::classify_call` does, and checks in this order:

| # | Check | Raises |
|---|-------|--------|
| 1 | `layout` not in {ROW_MAJOR, TILE}, or equal to `input_tensor.layout` | `ValueError` |
| 2 | `len(shape)` not in {2, 3, 4} | `ValueError` |
| 3 | `dtype == bfloat8_b` and `layout == ROW_MAJOR_LAYOUT` | `ValueError` |
| 4 | SUPPORTED per axis (tagger outputs) | `UnsupportedAxisValue` |
| 5 | EXCLUSIONS (Phase 0: none) | `ExcludedCell` |
| 6 | `memory_config` sharded, or `memory_config is None` while the input is sharded | `ValueError` |

Malformed calls (1-3) come before the support gate so a malformed call is a `ValueError` in every phase; check 6 is after the memory_layout refusal as the spec requires.

### INPUT_TAGGERS (names and values = TARGET of `feature_spec.py`)

| Axis | Tagger reads | Value |
|------|--------------|-------|
| `dtype` | `scenario["dtype"]` | ttnn.bfloat16 / ttnn.bfloat8_b |
| `layout` | `scenario["layout"]` | TILE / ROW_MAJOR |
| `output_dtype` | `out_dtype` if not None; else bfloat16 if dtype is bfloat8_b and target_layout is ROW_MAJOR; else dtype | bfloat16 / bfloat8_b |
| `memory_layout` | `scenario["memory_layout"]` | INTERLEAVED / HEIGHT_SHARDED / BLOCK_SHARDED |
| `alignment` | H = shape[-2], W = shape[-1]: W % 32 -> "w_non_aligned"; else H % 32 -> "h_non_aligned"; else "tile_aligned" | 3 strings |
| `rank` | `len(shape)` | 2 / 3 / 4 |

Phase 0 SUPPORTED: dtype [bfloat16], layout [TILE], output_dtype [bfloat16], memory_layout [INTERLEAVED], alignment [all three], rank [2, 3, 4]. EXCLUSIONS: none. Input buffer DRAM or L1 is not an axis: both are supported in Phase 0.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d0, ..., H, W)`, rank 2-4, any H >= 1, W >= 1 (W up to 131072 in production) |
| Dtype | bfloat16 (Phase 0); bfloat8_b (refinement, TILE only) |
| Layout | TILE (Phase 0); ROW_MAJOR (refinement) |
| Memory | INTERLEAVED in DRAM or L1 (Phase 0); HEIGHT_SHARDED / BLOCK_SHARDED in L1 (refinement) |

### Output

| Property | Value |
|----------|-------|
| Shape | `shape(input_tensor)` |
| Dtype | `dtype` keyword; else input dtype; bfloat8_b input to ROW_MAJOR -> bfloat16 |
| Layout | `layout` (the other one) |
| Memory | INTERLEAVED; buffer type of `memory_config`, else of the input. Allocated once per call by the host (`ttnn.allocate_tensor_on_device`) before dispatch; no other device tensor is allocated. |

## Blocking Model

### Axes

The op works in the tile index space of the TILE side. Leading dims fold with H **per image**: tile row `t` of the tensor is image `n = t / tiles_per_image_h`, image tile row `t % tiles_per_image_h`. `tiles_per_image_h = ceil(H/32)`, `tensor_tile_rows = N * ceil(H/32)`, `tensor_tile_cols = ceil(W/32)` (never `floor`, never `N*H/32`).

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| tile rows (leading dims x H, folded per image) | independent — each output value reads exactly one input value; no result spans rows | `block_tile_rows` | `clamp(floor(block_tiles_target / block_tile_cols), 1, max_rows_fit)` (p1: 32; Qwen W=2560: 1) | host `plan_blocking()` -> reader/writer CT arg | split across the grid as part of the 2-D work unit (tile row, width chunk) via `split_work_to_cores(row_wise=True)` | knob-turn |
| tile columns (W) | independent for bfloat16 — no result spans columns; for a bfloat8_b output an exponent group is 16 columns of one face row, always inside one tile, so any whole-tile cut keeps it whole | `block_tile_cols` (= width-chunk width) | whole row (`tensor_tile_cols`, rounded per the DEST cap) unless the L1 cap or occupancy forces chunks; chunks when `tensor_tile_rows < num_cores` | host `plan_blocking()` -> compute CT arg (untilize `block_width_tiles`), reader/writer CT arg | split across cores (width chunks) only when the tile-row axis cannot fill the grid or a tile row exceeds the L1 cap | knob-turn |
| row inside a tile (h mod 32) | independent, but fixed by the TILE storage unit: a TILE page holds 32 rows | none tunable — mechanism unit (32 rows = one tile) | 32 (whole tile) | tile geometry (constant of the format) | never split (a tile is the read unit) | none |
| face (16x16 quadrant) | independent; the read unit inside a tile | `live_faces` (which faces of a tail tile are read) | all 4, except tail tiles: faces {0,2} when `W%32 in 1..16`, {0,1} when `H%32 in 1..16`, {0} when both | host `plan_blocking()` -> reader CT args `w_tail_half`, `h_tail_half` | not assigned (inside a tile) | none (built) |

Knobs that are not per-axis:

| Knob | Phase 0 value | Source | Why this value |
|------|---------------|--------|----------------|
| `block_tiles_target` | 32 tiles (64 KiB bfloat16 per block) | host constant `BLOCK_TILES_TARGET` in the program-descriptor module | Bytes moved are constant in the extent here (every byte crosses once whatever the block), so the extent only amortizes fixed per-block cost: 1 read barrier, 1 write flush, CB handshakes. At 32 tiles that cost is ~1-2 % of a block's transfer time (`double_buffer`: gains saturate by 4-8 outstanding tiles; `trid_double_issue`: large CB ~ within 3 % of best). The largest block that fits (~164 tiles) only lengthens the un-overlapped first fill (p1: ~70 us per core at a DRAM share of ~4.5 GB/s per core), so it is the overlap lamp, measured below. A tile row wider than 32 tiles is never cut for this target: `block_tile_rows >= 1` keeps whole rows. |
| grid | `device.compute_with_storage_grid_size()` | host, per call (part of the cache key) | full grid; never a constant |
| `depth_in`, `depth_out` | 2, 2 | host constants `DEPTH_IN`, `DEPTH_OUT` | double buffering (see buffer-depth knobs) |
| `op_l1_budget` | `1347776 - out_l1_bytes_per_core` | host `plan_blocking()` | spec bound counted before the output exists; an L1 output's own pages come out of it (`out_l1_bytes_per_core = ceil(num_out_pages / num_l1_banks) * out_aligned_page_bytes`, 0 for a DRAM output) |
| `max_block_cols` | `floor_dest(floor(op_l1_budget / ((depth_in + depth_out) * tile_bytes)))` = 160 for a DRAM output | host `plan_blocking()` | L1 residency bound on the width chunk; `floor_dest(x)` = `x` if `x <= 8`, else `8 * floor(x/8)` |

#### Work-split formula (single source: host `plan_blocking()`)

```
tile_bytes          = 2048                       # bf16 / UInt16 tile (bf8b: 1088, refinement)
num_cores           = grid.x * grid.y
occupancy_chunks    = ceil(num_cores / tensor_tile_rows) if tensor_tile_rows < num_cores else 1
chunks0             = max(ceil(tensor_tile_cols / max_block_cols), min(tensor_tile_cols, occupancy_chunks))
block_tile_cols     = ceil(tensor_tile_cols / chunks0)
if block_tile_cols > DEST_TILES: block_tile_cols = min(align_up(block_tile_cols, DEST_TILES), max_block_cols)
num_col_chunks      = ceil(tensor_tile_cols / block_tile_cols)
max_rows_fit        = floor(op_l1_budget / ((depth_in + depth_out) * block_tile_cols * tile_bytes))
block_tile_rows     = clamp(floor(block_tiles_target / block_tile_cols), 1, max_rows_fit)
num_units           = tensor_tile_rows * num_col_chunks     # unit u -> tile_row = u / num_col_chunks, chunk = u % num_col_chunks
(cores, g1, g2, units_g1, units_g2) = split_work_to_cores(grid, num_units, row_wise=True)
num_blocks_this_core = ceil(units_this_core / block_tile_rows); last_block_extent = units_this_core - (num_blocks_this_core-1)*block_tile_rows
```

`DEST_TILES = 8` (16-bit DEST, half-sync; `tt_metal/hw/inc/api/compute/pack_untilize.h:102`). A block is `block_tile_rows` consecutive **units** of one core (a unit is one tile row of one width chunk: `block_tile_cols` tiles). Compute is oblivious to which tile row / chunk a unit is: every unit is one untilize row of nominal width `block_tile_cols`. Reader and writer derive `(tile_row, chunk)` per unit.

Production plans (grid 110 cores shown; the grid is a runtime input):

| Case | tensor_tile_rows x cols | block_tile_cols x chunks | block_tile_rows | units | cores |
|------|------------------------|--------------------------|-----------------|-------|-------|
| p1 `[1,2048,2048,3]` | 131072 x 1 | 1 x 1 | 32 | 131072 | all |
| p2 `[1,1024,2048,128]` | 65536 x 4 | 4 x 1 | 8 | 65536 | all |
| Qwen `[16,4096,2560]` | 2048 x 80 | 80 x 1 | 1 | 2048 | all |
| `[32,5120]` | 1 x 160 | 2 x 80 | 16 | 80 | 80 |
| `[1,131072]` | 1 x 4096 | 40 x 103 | 1 | 103 | 103 |
| `[16,1,2560]` (H=1) | 16 x 80 | 16 x 5 | 2 | 80 | 80 |
| `[1,185]` | 1 x 6 | 1 x 6 | 32 | 6 | 6 |
| `[1,32,2560]` L1 (2038) | 1 x 80 | 1 x 80 | 32 | 80 | 80 |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `depth_in` | 2 | reader fetches block k+1 (one barrier per block) while compute untilizes block k |
| `cb_output_rows` | `depth_out` | 2 | compute untilizes block k+1 while the writer drains block k's rows |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `pack_untilize` DEST capacity, 16-bit half-sync: 8 tiles per pack (`pack_untilize.h:102`); the helper splits a wider row into sub-blocks of the largest divisor <= 8 (`untilize_helpers.inl:69-76`) | `block_tile_cols` | when `> 8`, round **up** to a multiple of 8 (nominal width; the extra tiles are stale L1, untilized but never written out) | a prime width (e.g. 41) falls to 1-tile sub-blocks: correct but ~8x the pack calls |
| L1 residency (`op_l1_budget`) | `block_tile_cols * block_tile_rows` | `max_block_cols`, `max_rows_fit`; a wider row is cut into width chunks | out-of-memory / refused launch on `[1,131072]` (8 MiB per tile row) — a correctness failure under the spec |
| Untilize helper CT width (`block_width_tiles` is a template parameter, `untilize_helpers.hpp:163`) | `block_tile_cols` must be one value per program | ragged last width chunk keeps the nominal `block_tile_cols`; reader reads only live tiles, writer writes only live bytes | a second compute kernel or a recompiled width per chunk |
| DRAM read alignment 64 B on Blackhole (`noc_parameters.h:379`) | face reads of tail tiles | read whole faces (512 B at offsets 0 / 1024) or the face pair 0+1 (1024 B at offset 0) — never sub-face pieces | misaligned DRAM reads: wrong data with no error |
| NoC write alignment 16 B (L1) / DRAM write 16 B | output row write length | write `align_up(live_bytes, 16)` for the last chunk of a row, which stays inside the page's aligned slot (`aligned_page_size >= align_up(2W, 16)`); L1 source offset `r * block_tile_cols * 64` and DRAM offset `chunk * block_tile_cols * 64` are 64 B multiples | writes of 6 B / 370 B (W = 3, 185) mis-sized: wrong bytes or a NoC fault (R9) |
| Untilize output CB uses tile-sized symmetric pages (`untilize_helpers.hpp:112-113`) | `cb_output_rows` page size | page = `tile_bytes`, row r of a unit at byte `r * block_tile_cols * 64` | wrong row stride in the writer |
| Bit identity of bfloat16 through the engines | data format of both CBs | Phase 0 CBs are declared **`UInt16`** (opaque 16-bit, identity path in unpacker `cunpack_common.h:476-482` and packer `cpack_common.h:269-270`); bfloat16 bytes are moved as opaque 16-bit words | the Float16_b float path is not stated to keep subnormals / NaN payloads (`tensix.md#gaps`, R4): wrong bits only on special patterns |
| Fast-untilize gate accepts only Float16_b / Bfp8_b / Bfp4_b input (`untilize_helpers.inl:31-41`) | path selection | UInt16 CBs take the standard / block-based `pack_untilize` path (perf lamp L3 measures the fast path) | — |

### Regimes

Named minimum boundary: **each input page byte crosses its memory (DRAM or L1 bank) once to the core, each output row byte crosses back once.**

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| R1 `interleaved_untilize_2d` — 2-D unit split (tile rows x width chunks), whole-tile reads, opaque-UInt16 engine untilize, whole-row-chunk writes | **built** (Phase 0) | input TILE, INTERLEAVED (DRAM or L1), dtype bfloat16, output ROW_MAJOR bfloat16 | `block_tile_rows x block_tile_cols` tiles | input: minimum (each tile read once, padding of interior tiles included because a tile is one page); output: minimum, one write per (live row, chunk), each row byte once; rows of `2W < 64 B` (p1, p4) still occupy a 64 B DRAM slot and cost one DRAM burst each — structurally unreachable below one burst per row. Covers every Phase 0 shape: width chunking (`num_col_chunks > 1`) is a knob value of this regime, not another algorithm. | per block: 1 read barrier, 1 write flush, 2 CB handshakes per unit row; untilize init/uninit once per **kernel** (one helper call over all the core's units) |
| R1a `live_face_reads` (sub-case of R1's reader) | **built** (Phase 0) | per tail tile: `W%32 in 1..16` (last tile column) -> faces {0,2}; `H%32 in 1..16` (last tile row of each image) -> faces {0,1}; both -> {0} | same as R1 | input below the whole-tile figure: p1/p4 read 1 KiB of every 2 KiB tile (268 -> 134 MB on p1); `[16,1,2560]` reads 1 KiB per tile; faces not read hold stale L1 that is untilized and never written | same as R1 |
| R2 `sharded_in_place` — input CB backed on the core's shard (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy), no NoC read of the input; same compute + writer | **deferred** — the three sharded production cases (20, 37, 48) are all bfloat8_b inputs, so R2 lands with the bfloat8_b-input refinement (memory_layout + dtype axes) | input HEIGHT_SHARDED or BLOCK_SHARDED in L1 | one block = the whole live shard: `live_shard_tile_rows x shard_tile_cols` (case 37: 3 x 8, last shard 1 x 8; case 48: 3 x 4 / 1 x 4; case 20: 1 x 1) | input: 0 crossings (already resident); output: minimum (interleaved L1 pages on every bank) | one helper call per shard; no per-block barrier on the input side |
| R3 `rm_tilize_2d` — ROW_MAJOR -> TILE: reader reads the 32 rows of a unit as row chunks (`block_tile_cols*64` B each at page offset `chunk*block_tile_cols*64`), compute `tilize` (opaque UInt16 for bf16 out; Float16_b -> Bfp8_b packer for bf8b out), writer writes whole tiles | **deferred** — the layout axis value ROW_MAJOR is a refinement (19 cases, none flagged) | input ROW_MAJOR interleaved | same `block_tile_rows x block_tile_cols` units | input: minimum (each row read once; rows >= H not read); output: whole tiles, padding rows included (a tile is one page) | as R1 |
| R4 `bf8b_input_untilize` — R1/R2 with `cb_input_tiles` in Bfp8_b (1088 B tiles) and `cb_output_rows` in Float16_b; the unpacker expands Bfp8_b -> Float16_b exactly (`cunpack_common.h:425-445`) | **deferred** — dtype axis value bfloat8_b is a refinement | input bfloat8_b TILE | as R1 / R2 | as R1 / R2 | as R1 |
| R5 `dm_face_row_reorder` — no compute engine; a data-movement RISC moves the 64 face-row pieces (32 B) of every tile into row order (L1-to-L1 NoC or RISC-V loads) | **rejected** — superseded by R1: 64 transactions of 32 B per tile instead of one 2 KiB read + one packer pass; the bit-exactness it would buy is already given by the opaque UInt16 path | — | — | same bytes, 64x the transaction count on the issue-bound RISC | — |
| R6 `bank_run_writes` — write k rows of one DRAM bank (`p, p+NB, ...`, adjacent 64 B slots) in one transfer for narrow rows | **rejected** — needs the rows permuted into bank order in L1 first; that permutation is one local move per row, the same transaction count as the per-row DRAM writes it removes, so it is not structurally cheaper; the per-row issue cost is attacked by lamp L4 instead | — | — | same bytes | — |
| R7 `single_core` / `tile_row_only_split` | **rejected** — a tile-row-only split strands `[1,131072]`, `[32,5120]`, `[1,32,2560]` (one tile row) on one core; R1's width chunks supersede it | — | — | — | — |

Selection: Phase 0 builds R1 (+R1a, a predicate inside the reader). Later regimes select on the input placement / layout / dtype — host-checkable from the tensor alone, never from free L1. The op exposes the selected plan (BB-08): `g6_to_layout.plan_for(input_tensor, layout, dtype=..., memory_config=...)` returns the `plan_blocking()` dict (regime name, block extents, chunks, units, core ranges), and the program-descriptor builder logs nothing in the plain path.

### Traffic ranking

Every byte crosses its memory once each way under every split below (no operand is shared, no axis is dependent), so the ranking is by **transfer shape**: transaction count and size.

| Rank | Split | Input (DRAM) | Output (DRAM) | Cross-core | Verdict |
|------|-------|--------------|---------------|------------|---------|
| 1 | tile rows only, whole-row chunks | 1x, 2 KiB tile reads (1 KiB / 512 B for tail tiles, R1a) | 1x, one write of `2W` B per row | none | cheapest shape: fewest, largest row writes. Chosen whenever `tensor_tile_rows >= num_cores` and the row fits L1 |
| 2 | tile rows x width chunks (2-D) | 1x, same tile reads | 1x, `num_col_chunks` writes of `block_tile_cols*64` B per row | none | same bytes, more and smaller writes; used only when rank 1 cannot fill the grid or a row exceeds `max_block_cols` |
| 3 | width only | 1x | 1x, every row written in `num_cores` pieces | none | fragments every row on tall tensors; dominated by rank 2 |
| — | dependent-axis split + combine | n/a | n/a | n/a | no dependent axis exists (no result spans any axis for bfloat16; bfloat8_b groups stay inside a tile) |

Occupancy is applied *after* this ranking: width chunks are added only to fill the grid (`occupancy_chunks`) or to respect L1. Operand-reuse check: no operand is read by more than one core — every input tile belongs to exactly one unit — so no broadcast row exists.

Stall-shadow check: the stages that wait are the reader (on its read barrier), compute (on `cb_input_tiles`) and the writer (on `cb_output_rows`). The only work independent of a pending read is issuing the **next** block's reads before retiring the current one — that is lamp L6 (transaction-id double issue). No algebraic reorder exists (the op computes nothing).

### Block schedule

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);       // reader
    untilize_block(block_idx);   // compute
    store_block(block_idx);      // writer
}
```

| Operation | Block shape | Resident across it | Intended fixed-cost frequency |
|-----------|-------------|--------------------|-------------------------------|
| `load_block` | `extent x block_tile_cols` tiles, `extent = block_tile_rows` (last block: `last_block_extent`); per unit `(tile_row, chunk)`: live tiles `min(block_tile_cols, tensor_tile_cols - chunk*block_tile_cols)`, each read whole or as its live faces (R1a) into its nominal slot | the block's slot in `cb_input_tiles` | `cb_reserve_back` once per block (`extent * block_tile_cols` pages); all reads issued, **one** `noc_async_read_barrier` per block; `cb_push_back(block_tile_cols)` per unit (the compute's quantum) after the barrier |
| `untilize_block` | `extent` untilize rows of `block_tile_cols` tiles | nothing across blocks | **one** `compute_kernel_lib::untilize<block_tile_cols, cb_input_tiles, cb_output_rows>(units_this_core)` call per kernel: init/uninit once per kernel, waits per sub-block / unit row inside the helper |
| `store_block` | `extent` units; per unit the live rows `rows_live = min(32, H - (tile_row % tiles_per_image_h)*32)`, per row one write of `align_up(live_cols*2, 16)` B (last chunk of the row) or `block_tile_cols*64` B (interior chunk) to page `n*H + h` at offset `chunk*block_tile_cols*64` | the block's slot in `cb_output_rows` | `cb_wait_front(extent * block_tile_cols)` once per block; **one** `noc_async_writes_flushed` per block before `cb_pop_front`; one `noc_async_write_barrier` at kernel end (BB-03) |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 overlap | `block_tiles_target = 32` is below the L1 bound on purpose (fill latency); it may be too small where the per-block barrier is visible (narrow rows) | `block_tiles_target` in {16, 32, 64, largest-fit} on p1, p2, Qwen `[16,4096,2560]` |
| L2 grid synchronization | width chunks for occupancy give tiny shapes (`[1,185]`: 6 cores x 1 tile; `[32,5120]`: 80 cores x 2 tiles) launch-dominated work per core | a floor on units per core (e.g. >= 4 tiles) for shapes under ~1k tiles; judge on unprofiled trace-replay time (lesson RB-053) |
| L3 opaque UInt16 vs Float16_b fast untilize | UInt16 bypasses BH fast untilize (`untilize_helpers.inl:31-52`), which emits wider row-major strips | Float16_b CBs + fast untilize, accepted **only** if a probe proves all 65536 bfloat16 bit patterns survive bit-exactly (acceptance test `test_all_bit_patterns`) |
| L4 writer issue rate on narrow rows | p1 / p4 issue 32 writes of 16 B per tile against 2 reads per tile: the writer RISC is the critical path | split each block's row writes between both data-movement RISCs (`split_reader` pattern, reader has spare issue slots); stateful one-packet write API (`noc_async_write_one_packet_with_state`) |
| L5 core placement / bank stagger | `row_wise=True` placement and every core starting its walk at the same bank phase (`bank_stagger`, `noc_placement`) | rotate each core's unit order by `core_idx % num_dram_banks`; diagonal placement |
| L6 read issue depth | one barrier per block drains the read queue at every block | transaction-id double issue in the reader (`trid_double_issue`), depth 3-4 |
| L7 buffer depth | `depth_in = depth_out = 2` may not hide DRAM jitter on wide rows | depth 3 on Qwen `[16,4096,2560]` (fits: 6 x 160 KiB = 960 KiB) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| input DRAM/L1 -> core | TILE pages (2048 B bf16), interleaved | reader (NCRISC, NoC0): `TensorAccessor` `get_noc_addr(tile_id, face_offset)` + `noc_async_read`; whole tile, or live faces (R1a) | tile id = `tile_row * tensor_tile_cols + chunk * block_tile_cols + j`; only live tiles of a ragged chunk are read |
| reader -> compute | `cb_input_tiles`, UInt16 tile pages | CB, pushed per unit (`block_tile_cols` pages) | bfloat16 bytes declared UInt16 (opaque) |
| compute | tiles -> row-major | `compute_kernel_lib::untilize` (pack_untilize; block-based when `block_tile_cols > 8`) | no arithmetic; `fp32_dest_acc_en = False` |
| compute -> writer | `cb_output_rows`, UInt16, tile-sized pages; a unit = 32 rows of `block_tile_cols*64` B | CB | row r of a unit at `r * block_tile_cols * 64` |
| core -> output DRAM/L1 | ROW_MAJOR pages of `2W` B, interleaved | writer (BRISC, NoC1): `TensorAccessor` `get_noc_addr(page, chunk*block_tile_cols*64)` + `noc_async_write` per live row | rows `h >= H` of a tail tile row and columns `w >= W` are never written as values |

TARGET's `memory_layout` values as regimes of this scheme:

| TARGET value | Cut along | Cost class | Realization |
|--------------|-----------|------------|-------------|
| INTERLEAVED | 2-D logical split of R1 | built | R1 |
| HEIGHT_SHARDED (cases 37, 48) | tile rows (independent) | knob-turn: placement change | R2: `cb_input_tiles` backed on the shard (zero-copy); each core untilizes its own shard; `units_this_core = live_shard_rows/32` (last shard: 1 of 3 tile rows); the reader kernel only pushes the live pages (no NoC read); writer unchanged, output pages computed from the shard's first global row (shards laid on the grid range by range, each range row by row for ROW_MAJOR orientation) |
| BLOCK_SHARDED (case 20) | tile rows and tile columns (both independent) | knob-turn | R2 with one width chunk per core: `block_tile_cols = shard_cols/32`, `chunk = core's column index` in the shard grid; output writes are row pieces at offset `chunk*block_tile_cols*64` — exactly R1's chunk write |

No sharded value cuts a dependent axis, so none needs a combine.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one unit = one tile row of one width chunk (`block_tile_cols` tiles); a block = `block_tile_rows` consecutive units of one core |
| Grid | `device.compute_with_storage_grid_size()`; cores from `ttnn.split_work_to_cores(grid, num_units, row_wise=True)` -> at most two core groups, each a few rectangles (full grid rows + one partial row) — BB-17 |
| Per-core work | contiguous unit range `[start_unit, start_unit + units_this_core)`; `units_this_core` = `units_g1` or `units_g2`; RT args per core: `start_unit`, `units_this_core`; compute RT arg `units_this_core` |
| Remainder | units: the g1/g2 split of `split_work_to_cores`; blocks: `last_block_extent`; width: ragged last chunk `live_cols = W - chunk*block_tile_cols*32` (clamped to `block_tile_cols*32`); height: `rows_live = min(32, H - (tile_row % ceil(H/32))*32)` per image. All counts use `ceil` per image. |
| Idle cores | cores without units get no kernels (not in the `all_cores` set); no core waits on a CB that receives nothing |
| Kernel groups | one reader, one compute, one writer kernel over `all_cores`; per-core variation by RT args only (one binary per role) |

Regime selection is a pure function of the input placement/layout/dtype (Phase 0: one regime). Regime-pinned tests when R2-R4 land: each sharded case at its production shard spec, a ROW_MAJOR case, a bfloat8_b case.

### Host-side obligations (BB-07, BB-16, BB-21)

| Rule | Design decision |
|------|-----------------|
| BB-07 program cache | the program descriptor is built once per key and memoized; key = (shape, input dtype, layout, memory config incl. buffer type, output dtype, output memory config, grid). Only buffer addresses change between calls (RT args patched). |
| BB-16 trace | per call: one output allocation (the output tensor itself) and one `ttnn.generic_op` dispatch; no kernel build after the first call with a key |
| BB-21 zones | each kernel wraps `load_block` / `untilize_block` / `store_block` and every blocking wait in `DeviceZoneScopedN` zones (compiled out of the plain build); `plan_for(...)` reports each core's role |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_tiles` | 0 | `tile_bytes` (2048) | `depth_in * block_tile_rows * block_tile_cols` | spans tile rows (`block_tile_rows`) and tile columns (`block_tile_cols`) of one block; streams over blocks; x2 for overlap | UInt16 (Phase 0) | reader | compute | whole kernel, ring of blocks |
| `cb_output_rows` | 1 | `tile_bytes` (2048) | `depth_out * block_tile_rows * block_tile_cols` | same block span, row-major content; x2 for overlap | UInt16 (Phase 0) | compute | writer | whole kernel, ring of blocks |

Push / wait counts: `cb_input_tiles` — reader pushes `block_tile_cols` per unit; compute (helper) waits `sub_block_width` (block-based path) or `block_tile_cols` and pops the same; totals equal `units_this_core * block_tile_cols`; capacity is a multiple of both quanta. `cb_output_rows` — compute pushes `block_tile_cols` per unit; writer waits and pops `extent * block_tile_cols` per block; capacity is a multiple of both.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `extent x block_tile_cols` tiles | no (dataflow) | — (DRAM/L1 interleaved input via `TensorAccessor`) | `cb_input_tiles`, `extent*block_tile_cols` pages (stale slots for absent tiles / unread faces) | pushed |
| 2 | `untilize_block` | `extent` rows of `block_tile_cols` tiles | yes: `compute_kernel_lib::untilize` (one call over all `units_this_core`) | `cb_input_tiles`, waited and popped by the helper | `cb_output_rows`, `block_tile_cols` pages per unit | input popped, output pushed |
| 3 | `store_block` | `extent` units, live rows x live bytes | no (dataflow) | `cb_output_rows`, `extent*block_tile_cols` pages | — (interleaved output via `TensorAccessor`) | popped after `noc_async_writes_flushed` |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| (compute start) | raw_api | `compute_kernel_hw_startup(cb_input_tiles, cb_output_rows)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:113` | two-arg form | `cb_input_tiles` | `cb_output_rows` | — |
| `untilize_block` | helper | `compute_kernel_lib::untilize<block_width_tiles, input_dfb, output_dfb>(in, out, num_blocks)` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:163` (impl `untilize_helpers.inl:151-282`) | `block_width_tiles = block_tile_cols` (CT), defaults `InitAndUninit`, `WaitBlock`, `UnpackAndPackReconfigure`, `RemapMode::Configure`; `num_blocks = units_this_core` (RT) | `cb_input_tiles` | `cb_output_rows` | `block_width_tiles` |
| `load_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)`, `noc_async_read`, `noc_async_read_barrier` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165`, `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547`, `:1921` | tile page, face offsets 0 / 1024 / 512-B lengths | — | `cb_input_tiles` | extent via RT/CT `block_tile_rows`, `block_tile_cols` |
| `store_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)`, `noc_async_write`, `noc_async_writes_flushed`, `noc_async_write_barrier` | `tensor_accessor.h:165`, `dataflow_api.h:876`, `:1980`, `:1951` | row page, offset `chunk*block_tile_cols*64`, length `align_up(live_bytes,16)` | `cb_output_rows` | — | extent via `block_tile_rows`, `block_tile_cols` |
| host split | helper (host) | `ttnn.split_work_to_cores(grid, num_units, row_wise=True)` | `.claude/references/ttnn-python-utility-bindings.md:255` | — | — | — | `num_units` |

Helpers considered and rejected for the raw_api dataflow entries:

| Candidate | File:Line | Why it cannot express the shape |
|-----------|-----------|--------------------------------|
| `dataflow_kernel_lib::write_sticks_after_untilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:130` | writes whole sticks of a tile row; it has no width-chunk page offset (`chunk*block_tile_cols*64`) and no per-image live-row count, both needed by R1's 2-D units and per-image H tails. Gap recorded: a stick writer with `(page offset, live bytes, live rows)` would close it. |
| `read_sticks_for_tilize` | `tilize_helpers_dataflow.hpp:88` | reads sticks (ROW_MAJOR input) — wrong direction for Phase 0; candidate for R3's reader |
| `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:206` | RM -> TILE: R3's compute, not Phase 0 |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Special bfloat16 patterns changed by the engines (R4) | the golden suite injects -0, subnormals, inf, NaN 0x7FC1 into p1 / p2 / decode shapes; graded bit-exact | both CBs declared UInt16 (opaque identity path); acceptance test `test_all_bit_patterns` runs all 65536 patterns through a production shape — run it FIRST after the kernels compile. If UInt16 is not exact, stop and report (do not switch to Float16_b) |
| Input tile padding reaching the output (R8) | the model's producer leaves anything in padding; poisoned-padding golden cases on p1 and `[1,131072]` | writer writes only `rows_live` rows and `live_cols` columns; R1a does not even read padding faces; rows beyond `H` per **image** (tile row index modulo `ceil(H/32)`) — not per tensor |
| `[1,131072]` row of 8 MiB (R2) | a tile row exceeds L1 12x | width chunks capped by `max_block_cols`; occupancy chunks make it 103 chunks of 40 tiles |
| L1 output inside the 1347776 B (R11) | case 2038 keeps its L1 placement; refinement cases up to 153600 B per core | `op_l1_budget` subtracts `out_l1_bytes_per_core` before sizing CBs; CB total `(depth_in+depth_out)*block_tile_rows*block_tile_cols*tile_bytes <= op_l1_budget` asserted on host |
| Ragged chunk / stale tiles | the last width chunk and the `align_up(…, 8)` rounding leave CB slots never written by the reader | allowed: those slots' rows are untilized but the writer never reads them; push/pop counts stay nominal (`block_tile_cols` per unit) |
| Narrow-row write length (R9) | W = 3 -> 6 B rows, W = 185 -> 370 B | round the last chunk's length up to 16 B (inside the page's aligned slot); offsets are 64 B multiples |
| Untilize output layout assumption | the writer must know the row stride of the untilized unit | row stride = `block_tile_cols * 64` B (pack_untilize writes a unit's 32 rows full-width, `untilize_helpers.inl:248-256`) — verify with a 2-chunk probe |
| Idle cores / tiny shapes (R10) | 6-tile shapes on a 110-core grid | only cores with units get kernels |
| Program rebuilt per call (BB-07) | traced Qwen decode calls repeat 1920x | memoized descriptor keyed as in Host-side obligations |
| Structural impossibilities | none beyond `feature_spec.INVALID` | — |
