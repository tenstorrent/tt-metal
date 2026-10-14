# Operation Design: tilize

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (one compute phase: the tilize LLK re-lays faces; no arithmetic) |
| Goal | Re-lay a ROW_MAJOR tensor (or re-tile a TILE tensor) into TILE layout at the requested tile height, in one device dispatch, filling the grid on every work geometry the declared split reaches. |
| Math | `out[i, j] = x[i, j]` over the data region (`cast(x[i, j], out_dtype)` when `dtype=` is given); `out[p] = pad_value` over the pad region |
| Mode | Derivative (generic_op + ProgramDescriptor, Python entry point) |
| References | `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/generic_op_template/`, `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `eval/golden_tests/tilize/feature_spec.py`, `ttnn/ttnn/operations/examples/master.md` (entries cited per knob below) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on device; ROW_MAJOR, or TILE together with `tile=` | — | host |
| `memory_config` | `ttnn.MemoryConfig \| None` | no | any interleaved / sharded config | input's `memory_config()` | host |
| `dtype` | `ttnn.DataType \| None` | no (kw) | same-family pairs only (feature_spec INVALID prunes the rest) | input dtype (fp8_e4m3 → float32) | host → CB formats |
| `low_l1` | `bool` | no (kw) | — | `False` | host → `block_width_tiles` cap |
| `output_padded_shape` | `list[int] \| ttnn.Shape \| None` | no (kw) | ≥ input shape per dim; last two dims multiples of (tile_h, 32) | `None` | host → reader RT |
| `pad_value` | `float \| int \| None` | no (kw) | any; integer negatives bit_cast to unsigned | `None` | reader RT (fill encoded in the input format; see Dataflow) |
| `tile` | `ttnn.Tile \| None` | no (kw) | height ∈ {32,16,8,4,2,1}, width = 32 | `ttnn.Tile([32, 32])` | host → CB tile dims |

Index-axis canonicalization: none (no `dim` argument).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 0..6. Without a pad argument: `shape[-1] % 32 == 0`, `shape[-2] % tile_h == 0`, rank ≥ 2. |
| Dtype | TARGET `dtype`: bfloat16, float32, fp8_e4m3, uint32, int32, uint16, uint8. **Phase 0: bfloat16.** |
| Layout | ROW_MAJOR (TILE = retile path, requires `tile=`). **Phase 0: ROW_MAJOR.** |
| Memory | interleaved or sharded, DRAM or L1. **Phase 0: interleaved DRAM.** |

### Output

| Property | Value |
|----------|-------|
| Shape | logical = input logical shape (never grown); padded = `output_padded_shape`, else last two dims rounded to (tile_h, 32), rank<2 synthesized to `[32, 32]` / `[32, round_up(W,32)]` |
| Dtype | `dtype` if given, else input's (fp8_e4m3 → float32) |
| Layout | TILE_LAYOUT with tile `tile` (default 32×32) |
| Memory | `memory_config` if given, else input's |

## Blocking Model

The work is the 2-D grid of OUTPUT tiles, `tensor_rows_tiles × tensor_cols_tiles`:

```
tensor_rows_tiles = prod(padded_shape[:-2]) * ceil(padded_shape[-2] / tile_h)     # per-image ceil, then fold
tensor_cols_tiles = ceil(padded_shape[-1] / 32)
```

(Phase 0 has no padding, so `padded_shape == logical shape`; the formulas are already the per-image, ceil form so the padding refinement does not touch them.)

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `rows` — tile-rows; leading dims × `ceil(H/tile_h)` folded into one axis | independent — each output tile depends only on its own `tile_h` input sticks; the fold is legal because every image is tilized identically and image boundaries fall on tile-row boundaries (per-image ceil) | `block_height_tiles` | `core_rows_tiles` — the core's whole row assignment (one block spans it) | host: `block_height_tiles = core_rows_tiles` → reader/compute/writer RT arg | Phase 0: **split across the grid** with `ttnn.split_work_to_cores(grid, tensor_rows_tiles, row_wise=True)` (two core groups) | knob-turn (block_height below the assignment: only if the overlap lamp says so) |
| `cols` — tile-columns | independent — a tile's 32 columns depend on no other column | `block_width_tiles` (+ derived `last_block_width_tiles`) | `min(core_cols_tiles, max_block_width_tiles)` → for every Phase 0 shape this is the whole width (one column block) | host: `block_width_tiles` computed once → compute CT arg (helper template), reader/writer CT arg, CB page counts | Phase 0: **not assigned across cores** — every core owns all columns (`core_cols_tiles = tensor_cols_tiles`). Reason: Phase 0's declared `tile_grid` domain (single_tile / small / tall_narrow) has `R ≥ 16·C` or `R·C < 256`; the row axis carries the work there and a column cut only fragments the stick reads. The kernels already take `(col_start_tile, core_cols_tiles)` RT args, so assigning columns is host-only. | **knob-turn** — the 2-D rectangle split (`regime interleaved_2d_split`) only changes the host's per-core rectangle |
| `stick_in_tile` — the `tile_h` sticks inside one tile-row | independent, but fixed by tile geometry, not a knob — one tile-row is the helper's atomic unit (tilize_helpers.hpp:120-125) | `tile_h` | 32 | host: `tile.tile_shape[0]` → CB tile dims + reader CT arg | inside one core (never split — it is the tile) | n/a (tiny tiles change its value, not its role) |

No dependent axis exists: tilize is a pure re-lay, nothing combines across blocks. No operand is reuse-shared: there is one input and it varies along every axis that is cut. Consequently there is no cross-core traffic in any regime and no combine row in the regime table.

**Operand-reuse check (mechanical):** split = rows; `input_tensor` varies along rows → not reuse-shared. Nothing to broadcast.

**Stall-shadow check:** the only waits are reader→DRAM (read barrier), compute→`cb_rm_sticks`, writer→`cb_tiles_out`, writer→DRAM (write barrier). The work independent of a reader barrier is the next tile-row's reads; filling that window is exactly the buffer-depth / transactions-in-flight lamp below. No reorder needs an algebraic permission (no arithmetic), so no precision gate.

**Every knob is a parameter.** `block_height_tiles`, `block_width_tiles`, `last_block_width_tiles`, `in_depth_tile_rows`, `out_depth_tile_rows`, the grid (`device.compute_with_storage_grid_size()` at runtime), the per-core rectangle `(row_start_tile, core_rows_tiles, col_start_tile, core_cols_tiles)` — none is inlined. Each is defined once on the host and every CB page count / trip count is derived from it.

**`max_block_width_tiles` (single source, host):**

```
per_col_tile_bytes   = in_depth_tile_rows * in_page_bytes + out_depth_tile_rows * out_page_bytes
max_block_width_tiles = low_l1 ? LOW_L1_BLOCK_WIDTH_TILES                      # 8  (constant ⇒ O(1) L1)
                               : min(FAST_TILIZE_MAX_WIDTH_TILES,              # 255 (mechanism cap)
                                     L1_BLOCK_BUDGET_BYTES // per_col_tile_bytes)   # 512 KiB budget
max_block_width_tiles = align_down(max_block_width_tiles, dram_align_unit_tiles)   # mechanism cap, below
block_width_tiles      = min(core_cols_tiles, max_block_width_tiles)
num_col_blocks         = ceil(core_cols_tiles / block_width_tiles)
last_block_width_tiles = core_cols_tiles - (num_col_blocks - 1) * block_width_tiles
```

Phase 0 bf16 32×32: `per_col_tile_bytes = 2·2048 + 2·2048 = 8 KiB` → cap 64 tiles. Every Phase 0 INPUTS shape has `C ≤ 4`, so one column block = the whole width. `low_l1_forcing_width` / `short_wide_l1_forcing` (C = 256) land on `num_col_blocks = 4` (bf16) / 8 (fp32) under `low_l1=False` — bounded, never OOM.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_rm_sticks` | `in_depth_tile_rows` (capacity = `in_depth_tile_rows × block_width_tiles` pages) | 2 | the reader fills tile-row k+1 while compute tilizes tile-row k (catalog `double_buffer/`: 2× buffering + batched reads gives 2.78× over read-then-wait) |
| `cb_tiles_out` | `out_depth_tile_rows` (capacity = `out_depth_tile_rows × block_width_tiles` pages) | 2 | compute packs tile-row k+1 while the writer drains tile-row k |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Fast-tilize LLK legality (`can_use_fast_tilize`, tilize_helpers.inl:65-69) | `block_width_tiles < 256` | `FAST_TILIZE_MAX_WIDTH_TILES = 255` in `max_block_width_tiles` | helper silently selects the slow `tilize_block` path (correct, slower) |
| Helper width is a **template** param (tilize_helpers.hpp:187-197) | `block_width_tiles` must be one compile-time value per call site | ragged last column block uses a second CT arg `last_block_width_tiles` (second helper instantiation) — never shrink `block_width_tiles` to a divisor of C | prime C would collapse the block to 1 tile |
| DRAM/L1 read alignment of a stick **segment** (only when `num_col_blocks > 1`) | `block_width_tiles · 32 · in_elem_bytes` must be a multiple of `ttnn.get_dram_alignment()` (L1 source: `get_l1_alignment()`) so every segment offset `col_start · 32 · in_elem_bytes` is aligned | `dram_align_unit_tiles = align / gcd(align, 32·in_elem_bytes)`; `max_block_width_tiles = align_down(…, dram_align_unit_tiles)` (bf16 / fp32 / 16-bit: 1; uint8 on 64 B-aligned BH: 2) | misaligned NoC read → **wrong bytes**, silently |
| Helper asymmetric-page mode hard-codes 32 rows per tile-row (tilize_helpers.inl:200, 219) | `tile_h` | design uses the **symmetric** mode: `cb_rm_sticks` pages are tile-sized (`tile_h × 32 × in_elem_bytes`), `tile_h` sticks per tile-row, `block_width_tiles` pages per tile-row | with asymmetric pages a `tile_h < 32` tile-row would wait for 32 sticks → wrong tiles / hang |
| Fast tilize requires 32×32 output tiles and non-fp32 output (tilize_helpers.inl:67-68) | `tile_h`, output dtype | none needed — the helper falls back to `tilize_block` automatically | n/a (correct, slower) |
| fp32 fast tilize truncates to tf32 (tilize_helpers.hpp:53-71) | output dtype float32 / 32-bit int | refinement must set `fp32_dest_acc_en=True`, `UnpackToDestFp32` on `cb_rm_sticks`, `Fp32Mode::Lossless` | **wrong low mantissa bits** (spec requires bit-identical) |

### Regimes

Named boundary for every row: **DRAM/L1 tensor buffers — the minimum is "each input byte read once, each output byte written once, no cross-core traffic"**. Busiest-core share is `ceil(R_core) × C` tiles for row splits.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `interleaved_row_split` | **built** (Phase 0) | input and output INTERLEAVED (any buffer type); `tile_grid ∈ {single_tile, small, tall_narrow}`; RM input | `block_height_tiles = core_rows_tiles` × `block_width_tiles = min(C, max_block_width_tiles)` | **minimum** for both tensors. Input read as whole sticks (one NoC read per stick, `W·elem` bytes) when `num_col_blocks = 1`. Busiest core: `ceil(R / cores)` tile-rows × C tiles. Cores used = `min(R, grid)`. | One tilize init/uninit + reconfig per block (intended: **once per core** when `num_col_blocks = 1`); one CB reserve/push + one read barrier + one write barrier **per tile-row** (the streaming quantum); longer stick segments per NoC read as `block_width_tiles` grows (catalog `compute_block_size/`: one pass vs per-tile-row passes 1.65×) |
| `interleaved_2d_split` | **built** (Refinement 1: host selector `_choose_split` minimizes the busiest core's tiles when `R < grid`; cores grouped by column extent, one kernel/CB set per group; segment reader rotates each core's start stick, `SEGMENT_BANK_STAGGER`). Original deferral reason: on Phase 0's declared domain (`R ≥ 16C` or `R·C < 256`) the row split already places `min(R, grid)` cores and a column cut adds only fragmented reads; it is required for `short_wide` (row axis cannot fill the grid: `R ≤ C/16`) and `square_large`, which Phase 0 refuses by spec. Reachability: kernels already take the per-core rectangle `(row_start_tile, core_rows_tiles, col_start_tile, core_cols_tiles)`; only the host assignment changes (knob-turn). Also fills BH/large grids on `tall_narrow` when `R < grid` (lamp below). | any interleaved case where `R < num_cores` and `R·C ≥ num_cores` (selector: choose `cores_r × cores_c ≤ grid` maximizing `min(R,cores_r)·min(C_blocks,cores_c)`, prefer larger `cores_r` on ties) | per core: `block_height_tiles = core_rows_tiles` × `block_width_tiles = min(core_cols_tiles, max_block_width_tiles)` | bytes: **minimum** both tensors. Transactions: each stick read in `cores_c` segments of `core_cols_tiles·32·elem` bytes instead of 1 (more, smaller reads; same bytes). Busiest core: `ceil(R/cores_r) × ceil(C/cores_c)` tiles. Catalog `width_split/`: 6.25× at C=64, 7.76× at C=256 on 64 cores vs row split. | same as above; plus longer segments per read (fewer transactions per byte) |
| `low_l1_streaming` (a knob value of both regimes above, listed because it is selected by a caller flag) | **deferred** — knob-turn; Phase 0 SUPPORTED `low_l1 = [False]` per spec. Reachable: only `max_block_width_tiles` changes (`LOW_L1_BLOCK_WIDTH_TILES = 8`). | `low_l1 == True` | `block_height_tiles = core_rows_tiles` × `block_width_tiles = min(core_cols_tiles, 8)` | minimum bytes; more segments per stick when C > 8 | same; capped at 8 to keep L1 O(1) |
| `sharded_same_spec_in_place` | **built** (Refinement 2: `_shard_rects` maps every shard to its core's tile rectangle; both CBs backed on the shards, reader = one reserve/push of the whole shard, writer = one wait/pop; zero NoC traffic). Original note: compute kernel unchanged; reader/writer degenerate to CB push/no-op with `cb_rm_sticks` / `cb_tiles_out` backed on the shards via `ttnn.cb_descriptor_from_sharded_tensor`. | both sharded, same grid, same per-core rectangle in tile space, same orientation (HEIGHT/WIDTH/BLOCK/ND alike) | one block = **the whole resident shard**: `block_height_tiles = shard_rows_tiles`, `block_width_tiles = shard_cols_tiles` (clamped by the 255 fast-tilize cap; partial final shard = shorter runtime extent, same nominal push counts) | **zero** NoC/DRAM traffic — below the interleaved minimum; consumed and produced in place | one init per core; no reads/writes at all |
| `sharded_input_pinned` | **built** (Refinement 2; resident `cb_rm_sticks`, TensorAccessor writer — interleaved, L1/DRAM-sharded output) | input sharded on L1, output interleaved or differently sharded, AND `input_grid_cores ≥ min(R, grid)/2` (named knob `PIN_OCCUPANCY_RATIO = 1/2`) | the input shard (one block) | input: **0 crossings** (in-place CB); output: written once over NoC (min) | as above |
| `sharded_output_pinned` | **built** (Refinement 2; resident `cb_tiles_out`, TensorAccessor reader — interleaved or sharded input, pad fill included) | output sharded on L1, input interleaved or differently sharded, input not pinned, `output_grid_cores ≥ min(R, grid)/2` | the output shard's tile rectangle (one block); input sticks/segments read over NoC (shard rows of a sharded RM input are contiguous → one read per shard row-run, catalog `read_coalescing/`) | input read once (min); output **0 crossings** | as above |
| `sharded_full_grid` | **built** (Refinement 2; the interleaved split with sharded TensorAccessors; reader splits each stick segment at RM page boundaries for width/block-sharded inputs) | any sharded side whose grid fails the pin ratio (e.g. input+output on 2 cores while R ≥ 64) | as `interleaved_2d_split` with sharded TensorAccessors; any core holding a shard reads it locally | both tensors cross once; busiest core `total / grid` | as above |
| `retile_tile_input` | **built (Refinement 4)** — reader de-face stage; a resident TILE shard (holding whole `lcm` units) is the de-face source in place, otherwise tiles are read through a TensorAccessor into a 2-slot trid staging ring. Originally deferred. Reachable: same CBs and compute; the reader reads input TILE pages (`in_tile_h × 32` per face-pair) and de-faces them in L1 into the RM row layout `cb_rm_sticks` expects (local L1 copies, no DRAM round-trip). Unit = one output tile-row = `lcm(in_tile_h, tile_h)/tile_h` output tile-rows per staging step. | `input.layout == TILE` | `block_height_tiles = core_rows_tiles` (rounded to `lcm(in_tile_h, tile_h)` rows) × `block_width_tiles` | minimum: input read once, output written once; extra local L1 traffic (de-face copy) | same; plus de-face setup per staging step |
| `whole_width_unconditional` | **rejected** — dead end, superseded by `block_width_tiles` capping in every regime: an unconditional full-width CB is 2·256·4 KiB·2 = 2 MiB at C=256 fp32 → OOM on `low_l1_forcing_width` / `short_wide_l1_forcing` | — | `block_width_tiles = C` | — | — |
| `single_core` | **rejected** — superseded by `interleaved_row_split`; produces identical bytes at 1/64-1/110 of the throughput | — | — | — | — |
| `host_or_two_dispatch` (to_layout / pad / untilize + tilize) | **rejected** — forbidden by spec (one native dispatch; no layout-converting ttnn op; no host round trip) | — | — | — | — |

**Selection function (host, `tilize.py`):**

```
if input.layout == TILE:                                  -> retile_tile_input            (built, Refinement 4)
elif both sharded and same grid & per-core tile rectangle -> sharded_same_spec_in_place   (built, Refinement 2)
elif input sharded (L1) and pin ratio holds               -> sharded_input_pinned         (built, Refinement 2)
elif output sharded (L1) and pin ratio holds              -> sharded_output_pinned        (built, Refinement 2)
elif any side sharded                                     -> sharded_full_grid            (built, Refinement 2)
elif R < num_cores and R*C_blocks >= num_cores            -> interleaved_2d_split         (built, Refinement 1)
else                                                      -> interleaved_row_split        (built)
```

Phase 0 builds only the last line; validate() refuses the inputs that would reach the others (they are outside SUPPORTED). Regime-pinned tests are required as each deferred regime lands (the 2D split triggers on `R < num_cores`, which depends on the grid: test it with a shape where `R=1` so it triggers on every arch).

### Traffic ranking

| Candidate split | Input DRAM crossings | Output DRAM crossings | Read transactions | Cross-core | Grid fill |
|-----------------|----------------------|----------------------|-------------------|------------|-----------|
| rows only (built) | 1 | 1 | `R·tile_h` (one per stick) — the fewest possible on interleaved RM (consecutive sticks are different pages/banks, so no coalescing across sticks: catalog `read_coalescing/`) | none | `min(R, grid)` |
| 2D rows × cols | 1 | 1 | `R·tile_h·cores_c` | none | `min(R·C_blocks, grid)` |
| cols only | 1 | 1 | `R·tile_h·min(C, grid)` | none | `min(C_blocks, grid)` |
| dependent-axis split + combine | n/a — no dependent axis | | | | |

Bytes are equal across all candidates (every split reads each input byte and writes each output byte once); they differ only in transaction count. Rows-only moves the bytes in the fewest, largest transactions, so it is chosen wherever it fills the grid — which is Phase 0's domain. The 2D split is the occupancy fix where it does not.

### Block schedule

Per core (logical schedule; reader / compute / writer each realize their part):

```cpp
// num_blocks_this_core = num_col_blocks (Phase 0: 1)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    // block = core_rows_tiles tile-rows × block_width_tiles tile-cols (last: last_block_width_tiles)
    load_sticks_block(block_idx);    // reader
    tilize_block(block_idx);         // compute
    store_tiles_block(block_idx);    // writer
}
```

| Operation | Block shape | Resident across it | Intended fixed-cost frequency |
|-----------|-------------|--------------------|-------------------------------|
| `load_sticks_block` | `core_rows_tiles × tile_h` sticks, each a segment of `block_width_tiles·32` elements starting at column `col_start_tile + block_idx·block_width_tiles` | `in_depth_tile_rows` tile-rows of segments in `cb_rm_sticks` | TensorAccessor setup once per kernel; reserve/push + one `noc_async_read_barrier` **per tile-row** (`tile_h` reads in flight per barrier) |
| `tilize_block` | `block_height_tiles × block_width_tiles` tiles, one helper call with `num_blocks = block_height_tiles` | nothing beyond the CB windows | `compute_kernel_hw_startup` once per kernel; tilize init + uninit + data-format reconfig **once per block** (Phase 0: once per core); wait/pop + reserve/push per tile-row (inside the helper) |
| `store_tiles_block` | same tiles, written as TILE pages `out_tile_id = r·tensor_cols_tiles + c` | `out_depth_tile_rows` tile-rows in `cb_tiles_out` | wait/pop + one `noc_async_write_barrier` **per tile-row** (`block_width_tiles` writes in flight) |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Transactions in flight (overlap) | Perf-focus shapes have 64-128 B sticks (`[1,1,16384,64]`, `[...,32]`): per-transaction cost dominates, and one barrier per 32 reads may starve the NoC | `in_depth_tile_rows = 4` with issue-ahead (barrier on tile-row k only after issuing k+1, or transaction-id waits) — catalog `double_buffer/`, `trid_double_issue/` |
| Split reader | Narrow sticks are issue-bound; catalog `split_reader/` shows 1.36× at 256 B, 1.74× at 32 B | Reader issues half the sticks of each tile-row on each RISC (writer kernel takes the other half of reads) |
| Grid synchronization (inverse) on `tall_narrow` with `R < grid` | e.g. `tall_narrow_grid_scale` on Blackhole (R = 64, ~110 cores): row split uses 64 cores | `interleaved_2d_split` with `cores_c = 2` once that regime is built |
| Grid synchronization on `single_tile` / `small` | one or a few tile-rows per core; launch + first-read latency dominates | fewer cores with more rows each (e.g. ≥ 2 tile-rows per core) |
| Block width cap (budget 512 KiB) | wider segments amortize per-read cost but leave less L1 for depth | `L1_BLOCK_BUDGET_BYTES = 256 KiB` vs 512 KiB on `short_wide_l1_forcing` once 2D lands |
| Bank stagger / NoC placement | All cores start on the same DRAM bank; catalog `bank_stagger/` (1.01-1.07× interleaved), `noc_placement/` (row-wise core order 2.9×) | per-core start offset `k % tile_h` in the stick walk; `row_wise=True` is the default already |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM → `cb_rm_sticks` | RM sticks, input dtype | reader (RISCV_1, NoC0): `TensorAccessor` on the input (page = one stick); per tile-row `reserve_back(block_width_tiles)`, `tile_h` × `noc_async_read(acc.get_noc_addr(stick_id, seg_offset_bytes), l1 + i·seg_bytes, seg_bytes)`, barrier, `push_back(block_width_tiles)` | `stick_id = img·H + tr·tile_h + i` with `r = img·ceil(H/tile_h) + tr` (alignment-aware form; Phase 0 reduces to `r·tile_h + i`). `seg_bytes = bw·32·in_elem_bytes`, `seg_offset_bytes = (col_start_tile + block_idx·block_width_tiles)·32·in_elem_bytes`. Ragged column block: same push count of the nominal tile-row (pages = `last_block_width_tiles` for the tail instantiation — the tail block uses its own width consistently on both sides). |
| `cb_rm_sticks` → `cb_tiles_out` | RM → TILE (faces), cast at pack time when formats differ | compute: `compute_lib::tilize<block_width_tiles, cb_rm_sticks, cb_tiles_out>(block_height_tiles)` (symmetric pages) | output format = output dtype on `cb_tiles_out`; the packer converts (`dtype=` refinement) — no byte copy |
| `cb_tiles_out` → DRAM | TILE pages, output dtype | writer (RISCV_0, NoC1): `TensorAccessor` on the output; per tile-row `wait_front(bw)`, `bw` × `noc_async_write_page(out_tile_id, acc, l1_read_addr)` (or `get_noc_addr` + `noc_async_write`), barrier, `pop_front(bw)` | |
| Pad fill (**built**, Refinement 1, `pad_mode ≠ none`) | in `cb_rm_sticks`, input format | reader writes the data part of each segment and fills the rest (W tail, H-tail rows, whole pad tile-rows / pad tile-cols) with the fill value encoded in the **input** format (integer negative: `uint32(int)` bit_cast); fill source = a one-stick fill row in L1 prepared once (`cb_pad_fill`: 64 B RISC seed + doubling local NoC copies). Data bytes of a segment are read exactly; tail fills are issued after the read barrier. Leading-dim padding decoded per tile-row (≤ 4 leading dims). Output allocated at the padded shape, then a metadata-only `ttnn.experimental.view(out, logical, padded)` (binding overload added in Refinement 1) restores the logical shape — no second dispatch. | logical shape stays the input's; `tensor_rows_tiles`/`cols` computed from the padded shape; rank 0/1 synthesized to `[1,1,32,W']` view |
| Sharded in/out (deferred) | same | CBs backed on the shard buffers (`ttnn.cb_descriptor_from_sharded_tensor`); a core whose input shard is in its L1 never re-reads it through a TensorAccessor; a shard that needs reshaping (partial final shard, shard width ≠ block) is copied locally into a staging CB, not re-read over NoC | |

No regime has inter-Tensix communication (no dependent axis, no reuse-shared operand).

### TARGET axes → regimes (what each unsupported value costs)

| TARGET axis value(s) | Class | Unlock |
|----------------------|-------|--------|
| `tile_grid: short_wide, square_large` | knob-turn | `interleaved_2d_split` (host per-core rectangle) |
| `low_l1: True` | knob-turn | `max_block_width_tiles = LOW_L1_BLOCK_WIDTH_TILES` |
| `buffer: dram_to_l1, l1_to_l1, l1_to_dram` (interleaved) | knob-turn | TensorAccessor already buffer-type agnostic; segment alignment uses `get_l1_alignment()` for L1 sources |
| `out_scheme: HEIGHT/WIDTH/BLOCK/ND`, `shard_api`, `orientation` | knob-turn (placement) — no axis is dependent, so every shard flavor's cut is self-contained | the four sharded regime rows |
| `rank: 2, 3, 5, 6` | knob-turn | fold leading dims into `rows` (host only) |
| `rank: 0, 1`, `pad_mode`, `pad_value`, `alignment ≠ tile_aligned` | reader extension (same scheme) | pad fill in `load_sticks_block` |
| `tile_height: 16..1` | knob-turn | CB tile dims `ttnn.Tile([tile_h, 32])` on both CBs; helper falls back to `tilize_block` |
| `in_tile_height ≠ none` | scheme-change (reader de-face stage) | `retile_tile_input` |
| `dtype`, `output_dtype` | knob-turn + precision config | CB formats; fp32/32-bit int: `fp32_dest_acc_en`, `UnpackToDestFp32`, `Fp32Mode::Lossless` |

### Structural impossibilities

None missed by `feature_spec.py`'s INVALID.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block = `core_rows_tiles` tile-rows × `block_width_tiles` tile-cols; the split unit is a tile-row |
| Grid | `device.compute_with_storage_grid_size()` at runtime (never hardcoded) |
| Per-core work | `(all_cores, core_group_1, core_group_2, rows_g1, rows_g2) = ttnn.split_work_to_cores(grid, tensor_rows_tiles, row_wise=True)`; each core gets the rectangle `row_start_tile, core_rows_tiles ∈ {rows_g1, rows_g2}, col_start_tile = 0, core_cols_tiles = tensor_cols_tiles` |
| Remainder | rows: handled by the two core groups; cols: `last_block_width_tiles` (CT). Tile counts: `tensor_rows_tiles = prod(padded_shape[:-2]) · ceil(padded_shape[-2]/tile_h)`, `tensor_cols_tiles = ceil(padded_shape[-1]/32)` — per-image ceil, never `floor(N·H/32)` |
| Cores used | `min(tensor_rows_tiles, grid_x · grid_y)`; SUPPORTED `tile_grid` lists only values where this reaches the grid or where the shape has fewer tiles than cores |
| Program cache | per-core rectangle and buffer addresses are runtime args; nothing address-dependent enters CT args or the hash |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_rm_sticks` | 0 | `tile_h · 32 · in_elem_bytes` (one tile of RM data; CB tile dims = output tile) | `in_depth_tile_rows · block_width_tiles` | spans `cols` (`block_width_tiles`), streams `rows` (window of `in_depth_tile_rows` tile-rows), spans `stick_in_tile` (`tile_h` via page size) | input dtype (Phase 0 `Float16_b`) | reader | compute | whole kernel |
| `cb_tiles_out` | 16 | output tile size (`tile_h × 32`, output dtype) | `out_depth_tile_rows · block_width_tiles` | spans `cols`, streams `rows`, spans `stick_in_tile` | output dtype (Phase 0 `Float16_b`) | compute | writer | whole kernel |

Inventory: two CBs is the minimum for a re-lay (one RM staging buffer the unpacker reads, one tile buffer the packer writes; the helper requires `input_dfb ≠ output_dfb`, tilize_helpers.hpp:99). No intermediate exists to eliminate. Full ledger: `l1_ledger.md`.

Sync: reader pushes `block_width_tiles` per tile-row = helper waits `block_width_tiles` per tile-row (symmetric mode, tilize_helpers.inl:216-235); helper pushes `block_width_tiles` per tile-row = writer waits `block_width_tiles`. Tail block: all three use `last_block_width_tiles`. Ring-wrap: every push is `bw` and capacity is `depth·bw` — closes for the main width; for the tail width the implementer must keep the cycle closed (e.g. reserve/push the nominal `bw` pages and only fill `last_block_width_tiles`, or size capacity so both quanta divide it) — see blocking-model §2 *Ragged tails*.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_sticks_block` | `block_height_tiles·tile_h` stick segments × `block_width_tiles·32` elements | no (dataflow; TensorAccessor + NoC reads) | input tensor (DRAM) | `cb_rm_sticks`, `block_width_tiles` pages per tile-row | pushed |
| 2 | `tilize_block` | `block_height_tiles × block_width_tiles` tiles | yes — `compute_kernel_lib::tilize` | `cb_rm_sticks`, `block_width_tiles` per tile-row, waited & popped by helper | `cb_tiles_out`, `block_width_tiles` per tile-row, reserved & pushed by helper | both drained per tile-row |
| 3 | `store_tiles_block` | same tiles | no (dataflow) | `cb_tiles_out`, `block_width_tiles` per tile-row | output tensor (DRAM) | popped |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| (compute boot) | helper prerequisite | `compute_kernel_hw_startup(icb, ocb)` | tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:106 | `(cb_rm_sticks, cb_tiles_out)` | — | — | none |
| `tilize_block` | helper | `compute_kernel_lib::tilize` | ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:187-197 (impl tilize_helpers.inl:75-251) | `<block_width_tiles, cb_rm_sticks, cb_tiles_out>` defaults (InitAndUninit, WaitBlock, UnpackAndPackReconfigure, Fp32Mode::Fast for Phase 0 bf16); RT `num_blocks = block_height_tiles`; `total_input_pages` omitted (symmetric) | `cb_rm_sticks` | `cb_tiles_out` | `block_width_tiles` (template), `num_blocks` = `block_height_tiles` (RT). Tail: second call with `<last_block_width_tiles,…>` |
| `load_sticks_block` | raw_api (dataflow) | `TensorAccessor::get_noc_addr(page_id, offset)`, `noc_async_read`, `noc_async_read_barrier` | tt_metal/hw/inc/api/tensor/tensor_accessor.h:155; tt_metal/hw/inc/api/dataflow/dataflow_api.h:565, 2083 | stick id, segment offset, `seg_bytes` | DRAM | `cb_rm_sticks` | `block_width_tiles` (segment length), `in_depth_tile_rows` |
| `store_tiles_block` | raw_api (dataflow) | `noc_async_write_page` / `noc_async_write`, `noc_async_write_barrier` | tt_metal/hw/inc/api/dataflow/dataflow_api.h:1558, 952, 2113 | tile id | `cb_tiles_out` | DRAM | `block_width_tiles`, `out_depth_tile_rows` |

Helpers considered and rejected for the dataflow rows: the kernel-lib dataflow helpers are `reduce_helpers_dataflow.hpp` (scaler prep — not applicable) and `mcast/` (no cross-core traffic in this op). No kernel-lib helper covers interleaved stick-segment reads or tile-page writes; the dataflow APIs above are the intended mechanism.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Bit-exactness on float32 / 32-bit ints | Fast tilize truncates fp32 → tf32; standard unpack through SrcA also truncates | float32/int refinements: `fp32_dest_acc_en=True`, `UnpackToDestMode.UnpackToDestFp32` on `cb_rm_sticks`, `Fp32Mode::Lossless` (tilize_helpers.hpp:53-71); never combine fast tilize with UnpackToDestFp32 (tilize_helpers.inl:117-127 static_assert) |
| Asymmetric-page mode assumes 32-row tile-rows | tiny tiles (`tile_h < 32`) would wait for 32 sticks | symmetric pages (tile-sized), `tile_h` sticks per tile-row, chosen from Phase 0 |
| Row count from `shape[-2]` alone | `square_large_from_leading_dims` reaches R through the leading-dim fold with an H tail | `tensor_rows_tiles = prod(shape[:-2]) · ceil(H/tile_h)` and per-image stick indexing |
| Segment misalignment when columns are blocked | appears only when `num_col_blocks > 1` (C > cap) — not in Phase 0 tests | `dram_align_unit_tiles` clamp on `max_block_width_tiles` |
| `tile_grid` SUPPORTED drift | every geometry passes under any split | Phase 0 SUPPORTED `tile_grid = ["single_tile", "small", "tall_narrow"]`; validate() raises NotImplementedError for `short_wide` / `square_large` |
| Ragged column tail ring-wrap | tail pushes `last_block_width_tiles` into a CB sized `depth·bw` | keep the cycle closed (see Circular Buffers); only reachable when C > cap |
| Program cache | addresses or per-call work split baked into CT args → cache miss / stale addresses | addresses and the per-core rectangle are RT args; CT args are geometry only |
| Absent-argument sentinels | `pad_mode`, `pad_value`, `shard_api`, `orientation`, `in_tile_height` must tag `"none"` | SUPPORTED includes `"none"` on all five; validate never refuses `"none"` |
| Malformed vs unsupported | spec wants ValueError/RuntimeError for malformed calls, NotImplementedError for SUPPORTED gating | validate() checks malformed conditions (layout, on-device, alignment without pad, `output_padded_shape` too small, bad `tile`) first, raising ValueError, then the SUPPORTED gate |
