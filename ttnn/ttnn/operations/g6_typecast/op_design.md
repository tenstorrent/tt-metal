# Operation Design: g6_typecast

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement with a per-value format conversion (unpack → [SFPU convert] → pack) |
| Goal | Convert every value of one tensor to `dtype` in exactly one device program, keeping shape, layout and memory placement; bit-identical to `torch.Tensor.to` except bfloat8_b (≤ 1 quantization step per group of 16). |
| Math | `out[i] = conv_{src→dst}(x[i])` for every logical index `i`; Phase 0 pairs: bf16→f32 `v` (exact), bf16→bf8b block rendering, bf16→int32 `trunc(v)` |
| References | `docs/explorations/g6_typecast/{DOSSIER,tensix,risks,math}.md`; `eval/golden_tests/g6_typecast/{feature_spec,helpers}.py`; `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h`; `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.{hpp,inl}`; `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp`; `ttnn/ttnn/operations/examples/master.md` (entries `double_buffer`, `noc_placement`, `compute_block_size`, `bank_stagger`) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 1–4, on device, interleaved DRAM or L1 | — | buffer address RT; formats CT |
| `dtype` | `ttnn.DataType` | yes (positional or keyword) | TARGET `output_dtype` values | — | CT (selects CB output format, `fp32_dest_acc_en`, SFPU op) |
| `sub_core_grids` | `ttnn.CoreRangeSet \| None` | no, keyword-only | `None` only | `None` | — (any other value → `ValueError`) |

Entry-point validation (before `validate()`'s support check, raising `ValueError`): `dtype` not a `ttnn.DataType`; `sub_core_grids is not None`; rank outside 1–4. Then `validate()` (registry model, `eval/op_template.py`) raises `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract`. The scenario dict validate() builds: `{"input_shape": tuple(input_tensor.shape), "dtype": input_tensor.dtype, "output_dtype": dtype, "layout": input_tensor.layout, "buffer": "l1" if buffer_type == L1 else "dram"}`; taggers are exactly those listed in the spec (`tag_dtype`, `tag_output_dtype`, `tag_layout`, `tag_buffer`, `tag_alignment`, `tag_rank`), names matching TARGET in `feature_spec.py`. EXCLUSIONS stays empty in every phase.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(…, H, W)`, rank 1–4 (Phase 0: 3, 4); H, W any positive (non-tile-aligned handled natively) |
| Dtype | Phase 0: bfloat16. TARGET: bfloat16, float32, int32, uint32, uint16 |
| Layout | Phase 0: TILE. TARGET: TILE, ROW_MAJOR |
| Memory | interleaved, DRAM or L1 |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` |
| Dtype | `dtype` |
| Layout | `input_tensor.layout` |
| Memory | `input_tensor.memory_config()` (same interleaving, same buffer type) — allocated with `ttnn.allocate_tensor_on_device(shape, dtype, layout, device, input_tensor.memory_config())` |

## Blocking Model

The op is elementwise: no output value depends on any other input value. The one intra-value
coupling is the bfloat8_b shared exponent (16 consecutive values of one tile face row), which lives
entirely inside a tile and is owned by the packer. Every tile-index axis is therefore independent,
and the interleaved page order already linearizes them: TILE pages are numbered row-major over
`(leading, tile_row, tile_col)`, so all tile axes fold into one **page axis** with no change of
meaning. A block is a contiguous run of pages of one core's contiguous page range.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| leading dims (N·C, `tensor_leading`) | independent — no result spans images | folded into `block_tiles` (page axis) | folded | — (folded) | spread with the page axis | knob-turn |
| tile rows per image (`ceil(H/32)`) | independent — elementwise | folded into `block_tiles` | folded | — (folded) | spread with the page axis | knob-turn |
| tile cols (`ceil(W/32)`) | independent — elementwise; bf8b groups never cross a 16-value face column, so a tile boundary never cuts a group | folded into `block_tiles` | folded | — (folded) | spread with the page axis | knob-turn |
| **page axis** (`tensor_pages = prod(leading)·ceil(H/32)·ceil(W/32)` = `output.buffer_num_pages()`) | independent — the fold of the three above | `block_tiles` | `min(core_pages_max, BLOCK_TILES_DEFAULT=8)` | host constant `BLOCK_TILES_DEFAULT` in the program-descriptor module → CT arg `block_tiles` to reader, compute, writer; CB page counts derived from it | contiguous page ranges over the full grid via `ttnn.split_work_to_cores(grid, tensor_pages, row_wise=True)`; `num_cores = min(tensor_pages, grid cores)` | knob-turn (block size, depth, grid) |
| intra-tile elements (32×32; bf8b group = 16 values of a face row) | independent except the bf8b group, which is packer-internal | whole tile (1024 values) | 1 tile | hardware unit | never split | none — a tile is the atomic unit of unpack/pack and of the bf8b exponent |

Why `block_tiles` is not the whole per-core assignment (the bound): the op has **no reuse** — every
tile is read once, converted once, written once — so the only thing a bigger block buys is fewer
CB handshakes and fewer NoC barriers. A block equal to the whole assignment serializes read →
convert → write with zero overlap (overlap lamp). The measured catalog entry `double_buffer` puts
the streaming sweet spot at 4–8 tiles per barrier with CBs of `2·block`; 8 is the top of that range.
The L1 bound (≈109 tiles at the Phase 0 widest pair, see `l1_ledger.md`) is far above, so the value
is an overlap choice, recorded as a lamp, not an L1 response. For decode shapes (1–16 tiles,
one tile per core) the `min` makes the block equal the whole per-core assignment.

Compute is **not** chunked per block at the call level: one helper call per kernel walks the core's
whole assignment (`IterationShape::tiles(core_pages).block_size(block_tiles)`), so the typecast
init and data-format configuration run **once per core**, and only the CB handshake repeats per block.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `in_depth_blocks` (host constant, CT-derived capacity) | 2 | reader fills block k+1 while compute converts block k |
| `cb_output_tiles` | `out_depth_blocks` (host constant) | 2 | compute packs block k+1 while writer drains block k |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `typecast_tile` 32-bit formats (`typecast.h:63`: "For input/output to be UInt32, Int32, or Float32, Dest must be in 32 bit mode") | DEST width, not an extent | `fp32_dest_acc_en = (src ∈ {float32,int32,uint32}) or (dst ∈ {float32,int32,uint32})`; Phase 0: true for →float32, →int32; false for →bfloat8_b | int32/float32 values cut to 16 bits — wrong bits (R2) |
| Unpack path through srcA (19-bit) | source precision (refinement sources) | `unpack_to_dest_mode = UnpackToDestFp32` on `cb_input_tiles` when src ∈ {float32, int32, uint32}; Phase 0 (bf16 src): Default | float32 mantissa truncated to 10 bits before RNE; int32 bits mangled |
| bf8b shared exponent = 16 values of a face row | tile is atomic; never cut below 1 tile | block ≥ 1 whole tile (always true) | n/a |
| DEST slots per acquire (`DEST_AUTO_LIMIT`, halved under fp32 DEST) | tiles per dst-sync window | owned by the eltwise chain (block_size groups CB sync, the chain sizes its own DEST windows) | n/a — not a design knob |
| CB ring-wrap | pushed count per block | every push is `block_tiles` except the core's last block (`last_block_tiles`), which is the final push of the kernel; reader/compute/writer use the same `last_block_tiles` (compute: `BlockTailSync::ValidTiles`) | a mid-stream partial push would desynchronize the ring |
| Interleaved page alignment (DRAM 64 B / L1 16 B) | none for TILE (tile pages 1088/2048/4096 B are aligned) | — | ROW_MAJOR refinement only (R7) |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **TILE stream** (unpack → typecast chain → pack) | **built** (Phase 0) | `layout == TILE` and `(src,dst)` has a `typecast_tile` arm or an unpack/pack no-op arm (`typecast.h:150, 158, 315`) | `block_tiles` pages of the core's contiguous range | **minimum**: input crosses its memory (DRAM or L1 banks) once, output once; no DRAM staging, no re-read. For L1 placements, `(1 − 1/num_l1_banks)` of the pages cross the NoC between cores (see bank-affine row) | per block: one CB reserve/push + wait/pop pair on each CB, one `noc_async_read_barrier` (reader) and one write flush (writer). Init/reconfig: once per core (single helper call). Intended frequency: inits 1/core, handshakes 1/block, barriers 1/block |
| TILE bit copy (uint32↔int32 tiles, Q2230) | deferred — not in Phase 0 SUPPORTED (dtype uint32 is a refinement); built by the dtype refinement | `layout == TILE` and `{src,dst} == {uint32,int32}` | as TILE stream | minimum | same as TILE stream minus compute (reader → one CB → writer, no compute kernel; the 32 bits are the result by definition, no `typecast_tile` arm exists for this pair) |
| TILE bank-affine (L1 placements) | deferred — the contiguous split is correct for every L1 shape; the bank-affine split is a pure traffic refinement that needs the host-side L1 bank → logical core map, which must be verified per arch/harvest. Reachable: reader/writer take `(start_page, page_stride, num_pages)` RT args from Phase 0 (stride = 1) | `buffer == "l1"` | `block_tiles` pages of the core's own bank: `page = bank_id + j·num_l1_banks` | removes all cross-core NoC traffic for L1 placements (p4: 16 MiB in + 8.5 MiB out become core-local) | same as TILE stream |
| Masked bf8b pack | deferred — no production case reaches it (every bf8b case has W ∈ {128, 256}) and inputs produced by `ttnn` carry zero tile padding, which sets no exponent | `dst == bfloat8_b and W % 16 != 0` | as TILE stream | minimum | adds one mask element in the chain (multiply by a column mask tile) so padding never sets a group's exponent |
| ROW_MAJOR flat stream | deferred — ROW_MAJOR is not in Phase 0 SUPPORTED; built by the layout refinement | `layout == ROW_MAJOR` | `block_rows` RM pages gathered into a contiguous L1 run, converted as 1024-value pseudo-tiles (elementwise with identical in/out ordering, so no tilize/untilize), scattered back as RM pages | minimum in bytes; transfer count is the cost (p1: 4 194 304 pages of 6 B in / 3 B out, each in its own 64 B DRAM slot). Successor transfer shape: per-bank spans — pages `i, i+nb, i+2nb…` are consecutive 64 B slots of one bank, one read of `64·k` B moves k pages; DM RISC compacts slots | per block: transfer count ÷ k, barrier count, handshakes |
| ROW_MAJOR bit copy | deferred — as above (refinement) | `layout == ROW_MAJOR` and `{src,dst} == {uint32,int32}` | `block_rows` pages | minimum | no compute kernel |
| Whole-assignment resident block | rejected — superseded by TILE stream: no operand is reused, so residency saves no traffic and serializes read/convert/write | — | `core_pages` | minimum, same as stream | nothing beyond TILE stream at `block_tiles=8`; loses overlap |
| DRAM-staged layout round trip (to_layout/tilize/untilize around a tile conversion) | rejected — dead end, forbidden (spec rule 4, BB-01/02); superseded by ROW_MAJOR flat stream | — | — | 3× DRAM crossings per tensor | — |
| DM-RISC scalar conversion for TILE | rejected — dead end: 1024 scalar conversions per tile on a 32-bit RISC vs. hardware unpack/pack + one SFPU pass; superseded by TILE stream | — | — | minimum | — |
| Dependent-axis split + combine | rejected — not applicable: no axis is dependent (no result spans any axis) | — | — | — | — |

Regime selection (host, exact): `layout == TILE → TILE stream` (Phase 0's only built regime; the
compute kernel's chain is selected at compile time from the CB formats). Later refinements add the
predicates above; the op exposes the selected plan (`regime`, `num_cores`, `block_tiles`) via a
module-level `last_plan()` / plan object (BB-08), and any refinement adding a second built regime
adds regime-pinned tests.

### Traffic ranking

| Candidate split | DRAM crossings per tensor | Cross-core NoC | Fan-out | Rank |
|-----------------|---------------------------|----------------|---------|------|
| Page axis, contiguous ranges over the full grid (**chosen**) | x: 1, out: 1 | DRAM placement: none. L1 placement: `(1 − 1/nb)` of pages each way (pages live round-robin on all L1 banks) | 1 reader + 1 writer stream per core | 1 for DRAM placements; 2 for L1 |
| Page axis, bank-affine strided sets | x: 1, out: 1 | none for L1 placements (each core reads/writes its own bank); same as contiguous for DRAM | same | 1 for L1 placements (deferred row) |
| Tile-column (width) split | x: 1, out: 1 | as contiguous | same; fragments contiguous page runs; useless for W ≤ 32 (Wt = 1 for 23 of 40 Phase 0 cases) | 3 |
| Dependent-axis split + combine | n/a — no dependent axis | — | — | — |

Chosen: contiguous page ranges — it reaches the DRAM minimum for every DRAM case and is correct for
every L1 case; the cheaper L1 split is deferred with its delta. No operand is reuse-shared (the
operand-reuse check: the only input varies along every axis; `dtype` is a compile-time constant), so
there is no broadcast row. Stall-shadow: the only waits are CB waits between independent pipeline
stages; with depth 2 the next block's read runs in the shadow of the current block's conversion and
write — nothing else is independent of the waited data.

### Block schedule

```cpp
// per core: pages [start_page, start_page + core_pages), num_blocks_this_core = ceil(core_pages / block_tiles)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);      // reader: block_extent pages → cb_input_tiles
    convert_block(block_idx);   // compute: unpack (src fmt) → [SFPU typecast] → pack (dst fmt) → cb_output_tiles
    store_block(block_idx);     // writer: cb_output_tiles → block_extent output pages
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_block` | `block_extent = (block_idx < last) ? block_tiles : last_block_tiles` pages | nothing persists between blocks | 1 `cb_reserve_back`/`cb_push_back` per block; `block_extent` async reads, **one** `noc_async_read_barrier` per block; TensorAccessor built once per kernel |
| `convert_block` | same `block_extent` tiles | nothing | `compute_kernel_hw_startup` once per kernel; typecast/copy init + data-format reconfig **once per kernel** (one helper call over `core_pages` with `.block_size(block_tiles)`); CB wait/pop + reserve/push once per block; dst-sync windows owned by the chain |
| `store_block` | same `block_extent` pages | nothing | 1 `cb_wait_front`/`cb_pop_front` per block; `block_extent` async writes, one `noc_async_writes_flushed` before the pop; **one final `noc_async_write_barrier`** at kernel end (BB-03) |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (block size) | `block_tiles = 8` is below the L1 bound (~109); larger blocks halve barrier/handshake counts but delay pipeline fill | `block_tiles ∈ {4, 16, 32}` and `in/out_depth_blocks = 3` on p3 (Q2185, 2048 tiles) and p4 (P39, 8192 tiles) |
| Grid synchronization | decode shapes (1–16 tiles) put one tile on each of up to 16 cores; per-core launch cost may dominate | the same shapes on `min(tensor_pages, 4)` cores |
| NoC placement | `row_wise=True` is the catalog default (`noc_placement`); Blackhole may prefer a diagonal | `row_wise=False` / diagonal assignment on p3 |
| Bank stagger | contiguous ranges all start on bank `start_page % nb`; on Blackhole height-split rows may align on a bank (`bank_stagger`) | start each core's walk at `core % block_tiles` within its range and wrap |
| fp32 DEST for →bf8b | Phase 0 runs →bf8b with 16-bit DEST (2× DEST slots); bf16 values are exact in 16-bit DEST | none needed unless the device packer rounding (R5) misses the 1-step gate; then fp32 DEST |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| input interleaved buffer (DRAM or L1 banks) → `cb_input_tiles` | src tile format (Phase 0 Float16_b, 2048 B) | reader (NCRISC, NoC0): `TensorAccessor` over `input_tensor`, `noc_async_read_page` per page, one barrier per block | page ids `start_page + i` (Phase 0 stride 1) |
| `cb_input_tiles` → DEST | unpack Float16_b → srcA → DEST (fp32 DEST when dst ∈ {f32, int32}) | compute: `CopyTile` element of the typecast chain | Phase 0 sources never need unpack-to-dest |
| DEST convert | f32: none (`typecast.h:158` no-op arm); bf8b: none (`typecast.h:315`); int32: SFPU `calculate_typecast_fp32_to_int32` (`typecast.h:150`), truncation toward zero | compute: `Typecast<Float16_b, dst>` element | |
| DEST → `cb_output_tiles` | pack to dst format (Float32 4096 B / Bfp8_b 1088 B / Int32 4096 B) | compute: `PackTile` element (pack reconfig once) | bf8b exponent computed by the packer per 16 values of a face row |
| `cb_output_tiles` → output interleaved buffer | dst tile format | writer (BRISC, NoC1): `TensorAccessor` over `output_tensor`, `noc_async_write_page` per page | final write barrier before return |

Placement as a TARGET axis: `buffer ∈ {dram, l1}` are both interleaved; there is no sharded value in
TARGET. Both are the same regime (only the TensorAccessor's bank set differs) — a knob-turn, built in
Phase 0. No inter-Tensix protocol exists or is needed (no mcast, no semaphores).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block of `block_tiles` tile pages; per-core assignment = a contiguous page range |
| Grid | `grid = device.compute_with_storage_grid_size()` (single source, passed to `split_work_to_cores`); `sub_core_grids` is None by contract |
| Per-core work | `ttnn.split_work_to_cores(grid, tensor_pages, row_wise=True)` → `core_group_1` with `pages_per_core_1`, `core_group_2` with `pages_per_core_2`; `start_page` accumulates in core order (`ttnn.corerange_to_cores(group, None, True)`) |
| Remainder | handled by the two groups; per core `num_blocks_this_core = ceil(core_pages / block_tiles)`, `last_block_tiles = core_pages − (num_blocks_this_core − 1)·block_tiles` |
| Page count | `tensor_pages = output_tensor.buffer_num_pages()` (assert == `input_tensor.buffer_num_pages()`); equals `prod(shape[:-2])·ceil(H/32)·ceil(W/32)` — per-image `ceil`, never `floor(batch·H/32)` |
| Kernel groups | one reader, one compute, one writer descriptor, each on `all_cores` (≤ 2 rectangles: full rows + one partial row; BB-17). `core_pages` / `start_page` are **runtime** args so the two work groups do not split kernel groups |
| Cache | everything shape/dtype dependent is CT or CB config; buffer addresses are RT. The descriptor is built once per (shape, src, dst, layout, memory_config, grid) key and reused (BB-07); trace replay allocates only the output (BB-16) |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_tiles` | 0 | `input_tensor.buffer_page_size()` (Phase 0: 2048) | `in_depth_blocks · block_tiles` | spans the block's page extent ×depth; streams over the core's range | input dtype (Float16_b) | reader | compute | whole kernel, streaming |
| `cb_output_tiles` | 16 | `output_tensor.buffer_page_size()` (4096 / 1088 / 4096) | `out_depth_blocks · block_tiles` | same | output dtype (Float32 / Bfp8_b / Int32) | compute | writer | whole kernel, streaming |

No intermediate CB: the conversion packs straight into the destination CB (Rule 3, pattern 1).
Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_extent` pages | dataflow raw (TensorAccessor) | input buffer | `cb_input_tiles`, `block_extent` pushed | `block_extent` tiles visible to compute |
| 2 | `convert_block` | `block_extent` tiles (whole core range in one helper call, CB-synced per `block_tiles`) | yes — `compute_kernel_lib::typecast` | `cb_input_tiles`, waits/pops `block_extent` | `cb_output_tiles`, reserves/pushes `block_extent` | input freed, output visible to writer |
| 3 | `store_block` | `block_extent` pages | dataflow raw (TensorAccessor) | `cb_output_tiles`, waits/pops `block_extent` | output buffer | freed |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| (boot) | raw_api | `compute_kernel_hw_startup(cb_input_tiles, cb_output_tiles)` | required by `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:23-37` (caller-init contract) | first statement of the compute kernel | `cb_input_tiles` | `cb_output_tiles` | — |
| `convert_block` | helper | `compute_kernel_lib::typecast<Input, Output>(IterationShape, in, out)` | decl `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:91`, impl `convenience.inl:49-57` | `Input = input(cb_input_tiles, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize)`, `Output = output(cb_output_tiles, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)` (`chain.hpp:233-242, 386-392, 442`); shape `IterationShape::tiles(core_pages).block_size(block_tiles)` (`chain.hpp:159, 168`); buffers `DataflowBuffer in(cb_input_tiles), out(cb_output_tiles)` | `cb_input_tiles` | `cb_output_tiles` | `block_tiles` (via `.block_size`), `core_pages` (RT) |
| ↳ chain inside | helper | `unary<Typecast<in_df,out_df>>` → `CopyTile` + `Typecast` + `PackTile` | `convenience.inl:41-46`; `Typecast` struct `ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.inl:87-89` → `typecast_tile_init` / `typecast_tile` `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h:463, 91` | formats derived from the CB formats (`dfb_l1_format`, `ttnn/cpp/ttnn/kernel_lib/dfb_helpers_compute.hpp:36`) | | | — |
| `load_block` | raw_api | `TensorAccessor` + `noc_async_read_page` + `noc_async_read_barrier` | `tech_reports/tensor_accessor/tensor_accessor.md`; `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547, 1921` | CT: `block_tiles`, TensorAccessorArgs(input); RT: address, `start_page`, `page_stride`(=1), `core_pages` | input buffer | `cb_input_tiles` | `block_tiles` |
| `store_block` | raw_api | `TensorAccessor` + `noc_async_write_page` + `noc_async_writes_flushed` / `noc_async_write_barrier` | `dataflow_api.h:876, 1951` | CT: `block_tiles`, TensorAccessorArgs(output); RT: address, `start_page`, `page_stride`, `core_pages` | `cb_output_tiles` | output buffer | `block_tiles` |

**Helpers considered and rejected for the dataflow operations:** `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp` and `tile_column_dataflow.hpp` — local L1↔L1 copies and column walks, not interleaved page streams by TensorAccessor; there is no kernel_lib reader/writer helper for a contiguous interleaved page range, so the reader/writer are plain TensorAccessor loops at block granularity (the standard dataflow idiom, no compute-helper equivalent).

**Refinement notes recorded for the same helper:** `typecast()` refuses UInt8 CBs (`convenience.inl:53-55`) — bfloat16→uint8 must use `unary<Typecast<Float16_b, UInt8>, …>` explicitly. If a device run shows the float→int32 SFPU path rounding instead of truncating (R1), compose `Trunc` (`ttnn/cpp/ttnn/kernel_lib/eltwise/unary/rounding.hpp:15`) before `Typecast` in the same `eltwise_chain` — no extra CB. int32→bfloat16 must round through float32 (R4): if the `Int32→Float16_b` arm rounds once, chain `Typecast<Int32,Float32>` + `Typecast<Float32,Float16_b>` in one dst window.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| R2: 32-bit DEST omitted | →float32 and →int32 need 32-bit DEST (`typecast.h:63`); the ComputeConfigDescriptor default is 16-bit | host sets `fp32_dest_acc_en` from the `(src,dst)` pair (Mechanism caps); it is part of the program cache key |
| R1: bf16→int32 must truncate | the int32 SFPU routine's rounding mode is inferred, not stated; integer-valued inputs hide it | acceptance test feeds fractions (±0.5, ±2.75, 127.5) and large magnitudes (2^30, −(2^31−2^23)); fallback: `Trunc` element in the same chain |
| bf8b rounding vs host (R5) | packer rounding differs from `ttnn.from_torch` | gate is ≤ 1 step per group; nothing to do unless it misses |
| Garbage tile padding | 29 of 40 Phase 0 cases are partial tiles; padding converts like any value | output padding is unspecified; int32 conversion of garbage saturates, no trap; bf8b padding rows/cols are their own groups for every production shape (masked regime deferred for W % 16 ≠ 0) |
| L1 output placement (R8) | L1 inputs (Q2227, P22–P50) must give L1 outputs | output allocated with `input_tensor.memory_config()` verbatim |
| Write completion (BB-03) | writer returns while writes in flight → next program hangs | one `noc_async_write_barrier` at writer end; reader ends on a barrier |
| L1 budget (BB-05) | model leaves 1 347 776 B/core; P39 tensors live in L1 banks too | op footprint ≤ 98 304 B/core Phase 0 (≤ 131 072 B at TARGET's widest pair), independent of tensor size; the tensors themselves are allocated by the model/harness, not the op |
| Refinement: float32/int32/uint32 sources | srcA is 19-bit; fp32 RNE and int32 bits need unpack-to-dest | `unpack_to_dest_mode = UnpackToDestFp32` on `cb_input_tiles` for those sources (Mechanism caps) |
| Refinement: ROW_MAJOR W = 3 (p1/p2) | 4 194 304 pages; one transfer per page is transfer-bound | ROW_MAJOR flat-stream regime with per-bank span transfers (deferred row) |
| Profiling (BB-21) | time outside zones reads as waiting | each block operation sits in its own `DeviceZoneScopedN` zone in the zone build only; the plan object reports per-core roles (all cores: reader/compute/writer) |

### Structural impossibilities

None beyond `feature_spec.py`'s INVALID.
