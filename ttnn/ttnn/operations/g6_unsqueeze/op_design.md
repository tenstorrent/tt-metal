# Operation Design: g6_unsqueeze

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (pure bit move, no compute kernel, no arithmetic) |
| Goal | Append a size-1 axis last to a one-logical-row float32 TILE tensor in DRAM: every value moves from row 0 of its input tile to column 0 of an output tile, bit for bit, in one device program. |
| Math | `out[0,…,0, w, 0] = x[0,…,0, w]` for `0 ≤ w < W`; on tiles: input tile `t = w // 32`, position (row 0, col `w % 32`) → output tile `t`, position (row `w % 32`, col 0) |
| References | `eval/golden_tests/g6_unsqueeze/{feature_spec.py,helpers.py}`; `.claude/references/{blocking-model.md,l1-footprint-discipline.md}`; `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` (`get_noc_addr(page_id, offset)`); `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:375-380` (NoC alignments); catalog `ttnn/ttnn/operations/examples/master.md` entries `width_split`, `split_reader`, `double_buffer`, `tile_reorder`, `noc_placement` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 1..3, every dim but the last == 1, volume > 0, float32, TILE, DRAM interleaved | — | buffer address RT |
| `dim` | `int` (positional or `dim=`) | yes | `r` or `-1` (any other value in `[-(r+1), r]` refused; values outside refused) | none | host only (not a kernel arg) |

`dim` canonicalization: `k = dim if dim >= 0 else dim + r + 1`; accept iff `-(r+1) ≤ dim ≤ r` and `k == r`. Done in `validate()` before anything else touches the tensor.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(1, …, 1, W)`, rank `r ∈ {1,2,3}`, `W ≥ 1`, any W (no upper limit) |
| Dtype | `ttnn.float32` |
| Layout | TILE (32×32, four 16×16 faces, face-major, row-major in a face) |
| Memory | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `(1, …, 1, W, 1)` (rank `r+1`) |
| Dtype | input dtype (`ttnn.float32`) |
| Layout | TILE |
| Memory | input memory config (DRAM interleaved), a freshly allocated tensor (never the input buffer) |

Tile geometry (alignment-aware, both tensors): `tensor_w_tiles = ceil(W / 32)`. The input has exactly one tile row (all leading dims are 1, so `batch·ceil(H/32) = 1`): input page `t` is tile column `t`. The output has one tile column and `ceil(W/32)` tile rows: output page `t` is tile row `t`. Same page count, same page index for the source and destination of every value. Byte map inside a 4096 B fp32 tile (face = 1024 B, face row = 64 B):

| Values `w % 32` | Input bytes (source) | Output bytes (destination) |
|---|---|---|
| 0..15 (tile half 0) | face 0 row 0: offset `0 .. 63` | face 0 col 0 of rows 0..15: offset `64·r`, r = w%32 |
| 16..31 (tile half 1) | face 1 row 0: offset `1024 .. 1087` | face 2 col 0 of rows 0..15: offset `2048 + 64·(r−16)` |

Faces 2 and 3 of the input and faces 1 and 3 of the output hold only tile padding: never read, never written.

## Blocking Model

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| leading unit dims `d_0..d_{r-2}` | independent — extent 1 by contract, nothing spans them | folded (extent 1) | 1 | contract (validate) | not assigned — extent 1 by contract, nothing to split | none (another movement, refused) |
| `w_tile` (tile columns of the input row = tile rows of the output) | independent — each output tile depends on exactly one input tile | `block_w_tiles` | `core_w_tiles` (the whole per-core assignment; 1 tile for p1/p2 on the full grid) | `select_plan()` in `g6_unsqueeze_program_descriptor.py` | `split_work_to_cores(grid, tensor_w_tiles, row_wise=True)` over `num_cores = min(grid_cores, MAX_CORES or ∞, tensor_w_tiles)`; contiguous tile range per core | knob-turn (core count, block size) |
| `tile_half` (value columns 0..15 vs 16..31 of a tile; face 0→face 0, face 1→face 2) | independent — the two halves read disjoint 64 B rows and write disjoint 1024 B faces | `halves_per_risc` | 1 (half 0 on the reader-config RISC, half 1 on the writer-config RISC) | host constant `HALVES_PER_RISC = 1` (2 = one RISC does both halves) | the two data-movement RISCs of every core, one half each | knob-turn |
| `in_tile_row` (rows 0..31 of an input tile) | independent — only row 0 is logical; rows 1..31 are padding | none: only row 0 is read | 1 (row 0) | contract: one logical row | not assigned — padding is never read | none |
| new axis (output last dim) | independent — extent 1 by definition | none | 1 | op definition | not assigned — extent 1 | none |

Why `block_w_tiles` is the whole per-core assignment and why that is 1 tile for production: step 1 (occupancy) spreads `tensor_w_tiles` (16 / 8) over the full grid, one tile per core; step 2 takes the coarsest block in that assignment = the whole assignment. It is not a collapsed knob: for `core_w_tiles > 1` the block grows with it up to the L1 cap below. The grid-synchronization lamp measures fewer, fuller cores.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_face_stage_half0` | `buffer_depth` (from `select_plan()`, shared by both CBs) | 1 in regime `resident` (one block per core), 2 in regime `streaming` | depth 2: reads of block b+1 land while block b is scattered and written (read/write overlap on one RISC) |
| `cb_face_stage_half1` | same `buffer_depth` | same | same |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| L1 budget left by the model (BB-05): 1347776 B per core | `block_w_tiles · buffer_depth` | `block_w_tiles ≤ floor((L1_BUDGET_BYTES/2 − 64) / (buffer_depth · face_bytes))` = 658 (depth 1) / 329 (depth 2) for fp32 | CB allocation over budget → launch failure only in the model |
| BH DRAM read alignment 64 B (`noc_parameters.h:379`) | landing address of each 64 B face-row read | source offsets are `page + 0` / `page + 1024` (≡ 0 mod 64); landing = `align_up(cb_base, 64) + j·face_bytes` → ≡ 0 mod 64; CB capacity carries +64 B slack for the align-up | misaligned DRAM read lands shifted bytes → wrong values, no error |
| BH DRAM write alignment 16 B / L1 alignment 16 B (`noc_parameters.h:375-380`) | output write granularity | face writes of 1024 B at `page + 0` / `page + 2048`, from 64 B-aligned L1 | sub-16 B output writes are illegal → a 4-byte direct gather is impossible (see regimes) |
| NoC max burst (`NOC_MAX_BURST_SIZE`) | none — 64 B and 1024 B transfers are each one packet | none needed | — |
| Ragged last tile (`W % 32 ≠ 0`) | values of the last tile | half `h` of tile `t` is processed only if `32·t + 16·h < W`; a processed half moves its whole 64 B row (padding values land in output padding rows `≥ W`, which are not logical) | none harmful; the skip only saves a 1024 B write |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `resident` — face-row move, whole assignment resident | **built** (all production shapes) | `core_w_tiles_max ≤ cap(1)` where `core_w_tiles_max = ceil(tensor_w_tiles / num_cores)`, `cap(d) = min(MAX_BLOCK_W_TILES or ∞, floor((L1_BUDGET_BYTES/2 − 64)/(d·face_bytes)))` | `block_w_tiles = core_w_tiles` × 1 half per RISC; `buffer_depth = 1` | Boundary DRAM. Read: 64 B per tile half = 128 B/tile (only the logical row; minimum at 64 B read alignment). Write: 1024 B per tile half = 2048 B/tile; the aligned minimum is 32 × 16 B = 512 B/tile — exceeded 4× to use 2 transactions per tile instead of 32 (lamp below). Each input byte read once, each output byte written once, no cross-core traffic | Per block, per RISC: one read barrier, one write barrier (at kernel end, BB-03), one CB-address lookup, one runtime-arg fetch. A bigger block (more tiles per core) amortizes the barrier round trip and the kernel launch over more tiles |
| `streaming` — face-row move, blocks of a long assignment | **built** (no production shape reaches it; any W is accepted, BB-05 forbids refusing) | `core_w_tiles_max > cap(1)` | `block_w_tiles = cap(2)`, `num_blocks_this_core = ceil(core_w_tiles / block_w_tiles)`, `last_block_extent = core_w_tiles − (num_blocks_this_core−1)·block_w_tiles`; `buffer_depth = 2` | Same bytes as `resident` (no operand is re-read across blocks); only the barrier count grows to one read barrier per block | One read barrier per block and the read/write turnaround per block; the block is at its L1 cap |
| whole-tile read + scatter (read 4096 B pages) | rejected — superseded by `resident` | — | — | Reads 4096 B/tile instead of 128 B (32×), plus 4096 B L1 per tile half-pair | — |
| single core | rejected — superseded by `resident` (same bytes, grid-wide issue/latency parallelism) | — | — | same bytes, one core | — |
| compute-engine transpose (`transpose_wh` through unpack/math/pack) | rejected — superseded by RISC scatter in `resident` | — | — | fp32 through the SrcA/SrcB operand path is 19-bit → violates bit-exactness (NaN payloads, subnormals, mantissa LSBs); also adds a compute kernel and 2 CB hand-offs | — |
| direct 4-byte DRAM→L1 gather into column 0 (no scatter) | rejected — unreachable | — | — | Source offset `4·(w%16)` and destination offset `64·r` differ mod 64 (DRAM read alignment) for 15 of 16 values; illegal | — |
| split `w_tile` halves across cores (32 cores × 1 half) instead of across the 2 RISCs of 16 cores | deferred — the 2-RISC split already gives two independent issuers per tile at half the cores and half the launch footprint; reachable as a pure knob-turn (`HALVES_PER_RISC` + core count), measured by the grid-sync lamp | — | 1 tile half per core | same bytes | — |
| dependent-axis split + combine | rejected — no dependent axis exists (pure move; no value of the output spans more than one input value) | — | — | — | — |
| L1 / sharded placement of input or output | deferred — not in TARGET (no `memory_layout` axis), production calls DRAM interleaved only, `validate()` refuses it; reachable: a CB backed on the sharded buffer replaces the 64 B NoC reads with local reads of the same face rows, the scatter is unchanged | — | whole shard | — | — |

Regime selection function (pinned, host, in `select_plan()`):
```
num_cores        = min(grid.x*grid.y, MAX_CORES or inf, tensor_w_tiles)
core_w_tiles_max = ceil(tensor_w_tiles / num_cores)            # from split_work_to_cores
if core_w_tiles_max <= cap(1): regime="resident",  buffer_depth=1, block_w_tiles=core_w_tiles (per core)
else:                          regime="streaming", buffer_depth=2, block_w_tiles=cap(2)
```
p1 (16 tiles) and p2 (8 tiles) → `resident`, 16 / 8 cores, 1 tile per core. Regime-pinned test: the acceptance test forces `streaming` on p1 by setting `MAX_CORES = 2`, `MAX_BLOCK_W_TILES = 3` (8 tiles per core, blocks 3/3/2, ragged last block).

### Traffic ranking

| Candidate split | DRAM bytes (p1) | Cross-core | Fan-out / issue | Verdict |
|---|---|---|---|---|
| `w_tile` across cores + `tile_half` across the 2 RISCs | read 2048 B, write 32768 B | none | 16 cores × 2 issuers, 1 read + 1 write per issuer | **chosen**: fewest bytes, max parallel issuers, no combine |
| `w_tile` across cores, both halves on one RISC | same | none | 16 issuers, 2+2 transactions each | same bytes, half the issue parallelism (`HALVES_PER_RISC = 2` knob) |
| `tile_half` across cores (32 cores) | same | none | 32 issuers | same bytes, double the launch footprint (deferred row) |
| single core | same | none | 1 issuer, 32+32 transactions serial | rejected |
| whole-tile pages in and out | read 65536 B, write 65536 B | none | 16+16 large transactions | rejected (read 32×) |
| dependent-axis split | n/a | n/a | n/a | no dependent axis |

Operand-reuse check: one input operand; it varies along the split axis (`w_tile`) → nothing is re-read by any core, no broadcast row. Stall-shadow check: each RISC waits on its read barrier; the only remaining work of the block (scatter, write) depends on the landed data and output padding needs no initialization, so nothing fills the stall in `resident`. In `streaming` the next block's reads are issued before the current block's scatter and writes (the reorder is legal because blocks are independent; no floating-point op exists, so no precision gate).

### Block schedule

Per data-movement RISC (`tile_half = h`), identical kernel source, two instances:

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_face_rows_block(block_idx);      // block_w_tiles × 64 B DRAM reads → row 0 of each staged face
    scatter_row_to_column_block(block_idx); // in place: face[16·r] = face[r], r = 1..15, per tile
    store_column_faces_block(block_idx);  // block_w_tiles × 1024 B DRAM writes (face 0 or face 2)
}
drain_writes();                           // one noc_async_write_barrier before return (BB-03)
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|---|---|---|---|
| `load_face_rows_block` | `block_w_tiles` tiles × 1 half (64 B each) | staging slot `block_idx % buffer_depth` | reads issued back-to-back (no per-tile barrier); ONE `noc_async_read_barrier` per block; TensorAccessor built once per kernel |
| `scatter_row_to_column_block` | `block_w_tiles` faces × 15 word moves | same slot (in-place) | no barrier, no handshake; plain RISC-V 32-bit loads/stores |
| `store_column_faces_block` | `block_w_tiles` × 1024 B | same slot until the writes are flushed | writes issued back-to-back; in `streaming`, `noc_async_writes_flushed()` before the next block's reads are issued into the other slot (that slot's previous writes must have left L1); ONE `noc_async_write_barrier` per kernel at the end |

`streaming` ordering per RISC: `issue load(0)`; for each b: `read_barrier` → (if b+1 < n) `writes_flushed` then `issue load(b+1)` into the other slot → `scatter(b)` → `issue store(b)`. End: `write_barrier`.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| grid-synchronization | 1 tile per core on 16 cores: per-core launch and the 2-range core set may cost more than the ~2 µs of work they parallelize | `MAX_CORES` ∈ {16, 8, 4, 2, 1} on p1 (2/4/8/16 tiles per core) |
| output write shape | 2 × 1024 B face writes per tile move 4× the aligned-minimum bytes | 32 × 16 B column-segment writes per tile (scatter into a packed 16 B stride staging), and 1 × 4096 B whole-tile write (`tile_reorder`: whole pages) |
| RISC split | splitting halves across both RISCs helps only if issue is the bottleneck (`split_reader`) | `HALVES_PER_RISC = 2` (one RISC per core, other idle) |
| core placement | `split_work_to_cores(row_wise=True)` on a 13/14-wide grid gives 2 ranges for 16 cores (BB-17) | an explicit 8×2 rectangle / 4×4 rectangle, and NoC selection per instance (`noc_placement`) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM input page `t` → L1 | 64 B raw fp32 bytes (face 0 row 0 or face 1 row 0) | `noc_async_read(accessor.get_noc_addr(t, 1024·h), landing + j·face_bytes, 64)` | landing 64 B-aligned (cap above); only row 0 is read — poisoned padding rows never enter L1 |
| L1 in place | 16 fp32 words → column 0 of a 16×16 face | RISC-V `uint32_t` loads/stores, `face[16·r] = face[r]`, r ascending 1..15 | bit-exact (no FPU/SFPU, no conversion); row-0 elements 1..15 left behind become output padding |
| L1 → DRAM output page `t` | 1024 B face | `noc_async_write(landing + j·face_bytes, out_accessor.get_noc_addr(t, 2048·h), 1024)` | output faces 1, 3 never written (padding) |
| Cross-core | none | — | no Tensix-to-Tensix communication in any built or deferred regime |

TARGET connection: TARGET has only `dtype = [float32]` and `layout = [TILE]`; no placement axis, so no sharded realization is a support axis. SUPPORTED = TARGET at Phase 0. Shape class (`H = 1`, output `W = 1`, not multiples of 32) is the op itself — no alignment axis, no tagger, no EXCLUSIONS. Structural impossibilities: none beyond `INVALID = []`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block of `block_w_tiles` tile columns (= output tile rows); each tile split into 2 halves, one per data-movement RISC |
| Grid | `device.compute_with_storage_grid_size()`; cores from `ttnn.split_work_to_cores(grid, tensor_w_tiles, row_wise=True)` limited to `num_cores` (p1: 16 cores, p2: 8 cores) |
| Per-core work | contiguous tiles `[start_tile, start_tile + core_w_tiles)`; group 1 / group 2 of `split_work_to_cores` differ only in RT `core_w_tiles` — one kernel descriptor per RISC role over the union of both groups |
| Remainder | `tensor_w_tiles = ceil(W/32)`; core remainder handled by `split_work_to_cores` groups; block remainder by `last_block_extent`; tile remainder by the half-skip predicate `32·t + 16·h < W` |
| Runtime args (per core, both kernels) | input buffer addr, output buffer addr, `start_tile`, `core_w_tiles` (implementer derives the exact list) |
| Compile-time | `tile_half` (0/1), `block_w_tiles`, `buffer_depth`, `W`, CB id, TensorAccessorArgs(input), TensorAccessorArgs(output) |

Plan reporting (BB-08/BB-21): `g6_unsqueeze_program_descriptor.select_plan(input_tensor) -> G6UnsqueezePlan` (namedtuple: `regime`, `num_cores`, `core_w_tiles_max`, `block_w_tiles`, `buffer_depth`, `core_ranges`) is the single source of every knob; the program descriptor consumes it, never recomputes. Module-level knobs in the same file: `L1_BUDGET_BYTES = 1347776`, `MAX_CORES = None`, `MAX_BLOCK_W_TILES = None`, `HALVES_PER_RISC = 1`. Every knob value is part of the program-cache key (BB-07) together with input shape, dtype, layout, memory config.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_face_stage_half0` | 0 | `face_bytes` (1024) | `buffer_depth · block_w_tiles` (+64 B slack in `total_size` for 64 B alignment) | spans `w_tile` (block), streams over blocks; 1 face per tile (half 0) | Float32 | reader-config kernel (`tile_half=0`) — private scratch, no cross-kernel handshake | same kernel (private scratch) | whole kernel |
| `cb_face_stage_half1` | 1 | `face_bytes` (1024) | `buffer_depth · block_w_tiles` (+64 B slack) | same, half 1 | Float32 | writer-config kernel (`tile_half=1`) — private scratch | same kernel | whole kernel |

These are L1 allocations owned by one kernel each; no `cb_push_back`/`cb_wait_front` is used (no hand-off exists), so push count = wait count = 0 trivially. No compute kernel, no other CB.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_face_rows_block` | `block_w_tiles` × 64 B | raw NoC | DRAM input pages | `cb_face_stage_half{h}` slot `block_idx % buffer_depth`, `block_w_tiles` pages, row 0 filled | row 0 valid |
| 2 | `scatter_row_to_column_block` | `block_w_tiles` × 15 words | raw RISC stores | same slot | same slot (in place) | column 0 valid |
| 3 | `store_column_faces_block` | `block_w_tiles` × 1024 B | raw NoC | same slot | DRAM output pages | slot reusable after flush |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| address of a face row / face | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | `page_id = start_tile + block_idx·block_w_tiles + j`, `offset = 1024·h` (read) / `2048·h` (write) | — | — | — |
| `load_face_rows_block` | raw_api | `noc_async_read` + `noc_async_read_barrier` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547`, `:1921` | size 64 | DRAM | `cb_face_stage_half{h}` | trip count `block_w_tiles` |
| `scatter_row_to_column_block` | raw_api | RISC-V loads/stores via `addr_to_l1_ptr` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:30` | — | `cb_face_stage_half{h}` | same | trip count `block_w_tiles` |
| `store_column_faces_block` | raw_api | `noc_async_write` (+ `noc_async_writes_flushed` in streaming) + `noc_async_write_barrier` at end | `dataflow_api.h:876`, `:1951` | size 1024 | `cb_face_stage_half{h}` | DRAM | trip count `block_w_tiles` |
| slot address | raw_api | `get_write_ptr(cb)` | `dataflow_api.h:325` | — | — | — | — |

Helpers considered and rejected:

| Raw entry | Candidate helper | Mismatch (file:line) | Reason |
|---|---|---|---|
| scatter | `dataflow_kernel_lib::copy_tile_column` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:91-101` | reads `src[column_base(src_col) + row_offset(...)]` — a column of the source; this op reads a **row** (row 0, 16 consecutive words) and writes a column. A `copy_tile_row_to_column` helper would close the gap. |
| scatter | `fill_tile_column` | `tile_column_dataflow.hpp:68-77` | writes one constant value, not 16 distinct values |
| reads | `dataflow_kernel_lib::set_read_state` / `read_with_state` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:74-95` | self-aimed (local L1 → local L1) reads only; our source is a DRAM bank address that changes per page |
| all compute phases | eltwise / transpose / tilize helpers | — | no compute kernel: any FPU path narrows fp32 operands (bit-exactness rule) |

`compute_kernel_hw_startup()`: not applicable — no compute kernel.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Conversion through the FPU | fp32 through SrcA/B is 19-bit; randn passes a tolerance, special values fail | no compute kernel; values moved only by NoC and 32-bit RISC stores |
| Reading tile padding | the model's producer leaves arbitrary (NaN) padding in rows 1..31 | only row 0 of faces 0/1 is read; padding columns of a ragged last tile land only in output padding rows `≥ W` |
| 64 B DRAM read alignment on BH | a misaligned landing silently shifts bytes | landing = `align_up(cb_base, 64)`, +64 B slack in each CB, face pitch 1024 |
| In-place scatter order | overlapping read/write would corrupt values | reads indices 1..15 of row 0, writes indices 16·r (r ≥ 1) — disjoint; element 0 already in place |
| Writes in flight at return (BB-03) | next program starts immediately | one `noc_async_write_barrier` (and read barrier) before each kernel returns |
| Staging slot reuse in `streaming` | reading block b+1 into a slot whose writes are still in flight overwrites them | `noc_async_writes_flushed()` before issuing reads into a slot previously written from |
| Program cache key (BB-07) | p2 then p1 differ only in W; forced-streaming test differs only in knobs | key includes shape, and `MAX_CORES`, `MAX_BLOCK_W_TILES`, `HALVES_PER_RISC`, `L1_BUDGET_BYTES`; RT args (buffer addresses) are updated per call |
| Rank-1 input | `(W,)` TILE is still one tile row | same page map as `(1, W)`; output `(W, 1)` |
| Profiling (BB-21) | zones must cover read issue, read wait, scatter, write issue, write wait | five `DeviceZoneScopedN` zones per kernel, compiled only in the profiler build; `select_plan()` dumps core → (`start_tile`, `core_w_tiles`) and RISC → `tile_half` |
| Kernel image size (BB-04/18) | many builds and preceding programs share the config region | one small kernel source, two instances, no compute binary |

### Test scope note

Per BB-22 and the spec ("the golden suite runs the two production cases only"), the acceptance test is limited to p1/p2, `dim=-1`, special bit patterns, poisoned padding, one regime-pinned `streaming` case (forced through module knobs on the p1 shape), the plan report for production shapes, and the validation refusals — no shape sweep.
