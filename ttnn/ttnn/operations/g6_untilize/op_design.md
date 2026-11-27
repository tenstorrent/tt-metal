# Operation Design: g6_untilize

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no arithmetic, no compute kernel) |
| Goal | Copy the 32-bit patterns of one uint32 TILE, DRAM-interleaved tensor into a newly allocated uint32 ROW_MAJOR, DRAM-interleaved tensor of the same shape. One device program, byte moves only. |
| Math | `out[b, r, w] = x[b, r, w]` bit for bit. `x[b, r, w]` is element `(r mod 16, w mod 16)` of face `2·((r mod 32) div 16) + (w mod 32) div 16` of tile `(b, r div 32, w div 32)`. Output page `g = b·H + r` holds row `(b, r)`, W values in order |
| References | spec `docs/explorations/g6_untilize/spec/g6_untilize.txt`; `docs/explorations/g6_untilize/{DOSSIER,math,tensix,risks,lessons}.md`; golden suite `eval/golden_tests/g6_untilize/{feature_spec,helpers,test_regression}.py`; `.claude/references/{blocking-model,l1-footprint-discipline,op-design-template}.md`; catalog `ttnn/ttnn/operations/examples/master.md` |

The production call is `(1, 1, 32, 128)`: 4 tiles in, 32 rows out, 16384 B each way, 124 calls per traced
model pass. At this size the cost is fixed per-call overhead plus one DRAM read round trip and one DRAM write
round trip (risks.md R4). The design keeps both round trips on every core's critical path exactly once,
issues few transactions per core, and adds nothing else.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | device tensor, rank 4, uint32, TILE, DRAM interleaved | — | buffer address → common RT arg of the reader |
| `use_multicore` | `bool`, keyword-only | no | `True` only; any other value → `ValueError` | `True` | host only; decides nothing (core count belongs to the Blocking Model) |
| `sub_core_grids` | `ttnn.CoreRangeSet \| None`, keyword-only | no | `None` only; any other value → `ValueError` | `None` | host only; decides nothing |

There is no dtype, memory_config, output-tensor or compute-config keyword.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 4, `(d0, d1, H, W)`; production `(1, 1, 32, 128)` |
| Dtype | `ttnn.uint32` |
| Layout | TILE (32×32 tiles, four 16×16 faces TL, TR, BL, BR, row-major inside a face) |
| Memory | DRAM interleaved; one 4096-B page per tile, page `t` in bank `t mod n_banks` |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` |
| Dtype | `ttnn.uint32` (the input's) |
| Layout | ROW_MAJOR; one page per row `(b, r)`, `4·W` bytes (512 B in production) |
| Memory | DRAM interleaved (the input's placement), newly allocated with `ttnn.allocate_tensor_on_device` in every call. It shares no storage with the input. |

## Blocking Model

The work index space is the output: `tensor_rows = B·H` rows × `tensor_tile_cols = ceil(W/32)` tile columns.
No result depends on another element (math.md, Axes and dependence), so every axis is independent. There is
no dependent axis and so no combine. The only coupling is storage: one output row gathers two 64-B face-row
pieces from each of the `tensor_tile_cols` tiles of its tile row (math.md F1).

**Committed Phase 0 scheme:** `row_gather`. The rows are spread over one 8×4 core rectangle, so each of
32 cores owns 1 row. Each core's block is its whole assignment: 1 row × 4 tile columns. The reader gathers the
row's 8 face-row pieces (64 B each) from DRAM straight into their row-major positions in L1, with one read
barrier. The writer stores the row as one whole 512-B page with the untilize stick-writer helper. Every
knob below is a parameter, set once on the host and derived everywhere else. None is an inlined
constant.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` (B = d0·d1) | independent: `out[b,…]` reads `x[b,…]` only | folded into `rows` (`g = b·H + r`). Output pages of consecutive images are contiguous and every image is handled identically. The image is recovered per row (`b = g div H`) for the per-image tile-row index | B = 1 | `tensor_rows = B·H`, host-derived from `input_tensor.shape` → CT arg | with `rows` | knob-turn |
| `rows` (global output row g; `tensor_rows = B·H`) | independent: a row reads only its own `2·tensor_tile_cols` face-row pieces | `block_rows = min(core_rows − 32·k, TILE_HEIGHT)` for the core's k-th row block | 1 = the core's whole assignment (32 rows / 32 cores) | rows per core ← `num_cores` (host `choose_core_rectangle`, CT arg). Row-block cap `TILE_HEIGHT` = the CB's tile height, which the writer helper reads (`tilize_helpers_dataflow.inl:193,210`) | contiguous balanced ranges over the rectangle, derived in-kernel from the core's logical coordinate | knob-turn (`CORE_LIMIT`) |
| `tile_cols` (`tensor_tile_cols = ceil(W/32)`) | independent: each column range of a row is copied on its own; coupled only through the output page | `block_tile_cols` | 4 = whole row page | host `derive_block_tile_cols`: the largest divisor of `tensor_tile_cols` whose CB fits `L1_OP_BUDGET_BYTES`. Feeds a CT arg and the CB page count | not spread: `num_col_blocks = 1`. A core walks its col blocks in order | knob-turn (helper's `byte_offset_within_page`; assign col blocks to cores) |
| in-tile (face column `f ∈ {0,1}`, face row `i`) | independent | none: this is the transfer quantum, not a block axis. One piece is 16 elements = 64 B (Mechanism caps) | 2 pieces per (row, tile column) | TILE format (`tile_constants.h:18-22`) | with the row | — |

### Knob decisions per axis (Step 1.4), intermediate data included

| Axis | Character | Block factor (Phase 0) | Core assignment | Buffer depth | Why |
|------|-----------|------------------------|-----------------|--------------|-----|
| `batch` | independent | folded into `rows` | folded into `rows` | `cb_row_block` depth | B = 1 in production; folding keeps one row index and one page index |
| `rows` | independent | 1 row (whole per-core assignment) | 32 cores, 1 row each, one 8×4 rectangle | 1 (one block per core) | occupancy first (32 row units → 32 cores), then the coarsest block = the whole assignment. Splitting finer than a row means splitting `tile_cols` (see that row) |
| `tile_cols` | independent | 4 = whole width | not split across cores | same CB | splitting turns each 512-B whole-page write into four 128-B partial writes from 4 cores: same bytes, 4× write transactions (Traffic ranking). The coarse whole-page default wins on the traffic tiebreak |
| in-tile `f`, `i` | independent | fixed 64-B piece | with the row | — | the TILE format caps the contiguous run at 64 B |
| intermediate: the row block in `cb_row_block` (axes `rows` × `tile_cols`) | independent | = the core's block | produced and consumed on the same core | 1 | no stage collects or redistributes data. No core is designated to collect or coordinate, and no stage computes on one core while others wait |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_row_block` | `buffer_depth = min(ROW_BLOCK_DEPTH, max_blocks_per_core)`, with `ROW_BLOCK_DEPTH = 2` (host constant) and `max_blocks_per_core = ceil(max_core_rows / TILE_HEIGHT) · num_col_blocks` | 1: every core owns exactly one block | With ≥ 2 blocks per core the reader gathers block k+1 while the writer stores block k. With one block there is nothing to overlap and a second slot would be dead L1, so the clamp makes the knob honest at Phase 0 |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| TILE face geometry: the longest run that is contiguous in both x and out is one face row, 16 elements = 64 B (math.md F1, tensix.md Mechanism caps) | bytes per NoC read | piece bytes = `4·min(16, W − 32c − 16f)`; pieces with ≤ 0 bytes are skipped (pure tile padding) | a 128-B read treats a tile row as contiguous: columns 16–31 then come from the next face row of the left face (wrong values in a fixed pattern, risks.md R2) |
| Blackhole DRAM read alignment: `(l1_addr & 63) == (dram_addr & 63)` (`noc_parameters.h:379` `NOC_DRAM_READ_ALIGNMENT_BYTES 64`, checked at `sanitize.h:746`) | L1 destination offset of every piece, and therefore the row stride | row stride = `padded_chunk_row_bytes = block_tile_cols·128` (a multiple of 64). The CB base is DRAM-aligned by the allocator (`program.cpp:2505-2509`). Every source offset `(2h+f)·1024 + 64·i` is a multiple of 64 | with a 16-B-aligned stride every piece mis-lands silently. Under `--dev` the NoC sanitizer trips instead |
| writer helper quantum (`tilize_helpers_dataflow.inl:208-239`): rows per wait ≤ TILE_HEIGHT; wait/pop count = `width_in_tiles = ceil(chunk_row_bytes / 128)` tile pages; row stride `padded_row_bytes` | `block_rows`, pages pushed per block, CB page size | `block_rows = min(core_rows − 32k, 32)`. The reader reserves/pushes exactly `block_tile_cols` pages per block. CB page = one uint32 tile (4096 B). Capacity = `block_tile_cols · buffer_depth` pages | quanta disagree: either `cb_wait_front` never returns (hang) or the ring wraps mid-block and rows land on other rows' slots |
| writer helper contiguity: one call writes pages `start_page … start_page + total_num_rows − 1` (`tilize_helpers_dataflow.inl:233`) | the per-core row assignment | contiguous balanced ranges, never strided | a strided assignment cannot be expressed through the helper |
| CB ring closure (blocking-model.md §2): the push quantum must be the same for every block | `block_tile_cols` | `block_tile_cols` divides `tensor_tile_cols`. Every col chunk is then `block_tile_cols` tile pages wide, including the one holding a partial last tile | a ragged last chunk pushes fewer pages, so the ring wraps at a different offset and later blocks overwrite slots still being read |
| `NOC_MAX_BURST_SIZE = 16384 B` (`noc_parameters.h:289-290`) | size of a one-packet read | pieces are ≤ 64 B, always one packet. The writer helper uses `noc_async_write` (multi-packet safe, `dataflow_api.h:876`) | a one-packet API above 16 KB transfers a truncated packet |
| model L1 bound BB-05: 1347776 B per core | `buffer_depth · block_tile_cols · tile_bytes` | `derive_block_tile_cols` keeps the CB ≤ `L1_OP_BUDGET_BYTES`. Even `block_tile_cols = 1` (8 KB at depth 2) fits, so no shape needs a host refusal | the CB fails to allocate at launch, and only inside the model |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `row_gather` | **built** | always: every rank-4 uint32 TILE DRAM-interleaved input (all of TARGET and INPUTS). It is the only built regime, so `select_regime()` returns it unconditionally | `block_rows × block_tile_cols` (Phase 0: 1 × 4) | **Minimum.** Boundary: DRAM. x crosses once as a read (256 reads × 64 B, each piece read once by the core owning its row). out crosses once as a write (32 writes × 512 B, each page written whole, once, by one core). No cross-core traffic. No extra L1 pass: each byte lands in L1 once and leaves once | Per block, paid once each: one read barrier (one exposed DRAM read round trip), one CB reserve/push + wait/pop pair, one write barrier (the helper's per-block barrier), and the NoC read state set once per source tile touched. Phase 0 pays each once per core. A larger block (more rows per core) pays them fewer times per row but lowers occupancy, which is the grid-synchronization lamp |
| `tile_scatter` | **deferred.** It is the mirror of `row_gather`, with identical DRAM crossings and transaction count (32 reads × 512 B of contiguous face-row runs, 256 writes × 64 B). It only moves the 64-B transfers from the read side to the write side, and it makes 8 cores write parts of every output page. No INPUTS shape exists on which it moves fewer bytes or fewer transactions, and its store phase has no helper (`write_sticks_after_untilize` writes whole sticks). It stays reachable: same rectangle, same CB, same per-core row ranges, with the 64-B loop moved from reader to writer | always | 8 face rows of one face per core (512 B contiguous) | minimum bytes; output pages written in 64-B pieces by 8 cores | same terms as `row_gather` |
| `tile_stage_local_reorder` | **rejected**, superseded by `row_gather`. It reads whole tiles (4 × 4096 B), then reorders the same 256 64-B pieces in L1 (self-aimed NoC reads, `local_copy_helpers_dataflow.hpp:57-94`, or RISC-V stores) before writing whole pages. The local pass is exactly the 256 moves `row_gather` already makes as its DRAM reads, added on top. A tile spans all 32 rows, so the reorder concentrates on ≤ 4 cores | always | `tensor_tile_cols` tiles per core | DRAM minimum (4 + 32 transactions) plus one extra core-local pass of 16384 B in 256 transfers | — |
| `engine_untilize` | **rejected**, superseded by `row_gather`. Compute untilize (`untilize_helpers.hpp:152-163`) unpacks through `llk_unpack_A` (`pack_untilize.h:263`). UInt32 is legal only unpacked straight to DEST (`cunpack_common.h:467-473`), and that the packer keeps all 32 bits is unproven (risks.md R1). The spec forbids any matrix-engine operand or narrower path. It also adds a compute kernel (three more binaries, BB-18) | always | 1 × 4 tiles | DRAM minimum plus an unpack/pack pass | — |
| `replicated_tile_read` | **rejected, dead end**: every core reads the whole tile row (4 × 4096 B) and keeps its rows, so x crosses DRAM `num_cores` times (32×). Superseded by `row_gather`, which reads each piece once | always | — | x: 32 DRAM crossings | — |

Regime-selection function (host, `g6_untilize_program_descriptor.py`): `select_regime(input_tensor) -> "row_gather"`.
It is reported by `describe_plan()` (Host contract) and pinned by the acceptance test.

### Traffic ranking

Candidate splits, counted per memory tier. Bytes are identical for every split that reads each piece once,
so transaction count and transfer shape decide.

| Split | x DRAM crossings | out DRAM crossings | DRAM transactions | Cross-core | Cores (Phase 0) | Rank |
|-------|------------------|--------------------|-------------------|------------|-----------------|------|
| `rows` only (**built**) | 1 | 1 | 256 R × 64 B + 32 W × 512 B = 288 | 0 | 32 | **1** |
| `rows × tile_cols` | 1 | 1 | 256 R × 64 B + 128 W × 128 B = 384 | 0 | 128 | 2 (a knob-turn of `row_gather`; grid-sync lamp) |
| `tile_cols` only | 1 | 1 | 256 R + 128 W × 128 B = 384 | 0 | 4 | 3 (64 reads issued per core) |
| no split (one core) | 1 | 1 | 288 on one core | 0 | 1 | rejected: issue-bound, and Phase 0 must be multi-core |
| dependent axis + combine | — | — | — | — | — | n/a: no dependent axis exists |

Chosen: `rows` only. It ties the minimum bytes, needs the fewest DRAM transactions of any minimum-byte split,
writes every output page whole, and fills 32 cores.

**Operand-reuse check.** x varies along `rows` and along `tile_cols`. Each core reads only the pieces of its
own rows, so x is not reuse-shared under this split and there is no broadcast row. out is written once per
page.

**Stall-shadow check.** The writer's only wait is for the reader's block (one DRAM read round trip). The
reader's only wait is its read barrier. Every byte the writer moves depends on that barrier, so in Phase 0
(one block per core) no work independent of the wait exists. With ≥ 2 blocks per core (shapes outside
INPUTS), `buffer_depth = 2` lets the reader gather block k+1 while the writer stores block k. That reorder
needs no algebraic permission (no arithmetic) and changes no floating-point order.

### Block schedule

```cpp
// reader (NCRISC, NoC0) and writer (BRISC, NoC1) each realize their half of this schedule
for (uint32_t col_block_idx = 0; col_block_idx < num_col_blocks; ++col_block_idx) {
    for (uint32_t block_idx = 0; block_idx < num_row_blocks_this_core; ++block_idx) {
        gather_row_block(col_block_idx, block_idx);   // reader: DRAM face-row pieces -> row-major rows in cb_row_block
        store_row_block(col_block_idx, block_idx);    // writer: rows of cb_row_block -> ROW_MAJOR pages in DRAM
    }
}
```

`num_col_blocks = tensor_tile_cols / block_tile_cols`; `num_row_blocks_this_core = ceil(core_rows / TILE_HEIGHT)`.
Phase 0: 1 × 1.

| Block operation | Kernel | Block shape | Resident across it | Intended fixed-cost frequency |
|-----------------|--------|-------------|--------------------|-------------------------------|
| `gather_row_block` | reader | `block_rows` rows × `block_tile_cols` tile columns; per row `2·block_tile_cols` pieces of 64 B | the reserved `cb_row_block` slot (`block_tile_cols` pages) until push | `TensorAccessor` built once per kernel. CB reserve + push once per block. NoC read state set once per source tile touched by the block (tile-column-outer, rows-inner walk). **One** `noc_async_read_barrier` per block, after every piece of the block is issued. Never one barrier per piece or per row |
| `store_row_block` | writer | the same block; one write per row of `chunk_row_bytes` at `byte_offset_within_page = col_block_idx · block_tile_cols · 128` | the CB slot until pop | `TensorAccessor` built once per kernel. CB wait + pop once per block. One write per row. One `noc_async_write_barrier` per block (inside the helper, `tilize_helpers_dataflow.inl:238`), so every write has completed when the kernel returns (BB-03) |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Grid synchronization**: 32 cores × (8 reads + 1 write) | ~9 transactions per core; launch, dispatch and the two DRAM round trips dominate, so fewer cores with more rows may tie, or 4× more cores with fewer pieces each may cut issue time | `CORE_LIMIT ∈ {4, 8, 16}` (8/4/2 rows per core); `block_tile_cols = 1` with col blocks assigned across cores (128 cores, 2 reads + 1 128-B write each). Judge on unprofiled blocking trace-replay host time and the model step, not on profiled kernel duration alone (lessons.md RB-053) |
| reader/writer pair vs one fused kernel | one block per core still loads two binaries and pays a CB handoff; kernel-config bytes (BB-18) | one kernel on one RISC: gather → read barrier → stores → write barrier, using the same CB as scratch |
| issue split across both data-movement RISCs (catalog `split_reader`) | at lower core counts one RISC may be issue-bound on 64-B reads | each RISC gathers and stores half of the core's rows |
| bank stagger (catalog `bank_stagger`) | every core starts its gather on tile column 0, so the first request of all 32 cores hits one bank | start tile column = `core_idx mod block_tile_cols`, as a CT switch |
| read issue form | a plain one-packet read reprograms the full NoC command per 64-B piece | `noc_async_read_one_packet_set_state` once per source tile + `noc_async_read_one_packet_with_state` per piece (`dataflow_api.h:575,599`) |
| rectangle aspect / origin (catalog `noc_placement`) | an 8×4 rectangle at logical (0,0) is one choice of path to the DRAM endpoints | 4×8; 16×2 where the grid is ≥ 16 wide; other origins |

Overlap lamp: not attached. Every INPUTS shape has exactly one block per core, so there is nothing to overlap.

### Catalog entries consulted

| Entry | Knob it informed | How |
|-------|------------------|-----|
| `noc_placement` | core-assignment, kernel configs | reads on NoC0 (reader default), writes on NoC1 (writer default); the rectangle is wider than tall. Aspect is a lamp |
| `double_buffer` | block = the barrier unit | issue every read of a block, then **one** barrier; never read-one/barrier. Depth 2 only when a core owns ≥ 2 blocks |
| `tile_reorder` | `tile_cols` not split across cores | keep transfers coalesced where the format allows: whole 512-B page writes, not 128-B partial-page writes. The 64-B reads are forced by the face geometry |
| `width_split` | `rows` split finer than tile rows | a one-tile-row tensor split by tile rows strands on one core. Splitting the 32 rows inside the tile row fills 32 cores |
| `bank_stagger`, `split_reader` | perf lamps | gains measured as small at few blocks per core; recorded as lamps, not defaults |
| `read_coalescing`, `trid_double_issue` | not applied | interleaved pages and the 64-B face cap rule out wide coalesced reads; one block per core leaves no in-flight blocks for trids |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x in DRAM | TILE pages of 4096 B (4 faces × 16 rows × 64 B) | — | page `t = (b·tile_rows_per_image + r div 32)·tensor_tile_cols + c` |
| gather (reader, NCRISC, NoC0) | 64-B face-row pieces | one-packet NoC reads from `TensorAccessor::get_noc_addr(tile_id, piece_offset)` (`tensor_accessor.h:165`) | each piece lands at its row-major L1 position. No RISC-V load/store touches the values; no engine path (R1 holds by construction) |
| `cb_row_block` in L1 | row-major rows, stride `padded_chunk_row_bytes` | CB push (reader) / wait (writer) | the same layout `compute_kernel_lib::untilize` would produce for the writer helper |
| store (writer, BRISC, NoC1) | ROW_MAJOR rows of `chunk_row_bytes` | `dataflow_kernel_lib::write_sticks_after_untilize` | one write per row, barrier per block |
| out in DRAM | ROW_MAJOR pages of `4·W` bytes | — | page `g` in bank `g mod n_banks` |

No regime, built or deferred, uses Tensix-to-Tensix communication: there are no semaphores, multicast or combine.

**TARGET mapping.** TARGET has only `dtype: [uint32]` and `layout: [TILE]`. It has no `memory_layout`
axis, so no physical-shard placement exists to classify. The logical shard of the work split is a height
shard of `rows` over the rectangle. A future HEIGHT_SHARDED placement would cut an independent axis, which
makes it a knob-turn, but it is outside TARGET.

**Structural impossibilities:** none to add to `feature_spec.py` (INVALID stays empty).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one row block: `block_rows` (≤ TILE_HEIGHT) consecutive output rows × `block_tile_cols` tile columns |
| Grid | one rectangle `CoreRange((0,0), (rect_w−1, rect_h−1))` from `choose_core_rectangle(tensor_rows, device.compute_with_storage_grid_size(), CORE_LIMIT)`. Phase 0: 8 × 4 = 32 cores on any grid ≥ 8 wide and ≥ 4 tall. Both kernels use exactly this one range (BB-17) |
| Per-core work | contiguous balanced row range. `core_idx = get_relative_logical_y()·rect_w + get_relative_logical_x()`; `base = tensor_rows div num_cores`; `extra = tensor_rows mod num_cores`; `core_rows = base + (core_idx < extra)`; `core_row_start = core_idx·base + min(core_idx, extra)`. Phase 0: 1 row per core (same groups as `ttnn.split_work_to_cores`, row-major order) |
| Remainder | at most a 1-row imbalance (`extra`). `num_cores ≤ tensor_rows` by construction, so every core owns ≥ 1 row |
| Tile geometry (alignment-aware) | `tensor_rows = B·H` (ROW_MAJOR has no padding rows); `tile_rows_per_image = ceil(H/32)`; `tensor_tile_cols = ceil(W/32)`. For row `g`: `b = g div H`, `r = g mod H`, `tile_row = b·tile_rows_per_image + r div 32`, `h = (r mod 32) div 16`, `i = r mod 16`; piece `(c, f)` source offset `(2h + f)·face_bytes + i·face_row_bytes` in page `tile_row·tensor_tile_cols + c`; L1 offset `j·padded_chunk_row_bytes + ((c − c0)·32 + 16f)·4` for row slot `j` of the block; size per Mechanism caps. `face_bytes = tile_bytes / 4 = 1024`, `face_row_bytes = face_bytes / 16 = 64` |
| Selection | single regime (`row_gather`). The rectangle follows from `tensor_rows`, the grid and `CORE_LIMIT`. `describe_plan()` reports both |

`choose_core_rectangle(num_units, grid, core_limit)`: `cap = min(num_units, grid.x·grid.y, core_limit or ∞)`.
Pick `(w, h)` with `w ≤ grid.x`, `h ≤ grid.y`, `w·h ≤ cap` and maximal `w·h`; break ties by `w ≥ h`, then by
the smallest `w − h`. For 32 units this gives (8, 4).

Per-core work is derived in-kernel from the core's logical coordinate plus CT args (`rect_w`, `num_cores`,
`tensor_rows`). The only runtime args are **common** runtime args (the reader's input address, the writer's
output address). Dispatch therefore writes one runtime-arg set per kernel to the one rectangle, and no
per-core unicast is needed (lessons.md RB-036). Device-side grid origin: `get_relative_logical_x/y()`
(`dataflow_api.h:70-86`), and the rectangle starts at logical (0,0).

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_row_block` | 0 | `tile_bytes` = 4096 (one 32×32 uint32 tile, as the writer helper requires, `tilize_helpers_dataflow.inl:193-198`) | `block_tile_cols · buffer_depth` (Phase 0: 4 · 1 = 4 → 16384 B) | spans `tile_cols` (`block_tile_cols` pages) and `rows` up to TILE_HEIGHT (a tile page column holds 32 rows × 32 columns of a block); streams over blocks through `buffer_depth`. Row `j` of a block sits at `j · padded_chunk_row_bytes` | `ttnn.uint32` (UInt32). Dataflow-only payload, never unpacked | reader | writer | whole kernel; one block in flight per slot |

Full ledger, data-movement budget and symbol bounds: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `gather_row_block` | `block_rows × block_tile_cols` | no helper covers it (raw NoC reads; see API Mapping) | — (x in DRAM) | `cb_row_block`, `block_tile_cols` pages reserved then pushed | pushed. Rows `0 … block_rows−1` valid; the rest of the slot is untouched and never written out |
| 2 | `store_row_block` | `block_rows × block_tile_cols` | `dataflow_kernel_lib::write_sticks_after_untilize` | `cb_row_block`, `block_tile_cols` pages, waited | — (out in DRAM) | popped after the write barrier |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `store_row_block` | helper | `dataflow_kernel_lib::write_sticks_after_untilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:130`, impl `tilize_helpers_dataflow.inl:186-241` | `<cb_row_block>(output_accessor, core_rows, chunk_row_bytes, core_row_start, col_block_idx·block_tile_cols·128)`. One call per col block, covering all of the core's rows in ≤ 32-row blocks | `cb_row_block` | DRAM out | `chunk_row_bytes` and `byte_offset_within_page` carry `block_tile_cols`; the rows-per-wait cap (TILE_HEIGHT) is the helper's |
| `gather_row_block` | raw_api | `noc_async_read_one_packet` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:515` | `(input_accessor.get_noc_addr(tile_id, piece_offset), l1_dst, piece_bytes)` | DRAM x | `cb_row_block` | `block_rows`, `block_tile_cols` set the piece walk |
| `gather_row_block` (lamp form) | raw_api | `noc_async_read_one_packet_set_state` / `noc_async_read_one_packet_with_state` | `dataflow_api.h:575`, `dataflow_api.h:599` | state once per source tile, then one call per piece | DRAM x | `cb_row_block` | — |
| `gather_row_block` | raw_api | `noc_async_read_barrier` | `dataflow_api.h:1921` | once per block | — | — | — |
| `gather_row_block` | raw_api | `cb_reserve_back` / `get_write_ptr` / `cb_push_back` | `dataflow_api.h:406`, `:325`, `:208` | `block_tile_cols` pages per block (nominal, matching the helper's `width_in_tiles`, `tilize_helpers_dataflow.inl:209`) | — | `cb_row_block` | `block_tile_cols` |
| both | raw_api | `TensorAccessor` + `TensorAccessorArgs` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` (`get_noc_addr(page_id, offset)`) | CT args from `ttnn.TensorAccessorArgs(tensor).get_compile_time_args()` appended last; address from `get_common_arg_val<uint32_t>(0)` (`dataflow_api.h:171`); page size = the tensor's page size | — | — | — |
| both | raw_api | `get_relative_logical_x/y` | `dataflow_api.h:70-86` | core index in the rectangle | — | — | `num_cores`, `rect_w` |
| both (zone build) | raw_api | `DeviceZoneScopedN` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012` | zones named below; compiled out of the plain build | — | — | — |

**Helpers considered and rejected for `gather_row_block`:**

| Candidate | File:Line | Concrete mismatch |
|-----------|-----------|-------------------|
| `dataflow_kernel_lib::read_sticks_for_tilize` | `tilize_helpers_dataflow.hpp:88` | reads whole row-major **sticks**, one page per row via `get_noc_addr(page, byte_offset)`. Our source is TILE pages, where one output row is `2·tensor_tile_cols` non-contiguous 64-B pieces in `tensor_tile_cols` different pages |
| `dataflow_kernel_lib::set_read_state` / `read_with_state` / `local_addr` | `local_copy_helpers_dataflow.hpp:57,74-75,94` | self-aimed L1→L1 copies: the source is this core's own L1 (`local_addr()` builds `my_x/my_y`), never a DRAM bank. It fits only the rejected `tile_stage_local_reorder` |
| `compute_kernel_lib::untilize` | `untilize_helpers.hpp:152-163` | compute-engine path through `llk_unpack_A` (`pack_untilize.h:263`), illegal for UInt32 except unpack-to-DEST (`cunpack_common.h:467-473`) and forbidden by the spec's rule against an engine path (R1) |

Gap record: no kernel-lib helper gathers face rows of TILE pages into row-major sticks. A
`gather_sticks_from_tiles<cb>(accessor, num_rows, start_row, …)` dataflow helper, the producer counterpart
of `write_sticks_after_untilize`, would close it.

**Zones (BB-21, zone build only).** Reader: `G6U-gather-issue` (reserve + every piece issued),
`G6U-gather-wait` (read barrier), `G6U-gather-push`. Writer: `G6U-store-wait` (an explicit
`cb_wait_front(cb_row_block, block_tile_cols)` before the helper call; the helper's own `cb_wait_front`
then returns at once) and `G6U-store` (the helper call: writes plus its per-block write barrier). The plain
build is unchanged: `DeviceZoneScopedN` expands to nothing without the profiler.

## Host Contract

| Item | Decision |
|------|----------|
| Files | `ttnn/ttnn/operations/g6_untilize/{__init__.py, g6_untilize.py, g6_untilize_program_descriptor.py, kernels/g6_untilize_gather_reader.cpp, kernels/g6_untilize_row_writer.cpp}` |
| Exports (`__init__.py`) | `g6_untilize`, `describe_plan`, `validate`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS` (`eval/golden_tests/g6_untilize/axes.py` imports `INPUT_TAGGERS`) |
| Registry | `INPUT_TAGGERS = {"dtype": tag_dtype, "layout": tag_layout}`, which read `inputs[0]["dtype"]` and `inputs[0]["layout"]` of the scenario dict `{"input_shape", "dtype", "layout"}`. `SUPPORTED = {"dtype": [ttnn.uint32], "layout": [ttnn.TILE_LAYOUT]}`. `EXCLUSIONS = []`. No alignment, rank or size axis, no other tagger, no INVALID |
| `validate(input_tensor, *, use_multicore, sub_core_grids)`, first line of the entry point | in order: `use_multicore is not True` → `ValueError`; `sub_core_grids is not None` → `ValueError`; not a device tensor (`ttnn.is_tensor_storage_on_device`) → `ValueError`; rank ≠ 4 → `ValueError`; `memory_config().memory_layout != INTERLEAVED` or `buffer_type != DRAM` → `ValueError`. Each message names what it refuses, and the acceptance test matches `use_multicore`, `sub_core_grids`, `device`, `rank`, `DRAM` (case-insensitive). Then build the scenario dict, apply the taggers, run the SUPPORTED check (`UnsupportedAxisValue`, `_op_contract.py:27`) and the EXCLUSIONS check (`ExcludedCell`, `_op_contract.py:31`) |
| Entry flow | `validate` → descriptor-cache lookup → `ttnn.allocate_tensor_on_device(shape, uint32, ROW_MAJOR, device, input.memory_config())` → write the two buffer addresses into the cached descriptor's common runtime args → `ttnn.generic_op([input_tensor, output_tensor], descriptor)`. No other ttnn op is called, nothing is read on the host, and there is no layout, pad or slice call |
| Descriptor cache (BB-07) | module-level dict keyed by `(tuple(shape), dtype, layout, memory_layout, buffer_type, device id, grid size, CORE_LIMIT, ROW_BLOCK_DEPTH)`. A miss builds the `ProgramDescriptor` once. A hit only replaces `common_runtime_args` (reader: input address; writer: output address). On a device program-cache hit, `apply_descriptor_runtime_args` copies those values into the cached program (`tt_metal/impl/program/program_descriptors.cpp:188-211`). The device-side hash covers kernel sources, core ranges, CT args, CB config and arg counts (`generic_op_device_operation.cpp:78-101`), so every program-shaping value (shape-derived CT args included) is in the key |
| Program cache entries | exactly one per distinct shape; a repeat call with new buffers adds none |
| Plan report (BB-08, BB-21) | `describe_plan(input_tensor) -> dict` with keys `"regime"` (`"row_gather"`), `"core_ranges"` (`ttnn.CoreRangeSet`, one range), `"num_cores"`, `"block_tile_cols"`, `"buffer_depth"`, `"roles"` (`{(x, y): {"reader": "gather_row_block", "writer": "store_row_block", "row_start": s, "rows": n}}`). It is computed by the same host functions that build the descriptor, and it dispatches nothing |
| Mesh | `ttnn.generic_op(io_tensors, ProgramDescriptor)` wraps the descriptor as an SPMD `MeshProgramDescriptor` over the whole mesh (`ttnn/cpp/ttnn/operations/generic/generic_op.cpp:13-22`). Mesh buffers share one address on every device, so each device's kernels read that device's input and write that device's output |
| Trace (BB-16) | the output is allocated inside the call, so at capture time under trace; a replay allocates nothing and builds nothing. No host readback |
| L1 (BB-05) | per core: `buffer_depth · block_tile_cols · 4096` B (Phase 0: 16384 B) ≤ 1347776. No refusal path is needed (Mechanism caps) |

### Rules → design

| Rule | Where the design meets it |
|------|---------------------------|
| bit-identical, no path narrower than 32 bits | NoC byte moves DRAM→L1→DRAM only; no compute kernel; no RISC-V touches values |
| new storage, input unchanged | output allocated per call; the reader only reads x |
| own program, no delegated layout op; one dispatch (BB-01/02) | one `generic_op` of this op's descriptor and kernels |
| each tensor crosses DRAM once | `row_gather` data movement = minimum (Regimes) |
| mesh: every device written from its own input | SPMD mesh program (Host Contract) |
| keywords | `validate()` raises `ValueError` for anything but `True` / `None` |
| cores = the blocking decision | `choose_core_rectangle` + `CORE_LIMIT`; no shape-imposed limit for INPUTS |
| BB-03 | read barrier per block, write barrier per block inside the helper, before return |
| BB-04 / BB-18 | two small data-movement kernels, no compute kernel, built at the default optimization level |
| BB-06 | no races: the writer consumes only after the reader's barrier + push |
| BB-17 | one rectangle per kernel group (both kernels on the same single `CoreRange`) |
| BB-22 | the acceptance test covers only the production case and rule-required cases |

## Acceptance Test

`tests/ttnn/unit_tests/operations/g6_untilize/test_g6_untilize.py` (immutable spec), with
`conftest.py` applying `use_module_device`. BB-22 limits tests to the production case and rule-required
cases. That overrides the generic "≥ 4 shapes" rule: every value test runs `(1, 1, 32, 128)`.

| Test | Checks |
|------|--------|
| `test_production_case[explicit_keywords / default_keywords]` | full-range uint32 draw (`torch.manual_seed(42)`) vs `torch.clone`, bit-exact; output on device, shape, uint32, ROW_MAJOR, DRAM interleaved |
| `test_face_order_and_extremes` | each element encodes its own index (+2^31 on odd); row starts hold 0, 2^32−1, 2^31, 2^24+1 |
| `test_new_storage_input_unchanged` | output buffer address ≠ input's; input read back unchanged |
| `test_repeat_and_program_cache` | second call on a new input adds no program-cache entry and is correct; a repeat on the first input is bit-identical |
| `test_plan_single_rectangle_multicore` | `describe_plan`: regime `row_gather`, exactly one core range, ≥ 2 cores, stable across calls |
| `test_rejects_keywords`, `test_rejects_inputs` | `ValueError` for the keywords; `ValueError`/`RuntimeError` for a host tensor, rank 3, and L1 interleaved input |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| face order (R2) | a row takes faces `2h, 2h+1` (`h = (r mod 32) div 16`). Treating a tile as 32 contiguous rows of 128 B, or swapping faces 1 and 2, misplaces ½ or ¾ of the values | piece formula in Work Distribution; the acceptance test's index-coded input names any misplaced element |
| 64-B read alignment on Blackhole | DRAM reads require `l1 ≡ dram (mod 64)`; any stride or offset that is not a multiple of 64 corrupts silently | row stride `block_tile_cols·128`; CB base DRAM-aligned; offsets multiples of 64 (Mechanism caps) |
| reader/writer quantum mismatch | the writer helper waits `width_in_tiles` tile pages per ≤ 32-row block and computes it from `row_bytes` | the reader pushes `block_tile_cols` pages per block; `block_tile_cols` divides `tensor_tile_cols`; page size = 4096 |
| host plan vs kernel derivation drift | the per-core row range exists twice: the host `describe_plan` and the in-kernel formula | the same formula and the same CT inputs (`tensor_rows`, `num_cores`, `rect_w`). Any drift corrupts output rows and fails every value test |
| stale address on a cache hit | the descriptor is cached and the addresses live only in common runtime args | patch both addresses on every call before `generic_op`; `test_repeat_and_program_cache` uses fresh buffers on the second call |
| writer zone granularity (BB-21) | the helper's write barrier sits inside the `G6U-store` zone | an explicit pre-wait zone separates the CB wait. If the zone audit needs the barrier in its own zone, realize `store_row_block` as a thin custom block op with the helper's contract |
| `to_torch` readback dtype for uint32 | torch may return `uint32` or `int32` | tests compare 32-bit patterns through an `int32` view |
