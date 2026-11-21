# Operation Design: expand

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no NEO used) |
| Goal | `torch.Tensor.expand` made a tensor of its own. Phase 0: a rank-2 bfloat16 TILE tensor `[1, W]` in DRAM whose one row is repeated to `[H_out, W]`. Every output tile is the input tile of its column with row 0 replicated over all 32 rows, built **once** per source tile in L1 and written to every output tile row that a core owns. |
| Math | `output[..., h, w] = input[..., 0, w]` for every `h < H_out` (Phase 0: `changes = h`). In tiles: `out_tile(p, r, j) = row_bcast(in_tile(p, 0, j))` for every `r < output_h_tiles`. |
| References | Specification `eval/specs/expand/spec.py` (authoritative); target `.claude/references/targets/quasar.md`; backend `.claude/references/backends/ttsim.md`; host-program pattern `.claude/references/quasar_op_examples/` (README, `toy_thread_plan/`); DFB semantics `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` Part A; a DM-only reader→writer Quasar program with a scratch tile written many times: `ttnn/cpp/ttnn/operations/experimental/quasar/pad/device/pad_tile_multicore_program_factory.cpp:109-155` and `.../pad/device/kernels/dataflow/writer_pad_tiled.cpp:84-88`; perf catalog `ttnn/ttnn/operations/examples/master.md` (entries `noc_placement`, `width_split`, `double_buffer`, `tile_reorder`, `bank_stagger`, `trid_double_issue`). |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | `quasar` | target page |
| Compute grid | 8 × 4 = 32 clusters (fixed on this part); run time: `device.compute_with_storage_grid_size()` | target page |
| L1 per core | 4,194,304 B (4 MiB) per cluster | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` (`l1_reserve_bytes` = 0 in Phase 0); the figure the query returns on this target is not on the page: ≤ 4,194,304 B. Irrelevant to Phase 0 sizing (peak footprint over INPUTS: 2 × 8 × 2 KiB = 32 KiB bf16, 64 KiB float32; see `l1_ledger.md`) | `unknown` (exact value) |
| Host program | `ttnn.ProgramSpec` (Metal 2.0) + one `ttnn.generic_op` — `.claude/references/quasar_op_examples/` (`toy_thread_plan/toy_thread_plan_program_artifacts.py`) | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1,2,4} for a compute kernel), 6 user DM harts (R + W ≤ 6; on STRIDED rings R and W divide C); 32 DFBs per cluster, ring ≤ 1,048,560 B; 16 semaphores; DM binary ≤ 48 KiB | target page *Cluster model*, *Thread plan* |
| Unavailable here | `ttnn.ProgramDescriptor` / Metal-1 kernels and the legacy C++ program-factory op (refused at kernel creation). `bfloat8_b`, `bfloat4_b` (no path; outside TARGET anyway). Copy-path transposes are refused on Quasar (`tile_move_copy.h:53-58`) — irrelevant, nothing is transposed. | target page |

All worked numbers below are for **quasar (32 clusters, bf16 tile page 2,048 B)**. Every run-time predicate queries the device.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on device; Phase 0: bf16, TILE, DRAM interleaved, rank 2, `shape[-2] == 1` | — | tensor binding |
| `output_shape` | list or tuple of int (R3) | yes | `len ≥ rank`; per matched dim: `-1`, the input size, or any `n ≥ 1` where input size is 1; added (leading) dims: `n ≥ 1` (never `-1`) | — | host only (derived: `output_h_tiles`, `tensor_w_tiles`, `num_planes`) |

### Host-side contract (entry point, in this order)

| Step | Action | Error |
|------|--------|-------|
| 1 | `output_shape = [int(d) for d in output_shape]` — list and tuple both accepted (R3) | — |
| 2 | `len(output_shape) < input.rank` | `ValueError` (X1) |
| 3 | an added dim (index < `len - rank`) equal to `-1` | `ValueError` (X3) |
| 4 | per matched dim `i`: `-1` → keep; `== in[i]` → keep (R2); `in[i] == 1` and `n ≥ 1` → expand; `n < -1` or (`in[i] != 1` and `n != in[i]`) | `ValueError` (X2) |
| 5 | a size of `0` (unspecified by DECISIONS) | `NotImplementedError` |
| 6 | axes: `memory_axes(input)` (R7, `eval.memory.memory_axes`), dtype, layout, and `INPUT_TAGGERS = AXES` (imported from the spec, never redefined) on `(tuple(input.shape), output_shape)` | — |
| 7 | `validate()` against `SUPPORTED` (= `PHASE0`) | `NotImplementedError` (R8) |
| 8 | `result_shape` = output_shape with `-1` resolved; allocate output `ttnn.allocate_tensor_on_device(TensorSpec(result_shape, input.dtype, input.layout, memory_config=input.memory_config()), device)` — the input's buffer, interleaved (R6) | — |
| 9 | build the ProgramSpec and dispatch **one** `ttnn.generic_op([input, output], spec, run_args, tensor_indices)` | — |

There is no index axis (`dim`) to canonicalize; the only canonicalization is step 4's `-1` / equal-size reading (R2), done before the support check.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | Phase 0: `[1, W]`, any `W ≥ 1` (TARGET: rank 2-4, any dim of size 1 may expand) |
| Dtype | Phase 0: bfloat16 (TARGET adds float32) |
| Layout | Phase 0: TILE (TARGET adds ROW_MAJOR) |
| Memory | Phase 0: DRAM interleaved (TARGET adds L1 interleaved) |

### Output

| Property | Value |
|----------|-------|
| Shape | `result_shape` = `output_shape` with each `-1` replaced by the input's size (spec `_result_shape`) |
| Dtype | input's |
| Layout | input's |
| Memory | input's `memory_config()` (buffer type and interleaved) — R6 |

## Blocking Model

The op computes nothing; its cost is the output's bytes crossing DRAM once. The scheme is
**read once, build once, write many**: a core reads each source tile it needs once, builds the
broadcast tile in L1, and writes that one L1 tile to every output tile of its run that maps to it.

Index space (Phase 0 specialization in brackets):

| Symbol | Meaning | Formula |
|--------|---------|---------|
| `num_planes` | product of the result's leading dims (all but last two) [1 for rank 2] | `prod(result_shape[:-2])` |
| `tensor_w_tiles` | tile columns of input and output | `ceil(W / 32)` |
| `output_h_tiles` | tile rows of one output plane | `ceil(H_out / 32)` — per plane, never `floor(planes·H/32)` |
| `num_sources` | distinct source tiles | `num_planes · tensor_w_tiles` [= `tensor_w_tiles`] |
| `fan_out_tiles` | output tiles fed by one source (the expanded axis) | `output_h_tiles` |
| `total_out_tiles` | output pages | `num_sources · fan_out_tiles` = output `buffer_num_pages()` |

**Source-major order.** Output tiles are linearized with the source outermost and the expanded
(fan-out) axis innermost: `L = s · fan_out_tiles + r`, `s = p · tensor_w_tiles + j`.
Then `output page = p · output_h_tiles · tensor_w_tiles + r · tensor_w_tiles + j` and
`input page = p · tensor_w_tiles + j` (input has one tile row per plane). A core owns one contiguous
run `[run_start, run_start + run_tiles)` of `L`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| output tile rows `r` (expanded axis, `output_h_tiles`) | **independent**; and the input is **reuse-shared** along it (every row reads the same source tile) | `fan_extent` (rows written from one resident built tile) | the core's whole row segment of that source (no L1 cost: rows are writes from the same entry, not resident data) | derived in-kernel from `run_start`, `run_tiles`, `fan_out_tiles` (RT/CT args) | contiguous runs of `L` (source-major) over all engaged clusters | knob-turn |
| tile columns `j` (`tensor_w_tiles`) | **independent**; the source varies along it, results never span it | `block_src_tiles` (source tiles resident per block) | `min(max_src_tiles_per_core, block_src_tiles_cap)` — every source of the core's run in one block (traced call: 3) | one host constant computed in the factory, passed as CT arg; ring capacity derived from it | contiguous runs of `L` | knob-turn |
| leading planes `p` (`num_planes`) | **independent** (not expanded in Phase 0); when a leading dim is expanded it becomes part of the fan-out (reuse-shared) | folded into `block_src_tiles` (sources `s = p·Wt + j` are one linear axis) | `num_planes = 1` (rank 2) — not a collapsed knob: Phase 0 has no leading dims | host `num_planes` (CT arg) | contiguous runs of `L` | knob-turn (rank 3/4 `h`), build-variant (expanded leading dim) |
| rows inside a tile (32) | the replication axis of the build: row 0 copied to rows 1..31 | none — the tile is the atomic page | whole tile | — | not assigned (inside one block operation) | — |
| dependent axis | **none**: expand has no reduction, scan or combine | — | — | — | — | — |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_row_bcast_tiles` | `buffer_depth` (host constant) | 2 | The reader loads and builds block `b+1` while the writer is still fanning out block `b`. With one block per core (common in Phase 0) the second half is unused; it costs `block_src_tiles · page_size` and becomes live as soon as a core has ≥ 2 blocks (clamp binds, or the overlap lamp lowers `block_src_tiles`). |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DFB ring size ≤ 1,048,560 B (target page) | `buffer_depth · block_src_tiles · page_size` | `block_src_tiles_cap = floor(min(DFB_RING_MAX_BYTES, l1_budget) / (buffer_depth · page_size))` with `DFB_RING_MAX_BYTES = 1_048_560` (host constant from the target page; no Python query exists) and `l1_budget = get_max_worker_l1_unreserved_size() - l1_reserve_bytes`; bf16: 255 tiles, fp32: 127 | ring creation fails or overlaps other L1 |
| Ring blocks must not straddle the wrap | push/pop quantum | capacity is exactly `buffer_depth · block_src_tiles`; every push/pop is the nominal `block_src_tiles`, also for a ragged last block (unused entries are pushed and popped unwritten) | wrong entry addresses after a wrap; hang at kernel exit on unbalanced credits |
| L1 alignment 16 B / NoC copy alignment | replication copy offsets and sizes | all copies are multiples of `face_row_bytes = 16 · element_size` (32 B bf16, 64 B fp32) at offsets that are multiples of it | misaligned self-reads corrupt rows |
| Whole-page DRAM read (DRAM alignment 64 B) | input read granularity | Phase 0 reads the whole input page (`page_size`) into the entry | partial-page DRAM reads at non-matching alignment corrupt data (see lamp "input read granularity") |
| Stateful self-read path is single-packet (`local_copy_helpers_dataflow.hpp:62-76`) | replication copy size | Phase 0 replication uses plain `noc.async_read` from `local_addr(...)` (multi-size steps up to `2 · face_bytes` = 1,024 B bf16 / 2,048 B fp32); the stateful pair is only for fixed-size copies ≤ `NOC_MAX_BURST_SIZE` | static_assert / truncated transfer |
| 32-bit page ids and RT args | `total_out_tiles` | host asserts `total_out_tiles < 2^32` | wrap-around addressing |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R-H: tile row-broadcast, source-major runs** | **built** (Phase 0) | `layout == TILE and changes == "h"` (the only Phase 0 cell class; validate() refuses the rest) | `block_src_tiles` source tiles × each one's row segment (`fan_extent`) | Boundary: DRAM. Minimum: output once (`total_out_tiles` pages), input once (`num_sources`). Actual: output once; input `≤ num_sources + engaged_cores − 1` page reads (a run boundary that falls inside a source's rows reads that source on both cores). Traced call: 96 reads vs 80 minimum, 3,360 writes (input excess = 0.5 % of bytes moved). No cross-core traffic. Local: ~1 page of self-aimed L1 copies per source (build). | Per block: one `reserve/push` + one `wait/pop`, one DRAM-read barrier, 5 replication barriers, one write barrier. A block of all the core's sources pays them once per core; the writes themselves (one NoC write per output page) are irreducible on interleaved pages. |
| R-H-colalign: R-H with run boundaries rounded to source boundaries | deferred — saves ≤ `engaged_cores − 1` input page reads (traced: 16 × 2 KiB = 32 KiB, 0.5 % of bytes) at the price of up to one source column of imbalance per core (traced: busiest core 126 vs 105 output tiles, +20 % critical path) and of capping engaged cores at `num_sources` (narrow outputs, e.g. W=17 → 1 core). Reachable by a host-split change only: kernels take arbitrary runs. | as R-H | as R-H | input exactly once | as R-H |
| R-H-mcast: one injector reads each boundary source and multicasts it to the cores sharing it | rejected — superseded by source-major ordering, which already confines sharing to ≤ 1 source per run boundary; the multicast would save the same ≤ `engaged_cores − 1` page reads as R-H-colalign while adding semaphores and an injector role (operand-reuse check: the input is reuse-shared along the split axis `r`, but only boundary sources are actually shared) | — | — | input once + mcast fan-out | — |
| Row-split: each core a range of output tile rows, all columns | rejected (dead end) — every engaged core reads every source: input `engaged_cores · num_sources` reads (traced: 2,560 vs 80, +74 % of total bytes); superseded by R-H everywhere | — | — | input × cores | — |
| Row-major flat runs of output pages | rejected (dead end) — a run of ≥ `tensor_w_tiles` pages touches every source: same re-read as Row-split | — | — | input × cores | — |
| Column-only split (whole columns per core, `min(Wt, cores)` cores) | rejected — R-H-colalign is the same traffic with balanced occupancy where Wt < cores fails (catalog `width_split`: the converse trap of stranding cores) | — | — | input once | — |
| Materialize every output tile in the ring (implicit-sync writer, one entry per output page) | rejected (dead end) — the reader would copy the built tile `fan_extent` times in L1 (L1 traffic × fan-out) to feed a one-write-per-entry writer; superseded by write-many-from-one-entry | — | — | output once, + `total_out_tiles` local tile copies | — |
| Build on a NEO (unpack row-broadcast copy, pack) | rejected — adds DM→NEO→DM ring hops per source and a format-conversion path (R1: float32 must not pass a narrower format; DEST/unpack config would have to be proven exact) for a bound that is DRAM writes; the DM byte copy is exact for every dtype by construction | — | — | as R-H + 2 ring hops per source | — |
| R-LEAD: `changes ∈ {leading, added}` (whole planes repeated, TILE) | deferred — outside PHASE0. Same loop nest: build = identity (the source page is written as is), fan-out = the expanded leading indices; a CT `build_mode` and the index map `L → (source, fan)` generalize | `changes ∈ {leading, added}` and no H/W expansion | as R-H | minimum (input once per run boundary as R-H) | as R-H |
| R-W: `changes == w` (TILE column-broadcast) | deferred — outside PHASE0. Same loop nest; build = copy column 0 to columns 1..31 (`tile_column_dataflow.hpp:92` `copy_tile_column`, RISC stores + data-cache flush, line 23); fan-out axis = tile columns of the row | `changes == "w"` | source tiles × column segment | as R-H | as R-H |
| R-HW / combined (H and W, or H with leading dims, e.g. `(1,1)→(64,128)`, `(2,1,1,128)→(2,4,50,128)`) | deferred — outside PHASE0. Build = scalar broadcast (both builds), fan-out = product of every expanded axis; same structure | per `changes` + shape | as R-H | as R-H | as R-H |
| R-RM: ROW_MAJOR (`h`: the one input stick written to every output row; `w`: each stick built from one element) | deferred — outside PHASE0. Page = stick (`buffer_page_size()`), build = identity (`h`) or element fill (`w`); same ring, same runs over pages | `layout == ROW_MAJOR` | sticks | minimum | as R-H |

Placement/dtype (TARGET `memory_config: L1`, `dtype: float32`) are not regimes: an L1 input is read through the same accessor and the output is allocated in L1 (R6); float32 doubles `page_size` and `face_row_bytes`, both derived from the page size. Both are knob-turns of R-H.

### Traffic ranking

| Rank | Split | DRAM reads (input) | DRAM writes (output) | Cross-core | Verdict |
|------|-------|--------------------|----------------------|------------|---------|
| 1 | Column-aligned source-major runs (R-H-colalign) | `num_sources` (min) | once (min) | 0 | cheapest traffic; deferred (imbalance, core cap) |
| 1 + ε | **Source-major contiguous runs (R-H)** | `≤ num_sources + cores − 1` | once | 0 | **chosen**: traffic within 0.5 % of the minimum on the traced call, balanced to ±1 tile, fills the grid whenever `total_out_tiles ≥ cores` |
| 1 + ε | Same + multicast of boundary sources | `num_sources` | once | ≤ cores − 1 tiles | rejected (same saving as colalign, more machinery) |
| 3 | Row-split / row-major flat runs | `cores · num_sources` | once | 0 | rejected |
| — | Split a dependent axis + combine | n/a — expand has no dependent axis | | | |

Write order: reads on NoC0 (reader default), writes on NoC1 (writer default) and `split_work_to_cores(..., row_wise=True)` (catalog `noc_placement`).

### Block schedule

Per core (reader and writer are separate asynchronous kernels; blocks are pipelined through `cb_row_bcast_tiles`):

```cpp
// num_blocks_this_core = ceil(src_tiles_this_core / block_src_tiles)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_source_block(block_idx);      // reader
    replicate_rows_block(block_idx);   // reader
    store_fanout_block(block_idx);     // writer
}
```

| Operation | Kernel | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|--------|-------------|--------------------|-----------------------------------|
| `load_source_block` | reader | `block_src_tiles` input pages (sources `s_first .. s_first + n − 1` of this block, `n ≤ block_src_tiles`) | the reserved block of ring entries | 1 `reserve_back(block_src_tiles)` per block; `n` whole-page DRAM reads tagged `{.trid = 0}`; **1** read barrier per block |
| `replicate_rows_block` | reader | the same `n` entries, each a full tile | entries stay reserved until push | doubling copies inside faces 0 and 1 — rows `[0,1)→[1,2)`, `[0,2)→[2,4)`, `[0,4)→[4,8)`, `[0,8)→[8,16)` (2 self-aimed reads per tile per step) — then faces 0-1 → faces 2-3 (1 read of `2·face_bytes` per tile): 9 self-reads per tile, each step issued for **all `n` tiles** before **one** barrier per step: **5 barriers per block**; then **1** `push_back(block_src_tiles)` (nominal) |
| `store_fanout_block` | writer | `n` built tiles × each one's row segment `[r_begin(s), r_end(s))` | the block's entries, read-only, until pop | 1 `wait_front(block_src_tiles)` per block; one whole-page NoC write per output page from entry `k` (`page = p·output_h_tiles·Wt + r·Wt + j`), tagged `{.trid = 0}`; **1** write barrier per block; **1** `pop_front(block_src_tiles)` per block |

No inits (no compute kernel). Rows 1..31 of the input tile (padding, R4) are overwritten by the build and never reach the output. Padding columns of the last tile column and padding rows of the last output tile row are written with whatever the source tile / replication puts there; nobody reads them.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap | `block_src_tiles` = all of a core's sources gives one block per core: the writer starts only after every read and build of the core. Harmless when `fan_out_tiles` is large (traced: 3 reads vs 105 writes) but serializes when `output_h_tiles` is 1 (e.g. `(1,8192)→(17,·)`: 8 reads, 8 writes per core). | `block_src_tiles = ceil(max_src_tiles_per_core / 2)` (≥ 2 blocks per core, depth 2 then overlaps) |
| Grid synchronization | Full occupancy on tiny outputs leaves 1-2 output tiles per core (e.g. `(1,512)→(128,·)`: 64 tiles on 32 cores, 32 input reads vs 16 minimum); launch and per-core setup dominate. | `max_cores` knob (Phase 0: the full queried grid) lowered so each core gets ≥ `fan_out_tiles` tiles |
| Replication schedule | 5 dependent barriers per block; tiny 32 B NoC copies for the first steps. | (a) direct fan from row 0: 30 independent self-reads of `face_row_bytes` + 1 faces-copy, 2 barriers per block; (b) RISC stores + data-cache flush (`tile_column_dataflow.hpp:23`) |
| Input read granularity | The whole input page is read although only row 0 of faces 0 and 1 (`2 · face_row_bytes` = 64 B of 2,048 B) is used; matters only when `fan_out_tiles` is 1 (reads ≈ writes). | two partial reads (offset 0 and `face_bytes`, `face_row_bytes` each), alignment checked against `get_dram_alignment()` |
| DRAM bank order of writes | A core writes one source's rows: pages `j, j+Wt, j+2Wt, …` — the same bank whenever `Wt` is a multiple of the bank count (catalog `bank_stagger`). | within a block, write row-major across the block's sources (`for r: for k`) so consecutive writes hit consecutive pages |
| Writer threads | One writer hart issues every output write; the plan's (1,·,1) is the target page's data-movement default, unmeasured for explicit-sync fan-out writes. | W = 2 writer harts (blocks dealt round-robin over a STRIDED consumer), measured on `ttsim-perf` |
| Write barrier granularity | One write barrier per block drains the NoC once per block (catalog `trid_double_issue`). | two user trids alternating per block, barrier on the older one before its pop |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM input page → ring entry | TILE page (bf16 2,048 B) | reader, explicit sync: `reserve_back`, `noc.async_read<NocOptions::TXN_ID>(acc_in, cb_row_bcast_tiles, page_size, {.page_id}, {.offset_bytes = k·page_size}, {.trid = 0})`, barrier on trid 0 (pattern `toy_thread_plan/kernels/reader.cpp:29-43`) | one read per source per core |
| Ring entry → ring entry (row replication) | same bytes | reader: self-aimed `noc.async_read(UnicastEndpoint, cb_row_bcast_tiles, bytes, dataflow_kernel_lib::local_addr(src, noc.get_noc_id()), {.offset_bytes})`, trid 0, barrier per step; then `push_back` | byte copy: exact for every dtype (R1) |
| Ring entry → DRAM output pages (fan-out) | same | writer, explicit sync: `wait_front`, `noc.async_write<NocOptions::TXN_ID>(cb_row_bcast_tiles, acc_out, page_size, {.offset_bytes = k·page_size}, {.page_id}, {.trid = 0})` per output page, barrier, `pop_front` (pattern `toy_thread_plan/kernels/writer.cpp:30-33`) | the same entry is the source of `fan_extent` writes |
| Cross-core | — | none | R-H-mcast (rejected) would use `ttnn.Mcast` + `McastArgs` from an injector to the cores sharing a boundary source |

Placement axis (TARGET `sharding = [interleaved]` only): DRAM and L1 interleaved both read through `TensorAccessor`; output allocated in the input's buffer (R6). No sharded scheme is in TARGET; a sharded input is refused by name through `memory_axes` (R7).

Structural impossibilities: none beyond the spec's (`INVALID = []` is right: every TARGET cell is buildable).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one output tile in source-major order `L`; a core's contiguous run is its blocks |
| Grid | `device.compute_with_storage_grid_size()` (quasar: 8 × 4), optionally limited by host knob `max_cores` (Phase 0: all) |
| Per-core work | `ttnn.split_work_to_cores(grid, total_out_tiles, row_wise=True)`; `run_start` = cumulative, `run_tiles` = group count (pattern `toy_thread_plan_program_artifacts.py:48-60`) |
| Remainder | `split_work_to_cores` groups (±1 tile); `output_h_tiles = ceil(H_out/32)`, `tensor_w_tiles = ceil(W/32)` per plane; derived per core on host: `src_tiles_this_core = (run_start + run_tiles − 1) // fan_out_tiles − run_start // fan_out_tiles + 1`; `max_src_tiles_per_core` = max over cores → `block_src_tiles` |
| Bound | **data movement**: zero compute per tile; per output tile one 2 KiB DRAM write, per source one read plus ~1 page of local copies |
| Thread plan | `THREAD_PLAN = (R, C, W) = (1, 0, 1)` — one reader hart, **no compute kernel** (no NEO; the page's C ∈ {1,2,4} rule concerns a compute kernel, which this program does not have; precedent: tt-metal's Quasar pad tile program is reader + writer only), one writer hart. Host constant; counts passed as CT args `reader_threads`, `writer_threads`. Data-movement bound → the page's *Choosing the plan* says one of each. |
| Unit dealt to compute threads | n/a (no compute threads). Ring STRIDED DM→DM with one producer and one consumer thread. |
| Tests the plan needs | (1) every output page is written by the writer from a reader-built entry: the golden cells with `output_h_tiles ≥ 2` and `num_sources ≥ cores` (traced call) show both kernels' full work; a probe records `src_tiles_this_core` / reads per core (structural count: input reads ≤ `num_sources + engaged_cores − 1`). (2) In one device session: another program, then `expand`; and `expand`, then `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)` with the result checked. |

Worked Phase 0 values (quasar, bf16, page 2,048 B):

| INPUTS entry | output_h_tiles | tensor_w_tiles | total_out_tiles | cores | tiles/core | block_src_tiles | ring bytes | input reads (min) |
|--------------|---------------:|---------------:|----------------:|------:|-----------:|----------------:|-----------:|------------------:|
| `(1,2560)→(1330,-1)` | 42 | 80 | 3,360 | 32 | 105 | 3 | 12 KiB | 96 (80) |
| `(1,64)→(32,-1)` | 1 | 2 | 2 | 2 | 1 | 1 | 4 KiB | 2 (2) |
| `(1,512)→(128,512)` | 4 | 16 | 64 | 32 | 2 | 1 | 4 KiB | 32 (16) |
| `(1,128)→(1024,-1)` | 32 | 4 | 128 | 32 | 4 | 1 | 4 KiB | 32 (4) |
| `(1,4096)→(32,-1)` | 1 | 128 | 128 | 32 | 4 | 4 | 16 KiB | 128 (128) |
| `(1,8192)→(17,-1)` | 1 | 256 | 256 | 32 | 8 | 8 | 32 KiB | 256 (256) |
| `(1,50)→(100,-1)` | 4 | 2 | 8 | 8 | 1 | 1 | 4 KiB | 8 (2) |
| `(1,17)→(32,17)` | 1 | 1 | 1 | 1 | 1 | 1 | 4 KiB | 1 (1) |

One regime is built: no regime-selection function and no regime-pinned tests are needed. The `block_src_tiles` clamp is not a regime (same code, more blocks per core); a test with a forced small `block_src_tiles_cap` (e.g. 1) on the traced shape exercises the multi-block path and ring wrap.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `cb_row_bcast_tiles` | 0 | `page_size` = output `buffer_page_size()` (bf16 2,048 B) | `buffer_depth · block_src_tiles` | spans the source axis (`block_src_tiles` built tiles live while their fan-out is written) × depth; streams over the expanded axis (rows are writes from the same entry) | input dtype (bytes only; never unpacked) | reader | writer | explicit on both sides: listed in `disable_dfb_implicit_sync_for` of reader and writer; every NoC transfer of both kernels carries `{.trid = 0}` with barriers on the same id; both kernels end with `noc_async_read_set_trid(0, …)` / `noc_async_write_set_trid(0, …)` | whole program |

One buffer is the minimum for two kernels: the writer needs an L1 source, and the build happens in place in the entry the input page lands in (no separate input buffer: the DRAM read *is* the first step of the build).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_source_block` | `n ≤ block_src_tiles` input pages | raw NoC API (dataflow) | DRAM input via `TensorAccessor` | `cb_row_bcast_tiles`, `block_src_tiles` reserved, `n` filled | reserved, not pushed |
| 2 | `replicate_rows_block` | the same `n` tiles | `dataflow_kernel_lib::local_addr` (self-aimed source) + raw `noc.async_read` | `cb_row_bcast_tiles` (own reserved entries) | `cb_row_bcast_tiles`, `block_src_tiles` pushed | pushed (nominal count) |
| 3 | `store_fanout_block` | `n` tiles × row segments | raw NoC API (dataflow) | `cb_row_bcast_tiles`, `block_src_tiles` waited | DRAM output via `TensorAccessor` | popped after write barrier |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_source_block` | raw_api | `DataflowBuffer::reserve_back`, `Noc::async_read<NocOptions::TXN_ID>(TensorAccessor, DataflowBuffer, size, {.page_id}, {.offset_bytes}, {.trid=0})`, `Noc::async_read_barrier<NocOptions::TXN_ID>` | `.claude/references/quasar_op_examples/toy_thread_plan/kernels/reader.cpp:29-43` | `page_size`, `.offset_bytes = k·page_size` | DRAM input | `cb_row_bcast_tiles` | `block_src_tiles` (reserve count, loop extent) |
| `replicate_rows_block` | helper + raw_api | `dataflow_kernel_lib::local_addr(addr, noc.get_noc_id())`; `Noc::async_read(UnicastEndpoint, DataflowBuffer, bytes, src_args, {.offset_bytes})` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57` (usage `local_copy_helpers_dataflow.inl:16-28`) | source = entry address (`get_write_ptr()` + offsets), bytes ∈ {1,2,4,8}·`face_row_bytes`, `2·face_bytes`; `face_row_bytes = page_size/64`, `face_bytes = page_size/4` | `cb_row_bcast_tiles` | `cb_row_bcast_tiles` | `block_src_tiles` (tiles per step before each barrier) |
| `replicate_rows_block` (trid) | helper | `dataflow_kernel_lib::set_read_trid`, `async_read_barrier_with_trid` (or the `NocOptions::TXN_ID` overloads with `{.trid=0}`) | `local_copy_helpers_dataflow.hpp:147`, `:158` | trid 0 | — | — | — |
| `store_fanout_block` | raw_api | `DataflowBuffer::wait_front`, `Noc::async_write<NocOptions::TXN_ID>(DataflowBuffer, TensorAccessor, size, {.offset_bytes}, {.page_id}, {.trid=0})`, `Noc::async_write_barrier<NocOptions::TXN_ID>`, `pop_front` | `.claude/references/quasar_op_examples/toy_thread_plan/kernels/writer.cpp:30-33`; many writes from one scratch tile: `ttnn/cpp/ttnn/operations/experimental/quasar/pad/device/kernels/dataflow/writer_pad_tiled.cpp:84-88` | `page_size`, `.offset_bytes = k·page_size`, `.page_id` per output tile | `cb_row_bcast_tiles` | DRAM output | `block_src_tiles` (wait/pop count), `fan_extent` (writes per entry) |
| host | raw_api | `ttnn.experimental.ProgramSpec`, `KernelSpec(num_threads=…)`, `DataflowBufferSpec`, `DataMovement2XXConfig(disable_dfb_implicit_sync_for=["cb_row_bcast_tiles"])`, `ttnn.split_work_to_cores`, `ttnn.generic_op` | `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py:48-68,94-160` | reader: producer of the ring, tensor `input`; writer: consumer, tensor `output`; no compute KernelSpec | — | — | `block_src_tiles`, `buffer_depth`, `max_cores`, `THREAD_PLAN` |

**Helpers considered and rejected** (for every `raw_api` row):

| Raw-API block op | Candidate helper | File:Line | Why it cannot express the shape |
|------------------|------------------|-----------|---------------------------------|
| `load_source_block`, `store_fanout_block` | compute eltwise `copy` (`CopyTile → PackTile`) | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:109-113` | a compute-kernel helper (needs a NEO, unpack/pack); the op has no compute kernel (regime "Build on a NEO" rejected) and DRAM↔L1 transfers are data-movement work no compute helper performs |
| `load_source_block`, `store_fanout_block` | `local_copy_helpers_dataflow` stateful reads | `local_copy_helpers_dataflow.hpp:62-76` | self-aimed L1→L1 only (`local_addr` source); these transfers are DRAM↔L1 through a `TensorAccessor` |
| `store_fanout_block` | multicast `McastArgs` / `ttnn.Mcast` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp` | core-to-core multicast; the fan-out here is to interleaved DRAM pages, one write per page |
| `replicate_rows_block` | `tile_column_dataflow.hpp` `copy_tile_column` / `fill_tile_column` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:70,92` | replicates a **column** (the `w` build, R-W); a row broadcast copies whole face rows, which are contiguous (`face_row_bytes`) and go as NoC copies |
| `replicate_rows_block` | `l1_helpers.hpp` `fill_l1_range` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:100-101` | fills a constant value, not a copy of a row read at run time |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Sticky transaction tag | Both kernels use explicit sync on Quasar; an untagged transfer reuses the hart's last tag | every transfer (DRAM reads, self-aimed replication reads, DRAM writes) carries `{.trid = 0}`; the ring is listed in both kernels' `disable_dfb_implicit_sync_for`; each kernel's last statement resets its tag; the two one-session tests of Work Distribution are required |
| Writing many times from one entry | `pop_front` before the writes have left would let the reader overwrite the entry mid-write | write barrier on trid 0 before every `pop_front` (DFB doc Part A: firmware waits only for implicit-sync writes) |
| Ragged last block | a core's last block may have fewer than `block_src_tiles` sources | push/pop always `block_src_tiles` (nominal); `n` narrows only the reads, copies and writes |
| Run boundary inside a source | the first and last source of a run have partial row segments `[r_begin, r_end)` | segment bounds derived from `run_start`, `run_tiles`, `fan_out_tiles` (`r_begin = max(0, run_start − s·fan)`, `r_end = min(fan, run_start + run_tiles − s·fan)`); test cells with odd core boundaries (traced call: 16 split sources) cover it |
| Replication read-after-write | each doubling step reads rows the previous step wrote | one barrier per step (all block tiles batched); the DRAM read barrier precedes step 1 |
| RISC-store alternative | a DM core's stores land in its data cache (`tile_column_dataflow.hpp:23-24`) | Phase 0 replicates with NoC self-reads only; switching to stores (lamp) requires the cache flush before `push_back` |
| Padding rows (R4) | the input tile's rows 1..31 are padding of unknown content | rows 1..31 of faces 0/1 and all of faces 2/3 are overwritten from row 0 before push; only row 0 ever propagates |
| `h_non_aligned` results | last output tile row has rows ≥ H_out | written with the same built tile; they are padding of the output, never read |
| float32 later (R1) | a value must not pass a narrower format | no unpack/pack anywhere: the op copies bytes; `face_row_bytes` and `page_size` derive from the page size, so float32 needs no kernel change |
| Large outputs with tiny fan-out | `block_src_tiles` would exceed the ring cap | clamp `block_src_tiles_cap` from the ring maximum and the queried L1 budget; more blocks per core, same code |
| Single dispatch | the expansion must be one device program | host validation + allocation + exactly one `ttnn.generic_op`; no `ttnn.repeat` / `ttnn.copy` / `to_layout` around it |
