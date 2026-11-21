# Operation Design: scatter

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel: every output element is a bitwise copy of an `input` or `src` element, R1) |
| Goal | `torch.scatter(input, dim, index, src)` as one `ttnn.ProgramSpec` dispatch: the output is `input`'s line along `dim` with the places `index` names overwritten by `src`'s elements of the same line. Phase 0: rank 2, `dim` -2, bfloat16 TILE `input`/`src`, INT32 row-major `index`, all in DRAM interleaved. |
| Math | `out = input`; for each source coordinate c (ascending along `dim`): `p = index[c]`; if `0 <= p < input.shape[dim]`: `out[c with c[dim] := p] = src[c]`. Rank 2, dim -2: `out[index[i][j]][j] = src[i][j]`. |
| References | `eval/specs/scatter/spec.py` (authoritative); `.claude/references/targets/quasar.md`; `.claude/references/backends/ttsim.md`; `.claude/references/quasar_op_examples/` (host-program pattern); `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; `tt_metal/hw/inc/api/scratchpad.h`; `tt_metal/hw/inc/api/kernel_thread_globals.h`; `tt_metal/hw/inc/internal/tt-2xx/risc_common.h` (Quasar DM cache model); catalog `ttnn/ttnn/operations/examples/master.md` entries `width_split`, `split_reader`, `tile_reorder`, `double_buffer`, `noc_placement` |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | `quasar` | target page |
| Compute grid | 8 × 4 = 32 clusters (queried: `device.compute_with_storage_grid_size()` returned `8-4` on this run's simulator) | target page + probe `tests/ttnn/unit_tests/operations/scatter/probes/probe_002.py` |
| L1 per core | 4,194,304 B per cluster, shared by its 4 NEOs and 8 DM cores | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES` = 3,963,328 − 65,536 = **3,897,792 B** on this target (query value read on this run's simulator) | probe |
| Alignment | L1 16 B (`ttnn.get_l1_alignment()`), DRAM 64 B (`ttnn.get_dram_alignment()`) | target page + probe |
| Host program | `ttnn.ProgramSpec` (`ttnn.experimental.ProgramSpec` / `KernelSpec` / `ScratchpadSpec` / `WorkUnitSpec`, launched by one `ttnn.generic_op`) — reference `.claude/references/quasar_op_examples/` (README, `toy_thread_plan/toy_thread_plan_program_artifacts.py`) | target page |
| Threads per core | 4 NEOs (compute kernel threads ∈ {1,2,4}); 6 user DM harts (a DM kernel's `num_threads` ≤ 6, `tt_metal/impl/metal2_host_api/program_spec/validation/kernel_spec.cpp:123-129`; all DM kernels of a work unit together ≤ 6, `.../placement/work_unit.cpp:62-90`); a work unit needs no compute kernel (`work_unit.cpp` only bounds the counts from above). R \| C and W \| C hold for STRIDED rings only; this design has no ring. | target page *Thread plan* + host validation |
| DM-core caches | Every DM hart has a private 4 KB write-back L1 D$ and shares a 128 KB write-back L2; DM harts are coherent with each other but **not** with NoC traffic into L1 (`tt_metal/hw/inc/internal/tt-2xx/risc_common.h:194-215`). CPU access to bytes a NoC transfer fills or drains goes through `scoped_lock` (acquire = L2 invalidate of the range, release = L2 flush of the range; `tt_metal/hw/inc/internal/scoped_lock_cache_ops.h:14-30`). | tt-metal source |
| Unavailable here | `ttnn.ProgramDescriptor` / Metal 1 kernels and the legacy program-factory device op (refused at kernel creation, target page). `bfloat8_b` / `bfloat4_b` (no path; not in this TARGET anyway). uint16/uint32 as a DFB/scratchpad `data_format_metadata` (`validation/resource/scratchpad.cpp:30-36`) → the scratchpads here carry **no** format metadata. An untagged NoC transfer: not used (every transfer of this op is tagged). | target page |

All worked numbers below are for this target (32 clusters, 3,897,792 B budget). Every run-time predicate uses the queries, never these values.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input` | `ttnn.Tensor` | yes | rank 2–4 (TARGET); Phase 0: rank 2, bfloat16, TILE, DRAM interleaved | — | tensor |
| `dim` | `int` | yes | `[-rank, rank-1]`; canonicalized to the negative form (`dim - rank` when `dim >= 0`) **before** the support check (R4). Phase 0: -2 | — | host only (selects the regime) |
| `index` | `ttnn.Tensor` | yes | INT32, ROW_MAJOR, same rank as `input`, same placement axes; `index_form == "lines"` | — | tensor |
| `src` | `ttnn.Tensor` | yes | `input`'s dtype and layout, `index`'s shape | — | tensor |

Public signature is exactly `scatter(input, dim, index, src)` (spec stub). Blocking knobs and test probes are keyword arguments of the private program-artifacts function, never of `scatter`.

### Entry point: order of checks in `validate()`

| # | Check | Failure |
|---|-------|---------|
| 1 | `index.rank == src.rank == input.rank` (X2) | `ValueError` |
| 2 | `-rank <= dim < rank` (X1) | `IndexError` |
| 3 | canonicalize `dim` to negative (R4); build entry `(input.shape, dim, index.shape, src.shape)` | — |
| 4 | axes = `INPUT_TAGGERS` (= spec `AXES`, imported, never redefined) on the entry; `index_form == "other"` (X3, X4, R9) refused by name | `NotImplementedError` naming `index_form` |
| 5 | `index.dtype == ttnn.int32`, `index.layout == ROW_MAJOR`, `src.dtype == input.dtype`, `src.layout == input.layout` (R5) | `NotImplementedError` |
| 6 | `eval.memory.memory_axes(t)` for each of the three tensors (R8); all three must agree and lie in SUPPORTED | `NotImplementedError` |
| 7 | the cell (`dtype`, `layout`, `memory_config`, `sharding`, `dim`, `index_form`, `alignment`, `rank`) inside SUPPORTED (R10) | `NotImplementedError` (pipeline refusal type) |
| 8 | regime predicate `resident_line_fits` (Regimes) | `NotImplementedError` naming the deferred `banded_output_line` regime |

The output is `ttnn.allocate_tensor_on_device(input.spec, input.device())` — `input`'s shape, dtype, layout and memory config (R7), a new tensor (R3). `input` is never written.

## Tensors

### Input

| Property | `input` | `index` | `src` |
|----------|---------|---------|-------|
| Shape | `[H, W]` (Phase 0) | `[Hs, W]`, `Hs <= H` | `[Hs, W]` |
| Dtype | bfloat16 | INT32 | bfloat16 |
| Layout | TILE (32×32, four 16×16 faces) | ROW_MAJOR: one page per row, `W·4` bytes | TILE |
| Memory | DRAM interleaved | DRAM interleaved | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `input.shape` |
| Dtype | `input.dtype` |
| Layout | `input.layout` |
| Memory | `input.memory_config()` (R7) |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Phase 0 tensor coordinates: `dim` = -2 is the **line axis**; a line is one column of the tensor. In TILE layout a column lies in one column of tiles, and inside a tile in one half (faces 0+2 or 1+3). The **face-column** — 16 adjacent columns, all rows — is the smallest unit whose bytes are contiguous runs ≥ 512 B in every tile (one face = 16 rows × 16 elements), so it is the W work unit.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `width` (W, in face-columns of 16 elements; `tensor_width_faces = ceil(W/32)·(32/16)` incl. tile padding) | **independent** — a column's output depends only on that column of `input`, `index`, `src` | `block_width_faces` | the whole per-cluster assignment `max_cluster_faces = ceil(tensor_width_faces / clusters_engaged)`, clamped by `choose_blocking` (L1) | host `choose_blocking()` → one CT arg | **split across clusters**: `ttnn.split_work_to_cores(grid, tensor_width_faces, row_wise=True)`, contiguous ranges; inside a cluster not split (threads share the block) | knob-turn |
| `out_line` (H of `input`/output, in tile-rows `tensor_out_tile_rows = ceil(H/32)`) | **dependent** — any source row of a line may land on any output row, so each output row depends on the whole source line | `block_out_tile_rows` | `tensor_out_tile_rows` (whole extent, resident — the regime) | regime (`resident_line_block`) → derived on host | not split: whole extent per block, resident; splitting it is the deferred `banded_output_line` regime | scheme-change (band loop re-reading the source line) |
| `src_line` (Hs of `index`/`src`, in tile-rows `tensor_src_tile_rows = ceil(Hs/32)`) | **dependent** — the walk axis: ascending order decides which colliding source element stands within a thread; partial results of different source rows land in the same output tiles | `block_src_tile_rows` (source chunk staged at once) | `tensor_src_tile_rows` (whole extent) clamped by `choose_blocking` | host `choose_blocking()` → one CT arg | not split across clusters (would need a cross-cluster merge of output tiles: rejected regime); **split across the cluster's DM threads** (contiguous source-row ranges per chunk) | knob-turn (chunk size) |
| `batch` (leading dims for dim -2 at rank 3/4) | **independent** — images never exchange data along `dim` -2 | `block_images` | 1 (rank 2 has none; PHASE0 `rank` = [2]) | host | folds into the unit: units become (image, face-column) pairs over the same split | knob-turn |

Within a block every element move is one RISC-V load/store pair on a DM hart; the tile-level element offset is `((r/16)·2 + c/16)·256 + (r%16)·16 + c%16` elements for a 32×32 tile of 16×16 faces (the layout `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:36-56` encodes).

### Buffer-depth knobs

| Buffer | Depth knob | Phase 0 value | What the depth buys |
|--------|------------|---------------|---------------------|
| `scratch_out_column` | none (one resident block; depth is the regime) | 1 | — the block must be whole before its write-out |
| `scratch_src_rows` + `scratch_index_rows` | `src_stage_depth` | 1 | depth 2 would let chunk k+1's reads fly during chunk k's scatter (needs two user trids, 1 and 2); only matters when `block_src_tile_rows < tensor_src_tile_rows`, which no Phase 0 INPUTS shape triggers on this target (overlap lamp) |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DM harts per cluster (`QUASAR_USER_DM_CORES_PER_NODE` = 6, `kernel_spec.cpp:123-129`) | `SCATTER_THREADS` | `SCATTER_THREADS <= 6` asserted on host | program refused at validation |
| `sync_threads()` is a reusable barrier every thread of the kernel must call equally often (`kernel_thread_globals.h:266-287`) | thread participation | every thread runs the same block / chunk loop, including threads with zero source rows or zero transfers | hang (barrier never releases) |
| Per-cluster L1 (scratchpads) | `block_width_faces`, `block_src_tile_rows` | `choose_blocking()` (below) against `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES` | allocation failure at launch |
| Position bound (R6) | the row a position may address | `if ((uint32_t)p < H)` with `H` the real line length (not `tensor_out_tile_rows·32`); negative INT32 reads as a large unsigned and is skipped | write outside the slab / into tile padding |
| DRAM alignment 64 B for partial-page reads | index piece offset and size | index pieces start at `face_col·16·4 = face_col·64` B and are whole face-columns → always 64 B multiples while W is tile-aligned (Phase 0) | misaligned DRAM read (wrong bytes) |
| Face granularity of a half-owned tile | `block_width_faces` boundaries that split a tile | a tile with both face-columns in the block moves as one transfer of `tile_bytes`; a tile with one moves as two transfers of `face_bytes` (faces {0,2} or {1,3}) | two clusters writing the same bytes of one output tile |
| Scratchpad is not a NoC multicast target and carries no uint32/int32 format metadata | — | scratchpads declared without `data_format_metadata` | host validation refusal |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `resident_line_block` | **built** | `resident_line_fits`: `footprint(1, 1) <= budget`, i.e. `tensor_out_tile_rows·tile_bytes + tile_bytes + 32·16·4 <= budget` (on this target: `tensor_out_tile_rows <= 1,901`, H ≤ 60,832 rows; every INPUTS shape and the model's 2,766 rows) | `block_out_tile_rows (= all) × block_width_faces`, streaming `block_src_tile_rows` chunks | **minimum**: with DRAM as the boundary, `input`, `src`, `index` each cross once and the output once. No cross-core traffic. | per block: one slab read/write pass, 2 `sync_threads` + one L2 invalidate/flush of the slab, and the cluster's NoC issue setup — intended once per block; with Phase 0 knobs that is **once per cluster** (one block, one chunk). Per chunk: 2 extra `sync_threads` and one L2 invalidate/flush of the stage ranges. |
| `banded_output_line` | **deferred** — no INPUTS shape, no traced call and no model shape reaches it (it needs H > 60,832 rows here); `validate()` refuses those shapes by name | `!resident_line_fits` | `block_out_tile_rows < tensor_out_tile_rows` bands × `block_width_faces`, each band re-streaming the whole source line and writing only positions in `[band_lo, band_hi)` | `index` and `src` cross DRAM `ceil(tensor_out_tile_rows / block_out_tile_rows)` times; `input`, output once | fewer bands = fewer re-reads of the source line. Reachable: Phase 0's slab is already a block with the extent knob `block_out_tile_rows` (set to the whole extent); the delta is an outer band loop and a `[band_lo, band_hi)` filter in `scatter_chunk`. |
| `src_rows_across_clusters` | **rejected** — superseded by the width split, which reaches the minimum with no combine | — | source-row ranges per cluster, whole output line each | every participant holds a partial copy of the whole output column; merging them moves `(participants − 1) ×` output-column bytes over the NoC plus a per-element "written" mask, and order across clusters is lost | — (dead end: nothing in it survives into a good scheme) |
| `dram_element_writes` | **rejected** — superseded by `resident_line_block` | — | copy input→output, then one 2 B NoC write per source element into DRAM | 2 B writes are below the 64 B DRAM write granularity: read-modify-write per element, races between neighbours | — |
| `neo_select` | **rejected** — superseded by `resident_line_block` | — | NEO: `out = where(index_row_i == row_id, src_row_i, out)` per source row | O(Hs · tensor_out_tile_rows) tile operations against Hs element moves per column; int32 compare per tile | — |
| `row_lines` (`dim` -1) | **deferred** — outside PHASE0 (`dim` = [-2]); `validate()` refuses it | `dim == -1` | lines are rows: unit = face-row (16 rows) × whole W resident; index pieces are whole contiguous row segments | minimum (each tensor once) | same terms as `resident_line_block`. Reachable: the same kernel structure with rows and columns exchanged in the element offset and in the transfer shapes. |
| `cross_image_lines` (`dim` -3 / -4) | **deferred** — outside PHASE0; `validate()` refuses it | `dim <= -3` | lines run across images at a fixed (tile-row, face-column); unit = a face-column of one tile position, slab = that tile position across the `dim` extent | minimum | same terms. Reachable: the slab is a list of tiles addressed by page id; only the page-id formula and the unit enumeration change. |
| `row_major_lines` (ROW_MAJOR `input`/`src`) | **deferred** — outside PHASE0 (`layout` = [TILE]); `validate()` refuses it | `layout == ROW_MAJOR` | slab of row pieces: `H` rows × block columns; element offset `row·block_cols + c` | minimum, but one read per row per block; a block of one face-column is 32 B per row (< 64 B DRAM alignment) → RM needs `block_width_faces` ≥ 2 and 64 B-aligned starts (mechanism cap for that refinement) | larger blocks = longer row pieces per transaction |

Selection (host, exact): `dim == -2` and TILE → `resident_line_block` if `resident_line_fits`, else refuse. One program per call in every case.

### Traffic ranking

Bytes per call: `input` = `tensor_out_tile_rows · Wt · tile_bytes`, `src` = `tensor_src_tile_rows · Wt · tile_bytes`, `index` = `Hs · W · 4`, output = `input`. The index is the largest operand (INT32 is twice bfloat16) and the one with the most transactions: one piece per source row per block.

| Candidate split | DRAM crossings | Cross-core traffic | Verdict |
|-----------------|----------------|--------------------|---------|
| **width (face-columns) across clusters** | each tensor once (minimum) | none | **chosen** — cheapest; no combine |
| width in whole tile-columns across clusters | each tensor once | none | same bytes; under-fills the grid when `Wt < 32` (traced call: 16 tile-columns → 16 of 32 clusters idle). Face-columns double the unit count at the cost of two 512 B transfers instead of one 2 KB transfer at tile boundaries a block splits |
| out_line (dependent) across clusters | `index` + `src` once **per band** (every band must scan the whole source line), `input`/output once | none | more DRAM bytes; only useful when the line does not fit (deferred regime) |
| src_line (dependent) across clusters | each tensor once | `(participants−1)` × output column per line + masks | rejected |
| src_line across the **threads of one cluster** | each tensor once | none (shared L1; DM harts are cache-coherent) | **chosen inside a cluster** — the thread plan |

Occupancy is kept out of this ranking; it decides only the inner order (face-columns rather than tile-columns) between splits of equal bytes.

**Stall-shadow check.** Waits in the scheme and what fills them:

| Wait | Work independent of it | Decision |
|------|------------------------|----------|
| read barrier + `sync_threads` before a chunk's scatter (all of the slab and the chunk must have landed) | the slab's reads and chunk 0's `src`/`index` reads are independent of each other | **built**: a block issues all of them (slab + chunk 0) before one barrier per thread |
| chunk k's scatter while chunk k+1 could be read | chunk k+1's reads (disjoint stage) | deferred to `src_stage_depth = 2` (overlap lamp); no Phase 0 shape has >1 chunk on this target |
| block b's write-out drain before block b+1 reuses the slab | block b+1's `src`/`index` reads (disjoint buffers) | **built**: after block b's writes are issued, issue block b+1's chunk-0 stage reads, then `async_write_barrier<TXN_ID>(trid 0)` + `sync_threads`, then the slab reads |

No reorder changes a value (the op only moves bytes), so no precision gate applies.

### Block schedule

Per cluster, all `SCATTER_THREADS` threads run this schedule together (it is one kernel; thread-level dealing is inside each operation):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_line_block(block_idx);                 // input faces of the block -> scratch_out_column
    for (uint32_t chunk_idx = 0; chunk_idx < num_src_chunks; ++chunk_idx) {
        load_src_chunk(block_idx, chunk_idx);   // src faces + index pieces -> stages
        scatter_chunk(block_idx, chunk_idx);    // element moves stage -> slab, source rows dealt to threads
    }
    store_line_block(block_idx);                // slab faces -> output
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `load_line_block` | `block_out_tile_rows × block_tile_cols` tiles (only the block's faces) | slab, until `store_line_block` | tiles dealt round-robin over threads (`tile_k → thread k % T`); NoC issue with `TXN_ID` trid 0; **no barrier here** — its barrier is shared with chunk 0's |
| `load_src_chunk` | `block_src_tile_rows × block_tile_cols` `src` tiles (block's faces) + `min(32·block_src_tile_rows, rows left)` index pieces of `block_width_faces·64` B | stages, until the chunk's scatter ends | src tiles and index rows dealt round-robin over threads; **one** `async_read_barrier(trid 0)` per thread per chunk, then `sync_threads` |
| `scatter_chunk` | the chunk's real source rows × the block's real columns | slab (written), stages (read) | per chunk: every thread acquires `scoped_lock` on the stage ranges (and on the slab at chunk 0) → `sync_threads` → thread t moves its **contiguous** source-row range in ascending row order, columns innermost → releases the stage locks (and the slab lock at the last chunk) → `sync_threads`. Inside a thread no per-element init/sync. |
| `store_line_block` | same tiles/faces as `load_line_block` | — | tiles dealt round-robin; writes with trid 0; `async_write_barrier<TXN_ID>(trid 0)` + `sync_threads` before the slab is refilled (or the kernel ends; firmware waits for writes at return) |

Phase 0 on every INPUTS shape here: `num_blocks_this_core = 1`, `num_src_chunks = 1` — each cluster does one read pass, one scatter, one write pass.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Thread count (`SCATTER_THREADS` = 6) | 6 harts share one 128 KB L2; random row targets make every element store a line fill, so contention may flatten the gain | `SCATTER_THREADS` ∈ {1, 2, 4, 6} on the traced shape and the model shape (`ttsim-perf` model cycles) |
| Grid synchronization (dealing source rows to threads costs 2 `sync_threads` per chunk) | tiny blocks (traced call: 64 source rows × 16 columns per cluster ≈ 11 rows per thread) leave little work per barrier | threads own disjoint face-columns instead (no barriers, deterministic order) on wide shapes where each cluster has ≥ `SCATTER_THREADS` face-columns (`W ≥ 3072`); and fewer threads on the traced shape |
| Transfer granularity at split tiles | face-column ranges that split a tile turn one 2 KB transfer into two 512 B ones (`tile_reorder`: whole tiles move faster on DRAM-bound moves) | assign tile-column units (even-aligned face ranges) when `Wt >= clusters` |
| Overlap (`src_stage_depth` = 1) | with >1 source chunk the scatter waits on each chunk's read | `src_stage_depth = 2` with trids 1/2, measured with a forced `block_src_tile_rows` |
| NoC selection | one kernel reads and writes on one NoC (`noc_placement`: reads NoC0 / writes NoC1 measured faster) | issue the writes on the other NoC |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| `input` DRAM → `scratch_out_column` | bfloat16 tiles (faces) | `Noc::async_read<TXN_ID>` from `TensorAccessor(input)` page `tr·Wt + tc`, `offset_bytes` = face offset; trid 0 | page = tile; whole tile (2,048 B) when both face-columns are in the block, else two 512 B faces (offsets 0/1,024 or 512/1,536) |
| `src` DRAM → `scratch_src_rows` | bfloat16 tiles (faces) | same as above, `TensorAccessor(src)` | rows past `Hs` in the last src tile are tile padding, never read by the scatter |
| `index` DRAM → `scratch_index_rows` | INT32 row pieces | `Noc::async_read<TXN_ID>` from `TensorAccessor(index)` page = source row `i`, `offset_bytes = face_col0·64`, size `block_width_faces·64` | stage row stride = `block_width_faces·64` B (64 B multiple) |
| scatter (DM harts) | element moves, `uint16_t` (bf16) | RISC-V loads/stores inside `scoped_lock` windows | bitwise copy (R1); position bound `(uint32_t)p < H` (R6) |
| `scratch_out_column` → output DRAM | bfloat16 tiles (faces) | `Noc::async_write<TXN_ID>` to `TensorAccessor(output)` | two clusters may write different faces of one tile page: disjoint, 512 B-aligned byte ranges |

Ring protocols: **none** — the program has no dataflow buffer, so neither implicit nor explicit sync applies. Every NoC transfer carries `NocOptions::TXN_ID` with `{.trid = 0}` and every barrier waits on trid 0; the kernel's last statements are `noc_async_read_set_trid(0, noc.get_noc_id())` and `noc_async_write_set_trid(0, noc.get_noc_id())`.

### Placement / TARGET axes as later unlocks

| TARGET value not in PHASE0 | Class | What changes |
|----------------------------|-------|--------------|
| `memory_config` L1 | knob-turn | none in the kernels (accessors abstract placement); output already follows `input.spec`. L1-resident tensors share L1 with the scratchpads: the budget query covers the allocator's view |
| `sharding` | — | TARGET has `interleaved` only |
| `dtype` float32 | knob-turn | element type `uint32_t`, `tile_bytes` 4,096, `face_bytes` 1,024 (all derived from the tensor's page size); no conversion anywhere (R1) |
| `alignment` w_non_aligned | knob-turn | `tensor_width_faces` already counts tile padding (`ceil`); scatter skips columns `>= W`; index piece size clipped to the real columns of the block (page width `W·4`) |
| `rank` 3/4 with `dim` -2 | knob-turn | `batch` axis: units become (image, face-column); page ids gain `image·tensor_out_tile_rows·Wt` (tiles) and `image·Hs` (index rows) |
| `dim` -1, -3, -4 | scheme variants | deferred regime rows `row_lines`, `cross_image_lines` |
| `layout` ROW_MAJOR | scheme variant | deferred regime row `row_major_lines` |

No physical shard is in TARGET, so no regime consumes a resident shard.

### Structural impossibilities

None to add: the spec's `INVALID = []` matches this design.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one face-column (16 columns × the whole line); a block = `block_width_faces` contiguous face-columns × all `tensor_out_tile_rows` |
| Grid | `device.compute_with_storage_grid_size()`; `ttnn.split_work_to_cores(grid, tensor_width_faces, row_wise=True)` → per cluster `(face_col_start, num_face_cols)` RT args; clusters engaged = `min(grid clusters, tensor_width_faces)` |
| Per-core work | `num_face_cols` face-columns cut into `num_blocks_this_core = ceil(num_face_cols / block_width_faces)` blocks (last block ragged: same operation, shorter `width` extent) |
| Remainder | from `split_work_to_cores` (two groups); `tensor_width_faces = ceil(W/32)·2`, `tensor_out_tile_rows = ceil(H/32)`, `tensor_src_tile_rows = ceil(Hs/32)` — `ceil`, per image (rank ≥ 3 later) |
| Worked (this target) | traced `[70,512]`, Hs 64: 32 face-columns → 32 clusters × 1; block = 3 out tile-rows × 1 face-column, 2 src tile-rows; 64 source rows → threads get 11,11,11,11,10,10. Model `[2766,2560]`, Hs 2752: 160 face-columns → 5 per cluster; block = 87 × 5 face-columns (≤ 3 tile-columns), 86 src tile-rows; ~459 source rows per thread. `[64,8192]`: 512 → 16 per cluster. `[1024,64]` Hs 512: 4 face-columns → 4 clusters (the grid-sync lamp's other end) |
| Bound | **data movement on the DM harts, per element**: each source element costs an index load, a bounds compare, offset arithmetic, a 2 B load and a 2 B store on a DM hart, against 8 B of DRAM traffic; NEOs cannot express a data-dependent scatter. So threads of the DM kernel speed it; NEOs do not. |
| Thread plan | one DM kernel `scatter`, `num_threads = SCATTER_THREADS = 6` (all user DM harts), **no compute kernel**: in (R, C, W) terms (6, 0, 6) where the same 6 harts read, scatter and write. Host constant `SCATTER_THREADS`, passed as a CT arg; kernels size every dealing loop from it and `get_my_thread_id()` (not `get_num_threads()`), so a missing thread leaves visible work undone. |
| Unit dealt to threads | transfers: tiles / index rows round-robin (`k % T`); scatter: **contiguous source-row ranges per chunk** (`q = n / T`, `rem = n % T`, thread t gets `q + (t < rem)` rows from `chunk_row0 + t·q + min(t, rem)`). No ring → no padded rounds; a thread with zero units still calls every `sync_threads`. |
| Collisions (R2) | within a thread, ascending source rows → the later one stands (the reference's choice); between threads of a cluster, whichever store lands last stands — allowed by R2, whole 16-bit stores so never a mix |
| Tests the plan needs | (1) **threads ran**: a probe `probe_skip_thread = t` (default `T` = none) makes thread t skip its scatter rows only; for each t the output must equal the reference except at the targets of exactly the source rows the dealing formula gives thread t (which keep `input`'s value) — run on a distinct-index cell with ≥ 6 source rows. (2) **one device session**: the op after another program, and the op followed by `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)`. (3) **regime / knob pins**: forced `block_width_faces = 1` (several blocks per cluster) and forced `block_src_tile_rows = 1` (several chunks) on `[1024,64]`/Hs 512 and `[128,512]`/Hs 17 — no INPUTS shape reaches the clamps on this target. (4) host-only: `choose_blocking` with a tiny budget refuses `resident_line_fits = False` by name. |

### `choose_blocking` (host, single source of every blocking knob)

```
tile_bytes, face_bytes   = input page size, tile_bytes / 4
tc(wf)                   = wf // 2 + 1                     # tile-columns a wf-face range can touch
footprint(wf, sr)        = tensor_out_tile_rows·tc(wf)·tile_bytes      # scratch_out_column
                         + sr·tc(wf)·tile_bytes                        # scratch_src_rows
                         + sr·32·wf·16·4                               # scratch_index_rows
budget                   = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES
wf0, sr0                 = max_cluster_faces, max(1, tensor_src_tile_rows)
if footprint(wf0, sr0) <= budget:          (wf, sr) = (wf0, sr0)        # Phase 0 on every INPUTS shape
elif footprint(wf0, 1) <= budget:          sr = largest sr <= sr0 that fits;  wf = wf0
elif footprint(1, 1) <= budget:            wf = largest wf that fits with sr = 1; then sr = largest that fits
else:                                      refuse (banded_output_line deferred)
```

Shrinking `sr` or `wf` adds chunks or blocks but never DRAM bytes (each tensor still crosses once), so the order is: source chunk first (stages only), width second (slab). Every count (scratchpad sizes, loop trips, dealing) derives from `(wf, sr, SCATTER_THREADS)` and the shape; nothing restates them.

## Circular Buffers

This program has **no dataflow buffers**: the three buffers are `ScratchpadSpec`s bound to the one DM kernel (`scratchpad_bindings`), shared by its threads. Sizes are per cluster.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `scratch_out_column` | scratchpad `scatter_out_column` | `tile_bytes` | `block_out_tile_rows · (block_width_faces // 2 + 1)` | spans `out_line` (whole, the regime) and `width`; not `src_line` | none declared (bf16 payload, DM-only) | `scatter` kernel (NoC reads of `input`, then its threads' stores) | `scatter` kernel (NoC writes to output) | `scoped_lock` + `sync_threads` (protocol below) | one block |
| `scratch_src_rows` | scratchpad `scatter_src_rows` | `tile_bytes` | `block_src_tile_rows · (block_width_faces // 2 + 1) · src_stage_depth` | spans `src_line` chunk and `width` | none (bf16 payload) | `scatter` (NoC reads of `src`) | `scatter` (thread loads) | same | one chunk |
| `scratch_index_rows` | scratchpad `scatter_index_rows` | `block_width_faces · 64` B (one source row's piece) | `block_src_tile_rows · 32 · src_stage_depth` | spans `src_line` chunk and `width` | none (INT32 payload; uint32/int32 metadata not allowed) | `scatter` (NoC reads of `index`) | `scatter` (thread loads) | same | one chunk |

Each scratchpad is one kernel's (one producer kernel, one consumer kernel: `scatter`); the threads' accesses are ordered by the protocol below, not by credits.

**Cache / thread protocol (correctness, Quasar DM caches)** — per chunk, every thread, in this order:

| Step | Every thread does | Why |
|------|-------------------|-----|
| 1 | issue its dealt reads (trid 0); `noc.async_read_barrier(trid 0)` | its bytes are in L1 |
| 2 | `sync_threads()` | all threads' bytes are in L1 |
| 3 | acquire `scoped_lock` on `scratch_src_rows` and `scratch_index_rows` (whole chunk ranges) and, at chunk 0, on the block's `scratch_out_column` range | L2 invalidate: no stale cached copy of NoC-written bytes |
| 4 | `sync_threads()` | no thread stores into the slab while another is still invalidating it (an invalidate discards dirty lines) |
| 5 | scatter its source rows | stores land in the coherent DM caches |
| 6 | release the stage locks; at the last chunk release the slab lock | L2 flush: the slab reaches L1 before the NoC reads it |
| 7 | `sync_threads()` | every thread's flush done before any write-out or stage refill |

Then `store_line_block`: dealt writes (trid 0) → `noc.async_write_barrier<NocOptions::TXN_ID>({.trid = 0})` → `sync_threads()` before the slab is refilled. (`async_writes_flushed<TXN_ID>` is not used: it static-asserts on `DM_DYNAMIC_NOC`, `noc.h:788`.)

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input (semantic name, pages, state) | Output (semantic name, pages) | State after |
|---|-----------------|-------------|---------|-------------------------------------|-------------------------------|-------------|
| 1 | `load_line_block` | `block_out_tile_rows × tc` tiles, block's faces only | no (raw NoC; see API Mapping) | `input` DRAM pages | `scratch_out_column`, all pages of the block | in flight until chunk 0's barrier |
| 2 | `load_src_chunk` | `block_src_tile_rows × tc` src tiles + `≤ 32·block_src_tile_rows` index pieces | no | `src`, `index` DRAM pages | `scratch_src_rows`, `scratch_index_rows` | landed + visible after protocol steps 1–4 |
| 3 | `scatter_chunk` | chunk's real source rows × block's real columns | no (custom block operation, per-thread contiguous rows) | `scratch_src_rows`, `scratch_index_rows` (read), `scratch_out_column` (read-modify) | `scratch_out_column` | stages free after step 7; slab flushed at the last chunk |
| 4 | `store_line_block` | as 1 | no | `scratch_out_column` | output DRAM pages | slab free after flush + sync |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input | Output | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|-------|--------|------------------------------|
| host program | host | `ttnn.experimental.ProgramSpec`, `KernelSpec(num_threads=SCATTER_THREADS, scratchpad_bindings=[...], tensor_bindings=[input, src, index, output])`, `ScratchpadSpec(unique_id, size_per_node)`, `WorkUnitSpec(target_nodes=core_set)`, `ProgramRunArgs`, `ttnn.generic_op` | `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py:72-158`; `ttnn/cpp/ttnn-nanobind/program_spec.cpp:364-398, 704-718` | DM config: `E.create_reader_datamovement_config()`; no `disable_dfb_implicit_sync_for` (no rings) | — | — | `SCATTER_THREADS`, scratchpad sizes from `choose_blocking` |
| `load_line_block`, `load_src_chunk` | raw_api | `Noc::async_read<NocOptions::TXN_ID>(TensorAccessor, Scratchpad, size, {.page_id, .offset_bytes}, {.offset_bytes}, {.trid = 0})`; `async_read_barrier<TXN_ID>` | `tt_metal/hw/inc/api/dataflow/noc.h:246-284, 739-745`; `tt_metal/hw/inc/api/tensor/noc_traits.h:134-143`; `tt_metal/hw/inc/internal/local_mem/noc_traits.h:73-90` | src `offset_bytes` = face offset / index piece offset | DRAM | scratchpads | size = `tile_bytes` or `face_bytes` / `block_width_faces·64` |
| `scatter_chunk` | raw_api | `Scratchpad<T>` + `scoped_lock(offset, n)`; `sync_threads()`; `get_my_thread_id()` | `tt_metal/hw/inc/api/scratchpad.h:42-150`; `tt_metal/hw/inc/internal/scoped_lock_cache_ops.h:14-30`; `tt_metal/hw/inc/api/kernel_thread_globals.h:155, 281-287` | `T` = `uint16_t` (bf16) | stages | slab | chunk rows / thread from `block_src_tile_rows`, `SCATTER_THREADS` |
| `store_line_block` | raw_api | `Noc::async_write<NocOptions::TXN_ID>(Scratchpad, TensorAccessor, size, {.offset_bytes}, {.page_id, .offset_bytes}, {.trid = 0})`; `async_write_barrier<TXN_ID>` | `noc.h:404-470, 760-766` | — | slab | output DRAM | as load |
| kernel end | raw_api | `noc_async_read_set_trid(0, noc.get_noc_id())`, `noc_async_write_set_trid(0, noc.get_noc_id())` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2558, 2579` | — | — | — | — |

**Helpers considered and rejected** (every block operation is raw_api: the op has no compute phase and the dataflow helpers address whole tiles or DFB-shaped tiles):

| Candidate | File:Line | Why it cannot express this |
|-----------|-----------|----------------------------|
| `compute_kernel_lib::copy` / `eltwise_chain` (`CopyTile`) | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:109-113` | a NEO helper over DFB tiles in tile order: the destination of an element is its own position; the scatter's destination row is data (`index`), per element |
| `dataflow_kernel_lib::copy_tile_column` / `TileColumnLayout` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:36-56, 91-103` | copies a whole column with `dst row = src row`, and its layout is a template on a **DFB id** (`get_tile_r_dim<dfb_id>`, :38-39); the slab is a scratchpad and each element's destination row differs. Its offset formula is the one this design uses |
| `dataflow_kernel_lib::local_addr` / `set_read_trid` (local copies) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:43-60` | L1→L1 NoC copies of contiguous ranges; a 2 B element move by NoC per element is far costlier than one store |
| `tilize` / `untilize` helpers | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:206`, `untilize_helpers.hpp:64-72` | layout conversion on NEOs; no layout changes here |
| `McastArgs` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp:18-117` | no operand is shared between clusters under the width split |

What would close the gap: a dataflow `scatter_column_elements<T>(src_stage, index_stage, slab, rows, cols, bound)` block helper over scratchpad tiles; the implementer may write it as a kernel-local function.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Stale or lost bytes through the DM caches | DM stores sit in a write-back L1 D$/L2; NoC reads/writes go to L1 directly. Reading `input` tiles through a stale line, or writing out a slab whose stores are still in cache, gives silently wrong output | protocol steps 3/6 (`scoped_lock` acquire after landing, release before write-out) over exactly the NoC-touched ranges; if a stale-line symptom still appears, add `flush_l1_dcache(0)` (write back + invalidate this hart's D$, `risc_common.h:229-236`) after acquire — never `invalidate_l1_dcache(0)`, which discards the stack's dirty lines |
| One thread invalidating while another stores | an L2 invalidate discards dirty lines of other harts too | `sync_threads` between acquire and the first store (step 4) |
| Hang on `sync_threads` | barrier counts every thread of the kernel | every thread runs every block/chunk iteration, including those with no rows/transfers |
| Two clusters writing one output tile | face-column ranges split tiles between clusters | only the block's own faces are read and written (2 × 512 B for a half-owned tile); never a whole-tile write of a half-owned tile |
| Positions outside the line (R6) | INT32 data used as an address | `(uint32_t)p < H` with the real H; padding rows never receive a source element |
| R2 with all-zero index (traced call) | 64 source rows to row 0, dealt to 6 threads | any thread's element may stand; 16-bit stores are whole. The `index_all_zeros` loose case covers it |
| Transaction tag left for the next program | — | every transfer `TXN_ID` trid 0; trid reset as the kernel's last statements; session tests (Work Distribution, tests 2) |
| trid counters may be per NIU, shared by the cluster's harts | a thread's trid-0 barrier may also wait for other threads' reads | harmless for correctness (a `sync_threads` follows every barrier); recorded so nobody "fixes" it with untagged barriers |
| Index page offset | index is row-major INT32, one page per source row | `offset_bytes = face_col0·64`, size `block_width_faces·64`, page id = source row (rank 2) |
| Knob clamps never exercised by INPUTS on this target | 4 MB L1 holds every INPUTS shape at full extents | forced-knob tests (Work Distribution, tests 3) |
