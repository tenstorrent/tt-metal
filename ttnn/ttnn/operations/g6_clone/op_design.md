# Operation Design: g6_clone

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement |
| Goal | Copy one DRAM-interleaved device tensor into a newly allocated DRAM-interleaved tensor with the same `TensorSpec`, so that every logical element is bit-identical. The copy is one device program of NoC byte moves (reader → one CB → writer). It has no compute kernel. |
| Math | `bits(out[i]) = bits(x[i])` for every logical index `i`; `spec(out) = spec(x)`; `buffer(out)` shares no storage with `buffer(x)`; `x` unchanged |
| References | `docs/explorations/g6_clone/{DOSSIER.md, math.md, tensix.md, risks.md, lessons.md}`; `eval/golden_tests/g6_clone/{feature_spec.py, helpers.py, test_regression.py}`; `.claude/references/{blocking-model.md, l1-footprint-discipline.md, ttnn-cb-memory-fundamentals.md}`; `l1_ledger.md` (this directory) |

### Catalog entries that set knobs (`ttnn/ttnn/operations/examples/`)

| Entry | Knob it set | What it contributed |
|-------|-------------|---------------------|
| `width_split` | core-assignment | Split the flat page axis, never tile rows. A one-tile-row tensor (p4, `[1,1,1,62080]`) split by rows runs on one core. |
| `noc_placement` | core order, NoC per role | `split_work_to_cores(..., row_wise=True)`. Reads on NoC0 (reader, NCRISC) and writes on NoC1 (writer, BRISC), which are the `ReaderConfigDescriptor` / `WriterConfigDescriptor` defaults (`tt_metal/impl/kernels/kernel_types.cpp` reader `RISCV_1`/`NOC_0`, writer `RISCV_0`/`NOC_1`). |
| `double_buffer` | `block_pages`, `cb_depth_blocks` | Issue a whole block of reads, then one barrier. Never use read-one/barrier. Depth 2 buys overlap only when a core has more than one block. No gain once the copy is DRAM-bound with many cores. |
| `trid_double_issue` | overlap lamp | Tag reads with transaction ids (trid) and barrier on one tagged batch at a time; this is the alternative the overlap lamp measures. A large one-block loop is within 3% of the best. |
| `tile_reorder` | `in_page` extent | Move whole pages and batch the barriers. Sub-tile (face) transfers are rejected for aligned shapes. |
| `bank_stagger` | bank-stagger lamp | Rotate each core's issue order by core index mod bank count. |
| `split_reader` | split-reader lamp | Helps only if one data-movement RISC is issue-bound. |
| `read_coalescing` | `bank_affine_runs` regime | A single wide read cannot span interleaved pages taken in page order. A run is contiguous only inside one bank. Built in Refinement 2 with row placement next to each bank's DRAM endpoint (tech report `Saturating_DRAM_bandwidth`). |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | device tensor, rank 1–4, DRAM interleaved. TARGET dtypes: float32, bfloat16, int32, uint32. TARGET layouts: TILE, ROW_MAJOR. SUPPORTED since Refinement 1: all of TARGET (Phase 0: float32 + TILE) | — | host |

The op has no other parameter: no `memory_config`, no `dtype`, no compute config. Call form: `g6_clone(x)`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d_0, ..., d_{R-1})`, R in 1..4. H and W need not be multiples of 32. |
| Dtype | TARGET `float32, bfloat16, int32, uint32`, all SUPPORTED since Refinement 1 (Phase 0: `float32`). |
| Layout | TARGET `TILE_LAYOUT, ROW_MAJOR_LAYOUT`, both SUPPORTED since Refinement 1 (Phase 0: `TILE_LAYOUT`). |
| Memory | DRAM interleaved (ValueError otherwise) |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` (logical), and the same padded shape |
| Dtype | `input_tensor.dtype` |
| Layout | `input_tensor.layout` |
| Memory | `input_tensor.memory_config()` (DRAM interleaved), in a newly allocated buffer: `ttnn.allocate_tensor_on_device(input_tensor.spec, input_tensor.device())` |

Allocating from `input_tensor.spec` gives `out` the same padded shape as `x`. It therefore has the same page count, page size and page order, and page `p` of `x` is page `p` of `out`. The host asserts `out.buffer_num_pages() == x.buffer_num_pages()` and `out.buffer_aligned_page_size() == x.buffer_aligned_page_size()`.

## Blocking Model

### Index space

A copy sees its index space as **pages**: one 32×32 tile (TILE) or one row of the last dimension (ROW_MAJOR).

- TILE page id: `p = (n · h_tiles + ht) · w_tiles + wt`, with `n` running over the folded leading dims, `h_tiles = ceil(H/32)` and `w_tiles = ceil(W/32)`. Each image is tile-padded on its own; the padded spec gives these counts.
- ROW_MAJOR page id: `p` is the row index over `prod(d_0..d_{R-2})`.

All page-level tensor axes fold into one flat `page` axis. The fold is legal for two reasons:

1. The op treats every page identically.
2. Page ids are contiguous and in the same order in `x` and `out`, because the specs are identical.

So any run of consecutive page ids is a legal block, even when it crosses image or tile-row boundaries. The block is 2-D: `page × in_page`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Buffer depth | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|--------------|
| `page`: flat page id, extent `tensor_pages = x.buffer_num_pages()` (1–3072 on INPUTS) | independent: out page `p` reads x page `p` only, and no page reads another | `block_pages` | `min(core_pages_max, block_pages_l1_cap)`. This is the core's whole assignment on every INPUTS shape (1–28 pages at G = 110). It is 1 only where the core's whole assignment is one page. | host constant, passed as a CT arg to reader and writer. CB capacity, push/pop quanta and `num_blocks_this_core` are all derived from it. | contiguous page ranges over `num_cores = min(tensor_pages, G)` cores, with G the full compute grid. Assigned by `split_work_to_cores(grid, tensor_pages, row_wise=True)`. Grid comes from `device.compute_with_storage_grid_size()`, never a constant. | `cb_depth_blocks` (see next table) | knob-turn: `block_pages`, grid, depth |
| `lead` (TILE: `prod(d_0..d_{R-3})`, 1–16·12 on INPUTS) | independent: no result spans it | folded into `block_pages`; no knob of its own | the block covers whatever `lead` range its page run spans | — (through `block_pages`) | not assigned on its own. A split on this axis caps the core count at `lead` (=1 on p3, p4, `[1,1]`). | streams through `cb_page_staging` with `page` | knob-turn |
| `h_tiles` (`ceil(H/32)` per image) | independent: no result spans it | folded into `block_pages` | as above; a block may start and end mid-image | — | not assigned on its own. A tile-row split caps the core count at `lead·h_tiles`, which is 1 on p4 and `[1,1,1,62080]` (R4). | streams with `page` | knob-turn |
| `w_tiles` (`ceil(W/32)`) | independent: no result spans it | folded into `block_pages` | as above; a block may start and end mid-tile-row | — | not assigned on its own. A tile-column split caps the core count at 4 on p1, p2 and Q13. | streams with `page` | knob-turn |
| `rows` (ROW_MAJOR: `prod(d_0..d_{R-2})`, 1–48 on INPUTS) | independent: no result spans it | identical to `page` for ROW_MAJOR (one row = one page) | as `page` | — | through `page` | streams with `page` | knob-turn |
| `in_page`: bytes inside one page (a tile of four 16×16 faces, or one row plus its DRAM alignment gap); extent `aligned_page_bytes` | independent: no element reads another | `aligned_page_bytes`. This is a mechanism value, not a tunable knob: the buffer fixes it. | whole page (4096 B for a float32 tile) | host: `x.buffer_aligned_page_size()`, used as the CB page size. Device: `TensorAccessorArgs` AlignedPageSize. | not split across cores. A page is the DRAM interleave unit (one bank per page). Cutting it multiplies transactions, and on aligned shapes it moves no fewer bytes (catalog `tile_reorder`). | spans the CB page | knob-turn into regime `logical_rows_partial_tile` (built in Refinement 1) |

**The only intermediate data is the staged block in `cb_page_staging`.** Its axes are the axes of `x` (`page`, `in_page`), and its rows above apply unchanged. There is no coordinating core and no collecting core. No stage computes on one core while the others wait: every used core runs the same reader/writer pair on its own page range.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_page_staging` | `cb_depth_blocks = min(CB_DEPTH_KNOB, num_blocks_max)`, with `CB_DEPTH_KNOB = 2` (host constant) and `num_blocks_max = ceil(core_pages_max / block_pages)` | 1 on every INPUTS shape (one block per core). 2 when the L1 cap splits a core's assignment. | With ≥2 blocks per core, the reader fills block k+1 on NoC0 while the writer drains block k on NoC1. With one block per core a second slot would have nothing to hold. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| A TensorAccessor page transfer moves `get_aligned_page_size()` bytes (`tt_metal/hw/inc/api/tensor/tensor_accessor.h:157`; `noc_async_read_page` `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1243-1266`). That is the page plus its DRAM alignment gap (Blackhole DRAM alignment 64 B, `tt_metal/hw/inc/internal/tt-1xx/blackhole/core_config.h:42`). | `in_page`: the CB page size | CB `page_size := x.buffer_aligned_page_size()`, and the host asserts the output's equals it | A CB page sized to the raw page (4 B for `[1]` int32) is overrun by the 64 B transfer. Neighbouring slots are corrupted (ROW_MAJOR refinement). |
| Model L1 bound: 1347776 B per core for the op (BB-05) | `block_pages` | `block_pages_l1_cap = floor(1347776 / (CB_DEPTH_KNOB · aligned_page_bytes))`: 164 for float32 tiles, 329 for bfloat16 tiles, 131 for 5120 B rows. If the cap is < 1 (a page > 673888 B, which no TARGET shape has), the host raises RuntimeError. | The CB exceeds the L1 the model leaves free, and the launch fails only inside the model. |
| Core assignment: a block never exceeds the core's pages | `block_pages` | `block_pages ≤ core_pages_max`. The ragged last block reads and writes only `last_block_extent` pages. | Page ids past the core's range are copied twice, and on the last core they address past the buffer end. |
| CB ring wrap: the pointers reset only on an exact `fifo_limit` hit (`ttnn-cb-memory-fundamentals.md`, CB Ring-Wrap Invariant) | push/pop quantum | Every `cb_push_back` and `cb_pop_front` uses the nominal `block_pages`, and capacity is `cb_depth_blocks · block_pages`, so every cycle closes. A ragged block narrows only the NoC transfers. | The pointer walks past `fifo_limit`, causing silent corruption in release builds. |
| NoC DRAM read alignment of 64 B (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`) | CB slot addresses | No action needed. CBs are allocated at DRAM alignment (`tt_metal/impl/program/program.cpp:2509`), and the slot stride `aligned_page_bytes` is a multiple of 64. | — |
| `NOC_MAX_BURST_SIZE` = 16384 B (`noc_parameters.h:289-290`) | one page transfer | No action needed. `noc_async_read_page` / `noc_async_write_page` pass `max_page_size = NOC_MAX_BURST_SIZE + 1` and split into packets (`dataflow_api.h:1266`). The largest page on INPUTS is 5120 B. | — |

### Regimes

**Named minimum:** each stored byte of `x` crosses DRAM once (read), each stored byte of `out` crosses DRAM once (write), and there is zero cross-core traffic. Stored bytes include tile padding and the ROW_MAJOR alignment gap.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `flat_page_range` | **built** | Every input that passes `validate()` (TILE or ROW_MAJOR, rank 1–4, any alignment, DRAM interleaved, `block_pages_l1_cap ≥ 1`) and is selected by neither `logical_rows_partial_tile` nor `bank_affine_runs`: since Refinement 2, full-page inputs too small for every bank group core to get a page (fewer than 48 pages on the 11×10 grid). Selection: `partial_tile_walk(tile_geometry(x), aligned_page_bytes) is None` and the bank-affine gate fails. | `block_pages` consecutive page ids × `aligned_page_bytes` | **At the minimum** on stored bytes: x read once, out written once (p1: 12582912 B each way), no cross-core traffic. It exceeds the *logical*-byte minimum only on partial-tile shapes, where the tile padding moves too (see `logical_rows_partial_tile`). | Per block, the reader pays 1 `cb_reserve_back`, 1 `noc_async_read_barrier` and 1 `cb_push_back`. The barrier drains every outstanding read, so the reader's NoC queue is empty until the next issue. Per block, the writer pays 1 `cb_wait_front`, 1 `noc_async_writes_flushed` and 1 `cb_pop_front`. A block that spans the core's whole assignment keeps all of its reads in flight toward DRAM at once. **Intended frequency:** once per block, which is once per core per call on every INPUTS shape. TensorAccessor construction happens once per kernel. There are no LLK inits (no compute kernel), no multicast and no cross-core barrier. |
| `logical_rows_partial_tile` | **built** (Refinement 1). Pays on `[1,1,1,62080]` bf16 (Q8): 19.44 → 7.06 µs, and `[1,5120]` bf16 (F13): 2.39 → 1.59 µs. Same split, CB page and block loop as `flat_page_range`; only the per-page transfer inside `load_block` / `store_block` changes, under the CT flag `logical_rows`. | TILE with plain tile padding of the last two dims, and the image has a partial tile row or column (`H mod 32 ≠ 0` or `W mod 32 ≠ 0`), and at least one of the four page kinds moves fewer bytes than the whole page | Same `block_pages`. Each page is one of four kinds, `(tile row is the image's last) | (tile column is the image's last) << 1`, tracked by `PartialTileWalk` (`kernels/g6_clone_page_pieces.hpp`) without dividing page ids. A kind moves at most two byte ranges: the face rows holding logical rows (row `r` in faces `{0,1}` for `r < 16`, else `{2,3}`; faces 1 and 3 dropped when the last tile column holds ≤ 16 columns), each rounded up to the 64 B DRAM alignment and landed at its own in-page offset in the CB slot. Ranges at most `PIECE_MERGE_GAP_BYTES` (448) apart move as one span through the gap. Full pages (kind 0) are one whole-page range. | **Below** the stored-byte minimum. `[1,1,1,62080]` bf16: 3973120 → 1117440 B each way (one 576 B span per page; logical 124160 B). `[1,5120]` bf16: 327680 → 92160 B. `[1,1]` f32: 4096 → 64 B. One or two NoC requests per page. | Same as `flat_page_range`. Small ranges are read-request-bound, not byte-bound: on Q8, two 64 B reads per page run 9.35 µs, one 576 B span 7.06 µs and two 32 B reads 9.74 µs. Merging loses once the gap reaches 512 B (17×16 bf16 tiles +3%; one-row f32 tiles, 960 B gap, +10%; 17×16 f32 tiles, 1024 B gap, +53%). |
| `bank_affine_runs` | **built** (Refinement 2): the deferral premise ("changes only the transaction count; pays only when a DM RISC is issue-bound") is falsified. p1 is not issue-bound (44–110 cores flat at 63.8–64.4 µs) yet sat at 77% of DRAM peak; the bound was route overlap toward the DRAM endpoints, which bank affinity plus row placement removes. Blackhole p150b, one fresh profiled run each: p1 63.6 → **51.4 µs** (490 GB/s, 96% of 512 GB/s; at the 490 GB/s floor of the adjacent-core DRAM benchmark), p2 32.5 → 25.5, p3 25.7 → 21.5, p4 19.4 → 16.6, Q13 4.61 → 3.93, Q3 ROW_MAJOR 2.05 → 1.68 µs. Placement `dram_rows` (`BANK_AFFINE_PLACEMENT`): a bank's cores are the cores of its anchor row (the optimal worker of `BANK_AFFINE_ENDPOINT_NOC`) between its DRAM column and the next, 44 cores on the 11×10 grid. Both kernels address that one endpoint of the bank (`bank_endpoint_noc` CT arg, `TensorAccessor::get_noc_addr(page, 0, noc)`): Blackhole's NoC0 and NoC1 default endpoints sit on different rows, and writes to the NoC1 default ran 58.3 µs. Read data flows east along the row on NoC0 and write data west back along it on NoC1, so no two banks share a link. Measured alternatives (each kernel on its own NoC's endpoints): every grid core in row bands (`dram_side`) 72.6 µs, `interleave` 69.2 µs, `dram_rows` 58.2 µs; 2–8 blocks per core with depth 2 lose 4–24% (a bank's reads and writes then interleave in DRAM). | Not logical_rows (consecutive pages), more than one DRAM bank, and every core of every bank's group gets ≥ `BANK_AFFINE_MIN_CORE_PAGES` pages (48+ pages on this grid); else `flat_page_range`. `aligned_page_bytes` is the bank page stride for every interleaved buffer, so ROW_MAJOR qualifies too (the alignment gap moves with the run). | A run of k pages of one bank (ids `b, b+NB, b+2NB, …`, contiguous in bank `b`; `TensorAccessor::num_contiguous_pages` asserts it), moved as one transfer of `k · aligned_page_bytes` each way | At the minimum on stored bytes; one NoC command per block instead of per page | Same per-block terms as `flat_page_range`; one block per core on every INPUTS shape (up to 77 f32 pages, 315 KB, on p1). |
| `tile_row_split` | **rejected**, superseded by `flat_page_range` on every shape | — | `lead·h_tiles` tile rows per core | Same bytes, but the core count is capped at the tile-row count: 1 core on p4 and on `[1,1,1,62080]` (R4, catalog `width_split`) | — |
| `tile_col_split` | **rejected**, superseded by `flat_page_range` | — | tile columns per core | Same bytes, but the core count is capped at `w_tiles` (4 on p1, p2 and Q13) | — |
| `sub_page_split` (one page cut across cores) | **rejected**, superseded by `flat_page_range`. The shapes with fewer pages than cores hold ≤ 48 pages and ≤ 245760 B, and their time is per-call overhead. | — | a part of one page | Same stored bytes, but in more and smaller transactions (catalog `tile_reorder`: whole pages are at least as fast) | — |
| `single_risc_read_then_write` | **rejected**, superseded by the reader/writer pair. Reads and writes would serialize on one RISC and one NoC (catalog `noc_placement`: reads on NoC0, writes on NoC1). | — | as `flat_page_range` | At the minimum | — |
| `engine_copy` (unpack → DEST → pack, `compute_kernel_lib::copy`) | **rejected**, forbidden by spec rule 1 and risk R1. A 16-bit DEST rounds float32, and a matrix-engine operand truncates it. Superseded by NoC byte moves with no compute kernel. | — | tiles | At the minimum, but not a copy | — |
| `staged_dram_intermediate` (a second device tensor or a second pass) | **rejected**, forbidden by spec rule 4. It doubles DRAM crossings. Superseded by `flat_page_range`. | — | — | 2× the minimum | — |

**Regime-pinned tests:** `test_g6_clone_layout_matrix.py::test_g6_clone_plan_regime` pins which regime each input class runs (the merged pieces on Q8 / F13, `bank_affine_runs` on p1 / p4 / Q3 and its `flat_page_range` fallback below the gate); `test_g6_clone_plan_bank_affine_rows` pins the `dram_rows` placement and endpoint; `test_g6_clone_cache_key_partial_vs_full` pins that partial and full tiles never share a cached descriptor; `test_g6_clone.py::test_g6_clone_multi_block_ring` runs the depth-2 ring in both full-page regimes.

### Traffic ranking

The ranking is qualitative. Crossings are counted per tensor per tier.

| Candidate split | x DRAM reads | out DRAM writes | Cross-core | Fan-out | Verdict |
|-----------------|--------------|-----------------|------------|---------|---------|
| flat contiguous page ranges | 1× stored bytes | 1× stored bytes | 0 | none | **chosen**: at the minimum. It is also the only split that is not core-capped by a tensor axis. |
| bank-affine page sets | 1× | 1× | 0 | none | **chosen** on full-page shapes with ≥ 1 page per group core (Refinement 2): ties on bytes, wins on DRAM-route locality (p1 63.6 → 51.4 µs) |
| tile-row split | 1× | 1× | 0 | none | ties on bytes; rejected on work assignment (1 core on p4) |
| tile-column split | 1× | 1× | 0 | none | ties on bytes; rejected on work assignment (4 cores on p1) |
| sub-page split across cores | 1× | 1× | 0 | none | ties on bytes, needs more transactions; rejected |
| logical rows only (partial tiles) | logical face rows at 64 B granularity, merged across ≤ 448 B gaps (0.28 of stored on `[1,1,1,62080]` bf16) | same | 0 | none | **chosen** on partial-tile TILE shapes (Refinement 1) |
| split a dependent axis + combine | — | — | — | — | does not exist: no axis is dependent, so there is no combine to price |

- **Operand-reuse check:** `x` is the only input, and it varies along every axis. Under the chosen split, each page is read by exactly one core. No operand is reuse-shared, so there is no broadcast regime.
- **Stall-shadow check:** three stages wait. The reader waits on `noc_async_read_barrier` (DRAM latency). The writer waits on `cb_wait_front` (the reader's block) and on `noc_async_writes_flushed`. The only work that does not depend on a pending read is the write of an *earlier* block. That work exists only when a core has more than one block, and the scheme exposes it through `block_pages` and `cb_depth_blocks` (overlap lamp). The reorder is legal because a copy commutes with any reordering of page moves; no floating-point order is involved.

### Block schedule

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);    // reader: NCRISC, NoC0
    store_block(block_idx);   // writer: BRISC, NoC1
}
drain_writes();               // writer: every issued write acknowledged before the kernel returns (BB-03)
```

`num_blocks_this_core = ceil(num_pages_this_core / block_pages)` and `last_block_extent = num_pages_this_core − (num_blocks_this_core − 1) · block_pages`. Both are derived in the kernel from the CT arg `block_pages` and the per-core RT arg `num_pages_this_core`.

| Operation | Kernel | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|--------|-------------|--------------------|-----------------------------------|
| `load_block` | reader | `block_extent × aligned_page_bytes`, where `block_extent = block_pages`, or `last_block_extent` on the last block. Page ids run from `start_page + block_idx·block_pages`. | Its CB slot (`block_pages` pages) stays live until `store_block` pops it | Per block: 1 `cb_reserve_back(block_pages)`, then `block_extent` page reads issued back to back with no barrier between them, then 1 `noc_async_read_barrier`, then 1 `cb_push_back(block_pages)` (the nominal count even when ragged) |
| `store_block` | writer | same | — | Per block: 1 `cb_wait_front(block_pages)`, then `block_extent` page writes issued back to back, then 1 `noc_async_writes_flushed` (the data has left L1, so the slot may be reused), then 1 `cb_pop_front(block_pages)` |
| `drain_writes` | writer | — | — | Once per kernel: 1 `noc_async_write_barrier` |

**Profiler zones (BB-21).** Each issue and each blocking wait gets its own `DeviceZoneScopedN` zone. These compile out of the plain build.

| Kernel | Zones |
|--------|-------|
| reader | `g6c_rd_reserve`, `g6c_rd_issue`, `g6c_rd_barrier` |
| writer | `g6c_wr_wait`, `g6c_wr_issue`, `g6c_wr_flush`, `g6c_wr_drain` |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap | With one block per core, the writer is idle until the last read of the block lands, so read and write do not overlap within a core. Cores finish their reads at different times, so DRAM still sees mixed traffic across the grid. | (a) `block_pages = ceil(core_pages/2)` or `ceil(core_pages/4)` with depth 2. (b) Keep issuing the whole assignment, but tag sub-batches with trids (`dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid`, `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:147,158`) and push each sub-batch on its own trid barrier. Measure on p1, p2, p3 and p4 (catalog `trid_double_issue`, `double_buffer`). |
| Grid synchronization / occupancy | A full grid for a DRAM-bound copy may be more than needed: DRAM saturates before the grid fills (catalog `double_buffer`: 64 Wormhole cores reach about DRAM peak). Every extra core adds a launch message and RT args. The 1–48-page shapes run one page per core. | Cap `num_cores` (for example at 64, or at a floor of k pages per core) on p1–p4, and use fewer cores on shapes of ≤ 48 pages. Judge the result by trace-replay host time, not by profiled kernel time alone (lessons RB-053). |
| Bank stagger | Bank = page id mod 8. When a core's range length is ≡ 0 (mod 8), every core starts on the same bank. p1 (28 pages per core) alternates between banks 0 and 4. | Rotate each core's issue order inside its range by `core_idx mod num_banks` (catalog `bank_stagger`: 1.04–1.32× on Blackhole). |
| Split reader | One RISC per direction issues every transfer. | If the profile shows the reader issue-bound (not expected at ≥ 2 KB pages), split each core's reads across NCRISC and BRISC (catalog `split_reader`). |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM `x` → L1 `cb_page_staging` | Raw bytes of whole aligned pages: tiles in face order, or a row plus its alignment gap. No reinterpretation. | reader (NCRISC, NoC0): `TensorAccessor(x)` + `noc_async_read_page`, one read barrier per block | Every dtype and layout moves as bytes. The CB `data_format` is `x.dtype` only to label the descriptor; no unpacker ever reads it. |
| L1 reader → writer | the same bytes in place | CB handshake: push and wait `block_pages` | No compute kernel, so no TRISC touches the data. |
| L1 `cb_page_staging` → DRAM `out` | the same bytes | writer (BRISC, NoC1): `TensorAccessor(out)` + `noc_async_write_page`, `noc_async_writes_flushed` per block, `noc_async_write_barrier` at the end | The output's padding and alignment gaps receive the input's padding bytes. This is allowed: they are not part of the result, and no logical element reads them. |

**TARGET axes as blocking.** Every TARGET value is a knob-turn on the built regime, not a new algorithm:

| TARGET value | What changes | Cost class |
|--------------|--------------|------------|
| `dtype = bfloat16` | `aligned_page_bytes` 2048 (from the buffer), CB `data_format` | knob-turn (SUPPORTED only) |
| `dtype = int32`, `uint32` | CB `data_format`. Page bytes come from the buffer. | knob-turn |
| `layout = ROW_MAJOR_LAYOUT` | page = one row at its aligned size (64 B for 4/32/64 B rows, 128 B, 5120 B) | knob-turn: the kernels are unchanged, and the CB page uses the aligned size |

TARGET has no `memory_layout` axis. The flat page split is a logical one-dimensional shard: a contiguous page range per core, never pinned by a caller. No physical shard exists, so no CB is backed on a sharded buffer.

**Structural impossibilities:** none to add. `INVALID = []` is correct.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | A block of `block_pages` consecutive page ids. The page is the atom. |
| Grid | `G = grid.x · grid.y` from `device.compute_with_storage_grid_size()` (11×10 = 110 on this Blackhole box family). `num_cores = min(tensor_pages, G)`. |
| Per-core work | `split_work_to_cores(grid, tensor_pages, row_wise=True)`. The group-1 cores come first in row-major order and get `core_pages_g1 = ceil(P / num_cores)`. The group-2 cores get `core_pages_g1 − 1`. Each core's `start_page` is the running sum of page counts over the cores before it, in the same row-major order (`ttnn.grid_to_cores(num_cores, grid.x, grid.y, row_wise=True)`). |
| Remainder | Group-2 cores, and the last block of any capped core, keep the nominal `block_pages` handshake and move `last_block_extent` pages. |
| Tile geometry | `tensor_pages = x.buffer_num_pages()`, read from the padded spec. For TILE this equals `lead · ceil(H/32) · ceil(W/32)` per image, never `floor(volume/1024)`. |
| Kernel groups / core ranges | One reader and one writer `KernelDescriptor`, each over `all_cores`. That is ≤ 2 rectangles: the full rows plus one partial row (BB-17). The group 1 / group 2 distinction lives only in the RT args. |
| Per-core args (derived, not tabled) | CT: `block_pages`, the CB index, `TensorAccessorArgs`. Per-core RT: `start_page`, `num_pages_this_core`. Common RT (the same on every core): the buffer address, so a cache hit patches one value per kernel. |

Per-shape plan at G = 110. Each row is a direct consequence of the formulas above.

| INPUTS case | tensor_pages | aligned_page_bytes | num_cores | core_pages g1 × n / g2 × n | block_pages | blocks/core | core ranges | CB bytes per core |
|-------------|--------------|--------------------|-----------|-----------------------------|-------------|-------------|-------------|-------------------|
| p1 `(16,12,128,128)` f32 TILE | 3072 | 4096 | 110 | 28 × 102 / 27 × 8 | 28 | 1 | 1 | 114688 |
| p2 `(8,12,128,128)` f32 TILE | 1536 | 4096 | 110 | 14 × 106 / 13 × 4 | 14 | 1 | 1 | 57344 |
| `(1,12,128,128)` f32 TILE | 192 | 4096 | 110 | 2 × 82 / 1 × 28 | 2 | 1 | 1 | 8192 |
| `(1,1)` f32 TILE | 1 | 4096 | 1 | 1 × 1 | 1 | 1 | 1 | 4096 |
| p3 `(1,512,5120)` bf16 TILE | 2560 | 2048 | 110 | 24 × 30 / 23 × 80 | 24 | 1 | 1 | 49152 |
| p4 `(1,1,32,62080)` and `(1,1,1,62080)` bf16 TILE | 1940 | 2048 | 110 | 18 × 70 / 17 × 40 | 18 | 1 | 1 | 36864 |
| `(1,256,5120)` bf16 TILE | 1280 | 2048 | 110 | 12 × 70 / 11 × 40 | 12 | 1 | 1 | 24576 |
| `(1,8192,128)`, `(1,1,8192,128)` bf16 TILE | 1024 | 2048 | 110 | 10 × 34 / 9 × 76 | 10 | 1 | 1 | 20480 |
| `(1,1,8448,128)` bf16 TILE | 1056 | 2048 | 110 | 10 × 66 / 9 × 44 | 10 | 1 | 1 | 20480 |
| `(1,256,3072)` bf16 TILE | 768 | 2048 | 110 | 7 × 108 / 6 × 2 | 7 | 1 | 1 | 14336 |
| `(1,1,2304,128)` bf16 TILE | 288 | 2048 | 110 | 3 × 68 / 2 × 42 | 3 | 1 | 1 | 6144 |
| `(1,2048,128)`, `(1,1,2048,128)` bf16 TILE | 256 | 2048 | 110 | 3 × 36 / 2 × 74 | 3 | 1 | 1 | 6144 |
| `(1,5120)` bf16 TILE | 160 | 2048 | 110 | 2 × 50 / 1 × 60 | 2 | 1 | 1 | 4096 |
| `(1,1,256,128)` bf16 TILE | 32 | 2048 | 32 | 1 × 32 | 1 | 1 | 2 | 2048 |
| `(1,1)` bf16 TILE | 1 | 2048 | 1 | 1 | 1 | 1 | 1 | 2048 |
| `(16,3,2560)` bf16 RM | 48 | 5120 | 48 | 1 × 48 | 1 | 1 | 2 | 5120 |
| `(8,3,2560)` bf16 RM | 24 | 5120 | 24 | 1 × 24 | 1 | 1 | 2 | 5120 |
| `(1,3,2560)` bf16 RM | 3 | 5120 | 3 | 1 × 3 | 1 | 1 | 1 | 5120 |
| `(32,)`, `(1,1,1,32)` uint32 RM | 1 | 128 | 1 | 1 | 1 | 1 | 1 | 128 |
| `(16,)`, `(8,)`, `(1,)` int32/uint32 RM | 1 | 64 | 1 | 1 | 1 | 1 | 1 | 64 |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_page_staging` | 0 | `aligned_page_bytes` (= `x.buffer_aligned_page_size()`) | `cb_depth_blocks · block_pages` | Spans `page` → `block_pages` and `in_page` → the whole aligned page. Streams over the block walk (`block_idx`) with depth `cb_depth_blocks`. `lead`, `h_tiles`, `w_tiles` and `rows` are folded into `page`. | `x.dtype` (Float32 in Phase 0). The payload is dataflow-only and never passes through DEST. | reader | writer | whole program |

This is the only CB. Push count equals wait count (`block_pages` per block on both sides), and the ring closes after `cb_depth_blocks` blocks. Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_extent × aligned_page_bytes` | no: raw dataflow (see API Mapping) | — (DRAM `x`, pages `start_page + block_idx·block_pages + [0, block_extent)`) | `cb_page_staging`, `block_pages` reserved and pushed | `block_pages` pages visible to the writer; `block_extent` of them hold data |
| 2 | `store_block` | `block_extent × aligned_page_bytes` | no: raw dataflow | `cb_page_staging`, `block_pages` waited | — (DRAM `out`, the same page ids) | `block_pages` popped, after the writes have left L1 |
| 3 | `drain_writes` | — | no: raw dataflow | — | — | every write acknowledged |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_block` | raw_api | `TensorAccessorArgs<…>()` + `TensorAccessor(args, addr)` | `tt_metal/hw/inc/api/tensor/tensor_accessor_args.h:21`, `tt_metal/hw/inc/api/tensor/tensor_accessor.h:71,99` | CT `TensorAccessorArgs` (from `ttnn.TensorAccessorArgs(x)`); address from common RT | — | — | none (page size is the buffer's) |
| `load_block` | raw_api | `cb_reserve_back(cb, block_pages)` / `get_write_ptr(cb)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:406`, `:325` | `block_pages` | — | `cb_page_staging` | `block_pages` |
| `load_block` | raw_api | `noc_async_read_page(page_id, accessor, l1_addr)` | `dataflow_api.h:1243` | one per page, l1_addr += `aligned_page_bytes` | — | `cb_page_staging` | `block_extent` (trip count) |
| `load_block` (logical rows) | raw_api | `noc_async_read<NOC_MAX_BURST_SIZE>(page_noc_addr + offset, l1_addr + offset, length)` | `dataflow_api.h:546` | one or two per page, from `PartialTileWalk::pieces()` (CT table of four kinds) | — | `cb_page_staging` | `block_extent`; piece table from `PIECE_MERGE_GAP_BYTES` |
| `load_block` | raw_api | `noc_async_read_barrier()` | `dataflow_api.h:1921` | once per block | — | — | frequency set by `block_pages` |
| `load_block` | raw_api | `cb_push_back(cb, block_pages)` | `dataflow_api.h:208` | nominal `block_pages` | — | `cb_page_staging` | `block_pages` |
| `store_block` | raw_api | `cb_wait_front(cb, block_pages)` / `get_read_ptr(cb)` | `dataflow_api.h:479`, `:346` | `block_pages` | `cb_page_staging` | — | `block_pages` |
| `store_block` | raw_api | `noc_async_write_page(page_id, accessor, l1_addr)` | `dataflow_api.h:1418` | one per page | `cb_page_staging` | — | `block_extent` |
| `store_block` (logical rows) | raw_api | `noc_async_write<NOC_MAX_BURST_SIZE>(l1_addr + offset, page_noc_addr + offset, length)` | `dataflow_api.h:875` | the same pieces as the reader | `cb_page_staging` | — | `block_extent` |
| `store_block` | raw_api | `noc_async_writes_flushed()` | `dataflow_api.h:1980` | once per block, before the pop | — | — | frequency set by `block_pages` |
| `store_block` | raw_api | `cb_pop_front(cb, block_pages)` | `dataflow_api.h:260` | nominal `block_pages` | `cb_page_staging` | — | `block_pages` |
| `drain_writes` | raw_api | `noc_async_write_barrier()` | `dataflow_api.h:1951` | once per kernel | — | — | — |

**Helpers considered and rejected.** The rejection covers every `raw_api` row above.

| Candidate | File:Line of the mismatch | Why it cannot express `load_block` / `store_block` |
|-----------|---------------------------|---------------------------------------------------|
| `compute_kernel_lib::copy` (eltwise convenience) | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:109,113` ("Pure copy — CopyTile(D0) -> PackTile(D0)") | It routes every tile through unpack → DEST → pack, the engine path spec rule 1 forbids (R1). It also needs a compute kernel and a second CB. |
| `read_sticks_for_tilize` / `write_sticks_after_untilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:70` ("stick-indexed"), `:62` ("Pushes 1 page per row"), `:106` ("Always operates at TILE granularity — the compute_kernel_lib::untilize …") | Phase 0 pages are tiles, not sticks. Its ROW mode hands off one page per row, a unit-at-a-time handshake instead of `block_pages`. The writer half consumes the tile-sized output of an untilize, not a staged row. |
| `local_copy_helpers_dataflow` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:9` ("Local L1 -> L1 copy helpers: self-aimed NoC reads") | It does L1 → L1 self-aimed reads only, with no DRAM page addressing. Its `set_read_trid` / `async_read_barrier_with_trid` (`:147`, `:158`) are the tool for the overlap lamp's alternative, which Phase 0 does not need (one barrier per block). |
| `ccl_helpers_dataflow` `write_page` | `ttnn/cpp/ttnn/kernel_lib/ccl/ccl_helpers_dataflow.hpp:327` | It writes through a fabric/CCL channel, not a local DRAM page copy. |
| `mcast` (`McastArgs`) | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp` | There is no reuse-shared operand to broadcast. |

What would close the gap: a dataflow `copy_pages` block helper that reads a page range into a CB behind one barrier and writes it out behind one flush. It is not required. The two raw block operations above are that helper's body.

## Host Contract

### Registry (op file `ttnn/ttnn/operations/g6_clone/g6_clone.py`)

| Declaration | Content |
|-------------|---------|
| `INPUT_TAGGERS` | `{"dtype": tag_dtype, "layout": tag_layout}`. `tag_dtype(inputs, axes)` returns `inputs[0]["dtype"]`. `tag_layout(inputs, axes)` returns `inputs[0]["layout"]`. There is no other tagger: no alignment, rank or size axis. |
| `SUPPORTED` | `{"dtype": [ttnn.float32, ttnn.bfloat16, ttnn.int32, ttnn.uint32], "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]}` (Refinement 1; Phase 0 was float32 + TILE) |
| `EXCLUSIONS` | `[]` |
| INVALID | Not declared or checked in the op. |

### `validate(input_tensor)` order

| # | Check | Raises | Message must contain (the acceptance test matches it) |
|---|-------|--------|--------------------------------------------------------|
| 1 | `isinstance(input_tensor, ttnn.Tensor)` and `ttnn.is_tensor_storage_on_device(input_tensor)` | `ValueError` | `g6_clone: input_tensor must be a device tensor` |
| 2 | `1 ≤ len(input_tensor.shape) ≤ 4` | `ValueError` | `g6_clone: input_tensor rank must be 1 to 4` |
| 3 | `memory_config().buffer_type == DRAM` and `memory_layout == INTERLEAVED` | `ValueError` | `g6_clone: input_tensor must be DRAM interleaved` |
| 4 | Build `scenario = {"input_shape": tuple(shape), "dtype": input_tensor.dtype, "layout": input_tensor.layout}`, run `INPUT_TAGGERS` on `(scenario,)`, then check `SUPPORTED` per axis | `UnsupportedAxisValue` | free text |
| 5 | `EXCLUSIONS` (empty) | `ExcludedCell` | free text |

### Entry point `g6_clone(input_tensor)`

| Step | Action |
|------|--------|
| 1 | `validate(input_tensor)`. This is the first line. |
| 2 | `out = ttnn.allocate_tensor_on_device(input_tensor.spec, input_tensor.device())`. This is a new buffer on every call and is never `x` or a view of it. |
| 3 | Assert equal page count and aligned page size for `x` and `out`. |
| 4 | Fetch the cached `ProgramDescriptor` for the key, or build it on a miss (BB-07). |
| 5 | Write `x.buffer_address()` into the reader's common RT args and `out.buffer_address()` into the writer's. |
| 6 | `return ttnn.generic_op([input_tensor, out], descriptor)`. The output is last. This is exactly one program; no other ttnn op is called. |

**Cache key:** `(device id, grid.x, grid.y, dtype, layout, buffer_type, memory_layout, tensor_pages, aligned_page_bytes, tile_geometry(x))`. `tile_geometry` = `(h_tiles, w_tiles, rows_last, cols_last, elem_bytes)` or None; it separates `[1,1,1,62080]` from `[1,1,32,62080]`, which share page count and page size.

- Everything in the descriptor is a function of this key: CB size and format, CT args, core ranges, per-core `start_page` / `num_pages_this_core`. Only the two addresses vary, and they are patched per call.
- `dtype` is in the key even where the bytes coincide, because `[16]` int32 and uint32 differ in CB `data_format` (R6).
- `generic_op` refreshes RT values on its own program-cache hit, because its hash covers structure, not RT values (`ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:78-103`).
- Nothing else is allocated, and nothing is built on a hit (BB-16).

### Plan and role report (BB-08, BB-21)

The op module exposes `g6_clone_plan(input_tensor) -> dict` with these fields:

- `regime`: `"flat_page_range"` or `"logical_rows_partial_tile"`
- `partial_tile_walk`: None, or `h_tiles`, `w_tiles`, `rows_last`, `cols_last` and the `(offset, length)` pieces of the four page kinds
- `grid`
- `num_cores`
- `core_ranges`
- `core_pages_g1`, `core_pages_g2`
- `block_pages`
- `cb_depth_blocks`
- `num_blocks_max`
- `roles`: per core, `(core, "reader+writer", start_page, num_pages)`

The entry point uses the same function, so the report is the plan that ran.

`PROPERTIES` (optional): `multi_core` declared `True`.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| An engine path changes bits (R1) | The gate is bit-exact on random bit patterns: NaN payloads, subnormals, int32 values above 2^24. | No compute kernel. Bytes move only through the NoC. `engine_copy` is rejected. |
| The output aliases the input (R2) | Values would be right while the storage is shared. | `allocate_tensor_on_device(input.spec, …)` on every call. `x` is never written: the writer's accessor is `out` only. |
| Partial tiles (R3) | Poisoned-padding cases fill the input's padding with NaN. | Each logical element is copied from its own byte offset to the same offset of the same page id. A logical-rows range rounded to 64 B, or merged through a gap, carries padding bytes only into the output's padding. Pinned by the golden poisoned cases and `test_g6_clone_poisoned_padding_kinds` (all four kinds, bf16 / f32 / int32). |
| The work collapses onto few cores (R4) | p4 and `[1,1,1,62080]` are one tile row of 1940 tiles. | The split is on the flat page axis, never on tile rows or columns: 110 cores on p4. |
| Small ROW_MAJOR pages vs the 64 B DRAM alignment (R5) | Rows are 4, 32, 64 or 128 B against a 64 B DRAM page stride. | The CB page is `buffer_aligned_page_size()`, and the TensorAccessor moves the aligned page (Mechanism caps). The bytes in the output's alignment gap are not data. |
| An incomplete program cache key (R6, BB-07) | `[1,1]` f32 vs bf16, and `[16]` int32 vs uint32. | dtype, layout, page count and aligned page size are in the Python key, and the CB format is in generic_op's hash. Addresses are patched per call (test_regression `test_repeat`). |
| Group-2 cores push a short count | A core with one page fewer, if it pushed its actual count, would desync the ring. | Pushes and pops are always nominal `block_pages`; only the transfers narrow. |
| A write left in flight at return (BB-03) | Another program starts right after the call. | The writer ends with `noc_async_write_barrier`. The reader's last block ends with a read barrier before its push. |
| The slot is reused before a write leaves L1 | With depth 2, the reader refills a slot the writer is still sending from. | `noc_async_writes_flushed` before each `cb_pop_front`. |
