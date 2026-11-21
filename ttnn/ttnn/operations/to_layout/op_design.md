# Operation Design: to_layout

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (a layout change: one unpacker/packer pass per tile, no arithmetic) |
| Goal | Return a tensor with the input's shape, dtype and values in the requested layout (`TILE_LAYOUT` / `ROW_MAJOR_LAYOUT`), allocated in the input's buffer (DRAM or L1, interleaved), as **exactly one** `ttnn.generic_op` dispatch per call. |
| Math | `output[n, c, y, x] = input[n, c, y, x]` for every logical index; only the storage order changes. Padding of a TILE output (rows `y >= H` of an image's last tile-row) is written as 0. |
| Specification | `eval/specs/to_layout/spec.py` (governs where this design and it disagree) |
| References | `.claude/references/targets/quasar.md`, `.claude/references/backends/ttsim.md`, `.claude/references/quasar_op_examples/` (host program + ring protocols), `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` Part A, `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp`, `ttnn/ttnn/operations/examples/master.md` (entries `width_split`, `double_buffer`, `tile_reorder`, `compute_block_size`, `bank_stagger`, `noc_placement`) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on the ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 clusters (worked numbers below); the op queries `device.compute_with_storage_grid_size()` | target page |
| L1 per core | 4,194,304 B (4 MiB) per cluster, shared by its NEOs and DM cores | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES`; the query's value on this target is not recorded in the profile (≤ 4,194,304 B). Worked numbers below never come near it: the largest Phase 0 footprint is 128 KiB per cluster. | `unknown` (exact value) — read by the query at run time |
| Host program | `ttnn.ProgramSpec` (`ttnn.experimental.ProgramSpec`, `KernelSpec`, `DataflowBufferSpec`, `ttnn.generic_op`) — `.claude/references/quasar_op_examples/` (README, `toy_thread_plan/toy_thread_plan_program_artifacts.py`) | target page |
| Threads per core | 4 NEOs (C ∈ {1, 2, 4}), 6 user DM harts (R + W ≤ 6), on STRIDED rings R | C and W | C. This design: plan (1, 1, 1) (Work Distribution). | target page, *Thread plan* |
| DFB limits | 32 DFBs per cluster, ≤ 24 active per NEO, one ring ≤ 1,048,560 B on the compute side; one producer kernel and one consumer kernel per ring | target page, *Cluster model* |
| Alignment | L1 16 B (`ttnn.get_l1_alignment()`), DRAM 64 B (`ttnn.get_dram_alignment()`) | target page |
| Unavailable here | (1) `ttnn.ProgramDescriptor` / Metal 1 kernels and the legacy C++ program factory — refused at kernel creation. (2) `bfloat8_b`, `bfloat4_b` — no path on this family; refused by `validate()`, never queued. (3) uint16 / uint32 as a `DataflowBufferSpec` format — refused at build (`dfb.cpp:142-149`); a byte-moving ring carries uint16 as bfloat16 and uint32 as int32 (matters for dtype refinements only). (4) An unset `fpu_math_fidelity` — not used; set it on every compute `KernelSpec`. (5) The library's `read_sticks_for_tilize` / `write_sticks_after_untilize` dataflow helpers — Metal 1 CB API with untagged transfers (API Mapping). | target page *Not available* |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on device; Phase 0: bfloat16, rank 4 (rank 3 since Refinement 0), W % 32 == 0, DRAM or L1 interleaved | — | — |
| `layout` | `ttnn.Layout` | yes | `ttnn.TILE_LAYOUT`, `ttnn.ROW_MAJOR_LAYOUT`; anything that is not a `ttnn.Layout` raises `TypeError` (rejection X1) | — | host only (selects the regime) |

There is no compute config (eval/CONVENTIONS.md §2.2 exempts pure data movement) and no `memory_config` argument (R5 fixes the output placement).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[N, C, H, W]` (Phase 0 rank 4; rank 3 [B, S, H] since Refinement 0); the op internally folds every leading dim into `num_images = prod(shape[:-2])`, so ranks 2 and 3 need no kernel change |
| Dtype | Phase 0: `bfloat16` |
| Layout | `TILE_LAYOUT` or `ROW_MAJOR_LAYOUT` (axis `layout`) |
| Memory | interleaved, DRAM or L1 (axes `memory_config`, `sharding` from `eval.memory.memory_axes(input_tensor)`, R6) |

### Output

| Property | Value |
|----------|-------|
| Shape | logical shape of the input (TILE output is padded per image to `ceil(H/32)*32` rows; padding rows hold 0) |
| Dtype | input dtype |
| Layout | `layout` argument |
| Memory | `input_tensor.memory_config()` (same buffer type, interleaved; R5). Allocated with `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(input_tensor.shape, input_tensor.dtype, layout, input_tensor.memory_config()), device)` |

## Blocking Model

The op computes nothing, so a block is the unit that moves: one image tile-row band (32 rows) × `block_width_tiles` tile-columns. Its 32 sticks of `block_width_tiles·32` elements and its `block_width_tiles` tiles hold exactly the same bytes, so tilize / untilize is a block-local permutation.

### Axes

Leading dims fold into one axis: `tile_row = image · Ht + h`, where `Ht = ceil(H/32)` is computed **per image** and `image ∈ [0, num_images)`. Every image is tile-padded on its own.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `tile_row` (folded N·C·Ht, `total_tile_rows = num_images · ceil(H/32)`) | **independent**: each output element depends only on the input element at the same index, and two tile-rows share no tile and no stick | `block_height_tiles` | 1 tile-row. **Why 1:** (a) the compute fixed cost (helper init and reconfig) is paid once per **kernel**, because each compute kernel makes one helper call over all its units (catalog `compute_block_size`), so it does not depend on this extent. (b) The DM fixed cost (one barrier per block) is already spread over 32 stick transfers per block, 4–8× past the catalog's `double_buffer` sweet spot. (c) Phase 0 per-core assignments are 1–2 units on 13 of 15 PHASE0 shapes (worked table below), so a taller block would leave a core one block with no read/compute/write overlap. Overlap lamp below. | host constant `BLOCK_HEIGHT_TILES` (CT arg) | spread over the grid together with `tile_col` as one flattened unit index (`split_work_to_cores`, `row_wise=True`) | knob-turn |
| `tile_col` (`tensor_w_tiles = ceil(W/32)`) | **independent**: no element crosses tile-columns | `block_width_tiles` (`bw`) | `select_block_width_tiles(...)` (next table): occupancy first, then the coarsest divisor of `tensor_w_tiles` within the caps. Values: 2–16 on the PHASE0 shapes | one host function → one CT arg; `chunk_bytes = bw·32·elem_size`, ring capacities and `n_col_blocks = tensor_w_tiles / bw` derive from it | flattened with `tile_row`: unit `u` → `tile_row = u // n_col_blocks`, `col_block = u % n_col_blocks` | knob-turn |
| `row_in_tile` (32 rows of one band) | **dependent**: one tile spans all 32 rows of its band. Tilize needs all 32 sticks to produce any tile; untilize produces all 32 sticks from each tile | fixed = tile height 32 (mechanism cap) | 32 (whole extent, not split) | tile geometry (`unpack_tile_r_dim` of the ring) | not assigned across cores: a split needs a cross-core combine of half-tiles (regime `split_within_tile`, rejected) | scheme-change (rejected) |
| `col_in_tile` (32 elements) | **dependent**: one tile spans 32 columns | fixed = tile width 32 | 32 (whole extent) | tile geometry | not assigned across cores (same reason) | scheme-change (rejected) |
| `page` (identity regime only: `num_pages = buffer_num_pages()`) | **independent**: a page copy has no cross-page dependency | `copy_block_pages` (pages in flight per half-ring) | 4 (catalog `double_buffer`: 4–8 transfers outstanding; `tile_reorder`: whole pages per transfer) | host constant `COPY_BLOCK_PAGES` | `split_work_to_cores(grid, num_pages, row_wise=True)`, contiguous page ranges | knob-turn |

**Intermediate data.** The op has no intermediate tensor. The only internal data is the block in transit (rows in a ring, tiles in a ring), and it is consumed by the stage that follows on the same core. Every stage of every regime runs on every engaged core. No core collects, coordinates or computes for others.

**`block_width_tiles` selection** (single source, host, evaluated per call):

| Step | Rule (all quantities queried or derived; worked values for Quasar bf16) |
|------|------------------------------------------------------------------------|
| 1. floor | `bw_min = min(tensor_w_tiles, ceil(MIN_CHUNK_BYTES / (32·elem_size)))`, `MIN_CHUNK_BYTES = 256` (one NoC payload width on this part). bf16: 4 |
| 2. caps | `bw_max = min(floor(DFB_RING_MAX_BYTES / (BUFFER_DEPTH·tile_bytes)), floor(dfb_budget / (2·BUFFER_DEPTH·tile_bytes)))` with `dfb_budget = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES`, `DFB_RING_MAX_BYTES = 1,048,560`. bf16: 255 (ring cap binds) |
| 3. legal set | `D = {d : d divides tensor_w_tiles, bw_min ≤ d ≤ bw_max, (d·32·elem_size) % dram_alignment == 0}`. If `D` is empty: `D = {largest divisor of tensor_w_tiles ≤ bw_max}` |
| 4. occupancy first | `bw = max{d ∈ D : total_tile_rows · (tensor_w_tiles / d) ≥ num_cores}`; if no `d` qualifies, `bw = min(D)` |

`num_cores = grid.x · grid.y` from `device.compute_with_storage_grid_size()`.

**Worked Phase 0 values (Quasar, 32 clusters, bf16):**

| Input | total_tile_rows | tensor_w_tiles | bw | units | engaged clusters | units / cluster |
|-------|----------------|----------------|----|-------|------------------|-----------------|
| (1,1,400,512) | 13 | 16 | 4 | 52 | 32 | 1–2 |
| (1,1,1600,256) | 50 | 8 | 8 | 50 | 32 | 1–2 |
| (1,1,32,64) / (1,1,17,64) | 1 | 2 | 2 | 1 | 1 | 1 |
| (1,1,64,128) / (1,1,50,128) | 2 | 4 | 4 | 2 | 2 | 1 |
| (4,8,32,256) | 32 | 8 | 8 | 32 | 32 | 1 |
| (2,4,128,512) | 32 | 16 | 16 | 32 | 32 | 1 |
| (1,1,2048,256) / (4,8,47,256) | 64 | 8 | 8 | 64 | 32 | 2 |
| (4,1,512,512) | 64 | 16 | 16 | 64 | 32 | 2 |
| (1,1,32,4096) | 1 | 128 | 4 | 32 | 32 | 1 |
| (1,1,32,8192) | 1 | 256 | 8 | 32 | 32 | 1 |
| (1,1,128,4096) / (2,1,64,4096) | 4 | 128 | 16 | 32 | 32 | 1 |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_row_block` (tilize: reader → compute) | `BUFFER_DEPTH` | 2 blocks | reader fills block k+1 while the unpacker tilizes block k |
| `cb_tile_block` (tilize: compute → writer) | `BUFFER_DEPTH` | 2 blocks | packer fills block k+1 while the writer drains block k |
| `cb_tile_block_in` (untilize: reader → compute) | `BUFFER_DEPTH` | 2 blocks | reader's implicit reads of block k+1 overlap untilize of block k; also gives the DFB two halves for its two implicit-sync txn ids |
| `cb_row_block_out` (untilize: compute → writer) | `BUFFER_DEPTH` | 2 blocks | packer untilizes block k+1 while the writer's stick writes drain block k |
| `cb_pages` (identity: reader → writer) | `BUFFER_DEPTH` | 2 × `COPY_BLOCK_PAGES` entries | two implicit-sync txn-id halves: one half fills while the other drains |

One host constant `BUFFER_DEPTH = 2` serves every ring.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Quasar unpack-tilize MOP: `block_c_tiles` must equal the init's `full_ct_dim` (`llk_unpack_tilize_api.h:67-68`); the helper passes `block_width_tiles` to both (`tilize_helpers.inl:182,185,228,230`) | `block_width_tiles` must be one compile-time value for every block of a program | `bw` divides `tensor_w_tiles` (step 3), so no ragged column block exists | a narrower last block run on the init's width reads the next block's rows: wrong tiles |
| DFB ring size ≤ 1,048,560 B (target page) | `BUFFER_DEPTH · bw · tile_bytes` per ring | `bw ≤ floor(1,048,560 / (BUFFER_DEPTH · tile_bytes))` (step 2) | program build refuses the ring |
| Per-cluster L1 | `2 · BUFFER_DEPTH · bw · tile_bytes` (two rings live) | `≤ dfb_budget` (step 2) | rings collide with L1-resident tensors at launch |
| DRAM / L1 alignment of a stick-chunk transfer (src offset `col_block·chunk_bytes`, dst offset `row·chunk_bytes`) | `chunk_bytes = bw·32·elem_size` | multiple of `ttnn.get_dram_alignment()` (64 B) and `ttnn.get_l1_alignment()` (16 B), step 3. bf16: every `bw` passes; a 1-byte dtype needs `bw` even | misaligned DRAM reads return shifted bytes |
| Tile geometry 32 × 32 | `row_in_tile`, `col_in_tile` extents | fixed 32; every tile count is `ceil(·/32)` per image | — |
| Tilize ring contiguity: the unpacker reads 32 rows of `chunk_bytes` from the read cursor as one region (`llk_unpack_tilize_api.h:80-105`) | a block must never straddle the ring end | `cb_row_block` capacity = `BUFFER_DEPTH · bw` and every push / pop is exactly `bw` entries | the last rows of a block are read from past the ring: wrong tiles |
| Helpers `ASSERT(num_blocks > 0)` (`tilize_helpers.inl:95`, `untilize_helpers.inl:160`) | units per engaged core ≥ 1 | the work unit's `target_nodes` are only the clusters `split_work_to_cores` gives work | an assert / zero-length helper call on an idle cluster |
| Untilize DEST capacity: widths above `DEST_AUTO_LIMIT` use the block-based pack path with `sub_block_width` = largest divisor of `bw` ≤ the limit (`untilize_helpers.inl:78-87`) | the sub-block (helper-internal, not a block knob) | none needed for correctness: the helper chooses it | — (perf only) |

### Regimes

Selection predicate (host, before any allocation): `regime = "identity_copy" if input_tensor.layout == layout else ("tilize_stream" if layout == ttnn.TILE_LAYOUT else "untilize_stream")`. Each built regime is its own `ProgramSpec`, and every call dispatches exactly one of them.

Minimum (named boundary, DRAM or the tensor's L1 banks): every input byte crosses it once and every output byte crosses it once.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `tilize_stream` | **built** | input `ROW_MAJOR`, `layout == TILE` | `block_height_tiles` (1) tile-row × `bw` tile-columns: 32 sticks × `chunk_bytes` in, `bw` tiles out | **minimum**: each logical stick chunk is read once (padding rows are never read: they are zeroed in L1), and each output tile (including its required padding rows) is written once. No cross-core traffic. | a wider `bw`: fewer, larger stick reads (`chunk_bytes` per read), one barrier / reserve / push for more bytes. A taller block: fewer barriers (lamp). Helper init / reconfig are once per kernel whatever the block |
| `untilize_stream` | **built** | input `TILE`, `layout == ROW_MAJOR` | 1 tile-row × `bw` tiles in, 32 sticks × `chunk_bytes` out | minimum on writes (only valid sticks are written). Reads are minimum + the padding rows of each image's last tile-row: the tile page is the transfer unit, so those bytes cross the boundary. Example (1,1,400,512): 16 of 416 rows. Structural at page granularity; the half-tile read is a lamp | same as above, mirrored (larger stick writes, fewer write barriers) |
| `identity_copy` | **built** | `input_tensor.layout == layout` | `copy_block_pages` pages per txn-id half | **minimum**: each page read once, written once, whole page per transfer | fewer txn-id ISR rounds per byte; more bytes in flight |
| `identity_alias` (return `input_tensor`, no dispatch) | **rejected**: the run's integrity rule requires exactly one device-program dispatch per invocation, and zero is not one. The specification asks only for "a tensor equal to the input" (DECISIONS 2026-11-19). Superseded by `identity_copy` | same as above | — | zero traffic | — |
| `split_within_tile` (cut `row_in_tile` / `col_in_tile` across cores, cores exchange half-tiles) | **rejected**: every byte would cross the NoC once more, with nothing re-read to save (no residency gain, since each byte is used once). Superseded by the independent 2-D split, which already fills 32 clusters on all PHASE0 shapes with ≥ 32 tile-columns·rows | — | + one cross-core crossing of the whole tensor | — |
| `dm_only_layout_change` (DM harts permute sticks ↔ faces: 32-B face-row transfers or RISC-V stores) | **rejected** for bfloat16: 32-B pieces are below the 64-B DRAM alignment (they would need an L1 staging copy plus a second local shuffle), and they turn every 2 KiB tile into 64 sub-transfers. The unpacker / packer path (`tilize_stream` / `untilize_stream`) supersedes it for every dtype the unpacker carries bit-exactly | — | minimum DRAM bytes, but 64 sub-tile transfers or 512 RISC-V stores per tile | — |
| `exact32_stream` (float32 / int32 / uint32: tilize with `UnpackToDest` into 32-bit DEST, `Fp32Mode::Lossless`, `enable_32_bit_dest=True`; untilize from 32-bit DEST) | **deferred**: no PHASE0 dtype is 32-bit. The SrcA path of `tilize_stream` truncates fp32 to tf32 and would break R1 (`tilize_helpers.hpp:49-73`). Reachable on the same scheme: only the ring format, the compute `KernelSpec` (32-bit DEST, unpack mode) and the helper template arguments change. Blocking, split and rings stay as they are (`tile_bytes` doubles, so `bw_max` halves through step 2) | dtype ∈ {float32, int32, uint32} | as `tilize_stream` / `untilize_stream` | minimum | as above |
| `exact16_8_stream` (uint16 / uint8: a bit-exact path for integer bit patterns, which SrcA's float handling may canonicalize, e.g. uint16 values that are bf16 denormals) | **deferred**: no PHASE0 dtype is a narrow integer. uint16 rides a bfloat16-format ring (uint16 is not a DFB format) and needs a proven bit-exact unpack path. uint8 needs its own format or the DM shuffle. Reachable: same split and rings, compute path selected per dtype | dtype ∈ {uint16, uint8} | as above | minimum | as above |

### Traffic ranking

Single operand (`input_tensor`), single output. The operand varies along every candidate split axis, so the operand-reuse check finds no reuse-shared operand and no broadcast row applies.

| Rank | Candidate split | DRAM crossings (input / output) | Cross-core traffic | Transfer shape | Verdict |
|------|-----------------|------------------------------|--------------------|----------------|---------|
| 1 | `tile_row` × `tile_col` (2-D, both independent) — **chosen** | 1 / 1 | none | tile side: whole 2 KiB pages. Stick side: `chunk_bytes = bw·64 B` (bf16) per transfer, ≥ 256 B by the floor | fills 32 clusters on 13 of 15 PHASE0 shapes. The other two have 1–2 units in total |
| 1 (tie on bytes) | `tile_row` only (`bw = tensor_w_tiles`) | 1 / 1 | none | full-width stick transfers | same bytes, larger stick transfers, but only `total_tile_rows` clusters busy: 13 for (1,1,400,512), 1 for (1,1,32,8192). Occupancy breaks the tie (catalog `width_split`: up to 7.8× when tile-rows are few). Kept reachable: it is the `bw = tensor_w_tiles` value of the same knob, which step 4 picks whenever `total_tile_rows ≥ num_cores` |
| 3 | dependent split (`row_in_tile` across cores + half-tile exchange) | 1 / 1 | + the whole tensor once | sub-tile pieces | strictly more bytes, with no residency to win since nothing is re-read. Rejected |

**Cheapest-traffic split = implemented split** (rank 1; every independent split moves identical bytes, and the choice between them is occupancy and then transfer size).

**Residency.** No byte is used twice, so no regime holds a block resident beyond its pass. A predicate-guarded resident path would buy nothing.

**Stall-shadow check.** Writer waits on compute, compute waits on the reader, reader waits on DRAM. The only work independent of the block being waited on is the next block's transfer. `BUFFER_DEPTH = 2` already schedules it into the stall. The op has no algebraic reorder to exploit: it is a permutation with no arithmetic, and floating-point order is never touched.

### Block schedule

Per engaged cluster, `num_units_this_core` units starting at `unit_start` (runtime args), unit `u` → (`tile_row = u // n_col_blocks`, `col_block = u % n_col_blocks`), `image = tile_row // Ht`, `h = tile_row % Ht`, `valid_rows = min(32, H - 32·h)`.

**`tilize_stream`:**

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_row_block(block_idx);    // reader
    tilize_block(block_idx);      // compute
    store_tile_block(block_idx);  // writer
}
```

| Operation | Kernel | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|--------|-------------|--------------------|-----------------------------------|
| `load_row_block` | reader (explicit sync on `cb_row_block`) | 32 rows × `chunk_bytes` = `bw` tile-sized entries | nothing beyond the reserved block | per block: one `reserve_back(bw)`, `valid_rows` stick-chunk reads tagged trid 0 (src page `image·H + 32·h + r`, src offset `col_block·chunk_bytes`, dst offset `r·chunk_bytes`), one `async_write_zeros` of `(32 - valid_rows)·chunk_bytes` at dst offset `valid_rows·chunk_bytes` **only when `valid_rows < 32`**, followed by `write_zeros_l1_barrier()`, one read barrier on trid 0, one `push_back(bw)` |
| `tilize_block` | compute | 1 tile-row × `bw` tiles | nothing | **one** `compute_kernel_hw_startup` and **one** helper call per kernel (`num_blocks = num_blocks_this_core · block_height_tiles` tile-rows), so tilize init / reconfig / uninit happen once per kernel. Per tile-row the helper waits `bw`, tilizes, pushes `bw`, pops `bw` |
| `store_tile_block` | writer (implicit sync on `cb_tile_block`) | `bw` tiles | nothing | per tile: one tagged implicit write to page `tile_row·tensor_w_tiles + col_block·bw + j`. No barrier. At kernel end: `noc_async_write_set_trid(0, noc.get_noc_id())` |

**`untilize_stream`:**

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_tile_block(block_idx);   // reader
    untilize_block(block_idx);    // compute
    store_row_block(block_idx);   // writer
}
```

| Operation | Kernel | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|--------|-------------|--------------------|-----------------------------------|
| `load_tile_block` | reader (implicit sync on `cb_tile_block_in`) | `bw` tiles | nothing | per tile: one tagged implicit read of page `tile_row·tensor_w_tiles + col_block·bw + j`. At kernel end: `noc_async_read_set_trid(0, noc.get_noc_id())` |
| `untilize_block` | compute | 1 tile-row × `bw` tiles → 32 rows × `chunk_bytes` | nothing | one `compute_kernel_hw_startup` and one helper call per kernel (`num_blocks = num_blocks_this_core · block_height_tiles`), so pack-untilize init / uninit happen once per kernel |
| `store_row_block` | writer (explicit sync on `cb_row_block_out`) | 32 rows × `chunk_bytes` (`bw` entries) | nothing | per block: one `wait_front(bw)`, `valid_rows` stick-chunk writes tagged trid 0 (src offset `r·chunk_bytes`, dst page `image·H + 32·h + r`, dst offset `col_block·chunk_bytes`), one write barrier on trid 0, one `pop_front(bw)`. Padding rows are popped, never written |

**`identity_copy`:**

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    copy_page_block(block_idx);   // reader: implicit reads into cb_pages; writer: implicit writes out of it
}
```

| Operation | Kernel | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|--------|-------------|--------------------|-----------------------------------|
| `copy_page_block` | reader + writer (implicit sync on both sides of `cb_pages`) | `copy_block_pages` whole pages (tile pages or sticks: `entry_size = input_tensor.buffer_page_size()`) | nothing | per page: one tagged read, one tagged write. Credits come from the DM0 ISR once per txn-id half (`copy_block_pages` entries). Each kernel ends with its trid hand-back |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization / occupancy vs transfer size | Step 4 cuts the width to fill 32 clusters, which shrinks the stick transfer to `bw·64 B` (256 B on (1,1,400,512)) | `bw` one divisor larger (e.g. 8 on (1,1,400,512): 26 clusters, 512 B sticks) and `bw = tensor_w_tiles` (13 clusters, 1 KiB sticks), compared by model cycles on `ttsim-perf` |
| Overlap (block height) | `block_height_tiles = 1` pays one barrier per 32 stick transfers. Where a cluster holds many units (`total_tile_rows ≫ 32`, outside PHASE0) a taller block may batch better | `BLOCK_HEIGHT_TILES = 2, 4` against 1 on (1,1,2048,256) and on a large-H shape |
| Explicit-sync stick side | Here a barrier covers ≤ 32 transfers, but the barrier still drains the NoC | (a) two alternating user trids per block (catalog `trid_double_issue`); (b) an implicit-sync stick-entry ring (entry = `chunk_bytes`, tilize helper's asymmetric form) with padding entries re-read from DRAM. (b) costs up to 31 extra stick reads per image |
| Thread plan | (1,1,1) is the page's plan for a data-movement-bound op. It is unmeasured for this op on `ttsim-perf` | (2,2,2) with **BLOCKED** rings of `block_size = bw` (each NEO needs a contiguous tile-row block, so STRIDED dealing per entry cannot be used), against (1,1,1), on (1,1,1600,256) |
| Untilize padding reads | each image's last tile-row reads whole tiles even where rows 16–31 are all padding (`valid_rows ≤ 16`) | read only faces 0–1 (first half page) of those tiles |
| Bank stagger | every cluster starts its walk at its first unit, and neighbouring clusters' first pages may land on the same DRAM channel | start each cluster at unit `core_index % num_units_this_core` and wrap (catalog `bank_stagger`) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| tilize: DRAM/L1 sticks → `cb_row_block` | row-major chunks, 32 rows × `chunk_bytes`, contiguous (row stride `chunk_bytes` = `bw·32·elem`) | reader, `TensorAccessor(input)`, `noc.async_read<NocOptions::TXN_ID>(acc, cb, chunk_bytes, {.page_id, .offset_bytes}, {.offset_bytes}, {.trid = 0})`; padding rows `noc.async_write_zeros(cb, bytes, {.offset_bytes})` | explicit sync: ring listed in the reader's `disable_dfb_implicit_sync_for`; every transfer tagged trid 0 |
| tilize: `cb_row_block` → `cb_tile_block` | rows → 32×32 tiles (bf16, faces) | compute: unpacker tilize MOP (whole `bw`-tile row per call) → A2D datacopy → pack | `compute_kernel_lib::tilize<bw, cb_row_block, cb_tile_block>` |
| tilize: `cb_tile_block` → DRAM/L1 tiles | tile pages | writer, implicit `noc.async_write<NocOptions::TXN_ID>(cb, acc, {}, {.page_id})` | trid handed back at kernel end |
| untilize: tiles → `cb_tile_block_in` | tile pages | reader, implicit `noc.async_read<NocOptions::TXN_ID>(acc, cb, {.page_id}, {})` | trid handed back |
| untilize: `cb_tile_block_in` → `cb_row_block_out` | tiles → 32 rows × `chunk_bytes`, contiguous in `bw` tile-sized entries | compute: unpack → datacopy → pack_untilize (row stride `bw` tile-widths) | `compute_kernel_lib::untilize<bw, cb_tile_block_in, cb_row_block_out>` |
| untilize: `cb_row_block_out` → sticks | stick chunks | writer, `noc.async_write<NocOptions::TXN_ID>(cb, acc, chunk_bytes, {.offset_bytes}, {.page_id, .offset_bytes}, {.trid = 0})` | explicit sync: ring listed in the writer's `disable_dfb_implicit_sync_for`; only `valid_rows` written |
| identity: pages → `cb_pages` → pages | page bytes, unchanged | reader + writer, implicit sync both sides (DM → DM ring) | no compute kernel |

**No inter-Tensix communication** in any built or deferred regime.

**TARGET axes → refinement class** (classified against this op's axis characters):

| TARGET axis value outside PHASE0 | Class | What changes |
|------------------------------------|-------|--------------|
| `rank` 2, 3 | knob-turn (host) | `num_images = prod(shape[:-2])` (1 for rank 2). Kernels unchanged |
| `alignment` `w_non_aligned` | knob-turn (reader/writer) | `tensor_w_tiles = ceil(W/32)`. The last column block's stick transfer is `W·elem - col_block·chunk_bytes` bytes, and the reader zeroes the tail columns of each row. The writer writes only the valid bytes. Stick page size `W·elem` stays the accessor's page |
| `memory_config` L1 | built in Phase 0 (same accessor path; output follows R5) | — |
| `sharding` | only `interleaved` in TARGET | (if added) height = `tile_row` cut, width = `tile_col` cut: both independent axes, so placement only — the shard is consumed in place via a ring backed on the sharded buffer, never re-read over the NoC |
| `dtype` float32, int32, uint32 | scheme-change inside compute (regime `exact32_stream`) | 32-bit exact unpack path; split and rings unchanged |
| `dtype` uint16, uint8 | scheme-change inside compute (regime `exact16_8_stream`) | bit-exact narrow path |
| `dtype` bfloat8_b | never on this family (family fact) | refused by `validate()` |

**Structural impossibilities.** None found beyond the specification's INVALID.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | tilize / untilize: one block = (tile-row, column block) of 32 rows × `bw` tiles. identity: one page |
| Grid | `device.compute_with_storage_grid_size()` (8×4 = 32 on this target) |
| Per-core work | `ttnn.split_work_to_cores(grid, num_units, row_wise=True)`: contiguous unit ranges, `(unit_start, num_units_this_core)` per cluster as runtime args. `num_units = total_tile_rows · n_col_blocks` (tilize/untilize) or `num_pages` (identity). Only clusters with work are `target_nodes` |
| Remainder | `split_work_to_cores` group 1 / group 2 (counts differ by 1). Tile geometry is alignment-aware: `Ht = ceil(H/32)` per image, `total_tile_rows = num_images · Ht` (never `floor(num_images·H/32)`), `tensor_w_tiles = ceil(W/32)`, `valid_rows = min(32, H - 32·h)` per tile-row |
| Bound | **data movement**: per tile, 2 KiB crosses DRAM in and 2 KiB out, against one unpack-tilize (or unpack) → datacopy → pack pass with no SFPU and no second pass. This is the target page's "a layout change" case |
| Thread plan | **(1, 1, 1)**, host constant `THREAD_PLAN = (1, 1, 1)` in the op's program-artifacts module, passed to each `KernelSpec(num_threads=...)` and to the kernels as compile-time args `reader_threads`, `compute_threads`, `writer_threads`. Kernels size their loops from these and from `get_my_thread_id()`. Reason: data-movement bound; the page's plan for that bound. R = W = 1 divides C = 1 |
| Unit dealt to compute threads | the tile-row block (`bw` ring entries), larger than one entry. With C = 1 there is no dealing and no padded round. A plan with C > 1 must use BLOCKED rings (`block_size = bw`) and deal whole rounds of C blocks to clusters (lamp *Thread plan*) |
| Tests the plan needs | (1) A test that reads the plan off the built `ProgramSpec`: every `KernelSpec.num_threads` equals `THREAD_PLAN`. With one thread per kind, a missing thread leaves the whole output unwritten, so the exact golden check covers the rest. (2) One device session: tt-metal `ttnn.to_layout` (another program) → our `to_layout` (each built regime) → tt-metal `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)` on our TILE output and `ttnn.to_layout(output, ttnn.TILE_LAYOUT)` on our ROW_MAJOR output, values checked after each step. (3) Our tilize then our untilize then our identity back to back in one session |
| Regime selection | `identity_copy` iff `input.layout == layout`; else `tilize_stream` iff `layout == TILE`; else `untilize_stream`. **Regime-pinned tests** (all four (layout, target_layout) pairs, both memory configs) are the PHASE0 cells of the acceptance / golden suite. Grid-independent: the predicate reads only layouts |
| `bw`-selection tests | a host unit test of `select_block_width_tiles` on the worked table above (grid 8×4). Golden cells exercise both a column-split case (`bw < tensor_w_tiles`: (1,1,400,512), (1,1,32,4096)) and a full-width case ((2,4,128,512)) |

## Circular Buffers

Ring names are the `DataflowBufferSpec.unique_id`s and the kernels' binding names. Only one regime's rings exist in a program. `tile_bytes = input_tensor.buffer_page_size()` of the TILE-side tensor (bf16 32×32: 2048 B). Capacity is per NEO × C, with C = 1.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|----------|
| `cb_row_block` (tilize) | 0 | `tile_bytes` | `BUFFER_DEPTH · block_height_tiles · bw` | spans `tile_col` (bw) and `row_in_tile` (32 rows of a block = `bw` entries); streams over `tile_row`, `page` | bfloat16 (`input_tensor.dtype`) | reader | compute | producer **explicit** (opted out, trid 0); consumer compute | whole program |
| `cb_tile_block` (tilize) | 1 | `tile_bytes` | `BUFFER_DEPTH · block_height_tiles · bw` | spans `tile_col`, `row_in_tile`; streams over `tile_row` | bfloat16 | compute | writer | consumer **implicit** (tagged writes, trid handed back) | whole program |
| `cb_tile_block_in` (untilize) | 0 | `tile_bytes` | `BUFFER_DEPTH · block_height_tiles · bw` | spans `tile_col`, `row_in_tile`; streams over `tile_row` | bfloat16 | reader | compute | producer **implicit** (tagged reads, trid handed back) | whole program |
| `cb_row_block_out` (untilize) | 1 | `tile_bytes` | `BUFFER_DEPTH · block_height_tiles · bw` | spans `tile_col`, `row_in_tile`; streams over `tile_row` | bfloat16 | compute | writer | consumer **explicit** (opted out, trid 0) | whole program |
| `cb_pages` (identity) | 0 | `input_tensor.buffer_page_size()` | `BUFFER_DEPTH · copy_block_pages` | spans `page` window; streams over `page` | none (DM-only ring: metadata omitted) | reader | writer | both sides **implicit** | whole program |

Push count = wait count: tilize reader pushes `bw` per block, helper waits/pops `bw` per tile-row (`block_height_tiles = 1`), helper pushes `bw` per tile-row, writer drains `bw` entries per block. Untilize: reader posts `bw` per block, helper waits `bw` (or `sub_block_width` × `num_sub_blocks = bw`) and pushes `bw`, writer waits/pops `bw`. Every quantum is `bw` and capacity is `2·bw`, so the ring cycle closes.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_row_block` | 32 rows × `chunk_bytes` | raw NoC (see API Mapping) | DRAM/L1 sticks | `cb_row_block`, `bw` (reserved → filled + padding zeroed → pushed) | `bw` posted |
| 2 | `tilize_block` | 1 × `bw` tiles | yes: `compute_kernel_lib::tilize` | `cb_row_block`, `bw`, waited then popped by the helper | `cb_tile_block`, `bw` | input freed, output posted |
| 3 | `store_tile_block` | `bw` tiles | raw NoC (implicit) | `cb_tile_block`, `bw` | DRAM/L1 tile pages | entries acked by DM0 ISR |
| 4 | `load_tile_block` | `bw` tiles | raw NoC (implicit) | DRAM/L1 tile pages | `cb_tile_block_in`, `bw` | posted by DM0 ISR |
| 5 | `untilize_block` | 1 × `bw` tiles → 32 rows × `chunk_bytes` | yes: `compute_kernel_lib::untilize` | `cb_tile_block_in`, `bw` | `cb_row_block_out`, `bw` | input freed, output posted |
| 6 | `store_row_block` | 32 rows × `chunk_bytes` | raw NoC (explicit) | `cb_row_block_out`, `bw`, waited, then popped after the write barrier | DRAM/L1 stick chunks (`valid_rows` of them) | freed |
| 7 | `copy_page_block` | `copy_block_pages` pages | raw NoC (implicit both sides) | DRAM/L1 pages → `cb_pages` | `cb_pages` → DRAM/L1 pages | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| all compute | helper prerequisite | `compute_kernel_hw_startup(icb, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:113` | (`cb_row_block`, `cb_tile_block`) / (`cb_tile_block_in`, `cb_row_block_out`) | — | — | — |
| `tilize_block` | helper | `compute_kernel_lib::tilize<block_width_tiles, input_dfb, output_dfb>(in, out, num_blocks)` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:194-207`; loop `tilize_helpers.inl:214-239`; Quasar path `tt_metal/hw/inc/api/compute/tilize.h:67-79` (Quasar init incl. `llk_pack_init(ocb)`), `:445-480` (fast forwards to plain) | `<bw, cb_row_block, cb_tile_block>`, defaults (`InitAndUninit`, `WaitBlock`, `UnpackAndPackReconfigure`, `Fp32Mode::Fast` — bf16 only), symmetric form (no `total_input_pages`); `num_blocks = num_units_this_core · block_height_tiles` | `cb_row_block` | `cb_tile_block` | `block_width_tiles` (template), `num_blocks` (runtime) |
| `untilize_block` | helper | `compute_kernel_lib::untilize<block_width_tiles, input_dfb, output_dfb>(in, out, num_blocks)` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:152-163`; dispatch `untilize_helpers.inl:78-87, 241-271` | `<bw, cb_tile_block_in, cb_row_block_out>`, defaults; `num_blocks` as above | `cb_tile_block_in` | `cb_row_block_out` | `block_width_tiles`, `num_blocks` |
| `load_row_block` | raw_api | `Noc::async_read<NocOptions::TXN_ID>(acc, dfb, size, {.page_id, .offset_bytes}, {.offset_bytes}, {.trid = 0})`, `Noc::async_read_barrier<NocOptions::TXN_ID>({.trid = 0})`, `DataflowBuffer::reserve_back/push_back`, `Noc::async_write_zeros(dfb, size, {.offset_bytes})` + `Noc::write_zeros_l1_barrier()` | `tt_metal/hw/inc/api/dataflow/noc.h:246-258` (read), `:738-739` (barrier), `:847-848`, `:904` (zeros), `:22-23` (`DataflowBufferArgs.offset_bytes`); pattern `.claude/references/quasar_op_examples/toy_thread_plan/kernels/reader.cpp` | `chunk_bytes`, src offset `col_block·chunk_bytes`, dst offset `r·chunk_bytes` | DRAM/L1 sticks | `cb_row_block` | `bw` (via `chunk_bytes`, reserve count) |
| `store_tile_block` | raw_api | `Noc::async_write<NocOptions::TXN_ID>(dfb, acc, {}, {.page_id})` (implicit); `noc_async_write_set_trid(0, noc.get_noc_id())` at end | `noc.h:930-941`; `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2579`; pattern `toy_thread_plan/kernels/writer_implicit.cpp` | page id per tile | `cb_tile_block` | DRAM/L1 tiles | `bw` (tiles per block) |
| `load_tile_block` | raw_api | `Noc::async_read<NocOptions::TXN_ID>(acc, dfb, {.page_id}, {})` (implicit); `noc_async_read_set_trid(0, noc.get_noc_id())` at end | `noc.h:915-927`; `dataflow_api.h:2558`; pattern `toy_thread_plan/kernels/reader_implicit.cpp` | page id per tile | DRAM/L1 tiles | `cb_tile_block_in` | `bw` |
| `store_row_block` | raw_api | `DataflowBuffer::wait_front/pop_front`, `Noc::async_write<NocOptions::TXN_ID>(dfb, acc, size, {.offset_bytes}, {.page_id, .offset_bytes}, {.trid = 0})`, `Noc::async_write_barrier<NocOptions::TXN_ID>({.trid = 0})` | `noc.h:404-416` (write), `:759-760` (barrier); pattern `toy_thread_plan/kernels/writer.cpp` | `chunk_bytes`, src offset `r·chunk_bytes`, dst offset `col_block·chunk_bytes` | `cb_row_block_out` | DRAM/L1 sticks | `bw` |
| `copy_page_block` | raw_api | implicit read + implicit write as above, trid hand-back in both kernels | `noc.h:915-941` | page id per page | DRAM/L1 pages | DRAM/L1 pages | `copy_block_pages` (ring capacity) |

**Helpers considered and rejected (raw_api rows):**

| Raw block operation | Candidate helper | Mismatch (file:line) | Concrete reason |
|---------------------|------------------|----------------------|-----------------|
| `load_row_block` | `dataflow_kernel_lib::read_sticks_for_tilize<cb_id, TILE>` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.inl:117-127` (`cb_reserve_back(cb_id)`, `get_write_ptr(cb_id)`, untagged `noc_async_read`, untagged `noc_async_read_barrier()`), `:105` (`get_local_cb_interface`) | It uses the Metal 1 CB API and untagged NoC transfers. On Quasar an explicit-sync ring's transfers must carry a user trid (target page *Ring protocols*). It also leaves padding rows stale, where this design zeroes them. Its stick walk treats `total_num_rows` as one tall image (`:110-115`), so per-image padding needs one call per unit anyway |
| `store_row_block` | `dataflow_kernel_lib::write_sticks_after_untilize<cb_id>` | `tilize_helpers_dataflow.inl:229-239` (`cb_wait_front(cb_id)`, untagged `noc_async_write`, untagged `noc_async_write_barrier()`, `cb_pop_front`) | same protocol mismatch: untagged transfers on Quasar |
| `load_tile_block`, `store_tile_block`, `copy_page_block` | none in `kernel_lib` for a per-page implicit-sync transfer | — | the implicit-sync `Noc` overload is itself the one-call-per-entry primitive; there is no block helper to wrap |
| stick zeroing | `dataflow_kernel_lib` local-copy helpers (`local_copy_helpers_dataflow.hpp:46-150`) | they issue self-aimed NoC reads, which need a zero source buffer | `Noc::async_write_zeros` zeroes in place with no source buffer and no extra ring |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Transaction tags | Each program has one explicit-sync side (tilize reader, untilize writer) and one implicit side, and the op is itself followed by tt-metal programs (`ttnn.to_layout` is the canonical follower) | Explicit kernels tag **every** transfer with trid 0 and barrier on it. Their ring is listed in that kernel's `disable_dfb_implicit_sync_for`. Implicit kernels end with `noc_async_read_set_trid(0, …)` / `noc_async_write_set_trid(0, …)`. The tilize reader also ends with `noc_async_write_set_trid(0, …)`, because `async_write_zeros` reprograms cmd buffer 0's trid (`noc_zero_l1.inl:68`). Session tests in Work Distribution |
| `async_write_zeros` puts cmd buffer 0 in zero mode | Any NoC write on that hart before `write_zeros_l1_barrier()` would be corrupted (`noc.h:832-838`) | Only the tilize reader zeroes, and it issues reads only. Zero → `write_zeros_l1_barrier()` → read barrier → `push_back`, all before the next block |
| Quasar tilize width = init width | `llk_unpack_tilize_api.h:67-68`: a narrower block cannot run on a wider init | one `bw` per program, `bw | tensor_w_tiles` (Mechanism caps) |
| Per-image tile padding | `(4,8,47,256)`: each of 32 images has 2 tile-rows, the second with 15 valid rows. A global `floor(N·C·H/32)` walk would misplace every image after the first | `tile_row → (image, h)` with `Ht = ceil(H/32)`, stick page `image·H + 32·h + r`, `valid_rows = min(32, H - 32h)` |
| TILE output padding content | spec leaves it unspecified; tt-metal's op pads with 0 (`TT_METAL.fixed.pad_value = 0.0`) | the reader zeroes padding rows in L1, so the tile output's padding is 0 and deterministic |
| Fidelity | set on every compute kernel | every compute `KernelSpec`: `ComputeHardwareConfig(fpu_math_fidelity=ttnn.MathFidelity.LoFi, enable_32_bit_dest=False)`. A datacopy has no multiply, so fidelity cannot change a value |
| R1 exactness through SrcA | bf16 normal values pass the unpacker → 16-bit DEST → packer bit-exact. bf16 denormals / NaN payloads may be canonicalized by SrcA. The golden inputs are standard-normal (`NUMERIC["inputs"]`) and special values are UNRESOLVED in the spec | Phase 0 bf16 only. 32-bit and narrow-integer dtypes get their own exact regimes (deferred rows), never this path |
| DM → DM ring in `identity_copy` | the Quasar examples only show DM ↔ compute rings. DM → DM STRIDED is not refused by validation (`dfb.cpp:636-650` refuses only DM self-loops), but it is unexercised in the references | the identity regime is tested in its own session test. If the runtime refuses implicit sync on a DM → DM ring, both kernels switch to explicit sync (each ring side opted out, trid 0 on every transfer). The scheme is unchanged |
| Helper `num_blocks > 0` assert / idle clusters | (1,1,32,64) has one unit | `target_nodes` = only clusters with work (from `split_work_to_cores`) |
| L1-resident input/output plus rings | R3 / R5: in L1 cells the tensors occupy L1 banks beside the rings | rings ≤ 128 KiB on every PHASE0 shape. The budget predicate subtracts `L1_RESERVE_BYTES` (host constant, 1 MiB) from the queried unreserved size |
| One dispatch per invocation | the same-layout call is tempting to short-circuit | `identity_copy` dispatches one program. Every path calls `ttnn.generic_op` exactly once, after `validate()` |
| Validation order | X1 expects a non-`ttnn.Layout` `layout` to be refused. R2 requires block-float + `ROW_MAJOR` target to be refused in every phase. R6 requires gating on `memory_axes` | entry point: `isinstance(layout, ttnn.Layout)` else `TypeError` → `validate()`: axes `dtype`, `layout` (input), `target_layout` (argument), `eval.memory.memory_axes(input_tensor)` (`memory_config`, `sharding`), then `INPUT_TAGGERS = AXES` from the spec (`alignment`, `rank`) → explicit R2 check (block-float dtype with `target_layout = ROW_MAJOR` → `NotImplementedError`) → SUPPORTED gate (`NotImplementedError` / `UnsupportedAxisValue`) |
