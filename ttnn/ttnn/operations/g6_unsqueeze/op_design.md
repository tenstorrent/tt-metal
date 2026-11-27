# Operation Design: g6_unsqueeze

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement |
| Goal | Return `input_tensor` with a new last axis of extent 1. On tiles the logical row (row 0 of each input tile) becomes column 0 of the matching output tile, every value moved bit for bit, in one device program with one data-movement kernel and no compute kernel. |
| Math | `out[i_0, ..., i_{r-1}, 0] = x[i_0, ..., i_{r-1}]`; on tiles, for tile `t` and value `c` in `0..31`: `OutTile_t[c, 0] = InTile_t[0, c]` |
| References | `docs/explorations/g6_unsqueeze/spec/g6_unsqueeze.txt` (contract), `docs/explorations/g6_unsqueeze/math.md` (F1, F2), `docs/explorations/g6_unsqueeze/tensix.md` (tile byte layout, candidate primitives), `docs/explorations/g6_unsqueeze/risks.md`, `eval/golden_tests/g6_unsqueeze/feature_spec.py`, `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp`, `ttnn/ttnn/operations/examples/master.md` (width_split, double_buffer, noc_placement, read_coalescing) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | float32, TILE, DRAM interleaved, rank 1 to 3, shape `(1, ..., 1, W)`, `W >= 1` | none | buffer address: common RT arg |
| `dim` | Python `int` (not `bool`), positional or keyword `dim=` | yes | `r` or `-1` (`r` = input rank) | none | host only; the kernel never sees it |
| `max_cores` | host module constant | n/a | `1 .. grid_x * grid_y` | `grid_x * grid_y` of `device.compute_with_storage_grid_size()` | host only (shapes the core rectangle) |
| `L1_BUDGET_BYTES` | host module constant | n/a | contract value | `1347776` | host only (bounds the staging block) |
| `block_w_tiles` | derived on host by `_plan()` | n/a | `1 .. max_single_block_tiles` | see Blocking Model | CT |
| `staging_depth_blocks` | derived on host by `_plan()` | n/a | `1` or `2` | see Blocking Model | CT |
| `start_tile`, `core_w_tiles` | derived on host by `_plan()`, per core | n/a | `start_tile + core_w_tiles <= tensor_w_tiles` | n/a | unique RT args |
| tile geometry (`tile_bytes`, `face_bytes`, `face_row_bytes`, `element_bytes`) | derived from the input dtype | n/a | float32: 4096, 1024, 64, 4 | n/a | CT |

The public signature is exactly `g6_unsqueeze(input_tensor, dim)`. It has no `memory_config`, `dtype` or `compute_kernel_config`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d_0, ..., d_{r-1})`, `r` in 1..3, every `d_i` except the last equals 1, `W = d_{r-1} >= 1` |
| Dtype | `ttnn.float32` |
| Layout | TILE: one tile row of `tensor_w_tiles = ceil(W / 32)` tiles. Only row 0 of each tile is logical; rows 1..31, and columns `>= W mod 32` of the last tile, are padding and may hold any bits (NaN included). |
| Memory | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `(d_0, ..., d_{r-1}, 1)`, i.e. `list(input_tensor.shape) + [1]` |
| Dtype | the input dtype (float32) |
| Layout | TILE: one tile column of `tensor_w_tiles` tiles. Only column 0 is logical; tile `t` holds `x[32 t + c]` at row `c`, column 0. |
| Memory | `input_tensor.memory_config()` (DRAM interleaved); a new buffer allocated with `ttnn.allocate_tensor_on_device`, never the input's buffer |

## Blocking Model

The work unit is one tile pair: input tile `t` feeds output tile `t` and nothing else (math.md F2, valid because the input is a single logical row). No result spans any axis, so no combine exists, no operand is shared, and every split moves the same DRAM bytes. The split ranking therefore reduces to occupancy and transaction count. The extent ranking prices the fixed per-block costs: one read barrier, one fence, one store issue round.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `w_tiles`: input tile columns, which are also output tile rows (`tensor_w_tiles = ceil(W/32)`) | independent: output tile `t` reads input tile `t` only (math.md F2) | `block_w_tiles` | `min(core_w_tiles_max, max_single_block_tiles)` = the whole per-core assignment as one block. For p1 and p2 this is 1, because 16 and 8 tiles are spread one per core. That 1 is not a collapsed knob: it is the per-core assignment itself. | `_plan()` on the host: one value, passed as CT arg `block_w_tiles` | spread over a row-major core rectangle with `ttnn.split_work_to_cores(rect, tensor_w_tiles, row_wise=True)`. Phase 0 uses `min(tensor_w_tiles, max_cores)` cores. | knob-turn (core count, block size) |
| `n_rows`: input logical rows, leading dims folded | independent, but its extent is pinned to 1 by the contract: every dim except the last is 1, and the host refuses anything else | `block_n_rows` | 1: the contract domain, not a choice. A second row is a different tile map (math.md: for (2, 40), output tile 1 reads two input rows). | host validation (`validate()`): not a kernel arg | not assigned (extent 1) | scheme-change, and it needs a contract revision |
| `lane`: value index `c` inside a tile (row 0 column `c` → column 0 row `c`) | independent: each output value reads one input value | `block_lane_values` | 32, the whole tile row. The move is per whole tile. | fixed by the tile geometry (CT tile constants) | not split across cores. The sub-tile split is the `face_half_units` regime row, `deferred`. | scheme-change (`face_half_units`) |
| `new_axis`: the appended axis | no data along it (extent 1) | n/a | 1 | output shape on the host | not assigned | none |

The stage the scheme introduces is the in-L1 row-to-column transform. Its data is the staging page. It has the same `w_tiles` axis as the input, and that axis is assigned to the same core as its load: there is no collector core and no stage that runs on one core while others wait.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_output_tiles` (staging pages) | `staging_depth_blocks` | 1 when every core has a single block (`num_blocks_this_core == 1`, which holds for p1 and p2). 2 when any core has more than one block. | At depth 2, the loads of block `b+1` are issued before block `b` is transformed and stored, so block `b`'s transform and store run in the DRAM-read shadow of block `b+1`. At depth 1 there is nothing to overlap because there is one block. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DRAM read alignment: `NOC_DRAM_READ_ALIGNMENT_BYTES` = 64 on Blackhole (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`), 32 on Wormhole (`.../wormhole/noc/noc_parameters.h:295`). The DRAM→L1 destination must also be DRAM-aligned (`tt_metal/impl/allocator/l1_banking_allocator.cpp:218-221`). | the row-segment read: one face row = `FACE_WIDTH * element_bytes` = 64 B for float32, at tile offsets 0 and `face_bytes` (1024) | The read size is `align_up(face_row_bytes, NOC_DRAM_READ_ALIGNMENT_BYTES)`, landing at the same tile offsets in a staging page whose base is DRAM-aligned. For float32 on BH and WH this is exactly 64 B, so no clamp fires. | A sub-alignment read returns wrong bytes, or the watcher NoC sanitizer hangs (`tt_metal/hw/inc/internal/debug/sanitize.h:523-528`). |
| NoC L1 alignment, 16 B (`noc_parameters.h:375-376`) | intra-tile word placement: value `c < 16` moves from byte `4c` to `64c`; value `c >= 16` from `1024 + 4(c-16)` to `2048 + 64(c-16)` | Not clampable. The row-to-column move is done by RISC-V word loads and stores, never by the NoC. | A NoC-based scatter cannot place a value whose source and destination differ mod 16 (every `c mod 4 != 0`). |
| L1 budget, `L1_BUDGET_BYTES` = 1347776 (contract, BB-05) | `block_w_tiles * staging_depth_blocks` | `max_single_block_tiles = floor(L1_BUDGET_BYTES / tile_bytes)` (329 for float32). If `core_w_tiles_max <= 329`: `block_w_tiles = core_w_tiles_max`, depth 1. Otherwise `block_w_tiles = floor(L1_BUDGET_BYTES / (2 * tile_bytes))` (164), depth 2. | The CB allocation fails at launch for large W, or the op overlaps the model's own L1 tensors. |
| In-place transform disjointness (element size 4 B) | `lane`: the move happens in the destination page | Valid only for `element_bytes == 4`. Sources {4..60} ∪ {1024..1084} are disjoint from destinations {64..960} ∪ {2048..3008}; value 0 is in place (offset 0 → 0). Another element size requires this check to be redone. | Overlapping source and destination give order-dependent wrong values. |

### Regimes

The named boundary is DRAM. The minimum is that each logical input value crosses once and each output tile crosses once. Because of the 64 B read alignment, the input's logical values cross as `2 * tensor_w_tiles` face-row segments of 64 B. Because column 0 lies at a 64 B stride under a 16 B write alignment, the output crosses as whole tiles.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `row_gather_word_scatter`: read the two row-0 face-row segments of each tile into the output staging page at their own tile offsets; move value `c` to column 0 row `c` with RISC-V word copies in place; write the whole tile | **built** | Domain (host-checked in `validate()`): float32, TILE, DRAM interleaved, rank 1..3, single logical row, dim ∈ {r, -1}. Every legal call lands here. | `block_w_tiles` tiles along `w_tiles` × one tile row (input) / one tile column (output) × 32 lanes | Input: minimum. `tensor_w_tiles × 128 B` (p1 2048 B, p2 1024 B); padding rows 1..31 are never read. Output: `tensor_w_tiles × 4096 B`, which is 2× the bytes of the column-holding faces 0 and 2, paid to write each tile as one transaction (lamp `write_shape`). Cross-core traffic: 0. | Per block: one read barrier (with its BH `fence`, `dataflow_api.h:1932`), one store-ordering fence, and one flush when a slot is reused. Per kernel: TensorAccessor setup and RT-arg reads. A block of the whole per-core assignment pays each once per core. Production: 1 block per core, so each is paid once per core. |
| `compute_transpose`: unpack each tile to DEST, `transpose_dest`, pack | **rejected**, superseded by `row_gather_word_scatter` | n/a | one whole tile per pass through unpack/math/pack | Input: whole tiles, `tensor_w_tiles × 4096 B` (32× the built regime), because the unpacker reads whole tiles. Output: same as built. | n/a. Rejected because bit exactness is unverified: the face handling of the 32-bit DEST route (`tt_metal/hw/inc/api/compute/transpose.h:123-134`, tensix.md "UNVERIFIED"), and NaN-payload / subnormal pass-through in the unpacker and packer. It also adds a compute kernel (3 TRISC binaries per core, BB-18) to move 32 words per tile. The built regime is bit-exact by construction: integer loads and stores, no format path. |
| `per_value_dram_writes`: scatter each value straight into the output tile in DRAM | **rejected**, superseded by `row_gather_word_scatter` | n/a | one value | Output: 32 write transactions of 4 B per tile (512 for p1) instead of 1. Each value still needs 16 B-aligned L1 staging. | n/a. Transaction count is the cost at this size (double_buffer report: ~110–125 ns per transaction per core). |
| `face_half_units`: split each tile's 32 lanes into two face halves on two cores (core A: values 0..15 → face 0; core B: values 16..31 → face 2) | **deferred** | would apply when `tensor_w_tiles < max_cores / 2` | half a tile: 1 segment read of 64 B, 15 or 16 word moves, 1 write of 1024 B | Same input bytes. Output 2048 B per tile instead of 4096, spread over 2× the cores. Cross-core: 0. | Deferred for a positive reason: at one tile per core the critical path is two DRAM round trips (read, then write ack), and halving the 31 word moves does not shorten either. It doubles the launched cores and the per-core RT args. Reachable: `row_to_column_block` takes its lane range `[c_begin, c_end)` and the store takes its byte range, so the regime is a host-plan change plus two kernel args. |

There is only one built regime, so no regime-selection function and no regime-pinned tests are needed.

### Traffic ranking

| Candidate split | DRAM bytes (p1 / p2) | Cross-core traffic | Transactions per core (p1, 1 tile per core) | Rank |
|-----------------|----------------------|--------------------|---------------------------------------------|------|
| `w_tiles` across cores, whole-tile store (**chosen**) | read 2048 / 1024, write 65536 / 32768 | 0 | 2 reads + 1 write | 1 (tie on bytes with the next row, fewer transactions) |
| `w_tiles` across cores, face-0/face-2 store | read 2048 / 1024, write 32768 / 16384 | 0 | 2 reads + 2 writes | 1 on bytes. Same split; only the write shape differs, so it is lamp `write_shape`, not a separate split. |
| `lane` split within a tile (`face_half_units`) | read 2048 / 1024, write 32768 / 16384 | 0 | 1 read + 1 write on each of 2× cores | 2: 2× the cores for the same critical path |
| `n_rows` | n/a: extent 1 | 0 | n/a | not a candidate |
| dependent-axis split + combine | n/a: no axis is dependent (no result spans an axis) | n/a | n/a | not a candidate |
| one core for all tiles | same bytes | 0 | 32 reads + 16 writes + 16 serial transforms on one RISC-V | last. It serializes the transforms. Also ruled out by the multi-core Phase 0 rule and by the spec's cores rule. |

**Operand-reuse check:** the op has one operand (`input_tensor`), and it varies along `w_tiles`, the split axis. Every core reads disjoint bytes, so there is no reuse-shared operand and no broadcast row.

**Stall-shadow check:** each core has two waiting stages:

1. The read barrier. It waits on DRAM latency.
2. The final write barrier. BB-03 requires it before the kernel returns.

At one block per core (p1, p2), nothing independent of the awaited data remains. The transform needs the read data, and the store needs the transform. The only hoistable work is computing the output NoC addresses (TensorAccessor page lookups) for the block, which is issued before the read barrier.

At more than one block per core, the loads of block `b+1` are issued before `row_to_column_block(b)` and `store_block(b)`, so both run in the read shadow of `b+1`. This is legal because blocks are independent (no FP order is involved; this is a pure move).

### Block schedule

The mover kernel (one data-movement kernel per core) executes:

```cpp
// num_blocks_this_core = ceil(core_w_tiles / block_w_tiles); the last block has last_block_extent tiles
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_row_block(block_idx);        // row 0 of block_extent input tiles -> staging slot (block_idx % staging_depth_blocks)
    row_to_column_block(block_idx);   // in place, per staging page: value c of row 0 -> row c of column 0
    store_block(block_idx);           // block_extent whole-tile writes, staging page -> output tile
}
// end of kernel: noc_async_write_barrier (BB-03)
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed-cost work |
|-----------------|-------------|--------------------|---------------------------------------|
| `load_row_block` | `block_extent` tiles; per tile, 2 reads of one face row (64 B): input tile byte offset 0 → staging page offset 0, input byte offset `face_bytes` → staging page offset `face_bytes`. For the last tile of W, the second segment is read even when it is all padding: it lands in output rows ≥ W, which are padding. | The staging slot of this block. With depth 2, the other slot holds block `b-1` until its writes are flushed. | One `noc_async_read_barrier` per block (it fences on BH, `dataflow_api.h:1932`). Reads are never barriered per tile. With depth 2, the reads of `b+1` are issued before the transform of `b`. Zone: `unsq_load_issue`; the barrier is zone `unsq_load_wait`. |
| `row_to_column_block` | `block_extent` staging pages × 31 word moves each: value `c` (1..15) from byte `4c` → `64c`; value `c` (16..31) from byte `face_bytes + 4(c-16)` → `2 * face_bytes + 64(c-16)`. Value 0 is already in place. | the staging slot | No init. One store-ordering fence per block (`invalidate_l1_cache()`, which is `asm("fence")` on BH, `tt_metal/hw/inc/internal/tt-1xx/cache.h:12-16`) between the block's last store and its first NoC write. Zone: `unsq_row_to_column`. |
| `store_block` | `block_extent` writes of one whole tile page (`tile_bytes`), staging page → output page `start_tile + block_idx * block_w_tiles + i` | the staging slot, until the writes are flushed | Writes are issued back to back without a per-tile barrier. A `noc_async_writes_flushed` is issued only before a slot is reused, so never at depth 1. One `noc_async_write_barrier` per kernel, at the end. Zones: `unsq_store_issue`, `unsq_store_wait`. |

Every zone name above is a `DeviceZoneScopedN` (`tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012`). The macro compiles to nothing in the plain build, which leaves that build unchanged (BB-21).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| grid synchronization (`core_count`) | One tile per core leaves each core with 2 reads, 31 moves and 1 write. Per-core launch and RT-arg cost may outweigh the parallel transforms (width_split report, WH: +13 ns per added core; 8 cores × 1 tile = 1919 ns vs 1 core × 8 tiles = 4292 ns). | `max_cores` set so each core gets 2 or 4 tiles (p1: 8×1 and 4×1 rectangles). Judge on unprofiled host time of blocking trace replays, not on profiled kernel duration alone (lessons.md RB-053). |
| per-core RT args (`rt_args`) | Unique RT args (`start_tile`, `core_w_tiles`) are one write per core per dispatch. The common RT args (the two addresses) are written once. | Derive `start_tile` and `core_w_tiles` in the kernel from `get_relative_logical_x/y()` (`dataflow_api.h:70-86`) plus CT args (rectangle width, `q`, `r` of the split), leaving only common RT args. |
| `write_shape` | A whole 4096 B tile per store writes 2× the bytes that hold column 0 (faces 0 and 2). | 2 writes of `face_bytes` per tile (byte offsets 0 and `2 * face_bytes`). This halves the write bytes and adds one transaction. Faces 1 and 3 of the output are then never written, which is allowed (padding). |
| `read_shape` | Two 64 B reads per tile are two transactions; at this size the cost is per transaction, not per byte (split_reader report: 64 B reads are bound by transaction count). | One 4096 B whole-tile read per tile, landing in the staging page. It also lands the input padding rows in rows 1..31, which `row_to_column_block` never reads. |
| `noc_placement` | The mover runs on RISCV_0 with NoC0. With a latency-bound 1-tile-per-core load, the NoC choice is unmeasured here (noc_placement README: the effect "nearly vanishes" at block=1). | NoC1 for the mover. Or a split: the reader on NCRISC/NoC0 and the writer on BRISC/NoC1. |
| overlap (multi-block cores only, `W > 32 * max_cores * 329`; no production case) | One block of up to 329 tiles serializes all reads → all transforms → all writes. | A smaller `block_w_tiles` at depth 2. Or per-tile transaction ids (`set_read_trid` / `async_read_barrier_with_trid`, `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:147,158`), so the transform of tile `i` starts when its own reads land. |
| dual mover (multi-tile cores only) | One RISC-V does every transform of its core. | Both DM RISC-Vs, each owning half of the core's tiles (each with its own read, transform and store) and its own NoC. |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM → L1 | float32 bytes, two 64 B face-row segments per input tile | `noc_async_read` from `TensorAccessor::get_noc_addr(page_id, byte_offset)` into the staging page, at the same tile byte offsets | No padding row of the input reaches L1. Row-0 padding columns of the last tile land in output rows ≥ W. |
| in-L1 transform | 32-bit words, raw bits | RISC-V `volatile tt_l1_ptr uint32_t` loads and stores in the staging page (in place) | Bit-exact by construction: no FPU, no unpacker or packer, no format conversion. |
| L1 → DRAM | float32 tile page (4096 B) | `noc_async_write_page(page_id, accessor, staging_page_addr)` | Output padding holds input row-0 values (row 0, columns 1..31) and stale L1 (elsewhere). It is unspecified by the contract. |
| cross-core | n/a | none | No regime, built or deferred, has inter-Tensix traffic. |

TARGET has no `memory_layout` axis. The only placement is DRAM interleaved, so there is no sharded scheme to map. The `w_tiles` split is independent: a future sharded placement along `w_tiles` would be a knob-turn, with each core reading its resident input shard through a CB backed on the shard.

Structural impossibilities: none beyond `INVALID = []`.

The entry point canonicalizes `dim`: `k = dim + r + 1` if `dim < 0`. It refuses `k != r` before the support check.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one tile pair (input tile `t` → output tile `t`). A core's block is `block_w_tiles` consecutive units. |
| Grid | `(grid_x, grid_y) = device.compute_with_storage_grid_size()`. `num_work_cores = min(tensor_w_tiles, max_cores)`. Rectangle: `rect_rows = ceil(num_work_cores / grid_x)`, `rect_cols = ceil(num_work_cores / rect_rows)`, origin (0, 0). `ttnn.split_work_to_cores(ttnn.CoreCoord(rect_cols, rect_rows), tensor_w_tiles, row_wise=True)` returns `all_cores`, `core_group_1` (`q+1` tiles), `core_group_2` (`q` tiles). |
| Per-core work | `start_tile` = the running sum over cores in row-major order. `core_w_tiles` = the group's count. `num_blocks_this_core = ceil(core_w_tiles / block_w_tiles)`. `last_block_extent = core_w_tiles - (num_blocks_this_core - 1) * block_w_tiles`. |
| Remainder | `tensor_w_tiles = ceil(W / 32)`, never `W // 32`. The host asserts `input_tensor.buffer_num_pages() == output.buffer_num_pages() == tensor_w_tiles` (a RuntimeError otherwise). Uneven splits fall to `core_group_2` with the same kernel and a runtime extent. Cores of the rectangle that `split_work_to_cores` leaves without work are not in `all_cores` and get no kernel. |
| p1 `[1, 512]` | 16 tiles → rectangle 8×2 → 16 cores × 1 tile, 1 block, depth 1, **1 core range** (BB-17) |
| p2 `[1, 256]` | 8 tiles → rectangle 8×1 → 8 cores × 1 tile, 1 block, depth 1, **1 core range** |

Both production shapes take a single rectangle. The row-wise split over the full grid would instead give 13 + 3 cores for p1 on a 13-wide grid, which is 2 ranges.

**Plan reporting (BB-08, BB-21):** the op module exposes `plan_for(input_tensor)`. It returns the regime name (`row_gather_word_scatter`), the rectangle, `all_cores` with its range count, per core (`core`, role `mover`, `start_tile`, `core_w_tiles`), `block_w_tiles`, `staging_depth_blocks`, and L1 bytes per core. `_plan()` is the single source for every derived quantity, and the descriptor builder consumes it.

**Program description cache (BB-07, BB-16):** a module-level dict, keyed by (logical input shape, padded input shape, dtype, layout, buffer type, memory layout, device id, grid size, `max_cores`), holds the built kernel and CB descriptors.

- A hit rebuilds nothing. It writes only the two buffer addresses into the kernel's `common_runtime_args`, then calls `ttnn.generic_op([input_tensor, output], descriptor)`.
- No address appears in CT args or in the key. `TensorAccessorArgs` for interleaved DRAM carries no address.
- The implementer confirms that the mutation reaches the descriptor that `generic_op` receives, because `ProgramDescriptor.kernels` may return a copy (`ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:1137`). The `cache_two_widths` golden case checks this with fresh buffers.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_output_tiles` | 0 | `tile_bytes` (4096, `ttnn.tile_size(ttnn.float32)`) | `staging_depth_blocks * block_w_tiles` | spans `w_tiles` → `block_w_tiles` (one page per tile of a block); spans `lane` → a whole tile page (the store writes the whole page); `n_rows` → 1 (domain); depth from `staging_depth_blocks` | Float32 (dataflow-only payload, never in DEST) | mover (NoC reads + transform) | mover (NoC writes) | whole kernel |

This is the op's only CB. The mover uses it as an L1 allocation addressed through `get_write_ptr(cb_output_tiles)` plus `slot * block_w_tiles * tile_bytes`, with no reserve/push/wait/pop. One thread owns it end to end. Slot reuse is ordered by `noc_async_writes_flushed` (depth 2 only). The row segments need no buffer of their own: they land at their own offsets in the destination page, and the transform runs in place (l1-footprint-discipline Rule 3, patterns 1 and 2).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_row_block` | `block_extent` tiles × 2 face-row segments | raw dataflow API (no covering helper) | input: DRAM `input_tensor` pages `start_tile + block_idx*block_w_tiles ...` | `cb_output_tiles`, `block_extent` pages of slot `block_idx % staging_depth_blocks`; bytes [0,64) and [1024,1088) of each page valid after the read barrier | staging pages hold row 0 at face-row positions |
| 2 | `row_to_column_block` | `block_extent` pages × 31 word moves | custom block operation (the gap is recorded in API Mapping) | `cb_output_tiles`, same pages | `cb_output_tiles`, same pages (in place) | column 0 rows 0..31 of each page hold the 32 lane values; fence issued |
| 3 | `store_block` | `block_extent` whole-tile writes | raw dataflow API | `cb_output_tiles`, same pages | DRAM `output` pages, same ids | slot free once flushed; final write barrier at kernel end |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_row_block` | raw_api | `noc_async_read` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | `(src_noc_addr, dst_l1_addr, size = face_row_bytes)` | DRAM input | `cb_output_tiles` | loop extent `block_extent` (from `block_w_tiles`) |
| `load_row_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | `offset` ∈ {0, `face_bytes`} | n/a | n/a | none |
| `load_row_block` | raw_api | `noc_async_read_barrier` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1921` (fence at :1932) | none | n/a | n/a | called once per block |
| `row_to_column_block` | raw_api | RISC-V word loads/stores through `volatile tt_l1_ptr uint32_t*`; the destination offsets may reuse `detail::TileColumnLayout<cb>::column_base(0)` / `row_offset(face_row, row)` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:37-57` | `element_bytes` = 4 | `cb_output_tiles` | `cb_output_tiles` | loop extent `block_extent` |
| `row_to_column_block` | raw_api | `invalidate_l1_cache()` (`fence` on BH) | `tt_metal/hw/inc/internal/tt-1xx/cache.h:12-16` | none | n/a | n/a | once per block |
| `store_block` | raw_api | `noc_async_write_page` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1418` | `(page_id, accessor, src_l1_addr)`, size = page size | `cb_output_tiles` | DRAM output | loop extent `block_extent` |
| `store_block` | raw_api | `noc_async_writes_flushed` / `noc_async_write_barrier` | `dataflow_api.h:1980` / `dataflow_api.h:1951` | none | n/a | n/a | flush only on slot reuse; barrier once per kernel |
| all | raw_api | `DeviceZoneScopedN` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012` | zone names in Block schedule | n/a | n/a | none |
| host | host | `ttnn.split_work_to_cores(CoreCoord, units, row_wise)` | `ttnn/cpp/ttnn-nanobind/operations/core.cpp:470-472` | `(CoreCoord(rect_cols, rect_rows), tensor_w_tiles, True)` | n/a | n/a | `max_cores` shapes the rectangle |

**Helpers considered and rejected**

| Raw entry | Helper candidate | File:line of the mismatch | Concrete reason |
|-----------|------------------|---------------------------|-----------------|
| `row_to_column_block` | `copy_tile_column<dfb_id, T>` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:91-102` | It copies a column to a column. Its source walk is `src + column_base(src_col) + row_offset(face_row, row)` (a stride of FACE_WIDTH per row, faces 0/2 or 1/3). This op's source is row 0: consecutive elements in faces 0 and 1. The destination half (`column_base(0)`, `row_offset`) does match and is reused. Gap: a `copy_tile_row_to_column` beside `copy_tile_column` would close it. |
| `row_to_column_block` | `read_with_state` / `local_addr` (self-aimed L1→L1 NoC copies) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57-120`; L1 alignment `noc_parameters.h:375-376` | The NoC moves 16 B-aligned chunks with equal source and destination offsets mod 16. Value `c` sits at `4c` and goes to `64c`, so for `c mod 4 != 0` no NoC transfer can place it. |
| `row_to_column_block` | compute `transpose_tile` (no kernel_lib helper; LLK API) | `tt_metal/hw/inc/api/compute/transpose.h:120-137` | This is the rejected regime `compute_transpose`. Bit exactness of the 32-bit DEST route is unverified, and it adds a compute kernel. |
| `load_row_block` / `store_block` | `tilize_helpers_dataflow.hpp`, `reduce_helpers_dataflow.hpp`, `mcast/*` | n/a | These cover tilize staging, reduce scalers and multicast. None reads or writes a sub-page DRAM segment of a tile. The op has no compute kernel, so the compute helpers (tilize, untilize, reduce, eltwise, matmul) have no phase to cover. |
| `row_to_column_block` (optional padding fill) | `zero_tile` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:52-57` | Not used: the contract leaves output padding unspecified (`docs/explorations/g6_unsqueeze/sources/local/contract.md:63`). It is the mechanism if a consumer turns out to read padding. |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| A converting path breaks bit exactness (R1) | The gate is bit equality, including NaN payloads, -0.0 and subnormals. | No compute kernel. The move is integer words through RISC-V loads and stores. Never route a value through unpack/math/pack or a float type in C++: use `uint32_t`, not `float`. |
| Input padding reaches a logical output (R3) | Rows 1..31 of each input tile hold any bits; the golden suite poisons them with NaN. | Only row 0 is read. Row-0 padding columns (≥ W mod 32 of the last tile) map to output rows ≥ W, which are output padding. |
| The input's pages are returned unchanged (R2) | `check_exact` also refuses an aliased output buffer. | The output is a fresh `allocate_tensor_on_device` buffer, and every output tile is written by `store_block`. |
| The scatter reads stale data or the NoC write sends it | On BH the DM RISC-V has a data cache. | Reads: the transform runs only after `noc_async_read_barrier`, which issues `invalidate_l1_cache()` (`dataflow_api.h:1932`). Writes: one `invalidate_l1_cache()` (`fence`) per block between the last transform store and the first NoC write. |
| A sub-alignment DRAM read | The BH DRAM read alignment is 64 B. | The segment is one face row (64 B) at tile offsets 0 and 1024, read into a page base that is DRAM-aligned (Mechanism caps). The watcher (`--dev`) sanitizer verifies it. |
| Program cache miss or collision (R8, BB-07) | p1 and p2 differ only in W. | The Python key holds the shapes. The `generic_op` hash holds core ranges and CT args (`block_w_tiles`), which differ between p1 (8×2) and p2 (8×1). Addresses live in common RT args only. |
| Untested paths: the uneven split (`core_group_2`) and multi-block cores | The production shapes give 1 tile per core and the suite holds only production cases (BB-22). | These paths are the same block operations with runtime extents (`core_w_tiles`, `last_block_extent`), not separate code. During implementation, check once with `scripts/tt-probe.sh` on, for example, `W = 32*17 + 5` (17 tiles, non-aligned) and a W that gives 2+ tiles per core. Do not add these to the tests. |
| Refused calls must not build a program | `test_refused` counts program-cache entries. | `validate()` and the domain checks run first in the entry point, before the output is allocated and before any descriptor is built. Malformed requests raise `ValueError`. Support refusals raise `UnsupportedAxisValue` (a `NotImplementedError`, therefore a `RuntimeError`). |
| BB-03: a NoC transaction is left in flight | The next program starts right after the op. | The final `noc_async_write_barrier` on every core. No posted writes. |
| Output padding is nondeterministic | Stale L1 lands in output faces 1..3, and in face 0 outside column 0. | Allowed by the contract (padding unspecified). BB-06 compares logical values. `zero_tile` is the recorded mechanism if a consumer needs zeros. |
