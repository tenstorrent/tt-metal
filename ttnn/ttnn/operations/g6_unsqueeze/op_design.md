# Operation Design: g6_unsqueeze

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement |
| Goal | Return `input_tensor` with a new last axis of extent 1, in one device program. On tiles, the logical row (row 0 of every input tile) becomes column 0 of the matching output tile. Every value is moved bit for bit by one data-movement kernel per core, and there is no compute kernel. |
| Math | `out[i_0, ..., i_{r-1}, 0] = x[i_0, ..., i_{r-1}]`. On tiles, for tile `t` and lane `c` in `0..31`: `OutTile_t[c, 0] = InTile_t[0, c]` (math.md F2) |
| References | `docs/explorations/g6_unsqueeze/spec/g6_unsqueeze.txt` (contract), `docs/explorations/g6_unsqueeze/math.md` (F1, F2), `docs/explorations/g6_unsqueeze/tensix.md` (tile byte layout, candidate primitives), `docs/explorations/g6_unsqueeze/risks.md`, `docs/explorations/g6_unsqueeze/lessons.md` (RB-036, RB-053), `eval/golden_tests/g6_unsqueeze/feature_spec.py`, `eval/golden_tests/g6_unsqueeze/helpers.py`, `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp`. Catalog entries: `width_split` (core-count knob), `noc_placement` (row placement, NoC per direction), `tile_reorder` (whole-page writes), `double_buffer` (one barrier per block). |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | float32, TILE (32×32 tiles), on device, DRAM interleaved, rank 1 to 3, shape `(1, ..., 1, W)` with `W >= 1` | none | buffer address: common RT arg |
| `dim` | Python `int` (a `bool` is refused), positional or keyword `dim=` | yes | `r` or `-1`, where `r` is the input rank | none | host only; the kernel never sees it |
| `MAX_CORES` | host value | n/a | `1 .. grid_x * grid_y` | `grid_x * grid_y` from `device.compute_with_storage_grid_size()` | host only (it shapes the core rectangle) |
| `L1_BUDGET_BYTES` | host module constant | n/a | contract value | `1347776` | host only (it bounds the staging block) |
| `block_w_tiles` | derived on host by `_plan()` | n/a | `1 .. 329` | see Blocking Model | CT |
| `staging_depth_blocks` | derived on host by `_plan()` | n/a | `1` or `2` | see Blocking Model | CT |
| `start_tile`, `core_w_tiles` | derived on host by `_plan()`, per core | n/a | `start_tile + core_w_tiles <= tensor_w_tiles`; `core_w_tiles = 0` on idle tail cores | n/a | unique RT args |
| tile geometry: `tile_bytes`, `face_bytes`, `face_row_bytes` | derived from the input dtype and tile | n/a | float32: 4096, 1024, 64 | n/a | CT |

The public signature is exactly `g6_unsqueeze(input_tensor, dim)`. It has no `memory_config`, `dtype` or `compute_kernel_config`.

The module `ttnn/ttnn/operations/g6_unsqueeze/__init__.py` exports `g6_unsqueeze`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate` and `plan_for`. `eval/golden_tests/g6_unsqueeze/axes.py` imports `INPUT_TAGGERS` from the package.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d_0, ..., d_{r-1})`, `r` in 1..3. Every `d_i` except the last equals 1, and `W = d_{r-1} >= 1` (one logical row). |
| Dtype | `ttnn.float32` |
| Layout | TILE with 32×32 tiles. One tile row of `tensor_w_tiles = ceil(W / 32)` tile pages. Only row 0 of each tile is logical. Rows 1..31 of every tile, and columns `>= W mod 32` of the last tile when `W mod 32 != 0`, are padding and may hold any bits, NaN included. |
| Memory | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | `(d_0, ..., d_{r-1}, 1)` = `list(input_tensor.shape) + [1]` |
| Dtype | the input dtype (float32) |
| Layout | TILE: one tile column of `tensor_w_tiles` tile pages. Only column 0 is logical. Output tile `t` holds `x[32 t + c]` at row `c`, column 0. |
| Memory | `input_tensor.memory_config()` (DRAM interleaved), in a new buffer from `ttnn.allocate_tensor_on_device`, never the input's buffer |

## Blocking Model

The work unit is one tile pair: input tile `t` feeds output tile `t` and nothing else. This follows from math.md F2 and holds because the input is a single logical row. No result spans any axis, so the op has no combine and no shared operand, and every split moves the same DRAM bytes. The split ranking therefore reduces to which axis can be cut at all, plus transaction count. The extent ranking prices the per-block fixed costs, which are one read barrier and one slot flush per block.

Inside a tile the move has a uniform structure. Input row 0 consists of two face-row segments of 64 B: lanes 0..15 at tile byte 0 (face 0) and lanes 16..31 at byte `face_bytes` (face 1). Output column 0 lies in face 0 (lanes 0..15) and face 2 (lanes 16..31), at a 64 B stride.

The design lands each input segment at the base of the output face that receives it:

- segment 0 → staging byte 0
- segment 1 → staging byte `2 * face_bytes`

Both halves then become the same operation, relative to a face base `F`: move word `j` from byte `F + 4j` to byte `F + 64j`, for `j = 1..15`. Word 0 is already in place. Every source lies in face row 0 and every destination in face rows 1..15, so the sets are disjoint and the order of the moves does not matter.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `w_tiles`: input tile columns, which are also output tile rows (`tensor_w_tiles = ceil(W/32)`) | independent: output tile `t` reads input tile `t` only (math.md F2) | `block_w_tiles` | the whole per-core assignment as one block: `block_w_tiles = core_w_tiles_max` while `core_w_tiles_max <= 329`; balanced blocks of at most 164 at depth 2 beyond that (Mechanism caps). For p1 and p2 this is 1. That 1 is not a collapsed knob: it is each core's entire assignment, because 16 (p1) and 8 (p2) tiles are spread one per core. | `_plan()` on the host, passed as CT arg `block_w_tiles`. The kernel derives `num_blocks_this_core` and `last_block_extent` from it and the RT arg `core_w_tiles`. | spread over one row-major core rectangle with `ttnn.split_work_to_cores(CoreCoord(rect_cols, rect_rows), tensor_w_tiles, row_wise=True)`. `num_work_cores = min(tensor_w_tiles, MAX_CORES)`. | knob-turn: core count via `MAX_CORES`, block size via `block_w_tiles` |
| `lane`: value index `c` inside a tile (row 0, column `c` → column 0, row `c`) | independent: each output value reads one input value | `block_lanes` | 32, the whole tile row, as two face halves of 16 lanes each. The store writes whole tile pages, so a block always covers both halves. | tile geometry (CT `face_bytes`, `face_row_bytes`) | not split across cores. Cutting one tile across two cores is the `face_half_units` regime row (`deferred`). | scheme-change (`face_half_units`) |
| `n_rows`: input logical rows, leading dims folded | independent in the math, but its extent is pinned to 1 by the contract: every dim except the last is 1, and the host refuses anything else | `block_n_rows` | 1. This is the contract domain, not a choice. With a second row the tile map changes: for `(2, 40)`, output tile 1 reads two input rows (math.md). | `validate()` on the host; not a kernel arg | not assigned (extent 1) | scheme-change that needs a contract revision |
| `new_axis`: the appended axis | no data along it (extent 1) | n/a | 1 | output shape on the host | not assigned | none |

**Stages the scheme introduces.** The only intermediate stage is the in-L1 row-to-column move. Its data is the staging page, which has the same `w_tiles` and `lane` axes as the input. Its `w_tiles` axis is owned by the core that loaded the tile. There is no collector core and no stage that runs on one core while others wait.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_output_tiles` (staging pages) | `staging_depth_blocks` | 1 when every core has one block. This holds for every `W` with `core_w_tiles_max <= 329`, and so for p1 and p2. 2 when a core has more than one block. | At depth 2, the reads of block `b+1` are issued right after block `b`'s read barrier. They are then in flight while block `b` is moved and its stores are issued. At depth 1 there is one block per core, so there is nothing to overlap. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DRAM read alignment: `NOC_DRAM_READ_ALIGNMENT_BYTES` = 64 on Blackhole (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`). Source and destination of a DRAM→L1 transfer must both be DRAM-aligned (`tt_metal/impl/allocator/l1_banking_allocator.cpp:218-221`). | The row-segment read: one face row = `face_row_bytes` = 64 B for float32, read from input tile offsets 0 and `face_bytes`, landing at staging offsets 0 and `2 * face_bytes`. | No clamp fires for float32. The segment size equals the alignment, and every offset is a multiple of 64. The CB base is allocated at DRAM alignment (`tt_metal/impl/program/program.cpp:2505-2509`). The host asserts `face_row_bytes % ttnn.get_dram_alignment() == 0` before building the descriptor. | Wrong bytes land in L1, or the watcher NoC sanitizer halts the core (`tt_metal/hw/inc/internal/debug/sanitize.h:523-528`). |
| NoC L1 alignment: 16 B (`noc_parameters.h:375-376`) | intra-tile placement of a lane: byte `F + 4j` → byte `F + 64j` | Not clampable. A NoC transfer keeps the source offset modulo 16, and `4j mod 16 != 0` for every `j mod 4 != 0`. The row-to-column move is therefore done with RISC-V 32-bit word loads and stores, never with the NoC. | A NoC-based scatter places 3 of every 4 lanes at the wrong byte. |
| L1 budget: `L1_BUDGET_BYTES` = 1347776 (contract, BB-05) | `block_w_tiles * staging_depth_blocks * tile_bytes` | `max_single_block_tiles = floor(L1_BUDGET_BYTES / tile_bytes)` = 329. If `core_w_tiles_max <= 329`: `block_w_tiles = core_w_tiles_max`, depth 1. Otherwise: `num_blocks = ceil(core_w_tiles_max / 164)`, `block_w_tiles = ceil(core_w_tiles_max / num_blocks)` (≤ 164), depth 2. The cap 164 is `floor(L1_BUDGET_BYTES / (2 * tile_bytes))`. | The CB allocation fails at launch for large `W`, or the op overlaps the model's own L1 tensors. |
| Element width: 4 B words | `lane`: one lane is one `uint32_t` | Valid for float32 only, which is the only SUPPORTED dtype. The host asserts `input_tensor.element_size() == 4` (RuntimeError). A 2-byte dtype halves `face_row_bytes` below the BH 64 B read alignment and halves the move granularity, so this row and the move pattern would have to be re-derived. | Half-word lanes land at wrong offsets, and 32 B reads violate DRAM read alignment. |
| Tile shape: 32×32 with 16×16 faces (`tt_metal/hw/inc/api/tile_constants.h:18-23`) | the face-base offsets 0 / `face_bytes` (read) and 0 / `2 * face_bytes` (land) | The host refuses any tile other than 32×32 (`input_tensor.tile.tile_shape`, ValueError). | Segments are taken from the wrong rows. |

### Regimes

The named boundary is DRAM. The minimum is that each logical input value crosses once and each output tile crosses once.

- **Input.** The logical values cross as `2 * tensor_w_tiles` face-row segments of 64 B. The minimum unit is 64 B, set by the read alignment. Input padding rows 1..31 never cross. When `1 <= W mod 32 <= 16`, the last tile's second segment holds only padding. It is still read: that costs one extra 64 B segment per call, and it lands in output rows `>= W`.
- **Output.** Column 0 lies at a 64 B stride under a 16 B write alignment, so the output crosses as pages. Whole tiles are the default write shape; perf lamp `write_shape` measures writing faces 0 and 2 only.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `row_gather_face_scatter`: read the two row-0 segments of each input tile into the staging page at the bases of output faces 0 and 2; move lanes into column 0 with RISC-V word copies in place; write the whole tile page | **built** | Every call in the domain: float32, 32×32 TILE, DRAM interleaved, rank 1..3, one logical row, `dim ∈ {r, -1}`. `validate()` checks this on the host. It is the only built regime, so it needs no selection predicate. | `block_w_tiles` tiles along `w_tiles` × 32 lanes (two face halves) × 1 row | Input: the minimum, `tensor_w_tiles × 128 B` (p1 2048 B, p2 1024 B). Output: `tensor_w_tiles × 4096 B` (p1 65536 B, p2 32768 B). That is 2× the bytes of faces 0 and 2, paid to write each tile in one transaction (lamp `write_shape`). Cross-core: 0. | Per block: one `noc_async_read_barrier`, including its BH `fence` (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:1921-1936`), and one `noc_async_writes_flushed` before a slot is reused (depth 2 only). Per kernel: RT-arg reads, two TensorAccessor constructions and one final write barrier. A block covering the whole per-core assignment pays each per-block cost once per core, and that holds for every `W` up to `32 × 329 × MAX_CORES`. |
| `face_half_units`: cut each tile's 32 lanes across two cores. Core A reads segment 0, moves 15 words in face 0 and writes face 0 (1024 B). Core B does the same for segment 1 into face 2. | **deferred** | `tensor_w_tiles * 2 <= MAX_CORES` (the grid can hold two cores per tile) | one face half: 1 read of 64 B, 15 word moves, 1 write of `face_bytes` | Same input bytes. Output `tensor_w_tiles × 2048 B`, spread over 2× the cores. Cross-core: 0. | n/a. **Deferred because** it cuts a single block across two cores (a scheme-change on `lane`), and the built regime already serves every shape this one would. The only term it shortens is the 30-word move. The two DRAM round trips per core (read latency, then write acknowledgement) stay the same, and it doubles the launched cores, which is the direction the `grid_sync` lamp already questions. **Reachable:** `row_to_column_block` is already written as a per-face-half operation on a face base `F`, and the store's byte range is a per-face page offset. The regime becomes a host-plan change (unit = face half, `start_unit`/`core_units` RT args) plus a face-range store. |
| `compute_transpose`: unpack each tile to DEST, `transpose_dest`, pack | **rejected**, superseded by `row_gather_face_scatter` | n/a | one tile per unpack/math/pack pass | Input: whole tiles, `tensor_w_tiles × 4096 B`, 32× the built regime, because the unpacker consumes whole tiles. Output: the same as built. | n/a. Rejected because bit exactness is not established. For a Float32 input CB the route is unpack-to-DEST only with `fp32_dest_acc_en` and `UnpackToDestFp32` (`tt_metal/hw/inc/api/compute/transpose.h:123-134`). Its face swap on that route is UNVERIFIED (tensix.md), and the source-register route keeps 19 bits of a float32 (`transpose.h:135-137`). It also adds a compute kernel (3 TRISC binaries per core, BB-18) and two CB handshakes to move 32 words per tile. The built regime is bit-exact by construction: integer words, no format path. |
| `per_value_dram_writes`: write each lane straight into its output position in DRAM | **rejected**, superseded by `row_gather_face_scatter` | n/a | one lane | Output: 32 write transactions per tile (16 B each, 512 for p1) instead of 1. Each value still needs a 16 B-aligned L1 source, so the in-L1 move does not go away. | n/a. It needs the same move and then multiplies the write transactions by 32. At this size the cost is per transaction (double_buffer, split_reader reports). |

A single-core plan is not a regime: it is the built regime with `MAX_CORES = 1`, a knob setting ranked last below.

### Traffic ranking

| Candidate split | DRAM bytes (p1 / p2) | Cross-core traffic | Transactions per core (p1) | Rank |
|-----------------|----------------------|--------------------|----------------------------|------|
| `w_tiles` across cores, whole-tile store (**chosen**) | read 2048 / 1024, write 65536 / 32768 | 0 | 2 reads + 1 write (16 cores × 1 tile) | 1. It ties every other split on read bytes. It needs fewer transactions than the face-store shape and half the cores of `face_half_units`. |
| `w_tiles` across cores, face-0/face-2 store | read 2048 / 1024, write 32768 / 16384 | 0 | 2 reads + 2 writes | 1 on bytes. This is the same split; only the write shape differs, so it is lamp `write_shape`, not a separate split. |
| `lane` across cores (`face_half_units`) | read 2048 / 1024, write 32768 / 16384 | 0 | 1 read + 1 write on each of 32 cores | 2: twice the cores for the same two DRAM round trips per core |
| `n_rows` | n/a: extent 1 | 0 | n/a | not a candidate |
| dependent axis split + combine | n/a: no axis is dependent | n/a | n/a | not a candidate. No result spans an axis, so no combine exists to price. |
| one core for all tiles (`MAX_CORES = 1`) | same bytes | 0 | 32 reads + 16 writes + 16 serial tile moves on one RISC-V | last. It serializes every move on one RISC-V (catalog `width_split`: 1 core × 8 tiles is 2.24× slower than 8 cores × 1 tile), and it violates the multi-core Phase 0 rule and the spec's cores rule. |

**Operand-reuse check.** The op has one operand, `input_tensor`, and it varies along `w_tiles`, the split axis. Every core reads disjoint bytes, so no operand is reuse-shared and there is no broadcast row.

**Stall-shadow check.** Each core has two waits:

1. The read barrier, which waits on DRAM read latency.
2. The final write barrier, which BB-03 requires before the kernel returns.

Nothing that is independent of the awaited data can fill either wait. The move needs the landed segments, the store needs the move, and nothing follows the final barrier. Two pieces of work are placed ahead of the read wait: the construction of the output TensorAccessor and the RT-arg reads, both issued before the first read barrier.

Zero-filling the output padding would be independent work that fits the read shadow. It is not done, because the contract leaves output padding unspecified (`docs/explorations/g6_unsqueeze/sources/local/contract.md`, section 7). If a consumer turns out to need it, the mechanism is self-aimed zero reads into staging bytes `[64, 1024) ∪ [1024, 2048) ∪ [2112, 4096)`, which are disjoint from both landing segments, issued before the read barrier.

When a core has more than one block, the reads of block `b+1` are issued after block `b`'s barrier and before its move. This is legal because the blocks are independent. No floating-point order exists to change, since this is a pure move.

### Block schedule

There is one data-movement kernel per core, the `mover`. Its logical schedule:

```cpp
// num_blocks_this_core = ceil_div(core_w_tiles, block_w_tiles); last block has last_block_extent tiles
load_row_block(0);                                      // into slot 0
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    wait_row_block(block_idx);                          // one read barrier per block
    if (block_idx + 1 < num_blocks_this_core) {         // depth 2 only
        load_row_block(block_idx + 1);                  // after a writes_flushed: the slot held block_idx - 1
    }
    row_to_column_block(block_idx);                     // in place in the slot
    store_block(block_idx);                             // whole-tile writes from the slot
}
drain_stores();                                         // noc_async_write_barrier, once (BB-03)
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed-cost work |
|-----------------|-------------|--------------------|---------------------------------------|
| `load_row_block` | `block_extent` tiles. Per tile, 2 reads of `face_row_bytes`: input page byte 0 → staging page byte 0, and input page byte `face_bytes` → staging page byte `2 * face_bytes`. | the target slot `block_idx % staging_depth_blocks`. At depth 2 the other slot holds the block being moved and stored. | Reads are issued back to back, never barriered per tile. At depth 2, one `noc_async_writes_flushed` precedes the reads into a reused slot. Zones: `G6U-LOAD-ISSUE`, and `G6U-FLUSH-WAIT` around the flush. |
| `wait_row_block` | the slot's `block_extent` pages | the slot | One `noc_async_read_barrier` per block. Its built-in `invalidate_l1_cache()` (`fence` on BH) makes the landed bytes visible to RISC-V loads. Zone: `G6U-LOAD-WAIT`. |
| `row_to_column_block` | `block_extent` pages × 2 face halves × 15 word moves. For face base `F ∈ {0, 2 * face_bytes}` and `j = 1..15`: `word[F + 64j] = word[F + 4j]`. | the slot | No init. One `invalidate_l1_cache()` per tile, between that tile's last store and its NoC write. This is the store-ordering fence on BH (`tt_metal/hw/inc/internal/tt-1xx/cache.h:12-16`) and a no-op elsewhere. Zone: `G6U-MOVE-STORE-ISSUE`, shared with `store_block` because the two interleave per tile. |
| `store_block` | `block_extent` writes of one whole tile page (`tile_bytes`), staging page → output page `start_tile + block_idx * block_w_tiles + i` | the slot, until its writes are flushed or barriered | The write of tile `i` is issued as soon as tile `i` has been moved and fenced, so its NoC transfer overlaps the moves of tile `i+1`. There is no per-tile or per-block barrier. |
| `drain_stores` | all writes of the core | none | One `noc_async_write_barrier` per kernel. Zone: `G6U-STORE-WAIT`. |

The RT-arg reads and accessor construction run in zone `G6U-SETUP`. Every zone is a `DeviceZoneScopedN` (`tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012`), which compiles to `(void(name))` in the plain build (`kernel_profiler.hpp:1136`), so the plain build is unchanged (BB-21).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| `grid_sync` (core count) | At one tile per core, each core does 2 reads, 30 moves and 1 write. Per-core launch and RT-arg cost may outweigh the parallel moves. | `MAX_CORES` set to give each core 2 or 4 tiles (p1: 8×1 or 4×1 rectangles). Judge on the unprofiled host time of blocking trace replays, not on profiled kernel duration alone (lessons.md RB-053). |
| `rt_args` | `start_tile` and `core_w_tiles` are unique RT args: one dispatcher write per core per call. The two buffer addresses are common RT args, written once. | Derive `start_tile` and `core_w_tiles` in the kernel from `get_relative_logical_x/y()` (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:70-86`) plus CT args (`rect_cols`, `q`, `r` of the split), leaving only common RT args. |
| `write_shape` | A whole 4096 B tile per store writes 2× the bytes that hold column 0 (faces 0 and 2). | 2 writes of `face_bytes` per tile, from staging bytes 0 and `2 * face_bytes` to page offsets 0 and `2 * face_bytes`. This halves the write bytes and adds one transaction per tile, and output faces 1 and 3 are never written (allowed: they are padding). |
| `read_shape` | Two 64 B reads per tile are two transactions, and at this size the cost is per transaction (split_reader report). | One 4096 B whole-tile read per tile. It lands segment 1 at byte `face_bytes`, so lanes 16..31 move from `face_bytes + 4j` to `2 * face_bytes + 64j` (16 moves instead of 15). It also reads the input padding rows into staging, where nothing reads them. |
| `noc_placement` | The mover runs on BRISC with NoC1 (`ttnn.WriterConfigDescriptor()`, which maps to `WriterDataMovementConfig` at O2, `tt_metal/impl/program/program.cpp:568-575`; that config uses the preferred DRAM-write NoC, `tt_metal/api/tt-metalium/kernel_types.hpp:148-153`). The choice was made because writes carry 32× the bytes of reads. At one transfer per core the NoC effect is latency-bound and may vanish (noc_placement README:31-32). | NoC0 for the mover. Or reads on NoC0 and writes on NoC1 from one kernel under `NOC_MODE::DM_DYNAMIC_NOC` (`kernel_types.hpp:41`, `program_descriptors.cpp:611`). |
| `overlap` (multi-block cores only, `core_w_tiles_max > 329`, no production case) | A block of up to 164 tiles at depth 2 overlaps only at block granularity. | Per-tile transaction ids (`set_read_trid` / `async_read_barrier_with_trid`, catalog `trid_double_issue`), so the move of tile `i` starts when tile `i`'s reads land. |
| `dual_risc` (multi-tile cores only) | One RISC-V performs every move of its core. | Both DM RISC-Vs as movers, each owning half of the core's tiles with its own reads, moves and stores. This also halves the core count for the same parallelism of the moves, which feeds back into `grid_sync`. |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM → L1 | float32 bytes, two 64 B face-row segments per input tile | `noc_async_read(accessor.get_noc_addr(page, offset), slot_page + land_offset, face_row_bytes)`, with `(offset, land_offset)` ∈ {(0, 0), (`face_bytes`, `2 * face_bytes`)} | No input padding row reaches L1. Row-0 padding columns of the last tile reach output rows `>= W`, which are output padding. |
| in-L1 move | 32-bit words, raw bits | RISC-V `volatile tt_l1_ptr uint32_t` loads and stores in the staging page, in place | Bit-exact by construction: no FPU, no unpacker or packer, no format conversion. The kernel never touches a value as `float`. |
| L1 → DRAM | float32 tile page (4096 B) | `noc_async_write_page(page, accessor, slot_page)` | The output padding holds row-0 input lanes 1..15 at face 0 row 0, lanes 17..31 at face 2 row 0, and stale L1 elsewhere. The contract leaves it unspecified. |
| cross-core | n/a | none | No regime, built or deferred, has inter-Tensix traffic. |
| mesh | identical per device | `ttnn.generic_op(io_tensors, ProgramDescriptor)` broadcasts the program SPMD to every device of the mesh | The per-device shape is the same on every chip (contract section 5). |

**TARGET.** TARGET has no `memory_layout` axis: DRAM interleaved is the only placement, so there is no sharded scheme to map. A future width-shard along `w_tiles` would be a knob-turn, with each core's input shard consumed in place through a CB backed on the sharded buffer. A height or block shard has nothing to cut, because the input is one tile row.

**Structural impossibilities.** None beyond `INVALID = []`.

**Entry point and validation order.** `g6_unsqueeze(input_tensor, dim)` calls `validate(input_tensor, dim)` as its first line. `validate()` runs these checks in order, each before any allocation or descriptor work:

1. **Taggers and SUPPORTED.** Build the scenario dict `{"input_shape", "dim", "dtype", "layout"}` from the live tensor and apply `INPUT_TAGGERS` (`tag_dtype` reads `["dtype"]`, `tag_layout` reads `["layout"]`). Check SUPPORTED (`dtype: [ttnn.float32]`, `layout: [ttnn.TILE_LAYOUT]`); a value outside it raises `UnsupportedAxisValue`, a `NotImplementedError` and hence a `RuntimeError`.
2. **EXCLUSIONS.** The list is empty.
3. **Domain checks, raising `ValueError`:**
   - the input is on device;
   - `dim` is an `int` and not a `bool`;
   - rank `r` is in 1..3;
   - `dim` is in `[-(r+1), r]`;
   - the canonical `k = dim + r + 1` if `dim < 0`, else `dim`, equals `r`;
   - `numel > 0`;
   - every dim except the last is 1;
   - the buffer type is DRAM and the memory layout is INTERLEAVED;
   - the tile shape is 32×32.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | One tile pair (input tile `t` → output tile `t`). A core's block is `block_w_tiles` consecutive units. |
| Grid | `(grid_x, grid_y) = device.compute_with_storage_grid_size()`, `MAX_CORES = grid_x * grid_y`, and `num_work_cores = min(tensor_w_tiles, MAX_CORES)`. The rectangle has origin (0, 0), with `rect_rows = ceil(num_work_cores / grid_x)` and `rect_cols = ceil(num_work_cores / rect_rows)`. `ttnn.split_work_to_cores(ttnn.CoreCoord(rect_cols, rect_rows), tensor_w_tiles, row_wise=True)` assigns `q+1` / `q` tiles row-major. The kernel and the CB use the whole rectangle as **one core range**. Rectangle cores past the last assigned tile (at most `rect_rows - 1` of them) get `core_w_tiles = 0` and return after `G6U-SETUP`. |
| Per-core work | `start_tile` is the running sum of `core_w_tiles` over the rectangle in row-major order. `num_blocks_this_core = ceil_div(core_w_tiles, block_w_tiles)` and `last_block_extent = core_w_tiles - (num_blocks_this_core - 1) * block_w_tiles` are derived in the kernel. |
| Remainder | `tensor_w_tiles = ceil(W / 32)`, never `W // 32`. The host asserts `input_tensor.buffer_num_pages() == output.buffer_num_pages() == tensor_w_tiles`, and both page sizes `== tile_bytes`, raising RuntimeError otherwise. An uneven split falls to the `q` group with the same kernel and a runtime extent. |
| p1 `[1, 512]` | 16 tiles → 8×2 rectangle → 16 cores × 1 tile, 1 block, depth 1, **1 core range**, 4096 B L1 per core |
| p2 `[1, 256]` | 8 tiles → 8×1 rectangle → 8 cores × 1 tile, 1 block, depth 1, **1 core range**, 4096 B L1 per core |

The rectangle is wide rather than tall: the catalog's `noc_placement` row placement, and RB-036's few-range rule. Filling the grid row by row would give p1 13 + 3 cores on a 13-wide grid, which is 2 ranges.

There is one regime and one kernel, so there is no regime-selection function and no regime-pinned test. The single-block and multi-block cases are the same kernel with a different `block_w_tiles` / `staging_depth_blocks` (CT).

**Plan reporting (BB-08, BB-21).** `plan_for(input_tensor)` returns:

- the regime name (`row_gather_face_scatter`)
- the rectangle, and `core_ranges` with its range count
- per core: `core`, role (`mover` or `idle_tail`), `start_tile`, `core_w_tiles`
- `block_w_tiles`, `staging_depth_blocks`
- L1 bytes per core

`_plan()` is the single source of every derived quantity; the descriptor builder and `plan_for` both consume it.

**Program-description cache (BB-07, BB-16).** A module-level dict holds the built `KernelDescriptor`, `CBDescriptor` and `ProgramDescriptor` objects for each plan. Its key is: logical input shape, padded input shape, dtype, layout, tile shape, buffer type, memory layout, device id and grid size.

- **On a hit**, nothing is rebuilt. The entry point allocates the output, writes `[input_addr, output_addr]` into the cached kernel's `common_runtime_args`, assigns the kernel back to the descriptor's `kernels` list, and calls `ttnn.generic_op([input_tensor, output], descriptor)`.
- **Why the reassignment.** `ProgramDescriptor.kernels` is a by-value nanobind property (`ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:1137`), so a mutation through `descriptor.kernels[0]` does not reach the descriptor.
- **The program hash** excludes runtime-arg values (`ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:78-100`). Addresses never appear in CT args or in the key: interleaved `TensorAccessorArgs` carry no address.
- **Coverage.** The golden cases `cache_two_widths` and `traced_replay` exercise this path.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_output_tiles` | 0 | `tile_bytes` (4096, `ttnn.tile_size(ttnn.float32)`) | `staging_depth_blocks * block_w_tiles` | Spans `w_tiles` → `block_w_tiles` pages per slot. Spans `lane` → one whole tile page, because the store writes the whole page. `n_rows` → 1 (domain). Depth from `staging_depth_blocks`. | Float32: a dataflow-only payload that never passes through DEST (there is no compute kernel) | mover | mover | whole kernel |

This is the op's only CB. The mover uses it as an L1 allocation addressed by `get_write_ptr(cb_output_tiles) + slot * block_w_tiles * tile_bytes` (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:325`), with no reserve, push, wait or pop: a single thread owns it end to end. Slot reuse at depth 2 is ordered by `noc_async_writes_flushed`.

The row segments need no buffer of their own. They land at their destination face bases in the output page, and the move runs in place: l1-footprint-discipline Rule 3, pattern 1 (pack into the destination) and pattern 2 (transform in place).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_row_block` | `block_extent` tiles × 2 face-row segments | raw dataflow API; no covering helper (see API Mapping) | input: DRAM `input_tensor` pages `start_tile + block_idx * block_w_tiles + i` | `cb_output_tiles`, the `block_extent` pages of slot `block_idx % staging_depth_blocks` | reads in flight |
| 2 | `wait_row_block` | the slot's pages | raw dataflow API | `cb_output_tiles`, same pages | `cb_output_tiles`, same pages | bytes `[0, 64)` and `[2048, 2112)` of each page hold row 0 |
| 3 | `row_to_column_block` | `block_extent` pages × 2 halves × 15 word moves | custom block operation (gap recorded in API Mapping) | `cb_output_tiles`, same pages | `cb_output_tiles`, same pages (in place) | column 0, rows 0..31, of each page holds lanes 0..31, and each tile is fenced |
| 4 | `store_block` | `block_extent` whole-tile writes, interleaved per tile with #3 | raw dataflow API | `cb_output_tiles`, same pages | DRAM `output` pages, same ids | the slot is free once flushed (depth 2) or barriered (end of kernel) |
| 5 | `drain_stores` | all of the core's writes | raw dataflow API | n/a | n/a | no NoC transaction in flight (BB-03) |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_row_block` | raw_api | `noc_async_read` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | `(src_noc_addr, dst_l1_addr, size = face_row_bytes)` | DRAM input | `cb_output_tiles` | loop extent `block_extent` (from `block_w_tiles`) |
| `load_row_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | `offset ∈ {0, face_bytes}` | n/a | n/a | none |
| `load_row_block` | raw_api | `noc_async_writes_flushed` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1980` | none | n/a | n/a | before a reused slot only (depth 2) |
| `wait_row_block` | raw_api | `noc_async_read_barrier` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1921` (invalidate at :1932) | none | n/a | n/a | once per block |
| `row_to_column_block` | raw_api | RISC-V word loads and stores through `volatile tt_l1_ptr uint32_t*` | n/a (core-local L1 access) | face bases `0`, `2 * face_bytes`; `j = 1..15` | `cb_output_tiles` | `cb_output_tiles` | loop extent `block_extent` |
| `row_to_column_block` | raw_api | `invalidate_l1_cache()` (`fence` on BH) | `tt_metal/hw/inc/internal/tt-1xx/cache.h:12-16` | none | n/a | n/a | once per tile |
| `store_block` | raw_api | `noc_async_write_page` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1418` | `(page_id, accessor, src_l1_addr)`; size = accessor page size | `cb_output_tiles` | DRAM output | loop extent `block_extent` |
| `drain_stores` | raw_api | `noc_async_write_barrier` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1951` | none | n/a | n/a | once per kernel |
| all | raw_api | `DeviceZoneScopedN` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012` | zone names in Block schedule | n/a | n/a | none |
| host | host | `ttnn.split_work_to_cores(CoreCoord, units, row_wise)` | `ttnn/cpp/ttnn-nanobind/operations/core.cpp:470-475` | `(CoreCoord(rect_cols, rect_rows), tensor_w_tiles, True)` | n/a | n/a | `MAX_CORES` shapes the rectangle |
| host | host | `ttnn.generic_op(io_tensors, ProgramDescriptor)` | `ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp:62-68` | `[input_tensor, output]`, output last | n/a | n/a | none |

**Helpers considered and rejected**

| Raw entry | Helper candidate | File:line of the mismatch | Concrete reason |
|-----------|------------------|---------------------------|-----------------|
| `row_to_column_block` | `copy_tile_column<dfb_id, T>` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:91-102` | It copies a column to a column, applying the same offset `column_base(col) + row_offset(face_row, row)` to source and destination (`:97-99`). This op's source is a row: consecutive words of face row 0. The gap would be closed by a `copy_face_row_to_column<dfb_id, T>(face_base)` beside `copy_tile_column`, and the move can reuse `detail::TileColumnLayout::row_offset` (`:53-55`) for its destination walk. |
| `row_to_column_block` | `fill_tile_column` | `tile_column_dataflow.hpp:69-78` | It writes one constant down a column. This op writes 32 different values. |
| `row_to_column_block` | self-aimed L1→L1 NoC copies (`local_noc_addr`, `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:41-44`; `local_copy_helpers_dataflow.hpp`) | NoC L1 alignment `noc_parameters.h:375-376` | A NoC transfer keeps the source offset modulo 16. Lane `j` moves from `4j` to `64j`, so it cannot be placed for `j mod 4 != 0`. |
| `row_to_column_block` | compute `transpose_tile` (LLK API; no kernel_lib helper) | `tt_metal/hw/inc/api/compute/transpose.h:120-137` | This is the rejected regime `compute_transpose`: bit exactness of the 32-bit DEST route is unverified, and it adds a compute kernel. |
| `load_row_block` | `noc_async_read_page` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1243-1267` | It always transfers the accessor's full page size (`:1253-1266`); its `offset` shifts the source but not the size. The segment read needs `size = face_row_bytes`. |
| `load_row_block`, `store_block` | `tilize_helpers_dataflow.hpp`, `reduce_helpers_dataflow.hpp`, `mcast/*` | n/a | These cover tilize staging, reduce scalers and multicast. None reads or writes a sub-page DRAM segment of a tile. There is no compute kernel, so the compute helpers (tilize, untilize, reduce, eltwise, matmul) have no phase to cover. |
| (padding fill, not used) | `zero_tile` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:52-57` | Not used: the contract leaves output padding unspecified. It zeroes a whole tile, which would also clear the landing segments, so the Stall-shadow check records the disjoint-range variant instead. |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| A converting path breaks bit exactness (R1) | The gate is bit equality, including NaN payloads, -0.0, infinities and subnormals. | No compute kernel. The move uses `uint32_t` loads and stores only. The kernel never declares a value as `float`, never routes a value through unpack/math/pack, and never through a host conversion. |
| Input padding reaches a logical output (R3) | Rows 1..31 of each input tile hold any bits; the golden suite and the acceptance test poison them with NaN. | Only face row 0 of faces 0 and 1 is read. Row-0 padding columns (`>= W mod 32` of the last tile) map to output rows `>= W`, which are output padding. |
| The input's pages come back unchanged (R2) | `check_output` refuses an aliased output buffer and checks every value. | The output is a fresh `allocate_tensor_on_device` buffer, and every output tile is written by `store_block`. |
| Stale or reordered data on BH | The BH DM RISC-V has a 4×16 B write-through L1 cache (`tt_metal/hw/inc/internal/tt-1xx/cache.h:7-11`). | Reads: the move runs only after `noc_async_read_barrier`, which invalidates (`dataflow_api.h:1932`). Writes: one `invalidate_l1_cache()` (`fence`) per tile between its last store and its NoC write. |
| A sub-alignment DRAM read | The BH DRAM read alignment is 64 B. | 64 B segments at 64-aligned offsets into a CB base allocated at DRAM alignment (Mechanism caps). Run once under `--dev` so the watcher sanitizer checks it. |
| Program cache miss or collision (R8, BB-07) | p1 and p2 differ only in `W`. | The Python key holds the shapes. The `generic_op` hash covers core ranges and CT args, which differ between p1 (8×2) and p2 (8×1). Addresses live only in common RT args, and the descriptor's `kernels` list is reassigned on every hit (Work Distribution). |
| Paths no test reaches | Production gives 1 tile per core and the suite holds only production cases (BB-22). Uneven splits, idle tail cores, multi-block cores (depth 2), `W mod 32 != 0`, and rank-1 / rank-3 inputs run in no test. | These are the same block operations with runtime extents, not separate code. During implementation, check each once with `scripts/tt-probe.sh`, and do not add them to the tests. Suggested probes: `(1, 32*17 + 5)` (17 tiles, ragged, an idle tail core), `(W,)` rank 1 with `W = 40`, `(1, 1, 100)` rank 3, and a `W` with `core_w_tiles_max >= 2`. |
| A refused call builds a program | `test_validation_errors` counts program-cache entries. | `validate()` runs first, and every refusal is raised before the output is allocated or a descriptor is built. Malformed requests raise `ValueError`; support refusals raise `UnsupportedAxisValue` (a `RuntimeError`). |
| BB-03: a NoC transaction left in flight | The next program starts right after the op. | `drain_stores` (`noc_async_write_barrier`) ends every core's kernel, including idle tail cores, which issue nothing. No posted writes are used. |
| Output padding differs between runs | Stale L1 lands in output faces 1 and 3, and in faces 0 and 2 outside column 0 and row 0. | The contract allows it (padding unspecified, section 7), and BB-06 is graded on logical values. The read-shadow zero fill in the Stall-shadow check is the recorded mechanism if a consumer needs fixed padding. |
