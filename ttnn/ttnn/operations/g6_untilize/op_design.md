# Operation Design: g6_untilize

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no arithmetic) |
| Goal | Copy one DRAM-interleaved uint32 TILE tensor into a new DRAM-interleaved uint32 ROW_MAJOR tensor of the same shape, every logical element bit-identical, as exactly one device program per call. The reorder is done by the NoC: each core gathers 64 B face-row pieces from the tile pages straight into row order in its L1, then writes whole output pages. |
| Math | `out[b, r, w] = x[b, r, w]` (bit pattern), x stored TILE (32x32 tiles, four 16x16 faces TL/TR/BL/BR, row-major inside a face), out stored one page per row `p = b*H + r` |
| References | `docs/explorations/g6_untilize/DOSSIER.md`, `math.md` (M1-M4, F1), `tensix.md`, `risks.md`; `eval/golden_tests/g6_untilize/feature_spec.py`, `helpers.py`; `.claude/references/blocking-model.md`, `l1-footprint-discipline.md`; perf catalog `ttnn/ttnn/operations/examples/master.md` (entries cited per knob below) |
| Production case | (1, 1, 32, 128) uint32 TILE, DRAM interleaved, on a (1,4) mesh, 124 calls per traced pass. 4 input tiles (4096 B each), 32 output pages (512 B each). Phase 0 = TARGET = this case. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 4, uint32, TILE, DRAM interleaved, on device | - | buffer address is a common RT arg |
| `use_multicore` | `bool`, keyword-only | no | exactly `True` (checked with `is True`); anything else raises `ValueError` | `True` | host only; decides nothing |
| `sub_core_grids` | `ttnn.CoreRangeSet`, keyword-only | no | exactly `None`; anything else raises `ValueError` | `None` | host only; decides nothing |

No dtype, memory_config, output-tensor or compute-config keyword exists. The core count is the blocking decision (below), never these keywords.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 4 `(d0, d1, H, W)`; production `(1, 1, 32, 128)` |
| Dtype | `ttnn.uint32` (4 B per element) |
| Layout | TILE (32x32, faces 16x16) |
| Memory | DRAM interleaved; one page = one tile = `tile_bytes = 32*32*4 = 4096 B` |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` |
| Dtype | `input_tensor.dtype` (uint32) |
| Layout | ROW_MAJOR; one page per row, `page_bytes = W * 4` (512 B in production) |
| Memory | `input_tensor.memory_config()` (DRAM interleaved), newly allocated by the entry point with `ttnn.allocate_tensor_on_device(shape, dtype, ttnn.ROW_MAJOR_LAYOUT, device, memory_config)` (positional args); shares no storage with the input; the input is never written |

### Host contract (registry model)

| Item | Decision |
|------|----------|
| `INPUT_TAGGERS` | `{"dtype": tag_dtype, "layout": tag_layout}`; `tag_dtype(inputs, axes) -> inputs[0]["dtype"]`, `tag_layout(inputs, axes) -> inputs[0]["layout"]`; `inputs = (scenario,)`, `scenario = {"input_shape": tuple(shape), "dtype": dtype, "layout": layout}`. No other tagger (no alignment/rank/size axis). |
| `SUPPORTED` | `{"dtype": [ttnn.uint32], "layout": [ttnn.TILE_LAYOUT]}` (the implementer writes it; this is the Phase 0 claim) |
| `EXCLUSIONS` | `[]` |
| `validate(input_tensor, *, use_multicore=True, sub_core_grids=None)` | 1) `use_multicore is not True` -> `ValueError`; 2) `sub_core_grids is not None` -> `ValueError`; 3) not a device tensor (`ttnn.is_tensor_storage_on_device`) -> `ValueError`; 4) `len(shape) != 4` -> `ValueError`; 5) `memory_config().memory_layout != INTERLEAVED` or `buffer_type != DRAM` -> `ValueError`; 6) build the scenario dict, apply `INPUT_TAGGERS`, refuse a value outside `SUPPORTED` with `UnsupportedAxisValue`; 7) `EXCLUSIONS` with `ExcludedCell`. No INVALID check. Every refusal message (ValueError and support refusal) starts with `g6_untilize: ` and names the offending value; the acceptance test matches on the op name. |
| Entry point | `g6_untilize(input_tensor, *, use_multicore=True, sub_core_grids=None)`; first line `validate(...)`; then allocate output, fetch the cached program descriptor (see Work Distribution, BB-07), set the two address words, `return ttnn.generic_op([input_tensor, output_tensor], descriptor)`. No other ttnn op is called. |
| Plan report (BB-08 / BB-21) | module function `describe_plan(input_tensor) -> dict` returning `regime` (`"row_gather"`), `core_ranges`, `num_cores`, `block_rows`, `block_tile_cols`, `scratch_slots`, `blocks_per_core` (min/max), and the per-core role (`row_gather` for every core in the range; no other role exists). It reads the same cached plan the entry point uses. |
| Structural impossibilities | none beyond `feature_spec.py`'s empty `INVALID` |

## Blocking Model

The op's index space is the output's row-major space: `batch` (product of leading dims, `B = d0*d1`), `rows` (`H`, per image), `tile_cols` (`tensor_tile_cols = ceil(W/32)`). Inside a tile, a row of 32 values is two **face-row pieces** of 16 values = 64 B, which lie 1024 B apart in the tile page (faces `2h` and `2h+1`, `h = (r mod 32) div 16`), and are adjacent in the output row. No output element depends on more than one input element (math.md, axes table).

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` (B = d0*d1, extent 1) | independent: out[i] reads x[i] only | folded into `block_rows`: rows are flattened to output page index `p = b*H + r` (`tensor_rows = B*H`), because every output page is self-contained and its input address is recomputed per row (`t = b*Ht + r div 32`, `Ht = ceil(H/32)`, per image) | 1 (B = 1) | `block_rows` (host) | with `rows`: blocks of the flattened page axis are assigned to cores and may cross image boundaries | knob-turn |
| `rows` (H, extent 32) | independent: no result spans rows; one row is one output page | `block_rows` | **1**: the split spreads 32 rows over 32 cores, so each core's whole assignment is 1 row. The block equals the core's assignment, i.e. it is at bound 2 of blocking-model §1, not collapsed. | host constant in the program-descriptor module, computed once per plan: `block_rows = clamp(ceil(tensor_rows*col_blocks / core_budget), 1, rows_fit_cap)` | **split across cores (primary split)**: contiguous runs of blocks per core, 1 block per core in production | knob-turn |
| `tile_cols` (Wt, extent 4) | independent, but every output page spans all tile columns: a cut along it fragments each page write into partial-page writes | `block_tile_cols` | **4 = tensor_tile_cols** (whole width: one block row = one whole output page) | host constant, `block_tile_cols = min(BLOCK_TILE_COLS_REQ or tensor_tile_cols, tile_cols_fit_cap)` | **not split across cores in Phase 0**: splitting adds 3 write transactions per page (4 x 128 B instead of 1 x 512 B) and only shortens per-core read issue. That alternative is the grid-sync lamp below. | knob-turn (column offset and segment size are already parameters of `store_block`) |

Reuse check, run for every (operand, chosen split) pair: `x` varies along `rows` (each core reads only its own rows' pieces) and `out` varies along `rows`. No operand is re-read by several cores, so there is no reuse-shared operand and no broadcast regime.

Dependent axes: none. No "split the dependent axis + combine" candidate exists.

Stall-shadow check: each core waits twice, on the read barrier of `gather_block` and on the final write barrier. In production nothing is independent of either wait: the single page write needs all 8 pieces, and the end of the kernel needs the write acknowledged (BB-03). When `num_blocks_this_core > 1`, the gather of block k+1 is scheduled into the shadow of block k's writes through the `scratch_slots` ring (Block schedule). No floating-point order exists to perturb.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_row_staging` | `scratch_slots = min(SCRATCH_DEPTH, max_blocks_per_core)`, with `SCRATCH_DEPTH = 2` (host constant) | **1**: production has 1 block per core, so a second slot would be L1 that nothing uses | at `max_blocks_per_core >= 2`: the gather of block k+1 into the other slot runs while block k's page writes are in flight on the same RISC |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| TILE storage (math.md F1; tensix.md "Mechanism caps") | read transfer unit: the longest run contiguous in both x and out is one face-row piece, `face_row_bytes = 16 * 4 = 64 B` | every read is exactly one face-row piece: `2 * block_tile_cols` reads per row. A read never spans two face rows or two faces. | a 128 B read from face `2h` row `i` pulls row `i+1`'s first 16 values into columns 16-31 of row `i`, giving wrong values in a fixed pattern (risks.md R2) |
| BH DRAM read alignment: L1 dst must equal the DRAM src mod 64 (`NOC_DRAM_READ_ALIGNMENT_BYTES 64`, `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`; checked at `tt_metal/hw/inc/internal/debug/sanitize.h:746`) | staging layout | sources sit at multiples of 64 inside a 64 B-aligned tile page; the staging slot base is 64 B-aligned because CBs are allocated at DRAM alignment (`tt_metal/impl/program/program.cpp:2505-2509`); the row stride `block_row_bytes = block_tile_cols * 128` and piece offsets `(c - c0)*128 + fc*64` are multiples of 64 | the NoC drops the low address bits, so bytes land shifted (silent corruption), or the watcher hangs on `DebugSanitizeNocAlignment` |
| Output page size vs staged row | write size of a block row | write `min(block_row_bytes, page_bytes - c0 * 128)` bytes at page offset `c0 * 128` (`c0` = first tile column of the block) | when `W mod 32 != 0`, writing the whole staged row overruns the page into the next page of the bank and corrupts another row |
| Rows of a padded last tile row | `rows` | rows are enumerated over logical `r < H` per image only; the padded rows of a tile are never read or written | output gets rows that do not exist (shape error) or writes past the tensor |
| Single-packet NoC transfer (`NOC_MAX_BURST_SIZE`, `noc_parameters.h:289-290`) | `block_row_bytes` when a one-packet write API is used | one-packet write only while `block_row_bytes <= NOC_MAX_BURST_SIZE`; otherwise the multi-packet `noc_async_write` | a one-packet call with a longer size truncates the transfer |
| Staging L1 bound (model environment: 1347776 B per core, BB-05) | `block_rows * block_tile_cols` | `rows_fit_cap` and `tile_cols_fit_cap` chosen so that `block_rows * block_row_bytes * scratch_slots <= 1347776`; `block_tile_cols` drops below `Wt` only when one whole row times `scratch_slots` exceeds the bound (W above about 168k values) | launch refused, or L1 overlaps model tensors in the traced model |
| Core linear index from logical coordinates | `num_cores` | the plan's core range is a rectangle at a host-given origin; the kernel's index is `(y - y0) * cores_x + (x - x0)` | wrong rows per core (duplicate or missing pages) |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `row_gather`: per row, 2 x `block_tile_cols` reads of one 64 B face-row piece from the tile pages straight into row order in an L1 slot, then one write per row of `block_tile_cols * 128` bytes to the output page | **built** | always (only regime; covers every rank-4 uint32 TILE DRAM-interleaved tensor, the production case included) | `block_rows x block_tile_cols` (production: 1 x 4) | **minimum**. Boundary: DRAM. x crosses once (each 64 B piece read by exactly one core), out crosses once (each page written once). Transfer shape: 256 reads of 64 B and 32 writes of 512 B. No L1->L1 pass, no cross-core traffic. | Per-block fixed cost: one read barrier (intended once per block), one slot hand-over (`noc_async_writes_flushed`, intended once per block after the first `scratch_slots` blocks), the block's address setup. Per-kernel fixed cost: accessor construction, core-index and block-range arithmetic, one final write barrier. No LLK init, no reconfig, no CB handshake exists. In production each core runs exactly one block, so every fixed cost is paid once per core. Why no other regime applies to INPUTS: (1,1,32,128) gives 1 row per core, so the merged-read regime below has nothing to merge, and the tile-read regimes need more transactions per core. |
| `face_chunk_gather`: when a block holds g > 1 rows of one face half, read each face's g rows as one `g*64 B` run (8 reads per block instead of 8g), then reorder in L1 (RISC-V word copy) before whole-page writes | **deferred**: unreachable in INPUTS (production gives every core 1 row, so g = 1 and both forms are the same 8 reads). It only pays when `core_rows >= 2`. | `block_rows >= 2` and the block's rows lie in one 16-row face half | `block_rows x block_tile_cols` | minimum bytes; fewer, larger reads; adds a core-local L1 pass over the block | amortizes read issue over g rows. **Reachability:** only `gather_block`'s realization changes; the block shape, the slot ring and `store_block` stay as built. |
| `tile_read_scatter_write`: read whole tiles (or face chunks), write 64 B face-row pieces straight to their row offsets in the output pages | **rejected**: superseded by `row_gather`. Same DRAM bytes, but 256 write transactions instead of 32 (each page is assembled from 8 partial writes by up to 8 cores), and writes carry acks the kernel must barrier on. | - | `faces x face_rows` | minimum bytes; 8x the write transactions | - |
| `tile_read_local_reorder`: read whole tiles, reorder to rows in L1 (256 self-NoC copies of 64 B or 4096 RISC-V word copies), write pages | **rejected** (dead end): adds a full L1->L1 pass over the tensor that `row_gather` avoids by gathering at read time. Nothing builds on it. | - | tiles | minimum DRAM bytes + 16384 B core-local copy | - |
| `engine_pack_untilize`: unpack tiles to DEST (UInt32 is legal only unpacked straight to DEST, `tt_metal/tt-llk/tt_llk_blackhole/common/inc/cunpack_common.h:467-473`), `pack_untilize` to rows | **rejected**: superseded by `row_gather`. (1) Whether the packer keeps every UInt32 bit is unverified (risks.md R1, INFERENCE) and the gate is bit-exact. (2) Its unit is a whole tile row through DEST (32-bit DEST: at most 4 tiles in half-sync, `tt_metal/hw/inc/api/compute/pack_untilize.h:161-163`); production has `B*Ht = 1` tile row, so it runs on 1 core, or on 4 cores with 32 partial-page writes each. (3) It adds a compute kernel (3 TRISC binaries, BB-18) and CB handshakes. | - | tile row | minimum bytes; fewest DRAM transactions (4 + 32) but confined to one core | - |

### Traffic ranking

Every candidate split moves the same bytes: x crosses DRAM once (16384 B) and out once (16384 B), with no cross-core traffic. No dependent axis exists, so there is no combine candidate. No operand is shared, so there is no broadcast candidate. The splits differ only in transfer shape (count and size of DRAM transactions) and in per-core issue count:

| Rank | Split | DRAM reads | DRAM writes | Per-core issue (production) | Verdict |
|------|-------|-----------|-------------|-----------------------------|---------|
| 1 | `rows` (32 cores x 1 row) | 256 x 64 B | **32 x 512 B (whole pages)** | 8 reads + 1 write | **chosen**: fewest write transactions of any multi-core split; whole-page writes |
| 2 | `rows x tile_cols/2` (64 cores) | 256 x 64 B | 64 x 256 B | 4 + 1 | grid-sync lamp |
| 3 | `rows x tile_cols` (128 cores) | 256 x 64 B | 128 x 128 B | 2 + 1 | grid-sync lamp; needs >= 128 worker cores |
| 4 | `tile_cols` (4 cores, whole tiles, scatter writes) | 4 x 4096 B | 256 x 64 B | 1 + 64 | rejected (regime table) |
| 5 | whole tile row on 1 core (engine path) | 4 x 4096 B | 32 x 512 B | 4 + 32 + compute | rejected (regime table) |

Occupancy did not decide the ranking. The rows split spreads its 32 units over 32 cores (step 1 of the interleaved default), and each core's block is then its whole assignment (step 2).

### Block schedule

Logical schedule of the one data-movement kernel `untilize_dm` on every core of the range. There is no other kernel.

```cpp
// once per kernel: core index -> [first_block, num_blocks_this_core); accessors for x and out
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    gather_block(block_idx);   // DRAM tile pages -> slot (block_idx % scratch_slots), row order
    store_block(block_idx);    // slot -> DRAM output pages (whole page, or page segment when block_tile_cols < tensor_tile_cols)
}
drain();                       // write barrier: every NoC transaction of this core completed (BB-03)
```

| Operation | Block shape it acts on | Resident across it | Intended fixed-cost frequency |
|-----------|------------------------|--------------------|-------------------------------|
| `gather_block` | `block_rows x block_tile_cols` tiles' worth of rows: for each row `p` of the block (`b = p div H`, `r = p mod H`, `t = b*Ht + r div 32`, `h = (r mod 32) div 16`, `i = r mod 16`) and each tile column `c` in `[c0, c0+block_tile_cols)`, two reads of 64 B from input page `t*Wt + c` at offset `(2h+fc)*1024 + i*64`, `fc in {0,1}`, into slot row `row_in_block` at byte `(c - c0)*128 + fc*64`. All `2*block_rows*block_tile_cols` reads are issued back to back, then **one** read barrier. | the slot being filled. Before a slot is reused (`block_idx >= scratch_slots`), the earlier block's writes from it must have departed: `noc_async_writes_flushed` once per block. | read barrier: once per block. Address setup: one bank lookup per (row, tile column), with the 2 piece offsets added to it. No barrier per read (catalog `trid_double_issue`, `double_buffer`: issue all reads, then barrier). |
| `store_block` | `block_rows` page writes of `min(block_tile_cols*128, page_bytes - c0*128)` bytes at page offset `c0*128` of output page `p` | the slot until its writes depart | no barrier per write; writes are flushed only at slot reuse and fully barriered once in `drain` |
| `drain` | - | - | `noc_async_write_barrier` once per kernel |

Production instance: every core runs 1 block, which is 1 row = 8 reads of 64 B, 1 read barrier, 1 write of 512 B, 1 write barrier.

Profiler zones (BB-21, zone build only; `DeviceZoneScopedN`, `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012`, which compiles to nothing in the plain build): `G6U-GATHER-ISSUE`, `G6U-GATHER-WAIT` (read barrier), `G6U-STORE-ISSUE`, `G6U-SLOT-WAIT` (flush before slot reuse), `G6U-DRAIN` (final write barrier). Every piece of work and every blocking wait sits in exactly one zone.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Grid synchronization / participation** (blocking-model §4) | Production is per-transaction-bound (16 KB). Per-core read issue is the main per-core term (catalog `split_reader`: 64 B reads issue-bound, about 40 ns per read on WH B0). Rows-only uses 32 cores x 8 reads; more participation shortens issue but fragments page writes. | `BLOCK_TILE_COLS_REQ = 2` (64 cores, 4 reads + 1 x 256 B write) and `= 1` (128 cores, 2 reads + 1 x 128 B write, if the grid has >= 128 workers). Also fewer cores: `core_budget = 16` (2 rows per core). Judge on unprofiled blocking trace-replay host time and the model step, not on profiler kernel duration alone (lessons.md RB-053). |
| **Issue on both RISCs** (overlap lamp, issue form) | One RISC issues all 8 reads. Catalog `split_reader` measured 1.64x at 64 B when reads are split across NCRISC and BRISC. | Same 32-core rectangle, second copy of `untilize_dm` on the other RISC/NoC: each RISC owns half the tile columns (`block_tile_cols = 2`) and writes its own 256 B page segment. No cross-RISC handshake is needed. |
| **Stateful read issue** | Recomputing a full NoC address and request per 64 B piece costs more than the transfer. | `noc_async_read_one_packet_set_state` once per source tile page plus `noc_async_read_one_packet_with_state` per piece (`dataflow_api.h:575`, `:599`), against plain `noc_async_read` with `get_noc_addr(page, offset)` |
| **Core rectangle shape / placement** | 32 cores as 8x4 at origin (0,0) is the default. Catalog `noc_placement`: row or diagonal beat column lines, but "at block=1 the effect nearly vanishes". | 4x8, 16x2, and an origin away from the DRAM columns |
| **Kernel RISC / NoC** | Default `RISCV_0` + `NOC_0` (8 of 9 transactions are reads; catalog `noc_placement`: reads on NoC0). | `RISCV_1` + `NOC_0`; `RISCV_0` + `NOC_1` |
| **DRAM bank hot spot** | The 4 input tiles sit in 4 banks, so all 256 reads (64 per bank) hit 4 banks in the same window. Catalog `bank_stagger`: interleaved stagger gives 1.01-1.07x. | Rotate each core's tile-column order by `core_index mod block_tile_cols`, so the banks see interleaved arrivals |
| **Overlap lamp (large shapes only)** | Not active in production (1 block per core). For shapes with `core_rows` large, the coarsest block (all core rows in one slot) serializes gather against store. | `block_rows = core_rows / k` with `scratch_slots = 2` |

Catalog entries and the knob each informed: `width_split` (spreading a one-tile-row tensor over cores pays already at Wt = 8, so Phase 0 is multi-core); `double_buffer` and `trid_double_issue` (keep all of a block's reads in flight and use one barrier per block); `tile_reorder` (move whole pages, batch barriers, so writes are whole pages); `read_coalescing` (coalescing needs physical contiguity, so the 64 B piece cap stands); `split_reader`, `noc_placement`, `bank_stagger` (lamps above). No catalog entry covers launch, runtime-arg or core-range cost; those follow lessons.md RB-036 and the BB rules.

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM -> L1 (`gather_block`) | uint32 bytes, face-row pieces of 64 B | NoC reads from the input's interleaved tile pages via `TensorAccessor` (page id + byte offset), issued by `untilize_dm` | the reorder happens here: each piece lands at its row position; bytes only, never a conversion (R1) |
| L1 staging (`cb_row_staging`) | uint32 rows, `block_row_bytes` per row | a CB used only as an L1 allocation, addressed by `get_write_ptr(cb_row_staging) + slot*slot_bytes`; no reserve/push/wait/pop | producer and consumer are the same thread (`untilize_dm`). Order comes from the read barrier (gather before store) and the write flush (store before slot reuse). |
| L1 -> DRAM (`store_block`) | ROW_MAJOR pages | NoC write of a whole page (segment) via the output `TensorAccessor` | non-posted writes, barriered before return (BB-03) |
| Between RISCs / cores | - | none | no cross-core or cross-RISC communication exists in any built regime |

Placement as a TARGET axis: `feature_spec.py` TARGET has no `memory_layout` axis (DRAM interleaved only), so no sharded scheme is a support value. For reference, a height shard along `rows` would be a knob-turn (the rows split is the logical shard; the gather would read local L1 instead of DRAM), and a width shard along `tile_cols` would also be a knob-turn (partial-page segments). Neither is built or claimed.

Mesh: `ttnn.generic_op` with a `ProgramDescriptor` builds an SPMD `MeshProgramDescriptor` over the whole mesh (`ttnn/cpp/ttnn/operations/generic/generic_op.cpp:13-23`). Mesh buffers are allocated at the same address on every device, so the same common RT args (two buffer addresses) make each device read its own input and write its own output. No per-device logic exists.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: `block_rows` consecutive output rows (flattened page index `p`) x `block_tile_cols` tile columns |
| Grid | `core_budget = grid.x * grid.y` from `device.compute_with_storage_grid_size()` (host knob; may be lowered to measure the grid-sync lamp) |
| Geometry (alignment-aware, per image) | `B = d0*d1`; `Ht = ceil(H/32)`; `tensor_tile_cols = Wt = ceil(W/32)`; `tensor_rows = B*H` (logical rows only); `col_blocks = ceil(Wt / block_tile_cols)`; `row_blocks = ceil(tensor_rows / block_rows)`; `total_blocks = row_blocks * col_blocks`; `page_bytes = W*4`; input page of tile `(t, c)` is `t*Wt + c`, with `t = b*Ht + r div 32` |
| Knob order | 1) `block_tile_cols` (default `Wt`); 2) occupancy: `block_rows = clamp(ceil(tensor_rows*col_blocks / core_budget), 1, rows_fit_cap)`; 3) `num_cores = min(total_blocks, core_budget)`; 4) `scratch_slots = min(2, ceil(total_blocks / num_cores))` |
| Core range | **one rectangle** (BB-17, lessons RB-036): `cores_x` = the largest divisor `d` of `num_cores` with `d <= grid.x` and `num_cores/d <= grid.y`; `cores_y = num_cores / cores_x`; origin (0,0). Fallback when no divisor fits: `cores_x = grid.x`, full rows plus one partial row (2 ranges at most). |
| Per-core work | core index `k = (y - y0)*cores_x + (x - x0)` from `get_relative_logical_x/y` (`dataflow_api.h:70`, `:83`); `q = total_blocks div num_cores`, `rem = total_blocks mod num_cores`; core `k` owns blocks `[k*q + min(k, rem), +q + (k < rem))`; block `j` -> `row_block = j div col_blocks`, `col_block = j mod col_blocks` |
| Remainder | ragged in two places: a last row block shorter than `block_rows` (its `last_block_extent` is the actual row count), and a last column block whose write size is clipped to `page_bytes` (mechanism cap). Both are runtime extents of the same block operations. |
| Production | `tensor_rows = 32`, `Wt = 4`, `block_tile_cols = 4`, `col_blocks = 1`, `block_rows = 1`, `total_blocks = 32`, `num_cores = 32`, rectangle **8x4** at (0,0)-(7,3) (needs grid.x >= 8, grid.y >= 4; every BH worker grid satisfies this), 1 block per core, `scratch_slots = 1` |
| Runtime args | **no per-core runtime args.** Common runtime args = `[input buffer address, output buffer address]`, written once per range by the dispatcher. Everything else is a compile-time arg derived from the knobs above (part of the program hash). |
| Program cache (BB-07) | the entry point keeps a module-level cache `{key: (ProgramDescriptor, plan)}` with `key = (tuple(shape), tuple(padded_shape), dtype, layout, memory_config (layout, buffer type), grid.x, grid.y)`. On a hit it rewrites only the two common-RT-arg words: the `common_runtime_args` setter is `def_rw` (`ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:851-853`), and the cached program receives them in place on every hit (`tt_metal/impl/program/program_descriptors.cpp:202-212`). `ProgramDescriptor.kernels` is returned by value in Python, so the implementer holds the `KernelDescriptor` and re-assigns it, or rebuilds the thin `ProgramDescriptor` wrapper from cached parts. Nothing else is recomputed. |
| Trace (BB-16) | the output is allocated in the Python entry point, so under capture it is allocated at capture and replay allocates nothing. No host readback exists. The first (compiling) call precedes capture. |
| Regime selection | single regime (`row_gather`); no predicate; regime-pinned tests not applicable |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_row_staging` | 0 | `block_row_bytes = block_tile_cols * 32 * 4` (512 B) | `block_rows * scratch_slots` (1) | spans `rows` (`block_rows`) and `tile_cols` (page width `block_tile_cols`); streams over blocks via the `scratch_slots` ring; `batch` is folded into `rows` | UInt32 (dataflow payload; never enters DEST) | `untilize_dm` (NoC gather reads) | `untilize_dm` (NoC page writes); the same single thread, no concurrent access | whole kernel; slot `s` is live from its gather issue until its writes depart |

No other CB exists. Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `gather_block` | `block_rows x block_tile_cols` (1 x 4) | no (raw dataflow API; see API Mapping) | input tensor in DRAM (4 tile pages, read-only) | `cb_row_staging`, slot `block_idx % scratch_slots`, `block_rows` pages | slot holds `block_rows` complete rows after the read barrier |
| 2 | `store_block` | `block_rows` page segments of `min(block_tile_cols*128, page_bytes - c0*128)` B | no (raw dataflow API) | `cb_row_staging`, same slot, `block_rows` pages | output tensor in DRAM (`block_rows` pages) | slot free for reuse once `noc_async_writes_flushed` returns |
| 3 | `drain` | - | no | - | - | all writes acknowledged; kernel may return (BB-03) |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| setup | raw_api | `get_relative_logical_x`, `get_relative_logical_y` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:70`, `:83` | - | - | - | core index -> block range (with CT `num_cores`, `cores_x`, origin) |
| setup | raw_api | `get_common_arg_val<uint32_t>` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:168` | idx 0 = input addr, 1 = output addr | - | - | - |
| setup | raw_api | `TensorAccessor` (interleaved) from `TensorAccessorArgs` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:513-520` | CT accessor args of x and out, base address from common RT arg | - | - | - |
| setup | raw_api | `get_write_ptr` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:325` | `cb_row_staging` | - | `cb_row_staging` | slot stride `block_rows * block_row_bytes` |
| `gather_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:554` | input page `t*Wt + c`, offset `(2h+fc)*1024 + i*64` | x (DRAM) | - | `block_tile_cols`, `block_rows` (loop extents) |
| `gather_block` | raw_api | `noc_async_read` (or `noc_async_read_one_packet` / `_set_state` + `_with_state`, perf lamp) | `dataflow_api.h:547` (`:515`, `:575`, `:599`) | 64 B per call | x (DRAM) | `cb_row_staging` | - |
| `gather_block` | raw_api | `noc_async_read_barrier` | `dataflow_api.h:1921` | once per block | - | - | - |
| `store_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tensor_accessor.h:554` | output page `p`, offset `c0*128` | - | out (DRAM) | `block_tile_cols` (offset, size) |
| `store_block` | raw_api | `noc_async_write` | `dataflow_api.h:876` | size `min(block_tile_cols*128, page_bytes - c0*128)` | `cb_row_staging` | out (DRAM) | `block_tile_cols` |
| `store_block` | raw_api | `noc_async_writes_flushed` | `dataflow_api.h:1980` | before slot reuse | - | - | `scratch_slots` |
| `drain` | raw_api | `noc_async_write_barrier` | `dataflow_api.h:1951` | once per kernel | - | - | - |

**Helpers considered and rejected** (each applies to every `raw_api` row above, because this op has no compute phase and no helper moves DRAM bytes the way `gather_block` / `store_block` need):

| Candidate | File:Line of the mismatch | Concrete reason |
|-----------|---------------------------|-----------------|
| `compute_kernel_lib::untilize<block_width_tiles, input_dfb, output_dfb>` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:152-163` (signature), `:81-85` (requires `compute_kernel_hw_startup`), `:101-105` (unit = one tile-row through `pack_untilize`) | It is a compute-kernel helper whose unit is a whole tile row through DEST. Production has `B*Ht = 1` tile row, so the reorder would land on one core (or four cores with 32 partial-page writes each), against the 32-core row split the traffic ranking chose. UInt32 can reach DEST only through unpack-to-DEST (`cunpack_common.h:467-473`), DEST at 32 bits caps a block at 4 tiles in half-sync (`pack_untilize.h:161-163`), and the packer's UInt32 exactness is unverified (R1). This is the `engine_pack_untilize` row, rejected. |
| `dataflow_kernel_lib::local_addr` / `set_read_state` / `read_with_state` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57`, `:74-75`, `:94-95` | Self-aimed L1->L1 copies: the source is this core's own L1 (`local_addr`), so they cannot address a DRAM bank page. The gather's source is DRAM. |
| `dataflow_kernel_lib::read_sticks_for_tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:47-52` | Reads ROW_MAJOR sticks into a CB for tilize, the opposite direction; it has no face-row gather. |
| `fill_tile_column` / column copy | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:69-70` | RISC-V stores inside an L1 tile (column broadcast operand); no DRAM transfer and no row assembly. |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| R1: a path narrower than 32 bits | the gate is bit-exact over full-range uint32 | bytes move only by NoC reads and writes; no unpacker, DEST, packer or conversion is on the path. The engine regime is rejected. |
| R2: wrong face order | row `r` takes faces `2h`, `2h+1` of every tile, `h = (r mod 32) div 16`; rows 16-31 use faces 2 and 3 | the piece offset formula is pinned in Block schedule; the acceptance test has an index-encoding face-order case |
| 64 B DRAM read alignment | BH requires L1 dst == DRAM src mod 64 (`noc_parameters.h:379`) | staging layout keeps every piece offset a multiple of 64 from a 64 B-aligned CB base (Mechanism caps). Run `--dev` once to let the NoC sanitizer confirm. |
| Page overrun on non-multiple-of-32 W | staged row is `Wt*128` B but the page is `W*4` B | write size is clipped to the page (Mechanism caps); production is unaffected |
| R3: aliasing / input modified | values would still be right | output allocated fresh; the kernel only reads the input address; acceptance test checks the address and re-reads the input |
| R4: per-call overhead dominates 16 KB | launch, dispatch and core-range writes outweigh the data movement | one kernel on one RISC per core (smallest image, BB-18), no compute kernel, one rectangular core range (BB-17), zero per-core runtime args, descriptor cached (BB-07) |
| BB-03: transactions left in flight hang the next program | the kernel's last action is a write | `drain` = `noc_async_write_barrier` before return; reads are barriered per block |
| R6: mesh | production runs on a (1,4) mesh | SPMD `generic_op`; identical addresses per device; nothing is device 0-specific. Not testable from this single-chip slot; covered at model integration (DOSSIER Q1). |
| Stale addresses on a cache hit | the descriptor is cached; addresses change per call | only the two common-RT-arg words carry addresses and they are rewritten on every call; acceptance test moves addresses between calls and checks values |
| Core index on a sub-device | `get_relative_logical_*` is relative to the sub-device origin | `sub_core_grids` is refused (must be `None`); range origin passed as CT args |
