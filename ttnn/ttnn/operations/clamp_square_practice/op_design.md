# Operation Design: clamp_square_practice

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (pure elementwise, unary with two scalar parameters) |
| Goal | `y = clamp(x, min_val, max_val) ** 2` for a bf16 TILE interleaved-DRAM tensor, as **one** `ttnn.generic_op` dispatch whose compute is one fused SFPU chain (Copy → Clamp → Square → Pack) per core. |
| Math | `output[i] = (min(max(input[i], min_val), max_val))²` for every element `i` — matches `torch.clamp(x, min_val, max_val) ** 2` |
| Mode | Derivative (generic_op + ProgramDescriptor pattern, `.claude/references/generic_op_template/`) |
| References | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp`, `eltwise/unary/scalar.inl` (`Clamp`), `eltwise/unary/misc.inl` (`Square`), `tt_metal/hw/inc/api/compute/eltwise_unary/clamp.h`, `tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_clamp.h`, catalog `ttnn/ttnn/operations/examples/master.md` (entries `double_buffer`, `compute_fusion`, `compute_block_size`, `noc_placement`, `width_split`, `trid_double_issue`, `bank_stagger`) |

**Golden-test spec.** `eval/golden_tests/clamp_square_practice/feature_spec.py` was **not present** when this design was written. This document refers to the support axes by the names `eval/op_template.py` uses (`dtype`, `layout`, `alignment`, `memory_layout`). If `/golden-tests` later authors the spec with different names, the spec wins and the implementer maps the names. The planner did not create the spec.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | see Tensors → Input | — | tensor (TensorAccessor CT args + base address RT arg) |
| `min_val` | `float` (Python), positional or keyword | yes | finite fp32, `min_val <= max_val` | — | **RT** (compute), packed as the IEEE-754 **fp32 bit pattern** in a `uint32` |
| `max_val` | `float` (Python), positional or keyword | yes | finite fp32, `min_val <= max_val` | — | **RT** (compute), fp32 bit pattern |

The scalars are runtime args, not compile-time args. A new `(min_val, max_val)` pair must not trigger a kernel recompile. Pack them with `struct.unpack("<I", struct.pack("<f", v))[0]`. **Do not** use bf16-truncated `<< 16` packing: the LLK loads the whole 32-bit word as an fp32 constant (`ckernel_sfpu_unary_max_min.h:15`, `vConstIntPrgm0 = value`).

**Host validation** happens in Python before any device work. Each check raises `ValueError`, except where noted:

| Check | Error |
|-------|-------|
| `input_tensor.dtype != ttnn.bfloat16` | ValueError |
| `input_tensor.layout != ttnn.TILE_LAYOUT` | ValueError |
| `len(input_tensor.shape) < 2` | ValueError |
| logical `shape[-1] % 32 != 0` or `shape[-2] % 32 != 0` | ValueError |
| `memory_config().memory_layout != TensorMemoryLayout.INTERLEAVED` (any sharding) or `memory_config().buffer_type != BufferType.DRAM` | ValueError |
| `min_val > max_val` (`min_val == max_val` is legal) | ValueError |
| `min_val` / `max_val` are not real numbers (int is accepted and cast to float) | ValueError |

These checks double as the Phase 0 `SUPPORTED` gate (the implementer writes `SUPPORTED`/`validate()`). Values outside the Phase 0 rectangle raise `ValueError` because the requirement mandates that type for this op.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank ≥ 2, `[*leading, H, W]`, `H % 32 == 0`, `W % 32 == 0` |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | interleaved, DRAM |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | interleaved, DRAM (`ttnn.DRAM_MEMORY_CONFIG`). Allocated on device before the dispatch with `ttnn.allocate_tensor_on_device`; it is not a separate op dispatch. |

## Blocking Model

### Axes

The op is a pure elementwise map, so every output tile depends on exactly one input tile at the same page index. The interleaved TILE page order is `page = ((b · tensor_tile_rows_per_image) + ht) · tensor_tile_cols + wt`, where `b` is the flattened leading index. The op treats all axes identically and they are contiguous in page order, so they **fold honestly into one flattened `tile` axis**. That axis carries the block-extent knob. The three source axes are listed so each one has a recorded decision.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` (all leading dims, flattened: `tensor_batch = prod(shape[:-2])`) | independent: no element's result reads another element | folded into `tile`. It is the outermost run of `tile`, so it needs no separate knob. | folded | — | via `tile` (contiguous page ranges) | knob-turn |
| `tile_row` (`tensor_tile_rows_per_image = ceil(H/32)`) | independent: same reason | folded into `tile` | folded | — | via `tile` | knob-turn |
| `tile_col` (`tensor_tile_cols = ceil(W/32)`) | independent: same reason | folded into `tile`. It is the innermost, contiguous run. | folded | — | via `tile` | knob-turn |
| **`tile`** (flattened: `tensor_tiles = tensor_batch · tensor_tile_rows_per_image · tensor_tile_cols`) | **independent**: elementwise, no cross-tile dependency, so no combine exists anywhere | `block_tiles`: tiles per block. One block is one CB handshake, one NoC read batch plus barrier, one write batch plus barrier, and one DEST lane group. | `block_tiles = min(DEST_AUTO_LIMIT_host = 8, core_tiles_g1)`, with a minimum of 1 | **one host constant** in the program descriptor (`BLOCK_TILES_DEFAULT = 8`, clamped as stated), passed as **one CT arg** shared by reader, compute and writer. CB capacities are derived from it. | `ttnn.split_work_to_cores(core_grid, tensor_tiles, row_wise=True)` gives each core a **contiguous** page range `[start_tile, start_tile + core_tiles)` at tile granularity. Group 1 gets `core_tiles_g1` tiles and group 2 gets `core_tiles_g1 − 1`. | knob-turn (all) |

**Why `block_tiles` is 8, not the whole per-core assignment.** This is the coarsest legal value, not a collapsed knob. Four reasons:

1. **Mechanism cap.** The chain runs with `BlockTailSync::FullBlock` + `PerBlockSize` lifecycles, and in that mode the caller must supply a block size that fits DEST (`chain.hpp:504-508`). With bf16, `fp32_dest_acc_en=False` and half-sync, DEST holds 8 tiles (`dest_helpers.hpp:94-99`, `DEST_AUTO_LIMIT` at `:103`). The chain's lane width is 1 (every element uses `Dst::D0`).
2. **Bytes moved do not depend on the extent.** Each tile is touched once and there is no shared operand and no footprint overlap. Per `blocking-model.md` §4, the extent only amortizes per-block handshakes and barriers.
3. **The expensive fixed costs are already hoisted above the block.** These are the SFPU inits (`copy`, `clamp_tile_init`, `square_tile_init`), the data-format reconfig, and `compute_kernel_hw_startup`. The compute pass is **one** `eltwise_chain` call spanning the core's whole assignment (`IterationShape::tiles(core_tiles).block_size(block_tiles, FullBlock)`), so they are paid **once per core**, not once per block.
4. **Catalog evidence.** `double_buffer` measured a 4–8 tile batch with `2×` CB depth as the sweet spot, with "bigger wastes L1".

A larger NoC batch that is decoupled from the DEST group is a perf lamp below, not the default.

**Core grid knob.** `core_grid` is a host parameter of the program-descriptor builder. It defaults to the full `device.compute_with_storage_grid_size()` rectangle and is never inlined. All per-core quantities (`start_tile`, `core_tiles`) are derived from `split_work_to_cores` on it. Phase 0 is multi-core: every tensor with `tensor_tiles ≥ num_cores_in_grid` occupies the whole grid. Smaller tensors occupy `tensor_tiles` cores, one tile each, which is the work limit, not a single-core design.

**Intermediate data.** The scheme has no intermediate tensor. Clamp → Square stays in DEST (see Block schedule), so no later stage has axes to re-block. No core coordinates or collects data from another. Every core is a peer.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `input_depth_blocks` (host constant, single source) | 2 | The reader fills block `k+1` while compute drains block `k` (catalog `double_buffer`: 2.78× on one core vs. single-buffered). |
| `cb_output_tiles` | `output_depth_blocks` (host constant, single source) | 2 | Compute packs block `k+1` while the writer drains block `k`. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `eltwise_chain` with `BlockTailSync::FullBlock` (`chain.hpp:504-508`): the block uses `block_tiles` DEST lanes | `block_tiles ≤ DEST_AUTO_LIMIT`. That is 8 for bf16 half-sync without fp32 DEST, and 4 if `fp32_dest_acc_en` is ever turned on. | Host clamp: `block_tiles = max(1, min(BLOCK_TILES_DEFAULT, dest_limit(fp32_dest_acc_en, dst_full_sync_en), core_tiles_g1))`. `dest_limit` mirrors `get_dest_limit()` (`dest_helpers.hpp:93-100`) and is computed from the **same** compute-config flags the descriptor passes, so the two cannot drift. | FullBlock does not clamp itself. The lanes overflow DEST, so tiles are silently overwritten, giving wrong values (no error). |
| Core assignment bound (`blocking-model.md` §1) | `block_tiles ≤ core_tiles_g1` | Included in the clamp above | A block larger than the work wastes CB capacity. This is harmless but violates the bound. |
| CB ring wrap (`ttnn-cb-memory-fundamentals.md`, CB Ring-Wrap Invariant) | push/pop quanta vs. capacity | Every push, wait and pop is the **nominal** `block_tiles`, **including the ragged last block of a core** (FullBlock). Capacity is `depth · block_tiles`. | Mixed quanta that do not close the ring desynchronize the pointers, so data is read from the wrong slot. |
| Clamp scalar transport (`ckernel_sfpu_unary_max_min.h:15`) | The scalar encoding, not a block extent | Pass the full fp32 bit pattern | Packing bf16 bits (`<<16`) silently truncates bounds such as `0.1` and gives wrong clamp edges. |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `interleaved_stream` | **built** | `memory_layout == INTERLEAVED and buffer_type == DRAM` (the whole Phase 0 domain; every other placement is rejected by validation) | `[block_tiles]` on the flattened `tile` axis. A core runs `num_blocks_this_core = ceil(core_tiles / block_tiles)` blocks, and the last one has `last_block_extent = core_tiles − (num_blocks_this_core − 1) · block_tiles` valid tiles. | **Minimum.** Boundary: DRAM. Every input tile crosses DRAM → L1 once and every output tile crosses L1 → DRAM once, so traffic is `2T` (`T = tensor_tiles · 2048 B`). There is no cross-core traffic. | Fewer reader and writer barriers (1 per block each), fewer CB reserve/push/wait/pop handshakes (4 per block per CB pair), and fewer DEST acquire/commit/release windows (1 per block). Inits and reconfig are once per core regardless of block. There is no regime-added term. **Why no other regime applies in this domain:** there is no dependent axis (so no combine regime), no reuse-shared operand (the scalars are RT args, see Traffic ranking), and the input is not pre-placed in L1 (so no zero-copy regime). |
| `sharded_resident` | **deferred**: the Phase 0 input contract is interleaved DRAM only (the requirement mandates rejecting other placements), and `interleaved_stream` serves every shape in that contract | `memory_layout ∈ {HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED}` and the input is in L1 | The whole resident shard is one block (`core_tiles = shard tiles`). DEST lane groups of `block_tiles` are inside it. `cb_input_tiles` / `cb_output_tiles` are backed zero-copy on the input/output shard via `ttnn.cb_descriptor_from_sharded_tensor`. | Minimum. There is **no** NoC traffic: the input is consumed in place and the output is packed in place into the output shard. | One chain call over the whole shard with no per-block NoC barrier. Reachability: every axis is independent, so all three shard flavors are **knob-turns**. The compute chain keeps the same elements, and only the input lifecycle changes (one upfront wait of the shard). The reader and writer reduce to `cb_push_back` / `cb_wait_front` of the resident shard. The structure of `interleaved_stream` (one CB in, one CB out, one chain per core) is kept intact. |
| `per_tile_stream` | **rejected**: superseded by `interleaved_stream` | `block_tiles = 1`, read one tile then barrier | `[1]` | Minimum bytes, but latency-bound (one barrier per tile) | Superseded. It is the `block_tiles=1` corner of the built regime, measured 2.78× slower on one core (catalog `double_buffer`). It is not a separate algorithm and is not written. |
| `resident_assignment` | **rejected**: superseded by `interleaved_stream` | The block is the core's whole assignment (`block_tiles = core_tiles`): read everything, one barrier, compute, write | `[core_tiles]` | Minimum bytes | No reuse exists to justify residency (each tile is touched once). It serializes read → compute → write with no overlap (the overlap lamp at its extreme). CB size would scale with op size. It violates the FullBlock DEST cap unless it is re-chunked, which reduces it back to `interleaved_stream`. |
| `two_pass_clamp_then_square` | **rejected (dead end)**: superseded by the fused chain in `interleaved_stream` | — | — | Above the minimum. Either two device dispatches (`2T` extra DRAM traffic plus a forbidden second dispatch) or an intermediate L1 CB with an extra pack + unpack per tile. | Not built. Square is an SFPU consumer of Clamp's DEST result, so fusion through DEST wins (catalog `compute_fusion`). A two-dispatch version also violates the one-dispatch rule. |

**Selection predicate.** There is one built regime, so there is no runtime selection: a valid input always lands in `interleaved_stream`. Regime-pinned tests are not needed beyond the acceptance test. The acceptance test still exercises ragged per-core tails and tensors smaller than the grid, because those are the grid-dependent paths.

### Traffic ranking

`T = tensor_tiles · 2048 B` (bf16 tile). Every candidate is costed per memory tier.

| Rank | Candidate split | DRAM crossings | Cross-core | Occupancy note | Verdict |
|------|-----------------|----------------|-----------|----------------|---------|
| 1 | **Flattened `tile` axis, contiguous page ranges** (`split_work_to_cores(..., row_wise=True)`). *Perf 2: the counts are kept but ownership is now bank-affine (see changelog Perf 2)* | in 1× + out 1× = `2T` | 0 | Tile-granular balance for **any** shape, including `[32, 8192]` (one tile-row tall; catalog `width_split`) and `[8192, 32]` | **chosen** |
| 1 (tie on bytes) | Split by tile-rows (`batch · tile_row`), each core takes whole rows | `2T` | 0 | Strands the grid when `tensor_batch · tile_rows` is small (`width_split`: up to 7.8× loss) | rejected: same bytes, worse occupancy, and it is a coarser case of rank 1 |
| 1 (tie on bytes) | Split by tile-columns | `2T` | 0 | Each core reads strided page runs (one run per tile-row), so there are more and shorter NoC runs | rejected: same bytes, more fragmented transfers |
| — | Split a dependent axis + combine | — | — | **There is no dependent axis.** The op has no reduction or scan, so this candidate does not exist. | n/a |

**Operand-reuse check.** `input_tensor` varies along every axis, so it is not reuse-shared. `min_val` and `max_val` do not vary along the split, so they are reuse-shared by construction. They are two 32-bit runtime args per core that live in the kernel's RT-arg memory, with zero DRAM and zero NoC bytes, so a broadcast regime row would move nothing.

**Stall-shadow check.** Compute waits on `cb_input_tiles` (the reader's DRAM fetch), and the writer waits on `cb_output_tiles`. The only work independent of a pending block is the *other* blocks. Double buffering (`depth = 2`) already schedules the next fetch into the current block's compute window. There is no algebraic reorder to exploit: Clamp and Square do not commute (`clamp(x)² ≠ clamp(x²)`), and both are per-element, so nothing can be lifted out of the wait.

### Block schedule

Each core runs this logically. The reader, compute and writer are separate kernels, and blocks `k` / `k+1` pipeline through the depth-2 CBs.

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);          // reader
    clamp_square_block(block_idx);  // compute (one eltwise_chain spans all blocks of the core)
    store_block(block_idx);         // writer
}
```

| Operation | Block shape | Resident across it | Intended fixed-cost frequency |
|-----------|-------------|--------------------|-------------------------------|
| `load_block` | `[block_tiles]` pages starting at `start_tile + block_idx · block_tiles`. Only `valid = min(block_tiles, core_tiles − block_idx·block_tiles)` pages are read, but `block_tiles` pages are reserved and pushed. | nothing | 1 `cb_reserve_back` + `valid` async tile reads + **1** `noc_async_read_barrier` + 1 `cb_push_back(block_tiles)` per block. The TensorAccessor is built once per kernel. |
| `clamp_square_block` | `[block_tiles]` DEST lanes: Copy (unpack L1 → DEST) → Clamp(min,max) → Square → Pack, all in **one** DEST acquire window per block. Math and pack touch only valid tiles on the ragged last block. | DEST holds the block for the chain's duration. No L1 intermediate. | `compute_kernel_hw_startup` **once per kernel**. Element inits (`copy`, `clamp_tile_init`, `square_tile_init`) and the format reconfig are **once per core** (one chain call). 1 wait/pop on `cb_input_tiles`, 1 reserve/push on `cb_output_tiles`, and 1 DEST acquire/commit/wait/release per block. |
| `store_block` | `[block_tiles]` pages, of which `valid` are written | nothing | 1 `cb_wait_front(block_tiles)` + `valid` async tile writes + **1** `noc_async_writes_flushed`/barrier + 1 `cb_pop_front(block_tiles)` per block, plus a final `noc_async_write_barrier` once per kernel |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap / NoC batch size | `block_tiles` is pinned at 8 by the DEST cap, not by NoC efficiency. With enough cores the op is DRAM-bandwidth-bound (catalog `double_buffer`: 64 cores reached about DRAM peak untuned), so 8 is probably fine. On small grids or BH a deeper batch may help. | Issue `k` blocks of reads before one barrier (`trid_double_issue`: batch barriers across blocks, depth `cb_blocks/2`), or raise `input_depth_blocks` to 3–4. This keeps the compute quantum at `block_tiles`. |
| SFPU square vs. FPU square | Catalog `compute_fusion`: "never use the SFPU for what the FPU does" (SFPU mul measured 0.58× vs. FPU mul). Here the default keeps Square on the SFPU because it consumes Clamp's DEST result, so FPU square would need an L1 round trip. | Clamp → pack to a new `cb_clamped_tiles` → `square<input(cb_clamped_tiles), output(cb_output_tiles)>` (`convenience.hpp:80`, FPU `BinaryFpu` Mul of the tile with itself). Measure only if a profile shows compute-bound cores. Expect no gain while DRAM-bound. |
| Grid synchronization / occupancy | Tiny tensors (`tensor_tiles` < a few × the core count) give each core 1–2 tiles, so launch and pipeline fill dominate. | Cap the participating cores so each core gets `≥ block_tiles` tiles (`core_grid` knob), versus full participation. |
| Placement / bank stagger | Contiguous ranges start at `start_tile = Σ previous core_tiles`. When `core_tiles` is a multiple of the DRAM bank count, every core starts on the same bank (catalog `bank_stagger`). | Rotate each core's walk start (`bank_stagger`-style), and compare `row_wise=True` vs. `False` (`noc_placement`). |
| Format reconfig | The chain is all bf16, so its reconfig is redundant (catalog `compute_block_size` second lever). It runs only once per core, so the expected effect is about 0. | `DataFormatReconfig::Disabled` on input and output. |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM → L1 (`cb_input_tiles`) | bf16 tiles, 2048 B pages | Reader on NCRISC/NoC0 (`ReaderConfigDescriptor`). `TensorAccessor` over the interleaved input. `block_tiles` async page reads, one barrier per block. | Pages `start_tile .. start_tile+core_tiles−1` |
| `cb_input_tiles` → DEST → `cb_output_tiles` | bf16 in L1, 16-bit DEST | Compute: one `eltwise_chain` (Copy → Clamp → Square → Pack) | `fp32_dest_acc_en=False`, `math_fidelity` irrelevant (SFPU only) |
| L1 (`cb_output_tiles`) → DRAM | bf16 tiles | Writer on BRISC/NoC1 (`WriterConfigDescriptor`). `TensorAccessor` over the interleaved output. Same page indices as the input. | Output page index == input page index (shape-preserving) |

**TARGET axes and the regime table.**

| Support axis value (op_template names) | Phase 0 | Cost class of the unlock | How |
|------------------------------|---------|--------------------------|-----|
| `memory_layout = INTERLEAVED` (DRAM) | built | — | `interleaved_stream` |
| `memory_layout = INTERLEAVED` (L1 buffer) | rejected by the requirement | knob-turn | Same kernels. The TensorAccessor handles the L1 bank mapping. |
| `memory_layout = HEIGHT_SHARDED / WIDTH_SHARDED / BLOCK_SHARDED` | not supported | **knob-turn** for all three, because no axis is dependent | `sharded_resident` row: CBs backed on the shard and no NoC re-read |
| `dtype = float32` | not supported | knob-turn | CB page format Float32, and `fp32_dest_acc_en=True` makes `dest_limit` 4, so the `block_tiles` clamp follows automatically |
| `dtype = bfloat8_b` | not supported | knob-turn | CB page format Bfp8_b. The packer/unpacker handle the shared exponent. |
| `alignment = non_tile_aligned` (TILE) | not supported (the requirement rejects it) | knob-turn | The geometry is already `ceil` + per-image. The op computes on padding, which is never observed in the logical output. Relax the validation. |
| `layout = ROW_MAJOR` | not supported | scheme addition (tilize → chain → untilize inside the same dispatch), or a stick-paged reader | deferred to a refinement |

**Structural impossibilities to fold into the spec's INVALID:** `{dtype: bfloat8_b, layout: ROW_MAJOR}` (block-float has no RM form).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | A tile (the split granularity). Each core walks its range in blocks of `block_tiles`. |
| Grid | `core_grid` (host param, default `device.compute_with_storage_grid_size()` full rectangle) |
| Split | `(num_cores, all_cores, core_group_1, core_group_2, core_tiles_g1, core_tiles_g2) = ttnn.split_work_to_cores(core_grid, tensor_tiles, row_wise=True)` |
| Per-core work | `core_tiles ∈ {core_tiles_g1, core_tiles_g2}`, `start_tile` = running sum over the cores in the order returned by `ttnn.corerange_to_cores(all_cores, row_wise=True)`, `num_blocks_this_core = ceil(core_tiles / block_tiles)` |
| Remainder | Across cores: `split_work_to_cores` gives group 2 one tile fewer. Within a core: the ragged last block has `last_block_extent` valid tiles, with **nominal** CB counts (FullBlock). |
| Tile geometry | `tensor_tiles = prod(shape[:-2]) · ceil(H/32) · ceil(W/32)` (per image, `ceil`, never `floor(batch·H/32)`). Use the logical shape. Phase 0 shapes are aligned, so this is exact. |
| Idle cores | Cores in `core_grid` but not in `all_cores` get no kernels (the kernel core ranges are `all_cores`) |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_tiles` | 0 | `ttnn.tile_size(ttnn.bfloat16)` = 2048 | `input_depth_blocks · block_tiles` (Phase 0: 16) | spans `tile` → `block_tiles` × depth | Float16_b | reader | compute | per block, `load_block` → `clamp_square_block` |
| `cb_output_tiles` | 1 | 2048 | `output_depth_blocks · block_tiles` (Phase 0: 16) | spans `tile` → `block_tiles` × depth | Float16_b | compute | writer | per block, `clamp_square_block` → `store_block` |

There are only two CBs. There is no scaler and no intermediate, because Clamp and Square both act on DEST. The full ledger is in `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `[block_tiles]` pages (with `valid` real reads) | dataflow (TensorAccessor) | DRAM input | `cb_input_tiles`, reserve + push `block_tiles` | `block_tiles` pages visible to compute |
| 2 | `clamp_square_block` | `[block_tiles]` DEST lanes | **helper**: `eltwise_chain` | `cb_input_tiles`, wait + pop `block_tiles` (PerBlockSize, FullBlock) | `cb_output_tiles`, reserve + push `block_tiles` (PerBlockSize, FullBlock) | input freed, output visible to writer |
| 3 | `store_block` | `[block_tiles]` pages (with `valid` real writes) | dataflow (TensorAccessor) | `cb_output_tiles`, wait + pop `block_tiles` | DRAM output | output slot freed |

Compute realization (intended, one call per core, after `compute_kernel_hw_startup(cb_input_tiles, cb_output_tiles)`):

```
eltwise_chain(IterationShape::tiles(core_tiles).block_size(block_tiles, BlockTailSync::FullBlock),
    CopyTile<input(cb_input_tiles, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block)>{in},
    Clamp<>{min_bits, max_bits},
    Square<>{},
    PackTile<output(cb_output_tiles, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{out});
```

This is a contract sketch, not kernel code. The implementer confirms the policy and mapping cell is legal per the chain's static asserts, and adjusts spelling to the header.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `clamp_square_block` (driver) | helper | `eltwise_chain` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:619-620` | `IterationShape::tiles(core_tiles).block_size(block_tiles, BlockTailSync::FullBlock)` (`chain.hpp:146-173`, `tiles` `:159`, `block_size` `:168`) | — | — | `block_tiles` (≤ `DEST_AUTO_LIMIT`, `chain.hpp:504-508`). `core_tiles` is the assignment, not a knob. |
| load to DEST | helper element | `CopyTile<input(...)>` | `chain.hpp:553-557`. `input(...)` at `chain.hpp:385-391` | `input(cb_input_tiles, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block)`. Policies enum at `chain.hpp:233,236`. | `cb_input_tiles` | — (DEST D0) | sync quantum = `block_tiles` |
| clamp | helper element (SFPU) | `Clamp<Dst::D0>{min_bits, max_bits}` → `clamp_tile` | `eltwise/unary/scalar.inl:31-41`. Declaration `eltwise/unary/scalar.hpp:22`. LLK `tt_metal/hw/inc/api/compute/eltwise_unary/clamp.h:30` (`calculate_clamp`, `ckernel_sfpu_clamp.h:19-27`, fp32 param load `ckernel_sfpu_unary_max_min.h:15`) | runtime `uint32` fp32 bit patterns | DEST | DEST | — |
| square | helper element (SFPU) | `Square<Dst::D0>{}` → `square_tile` | `eltwise/unary/misc.inl:66-70`. Declaration `eltwise/unary/misc.hpp:35`. | — | DEST | DEST | — |
| pack | helper element | `PackTile<output(...)>` | `chain.hpp:587-592`. `output(...)` at `chain.hpp:442-450` | `output(cb_output_tiles, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)` | — | `cb_output_tiles` | sync quantum = `block_tiles` |
| startup | raw_api (required boot, not a phase) | `compute_kernel_hw_startup(cb_input_tiles, cb_output_tiles)` | contract stated at `chain.hpp:24-38` | — | — | — | — |
| `load_block` / `store_block` | dataflow | `TensorAccessor`, `noc_async_read_tile`/`noc_async_write_tile`, CB reserve/push/wait/pop | `tech_reports/tensor_accessor/tensor_accessor.md` | `TensorAccessorArgs` CT args from the host | — | — | `block_tiles` (batch size), `input_depth_blocks`/`output_depth_blocks` |

**Helpers considered and rejected.**
- `unary<SfpuOp, ...>` (`eltwise/api/convenience.hpp:86-87`): it takes **one** SFPU op type, while we need Clamp followed by Square in one DEST window. It also default-constructs the op, so it cannot carry Clamp's runtime `min/max` (`scalar.inl:35`, constructor takes `lo, hi`). Use `eltwise_chain` directly (the convenience header points there for fused chains, `convenience.hpp:28-31`).
- `square<...>` (FPU, `convenience.hpp:80`): its operand must come from an L1 CB, but the clamped value lives in DEST, so it would need an intermediate CB. This is recorded as the SFPU-vs-FPU perf lamp.
- `apply_activation_from_pack` / `sfpu_activation_helpers.hpp`: packer-thread activations, which have no clamp+square. Its `relu_max` (`sfpu_activation_helpers.hpp:94,136`) cannot express an arbitrary lower bound.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Clamp bounds packed as bf16 | The LLK loads the 32-bit RT word as an fp32 constant (`ckernel_sfpu_unary_max_min.h:15`) | Host packs with `struct.pack("<f")`. The acceptance test includes a bound that is not bf16-exact (`min_val=-0.3, max_val=0.7`) in addition to the three required call patterns. |
| Bounds as CT args | A recompile for every `(min, max)` pair, and the program cache thrashes | RT args on compute |
| FullBlock + `block_tiles > DEST_AUTO_LIMIT` | FullBlock does not clamp (`chain.hpp:507-508`), so the DEST lanes overflow silently | Host clamp derived from the same compute-config flags. If `fp32_dest_acc_en` is turned on (the float32 refinement), the limit drops to 4 automatically. |
| Ragged last block | `core_tiles % block_tiles ≠ 0` on most shapes and grids | Nominal push/wait/pop of `block_tiles` on all three kernels. Only `valid` tiles are read, computed and written. The test includes shapes with ragged tails on both WH (64 cores) and BH (≈110–130 cores). |
| `tensor_tiles < num_cores` | `split_work_to_cores` returns fewer cores, and kernels must target only `all_cores` | Kernel core ranges = `all_cores`. The test has a single-tile case. |
| Rank-2 and rank-3 inputs | Leading-dim product is empty or has length 1 | `prod(shape[:-2])` with an empty product of 1. The test covers rank 2 and rank 3. |
| `min_val == max_val` | Legal, and the output is constant | Validation uses a strict `>` |
| L1-interleaved input | Must be rejected (the requirement), even though the kernels would work | Check `buffer_type == DRAM` explicitly, not just `INTERLEAVED` |
