# Operation Design: g6_repeat

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no arithmetic) |
| Goal | Materialize `x.repeat(reps)` as a new tensor in one device program, bit for bit. Phase 0 is the production case: x `(1,1,1,W)`, reps `[1,1,R,1]`, `1 <= R < 32`, giving out `(1,1,R,W)`, one partial tile row. |
| Math | `out[i0,i1,i2,i3] = x[i0 mod d0, i1 mod d1, i2 mod d2, i3 mod d3]`; Phase 0: `out[0,0,h,w] = x[0,0,0,w]`, `h < R`, `w < W` |
| Tile form (F1 of `math.md`) | X_w is the input tile at tile column w, and only its row 0 is logical. B_w is X_w's row 0 copied into rows `0..R-1`. Out tile (0, w) = B_w. Rows `R..31` of B_w are out padding and may hold anything. |
| References | `docs/explorations/g6_repeat/{DOSSIER,math,tensix,risks}.md`; `eval/golden_tests/g6_repeat/{feature_spec,helpers}.py`; `.claude/references/{blocking-model,l1-footprint-discipline,noc-alignment-in-tt-metal,noc-alignment-theory}.md`; `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp`; catalog `ttnn/ttnn/operations/examples/{width_split,double_buffer,noc_placement,tile_reorder,split_reader}/README.md` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | `(1,1,1,W)`, `W % 32 == 0`, bfloat16, TILE, DRAM interleaved, on device | — | host |
| `reps` | `list` / `tuple` of 4 `int`, positional | yes | `[1, 1, R, 1]`, `1 <= R < 32` | — | host; R is lowered to CT `rows_to_fill` |
| `rows_to_fill` (derived) | int | — | `min(d2 * reps[2], tile_h)` = R in Phase 0 | — | CT |
| `block_width_tiles` (derived knob) | int | — | `1 .. max_block_width_tiles` | see Blocking Model | CT |
| `tensor_out_wt_tiles`, `tensor_out_ht_tiles` (derived) | int | — | `ceil(W_out/32)`; `d0*r0*d1*r1*ceil(H_out/32)` (= 1) | — | CT |
| `face_row_bytes`, `face_bytes`, `row0_faces_bytes`, `face_cols` (derived from `input_tensor.tile`) | int | — | bf16 32x32: 32, 512, 1024, 2 | — | CT |
| x / out buffer addresses | int | — | — | — | common RT (same value on every core) |
| `start_tile_col`, `core_w_tiles` | int | — | per core | — | per-core RT |

No `memory_config`, no output dtype, no `compute_kernel_config`. out takes the dtype, layout and placement of x.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(1, 1, 1, W)`, `W % 32 == 0` (one logical row; rows 1..31 of every tile are padding, content unspecified) |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | DRAM interleaved, on device |

### Output

| Property | Value |
|----------|-------|
| Shape | `(d0*reps[0], d1*reps[1], d2*reps[2], d3*reps[3])` = `(1, 1, R, W)` |
| Dtype | dtype of x (bfloat16) |
| Layout | layout of x (TILE) |
| Memory | memory config of x (DRAM interleaved); a new allocation, never a view of x |

## Blocking Model

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `out_batch` (out dims 0, 1 folded) | independent: no result spans it. Extent is 1 by contract (`d0 = d1 = 1`, `reps[0] = reps[1] = 1`). | `block_batch` | 1 = tensor extent. Validation pins it; there is nothing larger to block. | host constant in `g6_repeat_program_descriptor.py`; folds into `tensor_out_ht_tiles` (out page walk) | not split (extent 1) | knob-turn: it joins the tile-row page walk, and x varies along it, so it stays independent |
| `out_tile_row` (out dim 2, in tile rows) | independent; **x is reuse-shared along it**: no result spans it, and x does not vary along dim 2 because `d2 = 1`. | `block_height_tiles` | 1 = `tensor_out_ht_tiles` (R < 32 by contract) | host constant, derived from the validated shape; CT `tensor_out_ht_tiles` | not split (extent 1) | scheme-change (regimes 2 and 3): one built tile is written to every tile row |
| `row_in_tile` (the R logical rows inside a tile; sub-tile, not a page axis) | independent; **x reuse-shared**: every row h reads `x[0,0,0,:]`. | `rows_to_fill` | R, the whole logical extent. One slot holds the full tile, so the block spans every row. | CT `rows_to_fill = min(H_out, tile_h)`, computed once on the host | not split. A split would cut one page between two cores (regime 7, rejected). | none (always resident whole) |
| `out_tile_col` (out dim 3 = x dim 3, in tile columns) | independent: tile column w of out reads tile w of x only, and both x and out vary along it. | `block_width_tiles` | `min(core_w_tiles_max, max_block_width_tiles)` = the core's **whole assignment**. p1: 1. | host, in `g6_repeat_program_descriptor.py`: `block_width_tiles = min(ceil(tensor_out_wt_tiles / num_cores), max_block_width_tiles)`, passed as CT. The CB capacity and the kernel trip counts derive from it. | **split across the grid**: contiguous tile-column ranges, `split_work_to_cores(core_set, tensor_out_wt_tiles, row_wise=True)` over `min(tensor_out_wt_tiles, max_num_cores)` cores. p1: 8 cores, 1 tile each, one row rectangle. | knob-turn (`MAX_NUM_CORES`, `MAX_BLOCK_WIDTH_TILES`) |

Derived stage data: the built tile B_w is the only intermediate. It has the same axes as one out page, and it is built, stored and released inside its owning core's block, so no stage is left on one core while the others wait. No core coordinates, collects or hands out work: every active core runs the same three block operations on its own columns.

**Knob single sources.** Everything lives in `ttnn/ttnn/operations/g6_repeat/g6_repeat_program_descriptor.py`:

- `REPEAT_BUFFER_DEPTH = 1`
- `L1_BUDGET_BYTES = 1347776` (BB-05)
- `MAX_BLOCK_WIDTH_TILES = None`. None means the L1 cap.
- `MAX_NUM_CORES = None`. None means the full `device.compute_with_storage_grid_size()`.

Derived from them:

- `max_block_width_tiles = floor(L1_BUDGET_BYTES / (REPEAT_BUFFER_DEPTH * out_page_size))` (= 658 at 2048 B pages), further clamped by `MAX_BLOCK_WIDTH_TILES` when it is set.
- `num_cores`, `block_width_tiles`, the CB `total_size`, and every kernel trip count.

No second literal exists anywhere: the kernel receives `block_width_tiles` and the tile and face byte sizes as CT args and never restates 1024 / 512 / 32 / 16.

**Operand-reuse check (chosen split: `out_tile_col`).**

| Operand | Varies along `out_tile_col`? | Result |
|---------|------------------------------|--------|
| x | yes: core k needs only x tiles `[start_k, start_k + core_w_tiles)` | no re-read and no broadcast row. x's reuse along `row_in_tile` / `out_tile_row` is captured inside L1: row 0 is read from DRAM once and copied R-1 times locally. |
| out | yes | each page is written by exactly one core |

**Stall-shadow check.** The `replicate_row0_block` op waits on the DRAM read of row 0, about one DRAM read latency, and the store waits on the replicate.

- Nothing that depends on the data can enter that window. The copies read row 0, and the DRAM read also writes rows 1..15 of the slot with x padding, so a copy that landed early would be overwritten by padding (Key Risks R-A).
- What runs in the shadow:
  - all `block_width_tiles` reads of a block are issued before its single barrier, so their latencies overlap;
  - out page ids and write addresses are computed between issue and barrier.
- At p1 (1 tile per core) that is all there is.
- No algebraic reorder is involved (there is no arithmetic), so there is nothing to gate on precision.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_repeat_tiles` | `REPEAT_BUFFER_DEPTH` | 1 | Depth 2 lets block b's page writes drain while block b+1 is loaded and replicated into the other slot set. That overlap exists only when `num_blocks_this_core > 1`, which needs `tensor_out_wt_tiles > 658 * num_cores`; no golden shape and no realistic W gets there. Every golden shape has one block per core, so depth 1 leaves nothing to overlap and keeps the L1 cap at its maximum. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Blackhole DRAM read alignment: local and NoC addresses must match modulo 64 (`NOC_DRAM_READ_ALIGNMENT_BYTES 64`, `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`; rule in `noc-alignment-in-tt-metal.md:43`) | where the partial-page read of x starts and lands | The read starts at **page offset 0** and lands at the **slot base**. CB bases are allocated at DRAM alignment (`tt_metal/impl/program/program.cpp:2505-2509`) and slots step by `out_page_size` (2048, a multiple of 64), so both ends are 0 mod 64. Never read a single face row from DRAM into a 32 B sub-offset. | silent corruption (the watcher stops the core under `--dev`) |
| L1 to L1 NoC alignment: both ends 16 B (`noc_parameters.h:375-376`) | `face_row_bytes` (the loopback copy unit) and its destinations | host asserts `face_row_bytes % 16 == 0` (32 for bf16). Destinations are `slot + k*face_row_bytes (+ c*face_bytes)`, all 16 B aligned. | corrupted rows |
| Stateful single-packet read: `transfer_size <= NOC_MAX_BURST_SIZE` (`static_assert`, `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.inl:66`) | `face_row_bytes` | compile-time; 32 B is far below the burst size | compile error |
| One tile per slot | `rows_to_fill` | `rows_to_fill = min(H_out, tile_h)`; validation refuses R >= 32 | copies run past the slot into the next tile's slot and corrupt it |
| L1 residency, BB-05: `REPEAT_BUFFER_DEPTH * block_width_tiles * out_page_size <= 1347776` | `block_width_tiles` | `max_block_width_tiles` above; cores with more columns than that take `num_blocks_this_core = ceil(core_w_tiles / block_width_tiles)` blocks | CB allocation fails at launch (OOM) |
| Slot reuse across blocks: the NoC reads the write source after issue | the next block's loads into the same slot set | `noc_async_writes_flushed()` before block b+`REPEAT_BUFFER_DEPTH` loads into the slot set block b used | the next block's DRAM data overwrites a page that is still being sent: wrong out bits, nondeterministic (BB-06) |

### Regimes

| Regime | Status | Predicate | Block | Data movement (vs. minimum) | What a bigger block buys |
|--------|--------|-----------|-------|-----------------------------|--------------------------|
| 1. `column_split_row_broadcast`: out tile columns spread over the grid; each core loads the row-0 faces of its x tiles into the out-tile slots, replicates row 0 in place by NoC loopback, and writes each page once | **built** | the Phase 0 contract, checked by `validate()`: rank 4, `x.shape == (1,1,1,W)`, `W % 32 == 0`, `reps == [1,1,R,1]`, `1 <= R < 32`, bfloat16, TILE, DRAM interleaved. Every accepted call lands here, and every golden shape is in it. | `block_batch (1) x block_height_tiles (1) x block_width_tiles`, the whole core assignment, all `rows_to_fill` rows | Boundary: DRAM. **out**: each page crosses once as a whole 2048 B write and is never read back. That is the minimum. **x**: each page crosses once, partially: its two upper faces (1024 of 2048 B, containing row 0). That is the minimum transaction count, one per page (`noc-alignment-theory.md` §7.1). It is above the byte minimum (64 B per page at the 64 B alignment) by design (perf lamp L3). **Cross-core**: none. **Core-local loopback**: `2*(R-1)*face_row_bytes` per tile (960 B at R = 16). | Paid once per block: two read barriers (after the loads, after the loopback copies), one `writes_flushed`, and one exposed DRAM read latency (all loads of a block are in flight together). With one block per core, each is paid once per core. |
| 2. `multi_tile_row_broadcast`: out has 2 or more tile rows (`R >= 32`, or `reps[0]`/`reps[1] > 1`) | **deferred**: outside the contract. `validate()` refuses `R >= 32` and `reps[0..1] != 1`, no production case has it, and TARGET excludes it. | `tensor_out_ht_tiles > 1` | `block_height_tiles x block_width_tiles` | x crosses once per column. The built tile is written `tensor_out_ht_tiles` times from the same slot; x's reuse along tile rows is held in L1, not re-read. out crosses once. | Reachable without restructuring: `rows_to_fill = min(H_out, tile_h)` already fills a whole tile when `H_out >= 32`, and `store_block` already walks page ids `t*tensor_out_wt_tiles + w` for `t < tensor_out_ht_tiles` (1 iteration in Phase 0). |
| 3. `tile_row_split_x_broadcast`: regime 2 with out tile rows also split across cores | **deferred**: needs regime 2, which is outside the contract. It is a scheme-change. | `tensor_out_ht_tiles > 1` and the column split under-fills the grid | per core: `block_height_tiles` rows x `block_width_tiles` columns | Naive form: each extra core-row group re-reads the same x upper faces, adding one x crossing per group (x is reuse-shared along tile rows). Successor: one injector reads and builds B_w, then multicasts it to the row group (`ttnn.Mcast` + `McastArgs`), with x crossing once. | Fewer multicast rounds per built tile. |
| 4. `compute_row_broadcast`: unpack with row broadcast (`compute_kernel_lib::unary_bcast<BroadcastDim::Row,…>`), then pack | **rejected**: values pass through SrcB, DEST and the packer, and "SrcB is only 19bits wide" (`tt_metal/hw/inc/api/compute/bcast.h:86`). Bit identity of NaN payloads and denormals is not established (risk R2), while the contract gate is bit-exact. It would also add a compute kernel (3 TRISC binaries, BB-18) and 2 CBs. Superseded by regime 1: NoC byte moves cannot change bits. | — | — | same DRAM crossings plus an extra L1 round trip through the unpacker and packer | — |
| 5. `single_core` | **rejected**: Phase 0 must be multi-core. The bytes are the same as regime 1, with all per-tile work serialized on one core. Catalog `width_split/README.md:33-41` (Wormhole, Wt = 8): 4292 ns on 1 core vs 1919 ns on 8 cores. Superseded by regime 1. | — | — | same as regime 1 | — |
| 6. `row0_scatter_to_dram`: no L1 replication; write row 0 from L1 straight into each out-row offset (DRAM writes compare modulo 16) | **rejected**: `2*R` DRAM write transactions of 32 B per page instead of one. Each page would be assembled from 2R pieces, which is at odds with "every page of out written once". Catalog `tile_reorder/README.md:3-5`: whole-page writes beat sub-page writes. Superseded by regime 1. | — | — | out bytes are minimal, but out transactions are 32x | — |
| 7. `intra_tile_row_split`: split the R rows of one out tile row across cores | **rejected**: two cores would each write part of one page, and x's page would be read twice. Superseded by regime 1, which splits whole tiles. | — | — | x crosses twice and each out page is written in pieces | — |

Only regime 1 is built, so there is one plan. The plan report (BB-08) is constant: `"column_split_row_broadcast"`, plus `num_cores`, `block_width_tiles` and `num_blocks` for the call. There is no dependent axis, so "split the dependent axis and combine across cores" does not exist for this op.

### Traffic ranking

| Rank | Candidate split | x DRAM crossings | out DRAM crossings | Cross-core | Verdict |
|------|-----------------|------------------|--------------------|------------|---------|
| 1 | `out_tile_col` across cores (regime 1) | 1 per page, upper half (1024 B) | 1 per page (write) | 0 | **chosen**: the minimum on every tier |
| 2 | no split, single core (regime 5) | same | same | 0 | same bytes, but the per-tile issue, loopback and barrier work is serialized |
| 3 | rows inside the tile row across cores (regime 7) | 2 per page | 1 per page, in pieces | 0 | more bytes and split pages |
| 4 | `out_tile_row` across cores (regime 3, outside the contract) | 1 per page per core-row group (naive), or 1 + multicast | 1 | B_w multicast when the multicast successor is used | deferred |
| — | dependent axis + combine | — | — | — | none exists |

The cheapest-traffic split is the one implemented. Occupancy is then set by the interleaved default: spread across the grid (8 cores at p1), then take the coarsest block (the whole assignment).

### Block schedule

```cpp
// one kernel, `g6_repeat_dataflow` (data-movement, NOC_0), on every active core
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_row0_faces_block(block_idx);   // x upper faces -> out-tile slots
    replicate_row0_block(block_idx);    // row 0 -> rows 1..rows_to_fill-1, in place
    store_block(block_idx);             // slots -> out pages, each written once
}
// kernel end: noc_async_write_barrier() (BB-03)
```

| Block op | Block shape | Resident across it | Intended frequency of fixed costs |
|----------|-------------|--------------------|-----------------------------------|
| `load_row0_faces_block` | `block_extent` tile columns (`block_width_tiles`, or `last_block_extent` for a ragged last block) | the slot set `(block_idx % REPEAT_BUFFER_DEPTH)` of `cb_repeat_tiles`. For `block_idx >= REPEAT_BUFFER_DEPTH`, the writes that used this slot set must have flushed first. | `block_extent` DRAM read issues, each `row0_faces_bytes` from x page `start_tile_col + block_idx*block_width_tiles + j` at offset 0 into slot j. **One** `noc_async_read_barrier` per block. TensorAccessor construction once per kernel. |
| `replicate_row0_block` | the same `block_extent` tiles x `rows_to_fill` rows x `face_cols` face columns | slots (now holding x's upper faces) | Skipped entirely when `rows_to_fill == 1`. Per tile and per face column c: one `set_read_state<face_row_bytes>` (the source `slot_j + c*face_bytes` changes per tile and per c, so per-tile state is the minimum), then `rows_to_fill - 1` `read_with_state` issues. Their destinations are logical rows `r = 1..rows_to_fill-1` at `slot_j + (r / face_h)*face_cols*face_bytes + c*face_bytes + (r % face_h)*face_row_bytes`. **One** read barrier per block, after the last issue. |
| `store_block` | `block_extent` tiles x `tensor_out_ht_tiles` (1) | slots (built tiles B_w) | `block_extent * tensor_out_ht_tiles` whole-page `noc_async_write_page` issues to out page `t*tensor_out_wt_tiles + w`. `noc_async_writes_flushed` once per block, only when a later block will reuse the slot set. `noc_async_write_barrier` once per kernel. No inits, no semaphores, no multicast. |

The ragged last block keeps the nominal slot layout and changes only its trip count. There are no CB push/pop counts to keep nominal: the CB is single-thread workspace (see Circular Buffers).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 **grid synchronization**: 8 cores x 1 tile at p1 | Per-core work is one read, about 30 loopback issues and one write. The launch and core-range costs (BB-17, `lessons.md` RB-036) could be close to the work they parallelize. Catalog `width_split/README.md:33-41` favours 8 cores at Wt = 8 on Wormhole with a compute kernel; this op has none. | `MAX_NUM_CORES` = 4, 2, 1 (2, 4, 8 tiles per core, one block each) |
| L2 replicate realization | The default is helper loopback copies: `2*(R-1)` stateful NoC issues per tile (`noc-alignment-theory.md` §5.2 and §7.7 prefer NoC copies over RISC stores). On a data-movement RISC, 30 issues might cost more than the word stores they replace. | (a) RISC word copies: 16 loads + `2*(R-1)*8` stores per tile, after the same barrier. (b) Doubling NoC copies: rows `[0,k)` to `[k,2k)`, `log2(R)` rounds per face column, one barrier per round. |
| L3 x read size | One 1024 B read per tile brings 960 B of padding the op does not use. The default minimizes transaction count (§7.1). | two 32 B reads per tile, row 0 of faces 0 and 1, each landing at the slot offset 0 / `face_bytes` (both 0 mod 64) |
| L4 RISC / NoC placement | A single kernel puts reads, loopback copies and writes on NOC_0. The catalog (`noc_placement/README.md:35-42`) measures reads on NOC_0 with writes on NOC_1 at 2.5-4.8x, but says the effect "nearly vanishes" when latency-bound at 1 tile (README:30-32), which is p1. | (a) the same kernel on RISCV_0 / NOC_1. (b) a reader (NCRISC/NOC_0: load + replicate) and writer (BRISC/NOC_1: store) pair, with `cb_repeat_tiles` as a real handoff and `REPEAT_BUFFER_DEPTH = 2`, for multi-tile cores. |
| L5 **overlap** | When `core_w_tiles` is large (not p1), one whole-assignment block serializes load, then replicate, then store per core. | `block_width_tiles` of 4-8 (catalog `double_buffer/README.md:77-79`: plateau at 4-8 reads in flight) with `REPEAT_BUFFER_DEPTH = 2`, or per-tile transaction ids (`set_read_trid` / `async_read_barrier_with_trid`, `local_copy_helpers_dataflow.hpp:147,158`) so tile j's write issues while tile j+1 replicates |
| L6 write size | At `R <= 16` faces 2-3 of every out tile are padding, and the default writes the whole page. That only matters when bandwidth-bound (large W). | write only `row0_faces_bytes` per page when `rows_to_fill <= face_h`. Each page is still written once; out padding is unspecified (contract §7). |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x page (DRAM) to slot j, bytes `[0, row0_faces_bytes)` | raw bf16 tile bytes: faces 0 and 1 (row 0 plus padding rows 1..15) | `noc_async_read(x_accessor.get_noc_addr(page, 0), slot_j, row0_faces_bytes)` | Lands directly in the out-tile slot (Rule 3 pattern 1: pack into the destination). There is no input CB. Both ends are 0 mod 64. |
| slot j row 0 to rows `1..rows_to_fill-1` (faces 0/1, and 2/3 when `R > 16`) | raw bytes, `face_row_bytes` per copy | self-aimed stateful NoC reads (`dataflow_kernel_lib::set_read_state` / `read_with_state`) | In place (Rule 3 pattern 2). Must follow the load barrier. Rows `>= rows_to_fill` keep whatever they hold (x padding or stale L1); that is out padding. |
| slot j to out page (DRAM) | whole 2048 B tile page | `noc_async_write_page(page, out_accessor, slot_j)` | Each out page is written once and never read. BB-03: `noc_async_write_barrier` before return. |
| Inter-Tensix | — | none | No cross-core contract is needed by any built or deferred-reachable regime except regime 3's multicast successor, where one injector per row group builds B_w and runs `ttnn.Mcast` (fixed sender) to the group's receivers. |

**TARGET mapping.** TARGET has three axes: dtype, layout and alignment, one value each. All three are in Phase 0, so the refinement queue is empty. TARGET has no placement axis. If one were added:

- WIDTH_SHARDED out would cut `out_tile_col`, which is independent: a knob-turn. The slots would be backed on the out shard through `cb_descriptor_from_sharded_tensor`, and the store would become a no-op.
- A cut along `out_tile_row` is regime 3, a scheme-change.

**Structural impossibilities.** None to add; INVALID stays empty.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block: `block_width_tiles` contiguous out tile columns of tile row 0 (`tensor_out_ht_tiles = 1`) |
| Grid | `core_set = ttnn.num_cores_to_corerangeset(min(tensor_out_wt_tiles, max_num_cores), device.compute_with_storage_grid_size(), row_wise=True)`, then `ttnn.split_work_to_cores(core_set, tensor_out_wt_tiles, row_wise=True)`. `max_num_cores` is `MAX_NUM_CORES`, or the full grid when that is None. |
| Per-core work | core k: `core_w_tiles = n1` (group 1) or `n2` (group 2) columns starting at `start_tile_col`, which is the running sum over earlier cores in the same enumeration order; `num_blocks_this_core = ceil(core_w_tiles / block_width_tiles)`; `last_block_extent = core_w_tiles - (num_blocks_this_core - 1) * block_width_tiles` |
| Remainder | `split_work_to_cores` puts the extra column on group 1. Tile counts are alignment-aware: `tensor_out_wt_tiles = ceil(W_out / tile_w)`, `tensor_out_ht_tiles = d0*r0*d1*r1*ceil(H_out / tile_h)` (per image, never `floor(total_rows / 32)`). |
| p1 | 8 columns over 8 cores (0,0)-(7,0): one rectangular core range, 1 tile and 1 block per core, `block_width_tiles = 1`, CB = 2048 B |
| Kernel groups | one kernel (`g6_repeat_dataflow`, `ttnn.ReaderConfigDescriptor()`: RISCV_1 / NOC_0; no compute kernel and no second data-movement kernel) on `all_cores`; groups 1 and 2 differ only in RT args. Ranges: 1 at p1; at most a full-row rectangle plus a partial row in general (BB-17). |
| Regime selection | single regime, so there is no predicate beyond `validate()` |

### Host contract (validation, caching, plan report)

| Item | Decision |
|------|----------|
| Files | `__init__.py` (exports `g6_repeat`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`); `g6_repeat.py` (entry point, taggers, `validate()`); `g6_repeat_program_descriptor.py` (knob constants, descriptor builder, descriptor cache); `kernels/g6_repeat_dataflow.cpp` |
| `tag_alignment(inputs, axes)` | `s = inputs[0]`; `out = [d * r for d, r in zip(s["shape"], s["reps"])]`. Returns `"w_non_aligned"` if `out[-1] % 32`, else `"h_non_aligned"` if `out[-2] % 32`, else `"tile_aligned"`. Production gives `"h_non_aligned"`. |
| `SUPPORTED` / `EXCLUSIONS` | `{"dtype": [ttnn.bfloat16], "layout": [ttnn.TILE_LAYOUT], "alignment": ["h_non_aligned"]}` / `[]` |
| `validate(input_tensor, reps)`, first line of `g6_repeat` | (1) contract checks, raising `ValueError`: rank 4; `len(reps) == 4`; every rep `> 0`; on device; DRAM interleaved; `shape[:3] == (1,1,1)`; `reps[0] == reps[1] == reps[3] == 1`; `reps[2] < 32`. (2) SUPPORTED per axis, raising `UnsupportedAxisValue`; this is where `W % 32 != 0` lands, as `"w_non_aligned"`. (3) EXCLUSIONS, raising `ExcludedCell`. The contract checks run first so `R >= 32` raises `ValueError`, as the spec requires, rather than `NotImplementedError`. The storage check (`ttnn.is_tensor_storage_on_device`) precedes any call that needs a device tensor, so a host tensor gets the op's own `ValueError`. **Every refusal message starts with `"g6_repeat: "`**: the acceptance test matches `^g6_repeat:` through the `expect_error` fixture. |
| BB-07 | The module-level descriptor cache is keyed by `(tuple(x.shape), tuple(reps), dtype, layout, memory_config, device id, grid size, REPEAT_BUFFER_DEPTH, L1_BUDGET_BYTES, MAX_BLOCK_WIDTH_TILES, MAX_NUM_CORES)`. A hit only refreshes the x / out buffer-address common RT args, then calls `ttnn.generic_op([x, out], desc)`. |
| Output | `ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), x.dtype, x.layout, device, x.memory_config())`. Allocation is the only host-side tensor work: no pad, slice, to_layout or reshape (BB-02, rule 3). |
| BB-08 / BB-21 | A plan-report function returns the regime name, `num_cores`, `block_width_tiles` and per-core `(start_tile_col, core_w_tiles)`, which also serve as the role dump (every active core has role `repeat`). Kernel zones per block op go in the zone build only. |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_repeat_tiles` | 0 | `out_page_size` (`out.buffer_page_size()`, 2048) | `REPEAT_BUFFER_DEPTH * block_width_tiles` | Spans `out_tile_col` (`block_width_tiles`) and `row_in_tile` (a whole tile per slot). Streams over `out_tile_row` (one built tile serves every tile row; extent 1) and `out_batch` (extent 1). | bfloat16 (`x.dtype`); a dataflow-only payload that never passes through DEST | `g6_repeat_dataflow` (NoC read + loopback copies) | `g6_repeat_dataflow` (NoC write source) | per block: load, replicate, store. The slot set is reused after `writes_flushed`. |

`cb_repeat_tiles` is **single-thread workspace**. One RISC owns it: it never calls `cb_reserve_back`, `cb_push_back`, `cb_wait_front` or `cb_pop_front` (0 pushes = 0 waits). Its base comes from `get_write_ptr(cb_repeat_tiles)`, and slot j of set s sits at `base + (s*block_width_tiles + j)*out_page_size`. Ordering comes from program order plus the NoC barriers in the Block schedule. No second thread touches it, so the ownership invariant holds trivially. If lamp L4(b) is adopted, it becomes a real reader-to-writer handoff with push/pop of `block_extent` per block.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_row0_faces_block` | `block_extent` tile columns, `row0_faces_bytes` each | raw dataflow (no page-stream helper in `kernel_lib`) | x (DRAM), via TensorAccessor | `cb_repeat_tiles`, `block_extent` slots of set `block_idx % REPEAT_BUFFER_DEPTH` | slots hold x faces 0-1 (row 0 logical, rows 1..15 x padding) and are landed (read barrier, which also invalidates the Blackhole L1 cache, `dataflow_api.h:1932`) |
| 2 | `replicate_row0_block` | `block_extent` tiles x `rows_to_fill` rows x `face_cols` | **helper** `dataflow_kernel_lib` local copy | `cb_repeat_tiles`, the same slots, landed | `cb_repeat_tiles`, the same slots (in place) | slots hold B_w: rows `0..rows_to_fill-1` equal x row 0, bit for bit, and are landed (read barrier) |
| 3 | `store_block` | `block_extent` tiles x `tensor_out_ht_tiles` | raw dataflow | `cb_repeat_tiles`, the same slots, built | out (DRAM), pages `t*tensor_out_wt_tiles + w` | slot set free after `writes_flushed`; all writes acked at kernel end |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| all | raw_api | `TensorAccessor` + `TensorAccessorArgs` (x, out; interleaved, address as RT) | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:71`; `tt_metal/hw/inc/api/tensor/tensor_accessor_args.h:21` | CT args appended last on the host (`ttnn.TensorAccessorArgs(t).get_compile_time_args()`) | — | — | none |
| all | raw_api | `get_write_ptr(cb_repeat_tiles)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:325` | slot base | `cb_repeat_tiles` | — | slot stride `out_page_size`; slot-set stride `block_width_tiles` |
| 1 | raw_api | `noc_async_read(src_noc_addr, dst_l1, size)` with `x_accessor.get_noc_addr(page_id, 0)` | `dataflow_api.h:547`; `tensor_accessor.h:165` | `size = row0_faces_bytes` | x | `cb_repeat_tiles` | trip count `block_extent` |
| 1, 2 | raw_api | `noc_async_read_barrier()` (the helper's `noc.async_read_barrier()` is equivalent) | `dataflow_api.h:1921-1935` | — | — | — | once per block op |
| 2 | helper | `dataflow_kernel_lib::set_read_state<transfer_size>(Noc, src_addr)` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:75` (impl `.inl:62-70`) | `transfer_size = face_row_bytes` (CT); `src = slot_j + c*face_bytes` | `cb_repeat_tiles` | — | none (`face_row_bytes` is a tile-format constant) |
| 2 | helper | `dataflow_kernel_lib::read_with_state<transfer_size>(Noc, dst_addr, src_addr)` | `local_copy_helpers_dataflow.hpp:95` (impl `.inl:72-77`) | `dst` = row-r address of face column c (Block schedule) | `cb_repeat_tiles` | `cb_repeat_tiles` | `rows_to_fill` (trip count) |
| 3 | raw_api | `noc_async_write_page(id, out_accessor, src_l1)` | `dataflow_api.h:1418` | whole page (size 0, so the accessor page size applies) | `cb_repeat_tiles` | out | trip count `block_extent * tensor_out_ht_tiles` |
| 3 | raw_api | `noc_async_writes_flushed()` / `noc_async_write_barrier()` | `dataflow_api.h:1980` / `dataflow_api.h:1951` | — | — | — | per block (only before slot reuse) / per kernel |

**Helpers considered and rejected, per raw entry:**

| Raw entry | Candidate | File:Line | Concrete mismatch |
|-----------|-----------|-----------|-------------------|
| load (1) | `dataflow_kernel_lib::read_sticks_for_tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:87-88` | It reads row-major sticks into a CB for tilization. x is TILE and the op needs a partial-page (`row0_faces_bytes`) read of tile pages. `kernel_lib` has no TensorAccessor tile-page streaming helper; only `tilize_helpers_dataflow` and `ccl/*` use TensorAccessor. |
| load (1) | `noc_async_read_page` (accessor overload) | `dataflow_api.h:1243-1249` | It has no size argument and always reads the accessor's full aligned page (2048 B). The partial read needs the raw `noc_async_read` with an explicit size. |
| store (3) | `dataflow_kernel_lib::write_sticks_after_untilize` | `tilize_helpers_dataflow.hpp:129-130` | It writes row-major sticks after untilize; out is TILE pages. |
| barriers / accessor | — | — | These are primitives with no wrapping helper; the helper's own examples call them directly (`local_copy_helpers_dataflow.inl:28,50`). |
| replicate (2), compute alternative | `compute_kernel_lib::unary_bcast<BroadcastDim::Row, …>` | `ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp:33-34`; `tt_metal/hw/inc/api/compute/bcast.h:86` | Rejected as regime 4: the path through SrcB, DEST and the packer is not proven bit-preserving, the gate is bit-exact, and it would add a compute kernel and 2 CBs. |

No compute kernel exists, so `compute_kernel_hw_startup()` does not apply.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| R-A: a loopback copy issued before the load barrier | The DRAM load writes rows 1..15 of faces 0-1 with x **padding**. A copy that lands first is overwritten by padding, and its source row 0 may not be there yet. The result is wrong, nondeterministic logical rows (R1, BB-06), and it is invisible with zero padding. | `replicate_row0_block` starts strictly after the block's `noc_async_read_barrier`. The golden poisoned-padding cases (1000.0 and NaN) catch a violation. |
| R-B: whole-tile copy of x | A plain copy of x's tile puts padding rows into logical rows 1..R-1 (rule 2). | Every logical row r >= 1 is written by the replicate from row 0; only row 0 comes from the load. |
| R-C: write issued before the loopback barrier | The NoC write reads the slot while copies are in flight, so rows go out stale. | `store_block` issues only after the replicate's read barrier. With `rows_to_fill == 1` there are no copies and the load barrier suffices. |
| R-D: slot reuse across blocks | With depth 1 the next block's load overwrites slots that are still being written out. | `noc_async_writes_flushed()` before block b+`REPEAT_BUFFER_DEPTH` loads into the set block b used. This only triggers when `num_blocks_this_core > 1`; the acceptance test pins that path through `MAX_BLOCK_WIDTH_TILES`. |
| R-E: R > 16 crosses into faces 2/3 | Rows 16..R-1 sit in faces 2 and 3 at `+face_cols*face_bytes`, and their sources are still row 0 of faces 0 and 1. | Destination formula in the Block schedule. The acceptance test runs R = 17 and R = 31. |
| R-F: Blackhole RISC-V L1 cache | Blackhole data-movement cores have a small write-through L1 cache (`tt_metal/hw/inc/internal/tt-1xx/cache.h:7-15`). | The design does no RISC-V loads of NoC-written data; every byte moves by NoC. If lamp L2(a) (RISC copies) is adopted, its loads must follow `noc_async_read_barrier()`, which already invalidates (`dataflow_api.h:1932`). On Blackhole, RISC stores are ordered before a later NoC command (only Quasar needs a fence, `dataflow_api.h:2178-2183`). |
| R-G: kernel returns with writes in flight | The next model program starts immediately (BB-03). | `noc_async_write_barrier()` as the last statement on every core, including cores whose last block is ragged. |
| R-H: hard-coded tile geometry | 1024 / 512 / 32 only hold for bf16 32x32 tiles with 16x16 faces. | Every byte size is a CT arg derived from `x.tile.tile_shape`, `x.tile.face_shape` and `x.element_size()`. The host asserts `face_row_bytes % 16 == 0`. |
| R-I: stale descriptor on a cache hit | A cached descriptor with an old buffer address writes into a freed tensor (`repeat_determinism` case, BB-07). | The cache key covers every program-shaping value, and the addresses are refreshed on every call. |
| R-J: validation order | `reps[2] = 32` tags `"tile_aligned"` and would raise `NotImplementedError` if SUPPORTED ran first; the spec demands `ValueError`. | Contract checks run before the SUPPORTED / EXCLUSIONS checks inside `validate()`. |

## Acceptance test scope (BB-22)

The tests are the production case plus the cases its rules require. The planner's default "4 shapes incl. multi-batch" is replaced:

| Requirement | Coverage |
|-------------|----------|
| Multi-batch | outside the contract, so it is a refusal test |
| The Phase 0 claim "every W % 32 == 0, 1 <= R < 32" | `(W=32, R=1)`: no replicate branch. `(W=64, R=17)` and `(W=96, R=31)`: faces 2/3. `(W=9600, R=5)`: 300 columns, more than any Blackhole grid, so multi-tile blocks and both core groups. |
| Rule 1 (bit identity) | special values |
| Rule 2 (padding independence) | poisoned padding, 1000.0 and NaN |
| BB-06 / BB-07 | repeat determinism |
| BB-05 / R-D | the multi-block path via `MAX_BLOCK_WIDTH_TILES` / `MAX_NUM_CORES` |

There is no dtype or layout sweep.
