# Operation Design: g6_fill

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (a generator: writes only, reads nothing, no arithmetic on device) |
| Goal | Write one 32-bit element `e = cast_T(fill_value)` into every logical position of `output_tensor` with one device program of this op's own, and return `output_tensor`. The call is a few hundred bytes of writes, so the program's fixed cost is the whole cost, and the design minimises it. |
| Math | `e = cast_T(v)`; `out[i] = e` for every `i in [0, N)`; input values are never read |
| References | contract: task prompt (g6_fill spec, BB rules); `eval/golden_tests/g6_fill/feature_spec.py` (TARGET / INPUTS / LOOSE_CASES); `eval/golden_tests/g6_fill/helpers.py` (oracle `pytorch_g6_fill`, exact grade); `eval/op_template.py`; `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; `.claude/references/noc-alignment-in-tt-metal.md` §1, §6; `.claude/references/noc-alignment-theory.md` §4.1, §5.2, §7.7; `ttnn/ttnn/operations/examples/master.md` entries `width_split`, `noc_placement`, `tile_reorder`; `tt_metal/third_party/tt_ops_code_gen/skills/tooling/references/core-grid.md` ("Decision procedure", launch-dominated threshold and launch stagger, Blackhole p150b 11 x 10 grid, 8 DRAM banks) |

### Catalog evidence used

| Entry | What it measured | Knob it informs here |
|-------|------------------|----------------------|
| `width_split` | One tile-row-tall tensors run at single-core speed when split by tile-rows; splitting the `Wt` tile-columns across `min(Wt, grid)` cores wins, with no benefit only at `Wt = 1` | Core assignment: split `w_tiles` across `min(tensor_w_tiles, max_fill_cores)` cores |
| `noc_placement` | Row placement beats column placement for spread DRAM traffic; writes belong on NoC1 (the `WriterConfigDescriptor` default) | `split_work_to_cores(..., row_wise=True)`; writer kernel on BRISC / NoC1 |
| `tile_reorder` | Whole-page writes beat face-sized writes **on a DRAM-bandwidth-bound move** | Explains why it does **not** apply: this op moves 768–2,048 B per call, far below any bandwidth bound. It is bound by issue and launch costs, so 1/32 of the bytes in 2 x 64 B writes per tile is the right transfer shape (regime R3 rejected) |
| `core-grid.md` (tooling) | Launch-dominated when the kernel span is below 10 x the 600–1,300-cycle firmware overhead; launch stagger 0–624 cycles from 1 to 110 cores | Grid-synchronization perf lamp (fewer cores) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` (positional) | yes | rank 1, `[N]` with `N >= 1`; dtype `float32` / `int32`; on device; DRAM interleaved | — | Shape, dtype and layout set the regime, the core plan and the cache key. Its buffer is never touched. |
| `fill_value` | Python `float` or `int` (positional), never `bool` | yes | float32: finite with `abs(v) <= 3.4028234663852886e38`, or `inf` / `-inf` / NaN; an `int` with `abs(v) <= FLT_MAX` is admitted. int32: `int` in `[-2^31, 2^31 - 1]` only | — | RT: common runtime arg `fill_bits` (the 32-bit pattern of `e`). **Never** a CT arg or define. |
| `output_tensor` | `ttnn.Tensor` (keyword-only, no default) | yes | same logical shape, dtype, layout, memory config and device as `input_tensor`; may be `input_tensor` itself | — | RT: common runtime arg `dst_addr` (`output_tensor.buffer_address()`) |

No `memory_config` and no `compute_kernel_config`: the result is placed where `output_tensor` is, and no device arithmetic exists.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[N]`, rank exactly 1, `N >= 1`. As TILE it is the tile plane `[1, N]` padded to `32 x 32*ceil(N/32)`. |
| Dtype | `ttnn.float32` (Phase 0); `ttnn.int32` (refinement) |
| Layout | TILE (Phase 0); ROW_MAJOR (refinement: one page = the whole row, `4N` bytes) |
| Memory | DRAM interleaved, on device. **Never read**: not passed to `generic_op` and has no accessor in any kernel. |

### Output

| Property | Value |
|----------|-------|
| Shape | `[N]` (= input shape) |
| Dtype | = input dtype |
| Layout | = input layout |
| Memory | = input memory config (DRAM interleaved), same device. The op returns `output_tensor` itself (same buffer). With a separate output, `input_tensor` is untouched bit for bit. |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Everything below is the realization of this section.

**The op's index space (TILE, regime R1).** The tile plane `[1, N]` has one logical row. Tile `t` (`0 <= t < tensor_w_tiles = ceil(N/32)`) holds positions `32t .. 32t+31` in its **row 0**. That row is split across two faces: positions `32t+0..15` are face 0 row 0, at byte offset `0` of the page. Positions `32t+16..31` are face 1 row 0, at byte offset `face_bytes = face_h * face_w * 4 = 1024`. Each of these is a **row-0 segment** of `segment_bytes = face_w * 4 = 64` contiguous bytes. Every other byte of the tile (rows 1–31, faces 2–3) is padding, and the op does not write it.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `w_tiles` (tile index along N) | **independent**: no output position depends on any other position or on any input value | `block_w_tiles` (tiles per block) = `core_w_tiles` (the core's whole assignment); `num_blocks_this_core = 1` | the whole per-core assignment. For every production shape the full-grid spread gives each core exactly **1** tile (`tensor_w_tiles` is 16/8/7/7/6, fewer than the grid's cores). This is the occupancy step, not an L1 limit: the block's L1 footprint does not depend on this extent at all. | host split: `ttnn.split_work_to_cores(grid, tensor_w_tiles, row_wise=True)` → per-core `(start_tile, core_w_tiles)`; `tensor_w_tiles` from `output_tensor.buffer_num_pages()` (asserted `== ttnn.div_up(N, 32)`) | contiguous tile ranges over `num_cores = min(tensor_w_tiles, max_fill_cores)` cores, row-wise from core (0,0); `max_fill_cores` = full compute grid in Phase 0 | knob-turn (`max_fill_cores`; the grid-sync lamp) |
| `row0_segment` (face 0 / face 1 half of row 0) | **independent**: the two 64-B runs of one tile share no data except the constant `e` | `segments_per_tile = tile_w / face_w` | **2** (whole: both segments of every tile, including the last tile, whose face-1 segment may be all padding) | CT arg derived from `output_tensor.tile.tile_shape` and `.face_shape` | not split: a core owns whole tiles. A 64-B write is far below any per-core fixed cost. | knob-turn (not expected) |
| `lane` (16 consecutive elements of one segment) | **independent**: element-wise constant | `segment_elems = face_w` (the NoC transfer unit is one segment) | **16** (whole face row = one 64-B write) | CT arg `segment_bytes = face_w * element_size` | not split | none: fixed by the face geometry |
| `plane_h` (rows of the tile plane) | **independent**; extent **1** because the rank-1 tensor is the plane `[1, N]` | `rows_written` | **1**. This is the tensor's own height, not a collapsed knob. Rows 1–31 are padding, and the contract lets the op leave them unwritten. | implied by rank 1 (validated) | not split | none (rank-1 contract) |
| `rm_bytes` (positions along the single ROW_MAJOR page; regime R2, deferred) | **independent** | `rm_chunk_bytes` (bytes per NoC write = bytes of the L1 fill run) | refinement value `min(align_up(4N, 16), RM_CHUNK_CAP_BYTES)` with `RM_CHUNK_CAP_BYTES = 2048`. Production `[1]`: one chunk, write length 4 B. | host constant `RM_CHUNK_CAP_BYTES` → CT arg `rm_chunk_bytes`; `num_rm_chunks = ceil(4N / rm_chunk_bytes)` | contiguous chunk ranges over `min(num_rm_chunks, max_fill_cores)` cores (production `[1]`: 1 core) | knob-turn |

**Operand-reuse check (for the chosen `w_tiles` split).**

| Operand | Varies along `w_tiles`? | Consequence |
|---------|------------------------|-------------|
| `fill_value` (`e`) | no: one scalar for every position | **Reuse-shared by construction.** It is broadcast as a *common* runtime arg, which the dispatcher writes once per core range (multicast). Each core materialises the 64-B run locally with 16 word stores. There is zero NoC data traffic and no injector. |
| `input_tensor` | — | never read (contract): zero bytes, zero crossings |

**No dependent axis exists.** No result spans more than one position, so no split ever needs a combine.

**Stall-shadow check.** The only stage that waits is `drain_writes`, the final write barrier waiting for DRAM acks. Every other piece of work, building the fill row and issuing every write, must precede it and already does. Nothing independent of the wait remains to fill that window: the kernel returns as soon as the barrier clears.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_fill_row` | `fill_row_depth` | 1 | Nothing. The content is constant, written once per core and read by every segment write, with no producer/consumer cycle, refill or overlap to buy. A depth above 1 is a defect. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| One-packet NoC write (`noc_async_write_one_packet`, `dataflow_api.h:841`) | `segment_bytes` (R1), `rm_chunk_bytes` (R2) `<= NOC_MAX_BURST_SIZE` | R1: `segment_bytes = 64`, fixed by geometry. R2: `RM_CHUNK_CAP_BYTES = 2048`, well below the WH and BH burst size. Host asserts `rm_chunk_bytes <= NOC_MAX_BURST_SIZE`. | a one-packet write larger than a burst is malformed (silent corruption or hang) |
| Blackhole DRAM write rule: local L1 offset ≡ NoC offset (mod 16) (`noc_parameters.h` `NOC_DRAM_WRITE_ALIGNMENT_BYTES 16`; `noc-alignment-in-tt-metal.md` §1) | segment start offsets inside a page | R1 offsets are `{0, face_bytes = 1024}`, both ≡ 0 mod 16, and the `cb_fill_row` base is CB-allocated, so 16-B aligned. R2 chunk starts are `k * rm_chunk_bytes`, so `rm_chunk_bytes % 16 == 0`. Only the ragged last chunk's **length** may be any size. | silent data corruption (the watcher stops it under `--dev`) |
| Tile / face geometry | `segments_per_tile`, `segment_bytes`, `face_bytes` | derive all three from `output_tensor.tile.tile_shape` / `.face_shape` and the element size. Host asserts tile `(32, 32)`, face `(16, 16)` and element size 4. | positions 16–31 of each tile land in the padding of face 0 instead of face 1 row 0: wrong result for every `N > 16` |
| Core count | `num_cores <= grid_x * grid_y` | `num_cores = min(tensor_w_tiles, max_fill_cores)`, `max_fill_cores <= grid cores` | launch on non-existent cores is refused |
| Page index width (`uint32_t` page id, `TensorAccessor::get_noc_addr`) | `tensor_w_tiles` | `tensor_w_tiles < 2^32` holds for every allocatable tensor; host asserts it | page id wraps and writes the wrong page |
| Profiler marker budget (250 markers per RISC) | zones per core | 3 zones, each executed **once per core**, never inside the tile loop: 6 markers | zones past the budget silently vanish (BB-21 attribution breaks) |

### Regimes

Selection predicate (host, exact): `regime = "R1_tile_row0_segments" if input_tensor.layout == ttnn.TILE_LAYOUT else "R2_rm_chunks"`. It depends on layout only, never on grid size or free L1. Both regimes are dtype-agnostic because they move 32-bit patterns.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 tile row-0 segment scatter** | **built** | `layout == TILE` (all Phase 0 cells: float32 TILE `[512]/[256]/[207]/[196]/[185]`) | `block_w_tiles x segments_per_tile x segment_elems` = `core_w_tiles x 2 x 16`, one block per core | Minimum against the DRAM boundary: each logical output byte (`4N`) is written once and the input is read 0 times. Actual: `128 * tensor_w_tiles` B written (2,048 / 1,024 / 896 / 896 / 768 B for 512 / 256 / 207 / 196 / 185) and 0 B read. The excess over `4N` is padding in the last tile's row 0 (`4 * (32*tensor_w_tiles - N) < 128` B), which is unavoidable at segment granularity. Each segment is one 64-B-aligned, 64-B DRAM burst. | Paid once per block, which is once per core: the RT-arg fetch, accessor construction, the 16-store fill-row build, zone entry/exit and **one** write barrier. No inits, no CB handshakes, no reconfigs exist. A bigger block (fewer cores) amortises these over more tiles; that is exactly the grid-sync lamp. |
| **R2 row-major chunk scatter** | **deferred**: int32 / ROW_MAJOR is outside Phase 0 SUPPORTED by the contract (the refinement). R1's structure keeps it reachable: the same writer and fill-row CB, with the unit "chunk of page 0 at offset `k * rm_chunk_bytes`" instead of "tile page at offsets `{0, face_bytes}`", and a ragged last-chunk length. That is a CT-geometry change plus host split, not a new loop nest. | `layout == ROW_MAJOR` (production: int32 `[1]`) | `core_rm_chunks x rm_chunk_bytes`, one block per core | Minimum: `4N` B written once, 0 read. Actual: exactly `4N` B (`[1]`: 4 B), which is the minimum. | Same terms as R1, plus fill-row stores proportional to `rm_chunk_bytes / 4` (once per core) |
| R3 whole-tile write | **rejected**, superseded by R1 | — | `core_w_tiles` x one 4,096-B page | `4096 * tensor_w_tiles` B (65,536 B for `[512]`): 32 x R1's bytes. It needs a 4-KB fill tile (1,024 RISC stores, or a 6-step NoC doubling with barriers) and halves the write count from `2W` to `W`. A contiguous `[0, face_bytes + 64)` span is the same trade at 17 x the bytes. | — |
| R4 compute-kernel fill (`fill_tile` → pack → writer) | **rejected**, superseded by R1 | — | — | Same writes as R1 or R3, plus a compute kernel (three more RISCs to launch), a CB handoff, and DEST/packer handling of `e`. Subnormals (`1e-40`) and NaN payloads are not guaranteed to survive SFPU/pack, and `fill_tile` takes a `float` (`fill.h:29`). The contract demands all 32 bits survive. | — |
| R5 host-built upload / read-modify-write | **rejected**: forbidden by the contract | — | — | uploads `4N` from the host, or reads the output/input; both are MUST NOTs | — |
| R6 injector + multicast of the fill row | **rejected**, superseded by R1's common-RT-arg broadcast | — | — | adds `64 * (num_cores - 1)` B of NoC payload plus a semaphore round trip, to deliver a value the dispatcher already places on every core in the common RT args | — |

Regime-pinned tests: R1 is pinned by every float32 TILE case of the acceptance test. R2 is pinned by the int32 ROW_MAJOR `[1]` cases, which xfail on a support refusal in Phase 0 and must pass after the refinement. Selection does not depend on the grid, so no regime triggers on only some devices.

### Traffic ranking

| Candidate split | DRAM writes | DRAM reads | Cross-core NoC | Fan-out | Verdict |
|-----------------|-------------|------------|----------------|---------|---------|
| `w_tiles` across `min(tensor_w_tiles, max_fill_cores)` cores (**chosen**) | `128 * tensor_w_tiles` B | 0 | 0 | each core writes its own tiles' banks | cheapest (tie) and fills the grid |
| no split (1 core) | same | 0 | 0 | one NIU issues all `2W` writes | ties on bytes; loses occupancy |
| `row0_segment` across cores | same | 0 | 0 | 2 x the cores for 64-B units | ties on bytes; doubles per-core fixed cost |
| dependent axis + combine | — | — | — | — | not applicable: no dependent axis |
| injector + fill-row mcast (R6) | same | 0 | `+64 * (num_cores - 1)` B + semaphores | 1 → all | strictly more traffic |

Every split moves identical DRAM bytes, so traffic is invariant under the split. The tiebreaker (independent axis, full occupancy) picks the `w_tiles` split. The occupancy-versus-launch trade is a perf lamp, not a traffic question.

### Block schedule

```cpp
build_fill_row();                                         // once per core: e -> cb_fill_row (segment_elems words)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // num_blocks_this_core == 1
    scatter_row0_block(block_idx);                        // block_w_tiles tiles x segments_per_tile writes, all sourced from cb_fill_row
}
drain_writes();                                           // once per core: every write acked before the kernel returns
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `build_fill_row` | `1 x segment_elems` (16 x 32-bit) | `cb_fill_row` holds `e` x 16 from here to kernel exit | once per core: 1 common-arg read, `segment_elems` word stores. Zone `G6FILL-BUILD-ROW`. |
| `scatter_row0_block` | `block_w_tiles x segments_per_tile x segment_elems` | `cb_fill_row` (read-only NoC source, never refilled, never popped) | accessor constructed once per core. Per tile: one page-address lookup and `segments_per_tile` one-packet writes. No barrier, CB handshake or init inside. Zone `G6FILL-ISSUE` around the whole loop. |
| `drain_writes` | all writes of the core | — | **one** `noc_async_write_barrier` per core (BB-03). Zone `G6FILL-DRAIN`. |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Grid synchronization** (core count) | Per-core work is 2 one-packet writes, far below the 600–1,300-cycle firmware overhead per launch, and launch stagger grows with core count (`core-grid.md`). The op is launch-dominated by that file's own threshold, so full occupancy may lose to fewer cores. | `max_fill_cores` in {1, 2, 4, 8} against the default `min(W, grid)`, on the PERF-FOCUS case `[512]` (LOOSE_CASES[0]) and `[207]`, by device kernel duration. If fewer cores wins, also measure splitting each core's tiles across BRISC + NCRISC (two writer kernels, disjoint tile halves). |
| Per-core unique RT args | Each core's `(start_tile, core_w_tiles)` is a separate dispatcher write per call and per trace replay. Common args are one write per core range. | Derive `start_tile` / `core_w_tiles` in-kernel from `get_relative_logical_x/y()` (`dataflow_api.h:70`) and common args `(tensor_w_tiles, num_cores, cores_x)`, so the kernel has zero unique RT args. |
| Fill-row source | `cb_fill_row` costs a CB config entry per launch plus 16 RISC stores per core. | Put `fill_bits` 16 times in the common RT args, starting at a 16-B-aligned index (common-arg offsets are L1-aligned, `dispatch.cpp` `configure_crta_offsets_for_kernel_groups`), and source each write from `get_common_arg_addr(idx)` (`dataflow_api.h:127`). That removes the CB and the stores. The implementer must verify the alignment on device before adopting it. |
| Write issue path | Plain one-packet writes recompute the full NoC command per segment. | `noc_async_write_one_packet_set_state` once per tile (same bank, same size) followed by 2 `noc_async_write_one_packet_with_state` (`dataflow_api.h:982/1008`). This only matters once `block_w_tiles >> 1` (the fewer-cores arm of the lamp above). |
| Overlap | not applicable: there is no compute stage and no CB pipeline to overlap (empty claim) | — |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Host: `fill_value` → `e` | 32-bit pattern `fill_bits` | Python conversion, once per call (see Host Contract) | float32 rounds **once** from the exact value; int32 exact |
| Host → every core | `fill_bits`, `dst_addr` as common RT args; `(start_tile, core_w_tiles)` as unique RT args | dispatcher: common args multicast per core range | the cached descriptor is patched with the two common words per call |
| Writer (BRISC, NoC1): build | `e` x `segment_elems` in `cb_fill_row` | RISC word stores into `get_write_ptr(cb_fill_row)` | once per core, before any write is issued |
| Writer: scatter | 64-B runs | `TensorAccessor::get_noc_addr(tile, offset)` for `offset in {0, face_bytes}` → `noc_async_write_one_packet(cb_fill_row_addr, noc_addr, segment_bytes)` | non-posted writes, so each is acked |
| Writer: drain | — | `noc_async_write_barrier()` | the kernel returns only after every write has landed (BB-03) |
| `input_tensor` | — | none | never passed to `generic_op`, no accessor, no NoC transaction |

No regime, built or deferred, needs inter-Tensix communication: no semaphore, multicast or ring. TARGET has no placement axis, because the contract fixes DRAM interleaved. The logical shards are the per-core contiguous tile ranges of the `w_tiles` split, and no physical-shard regime exists. **Structural impossibilities:** none beyond `INVALID = []`. Both regimes accept either 32-bit dtype, so a non-production pairing (int32 TILE, float32 ROW_MAJOR) is still computable.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | R1: one tile page (both row-0 segments). R2 (deferred): one `rm_chunk_bytes` chunk of page 0. |
| Grid | `device.compute_with_storage_grid_size()` (Blackhole p150b: 11 x 10). `max_fill_cores` = `grid_x * grid_y` (host constant `MAX_FILL_CORES = None` → full grid; part of the cache key). |
| Per-core work | `ttnn.split_work_to_cores(grid, tensor_w_tiles, row_wise=True)` → `core_group_1` gets `ceil` and `core_group_2` gets `floor` tiles. Contiguous `start_tile` in core order. One kernel descriptor over `all_cores`. |
| Remainder | `tensor_w_tiles = ceil(N/32)` (per image: rank 1 has one image). The last tile holds `r = N - 32*(tensor_w_tiles - 1)` logical positions (`[207]`: 15, `[196]`: 4, `[185]`: 25), and both of its segments are written regardless. Ragged core split: `split_work_to_cores` groups (none for production shapes). |

Production plans (BH 11 x 10, row-wise, 1 tile per core):

| Shape | `tensor_w_tiles` | Cores | Core ranges (BB-17) | Bytes written |
|-------|------------------|-------|---------------------|---------------|
| `[512]` | 16 | 16 | `(0,0)-(10,0)`, `(0,1)-(4,1)` | 2,048 |
| `[256]` | 8 | 8 | `(0,0)-(7,0)` | 1,024 |
| `[207]` | 7 | 7 | `(0,0)-(6,0)` | 896 |
| `[196]` | 7 | 7 | `(0,0)-(6,0)` | 896 |
| `[185]` | 6 | 6 | `(0,0)-(5,0)` | 768 |
| `[1]` int32 RM (R2, refinement) | 1 chunk | 1 | `(0,0)` | 4 |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_fill_row` | 0 | R1: `segment_bytes = face_w * element_size` (64). R2: `rm_chunk_bytes`. | `fill_row_depth` (1) | Spans `lane` (one whole face row of `e`). Streams `w_tiles`, `row0_segment` and `plane_h`: the same constant row is the source of every write. | output dtype (`Float32` in Phase 0; `Int32` in R2): a dataflow-only payload that never passes through DEST | writer (RISC stores, once) | writer (NoC source of every write; same kernel thread, so there is no reserve/push/wait/pop) | whole kernel |

The ledger, with capacity vs. live set, axis accounting and sharing, is `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `build_fill_row` | `1 x segment_elems` | no (raw stores; no helper writes an arbitrary 32-bit constant, see API Mapping) | — (`fill_bits` from common RT arg) | `cb_fill_row`, 1 page | holds `e` x `segment_elems` until kernel exit; no push |
| 2 | `scatter_row0_block` | `block_w_tiles x segments_per_tile x segment_elems` | no (raw NoC; gap recorded below) | `cb_fill_row`, 1 page, read-only | DRAM pages `[start_tile, start_tile + core_w_tiles)` of `output_tensor` at offsets `{0, face_bytes}` | unchanged |
| 3 | `drain_writes` | every write of the core | no | — | — | every write acked |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `build_fill_row` | raw_api | `get_write_ptr(cb_fill_row)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:325` | CB index | — | `cb_fill_row` | — |
| `build_fill_row` | raw_api | `get_common_arg_val<uint32_t>(idx)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:168` | `fill_bits`, `dst_addr` | — | — | — |
| `build_fill_row` | raw_api | `volatile tt_l1_ptr uint32_t*` word stores (`addr_to_l1_ptr`) | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:30` | `segment_elems` stores | — | `cb_fill_row` | `segment_elems` |
| `scatter_row0_block` | raw_api | `get_arg_val<uint32_t>(idx)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:148` | `start_tile`, `core_w_tiles` | — | — | `block_w_tiles = core_w_tiles` |
| `scatter_row0_block` | raw_api | `TensorAccessorArgs<...>()` / `TensorAccessor(args, dst_addr, page_size)` | `tt_metal/hw/inc/api/tensor/tensor_accessor_args.h:21`, `tt_metal/hw/inc/api/tensor/tensor_accessor.h:71` | output accessor CT args from `ttnn.TensorAccessorArgs(output_tensor)` | — | DRAM | — |
| `scatter_row0_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | `page_id = tile`, `offset in {0, face_bytes}` | — | DRAM | `segments_per_tile` |
| `scatter_row0_block` | raw_api | `noc_async_write_one_packet(src_l1, dst_noc, size)` (or `noc_async_write<segment_bytes>`, `:876`) | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:841` | `size = segment_bytes`; non-posted | `cb_fill_row` | DRAM | `segment_bytes` |
| `drain_writes` | raw_api | `noc_async_write_barrier()` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1951` | — | — | — | — |

**Helpers considered and rejected** (they apply to every `raw_api` row above):

| Candidate | File:Line | Why it cannot express this shape |
|-----------|-----------|----------------------------------|
| `dataflow_kernel_lib::zero_tile` / `prepare_zero_tile` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:53` (`async_write_zeros`, `:55`) | writes zeros only, over a whole DFB entry in local L1. It cannot write an arbitrary 32-bit `e` and cannot target DRAM pages. |
| `prepare_reduce_scaler` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:70` | fills a CB tile with a reduce-specific scaler pattern (pool-type/face layout, 16-bit packing). Wrong geometry, and it does not carry a 32-bit pattern. |
| local L1 copy helpers (`set_read_state` / `read_with_state`) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:75`, `:95` | self-aimed L1 → L1 reads. The destination here is interleaved DRAM pages. |
| `Mcast` / `McastArgs` | `ttnn/cpp/ttnn/kernel_lib/mcast/` | core-to-core multicast. The scalar already reaches every core as a common RT arg (R6 rejected). |
| tilize / untilize / eltwise / reduce / matmul helpers | `ttnn/cpp/ttnn/kernel_lib/*_helpers.hpp` | compute helpers. This op has no compute phase (R4 rejected). |

Gap, recorded the way a raw-LLK bypass is: no kernel_lib dataflow helper writes **one constant L1 run to a set of `(page, in-page offset)` destinations of an interleaved tensor**. A helper `scatter_constant_run(accessor, src_l1, first_page, num_pages, offsets[], run_bytes)` would close it, and both R1 and R2 would be calls to it. Not required for Phase 0.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Positions 16–31 written as a contiguous run | Row 0 of a tile is not contiguous in memory: positions 16–31 sit in face 1 at byte 1024. A 128-B run from byte 0 writes face-0 rows 0–1 and leaves 16–31 unwritten. | Two segments per tile at `{0, face_bytes}`, derived from `tile.face_shape`. `[185]` (25 positions in the last tile) and every full tile detect it. |
| A new `fill_value` replays the cached program's old value | `fill_bits` is the only thing that changes. If it were a CT arg or define it would be hashed (`generic_op_device_operation.cpp:83-101` hashes CT args and defines); if absent from RT args it would never be refreshed. | `fill_bits` and `dst_addr` are plain-`uint32` **common runtime args**. On a cache hit, `generic_op` rebuilds runtime args from the passed descriptor (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:958-962` → `tt_metal/impl/program/program_descriptors.cpp:188-216`, which copies per-core and common args). Do **not** bind the output by `Buffer*` / `emplace_runtime_args`: that switches to the fast path, which patches addresses only (`mesh_device_operation_adapter.hpp:936-939`) and would replay the first value. |
| Patching the cached Python descriptor does nothing | `ProgramDescriptor.kernels` is a by-value `std::vector` property (`ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:1137`), so `desc.kernels[0].common_runtime_args[i] = x` mutates a temporary copy | keep the `KernelDescriptor` object, patch its `common_runtime_args`, then reassign `desc.kernels = [kernel]` before `ttnn.generic_op`. The repeat-value test (`1.0, 1.0, 0.1, 1.0`) detects a failed patch. |
| int → float32 double rounding | `struct.pack('<f', float(v))` rounds the int to float64, then to float32. `v = 2**60 + 2**36 + 1` gives `2**60`; the contract (one rounding from the exact value) gives `2**60 + 2**37`. Verified on host: torch's `int64 → float32` cast rounds once. | For an `int` with `abs(v) > 2**53`, round the integer directly to 24 significant bits, ties to even (Host Contract). |
| Over-range floats slip through | `struct.pack('<f', 3.4028235e38)` silently returns FLT_MAX (verified), but the contract refuses every finite `abs(v) > 3.4028234663852886e38`. | Explicit range check before packing. `inf` / `-inf` / NaN are admitted. |
| `bool` accepted as an int | `isinstance(True, int)` is true | check `isinstance(v, bool)` first and raise ValueError |
| `-0.0` / NaN normalised | The contract needs `0x80000000` and `0x7fc00000` bit for bit; a `v == 0` shortcut or a float-to-int path loses the sign bit | pack the Python float directly (`struct` matches torch for `1/3, -0.0, 1e-40, inf, nan, 0.1, FLT_MAX`, verified) |
| Fill row read by the NoC before the stores land | The NoC reads `cb_fill_row` straight from L1. A stale row holds the previous call's value at the same CB address. | build the row before the first write is issued. The `[1.0, 1.0, 0.1, 1.0]` repeat case is the detector. Under `--dev`, a stale row shows as `0x3f800000` where `0x3dcccccd` is expected. |
| Writes in flight at return | BB-03: the next program starts immediately. Posted writes or `noc_async_writes_flushed` guarantee departure only. | non-posted writes plus one `noc_async_write_barrier()` as the kernel's last action |
| Contract refusal mistaken for support refusal | `UnsupportedAxisValue` subclasses `NotImplementedError`, a `RuntimeError`. If SUPPORTED is checked first, a bfloat16 input or a float for int32 raises a support refusal, which the harness reads as "not yet supported" instead of a refusal. | `validate()` runs every contract check (ValueError) **before** the SUPPORTED/EXCLUSIONS gate |
| In-place aliasing | `output_tensor is input_tensor` | `ttnn.generic_op([output_tensor], desc)`: one io tensor, so no aliasing ambiguity in binding resolution, and the input is never an operand |
| Allocation under trace (BB-16) | the model calls inside a captured trace | no `allocate_tensor_on_device`, no host tensor build, nothing compiled after warm-up; the descriptor cache is warm before capture |

## Host Contract

### Registry declarations (op file `g6_fill.py`)

| Item | Phase 0 value |
|------|---------------|
| `INPUT_TAGGERS` (in this order) | `tag_dtype(inputs, axes) -> inputs[0]["dtype"]`; `tag_layout(inputs, axes) -> inputs[0]["layout"]`; `tag_alignment(inputs, axes) -> "w_non_aligned" if inputs[0]["shape"][-1] % 32 else "h_non_aligned"` (the rank-1 shape read as the plane `[1, N]`; `"tile_aligned"` never occurs) |
| `SUPPORTED` | `{"dtype": [ttnn.float32], "layout": [ttnn.TILE_LAYOUT], "alignment": ["w_non_aligned", "h_non_aligned"]}` |
| `EXCLUSIONS` | `[]`. Never list `"w_non_aligned"` and never split a cell by size. |

### `validate(input_tensor, fill_value, output_tensor)`: first line of `g6_fill`

| Order | Check | Raises |
|-------|-------|--------|
| 1 | `input_tensor` is a `ttnn.Tensor`, on device, `len(shape) == 1`, `N >= 1`, dtype in {float32, int32} | ValueError |
| 2 | input memory config is DRAM + INTERLEAVED | ValueError |
| 3 | `fill_value` is `int` or `float` and not `bool`. int32: not a `float`, and in `[-2**31, 2**31 - 1]`. float32: not finite with `abs(v) > 3.4028234663852886e38` (ints compared exactly against `int(FLT_MAX)`) | ValueError |
| 4 | `output_tensor` is not `None`; is a `ttnn.Tensor` on the same device; same logical shape, dtype, layout and memory config as `input_tensor` | ValueError |
| 5 | build `(scenario,)` with keys `shape`, `dtype`, `layout`, `fill_value`, then apply `INPUT_TAGGERS` → SUPPORTED per axis → EXCLUSIONS | `UnsupportedAxisValue` / `ExcludedCell` |

A refusal happens before any dispatch, so `output_tensor` is untouched.

### Element conversion `e`

| Case | Rule |
|------|------|
| float32, `float` v | `fill_bits = struct.unpack('<I', struct.pack('<f', v))[0]` after check 3 (one rounding from the exact double, ties to even; `-0.0`, inf, NaN and subnormals preserved) |
| float32, `int` v, `abs(v) <= 2**53` | `float(v)` is exact, then as above |
| float32, `int` v, `abs(v) > 2**53` | exact integer rounding: `b = abs(v).bit_length()`, `shift = b - 24`, `q, r = divmod(abs(v), 1 << shift)`; round `q` up if `r > half` or (`r == half` and `q` odd); if `q == 1 << 24`, then `q >>= 1` and `shift += 1`. `float(q << shift)` is now exact; pack it with the sign. |
| int32, `int` v | `fill_bits = v & 0xFFFFFFFF` |

### Descriptor cache and dispatch (BB-01, BB-07, BB-16)

| Item | Decision |
|------|----------|
| Cache key | `(device id, tuple(shape), dtype, layout, memory_config, MAX_FILL_CORES)` → a cached `(ProgramDescriptor, writer KernelDescriptor)` |
| Per call | patch the writer's common RT args `[dst_addr, fill_bits]`, reassign `desc.kernels`, then `ttnn.generic_op([output_tensor], desc)` and return `output_tensor` |
| Dispatch count | exactly one `generic_op`. No `to_layout` / `pad` / `slice` / `typecast` / allocation. |
| Mesh | `generic_op` runs the same descriptor SPMD on every device. `dst_addr` is identical across the mesh for the model's replicated buffers. |

### Plan and role report (BB-08, BB-21)

| Item | Decision |
|------|----------|
| `g6_fill_plan(input_tensor) -> dict` (module-level, no dispatch) | `{"regime": "R1_tile_row0_segments" or "R2_rm_chunks", "num_cores", "core_ranges", "tiles_per_core" (or "chunks_per_core"), "segment_bytes", "segments_per_tile", "roles": {core: "fill_writer"}}`. It uses the same function the descriptor build uses, so the report cannot drift from the dispatched plan. |
| Zones | `G6FILL-BUILD-ROW`, `G6FILL-ISSUE`, `G6FILL-DRAIN` (`DeviceZoneScopedN`, executed once per core). They compile out of the plain build. |
