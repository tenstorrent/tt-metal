# Operation Design: permute

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement |
| Goal | Reorder tensor dimensions (`torch.permute` semantics) in a single native TTNN device program, at DRAM bandwidth. |
| Math | `output[i_{dims[0]}, ..., i_{dims[r-1]}] = input[i_0, ..., i_{r-1}]` — pure relabeling, no arithmetic, dtype preserved. |
| Mode | Derivative (generic_op / ProgramDescriptor) |
| References | `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/generic_op_template/`, `ttnn/ttnn/operations/examples/tile_reorder/README.md`, `ttnn/ttnn/operations/examples/double_buffer/README.md`, `ttnn/ttnn/operations/examples/noc_placement/README.md`, `ttnn/ttnn/operations/examples/width_split/README.md`, `tech_reports/tensor_accessor/tensor_accessor.md` |

Phase 0 realizes the **whole-tile relocation** regime: `dims` keeps the innermost two dims in
place (`dims[-1] == r-1` and `dims[-2] == r-2`), so the permutation acts entirely on the *outer*
axes and every 32x32 tile moves intact. There is no compute phase; the program is
reader (NoC0) → CB → writer (NoC1), one dispatch.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | fp32 (Phase 0), TILE, tile-aligned, rank 4, interleaved DRAM | — | tensor |
| `dims` | `tuple[int,...]` | yes | permutation of `range(rank)`; Phase 0 additionally `dims[-1]==r-1 and dims[-2]==r-2` | — | derived → CT/RT |
| `memory_config` | `ttnn.MemoryConfig` | no | interleaved DRAM (Phase 0); L1 sharded deferred (R2) | `input_tensor.memory_config()` (DRAM interleaved) | host only |
| `block_tiles` | host constant | — | `1 .. tiles_per_plane`, clamp §Mechanism caps | **8** | CT arg |
| `buffer_depth` | host constant | — | `>= 2` | **2** | CT arg (via CB size) |
| `grid` | `CoreCoord` | — | `<= device.compute_with_storage_grid_size()` | full grid | host, RT-arg-driven |

`dims` is canonicalized to non-negative form (`d % rank`) by the entry point before the support
check. `swap_hw = (dims[-1] != rank-1)`, `inner_pair = "preserved" if dims[-2]==rank-2 else "moved"`,
`mem = "l1_sharded" if output memory_config is sharded else "dram_interleaved"` — all three are
derived in `validate()`, not taggers. `INPUT_TAGGERS = {"alignment": tag_alignment, "rank": tag_rank}`.

Phase 0 `SUPPORTED`: `dtype=[float32]`, `layout=[TILE]`, `alignment=["tile_aligned"]`, `rank=[4]`,
`swap_hw=[False]`, `inner_pair=["preserved"]`, `mem=["dram_interleaved"]`. `EXCLUSIONS = []`.
`inner_pair` is a validate-only axis: the harness never generates it (it is not in TARGET), but an
external caller passing e.g. `(0,2,1,3)` must be refused with `UnsupportedAxisValue`, because that
permutation *re-tiles* (a non-innermost axis becomes the tiled H axis) and is not whole-tile
relocation. Do not silently accept it.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 4 `(N, C, H, W)`, `H % 32 == 0`, `W % 32 == 0` |
| Dtype | `float32` |
| Layout | TILE |
| Memory | interleaved DRAM |

### Output

| Property | Value |
|----------|-------|
| Shape | `tuple(input.shape[d] for d in dims)` |
| Dtype | same as input |
| Layout | TILE |
| Memory | `memory_config` (Phase 0: interleaved DRAM) |

## Blocking Model

Index space is the **tile** index space: tile coords `(n, c, ht, wt)` with
`tensor_ht_tiles = N_batchdim_independent ceil(H/32)`, `tensor_wt_tiles = ceil(W/32)`.
Total tiles `tensor_tiles = N * C * ceil(H/32) * ceil(W/32)` (per-image `ceil`, never
`floor(N*H/32)`).

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `n` (dim 0) | independent — each tile's destination is a pure function of its own index; no cross-tile dependency | `block_n` | 1 (an outer axis is never contiguous with the next in linear tile order once permuted; extent >1 would break the single-run coalescing the block relies on — explicit reason per §Bounds) | host constant `BLOCK_N = 1` | folded into the linear output-tile range each core owns | knob-turn (only useful if a future regime blocks whole planes) |
| `c` (dim 1) | independent — same reason | `block_c` | 1 (same reason as `block_n`) | host constant `BLOCK_C = 1` | folded into the linear output-tile range | knob-turn |
| `ht` (dim 2, tiled H) | independent, and **contiguity-carrying**: within a plane, `(ht, wt)` is linear in both input and output because `dims` preserves the inner pair | `block_ht` | implicit — `block_tiles` spans `(ht,wt)` jointly as one linear run | `BLOCK_TILES` (single source) | linear output-tile range per core, `split_work_to_cores(row_wise=True)` | knob-turn (raise `BLOCK_TILES`) |
| `wt` (dim 3, tiled W) | independent, contiguity-carrying (innermost, stride 1 in tile order) | `block_wt` | implicit — see `ht`; the pair is blocked as `block_tiles` consecutive tiles | `BLOCK_TILES` | same linear range | knob-turn |

**There is no dependent axis anywhere in this op**: nothing is reduced, scanned or accumulated, so
no cross-core combine exists to design and none is deferred. There is no reuse-shared axis either:
every output tile reads exactly one input tile, so no operand is re-read by more than one core
(operand-reuse check run over the single input against the chosen split: the input *does* vary
along the split axis — it is the split axis — so no broadcast row is needed).

**Stall-shadow check.** The only waits are `cb_wait_front` / `cb_reserve_back` between reader and
writer and the NoC barriers inside each. They are shadowed structurally, not by reordering: the CB
depth (`buffer_depth = 2`) lets the reader fetch block `i+1` while the writer drains block `i`, and
reader on NoC0 / writer on NoC1 (`examples/noc_placement`: 2.5–4.8x vs. reversed) lets the two
directions overlap. `BLOCK_TILES` reads are issued before one barrier so the NoC pipeline stays
full (`examples/double_buffer`: batching + double buffering 6.5 → 17.9 GB/s single core, 2.78x).
No stage computes on one core while others wait — every core does identical streaming work.

**Blocking the intermediate.** The only intermediate is the CB itself; its axes are the same
`(ht,wt)` linear run, blocked identically by `BLOCK_TILES` — re-running the axis table on it adds
no new decision.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_tiles` | `BUFFER_DEPTH` | 2 | reader fetches block `i+1` while writer drains block `i` — the read/write overlap that batching alone cannot deliver (`examples/double_buffer`, "batching alone saturates ~13 GB/s") |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Plane contiguity (`dims` permutes only outer axes) | `block_tiles` must not straddle a `(ht,wt)` plane boundary | `run = min(BLOCK_TILES, tiles_per_plane - offset_in_plane, tiles_left_this_core)` | tiles inside one block would need two different input base offsets → **wrong data**, not an error |
| L1 residency | `BUFFER_DEPTH * BLOCK_TILES * tile_bytes` <= working-set budget | `BLOCK_TILES <= l1_budget / (BUFFER_DEPTH * tile_bytes)` | CB allocation failure at program creation |
| Core assignment | `block_tiles <= tiles_this_core` | `min(...)` as above (ragged last block passes its actual extent) | reads/writes past the core's range → cross-core corruption |
| DRAM page granularity | block is a whole number of **tile pages**; page size is `ttnn.tile_size(dtype)` and must be the tensor's aligned page size | never sub-tile: one NoC transaction per tile page (`examples/tile_reorder`: whole-tile writes >= 4 face writes) | sub-tile faces fragment the transaction and lose DRAM bandwidth |
| `tensor_accessor` page index width | linear tile index `< 2^32` | 32-bit page indices; assert `tensor_tiles < 2^31` on host | index wraparound → silent wrong addresses |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `whole_tile_relocation` | **built** | `layout==TILE and dims[-1]==r-1 and dims[-2]==r-2 and mem==dram_interleaved` | `BLOCK_TILES` consecutive tiles of one `(ht,wt)` plane; `block_n=block_c=1` | **minimum**: input crosses DRAM once, output crosses DRAM once, nothing crosses twice, no cross-core traffic | amortizes: NoC read/write barrier count (`BLOCK_TILES` transactions per barrier, intended once per block), CB reserve/push/wait/pop (once per block per kernel), pipeline fill/drain (once per core). No regime-added term. |
| `sharded_output` | **deferred** — R2; positive reason: Phase 0's SUPPORTED rectangle is `dram_interleaved`, and the built regime already serves every interleaved shape in `INPUTS`; the structure stays reachable because the writer's only coupling to placement is its output CB/accessor, so swapping the output CB for `ttnn.cb_descriptor_from_sharded_tensor(output)` and dropping the writer's DRAM store is a placement change, not a new algorithm | `mem==l1_sharded` | one shard per core, `block_tiles = shard tiles` (sharded default: one block = the whole resident shard) | **below** the interleaved minimum: output never crosses DRAM (halves total DRAM bytes for a same-dtype copy); consumed in place through a CB backed on the sharded buffer, never re-read over the NoC | one block per shard = one barrier set per core |
| `within_tile_transpose` | **deferred** — R3; positive reason: needs a compute phase (`transpose_init`/`transpose_tile`, `tt_metal/hw/inc/api/compute/transpose.h:38,108`) which is an added stage, not a knob; reachable because the reader/CB/writer split and the linear-block schedule are unchanged — a compute kernel is inserted between the two existing CBs | `swap_hw==True` (i.e. `dims[-1]!=r-1`) | `BLOCK_TILES` tiles; destination tile index transposes `(ht,wt)`→`(wt,ht)` | input once, output once (minimum); output tile order is strided in `wt`, so the writer's transactions are per-tile rather than per-run — above the *transaction-count* optimum, not the byte optimum | amortizes the transpose LLK init once per block instead of per tile |
| `row_major_stick` | **deferred** — R5; positive reason: the stick (not the tile) is the transfer unit, so the ceiling and the block extent are computed from `W*element_size`, a different extent ranking; reachable because the block schedule is index-space-agnostic — only the page size and index math change | `layout==ROW_MAJOR` | `BLOCK_STICKS` consecutive sticks | input once, output once | amortizes per-stick transaction overhead; stick size may be below the DRAM-efficient transfer size, which is exactly what R5 must measure |
| `retile` (permutation moving a non-innermost axis into the tiled H/W position) | **deferred** — needed for TARGET `dims` beyond the inner-pair-preserving set; positive reason: no `INPUTS`/`swap_hw` cell generated by the golden harness requires it (`dims_for` only ever produces inner-pair-preserving or inner-pair-swapped permutations), so no shape is left unserved; reachable as a gather regime over the same linear block schedule | `dims[-2] != r-2 and dims[-1] == r-1` | gather of 32 sub-tile rows per output tile | **above** minimum in transaction count (32 face-row reads per output tile), bytes still once each | amortizes the gather setup; the row is the reason this is not Phase 0 work |
| `host_roundtrip_or_multi_dispatch` (untilize → row-major permute → tilize as separate ops) | **rejected** — superseded by `whole_tile_relocation` and `row_major_stick`, which both move each byte exactly once; this regime adds two extra full-tensor DRAM passes (3x the DRAM bytes) and violates the one-native-dispatch rule | — | — | 3x minimum | nothing — a dead end; no good scheme passes through it |

Selection predicate (host, exact), evaluated in order: `layout==ROW_MAJOR → row_major_stick`;
else `dims[-1]!=r-1 → within_tile_transpose`; else `dims[-2]!=r-2 → retile`; else
`output memory_config is sharded → sharded_output`; else `whole_tile_relocation`. Phase 0 supports
only the last; the others raise from `validate()`. **Regime-pinned tests are required** once more
than one regime is built — the acceptance test pins `dims=(1,0,2,3)` and `dims=(0,1,2,3)`
explicitly rather than relying on a shape to select a path.

### Traffic ranking

Bytes are identical under every candidate split (each tile is read once and written once — §4 of
`blocking-model.md`: no overlapping footprints, no shared operand), so the ranking is over
**transaction shape and count**, tier by tier.

| Candidate split | DRAM crossings | Transaction shape | Cross-core | Rank |
|---|---|---|---|---|
| linear **output**-tile range per core (chosen) | in 1x, out 1x | writes are a contiguous run of output tile pages; reads are a contiguous run of input tile pages (inner pair preserved) → both sides run-contiguous | none | **1** |
| linear **input**-tile range per core | in 1x, out 1x | reads contiguous, writes contiguous (symmetric here, because the permutation maps whole runs to whole runs) — equivalent; chosen the output form because the output range also fixes the output shard mapping R2 needs | none | 1 (tie) |
| one plane (`ht*wt` tiles) per core | in 1x, out 1x | maximally contiguous | none | 3 — strands work when `N*C < grid` (e.g. `(1,1,2048,256)` → 1 plane → 1 core); `width_split` measures up to 7.8x for spreading a wide/short tensor instead |
| `wt`-only split (columns of tiles) | in 1x, out 1x | every core's tiles are `tensor_wt_tiles`-strided → no run contiguity | none | 4 |
| split a dependent axis + cross-core combine | n/a | n/a | n/a | **not applicable — the op has no dependent axis** |

Chosen: **linear output-tile range, `row_wise=True`** (`examples/noc_placement`: row placement
~2.9x over the column-major default). It is also the cheapest-traffic split; nothing is deferred on
traffic grounds within the interleaved regime. The one strictly cheaper *scheme* is
`sharded_output` (removes the output DRAM crossing entirely) — a deferred regime row above, and the
R2 lever.

### Block schedule

Logical schedule (reader and writer are separate async kernels; adjacent blocks pipeline through
`cb_tiles` at depth 2):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);    // reader: BLOCK_TILES input tile pages -> cb_tiles, one barrier
    store_block(block_idx);   // writer: cb_tiles -> BLOCK_TILES output tile pages, one barrier
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|---|---|---|---|
| `load_block` | `block_extent = min(BLOCK_TILES, plane_remainder, core_remainder)` tile pages, one contiguous input run | the destination CB pages | 1 `cb_reserve_back` + `block_extent` `noc_async_read` + **1** `noc_async_read_barrier` + 1 `cb_push_back` per block; accessor/base-address setup once per kernel |
| `store_block` | same extent, one contiguous output run | the source CB pages | 1 `cb_wait_front` + `block_extent` `noc_async_write` + **1** `noc_async_write_barrier` + 1 `cb_pop_front` per block; setup once per kernel |

No compute kernel and therefore no LLK init at all in Phase 0. The per-block index math
(linear output tile → `(n,c,ht,wt)` → inverse-permute → linear input tile) is computed **once per
block** for the run base, then incremented, not recomputed per tile.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Extent (transactions per barrier) | `BLOCK_TILES=8` is the measured sweet spot for outstanding reads per barrier (`examples/double_buffer`: 4–8), but the op is fp32 (4 KB tile pages), where fewer, larger transactions may already saturate | `BLOCK_TILES ∈ {4, 8, 16, 32}` at fixed depth 2 |
| Overlap | a large block can serialize read against write inside a core, and depth 2 may be too shallow to hide DRAM latency at full grid | `BUFFER_DEPTH ∈ {2, 3, 4}` at the winning `BLOCK_TILES` |
| Grid synchronization / occupancy | at full 8x8 grid, small tensors (e.g. `(1,1,32,64)` = 2 tiles) leave <1 block per core, so dispatch overhead dominates the transfer | cap `num_cores = min(grid_tiles, ceil(tensor_tiles / MIN_TILES_PER_CORE))` and measure `MIN_TILES_PER_CORE ∈ {1, 8}` |
| NoC assignment | reader-on-NoC0 / writer-on-NoC1 is the default and the measured winner, but this op saturates both directions simultaneously | swap the assignment; also measure `split_reader` (both DM RISCs reading) **only after** confirming issue-bound rather than bandwidth-bound |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| input DRAM → L1 | tiled, `tile_size(dtype)`-byte pages | `TensorAccessor` on the input tensor + `noc_async_read_page`-class calls, NoC0 (reader/NCRISC) | one transaction per tile page; `block_extent` issued per barrier |
| L1 reader → writer | tiled pages | `cb_tiles` (one producer: reader; one consumer: writer) | depth 2 x `BLOCK_TILES` pages |
| L1 → output DRAM | tiled pages | `TensorAccessor` on the output tensor, NoC1 (writer/BRISC) | destination page index = linear **output** tile index, contiguous within the core's run |

Deferred `sharded_output` (R2) contract: the output CB becomes
`ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, output_tensor)`
(`ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:517`, impl `ttnn/core/tensor/tensor_utils.cpp:44`),
the reader writes directly into that zero-copy CB and the writer stage disappears; the core grid is
then dictated by the output shard grid, not by `split_work_to_cores`. No Tensix-to-Tensix
communication is needed by any deferred regime — every output element depends on exactly one input
element, so no mcast/semaphore/ring contract is required anywhere in this op.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: up to `BLOCK_TILES` consecutive **output** tile pages within one `(ht,wt)` plane |
| Grid | `device.compute_with_storage_grid_size()`, via `ttnn.split_work_to_cores(grid, tensor_tiles, row_wise=True)` (`ttnn/cpp/ttnn-nanobind/operations/core.cpp:469`); `row_wise=True` is mandatory (column placement measured ~2.9x slower) |
| Per-core work | a contiguous half-open range `[start_tile, start_tile + tiles_this_core)` of the linear output tile index; `num_blocks_this_core = ceil_div(tiles_this_core, BLOCK_TILES)` |
| Remainder | `split_work_to_cores` returns two core groups (`units_per_core_group_1/2`); the last block of a core passes its **actual** extent to `load_block`/`store_block` (same operation, shorter runtime extent). Tile geometry is `ceil`-based per image: `tensor_tiles = N * C * ceil_div(H,32) * ceil_div(W,32)` — never `floor(N*H/32)`, so R5's non-aligned shapes hit the same formula. |

Per-core RT args: `[in_addr, out_addr, start_tile, tiles_this_core]`; CT args:
`[BLOCK_TILES, rank, out_dim_sizes..., in_stride_per_out_dim...]` then
`TensorAccessorArgs(input)` / `TensorAccessorArgs(output)` **last**. The input tile index for an
output tile is `sum_d out_coord[d] * in_tile_stride[dims[d]]`, so the permutation is entirely
absorbed into a host-computed stride vector — the kernel never branches on `dims`.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_tiles` | 0 | `ttnn.tile_size(input.dtype)` (fp32: 4096 B) | `BUFFER_DEPTH * BLOCK_TILES` (Phase 0: `2*8 = 16`) | live set **spans** the joint `(ht,wt)` block run (`BLOCK_TILES` pages) and **streams over** `n`/`c` (one plane run at a time); the extra factor is the depth knob, i.e. double buffering | `input.dtype` (`Float32`) — pure movement, no DEST involvement, so the page format is exactly the tensor format; widening or narrowing it would corrupt or bloat the copy | reader | writer | whole program |

One CB, one producer, one consumer. Inventory-before-solve: no intermediate CB is added because
there is no compute phase and no format change — the phase boundary between read and write is the
CB itself, and no scratch buffer can be justified. No L1 budget solve, safety fraction or blocking
search is used; the footprint is a closed-form 64 KB at Phase 0 values.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_extent` tile pages (contiguous input run) | no (dataflow; `TensorAccessor` + batched `noc_async_read`) | — (DRAM source) | `cb_tiles`, `block_extent` pages | `block_extent` pages pushed |
| 2 | `store_block` | `block_extent` tile pages (contiguous output run) | no (dataflow; `TensorAccessor` + batched `noc_async_write`) | `cb_tiles`, `block_extent` pages, front-valid | — (DRAM sink) | `block_extent` pages popped |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| host: program | helper | `ttnn.ProgramDescriptor` / `ttnn.CBDescriptor` / `ttnn.CBFormatDescriptor` / `ttnn.KernelDescriptor` | `.claude/references/generic_op_template/template_op_program_descriptor.py:83,131,149,163` | `total_size = BUFFER_DEPTH*BLOCK_TILES*page_size`, `core_ranges = all_cores` | — | `cb_tiles` | `BLOCK_TILES`, `BUFFER_DEPTH` (single source: module-level constants in the program-descriptor file) |
| host: work split | helper | `ttnn.split_work_to_cores` | `ttnn/cpp/ttnn-nanobind/operations/core.cpp:469` | `(grid, tensor_tiles, row_wise=True)` | — | — | grid (core count) |
| host: accessors | helper | `ttnn.TensorAccessorArgs(...).get_compile_time_args()` | `.claude/references/generic_op_template/template_op_program_descriptor.py:109` | input, then output; **appended last** | — | — | — |
| host: dispatch | helper | `ttnn.allocate_tensor_on_device` + `ttnn.generic_op` | `.claude/references/generic_op_template/template_op.py:48,59` | `([input, output], program_descriptor)` — exactly one dispatch | — | — | — |
| `load_block` | raw_api | `TensorAccessor::get_noc_addr(page_idx)` + `noc_async_read` + one `noc_async_read_barrier` | `tech_reports/tensor_accessor/tensor_accessor.md` (API surface); `tt_metal/hw/inc/dataflow_api.h` | `block_extent` reads per barrier | — | `cb_tiles` | `BLOCK_TILES` |
| `store_block` | raw_api | `noc_async_write` + one `noc_async_write_barrier` | `tt_metal/hw/inc/dataflow_api.h` | `block_extent` writes per barrier | `cb_tiles` | — | `BLOCK_TILES` |

**Helpers considered and rejected (for the two `raw_api` dataflow entries).** `ttnn/cpp/ttnn/kernel_lib/`
exposes no dataflow helper for a plain page-indexed DRAM→L1→DRAM stream: `mcast_pipe.hpp`
(`McastArgs`/`SenderPipe`/`ReceiverPipe`) is a NoC-multicast + semaphore handshake, and this op has
no shared operand and no cross-core dependency at all (see §Axes: no reuse-shared axis), so its
sender/receiver contract has nothing to express here. `tilize_helpers_dataflow.hpp` /
`untilize_helpers_dataflow.hpp` convert between stick and tile pages; Phase 0 input and output are
both TILE, so there is no conversion to perform (they become relevant in R5). No **compute** phase
exists in Phase 0, so no compute helper is bypassed. For R3 the compute phase will use
`transpose_init` / `transpose_tile` (`tt_metal/hw/inc/api/compute/transpose.h:38,108`) — verified:
`kernel_lib` contains no transpose wrapper, so raw compute API is the only option there, and
`compute_kernel_hw_startup()` must precede it.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| A `swap_hw=False` permutation that still moves the inner pair (e.g. `(0,2,1,3)`) | `axes.classify_call` derives `swap_hw` from `dims[-1]` only, so such a call lands in the Phase 0 cell but is *not* whole-tile relocation — it would silently produce wrongly re-tiled data | `validate()` derives the extra `inner_pair` gate and raises `UnsupportedAxisValue`; the `retile` regime row records the real algorithm |
| Block straddling a plane boundary | the input base offset changes at the boundary; a run computed from a single base would read the wrong plane | `block_extent` clamped by `tiles_per_plane - offset_in_plane` (Mechanism caps) |
| Identity permute `(0,1,2,3)` | must still be one native dispatch and must still honor `memory_config`, so it cannot be short-circuited to returning the input when placement differs | the same regime handles it with `in_tile_idx == out_tile_idx`; the acceptance test pins it |
| `floor` tile math | with R5's non-aligned shapes, `N*H/32` differs from `N*ceil(H/32)` and every index downstream shifts | `ceil_div` per image from Phase 0 onward, even though Phase 0 only tests aligned shapes |
| Column-major core placement | `split_work_to_cores` defaults to `row_wise=False`, measured ~2.9x slower on a spread line | `row_wise=True` is specified, not optional |
| Sub-tile (face) transfers | a generic implementation naturally writes 4x512 B faces and loses DRAM bandwidth (`examples/tile_reorder`) | whole tile page is the transfer unit, declared as a mechanism cap |
| Per-tile barriers | one barrier per read destroys the NoC pipeline (6.5 vs 17.9 GB/s) | one barrier per block, stated as intended frequency in the Block schedule |

## Structural impossibilities (candidates for a future `/golden-tests` pass)

- `{"rank": 2, "swap_hw": False}` is the **identity** permutation, not a movement case; it is a
  legal cell (and useful as a copy path) but carries no permutation semantics — worth tagging so a
  green identity cell is not read as evidence for the movement path.
- No further INVALID candidates: `dims` is always a permutation by contract, and every
  (dtype, layout, alignment, rank, mem) combination other than the recorded bf8b+ROW_MAJOR cell is
  representable.
