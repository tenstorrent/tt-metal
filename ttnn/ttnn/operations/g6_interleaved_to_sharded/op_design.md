# Operation Design: g6_interleaved_to_sharded

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no CB, no semaphore) |
| Goal | Move every tile page of a DRAM-interleaved TILE tensor into the L1 shard `memory_config` assigns it, in one program, bit for bit. Each shard-owning core reads its own pages from DRAM straight into its own shard (owner-fetch), and for tile rows that are only partly valid it reads only the valid face rows. |
| Math | `y[i] = x[i]` for every logical index; page `T(r, c)` moves from DRAM page id `r*C + c` to shard slot `(r - r0)*shard_cols_tiles + (c - c0)` on core `G[s]` |
| References | `eval/prompts/g6_interleaved_to_sharded.txt` (contract); `eval/golden_tests/g6_interleaved_to_sharded/{feature_spec,helpers,test_regression}.py`; `docs/explorations/g6_interleaved_to_sharded/{DOSSIER,math,tensix,risks,lessons}.md`; `.claude/references/{blocking-model,l1-footprint-discipline,op-design-template}.md`; `tech_reports/tensor_layouts/tensor_layouts.md:59-67` (face order); catalog `ttnn/ttnn/operations/examples/master.md` (entries listed under Catalog Evidence) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes, positional | rank 2 or 4, bfloat16, TILE, DRAM interleaved, on device | — | input address: common RT; shape-derived quantities: CT |
| `memory_config` | `ttnn.MemoryConfig` | yes, positional | L1, HEIGHT_SHARDED or WIDTH_SHARDED, shard spec covering the input's padded tile grid (see Validation) | — | output address: common RT; shard rectangle per core: per-core RT; shard shape: CT |

No `dtype`, `layout`, `compute_kernel_config` or output-tensor parameter (contract).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(..., H, W)`, rank 2 or 4. Leading dims fold into tile rows. Production: `(1, 5120)` (Phase 0), `(1, 1, 1, 128)` (refinement) |
| Dtype | bfloat16 |
| Layout | TILE (32x32 tile, four 16x16 faces, order f0→f1→f2→f3, `tensor_layouts.md:60`) |
| Memory | DRAM interleaved. L1 interleaved is refused (ValueError). |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` |
| Dtype | `input_tensor.dtype` |
| Layout | TILE |
| Memory | Exactly `memory_config`: L1, its memory layout, shard grid, shard shape and orientation. The entry point allocates it with `ttnn.allocate_tensor_on_device(shape, dtype, TILE, device, memory_config)`, so the allocator's page map is the placement contract. Tiles inside a shard are row-major. Padding rows of partial tile rows are left unwritten (allowed by the contract, see Key Risks). |

## Blocking Model

Every output page depends on exactly one input page. No axis is spanned by a result, so there is no dependent axis and no combine. The output placement pins which core owns each page. The blocking decisions are therefore: who moves each page (the split), how many bytes of each page cross DRAM (the transfer shape), and how many reads are in flight before the core's one barrier.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| leading dims `L` (images) | independent: no output element depends on another image | folded into `tensor_rows_tiles` (`R = L * tensor_rows_per_image_tiles`) | folded (production `L = 1`) | host: `prod(shape[:-2])` | through the tile-row axis (height sharding cuts folded tile rows) | knob-turn (rank 4 = folding only) |
| folded tile rows `r` | independent: every output page is one input page | `block_rows` = this core's shard rows clipped to the tensor | width sharded: `min(shard_rows_tiles, R)` = `R` = 1 (p1). Height sharded: `min(shard_rows_tiles, R - s*shard_rows_tiles)` = 1 (p2). The whole shard: the caller pinned the extent | per-core RT, derived on host from `memory_config.shard_spec.shape[0] // tile_h` and `R` | height sharded: shard `s` → `G[s]` (caller-pinned). Width sharded: not split, because every core owns all rows of its columns | knob-turn (height sharding = another rectangle per core, same kernel) |
| tile columns `c` | independent: every output page is one input page | `block_cols` = this core's shard columns clipped to the tensor | width sharded: `min(shard_cols_tiles, C - s*shard_cols_tiles)` = 5 (p1). Height sharded: `C` = 4 (p2). The whole shard | per-core RT, derived on host from `memory_config.shard_spec.shape[1] // tile_w` and `C` | width sharded: shard `s` → `G[s]` (caller-pinned). Height sharded: not split | knob-turn |
| datum rows inside a tile | independent: rows of a tile are separate logical values, and rows ≥ valid count are padding outside the result | `transfer_rows` per tile-row class: full tile rows → all 32 rows (one 2048 B segment). The last tile row of each image when `H % 32 != 0` → only the valid rows, rounded up to the DRAM read granule inside each face pair | full rows: 32. Partial rows at `H = 1`: 2 face rows × faces f0, f1 = two 64 B segments per tile. The value is the binding traffic-ranking choice, not a collapsed knob | CT segment table, derived on host from `H`, `tile.tile_shape`, `tile.face_shape`, datum bytes and `ttnn.get_dram_alignment()` (64 on Blackhole). Knob `partial_tile_transfer ∈ {valid_rows, whole_page}` (host constant, in the cache key) | not split across cores: a page is one NoC destination | knob-turn (perf lamp: whole page vs one contiguous read) |

Derived names, scoped per `blocking-model.md` §3, each computed once on host:

| Name | Formula | p1 `(1,5120)` width | p2 `(1,1,1,128)` height |
|------|---------|----|----|
| `tensor_rows_per_image_tiles` | `ceil(H / tile_h)` | 1 | 1 |
| `tensor_rows_tiles` (`R`) | `L * tensor_rows_per_image_tiles` (per image, never `floor(L*H/32)`) | 1 | 1 |
| `tensor_cols_tiles` (`C`) | `ceil(W / tile_w)` | 160 | 4 |
| `last_row_valid_rows` | `H - tile_h * (tensor_rows_per_image_tiles - 1)` | 1 | 1 |
| `shard_rows_tiles`, `shard_cols_tiles` | `shard_spec.shape // (tile_h, tile_w)` | 1, 5 | 1, 4 |
| `num_shards` | height: `ceil(R / shard_rows_tiles)`; width: `ceil(C / shard_cols_tiles)` | 32 | 1 |
| `(r0, c0)` of shard `s` | height: `(s*shard_rows_tiles, 0)`; width: `(0, s*shard_cols_tiles)` | `(0, 5s)` | `(0, 0)` |
| `num_blocks_this_core` | 1 (the whole clipped shard) | 1 | 1 |
| tile-row class of `r` | `partial` iff `last_row_valid_rows < tile_h` and `r % tensor_rows_per_image_tiles == tensor_rows_per_image_tiles - 1`, else `full` | all partial | all partial |

### Buffer-depth knobs

There are no CBs: the destination of every read is the output shard itself, so there is no staging slot whose depth could be set. The knob that plays the role of depth is the number of NoC reads in flight before the barrier.

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| (none: reads land in the output shard) | `reads_in_flight`: segment reads issued before the core's single `noc_async_read_barrier` | all segments of the block (p1: 10, p2: 8) | overlap of every DRAM round trip on the core. Catalog `double_buffer`: gains flatten at ≈4 outstanding reads, and the block already exceeds that. There is no L1 cost, because the slots are the shard |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| NoC DRAM read alignment, Blackhole: `(dst_l1 & 63) == (src_dram & 63)` (`tt_metal/hw/inc/internal/debug/sanitize.h:746`, `NOC_DRAM_READ_ALIGNMENT_BYTES 64` at `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`) | segment offset and size inside a page | Segment offsets sit only at face boundaries (0, 512, 1024, 1536), which are multiples of 64. Segment sizes are `align_up(rows*face_row_bytes, 64)`, capped at `face_bytes`. Holds because L1 buffers (the shard) are allocated at DRAM alignment (`tt_metal/impl/allocator/l1_banking_allocator.cpp:155-156`, `tt_metal/impl/allocator/bank_manager.cpp:176`) and DRAM page slots are 64-aligned with 2048 B pages | watcher alignment hang under `--dev`; silently shifted bytes in a plain build |
| Face geometry of the valid-rows transfer | `transfer_rows` | Valid only for `tile.tile_shape == (32,32)`, `tile.face_shape == (16,16)`, `transpose_of_faces == transpose_within_face == False`, and a non-block dtype (bfloat16). Any other tile or dtype → `whole_page` | wrong bytes: rows taken from the wrong face |
| `NOC_MAX_BURST_SIZE` = 256 × 64 B = 16384 B (`noc_parameters.h:289-290`) | bytes per segment | Every segment is ≤ `page_bytes` = 2048 < 16384, so each is one packet. `noc_async_read` splits larger sizes itself | none (multi-packet path) |
| Shard → core map of the allocator: 1D sharding lists cores with `corerange_to_cores(grid, grid.num_cores(), row_wise = orientation == ROW_MAJOR)` and gives shard `s` to entry `s` (`tt_metal/impl/buffers/buffer_distribution_spec.cpp:168-173`) | core-assignment | The host builds the core list with the same call (`ttnn.corerange_to_cores(grid, grid.num_cores(), row_wise)`) and takes its first `num_shards` entries. The orientation is read from `memory_config`, never assumed | pages on the wrong core. Values read back stay right, and only the placement check and `test_orientation_col_major` show the defect |
| Page stride inside a shard: positions follow the full shard shape even for a clipped last shard (`buffer_distribution_spec.cpp:49-50`, shard strides) | local slot of a page | `slot = (r - r0)*shard_cols_tiles + (c - c0)`, where the stride is `shard_cols_tiles`, never the clipped `block_cols` | pages of a clipped last width shard land at wrong slots |
| One shard per core | `num_shards` | Validation refuses `num_shards > len(G)` with ValueError | allocator multi-shard-per-core layout the kernel does not address |
| BB-05 free L1 per core = 1347776 B, the output shard included | shard bytes | Host check `shard_rows_tiles * shard_cols_tiles * page_bytes + 0 ≤ 1347776` (the op adds 0 B), else ValueError | refused launch beside the model's tensors |

### Regimes

Named boundary: **each input page's valid bytes cross DRAM once. Each output page is written once, into its owner's L1 shard, by the response of that read. No page crosses core-to-core.** The data minimum for `H = 1` is 64 logical bytes per tile. The DRAM read granule (64 B per segment, two face segments) makes the reachable minimum 128 B per tile. This is structural: row 0 lies in two faces 512 B apart (`tensor_layouts.md:60`).

Per-tile-row selection function (one program, both rows can occur in one call):
`transfer(r) = valid_rows if (partial_tile_transfer == valid_rows and last_row_valid_rows < tile_h and r % tensor_rows_per_image_tiles == tensor_rows_per_image_tiles - 1 and face-geometry cap holds) else whole_page`.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| A. owner-fetch, whole-page transfer | **built** | tile row is `full` under `transfer(r)`. In Phase 0 this occurs only for `H > 32` non-aligned shapes (design-forced test `(33, 5120)`), never in production | `block_rows × block_cols` = the core's clipped shard, one segment `(0, page_bytes)` per page | minimum: a full tile row is all valid, so 2048 B per page crosses DRAM once, with 0 core-to-core hops | the block is already the whole shard (caller-pinned). The fixed costs it pays once are one barrier and one kernel launch per core. Per-page cost (one NoC read issue) cannot be amortised further, because interleaved pages lie in different banks (catalog `read_coalescing`) |
| B. owner-fetch, valid-rows transfer | **built** (every production page) | tile row is `partial` under `transfer(r)` | as A. For each page, 2 segments: `v ≤ 16`: `(0, s)`, `(512, s)`, `s = align_up(v*32, 64)`; `v > 16`: `(0, 1024 + s)`, `(1536, s)`, `s = align_up((v-16)*32, 64)`; merged when contiguous. At `H = 1`: `(0,64)`, `(512,64)` | 128 B per page instead of 2048 B (p1: 20480 B from DRAM instead of 327680 B, against 10240 B logical). The 64 B granule is the only excess | as A. It also pays the NoC read issue twice per page instead of once (see perf lamp L1) |
| C. reader-and-forward: non-owner cores read DRAM pages into their own L1 and `noc_async_write` them into the owner's shard | **deferred**: owner-fetch already meets the named minimum (1 DRAM crossing, 0 core-to-core hops), and in every production case no owner holds more than 5 pages (p1) or 4 pages (p2). The only gain of C is issue parallelism for owners with many pages, which no production shape has | `block_pages_this_owner > forward_threshold` (would be a host predicate) | helper block = a slice of an owner's shard rectangle, staged in a helper CB of `k` pages | +1 core-to-core crossing per forwarded page and +`k` staging pages per helper. DRAM crossings unchanged | fewer per-page NoC issues per RISC. Reachable: the host already computes each page's owner and slot, so C adds helper roles (core ranges outside the active set) without changing A/B |
| D. two-step: interleaved L1 copy then shard (or calling an existing conversion op) | **rejected**: forbidden by spec rule 1 and BB-01, pays traffic twice. Superseded by A/B | — | — | 2 DRAM/L1 crossings per page and 2 programs | — |
| E. one core reads every page and writes the 31 other shards remotely | **rejected** (dead end): superseded by A/B everywhere. Same DRAM bytes, plus 31/32 of the pages crossing core-to-core, all serial on one RISC | — | whole tensor on one core | +1 core-to-core hop for every non-local page | — |
| F. copy through the compute engine (unpack → DEST → pack) | **rejected**: can rewrite NaN payloads or flush denormals (rule 4), and adds two CBs and three threads. Superseded by A/B | — | — | +2 L1 passes per page | — |
| G. host round trip (read to host, write sharded) | **rejected**: BB-01 (host read of values, extra programs) | — | — | PCIe crossing twice | — |

No split of a dependent axis exists: no axis is spanned, so there is no "split + combine" row.

### Traffic ranking

| Rank | Candidate split | DRAM crossings (input) | Core-to-core traffic | Notes |
|------|-----------------|------------------------|-------------------|-------|
| 1 (chosen) | split by output shard, owner reads its own pages, valid rows only (A+B) | 1 per page, valid bytes only: p1 20480 B, p2 512 B | 0 | the page's destination is the reading core's L1, so the read response is the store |
| 2 | same split, whole pages (A everywhere) | 1 per page: p1 327680 B, p2 8192 B | 0 | 16× the DRAM bytes at `H = 1`. Kept as knob `partial_tile_transfer = whole_page` |
| 3 | reader-and-forward (C) | 1 per page | 1 per forwarded page | strictly more traffic than rank 1. Deferred row C |
| 4 | one mover core (E) | 1 per page | (N-1)/N of pages | dead end, rejected |
| 5 | two-step (D) | 2 | — | forbidden |

Operand-reuse check: the only operand, `input_tensor`, varies along both split axes (each page is read by exactly one owner), so nothing is reuse-shared and there is no broadcast row.
Stall-shadow check: the only wait is the core's final read barrier. No work on the core is independent of it (the kernel has no other work), so nothing can be scheduled into it.
Occupancy (kept separate from the ranking): p1 uses all 32 cores of the 8×4 shard grid. p2 uses 1 core, which is pinned by its 1-core shard grid, and its 4 pages do not justify regime C (row C).

### Block schedule

The schedule runs per active core. A block is this core's clipped shard rectangle, and each core has exactly one.

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {  // num_blocks_this_core == 1
    fetch_block_into_shard(block_idx);   // every segment read of every page of the block, issued back to back
    retire_block(block_idx);             // one noc_async_read_barrier: all responses landed in the shard
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `fetch_block_into_shard` | `block_rows × block_cols` pages. Per page, the segment list of its tile-row class (A: 1 segment, B: 2). Inside the operation the necessary traversal is one NoC read issue per segment (pages lie in different DRAM banks, so they cannot coalesce) | the output shard (destination), nothing else | input TensorAccessor built once per kernel. Tile-row class decided once per tile row, and compiled out entirely when `tensor_rows_per_image_tiles == 1` (every row partial). No CB handshake, no init, no multicast, no semaphore. Profiler zone once per block |
| `retire_block` | the block | output shard, complete | exactly one `noc_async_read_barrier` per core per call, before the kernel returns (BB-03). Profiler zone around the wait (BB-21) |

There is no `store_block`: the read response is the store. There is no compute kernel and no writer kernel.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 segment shape of the partial tile row | Two 64 B reads per tile double the NoC issues (10 instead of 5 per p1 core). Per-transaction cost can outweigh the bytes saved at 5 pages per core | one contiguous read per partial tile `(0, face_bytes + s)` = 576 B at `H = 1` (5 issues per core, 92160 B total), and `partial_tile_transfer = whole_page` (5 issues of 2048 B, 327680 B total) |
| L2 single data-movement RISC | All segment issues of a core run on NCRISC/NoC0. Catalog `split_reader`: splitting helps only when issue-limited, which 8–10 reads are probably not | split each block's pages across two reader kernels, NCRISC/NoC0 and BRISC/NoC1 (two zones, two barriers) |
| L3 issue API | `noc_async_read` takes the generic path. Every segment is one packet (≤ 16384 B) | `noc_async_read_one_packet` (`dataflow_api.h:515`) for each segment |
| L4 bank order of first issues | All 32 cores start at the same instant, and core `s`'s first page sits in bank `5s mod 8`, which already rotates (catalog `bank_stagger`: 1.04–1.07× at one block per core) | rotate the in-block issue order by `s mod block_cols` |
| Grid synchronization | does not apply: there is no collective, barrier across cores, or multicast. The core count is pinned by the shard spec, and launch cost scales with core ranges (1 rectangle for p1) | — |
| Overlap | does not apply: there is no compute phase to overlap with. All reads are already in flight together | — |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM interleaved input page `p = r*C + c` | bfloat16 TILE page, 2048 B (faces f0..f3, 512 B each) | `TensorAccessor` over `TensorAccessorArgs(input_tensor)`, `get_noc_addr(p, seg_offset)` (`tensor_accessor.h:165`) | page `p` is in bank `p mod n_banks`. Consecutive pages are in different banks, so there is no multi-page read |
| DRAM → owner core L1 | the segment bytes, unchanged | `noc_async_read(src_noc_addr, shard_base + slot*page_bytes + seg_offset, seg_bytes)` on NCRISC / NoC0 (`dataflow_api.h:547`) | `shard_base` = `output_tensor.buffer_address()`, which is the same L1 address on every core of an L1-sharded buffer |
| Completion | — | `noc_async_read_barrier()` (`dataflow_api.h:1921`) once | BB-03: nothing in flight at return |
| Output | bfloat16 TILE pages in each core's shard, row-major within the shard | — (no CB: the shard is written in place by the read responses) | Padding rows of partial tiles are not written (contract: unspecified) |

TARGET placement axis, seen against the axis characters:

| `out_scheme` value | Axis it cuts | Character | Cost class | Phase |
|--------------------|--------------|-----------|------------|-------|
| `width_sharded` | tile columns | independent | knob-turn: rectangle `[0,R) × [s*sw, …)` per core | Phase 0 |
| `height_sharded` | folded tile rows | independent | knob-turn: rectangle `[s*sh, …) × [0, C)` per core. Same kernel, same segment table (tile-row class by `r % tensor_rows_per_image_tiles`) | refinement, together with rank 4 (folding only) |

Neither scheme needs core-to-core communication, because a physical shard here is the destination, not a source. The host and kernel are written rectangle-generic from Phase 0, so the refinement changes `SUPPORTED`, not the data path.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block = one core's clipped shard rectangle, `block_rows × block_cols` pages |
| Grid | the active cores: the first `num_shards` cores of `ttnn.corerange_to_cores(shard_grid, shard_grid.num_cores(), row_wise = orientation == ROW_MAJOR)`. They are expressed as core ranges: full rows (ROW_MAJOR) or full columns (COL_MAJOR) of each grid range plus at most one partial row/column, i.e. ≤ 2 rectangles per grid range (BB-17). p1: 1 rectangle (0,0)-(7,3). p2: 1 core. Cores of `G` beyond the last shard get no kernel. |
| Per-core work | shard `s` on core `G[s]`: rows `[r0, r0 + block_rows)`, columns `[c0, c0 + block_cols)`, with `block_rows = min(shard_rows_tiles, R - r0)` and `block_cols = min(shard_cols_tiles, C - c0)` |
| Remainder | only the last shard can be clipped (by the `min` above). `R` is per-image tile-padded (`L * ceil(H/32)`) and `C = ceil(W/32)`. There is no `floor` and no `//` on element counts. |
| `split_work_to_cores` | not used: the placement fixes both the core-assignment and each core's extent, and owner-fetch needs no further balancing |
| Kernel roles | one reader kernel (NCRISC, `ttnn.ReaderConfigDescriptor()`, NoC0) on the active cores. No writer and no compute kernel. |
| Knob sources (single place each) | shard rectangle per core: host, one function `_plan(input_tensor, memory_config)` that also feeds the plan report. Segment table: host constant derived in `_plan`. `partial_tile_transfer`: module constant `PARTIAL_TILE_TRANSFER = "valid_rows"`. Grid: `memory_config.shard_spec.grid` (runtime input, never a literal) |

Regime-pinned tests (both regimes occur per tile row, not per device grid):

| Test | Covers |
|------|--------|
| `flux58` production `(1, 5120)` width `(32,160)` | B only (every row partial) |
| `h33_width` `(33, 5120)` width `(64,160)` on (0,0)-(7,3) | A (tile row 0) and B (tile row 1) on every core |
| `height_multi_shard` `(2,1,33,128)` height `(32,128)` on (0,0)-(3,0) (refinement, xfail in Phase 0) | A on cores 0 and 2, B on cores 1 and 3, multi-image folding |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| — (none) | — | — | 0 | Inventory before solve: each phase boundary is "DRAM page → its final L1 slot", and the read response writes the destination directly (pack-into-destination, `l1-footprint-discipline.md` Rule 3.1). A CB aliased on the output shard would only supply an address that `output_tensor.buffer_address()` already gives, at the cost of a CB config entry and a producer-only handshake with no consumer kernel | — | — | — | — |

Full accounting: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `fetch_block_into_shard` | `block_rows × block_cols` pages × segments of the row class | raw dataflow API (no helper covers DRAM→own-L1 reads, see API Mapping) | none (source = input DRAM via TensorAccessor) | none (destination = output shard at `shard_base`) | reads in flight |
| 2 | `retire_block` | the block | raw dataflow API | — | — | shard complete. Nothing in flight |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `fetch_block_into_shard` | raw_api | `TensorAccessor` + `TensorAccessorArgs<…>` (input, interleaved DRAM) | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:71` | CT args from `ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()` (placed last in the CT list), address = common RT arg, page size = `page_bytes` | — | — | none |
| `fetch_block_into_shard` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tensor_accessor.h:165` | `page_id = r*C + c`, `offset = seg_offset` | — | — | `seg_offset` (segment table) |
| `fetch_block_into_shard` | raw_api | `noc_async_read(src_noc_addr, dst_local_l1_addr, size)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | `dst = shard_base + ((r-r0)*shard_cols_tiles + (c-c0))*page_bytes + seg_offset`, `size = seg_bytes` | — | — | `block_rows`, `block_cols`, `seg_bytes` |
| `retire_block` | raw_api | `noc_async_read_barrier()` | `dataflow_api.h:1921` | default NoC | — | — | — |
| both | raw_api | `DeviceZoneScopedN("I2S-FETCH")`, `DeviceZoneScopedN("I2S-BARRIER")` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012` (compiles to nothing without the profiler, `:1047`) | — | — | — | — |

**Helpers considered and rejected** (they apply to every `raw_api` row above):

| Candidate | File:Line | Why it cannot express this block |
|-----------|-----------|----------------------------------|
| `dataflow_kernel_lib::set_read_state` / `read_with_state` / `local_addr` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57,74-75` | They are self-aimed L1→L1 copies: `local_addr()` builds the source from this core's own `my_x/my_y` (`:57`), and `set_read_state` programs one fixed source for the loop (`:74-75`). Here every page's source is a different DRAM bank |
| `dataflow_kernel_lib` mcast (`McastArgs`, `ttnn.Mcast`) | `ttnn/cpp/ttnn/kernel_lib/mcast/README.md` | no reuse-shared operand (operand-reuse check above), so there is nothing to multicast |
| `tilize`/`untilize`/`eltwise`/`reduce`/`matmul_block` compute helpers | `ttnn/cpp/ttnn/kernel_lib/*_helpers.hpp` | the op has no compute phase, and a compute-engine path is rejected (regime F, rule 4) |
| `dfb_helpers_dataflow` | `ttnn/cpp/ttnn/kernel_lib/dfb_helpers_dataflow.hpp` | DataflowBuffer/CB handshakes. The design has no CB (Circular Buffers) |
| `TensorAccessor` of the **output** (sharded) | `tensor_accessor.h:165` | its `get_noc_addr` resolves a page to a (core, address) NoC address. A read's destination must be a local L1 address, which the host-computed slot formula gives directly without the distribution-spec CT args (BB-18 text size) |

What would close the gap: a `kernel_lib` dataflow helper "read a page rectangle from an interleaved accessor into a local L1 region with a per-page segment list". None exists, so the implementer writes `fetch_block_into_shard` as one block function, with the per-segment issue loop inside it.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Core order differs from the allocator's | The values read back by `to_torch` follow the allocator. A core order that differs from it puts pages on cores the allocator does not map to those shards → wrong values or a failed placement | the host uses the allocator's own ordering call (Mechanism caps, `buffer_distribution_spec.cpp:168-173`) with `row_wise` from `memory_config.shard_spec.orientation`. Test `col_major` |
| Partial tile rows read at wrong face offsets | Row `v` of a tile lives in two faces 512 B apart, not contiguously | segment table derived on host from `tile.face_shape` (single source). Tests `special_values` and `poisoned_padding` (NaN in rows 1-31 must not reach row 0), and `h33_width` for `v`-class mixing |
| DRAM read alignment | Blackhole requires `dst ≡ src (mod 64)` for DRAM reads | segment offsets only at face boundaries, sizes aligned to 64 (Mechanism caps) |
| Output padding rows are stale L1 | The valid-rows transfer leaves rows ≥ `last_row_valid_rows` of partial tiles unwritten. Their content varies between runs, so BB-06 holds on the result (logical values) but not on padding bytes. A consumer that reads padding as values (e.g. a K-padded matmul operand: `0 × inf = NaN`) would see it | the contract states padding is unspecified (spec Shape handling; `docs/explorations/g6_interleaved_to_sharded/sources/local/contract_r3.md:64`). The fallback is one constant: `PARTIAL_TILE_TRANSFER = "whole_page"` copies the input's padding verbatim (traffic rank 2) |
| Clipped last shard | `C % shard_cols_tiles != 0` (or the R analogue) | `min` clipping on host, and stride `shard_cols_tiles` (Mechanism caps). Not in production, so the formula is anchored to `buffer_distribution_spec.cpp:49-50` |
| Rank-4 folding (refinement) | Partial tile rows recur once per image, not once per tensor | tile-row class by `r % tensor_rows_per_image_tiles`, R per image (`test_height_multi_shard`) |
| BB-03: transfer in flight at return | the next model program starts at once | the barrier is the last statement of the kernel. `test_repeat_then_other_program` |
| BB-07 / BB-16: cached descriptor replays a stale address or the wrong placement | input and output addresses change per call, and the shard spec separates programs of one shape | see Host Contract: cache key and per-call patch of exactly the two address common RT args. `test_cache_key_shard_spec`, `test_new_input_address` |
| BB-17: scattered kernel core ranges | a clipped last row/column of `G` | ≤ 2 rectangles per grid range (Work Distribution). p1 = 1 range |
| BB-18 / BB-04: kernel text | one tiny reader kernel. The whole-page branch exists only when `tensor_rows_per_image_tiles > 1` | `if constexpr` on CT args. No per-page RT args (RT count is 4 per core + 2 common, independent of shape) |

## Host Contract

### Registry declarations (op file)

| Item | Value |
|------|-------|
| `INPUT_TAGGERS` | `dtype` ← `scenario["dtype"]`. `layout` ← `scenario["layout"]`. `rank` ← `len(scenario["shape"])`. `alignment` ← `H = shape[-2]`, `W = shape[-1]`: `"tile_aligned"` if both are multiples of 32, `"w_non_aligned"` if `W % 32`, else `"h_non_aligned"`. `out_scheme` ← `scenario["out_memory"]["scheme"]`. Each tagger reads `inputs[0]` (the scenario dict) |
| `SUPPORTED` (Phase 0, exactly the spec list) | `dtype: [ttnn.bfloat16]`, `layout: [ttnn.TILE_LAYOUT]`, `rank: [2]`, `alignment: ["h_non_aligned"]`, `out_scheme: ["width_sharded"]` |
| `EXCLUSIONS` | `[]` |
| `validate(input_tensor, memory_config)` | First, the malformed-request checks below (ValueError/RuntimeError). Then it builds the scenario dict from the call (`shape`, `dtype`, `layout`, `in_memory`, `out_memory` with `scheme` = `"height_sharded"`/`"width_sharded"`) and applies `INPUT_TAGGERS` (one tagging path for tests and runtime). Then SUPPORTED per axis → `UnsupportedAxisValue`, then EXCLUSIONS → `ExcludedCell` (`ttnn/ttnn/operations/_op_contract.py:27,31`) |
| Entry point | `g6_interleaved_to_sharded(input_tensor, memory_config)`: `validate(...)` is its first line |
| Index axes | none |
| Structural impossibilities | none beyond `feature_spec.INVALID = []` |

### Validation (ValueError, before the support check)

| Check | Refused when |
|-------|--------------|
| input placement | not on device, `layout != TILE`, `memory_layout != INTERLEAVED`, or `buffer_type != DRAM` |
| dtype / rank | `dtype != bfloat16`, or `rank ∉ {2, 4}` |
| target | `buffer_type != L1`, `memory_layout ∉ {HEIGHT_SHARDED, WIDTH_SHARDED}`, or no `shard_spec` |
| shard covers the padded tile grid | a shard dim not a multiple of 32. HEIGHT: `shard_w < C*32` (narrower than the tensor). WIDTH: `shard_h < R*32` (shorter than the tensor). Either: `num_shards > shard_grid.num_cores()` |
| BB-05 | `shard_rows_tiles * shard_cols_tiles * page_bytes > 1347776` |

### Program caching, plan report, zones (BB-07, BB-08, BB-16, BB-21)

| Item | Decision |
|------|----------|
| Descriptor cache | module-level dict keyed by plain ints/strings: `(op name, device id, logical shape, padded shape, dtype, tile shape, input buffer type + memory layout, output memory layout, buffer type, shard grid ranges, shard shape, orientation, PARTIAL_TILE_TRANSFER)`. A hit reuses the built `ProgramDescriptor` and patches only the two common RT args, input address and output address (`output_tensor.buffer_address()`). Everything else in the program is a function of the key. `generic_op` hashes the descriptor without RT values (`ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:78-146`). A `custom_program_hash` is allowed only if it is derived from this same key |
| Plan report (BB-08, BB-21) | `describe_plan(input_tensor, memory_config) -> dict`, the same `_plan` the builder uses: regime (`owner_fetch`), per-tile-row transfer class and segment table, active core ranges, and per core `{core, shard, r0, c0, block_rows, block_cols, role: "owner-reader"}` |
| Zones | `I2S-FETCH` around `fetch_block_into_shard`, `I2S-BARRIER` around the barrier. Both vanish in the plain build |
| One dispatch | exactly one `ttnn.generic_op([input_tensor, output_tensor], descriptor)` per call. No other ttnn op is called on the data path |

## Catalog Evidence

| Entry (`ttnn/ttnn/operations/examples/`) | Knob it informed |
|------------------------------------------|------------------|
| `double_buffer/README.md` | `reads_in_flight` = whole block, one barrier (gains flatten at ≈4 outstanding) |
| `read_coalescing/README.md` | no multi-page reads: interleaved pages are in different banks |
| `split_reader/README.md` | single reader RISC by default. Lamp L2. Smaller transactions for the same bytes are slower (motivates lamp L1, not a sub-page split of full pages) |
| `noc_placement/README.md` | DRAM reads on NCRISC/NoC0 |
| `bank_stagger/README.md` | lamp L4 |
| `width_split/README.md` | a fixed per-call cost dominates tiny moves. Spreading over the shard cores costs little |
