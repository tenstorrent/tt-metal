# Operation Design: g6_interleaved_to_sharded

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement — one reader kernel, no compute kernel, no writer kernel, no CB |
| Goal | Move every tile page of a DRAM-interleaved bfloat16 TILE tensor into the L1 height or width shard that `memory_config` names, bit for bit, in one program. Each shard's own core reads its pages from DRAM straight into its shard (owner-fetch), and for the last tile row of each image it reads only the face rows that hold logical values. |
| Math | `y[i] = x[i]`. Page `T(r, c)` (page id `p = r·tensor_cols_tiles + c`) moves from its interleaved DRAM slot to core `G[s]` at shard-local page `(r − block_row_start)·shard_cols_tiles + (c − block_col_start)`. |
| References | the spec (prompt); `docs/explorations/g6_interleaved_to_sharded/{DOSSIER,math,tensix,risks,lessons}.md`; `eval/golden_tests/g6_interleaved_to_sharded/{feature_spec,helpers,test_regression}.py`; `.claude/references/{blocking-model,l1-footprint-discipline,noc-alignment-in-tt-metal}.md`; catalog entries listed under Dataflow Strategy |

Symbols used throughout. Every one of them has a single source, the host function `derive_plan(input_tensor, memory_config)` in the program-descriptor module. The descriptor builder and `describe_plan` both read it.

| Symbol | Definition | p1 (1, 5120) width | p2 (1, 1, 1, 128) height |
|--------|------------|--------------------|--------------------------|
| `tiles_h_per_image` | `padded_shape[-2] / 32` = `ceil(H/32)` | 1 | 1 |
| `num_images` | product of the leading dims (`1` for rank 2) | 1 | 1 |
| `tensor_rows_tiles` | `num_images · tiles_h_per_image` (each image is padded separately) | 1 | 1 |
| `tensor_cols_tiles` | `padded_shape[-1] / 32` = `ceil(W/32)` | 160 | 4 |
| `last_row_valid_rows` (v) | `H − 32·(tiles_h_per_image − 1)` | 1 | 1 |
| `shard_rows_tiles`, `shard_cols_tiles` | `shard_spec.shape / 32` | 1, 5 | 1, 4 |
| `num_shards` | height: `ceil(tensor_rows_tiles / shard_rows_tiles)`; width: `ceil(tensor_cols_tiles / shard_cols_tiles)` | 32 | 1 |
| `shard_cores` (G) | `ttnn.corerange_to_cores(shard_spec.grid, row_wise=(orientation == ROW_MAJOR))` | (0,0),(1,0)…(7,0),(0,1)…(7,3) | (0,0) |
| `page_size` | `input_tensor.buffer_aligned_page_size()` | 2048 | 2048 |
| `face1_offset_bytes` | `face_h · face_w · element_size` | 512 | 512 |
| `trim_span_bytes` | `round_up(v · face_w · element_size, ttnn.get_dram_alignment())` when the trim predicate holds, else 0 | 64 (Blackhole) | 64 |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes, positional | rank 2 or 4, bfloat16, TILE, DRAM interleaved, on device | — | geometry → CT; buffer address → common RT (patched on every call) |
| `memory_config` | `ttnn.MemoryConfig` | yes, positional | L1, HEIGHT_SHARDED or WIDTH_SHARDED, a legacy `ShardSpec` (grid, shape in multiples of 32, orientation) that covers the padded tile grid | — | shard geometry → CT, plus the per-core block rectangle as per-core RT; output buffer address → common RT (patched on every call) |

No dtype, layout, or compute_kernel_config parameter: the op neither casts nor computes.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(..., H, W)`, rank 2 or 4. Leading dims fold into tile rows, and each image is tile-padded separately. Production: `H = 1`, `W ∈ {5120, 128}` |
| Dtype | bfloat16 |
| Layout | TILE (32×32 tiles, 16×16 faces) |
| Memory | DRAM interleaved. An L1-interleaved input is malformed and is refused |

### Output

| Property | Value |
|----------|-------|
| Shape | the input's logical shape |
| Dtype | the input's (bfloat16) |
| Layout | TILE |
| Memory | exactly `memory_config`: L1, the same memory layout, shard grid, shard shape and orientation. Allocated by the entry point with `ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, TILE_LAYOUT, device, memory_config)` before dispatch. Positions beyond the logical shape (tile padding) are unspecified and are not written |

## Validation and Support Contract

`validate(input_tensor, memory_config)` is the first line of the entry point. Malformed requests raise `ValueError`, in this order, **before** the support check. A malformed height-sharded rank-4 request therefore raises `ValueError`, not a support refusal.

| # | Malformed when | Source of the rule |
|---|----------------|--------------------|
| 1 | input not on a device (`ttnn.is_tensor_storage_on_device(input_tensor)` false) | spec Validation |
| 2 | `input_tensor.layout != TILE_LAYOUT` | spec Validation |
| 3 | input memory layout not INTERLEAVED, or input buffer type not DRAM | spec Validation; contract r3 (L1 interleaved refused) |
| 4 | `input_tensor.dtype != bfloat16` | spec Validation |
| 5 | rank not in {2, 4} | spec Validation |
| 6 | `memory_config.buffer_type != L1`, or memory layout not in {HEIGHT_SHARDED, WIDTH_SHARDED}, or `shard_spec is None` | spec Validation |
| 7 | either shard dimension is not a multiple of 32 | spec Validation |
| 8 | height sharded and shard width != `32·tensor_cols_tiles` (narrower is the spec's case; wider is rejected by `tt_metal/impl/tensor/spec/tensor_spec.cpp:52-56`) | spec Validation; tensor_spec.cpp |
| 9 | width sharded and shard height != `32·tensor_rows_tiles` (shorter is the spec's case; taller is rejected by `tensor_spec.cpp:62-67`) | spec Validation; tensor_spec.cpp |
| 10 | `num_shards > shard_spec.grid.num_cores()` | spec Validation; `tensor_spec.cpp:57-60,68-71` |
| 11 | `memory_config.experimental_is_per_core_allocation()`: shard bases may differ per core, and the design passes one output address to every core | mechanism cap "lockstep shard address" (Blocking Model) |

After these checks comes the support check. The tags are derived from the tensor and `memory_config` by the same rules the INPUT_TAGGERS apply to scenario dicts: `dtype`, `layout`, `rank = len(shape)`, `alignment` (both H and W multiples of 32 → `"tile_aligned"`; W not → `"w_non_aligned"`; else `"h_non_aligned"`), and `out_scheme` from the memory layout. A value outside SUPPORTED raises `UnsupportedAxisValue`, and an EXCLUSIONS hit raises `ExcludedCell`. Both come from `ttnn.operations._op_contract`.

| Item | Phase 0 |
|------|---------|
| SUPPORTED (implementer writes it) | dtype `[bfloat16]`, layout `[TILE_LAYOUT]`, rank `[2]`, alignment `["h_non_aligned"]`, out_scheme `["width_sharded"]` |
| EXCLUSIONS | `[]` |
| Index axes | none: the signature has no dim/axis argument |
| Structural impossibilities (INVALID candidates) | none: every cell is a production scenario |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Nothing is computed. The block is the set of pages one core receives and drives through one uninterrupted read pass: every read of the block in flight, then one barrier.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Buffer depth | Later unlock | Why |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|--------------|-----|
| `image` (leading dims, folded) | independent: every output page is exactly one input page, and no result spans images | folded into `block_rows`; the fold uses `num_images` and `tiles_h_per_image` | rank 2: 1 image | host `derive_plan` (from `input_tensor.padded_shape`) | through `tile_rows` | n/a (no CB) | knob-turn: rank 4 is the same fold | Images are contiguous in tile rows and are treated identically, so they fold honestly. Per-image padding stays visible through `tiles_h_per_image`, which picks out the last tile row of each image |
| `tile_rows` (r < `tensor_rows_tiles`) | independent: no result spans rows | `block_rows` = this core's rows inside the tensor. Height: `min(shard_rows_tiles, tensor_rows_tiles − s·shard_rows_tiles)`; width: `tensor_rows_tiles` | p1: 1 (the whole tensor height) | host `derive_plan` → per-core RT `block_row_start`, `block_rows` | height sharded: `G[r div shard_rows_tiles]`; width sharded: every owner holds all rows | whole block in flight | knob-turn: height sharding cuts this axis with the same kernel | The shard fixes the per-core extent. One block = the whole resident shard (the default is one block). There is no L1 pressure to sub-chunk, because the op adds 0 B of L1 |
| `tile_cols` (c < `tensor_cols_tiles`) | independent: no result spans columns | `block_cols` = this core's columns inside the tensor. Width: `min(shard_cols_tiles, tensor_cols_tiles − s·shard_cols_tiles)`; height: `tensor_cols_tiles` | p1: 5 (the whole shard) | host `derive_plan` → per-core RT `block_col_start`, `block_cols`; the shard-local row stride `shard_cols_tiles` is CT | width sharded: `G[c div shard_cols_tiles]` (the Phase 0 split: 32 cores for p1) | whole block in flight | knob-turn | As for `tile_rows`. The ragged last shard keeps the **nominal** stride `shard_cols_tiles` and narrows only `block_cols` |
| `tile_face_rows` (rows 0..31 inside a page) | independent: padding rows are not part of the result, so a page may be moved partially | `trim_span_bytes`: bytes moved per top face (face 0, face 1) of a page in the last tile row of an image. Interior tile rows always move the whole page | `round_up(v·32 B, dram_alignment)` = 64 B at `H mod 32 = 1`; 0 (whole page) when the trim predicate fails | host `derive_plan` → CT `trim_span_bytes`, CT `face1_offset_bytes` | not split (it lies inside one page) | n/a | knob-turn (perf lamp L1) | This is the binding traffic choice. For `H = 1` it moves 128 B per tile instead of 2048 B (16× fewer DRAM bytes). The structure is unchanged: the same page loop issues two short reads instead of one long one |

**Trim predicate** (host-exact, in `derive_plan`): `input_tensor.tile.tile_shape == [32, 32]` ∧ `tile.face_shape == [16, 16]` ∧ `not tile.transpose_within_face` ∧ `not tile.transpose_of_faces` ∧ `last_row_valid_rows ≤ face_h (16)`. When it holds, a last-row page moves as two spans `[0, trim_span_bytes)` and `[face1_offset_bytes, face1_offset_bytes + trim_span_bytes)`. When it fails, `trim_span_bytes = 0` and every page moves whole. Interior tile rows (`(r + 1) mod tiles_h_per_image != 0`) always move whole pages.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| none. No CB exists: the read response lands in the output shard itself | reads in flight per barrier = every span of the block | p1: 10 spans (5 pages × 2); p2: 8 | One DRAM round trip per core per call instead of one per page. This is above the catalog's plateau of about 4 outstanding transactions (`double_buffer/README.md:96-99`), and there is no L1 cost to holding more |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DRAM→L1 read alignment, Blackhole 64 B (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`): the source and destination must agree modulo 64 (`noc-alignment-in-tt-metal.md` §1) | span offsets and the shard base | Span offsets are `0` and `face1_offset_bytes` (512), both multiples of 64 inside 2048-aligned pages. Shard bases are aligned to the DRAM alignment by the allocator (`tt_metal/impl/allocator/bank_manager.cpp:176`, `init_allocators(..., dram_alignment_bytes, ...)`). The host asserts `output.buffer_address() % ttnn.get_dram_alignment() == 0` and rounds `trim_span_bytes` up to `ttnn.get_dram_alignment()` | silent corruption; the watcher stops under `--dev` |
| Face geometry: rows 0..15 of a 32×32 tile lie in faces 0 and 1, row-major inside each face and face-major across the tile (`ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:11-12`) | `tile_face_rows` (trim) | The trim predicate above (32×32 tile, 16×16 faces, no transpose) | wrong rows copied, so wrong logical values |
| Top-faces-only trim | `last_row_valid_rows` | Trim only when `v ≤ 16`; otherwise the last row moves whole pages | rows 16..v−1 are never copied, so wrong values |
| Shard geometry (`tensor_spec.cpp:52-71`; `buffer_distribution_spec.cpp:172`: shard `s` → `G[s]`, `G` from `corerange_to_cores(grid, row_wise = ROW_MAJOR)`) | `block_rows`, `block_cols` | The block is the shard rectangle clipped to the tensor's tile grid. The shard-local index uses the nominal `shard_cols_tiles` stride. Cores `G[s ≥ num_shards]` get an empty block | A partial last shard lands at the wrong stride, or a write goes past the shard |
| `noc_async_read` packet size (`dataflow_api.h:547`, which splits above `NOC_MAX_BURST_SIZE`) | bytes per span | none needed: spans are at most 2048 B | — |
| Lockstep shard address: with default and range-lockstep allocation, an L1 sharded buffer has one address on every core of its grid. Per-core allocation does not (`ttnn/cpp/ttnn-nanobind/tensor.cpp`, `experimental_is_per_core_allocation`) | the single `output_shard_base_addr` common RT arg | validate rule 11 refuses per-core allocation with `ValueError` | every core but one writes at another core's address, corrupting the model's L1 |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `owner_fetch` | **built** | every validated call (output L1 HEIGHT/WIDTH sharded, input DRAM interleaved). The per-row transfer form inside it is set by the trim predicate | One block per owner core `G[s]`, `s < num_shards`: `block_rows × block_cols` pages at (`block_row_start`, `block_col_start`). Last-row pages are trimmed when `trim_span_bytes > 0`; interior-row pages move whole. Cores `G[s ≥ num_shards]`: `num_blocks_this_core = 0` | **Named boundary:** each logical value of x crosses DRAM once, y never crosses DRAM, and no byte crosses Tensix to Tensix. **Actual:** x is read once; a trimmed page moves `2·trim_span_bytes` (128 B at H mod 32 = 1), so p1 moves 20480 B against 10240 logical B. The 2× excess is the 64 B DRAM read granularity applied to 32 B face rows, which is structurally unreachable below that. Interior rows move whole pages (all 32 rows are logical). y: 0 DRAM bytes, 0 cross-core bytes, because the read response lands in the owner's shard | Fixed cost per block: one kernel start and runtime-arg fetch, one TensorAccessor construction, and **one** `noc_async_read_barrier`. No CB handshakes, inits, or reconfigs exist. The whole shard as one block pays the barrier and the DRAM round trip once per core per call instead of once per page |
| `owner_fetch_whole_pages` | **rejected**: superseded by `owner_fetch`, which issues the same page reads, moves 16× fewer DRAM bytes at H mod 32 = 1, and already moves whole pages wherever trimming saves nothing (interior rows, `v > 16`, non-standard tiles) | never selected | as `owner_fetch` | x: 2048 B per page. p1 moves 327680 B (32× logical) | as `owner_fetch` |
| `reader_and_forward` (helper cores read sub-rectangles of a shard into a staging CB and NoC-write them into the owner's shard) | **deferred**: `owner_fetch` keeps every read of a core in flight together, so an owner's time is one DRAM round trip plus its issue cost. Forwarding adds a NoC write hop, a staging CB and a write barrier on every forwarded page. It pays only when one owner's issue time exceeds a NoC write round trip, i.e. a large shard on few cores. No production shape is like that: 5 and 4 pages per owner. For p2 (4 pages on one core) forwarding is strictly slower: a DRAM read plus a NoC write, where owner-fetch needs one DRAM read. **Reachable:** the per-core work is already an RT rectangle. A forwarder adds the owner's NoC coordinates (`device.worker_core_from_logical_core`) and a staging CB, and reuses the same page map | if built: `block_rows·block_cols > forward_threshold_pages` ∧ `num_shards < grid cores` (threshold measured) | a sub-rectangle of one shard per helper core | x is read once. Adds +1 NoC write per forwarded page and `forward_depth · page_size` of staging L1 on each helper | adds a write barrier per helper block |
| `single_core_mover` | **rejected**: superseded by `owner_fetch` (risks.md R7). One core issues all 160 p1 reads, and 31/32 of the pages cross the NoC a second time | never selected | the whole tensor | x read once; +155 NoC page writes on p1 | — |
| `two_step` (to L1 interleaved, then reshard) | **rejected**: forbidden by the contract (it pays the traffic twice; BB-01 one program) | never selected | — | x crosses DRAM, then the whole tensor crosses L1 twice | — |

Only `owner_fetch` is built, so there is no regime-selection function. The **row-transfer-form selector** is the trim predicate above. The pinned tests are: trimmed form by p1 (`H = 1`, every row last); whole-page form by the ragged width case `(33, 5120)` of the acceptance test (row 0 is interior); both forms on one core by `height_multi_shard` `(2, 1, 33, 128)` once height sharding is in SUPPORTED.

### Traffic ranking

Operand-reuse check: x varies along the split axis, because every owner reads a disjoint page set. No operand is reuse-shared, so there is no broadcast/multicast row. Dependent-axis split + combine: not applicable, since no axis is dependent (`math.md` "Axes and dependence").

| Rank | Candidate split | DRAM crossings (x) | DRAM bytes p1 / p2 | Cross-core bytes p1 / p2 | Fan-out | Decision |
|------|-----------------|--------------------|--------------------|--------------------------|---------|----------|
| 1 | owner-fetch along the axis the shards cut (tile_cols for width, tile_rows for height), trimmed last rows | 1 | 20480 / 512 | 0 / 0 | each page → 1 core (its owner) | **built** |
| 2 | owner-fetch, single span `[0, face1_offset_bytes + trim_span_bytes)` per trimmed page | 1 | 92160 / 2304 | 0 / 0 | 1 | perf lamp L1: half the read count, 4.5× the bytes |
| 3 | owner-fetch, whole pages | 1 | 327680 / 8192 | 0 / 0 | 1 | rejected |
| 4 | reader-and-forward (k helper cores per owner) | 1 | as rank 1 | forwarded pages × 2048 | 1 extra hop per page | deferred |
| 5 | single-core mover | 1 | as rank 1 | 155 pages (p1) | 1 core → 31 cores | rejected |
| 6 | two-step via L1 interleaved | 1 + an L1 round trip of the whole tensor | — | whole tensor twice | — | rejected (contract) |

The chosen split is also the cheapest-traffic split. Occupancy follows the placement: p1 fills 32 cores (all of its owners). p2 has one owner, and its only multi-core alternative (reader-and-forward, ranking row 4) is slower (see the deferred regime row).

### Block schedule

```cpp
// num_blocks_this_core = 1 on an owner core G[s], s < num_shards; 0 on G[s >= num_shards].
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    fetch_shard_block(block_idx);   // every span read of the block, DRAM -> this core's own shard
    drain_block(block_idx);         // one read barrier for the whole block
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `fetch_shard_block` | `block_rows × block_cols` pages, rows outer and columns inner, in shard-local order. Per page: 2 spans of `trim_span_bytes` at offsets 0 and `face1_offset_bytes` when the page is in the last tile row of an image and `trim_span_bytes > 0`; otherwise 1 span of `page_size` | the output shard (destination), which is the result | TensorAccessor built once per kernel. Reads are issued back to back with **no barrier, wait or handshake between pages**. No init, no CB, no multicast |
| `drain_block` | the same block | the shard, now complete | exactly one `noc_async_read_barrier()` per block, i.e. per core per call. It is the kernel's last NoC action (BB-03) |

`store_block` does not exist: the read lands in the output. Splitting the barrier per page or per row is the anti-pattern this schedule forbids.

**Stall shadow:** the only waiting stage is `drain_block`, and the op has no remaining work that does not depend on the reads being waited for. Nothing to reorder.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 trim span form | The default moves the fewest bytes (2 × 64 B per tile) but issues twice the reads. Per-transaction cost dominates small reads (`read_coalescing/README.md`, `double_buffer/README.md:96-99`) | one span `[0, face1_offset_bytes + trim_span_bytes)` per trimmed page (576 B, one read); also the whole page. Measure on p1 |
| L2 reader RISC count | One reader RISC (NCRISC, NOC0) issues 10 reads per owner. Splitting only helps when a DM RISC is issue-bound (`split_reader/README.md:10-15`) | split each block's pages across NCRISC/NOC0 and BRISC/NOC1: a second kernel on the same core range, with each kernel taking half the block's columns |
| L3 NoC of the single reader | NOC0 vs NOC1 read cost depends on the core line versus the DRAM column positions, and is unmeasured on Blackhole (`noc_placement/README.md:30-32`) | reader on NOC1 (`ttnn.DataMovementConfigDescriptor(processor=RISCV_1, noc=NOC_1)`) |
| L4 span issue cost | Each span is a full `noc_async_read` (complete register setup). Both spans of a trimmed page share the bank and the size | `noc_async_read_one_packet_set_state` once per page, then `noc_async_read_one_packet_with_state` per span (`dataflow_api.h:575,599`) |

Overlap lamp: not attached. There is no compute to overlap, and all of a block's reads are already in flight together. Grid-synchronization lamp: not attached. There is no collective, and the only fewer-core alternative is the deferred `reader_and_forward` regime.

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| 1. page lookup | interleaved DRAM page `p = r·tensor_cols_tiles + c` | `TensorAccessor` (interleaved, DRAM), from `TensorAccessorArgs(input_tensor)` CT args and the input base address (common RT) | `get_noc_addr(p, span_offset)` → bank `p mod num_banks`. The input side cannot coalesce: consecutive pages sit in different banks (`read_coalescing/README.md:85-87`) |
| 2. move | bfloat16 tile bytes, either 2 trimmed spans or the whole 2048 B page | `noc_async_read` on the reader RISC (NCRISC, NOC0) into `output_shard_base_addr + local_idx·page_size + span_offset` on **this** core | No CB, no staging, no Tensix-to-Tensix traffic, no compute engine. The bytes never touch unpack/pack, so NaN payloads and denormals survive (rule 4) |
| 3. complete | — | one `noc_async_read_barrier` | The kernel returns only after it (BB-03). Idle cores issue nothing |
| deferred `reader_and_forward` contract | bfloat16 page | helper: `noc_async_read` into `cb_forward_staging`, `noc_async_write` to the owner's NoC coordinates (`device.worker_core_from_logical_core(G[s])`) at the same shard-local address, `noc_async_write_barrier` before return | No semaphore needed: program completion orders the writes before the next program. The page map is the same as owner-fetch |

**TARGET axes as placements.** Both shard schemes cut an independent axis, so both are knob-turns. The kernel is scheme-agnostic: a core receives a rectangle in tile space plus the nominal shard stride.

| TARGET value | Phase 0 | Axis the cut is along | Cost class | What the refinement changes |
|--------------|---------|----------------------|------------|------------------------------|
| out_scheme `width_sharded` | built | `tile_cols` (independent) | — | — |
| out_scheme `height_sharded` | refinement | `tile_rows` (independent) | knob-turn | add to SUPPORTED; `derive_plan` already produces height rectangles; kernel unchanged |
| rank `4` | refinement (with height_sharded, its only witness) | `image` folded into `tile_rows` | knob-turn | add to SUPPORTED; the fold (`num_images · tiles_h_per_image`, per-image last-row test) is already in `derive_plan` and the kernel |
| alignment `h_non_aligned`, dtype bf16, layout TILE, rank 2 | built | — | — | — |

**Catalog entries that set knobs:**

| Entry | Finding used | Knob it set |
|-------|--------------|-------------|
| `double_buffer/README.md:96-99` | gain plateaus by about 4 outstanding transactions | the whole block in flight, one barrier |
| `trid_double_issue/README.md:166-169` | transaction IDs pay only for a CB-capped stream | plain reads, one barrier, no trids |
| `split_reader/README.md:10-15` | helps only when a DM RISC is issue-bound | one reader RISC by default (lamp L2) |
| `read_coalescing/README.md:85-87` | invalid across interleaved pages | one transfer (or two trimmed spans) per page is the minimum count (lamp L1) |
| `bank_stagger/README.md:59-60` | the trap is every core starting on the same bank | none: p1 owner `s` starts at page `5s`, and `5s mod 8` cycles all 8 banks, so no stagger knob |
| `noc_placement/README.md:30-32` | the NoC choice matters only when bandwidth-bound | NOC0 by default (lamp L3) |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block = one core's shard rectangle clipped to the tensor (`block_rows × block_cols` pages) |
| Grid | the kernel's core ranges are **`memory_config.shard_spec.grid` as given**: p1 is one range (0,0)-(7,3), p2 is one range (0,0). That is the caller's own range count (BB-17). Cores of the grid after the last shard run the kernel with an empty block and return |
| Core order | `G = ttnn.corerange_to_cores(grid, row_wise=(orientation == ttnn.ShardOrientation.ROW_MAJOR))`, the same function and argument the buffer allocator uses (`tt_metal/impl/buffers/buffer_distribution_spec.cpp:172`). Shard `s` → `G[s]`. Never hard-wired |
| Per-core work | width: `block_row_start = 0`, `block_rows = tensor_rows_tiles`, `block_col_start = s·shard_cols_tiles`, `block_cols = min(shard_cols_tiles, tensor_cols_tiles − block_col_start)`. Height: `block_row_start = s·shard_rows_tiles`, `block_rows = min(shard_rows_tiles, tensor_rows_tiles − block_row_start)`, `block_col_start = 0`, `block_cols = tensor_cols_tiles`. `s ≥ num_shards`: all four are 0 |
| Remainder | The ragged last shard narrows `block_cols`/`block_rows` and keeps the nominal stride `shard_cols_tiles`. Tile counts use `ceil` per image (`tensor_rows_tiles = num_images · ceil(H/32)`, `tensor_cols_tiles = ceil(W/32)`), and the last tile row of image `i` is `r = i·tiles_h_per_image + tiles_h_per_image − 1` |
| p1 / p2 | 32 owners × 1×5 pages, each read as 10 spans of 64 B / 1 owner × 1×4 pages, read as 8 spans |
| Per-call values | Only `input_tensor.buffer_address()` and `output.buffer_address()` change between calls of one key. Both travel as **common runtime args**, which `apply_descriptor_runtime_args` re-applies on a program-cache hit (`tt_metal/impl/program/program_descriptors.cpp:202-212`) |
| Descriptor cache (BB-07) | A module-level dict maps the key to the built `ProgramDescriptor`. A repeat call patches only the two address args and calls `ttnn.generic_op([input, output], descriptor)`, with no rebuild. The **key**: input logical shape, padded shape, dtype, layout, tile/face shape and transpose flags, input memory config (buffer type, layout), output memory config (buffer type, memory layout, shard grid, shard shape, orientation), device arch / DRAM alignment. The logical H is in the key because it sets `trim_span_bytes`, while the padded shape alone does not |
| Plan report (BB-08, BB-21) | `describe_plan(input_tensor, memory_config) -> dict`, exported from `ttnn.operations.g6_interleaved_to_sharded`. It runs `validate` and returns `derive_plan`'s view: `{"regime": "owner_fetch", "last_row_form": "trimmed" \| "whole_page", "trim_span_bytes": int, "num_shards": int, "cores": [{"core": (x, y), "role": "owner" \| "idle", "shard": s \| None, "block_rows": int, "block_cols": int}, ...]}`, with `cores` in G order. p1 → trimmed, 64, 32 owners |
| Zones (BB-21, profiler build only) | The reader kernel wraps the issue loop in `DeviceZoneScopedN("G6I2S-FETCH")` and the barrier in `DeviceZoneScopedN("G6I2S-DRAIN")`. Idle cores record `DeviceZoneScopedN("G6I2S-IDLE")`. The plain build is unchanged |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| — none — | — | — | — | No CB is allocated. Reads land directly in the output shard, whose address arrives as a common RT arg. A CB globally allocated on the output buffer was considered and not used: it would carry only that address, with no in-program consumer. The address arg is patched with one write on the cached descriptor and adds no CB config to the kernel-config region (BB-04/BB-18). Full accounting: `l1_ledger.md` | — | — | — | — |

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input (semantic name, pages, state) | Output (semantic name, pages) | State after |
|---|-----------------|-------------|---------|-------------------------------------|-------------------------------|-------------|
| 1 | `fetch_shard_block` | `block_rows × block_cols` pages; 2 spans per last-row page when `trim_span_bytes > 0`, else 1 | no: raw NoC (see API Mapping) | `x` (DRAM interleaved), pages `r·tensor_cols_tiles + c` for the block's rectangle | `output_shard` (this core's L1 shard), local pages `(r − block_row_start)·shard_cols_tiles + (c − block_col_start)` | reads in flight |
| 2 | `drain_block` | same | no: raw NoC | — | — | every page of the block landed; the kernel may return |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input | Output | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|-------|--------|------------------------------|
| host: core order | raw_api | `ttnn.corerange_to_cores(core_range_set, max_cores=None, row_wise)` | `ttnn/cpp/ttnn-nanobind/tensor.cpp:659-665`; allocator use `tt_metal/impl/buffers/buffer_distribution_spec.cpp:172` | `row_wise = orientation == ROW_MAJOR` | shard grid | G | — |
| host: input accessor args | raw_api | `ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()` | `.claude/references/generic_op_template/template_op_program_descriptor.py` (step 4) | appended last in the reader's CT args | — | — | — |
| host: launch | raw_api | `ttnn.generic_op([input_tensor, output_tensor], descriptor)` | `.claude/references/generic_op_template/template_op.py` | output last | — | — | — |
| `fetch_shard_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` (interleaved specialization) | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:554` | `offset ∈ {0, face1_offset_bytes}` | x | NoC address | `trim_span_bytes` (via offset/size) |
| `fetch_shard_block` | raw_api | `noc_async_read(src_noc_addr, dst_local_l1_addr, size)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | `size ∈ {trim_span_bytes, page_size}` | x | output shard | `block_rows`, `block_cols`, `trim_span_bytes` |
| `drain_block` | raw_api | `noc_async_read_barrier()` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1921` | — | — | — | — |

**Helpers considered and rejected** (raw NoC is used because kernel_lib has no DRAM→L1 page-read helper):

| Candidate | File:Line | Why it cannot express this op |
|-----------|-----------|-------------------------------|
| eltwise `copy` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:113` | It routes tiles through unpack → DEST → pack, the engine path spec rule 4 forbids (NaN payloads, denormals), and it needs a compute kernel plus two CBs for a pure move |
| `set_read_state` / `read_with_state` / `local_addr` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:9,57,75` | self-aimed L1→L1 only: `local_addr()` builds an endpoint at this core's own coordinates. The source here is a DRAM bank |
| `McastArgs` / `ttnn.Mcast` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp:18` | no page reaches more than one core (the operand-reuse check is negative) |
| tilize / untilize, reduce, `tile_column_dataflow`, `index_tile_dataflow` | `tilize_helpers_dataflow.hpp`, `reduce_helpers_dataflow.hpp`, `tile_column_dataflow.hpp:9`, `index_tile_dataflow.hpp:23` | no layout change, no reduction, no per-element stores |

The realization of each block operation (direct calls, a thin wrapper, or the stateful-read variant of lamp L4) is the implementer's choice. The contract is one barrier per block and no per-page synchronization.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Trimmed read misaligned | A DRAM read requires src ≡ dst mod 64 on Blackhole, and a mismatch corrupts silently | offsets 0/512 inside 2048-aligned pages; shard base DRAM-aligned by the allocator, asserted on the host per call; `trim_span_bytes` rounded to `ttnn.get_dram_alignment()` |
| Trim applied where rows exceed the top faces | `v > 16` puts logical rows in faces 2/3 | the trim predicate requires `v ≤ 16`; otherwise the last row moves whole pages |
| Trim decision taken per tensor instead of per image | rank 4 with `tiles_h_per_image > 1`: only the last tile row **of each image** is partial | per-row test `(r + 1) mod tiles_h_per_image == 0`; pinned by `height_multi_shard` and by the ragged width case |
| Ragged last shard at the wrong stride | the shard-local index must use the nominal `shard_cols_tiles`, not `block_cols` | stated in Work Distribution; pinned by the acceptance case `(33, 5120)` → 27 shards of 6 columns, the last one holding 4 |
| Hard-wired core order | both production placements are ROW_MAJOR | `G` from `corerange_to_cores(..., row_wise=orientation == ROW_MAJOR)`; acceptance and regression col-major cases |
| Stale program replay | `trim_span_bytes` depends on the logical H, which the padded shape does not show; generic_op hashes CT args but not RT values | the Python key includes the logical shape, the full shard spec and the orientation; generic_op re-applies RT args on every hit |
| Address patch lost | nanobind vector properties (`descriptor.kernels`, `common_runtime_args`) may return copies, so an in-place patch can miss the cached descriptor | re-assign the patched list onto the descriptor; pinned by the regression `new_input_address` and the acceptance cache test |
| Output padding holds stale L1 | a trimmed page writes only face rows `< round_up(v·32, 64)/32` of faces 0/1, and the rest of the output tile keeps prior L1 bytes | allowed: padding is not part of the result (contract r3:64). Do **not** add a fill; it would cost writes and buy nothing. The logical values are bit-identical run to run (BB-06) |
| Transfers in flight at return | the next program starts immediately (BB-03) | `drain_block` is the last NoC action on every owner; idle cores issue nothing |
| Malformed height request answered with a support refusal | height_sharded and rank 4 are outside Phase 0 SUPPORTED | all eleven malformed checks run before the support check; acceptance case `height_shard_narrower_than_tensor` pins it |
| ttnn rejects an inexact shard dimension at allocation | `tensor_spec.cpp:52-71` requires shard width == padded width (height) and shard height == padded height (width) | validate rules 8/9 raise `ValueError` first, with the spec's wording |
| 2×2 mesh in the model | one descriptor runs on every device | lockstep allocation gives identical addresses on all devices; no per-device arg |
