# Operation Design: g6_sharded_to_interleaved

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (placement change: one NoC write per tile, no arithmetic, no compute kernel) |
| Goal | Copy every tile of a height- or width-sharded L1 TILE tensor into a newly allocated interleaved TILE tensor (DRAM or L1, as `memory_config` names), bit for bit, as one device program in which each core writes the tiles of its own resident shard straight to their output pages. |
| Math | `out[i] = input_tensor[i]` for every logical index i; tile `(a, b)` of the 2-D tile view goes to interleaved page `p = a * C + b` |
| References | Contract: `docs/explorations/g6_sharded_to_interleaved/spec/g6_sharded_to_interleaved.txt`; math (F1 page map): `docs/explorations/g6_sharded_to_interleaved/math.md`; counts and shard geometry: `docs/explorations/g6_sharded_to_interleaved/tensix.md`; risks: `docs/explorations/g6_sharded_to_interleaved/risks.md`; golden universe: `eval/golden_tests/g6_sharded_to_interleaved/feature_spec.py`; exact gate: `eval/golden_tests/g6_sharded_to_interleaved/helpers.py` (`TOLERANCES`); block semantics: `.claude/references/blocking-model.md`; sharded page layout: `tt_metal/impl/buffers/buffer_distribution_spec.cpp:41-80` (`iterate_within_shard` / `iterate_over_shards`), `:168-174` (core list), `:399-449` (`squeeze_shape_ranks`); legacy shard validation: `tt_metal/impl/tensor/spec/tensor_spec.cpp:50-72` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes, positional | on device; TILE; HEIGHT_SHARDED or WIDTH_SHARDED in L1; shard shape a whole number of 32x32 tiles; rank 2 or 4; dtype bfloat16 (Phase 0) or bfloat8_b (TARGET) | none | geometry derived on host: shard-dependent values become CT args (per program) and per-core RT args; buffer address bound per call through the sharded CB |
| `memory_config` | `ttnn.MemoryConfig` | yes, positional | `memory_layout == INTERLEAVED`; buffer type DRAM (Phase 0) or L1 (TARGET) | none | buffer type is a CT arg (inside the output's `TensorAccessorArgs`); output address is per call |

No output dtype, no compute configuration, no `memory_format`, no output-tensor argument: the op converts placement, never values.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d_0, ..., d_{r-1})`, r ∈ {2, 4}, every d_k ≥ 1 |
| Dtype | bfloat16 (Phase 0); bfloat8_b (TARGET, knob-turn) |
| Layout | TILE (32x32) |
| Memory | L1, HEIGHT_SHARDED or WIDTH_SHARDED; shard spec (grid as an ordered list of core ranges, shard shape `(s_h, s_w)` elements with `s_h % 32 == 0` and `s_w % 32 == 0`, orientation) read from the tensor; a shorter last shard is legal |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` (per device) |
| Dtype | `input_tensor.dtype` |
| Layout | TILE |
| Memory | INTERLEAVED, buffer type of `memory_config` (DRAM or L1); newly allocated by this call on every device holding the input (`ttnn.allocate_tensor_on_device` on `input_tensor.device()`); page `p` holds tile `(p // C, p % C)` |

Tile geometry (alignment-aware, per image — never `floor`):

| Symbol | Formula | p2 `(1, 5120)` | p1 `(1, 128, 256, 32)` |
|--------|---------|----------------|------------------------|
| `tensor_rows_tiles` R | `prod(padded_shape[:-1]) / 32` (each image's `d_{r-2}` already rounded up to 32 in `padded_shape`) | 1 | 1024 |
| `tensor_cols_tiles` C | `padded_shape[-1] / 32` = `ceil(d_{r-1} / 32)` | 160 | 1 |
| P (pages) | `R * C` | 160 | 1024 |
| page bytes | `output.buffer_page_size()` = `input.buffer_page_size()` (same dtype): 2048 (bfloat16), 1088 (bfloat8_b) | 2048 | 1088 |
| `shard_rows_tiles` | `s_h / 32` | 1 | 10 |
| `shard_cols_tiles` | `s_w / 32` | 5 | 1 |
| shards | `ceil(R / shard_rows_tiles) * ceil(C / shard_cols_tiles)` (one factor is 1 for height/width sharding) | 32 | 103 (102 of 10 tiles, last of 4) |

## Support Contract (registry)

| Item | Decision |
|------|----------|
| `INPUT_TAGGERS` | exactly `tag_dtype`, `tag_layout`, `tag_in_scheme`, `tag_buffer`, each reading `inputs[0][<axis>]` of the scenario dict `{"shape", "dtype", "layout", "in_scheme", "shard_shape", "grid", "orientation", "buffer"}` |
| `SUPPORTED` (Phase 0) | `dtype: [ttnn.bfloat16]`, `layout: [ttnn.TILE_LAYOUT]`, `in_scheme: ["width_sharded"]`, `buffer: ["dram"]` |
| `EXCLUSIONS` | `[]` — never a size, shape, rank or shard-geometry entry |
| `validate()` order | (1) structural requirements → `ValueError`: not on device, not TILE, not sharded, not L1, memory layout other than HEIGHT/WIDTH_SHARDED, shard shape not whole tiles, rank ∉ {2, 4}, `memory_config` not INTERLEAVED; (2) build the scenario dict (in_scheme from `memory_layout`, shard_shape/grid ranges in stored order/orientation from `memory_config().shard_spec`, buffer from `memory_config.buffer_type`) and run the taggers; (3) `SUPPORTED` per axis → `UnsupportedAxisValue`; (4) `EXCLUSIONS` → `ExcludedCell`; (5) BB-05 bound: when the output is L1, the output's bytes per L1 bank must be ≤ 1347776 → `RuntimeError` otherwise (never reached by a golden shape) |
| Entry point | first line `validate(input_tensor, memory_config)` |
| Refinement path | every `TARGET − SUPPORTED` value is a knob-turn on the same program (see Dataflow Strategy, TARGET classification): `in_scheme: height_sharded`, `buffer: l1`, `dtype: bfloat8_b` |
| Structural impossibilities | none beyond `feature_spec.py`'s `INVALID = []` |

## Blocking Model

The op is an identity on values; the only freedom is which core writes which tile, in what order, with how many issuers. The input's shard spec has already fixed where every tile lies, and spec rule 4 (one NoC crossing per tile) makes the holder of a tile the only core that can write it in one crossing: a NoC write sources from the issuer's own L1 (`dataflow_api.h:876-885`), so any other core would first have to fetch the tile — a second crossing. The blocking therefore reads the per-core assignment *off the shard* and makes the whole resident shard one block.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `tile_rows` (all dims but the last, folded; R tiles) | independent — `out[i]` depends on `x[i]` alone, nothing spans it | `block_extent_rows` = `min(shard_rows_tiles, R − a0)` for this core's shard | the core's whole shard height (p2: 1; p1: 10, last core 4) | host function deriving the per-core shard rectangle from `input_tensor.memory_config().shard_spec` + padded shape; passed as per-core RT arg; every count below derives from it | the input's shard spec: shard k on core k of the grid walk (height sharding cuts this axis) | knob-turn (height_sharded is the same formula) |
| `tile_cols` (last dim; C tiles) | independent — elementwise identity | `block_extent_cols` = `min(shard_cols_tiles, C − b0)` | the core's whole shard width (p2: 5; p1: 1) | same host function; per-core RT arg | the input's shard spec (width sharding cuts this axis) | built (width_sharded is Phase 0) |
| block count per core | — (not a data axis: number of blocks a core walks) | `num_blocks_this_core` | 1 for a core holding a shard, 0 for a grid core past the last shard | same host function; per-core RT arg | — | — |
| issue lanes within a core (not a data axis: split of one block's local tile range across the two data-movement RISCs) | — | `num_lanes`; lane `j` owns local tile indices `[lane_start_j, lane_start_j + lane_tiles_j)`, `lane_tiles_0 = ceil(n_k / num_lanes)`, `n_k = block_extent_rows * block_extent_cols` | 2 (lane 0 = BRISC/NoC1, lane 1 = NCRISC/NoC0) | one host module constant; kernel-descriptor count, per-lane RT ranges derive from it | both data-movement RISCs of every shard-holding core | knob-turn (perf lamp L1) |

Step-1 table (one row per axis, every cell decided):

| Axis | Character | Block factor (Phase 0) | Core assignment | Buffer depth | Why |
|---|---|---|---|---|---|
| `tile_rows` | independent | whole shard height of this core (`block_extent_rows`) | fixed by the input's shard spec (shard k → core k of the ordered grid walk) | not a CB knob: the block is the resident shard; writes in flight = the lane's whole range before one barrier | the shard hands over the per-core extent; one block = whole resident shard (l1-footprint Rule 2, sharded) — no L1 cost, no reason to cut |
| `tile_cols` | independent | whole shard width of this core (`block_extent_cols`) | fixed by the input's shard spec | same | same |

No stage produces intermediate data: the scheme has exactly one stage (store), so there is no derived axis to re-block and no coordinating core.

Commitment: Phase 0 realizes the owner-writes scheme on every core of the input's grid that holds a shard, with both data-movement RISCs issuing. Every knob above — block extents, block count, lane count, grid — is a parameter derived from the input's shard spec or a single host constant; nothing is inlined for a production spec.

Why this is multi-core and why the core count is the blocking decision's choice: owner-writes puts every tile-holding core to work (p2: 32 cores × 2 lanes; p1: 103 cores × 2 lanes). A larger set is illegal (a non-owner needs a second crossing to touch a tile), a smaller set is illegal for the same reason, so the participating set is the maximum the traffic ranking allows — decided by the regime, not by a `min(num_tiles, grid)` shape limit.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_shard_lane0`, `cb_shard_lane1` | none — capacity is pinned to the input shard (`shard_rows_tiles * shard_cols_tiles` pages, globally allocated on the input's buffer) | whole shard | the shard is resident before the program starts; there is nothing to pipeline into it |
| (NoC, not a CB) write transactions in flight per barrier | `writes_per_barrier` | the lane's whole range (`lane_tiles_j`), one `noc_async_write_barrier` per lane at the end | maximal overlap of write issue and write flight; hardware back-pressures issue when the NoC command queue is full, so no software window is needed |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| bfloat8_b shared exponents (1 per 16 datums of a face row; tile = 1024 B mantissas + 64 B exponents, `tt_backend_api_types.hpp:129`) | transfer unit | the transfer unit is one whole page (1088 B / 2048 B); never split a page | a cut tile carries another tile's exponents or loses its own: wrong values on bfloat8_b only (risk R5) |
| Interleaved page placement (consecutive pages go to consecutive banks) | coalescing of a lane's writes | one NoC write per page; no multi-page transfer | a multi-page write to `get_noc_addr(p)` spills page `p+1` into bank `b`'s next slot (page `p + num_banks`): wrong page map |
| Sharded in-shard layout: slot `(r, c)` at `r * shard_cols_tiles + c`, row stride the *full* shard width even on a shorter last width shard (`buffer_distribution_spec.cpp:41-52` `iterate_within_shard` with shard strides; `buffer.cpp:468` legacy map) | source slot index | slot stride = `shard_cols_tiles` (from the shard spec), never `block_extent_cols` | the shorter last width shard reads the wrong slots when `shard_rows_tiles > 1` |
| Slot byte stride | source address | `input.buffer_aligned_page_size()` (the CB page size, `tensor_utils.cpp:67`) | misaligned slot reads if a dtype's page were not a multiple of the L1 alignment (both are: 1088 = 68·16, 2048 = 128·16) |
| NoC/bank alignment: DRAM 64 B, L1 16 B on Blackhole (`core_config.h:42-43`) | transfer size and destination offset | transfer = page bytes (1088 = 17·64, 2048 = 32·64), destination from the output `TensorAccessor` | a sub-page transfer could violate DRAM alignment (relevant only to perf lamp L2) |
| NoC burst (`NOC_MAX_BURST_SIZE` = 256 × 64 B = 16384 B, `noc_parameters.h:288-290`) | bytes per write | one page (≤ 2048 B) is one packet | — (never reached) |
| Legacy shard validation: `num_shards ≤ num_cores`, shard width = physical width (height), shard height = physical height (width) (`tensor_spec.cpp:50-72`) | `num_blocks_this_core` | ≤ 1 by construction; grid cores with index ≥ num_shards get 0 | — (enforced at tensor creation) |
| Shard → core order: shard k on the k-th core of `corerange_to_cores(grid, num_cores, orientation == ROW_MAJOR)` walking ranges in stored order (`buffer_distribution_spec.cpp:168-174`, `core_coord.cpp:604-626`) | core-assignment | host walks the *same* `CoreRangeSet` object read from the input, in stored order, with `ttnn.corerange_to_cores(grid, None, row_wise)`; never re-sorts or rebuilds it | every shard written to another shard's pages (risk R1/R12) |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **owner-writes**: each shard-holding core writes the tiles of its own resident shard, sourced in place from its L1, to their interleaved output pages; the block's local range is split across `num_lanes` RISCs | **built** (Phase 0, all of TARGET) | the validated domain: TILE, HEIGHT_ or WIDTH_SHARDED L1 input with whole-tile shards; DRAM or L1 interleaved output | `block_extent_rows × block_extent_cols` = this core's whole shard | **minimum.** Named boundary: the NoC. Each tile crosses it exactly once, as the write to its output page (p2: 160 × 2048 B to DRAM; p1: 1024 × 1088 B to L1 banks). The input crosses no tier (it is the write's local source). No DRAM crossing for p1; one DRAM write crossing of `out` for p2. | The block is already the whole shard (its bound). Fixed costs paid once per core per call: kernel launch and RT-arg load, `TensorAccessor` construction, the lane's start-index setup (one division), and one write barrier per lane. A smaller block would repeat the barrier and index setup for no traffic saving. |
| gather-then-write: fewer cores (or other cores) fetch remote shards over the NoC and write the pages | **rejected** — superseded by owner-writes everywhere | — | — | 2 NoC crossings per tile (fetch + write); forbidden by spec rule 4; no shape benefits since every tile already sits on a core that can write it | — |
| bank-affine re-split: each core writes the pages of one output bank so a core's pages share a bank | **rejected** — superseded by owner-writes | — | — | needs tiles from other shards: 2 crossings per tile; and pages adjacent in a bank (`p`, `p + num_banks`) are never adjacent in a shard, so no write coalescing is gained | — |
| compute-engine pass: unpack the shard to DEST and pack into a staging CB, then write | **rejected** — superseded by owner-writes | — | — | same NoC traffic plus a local unpack/pack pass and a staging CB; a bfloat8_b repack recomputes shared exponents (spec rule 7) | — |
| local staging copy: copy shard tiles into a scratch CB, then write from it | **rejected** — superseded by owner-writes | — | — | an extra L1→L1 pass and a scratch buffer; the NoC write can source the shard directly | — |
| split a dependent axis + cross-core combine | **rejected** — no dependent axis exists (math.md, "Axes and dependence") | — | — | nothing to combine; would only add traffic | — |
| host round trip or an existing placement op | **rejected** — forbidden (spec rules 3, BB-01/BB-02) | — | — | host crossing or a second program | — |
| block-sharded input | **deferred** — positive reason: BLOCK_SHARDED is outside TARGET (no production call); `validate()` refuses it with `ValueError` | `memory_layout == BLOCK_SHARDED` | same block | same minimum | same. Reachable: the per-core rectangle formula below is already the 2-D one (`(k_r, k_c)` over a shard grid); only the core walk changes to the trimmed 2-D grid (`buffer_distribution_spec.cpp:177-200`) |

Single built regime: there is no selection predicate and no regime-pinned test requirement beyond running both sharding strategies.

### Traffic ranking

Counted per tile, by tier (cheapest first: core-local L1, NoC, DRAM).

| Rank | Candidate split | Input crossings | Output crossings | Cross-core traffic | Verdict |
|------|-----------------|-----------------|------------------|--------------------|---------|
| 1 | the input's own shard split (owner-writes) | 0 (local source of the write) | 1 page write per tile: DRAM (p2) or L1 bank (p1) | P unicast writes from S cores, no rendezvous | **chosen** — the minimum; also the cheapest-traffic split |
| 2 | output-page split (contiguous page ranges per core, `split_work_to_cores`) | 1 NoC fetch per tile not owned by the writer (almost all) | 1 | 2P transfers | rejected (rule 4) |
| 3 | output-bank split | 1 NoC fetch per tile | 1 | 2P transfers | rejected (rule 4) |
| 4 | one or few gathering cores | 1 NoC fetch per tile | 1 | 2P transfers, serialized on few issuers | rejected (rule 4, risk R11) |
| — | dependent-axis split + combine | not applicable: no dependent axis | | | — |

Occupancy did not choose the split: rank 1 wins on crossings, and it happens to occupy every tile-holding core.

Operand-reuse check: one input operand; under the shard split no two cores need the same bytes, so no operand is reuse-shared and no broadcast row exists.

Stall-shadow check: the only wait is each lane's final `noc_async_write_barrier` (acks of its own writes). Every write of the lane is issued before it, and the other lane issues concurrently on the other NoC; no work independent of the acks remains to schedule into the wait.

### Block schedule

Logical schedule per lane (the two lanes of a core run concurrently, each on its own NoC):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // 0 or 1
    bind_resident_block(block_idx);   // shard base from cb_shard_laneX; (a0, b0, block_extent_rows, block_extent_cols); lane range
    store_block(block_idx);           // every tile of the lane's range: shard slot -> its interleaved output page, one write each
}
retire_writes();                      // one write barrier per lane before return (BB-03)
```

| Operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|-----------|------------------------|--------------------|-----------------------------------|
| `bind_resident_block` | `block_extent_rows × block_extent_cols` (the shard) | the whole shard (pre-placed input) | once per core per call: read the shard base address (`get_read_ptr`), construct the output `TensorAccessor`, compute the lane's starting `(r, c)` with one division |
| `store_block` | the lane's contiguous local range `[lane_start, lane_start + lane_tiles)` of the block, walked row-major with incremental `(r, c)` (no per-tile division) | the shard | one NoC page write per tile (the irreducible transaction: interleaved neighbours change bank); no barrier, no CB handshake inside the block |
| `retire_writes` | — | — | once per lane per call: `noc_async_write_barrier` on the lane's NoC |

Per tile `(r, c)` of the block: source = `shard_base + (r * shard_cols_tiles + c) * input_aligned_page_bytes`; destination page = `(a0 + r) * C + (b0 + c)`; size = page bytes.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 — issue lanes (`num_lanes` = 2) | p2 writes only 5 tiles per core; a second data-movement binary adds dispatch bytes and a second kernel group per core range, which may cost more than halving 5 issues saves (risk R6) | `num_lanes = 1` (BRISC/NoC1 only); judge on unprofiled trace-replay host time and model step, not profiled kernel time alone (lesson RB-053), separately for p2 and p1 |
| L2 — whole-page store of padded tiles | p2's tiles hold 1 logical row of 32: 2048 B written per tile where 64 B (row 0 of faces 0 and 1) carry data; DRAM bytes may weigh against the per-call fixed cost | bfloat16 only (spec rule 7 requires whole bfloat8_b tiles): store only the face rows that hold logical rows for tiles of a partially-filled last tile row — 2 writes of 32 B per p2 tile instead of 1 of 2048 B; check the Blackhole DRAM write alignment (64 B, `core_config.h:42`) accepts 32-B pieces before measuring |
| L3 — issue order | a shard whose tile count is a multiple of the bank count (e.g. 8 tiles to 8 DRAM banks) makes every core hit the same bank at the same step; production specs do not (p2: page `5k + j` spreads 32 cores over all 8 banks since gcd(5, 8) = 1; p1: L1 banks) | start each core's walk at local index `k mod lane_tiles` and wrap (catalog `bank_stagger`) |
| L4 — kernel core ranges | the kernel group uses the input grid as given (1 range for p2, 2 for p1), including grid cores with no shard that exit at once; dropping them could add ranges (lesson RB-036, BB-17) | kernel ranges trimmed to shard-holding cores; count ranges per RISC and judge on trace-replay start gap |

Overlap lamp: does not apply — there is no compute phase to overlap with; write issue already overlaps write flight (one barrier at the end). Grid-synchronization lamp: the alternative "fewer participating cores" is illegal (rule 4), so participation is fixed; L1 is the only participation knob.

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| input shard | TILE pages (Float16_b 2048 B / Bfp8_b 1088 B) at the shard's L1 address on its core | pre-placed by the producer; bound zero-copy to `cb_shard_lane0` and `cb_shard_lane1` (`ttnn.cb_descriptor_from_sharded_tensor`, two CB indices on the same buffer) | never read over the NoC; never through a `TensorAccessor`; no CB handshake (address binding only) |
| store | the same bytes, whole pages | `noc_async_write_page(page_id, out_accessor, src_l1_addr, page_bytes)`; lane 0 on BRISC (`RISCV_0`, NOC_1, `WriterConfigDescriptor` default), lane 1 on NCRISC (`RISCV_1`, NOC_0, `ReaderConfigDescriptor` default) | non-posted writes (acked), so the barrier proves completion |
| output | interleaved pages in DRAM banks (p2) or L1 banks (p1) | output `TensorAccessor` (interleaved; DRAM/L1 flag in its CT args) resolves bank and offset per page | no compute kernel, no TRISC binaries |

TARGET classification (the placement axes as regime rows):

| TARGET value | Cut it represents, and that axis's character | Class | What changes |
|--------------|-----------------------------------------------|-------|--------------|
| `in_scheme: width_sharded` | cut along `tile_cols` (independent) | built (Phase 0) | — |
| `in_scheme: height_sharded` | cut along `tile_rows` (independent) | knob-turn | nothing in the kernels: the per-core rectangle formula covers it (`n_cols_shards = 1`); widen `SUPPORTED` |
| `buffer: l1` | destination banks only | knob-turn | output `TensorAccessorArgs` carry L1; the BB-05 footprint check counts the output's L1 pages; widen `SUPPORTED` |
| `dtype: bfloat8_b` | page bytes 1088 | knob-turn | page and slot sizes are read from the tensors; whole pages move, so exponents travel with their tile; widen `SUPPORTED` |
| BLOCK_SHARDED (not in TARGET) | cut along both independent axes | deferred regime row | core walk only |

Mesh: one `ProgramDescriptor` (identical per-device geometry) dispatched by one `ttnn.generic_op` call runs on every device of the mesh; the output is allocated on the input's device object (the mesh) with the per-device shape, so every device's pages are written.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block = one core's whole resident shard |
| Grid | the input's shard grid as given (`input_tensor.memory_config().shard_spec.grid`, ranges in stored order); kernel groups and the two CBs use exactly that `CoreRangeSet` (1 range for p2, 2 for p1) |
| Core order | `cores = ttnn.corerange_to_cores(grid, None, orientation == ttnn.ShardOrientation.ROW_MAJOR)`; shard k ↔ `cores[k]` |
| Per-core work | `n_cols_shards = ceil(C / shard_cols_tiles)`; for k < num_shards: `k_r = k // n_cols_shards`, `k_c = k % n_cols_shards`, `a0 = k_r * shard_rows_tiles`, `b0 = k_c * shard_cols_tiles`, `block_extent_rows = min(shard_rows_tiles, R − a0)`, `block_extent_cols = min(shard_cols_tiles, C − b0)`, `n_k = block_extent_rows * block_extent_cols`, `num_blocks_this_core = 1`; for k ≥ num_shards: `num_blocks_this_core = 0` |
| Lanes | `lane_tiles_0 = ceil(n_k / num_lanes)`, `lane_start_0 = 0`; `lane_tiles_1 = n_k − lane_tiles_0`, `lane_start_1 = lane_tiles_0` (num_lanes = 2) |
| Remainder | the shorter last shard is a shorter runtime extent of the same block operation (p1: 4 rows on the last core; p2 other spec: 1 column); R and C use the per-image tile-padded `padded_shape` (`ceil`), never `floor(batch * H / 32)` |
| Why not `split_work_to_cores` | the shard spec already fixes the assignment; any other split adds a NoC crossing per tile (Traffic ranking) |

Production instances:

| Case | Cores (ranges) | Tiles per core | Lane split | Pages per destination bank |
|------|----------------|----------------|------------|----------------------------|
| p2 `(1, 5120)` bf16 width → DRAM | 32 (1: (0,0)-(7,3)) | 5 | 3 + 2 | 20 per DRAM bank (8 banks) |
| p1 `(1, 128, 256, 32)` bf8 height → L1 | 103 (2: (0,0)-(10,8), (0,9)-(3,9)) | 10 (last: 4) | 5 + 5 (last: 2 + 2) | `ceil(1024 / N_l1_banks)` |
| p2 other spec, col-major → DRAM | 54 (1: (0,0)-(8,5)) | 3 (last: 1) | 2 + 1 (last: 1 + 0) | 20 |
| p1 other spec, col-major → L1 | 94 (2: (0,0)-(8,9), (9,0)-(9,3)) | 11 (last: 1) | 6 + 5 (last: 1 + 0) | `ceil(1024 / N_l1_banks)` |

Plan report (BB-08, BB-21): the host exposes the per-core assignment of a call — core, shard index k, `(a0, b0, block_extent_rows, block_extent_cols)`, lane ranges — so a dump names each core's role; there is one plan, selected for every input.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_shard_lane0` | 0 | `input.buffer_aligned_page_size()` | `shard_rows_tiles * shard_cols_tiles` (the input buffer's `aligned_size_per_bank`; `cb_descriptor_from_sharded_tensor` default) — globally allocated on the input's shard, 0 new bytes | spans both block axes (it *is* the resident shard); capacity pinned by the shard | input dtype (Float16_b / Bfp8_b) | none — pre-placed by the producer op, no kernel pushes | writer lane 0 (BRISC) | whole program |
| `cb_shard_lane1` | 1 | same | same, same buffer | same | same | none — pre-placed | writer lane 1 (NCRISC) | whole program |

CB sync: push count = wait count = 0 on both CBs. The lanes use `get_read_ptr` as an address binding only; no `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` is issued, so no FIFO pointer moves. Two indices on one buffer keep one consumer per CB. Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `bind_resident_block` | `block_extent_rows × block_extent_cols` | no (raw dataflow API) | `cb_shard_laneX`, whole shard, resident since program start | — | unchanged (no pointer moved) |
| 2 | `store_block` | lane range of the block: `lane_tiles_j` tiles | no (raw dataflow API) | `cb_shard_laneX`, read by address | output tensor pages (no CB) | unchanged; shard bytes unmodified (input stays intact) |
| 3 | `retire_writes` | — | no (raw dataflow API) | — | — | all of the lane's writes acked |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| host: bind shard | raw_api (host) | `ttnn.cb_descriptor_from_sharded_tensor(cb_index, input_tensor)` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:517-523`; `ttnn/core/tensor/tensor_utils.cpp:48-72` | defaults: total size = per-bank shard bytes, page = aligned page size, core ranges = shard grid | `cb_shard_lane0`, `cb_shard_lane1` | — | none (pinned by the shard) |
| host: core walk | raw_api (host) | `ttnn.corerange_to_cores(grid, None, row_wise)` | `tt_metal/common/core_coord.cpp:604-626` | `row_wise = orientation == ROW_MAJOR` | — | — | core-assignment |
| `bind_resident_block` | raw_api | `get_read_ptr(cb)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:346` | lane's CB index (CT) | `cb_shard_laneX` | — | — |
| `store_block` | raw_api | `TensorAccessor(TensorAccessorArgs<…>(), out_addr, page_bytes)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:71-100`; `tt_metal/hw/inc/api/tensor/tensor_accessor_args.h:21` | output accessor (interleaved, DRAM/L1 in CT args) | — | output pages | — |
| `store_block` | raw_api | `noc_async_write_page<AddrGen>(page_id, out_accessor, src_l1_addr, page_bytes)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1417-1450` | `page_id = (a0 + r) * C + (b0 + c)`, `src = shard_base + (r * shard_cols_tiles + c) * slot_bytes`, non-posted | `cb_shard_laneX` | output pages | `block_extent_rows`, `block_extent_cols`, lane range (RT) |
| `retire_writes` | raw_api | `noc_async_write_barrier()` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1951` | lane's own NoC (default `noc_index`) | — | — | — |

Helpers considered and rejected (all `raw_api` rows above share this justification — the op has no compute phase and no helper moves resident pages to remote pages):

| Candidate | File:Line | Why it cannot express the operation |
|-----------|-----------|-------------------------------------|
| eltwise `copy` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:113` | a compute helper: unpacks to DEST and packs to an output dataflow buffer — adds an engine pass and a staging CB, and a bfloat8_b repack recomputes shared exponents (spec rule 7); the tile still needs a writer afterwards |
| local copy helpers | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:12-21` | L1 → L1 self-aimed *reads* into a local typed buffer; the op needs local → remote page writes, and a local copy is the rejected staging regime |
| `tilize` / `untilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:206`, `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:163` | layout conversion; input and output are both TILE |
| `McastArgs` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp:117-125` | no operand is shared by several cores; each tile has exactly one destination |
| `reduce` / `prepare_reduce_scaler` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:673` | no reduction |

No compute kernel exists, so `compute_kernel_hw_startup()`, `fp32_dest_acc_en` and math fidelity do not apply.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Wrong page map (R1) | orientation, two core ranges, shorter last shard and width-vs-height all change which tiles a core holds | one per-core rectangle formula; shard k ↔ `corerange_to_cores` of the input's own `CoreRangeSet` in stored order; slot stride `shard_cols_tiles`; acceptance test covers both production specs plus a column-major, uneven spec per case, with per-tile-distinct values, a dirty destination and special values |
| Hard-coded spec (R12) | every production cell passes with a hard-coded 32×5 / 103×10 layout | nothing is inlined: grid, shard shape, orientation and strategy are read from the input on every call and enter the program through CT/RT args |
| Shorter last width shard with `shard_rows_tiles > 1` | its in-shard row stride is the full shard width (padding slots) | slot index uses `shard_cols_tiles`, extents use `block_extent_cols` |
| Writes in flight at return (R3, BB-03) | the next program starts right after | non-posted writes and one `noc_async_write_barrier` per lane on its own NoC before return |
| bfloat8_b page size (R5) | 1088 B, not 1024 | page bytes and slot stride read from the tensors, never derived from element size |
| Cache key (R8, BB-07) | orientation alone changes only per-core RT values; dtype, shape, scheme, destination change CT args / CB format | per-core geometry is RT (bulk-copied on a C++ cache hit, `ttnn/api/ttnn/mesh_device_operation_adapter.hpp:345-348`); CT args carry `C`, `shard_cols_tiles`, page bytes, CB index and the output accessor (buffer type); the Python side memoizes the descriptor per key (padded shape, dtype, memory layout, shard shape, ordered grid ranges, orientation, output buffer type, device) and refreshes only the output address and the sharded-CB binding on a hit — it never rebuilds the descriptor |
| Input mutated or freed (R13) | the CBs alias the input buffer | lanes only read the shard; no pointer moves, no writes target input addresses; the input is never deallocated |
| Trace (R10, BB-16) | the model replays captured calls | the first eager call builds and caches the program; the output allocation is a host allocator action outside the recorded commands; no kernel is built after capture |
| Mesh (R9) | FLUX runs on 4 chips | one descriptor, one `generic_op` call on the mesh; output allocated on the mesh device |
| L1 budget (R4, BB-05) | p1's output pages land in L1 on the cores that hold the shards | the op allocates no L1 of its own (CBs alias the input); the only term is the output's L1 pages per bank, checked on host against 1347776 B |
| Profiling zones (BB-21) | time outside a zone reads as waiting | each lane wraps `store_block` and `retire_writes` in its own named zone in the zone build only |
