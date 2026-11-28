# Operation Design: g6_sharded_to_interleaved

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no arithmetic) |
| Goal | Copy a height- or width-sharded L1 TILE tensor into a newly allocated interleaved tensor (DRAM or L1, as `memory_config` names), bit for bit, in one device program. Each shard core writes its own resident tiles straight to their output pages, so every tile crosses the NoC once and nothing is read over the NoC. |
| Math | `out[a, b] = in[a, b]` for every tile (a, b) of the padded 2-D tile grid; output page `p = a * tensor_tile_cols + b`. Logical values equal `torch.clone(input_tensor)`. |
| References | `eval/golden_tests/g6_sharded_to_interleaved/feature_spec.py` (TARGET, INPUTS, LOOSE_CASES); `eval/golden_tests/g6_sharded_to_interleaved/helpers.py` (oracle, exact gate); `eval/op_template.py` (registry model); `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; `ttnn/ttnn/operations/examples/master.md` (perf catalog); `tt_metal/impl/buffers/buffer_distribution_spec.cpp` (shard page map); `ttnn/cpp/ttnn/operations/generic/` (generic_op caching and mesh dispatch) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes, positional | on device; rank 2 or 4; TILE; HEIGHT_SHARDED or WIDTH_SHARDED in L1; shard shape a whole number of 32x32 tiles; dtype in TARGET | none | shard base address: RT, through the binding of `cb_input_shard`; shard geometry: per-core RT; page size: CT |
| `memory_config` | `ttnn.MemoryConfig` | yes, positional | INTERLEAVED, `buffer_type` DRAM or L1 | none | bank layout: CT (output `TensorAccessorArgs`); output base address: common RT |

The shard spec (grid, shard shape, orientation) is read from `input_tensor`; it is never a parameter. There is no dtype, layout, `memory_format` or compute-config parameter.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d0, d1)` or `(d0, d1, d2, d3)`, every size ≥ 1 |
| Dtype | `bfloat16` (Phase 0); `bfloat8_b` (TARGET, refinement) |
| Layout | TILE |
| Memory | L1, HEIGHT_SHARDED or WIDTH_SHARDED; shard shape `(rows, cols)` both multiples of 32; any grid, any number of core ranges, ROW_MAJOR or COL_MAJOR orientation; the last shard may hold fewer tiles |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` (per device) |
| Dtype | `input_tensor.dtype` |
| Layout | TILE |
| Memory | INTERLEAVED in `memory_config.buffer_type` (DRAM or L1); a new buffer allocated by this call on every device that holds the input |

### Validation and support gate

`validate(input_tensor, memory_config)` is the entry point's first line. Structural checks come first and raise `ValueError` (a refusal must never be a support refusal for these); the support gate comes after, on the scenario built from the tensors.

| Order | Check | Raises |
|-------|-------|--------|
| 1 | `input_tensor` is a `ttnn.Tensor` stored on a device | `ValueError` |
| 2 | rank ∈ {2, 4} | `ValueError` |
| 3 | `layout == TILE_LAYOUT` | `ValueError` |
| 4 | input memory config is sharded and `buffer_type == L1` | `ValueError` |
| 5 | `memory_layout ∈ {HEIGHT_SHARDED, WIDTH_SHARDED}` (BLOCK_SHARDED refused) | `ValueError` |
| 6 | `shard_spec.shape[0] % 32 == 0 and shard_spec.shape[1] % 32 == 0` | `ValueError` |
| 7 | `memory_config.memory_layout == INTERLEAVED` and `buffer_type ∈ {DRAM, L1}` | `ValueError` |
| 8 | build `scenario = {shape, dtype, layout, in_scheme, shard_shape, grid, orientation, buffer}` from the tensors (`grid` = the shard spec's core ranges in their order, `buffer` from `memory_config`), then apply `INPUT_TAGGERS` in the order `dtype, layout, in_scheme, buffer` | none |
| 9 | every axis value in `SUPPORTED` | `UnsupportedAxisValue` |
| 10 | no `EXCLUSIONS` entry matches (Phase 0: empty list) | `ExcludedCell` |

| Registry item | Phase 0 value |
|---------------|---------------|
| `INPUT_TAGGERS` | `tag_dtype` → `inputs[0]["dtype"]`; `tag_layout` → `inputs[0]["layout"]`; `tag_in_scheme` → `inputs[0]["in_scheme"]` (`"width_sharded"` / `"height_sharded"`); `tag_buffer` → `inputs[0]["buffer"]` (`"dram"` / `"l1"`). No other tagger, no size, shape, rank or shard-geometry axis. |
| `SUPPORTED` | `dtype: [ttnn.bfloat16]`, `layout: [ttnn.TILE_LAYOUT]`, `in_scheme: ["width_sharded"]`, `buffer: ["dram"]` |
| `EXCLUSIONS` | `[]` |
| Package exports | `g6_sharded_to_interleaved`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS` from `ttnn/operations/g6_sharded_to_interleaved/__init__.py` (the golden suite imports all four) |
| Structural impossibilities | none beyond `feature_spec.py` (INVALID is correctly empty) |

The kernels and the descriptor builder carry no branch on dtype, `in_scheme` or buffer type: each enters as a value read off the tensors (page size, shard extents, output accessor). Widening `SUPPORTED` to TARGET is the whole of each refinement, plus its verification.

## Blocking Model

### Axes

Index space: the padded 2-D tile grid of the input, rows = every dimension but the last. `tensor_tile_rows = prod(padded_shape[:-1]) / 32` (rank 4: `d0 · d1 · ceil(d2/32)`, tile padding per image; rank 2: `ceil(d0/32)`), `tensor_tile_cols = padded_shape[-1] / 32`. Leading dims fold into `tile_row`: the op treats them identically and they are contiguous in the 2-D view that both the shard spec and the interleaved page index are defined on.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `tile_row` (a) | independent: `out[a,b]` reads only `in[a,b]`, no result spans rows | `block_tile_rows` | the core's whole shard extent along rows: `min(shard_tile_rows, tensor_tile_rows − block_first_tile_row)` (height sharded); `tensor_tile_rows` (width sharded, `shard_tile_rows == tensor_tile_rows`) | host, once per descriptor build, from `shard_spec.shape[0] / 32`; per-core RT arg | pinned by the physical shard: shard k is owned by `cores[k]` | none needed. Sub-blocking is a knob-turn but buys nothing (no staging buffer, no overlap partner) and adds barriers |
| `tile_col` (b) | independent: same reason, no result spans columns | `block_tile_cols` | `min(shard_tile_cols, tensor_tile_cols − block_first_tile_col)` (width sharded); `tensor_tile_cols` (height sharded, `shard_tile_cols == tensor_tile_cols`) | host, from `shard_spec.shape[1] / 32`; per-core RT arg | pinned by the physical shard | as for `tile_row` |
| tile interior (32x32 elements, faces, bfp8 exponent groups) | independent, but the page is the atom: a tile is one interleaved page and one NoC transfer | whole page, 1 tile | 1 page | `ttnn.tile_size(dtype)` → CT page size | not split: a sub-tile write fragments the transfer (catalog `tile_reorder`) and a bfp8 tile must move with its shared exponents | none |
| device (mesh coordinate) | independent: each chip copies its own per-device tensor, nothing crosses chips | 1 device per program instance | 1 | mesh shape of `input_tensor.device()` | SPMD: the same program on every chip (`generic_op` with a `ProgramDescriptor`, `generic_op.cpp:13-23`) | none |

Extents are named by scope: `tensor_tile_rows/cols` (tensor), `shard_tile_rows/cols` (the shard spec, nominal), `block_first_tile_row/col` and `block_tile_rows/cols` (this core's block). Phase 0 cuts each shard into exactly one block, so `num_blocks_this_core ∈ {0, 1}` (0 only for a grid core without a shard).

Why the whole shard is the block: the block is consumed in place from the resident shard and written straight to its pages, so its working set adds no L1 at any extent; the only per-block fixed costs (the write barrier, the zone pair, the RT-arg and accessor setup) are paid fewest times at one block. Splitting into more blocks has no L1 pressure to answer (`l1-footprint-discipline.md` Rule 2, sharded case).

| Additional core-assignment knob | Phase 0 value | Source | What it is |
|---------------------------------|---------------|--------|------------|
| `num_write_lanes` | 1 | host constant in the descriptor builder, threaded to the per-lane RT args | data-movement RISCs per shard core that issue the block's writes. 1 = the writer on BRISC / NoC1. At 2 the host cuts the block into two rectangles along its longer tile axis and adds a second instance of the same kernel on NCRISC / NoC0. A knob-turn: no kernel change, no loop-nest change. |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_shard` | none: its capacity is the input's own shard buffer, pinned by the caller | `shard_tile_rows · shard_tile_cols` pages (the whole shard) | nothing to overlap: the data is resident before launch and is read in place |
| (no streaming CB exists) | — | — | The in-flight depth of this op is the NoC's outstanding-write window. One barrier at the end of the block leaves every write of the block in flight together (catalog `double_buffer`, `trid_double_issue`: issue the batch, then one barrier). |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| NoC packet: `NOC_MAX_BURST_SIZE` = 256 words × 64 B = 16384 B on Blackhole (`noc_parameters.h:288-290`) | transfer unit = one page | host asserts `page_size ≤ 16384` (2048 B bf16, 1088 B bf8b); the write uses the one-packet path with `max_page_size = page_size` | a size above the one-packet bound given to the one-packet path truncates the transfer |
| Interleaved bank map: page p lives in bank `p mod n_banks` | transfer extent = 1 page; consecutive output pages are in different banks | one NoC write per page, never a multi-page write | a coalesced write puts the following pages into the first page's bank: wrong page map |
| Shard geometry (caller-pinned) | `block_tile_rows`, `block_tile_cols` ≤ the shard; the last shard is clipped at the tensor edge | `block_tile_rows = min(shard_tile_rows, tensor_tile_rows − block_first_tile_row)`, same on cols; cores `k ≥ num_shards` get 0 blocks | the padding slots of a short last shard are written to pages past `tensor_tile_rows · tensor_tile_cols`: writes beyond the output buffer |
| Slot stride of a short shard: slots keep the full shard stride (`buffer_distribution_spec.cpp:41-52` advances the slot by `shard_strides`, the nominal shard shape) | source slot of local tile (r, c) | `slot = r · shard_tile_cols + c`, the nominal `shard_tile_cols`, never `block_tile_cols` | a width-sharded tensor with `tensor_tile_rows > 1` and a short last shard reads the wrong tiles from it |
| Page alignment | source slot stride, destination page size | host asserts `input_tensor.buffer_aligned_page_size() == output_tensor.buffer_page_size() == ttnn.tile_size(dtype)`; both tile sizes are multiples of the 64 B DRAM and 16 B L1 alignment | source and destination at different alignment offsets: shifted bytes or a NoC alignment fault |
| Shards per core (legacy height/width sharding: one shard per core) | `num_shards ≤ len(cores)` | host assert at descriptor build | a shard without an owner: its pages are never written, which a dirty destination exposes |

No DEST, no accumulator and no compute capacity bound exists: the op has no compute kernel.

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `owner_writes` | built | the sole built regime: every input that passes `validate()` (height or width sharded L1 TILE, either output buffer type, any shard spec) | `block_tile_rows × block_tile_cols` = the core's whole shard rectangle clipped to the tensor; one block per shard core | **Minimum.** Boundary: the NoC. Minimum = each tile crosses once, as the write to its output page; no NoC read. Actual: one write per tile, zero reads. DRAM: input 0 crossings (it lives in L1); output 1 write crossing when `buffer == "dram"`, 0 when `"l1"`. | The fixed cost per block is one `noc_async_write_barrier`, one zone pair, one RT-arg load with accessor and read-pointer setup. Intended frequency: once per core per call. One block per core is the minimum, so no larger block exists. |
| `destination_pull` | rejected, superseded by `owner_writes` | — | an output bank core's pages | Also one crossing per tile, as a NoC read into the page. Exists only for L1 outputs: a DRAM bank cannot issue a read. | Same participation at the production shapes (p1: 103 owners against ≤ one bank core per worker core, about 10 pages each), a second code path that covers only half of TARGET's buffers, and a per-page reverse map (page → shard core, slot). The contract also names the write to the output page as the single crossing. |
| `relay_through_non_owner` (any split that is not the shard split: tile rows/cols over the full grid, a single core, a gatherer) | rejected, superseded by `owner_writes` | — | any | Two crossings per tile: a fetch from the owner, then the write. The contract forbids fetching a tile from another core's shard. | — |
| `staged_copy_through_cb` (a reader copies the shard into a CB, a writer drains it) | rejected, superseded by `owner_writes` | — | a CB window | An extra self-aimed NoC read per tile plus a CB handshake, with no compute to overlap: two crossings. The zero-copy read pointer of `cb_input_shard` supersedes it. | — |
| `two_dispatch` / DRAM-staged (an existing reshard or copy op, or a scratch interleaved buffer) | rejected, superseded by `owner_writes` | — | — | A second program, an extra full-size buffer, every tile twice. Forbidden (BB-01, contract). | — |

Why one row covers the domain: every axis is independent and the shard pins each tile to the core whose L1 holds it. For a DRAM destination only that core can issue the tile's single crossing, so `owner_writes` is the only one-crossing regime that covers both buffer types. No dependent axis exists, so no combine regime exists. No operand feeds more than one destination, so no multicast regime exists.

### Traffic ranking

| Candidate split | Who moves a tile | NoC crossings per tile | DRAM crossings (input / output) | Covers | Rank |
|-----------------|------------------|------------------------|---------------------------------|--------|------|
| by physical shard (owner) | the owner writes from its L1 | 1 (write) | 0 / 1 if DRAM output, 0 if L1 | both buffers | 1, chosen |
| by output bank (destination pull) | the bank core reads | 1 (read) | 0 / 0 | L1 output only | 1 (tie), rejected on coverage |
| by tile row / tile col over the full grid (`split_work_to_cores`) | a worker fetches from the owner, then writes | 2 | 0 / 1 if DRAM | both | 3 |
| single core | one core fetches every remote tile, then writes | 2 (1 for its own shard) | 0 / 1 if DRAM | both | 4 (also serial) |
| dependent axis + cross-core combine | n/a: no dependent axis | — | — | — | — |
| broadcast of a reuse-shared operand | n/a: the input varies along both split axes, so it is not reuse-shared; each tile has exactly one destination page | — | — | — | — |

Chosen: the owner split. It is the traffic minimum and it puts work on every shard core: p2 32 cores × 5 tiles, p1 103 cores × 10 tiles (the last core 4). The core count is the blocking decision's consequence, not a shape limit. The single-crossing rule pins each tile to its owner, and every owner works.

Stall-shadow check: the only wait is the final write barrier. Every write of the block is issued before it, so no independent work is left to schedule into that window.

### Block schedule

```cpp
// writer kernel, one instance per lane on every core of the input shard grid
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {  // num_blocks_this_core ∈ {0, 1}
    store_block(block_idx);  // every tile of the resident shard rectangle -> its output page, one NoC write each
}
drain_writes();              // one write barrier per kernel
```

| Block operation | Block shape | Resident across it | Intended frequency of inits / multicast / barriers |
|-----------------|-------------|--------------------|----------------------------------------------------|
| `store_block` | `block_tile_rows × block_tile_cols` tiles (the lane's rectangle when `num_write_lanes = 2`) | the whole input shard, read in place through `cb_input_shard` (never waited on, never popped, never written); the output accessor | read pointer and accessor built once per kernel; no barrier inside; no multicast; one non-posted single-packet NoC write per tile. Source slot `r · shard_tile_cols + c`, destination page `(block_first_tile_row + r) · tensor_tile_cols + block_first_tile_col + c`. Inside a row the page and the slot both advance by one per tile. |
| `drain_writes` | — | — | exactly one `noc_async_write_barrier` per kernel, also on a core with 0 blocks (a no-op there). After it every write has been acked, as BB-03 requires. |

There is no `load_block`: the block is already in the core's L1. The kernel never issues a NoC read.

Profiler zones (BB-21): `store_block` runs inside `DeviceZoneScopedN("S2I-STORE-BLOCK")` and `drain_writes` inside `DeviceZoneScopedN("S2I-WRITE-BARRIER")`. Both compile to nothing outside the zone build (`kernel_profiler.hpp:1047`).

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| write lanes (`num_write_lanes = 1`) | p1 issues 10 writes of 1088 B per core from one RISC. If the writer is issue-bound rather than NoC-bound, a second RISC on the other NoC halves the issue path. Catalog `split_reader`: this pays only when a data-movement RISC is the bottleneck. Catalog `noc_placement`: writes to interleaved DRAM on NoC0 measured 2.5–4.8× slower than on NoC1, so judge the DRAM (p2) and L1 (p1) destinations separately. | `num_write_lanes = 2`: the block cut along its longer tile axis (p1 rows 5+5, p2 cols 3+2), lane 1 on NCRISC / NoC0. Judge on unprofiled trace-replay host time, not the profiled kernel time (lesson RB-053). **Refinement 2:** `default_knobs` turns on 2 lanes (with `bank_stagger`) for an L1 destination with ≥ `TWO_LANE_MIN_SHARD_PAGES` (8) pages per shard: p1 kernel 2119 → ~1712 ns, replay flat. p2 to DRAM (+7% kernel) and p2 to L1 (+4–5% replay) keep 1 lane. `num_write_nocs = 2` (one lane alternating both NoCs, dynamic NoC mode) is a parked knob: p1 1804 ns, slower than 2 lanes. |
| page address per tile (`TensorAccessor::get_noc_addr(page)`) | The accessor does a bank div/mod per page on the issue path; with 5 or 10 pages per core the per-page arithmetic is a visible share of the core's time. | An incremental bank walk inside a row: the bank index +1 with wrap, the bank offset +page_size on wrap. |
| overlap lamp | does not apply: no compute phase exists to overlap with, and the whole block is in flight before the single barrier | — |
| grid-synchronization lamp | does not apply: no collective, semaphore or multicast exists. Participation is pinned by ownership: a "fewer cores" alternative would move tiles twice. | — |

Catalog entries that set the knobs (`ttnn/ttnn/operations/examples/master.md`):

| Knob / decision | Entry | What it set |
|-----------------|-------|-------------|
| one barrier per block, all writes in flight | `double_buffer`, `trid_double_issue` | never write-one / barrier; batch, then one barrier |
| transfer unit = one whole page per write | `tile_reorder` | whole 2 KB tile in one NoC write, no face scatter |
| writer on NoC1 | `noc_placement` | writes on NoC1 for spread placements into interleaved DRAM |
| `num_write_lanes` default 1, lamp | `split_reader` | a second issuer only when issue-bound |
| no bank stagger | `bank_stagger` | not needed: shard k starts at page `k·block_tiles`, so at write j the cores hit banks `(k·block_tiles + j) mod n_banks`, already spread (p2: banks `5k mod 8`). **Refinement 2:** wrong for p1. The L1 destination has 110 banks and `10k mod 110` takes 11 values, so all cores of a column hit the same bank at each step. `bank_stagger` (rotation `(k // issue_period) % rect_rows`, two-segment walk) is now a knob, on whenever 2 lanes are. It is slower on one NoC (2239 vs 2139 ns) and faster on two (1712 vs 1868 ns). |
| core participation for a one-tile-row tensor | `width_split` | p2 (`tensor_tile_rows = 1`) is width sharded by the caller; the owner split puts its 160 tiles on 32 cores. A tile-row split would strand it on one core. |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input shard (resident before launch) | TILE pages, `page_size` = 2048 B (bf16) / 1088 B (bf8b, mantissas + shared exponents); slots row-major within the shard at the nominal shard stride | `cb_input_shard`, globally allocated on the input's buffer by `ttnn.cb_descriptor_from_sharded_tensor`; the writer takes the base from `get_read_ptr(cb_input_shard)` | zero-copy; never read over the NoC; never written. The input stays allocated, sharded and unchanged. |
| Shard slot → output page | the same bytes, unchanged | one non-posted single-packet NoC write per page from the owner (BRISC, NoC1) to `TensorAccessor(output).get_noc_addr(page)` | bank set (DRAM banks or L1 banks) comes from the output's `TensorAccessorArgs`. A bfp8 tile moves as raw bytes, never through unpack/pack. |
| Completion | — | `noc_async_write_barrier` once per kernel | no transaction outlives the kernel (BB-03) |
| Mesh | per-device tensors at identical addresses (lock-step mesh allocation) | `ttnn.generic_op([input, output], ProgramDescriptor)` → SPMD over the whole mesh shape (`generic_op.cpp:13-23,34-36`) | one RT-arg set serves every chip. The output is allocated on every device by `allocate_tensor_on_device(..., mesh_device, ...)`. |

No `deferred` regime needs Tensix-to-Tensix communication; the op has none.

### Host path (per call)

| Step | What | When |
|------|------|------|
| `validate()` | the table under Tensors | every call, first line |
| allocate output | `ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, TILE_LAYOUT, device, memory_config)`, positional args | every eager call. Under trace this runs at capture time only; a replay executes the recorded program and allocates nothing. |
| plan + descriptor build | the plan (cores, per-core blocks, coverage assert, L1 bound check) and the `ProgramDescriptor`, memoized in a module-level dict | cache miss only. Key: device / mesh identity, `shape`, `dtype`, input `memory_layout`, shard shape, the ordered core ranges, orientation, output `buffer_type`, `num_write_lanes`. **Orientation and range order must be in the key**: they change the per-core RT values without changing the C++ hash. |
| patch | (1) the output base address, a common RT arg of the writer; (2) the backing of `cb_input_shard`, rebound to this call's `input_tensor` | every call, before dispatch |
| dispatch | `ttnn.generic_op([input_tensor, output_tensor], descriptor)`, output last | every call, exactly one program |

Why the patch is enough: the C++ program-cache hash of a descriptor covers kernel sources, CT args, core ranges, RT-arg counts and CB layout, never RT values or addresses (`generic_op_device_operation.cpp:78-147,307-324`). On a hit the framework copies the per-core and common RT values and rebinds every buffer-backed CB from the descriptor passed in (`program_descriptors.cpp:202-242`). Every per-call value is therefore an RT value or the CB binding, and each per-core shard quantity is a per-core RT arg, never a CT arg.

Plan report (BB-08, BB-21): the module exposes `describe_plan(input_tensor, memory_config) -> dict` with `plan = "owner_writes"`, `num_write_lanes`, the per-core role (`"writer"` for cores with a block, `"idle"` otherwise) and each block `(first_row, first_col, rows, cols)`, all read from the memoized plan.

### TARGET axes as blocking

| TARGET value | What it changes | Class |
|--------------|-----------------|-------|
| `in_scheme = "width_sharded"` | shards cut `tile_col`, an independent axis: each core's block is self-contained | knob-turn: the block rectangle comes from the shard spec |
| `in_scheme = "height_sharded"` | shards cut `tile_row`, an independent axis | knob-turn: the same block formula, clipped on rows instead of cols; no new kernel path |
| `buffer = "l1"` | the output's bank set | knob-turn: the output `TensorAccessorArgs` CT args |
| `dtype = bfloat8_b` | page size 1088 B | knob-turn: the CT page size; the bytes move unchanged with their exponents |

Each physical shard is the realized form of the logical shard the work-split already defines. None of these values needs a new algorithm or a new regime row.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block = the core's shard rectangle, `block_tile_rows × block_tile_cols` |
| Grid | the input's own shard grid, its core ranges as given (p2: 1 range, 32 cores; p1: 2 ranges, 103 cores; p2-other: 1 range, 54 cores; p1-other: 2 ranges, 94 cores). The writer kernel and `cb_input_shard` take exactly these ranges, so the kernel group has as many core ranges as the input (BB-17). |
| Shard → core | `cores = ttnn.corerange_to_cores(shard_spec.grid, None, row_wise=(orientation == ROW_MAJOR))`, computed from the tensor's own `shard_spec.grid` object, never from a rebuilt or sorted range list (the buffer's own map uses this function: `buffer.cpp:440`, `buffer_distribution_spec.cpp:172-174`; ranges walked in order, `core_coord.cpp:604-625`). Shard k → `cores[k]`. |
| Shard grid | `shards_along_rows = ceil(tensor_tile_rows / shard_tile_rows)`, `shards_along_cols = ceil(tensor_tile_cols / shard_tile_cols)`, `num_shards = shards_along_rows · shards_along_cols` (one factor is 1: height → `shards_along_cols = 1`, width → `shards_along_rows = 1`). Shard k sits at shard-grid position `(k // shards_along_cols, k % shards_along_cols)`, row-major as in `iterate_over_shards`. |
| Per-core work | `block_first_tile_row = (k // shards_along_cols) · shard_tile_rows`, `block_first_tile_col = (k % shards_along_cols) · shard_tile_cols`; extents per the mechanism-cap clamp |
| Remainder | the last shard is clipped (p1: core 102 holds 4 tile rows; p2-other: the last shard holds 1 tile column; p1-other: 1 tile row). Cores `k ≥ num_shards` get `num_blocks_this_core = 0` and still receive RT args. Tile geometry uses `ceil` and the per-image padded height. |
| Coverage invariant | host-asserted at descriptor build: `Σ block_tile_rows · block_tile_cols == tensor_tile_rows · tensor_tile_cols`, and the blocks are disjoint. Every output page is written by the call. |
| Geometry asserts (host, `RuntimeError`) | height sharded: `shard_tile_cols == tensor_tile_cols`; width sharded: `shard_tile_rows == tensor_tile_rows`; `num_shards ≤ len(cores)` |
| L1 bound (BB-05) | on a cache miss: when `buffer == "l1"`, `ceil(tensor_tile_rows · tensor_tile_cols / n_l1_banks) · page_size + 0 ≤ 1347776`, else `RuntimeError` (`n_l1_banks` from the device's L1 memory view). Never a support refusal and never a tag. Every INPUTS shape passes even at one bank (1,114,112 B). |
| Regime selection | single regime, no predicate; reported by `describe_plan` |
| `split_work_to_cores` | not used: the physical shard pins the core assignment |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_shard` | 0 | `input_tensor.buffer_aligned_page_size()` (2048 / 1088) | `shard_tile_rows · shard_tile_cols`, set by `ttnn.cb_descriptor_from_sharded_tensor(0, input_tensor)` (total size = the input's bank size) | It is the shard: it spans `tile_row` (`shard_tile_rows`) and `tile_col` (`shard_tile_cols`) and streams over nothing. Globally allocated on the input buffer, so it allocates no L1. | input dtype (`Float16_b` / `Bfp8_b`), a dataflow-only payload that never enters DEST | none in this program: the shard is resident, written by the program that produced `input_tensor` | writer | whole program, read-only, no CB handshake |

No other CB exists. The output is written through `TensorAccessor` to its interleaved pages, never staged.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `store_block` | `block_tile_rows × block_tile_cols` | no (raw dataflow; see API Mapping) | `cb_input_shard`, the whole shard, resident, read through `get_read_ptr`, never waited on or popped | none: output pages via `TensorAccessor` (`buffer_type` from `memory_config`) | `cb_input_shard` unchanged; writes in flight |
| 2 | `drain_writes` | — | no | — | — | every write acked |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `store_block` (source base) | raw_api | `get_read_ptr(cb_input_shard)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:346` | CB index | `cb_input_shard` | — | none |
| `store_block` (destination) | raw_api | `TensorAccessorArgs<cta_offset>()`, `TensorAccessor(args, output_addr, page_size)`, `get_noc_addr(page_id)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:71,165`; `tech_reports/tensor_accessor/tensor_accessor.md:61-72` | host: `ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()` appended last to the CT args | — | output pages | none |
| `store_block` (move) | raw_api | `noc_async_write<page_size>(src_l1_addr, dst_noc_addr, page_size)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:875-883` | `max_page_size = page_size` (≤ `NOC_MAX_BURST_SIZE`, so the one-packet path); `posted = false` | `cb_input_shard` | output pages | the call count is `block_tile_rows · block_tile_cols`, the extent knobs |
| `drain_writes` | raw_api | `noc_async_write_barrier()` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1951` | — | — | — | none |
| zones | raw_api | `DeviceZoneScopedN(name)` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012,1047` | `"S2I-STORE-BLOCK"`, `"S2I-WRITE-BARRIER"` | — | — | none |
| host: shard CB | host | `ttnn.cb_descriptor_from_sharded_tensor(cb_index, tensor)` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:517-555` | `cb_index = 0`; default size and core ranges | — | — | none |
| host: shard→core list | host | `ttnn.corerange_to_cores(core_range_set, max_cores, row_wise)` | `ttnn/cpp/ttnn-nanobind/tensor.cpp:659-665`; `tt_metal/common/core_coord.cpp:604-625` | `row_wise = orientation == ROW_MAJOR` | — | — | sets `num_blocks_this_core` per core |
| host: dispatch | host | `ttnn.generic_op(io_tensors, program_descriptor)` | `ttnn/cpp/ttnn/operations/generic/generic_op.cpp:34-36` | `[input_tensor, output_tensor]`, output last | — | — | — |

**Helpers considered and rejected** (they apply to every `raw_api` row above):

| Candidate | File:Line | Why it cannot express this op |
|-----------|-----------|-------------------------------|
| eltwise `copy` (CopyTile → PackTile) | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:109-113` | Routes each tile through unpack → DEST → pack in a compute kernel between two CBs. That is an extra L1 staging copy plus a compute kernel for an op with no arithmetic, and for bfloat8_b the unpack/pack round trip re-derives the shared exponents, which the bit-exact rule forbids. |
| local L1→L1 copy helpers | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:12-23` | Self-aimed NoC reads into a typed local buffer. This op's destination is a NoC-addressed interleaved page (a DRAM bank or another core's L1); the header states a NoC destination needs `Noc::async_write` (lines 20-21). |
| multicast channel | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp` | No operand has more than one destination. |
| tilize / untilize helpers | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `untilize_helpers.hpp` | The layout is unchanged (TILE → TILE). |
| reduce / matmul helpers | `reduce_helpers_compute.hpp`, `matmul_block_helpers.hpp` | No reduction, no contraction. |

No helper gap needs closing: `store_block` is a block operation over raw NoC writes, with its per-tile walk inside it and its one barrier outside the walk.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Wrong shard → core order | Multiple core ranges (p1: 2) and column-major walks (both other specs) are in the suite; a bounding-box walk, a sorted range list or a hard-coded row-major walk passes the production cells and fails the others | `corerange_to_cores` on the tensor's own `shard_spec.grid` with `row_wise` from the orientation, exactly as the buffer maps itself |
| Short-shard slot stride | the buffer keeps the nominal shard stride inside a short last shard (`buffer_distribution_spec.cpp:41-52`) | slot row stride = `shard_tile_cols` (mechanism cap). Not exercised by INPUTS (p2 has one tile row), so stated here for correctness on any spec. |
| Tile-row count of a rank-4 tensor | tile padding is per image on `d2`; `floor(volume/32)` or unpadded rows give the wrong page count when `d2 % 32 ≠ 0` | `tensor_tile_rows = prod(padded_shape[:-1]) / 32` |
| Stale binding in the memoized descriptor | the cached descriptor holds the first call's input `Buffer*` and output address; an unpatched hit reads a freed or different shard and writes into the previous output | rebind `cb_input_shard` and set the output-address common RT arg on every call. The acceptance test alternates two live inputs of one spec, which catches a stale binding. |
| Orientation-only change hits the same C++ program | same core ranges, CB layout and CT args, so the hash is equal (cache-order loose case, 3rd entry) | orientation is in the Python memo key, and per-core shard quantities are RT args that the hit path re-copies |
| Pages left unwritten | a fresh buffer or a dirty-destination reuse hides or exposes an unwritten page; idle grid cores or a dropped last shard leave pages untouched | coverage invariant asserted on the host; one block per shard |
| bf8b regrouping | any path through the compute engine re-derives the shared exponents | raw byte NoC write of the whole 1088 B page; no compute kernel exists |
| Transaction left in flight | another program starts right after the call (BB-03) | non-posted writes and one `noc_async_write_barrier` at the end of every kernel instance, including 0-block cores |
| Mesh: only device 0 written | the four-chip p2 case is not visible to a single-device test | a plain `ProgramDescriptor` through `generic_op` is SPMD over the full mesh shape (`generic_op.cpp:13-23`); the output is allocated on the mesh device |
| Trace | the signature has no output parameter, so the call itself allocates its output | allocation happens in the host path, which runs at capture time only. Replay allocates and builds nothing (BB-16), provided the model warms the program cache before capture. |
| Input mutated or aliased | an L1 output could be returned as the input relabelled | the writer only reads the shard; the output is a new `allocate_tensor_on_device` buffer; the acceptance test checks the address, placement and values of the input |
| Validation order | a structural error refused as `UnsupportedAxisValue` (a `NotImplementedError`) violates the `ValueError` / `RuntimeError` contract | structural checks 1-7 precede the support gate |
| Production grids need cores up to (10, 9) | a device with a smaller worker grid cannot hold the input | the tests skip there (as the golden helper does); the op itself never assumes a grid |
| Kernel text and config size (BB-04, BB-18) | the op runs inside traced models | one small writer kernel, no compute or reader kernel, a handful of per-core RT args; zones vanish in the plain build |
