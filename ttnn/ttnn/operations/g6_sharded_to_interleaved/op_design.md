# Operation Design: g6_sharded_to_interleaved

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (placement change: one local read and one NoC write per tile, no arithmetic) |
| Goal | Copy every tile of a height- or width-sharded L1 TILE tensor into a newly allocated interleaved TILE tensor (DRAM or L1, as `memory_config` names), bit for bit, as one device program. |
| Math | `out[i] = input_tensor[i]` for every logical index i; in tile terms, output page `p = a * C + b` holds the input tile at tile row `a`, tile column `b` |
| References | `docs/explorations/g6_sharded_to_interleaved/` (DOSSIER.md, math.md F1, tensix.md, risks.md), `eval/golden_tests/g6_sharded_to_interleaved/feature_spec.py`, `eval/golden_tests/g6_sharded_to_interleaved/helpers.py`, `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `ttnn/ttnn/operations/examples/master.md` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes, positional | on device, TILE (32x32 tiles), HEIGHT_SHARDED or WIDTH_SHARDED, L1, shard shape a whole number of tiles, rank 2 or 4 | none | shape, dtype, shard spec → host plan; buffer → CB binding per call |
| `memory_config` | `ttnn.MemoryConfig` | yes, positional | `memory_layout == INTERLEAVED`, `buffer_type` DRAM or L1 | none | buffer type → output `TensorAccessorArgs` (CT); output address → common RT arg |

No other parameter exists: no output dtype (placement only), no compute config (no arithmetic), no memory_format (host stride option of the reference). The shard spec (grid, shard shape, orientation) is read from `input_tensor`, never passed.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d_0, …, d_{r-1})`, r ∈ {2, 4}, every d ≥ 1 |
| Dtype | bfloat16 (Phase 0); bfloat8_b (refinement) |
| Layout | TILE, 32x32 tiles |
| Memory | L1, HEIGHT_SHARDED or WIDTH_SHARDED; shard shape `(s_h, s_w)` with `s_h % 32 == 0`, `s_w % 32 == 0`; any grid (any number of core ranges, in the order the spec holds them), either orientation; the last shard may hold fewer tiles |

### Output

| Property | Value |
|----------|-------|
| Shape | `shape(input_tensor)` (per device) |
| Dtype | `dtype(input_tensor)` |
| Layout | TILE |
| Memory | INTERLEAVED, `memory_config.buffer_type` (DRAM or L1); newly allocated by every call on every device that holds the input |

### Entry-point contract

| Item | Decision |
|------|----------|
| First line | `validate(input_tensor, memory_config)` |
| Structural checks (inside `validate()`, before the support gate) | raise `ValueError` for: not on device; layout ≠ TILE; tile shape ≠ (32, 32); not sharded; `buffer_type` ≠ L1; memory layout not HEIGHT_SHARDED / WIDTH_SHARDED (BLOCK_SHARDED included); shard shape not whole tiles; `memory_config.memory_layout` ≠ INTERLEAVED; rank ∉ {2, 4}. These must be `ValueError`, never a `SupportRefusal` (`UnsupportedAxisValue` subclasses `NotImplementedError`, which is a `RuntimeError`, so the type is what tells the two apart) |
| Support gate | build the scenario `{"shape", "dtype", "layout", "in_scheme", "shard_shape", "grid", "orientation", "buffer"}` from the arguments (grid = the shard spec's ranges in their stored order as `((x0, y0), (x1, y1))`; in_scheme `"width_sharded"` / `"height_sharded"`; orientation `"row_major"` / `"column_major"`; buffer `"dram"` / `"l1"`), run `INPUT_TAGGERS` (`tag_dtype`, `tag_layout`, `tag_in_scheme`, `tag_buffer`, each reading `inputs[0][key]`), then SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`, empty) |
| L1 bound (BB-05) | when buffer = L1: `ceil(num_pages / num_l1_banks) * out_aligned_page_bytes > L1_FREE_BYTES (1347776)` → `RuntimeError` (not a support refusal, not an axis). Every golden INPUTS shape is far below it (l1_ledger.md) |
| Output allocation | `ttnn.allocate_tensor_on_device(shape, dtype, TILE_LAYOUT, input_tensor.device(), memory_config)` on every call: new storage per call (repeat case needs a distinct buffer), host-side only, allocated on every device of a mesh |
| Dispatch | exactly one `ttnn.generic_op([input_tensor, output_tensor], descriptor)` with a `ProgramDescriptor` (SPMD: the same program on every chip of a mesh) |
| Input | never written, never freed, never relabelled; the output never aliases it |
| Package exports | `__init__.py` exports `g6_sharded_to_interleaved`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS` (the golden suite and the acceptance test import them from the package) and `describe_plan` |
| Acceptance test | `tests/ttnn/unit_tests/operations/g6_sharded_to_interleaved/test_g6_sharded_to_interleaved.py` — the production cases, the other shard spec of each, p2 → L1, dirty destination, repeat, cache order (orientation alone differs), special values, structural refusals; exact gate. A case outside SUPPORTED must raise a `SupportRefusal` and reports xfail |

## Blocking Model

### Axes

The op's tile index space is the 2-D view `[R, C]`: `R = tensor_tile_rows = prod(d_0 … d_{r-3}) * ceil(d_{r-2} / 32)` (leading dims fold into tile rows: the op treats them identically and they are contiguous; tile padding is per image) and `C = tensor_tile_cols = ceil(d_{r-1} / 32)`. Both are read from `input_tensor.padded_shape` (`R = volume(padded) / padded[-1] / 32`, `C = padded[-1] / 32`), never computed with floor.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| tile rows (`R`, leading dims folded) | independent — `out[i]` depends on `input[i]` alone, no result spans the axis | `block_tile_rows` | the whole extent this core's shard holds: `min(shard_tile_rows, R − block_origin_row)` (= `R` on every width shard; = `shard_tile_rows` except on the last height shard) | host plan, from `shard_spec.shape[0] / 32` and `R`, passed as a per-core RT arg | the core that holds the shard (shard k → k-th core of the input grid walk); on a width-sharded input every core holds the whole `R` | knob-turn: cut the shard into `num_writer_riscs` blocks, one per DM RISC of the same core. Cross-core reassignment: rejected (spec rule 4, regime table) |
| tile columns (`C`) | independent — same reason | `block_tile_cols` | `min(shard_tile_cols, C − block_origin_col)` (= `C` on every height shard; = `shard_tile_cols` except on the last width shard) | host plan, from `shard_spec.shape[1] / 32` and `C`, per-core RT arg | as above; on a height-sharded input every core holds the whole `C` | knob-turn, as above |

Derived quantities (each computed once on the host from the knobs and the shard spec, never restated):

| Quantity | Formula | Source |
|----------|---------|--------|
| `shard_tile_rows`, `shard_tile_cols` | `shard_spec.shape[0] / 32`, `shard_spec.shape[1] / 32` | input shard spec |
| `num_shard_cols`, `num_shards` | `ceil(C / shard_tile_cols)`; `ceil(R / shard_tile_rows) * num_shard_cols` | derived |
| shard k origin | `block_origin_row = (k div num_shard_cols) * shard_tile_rows`, `block_origin_col = (k mod num_shard_cols) * shard_tile_cols` | derived (one formula for both strategies: height has `num_shard_cols = 1`, width has `ceil(R / shard_tile_rows) = 1`) |
| core of shard k | `ttnn.corerange_to_cores(shard_spec.grid, None, row_wise = (orientation == ROW_MAJOR))[k]` | the input's own grid object, ranges in stored order — the exact walk the sharded buffer uses (`tt_metal/impl/buffers/buffer.cpp:439-441`, `buffer_distribution_spec.cpp:172-173`) |
| `num_blocks_this_core` | 1 for the first `num_shards` cores of the walk, 0 for the rest | derived; a height/width shard spec places at most one shard per core (`buffer.cpp:437-447`); host asserts `num_shards <= len(cores)` |
| local L1 offset of tile `(ρ, γ)` of the block | `(ρ * shard_tile_cols + γ) * in_aligned_page_bytes` — the **nominal** shard width, also on a short last shard (`buffer.cpp:462-466`: `core_page_idx = shard_page_x * shape_in_pages[1] + shard_page_y`) | CT: `shard_tile_cols`, page bytes |
| output page of tile `(ρ, γ)` | `(block_origin_row + ρ) * C + (block_origin_col + γ)` | CT: `C`; RT: origins |
| `num_writer_riscs` | 1 | host constant (Phase 0) |

The kernel is scheme-agnostic: the strategy only chooses `num_shard_cols` on the host. Height sharding, bfloat8_b and the L1 destination therefore need no kernel change.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_shard` | none — pinned to one shard by the sharded buffer it is backed on | 1 shard (`shard_tile_rows * shard_tile_cols` pages) | nothing: no producer kernel exists to overlap with; the shard is resident before the program starts |

No streamed CB exists, so the op has no buffer-depth knob. Write overlap comes from issuing all writes of the block back to back before the single barrier (the NoC keeps them in flight), not from a CB.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| interleaved round-robin placement (`tensor_accessor.h:560`: pages round-robin across banks) | transfer length of one write: 1 page (consecutive output pages lie in different banks) | one `noc_async_write_page` per tile | a multi-page write spills tile `p+1` into bank `p`'s next slot → wrong pages, other pages overwritten |
| shard geometry (short last shard) | `block_tile_rows`, `block_tile_cols` ≤ the tiles the shard actually holds | `min(nominal, R − origin_row)`, `min(nominal, C − origin_col)` on the host | the last core writes padding pages to page ids ≥ `R*C` or into a neighbouring row → out-of-bounds or wrong pages |
| in-shard stride | the local row stride is the nominal `shard_tile_cols`, not `block_tile_cols` | stride CT arg = `shard_tile_cols` | on a short last width shard with `R > 1`, every row after the first reads the wrong tile |
| whole tiles (bfloat8_b shared exponents, `tt_backend_api_types.hpp:129`: 1024 + 64 B) | transfer size | the full page bytes from the tensor (`buffer_page_size()`): 2048 bfloat16, 1088 bfloat8_b; never a literal, never keyed on dtype | a 1024-B move drops the exponent section; a repack through compute recomputes exponents (rule 7) |
| NoC alignment (Blackhole: DRAM 64 B, L1 16 B, `core_config.h:42-43`) | page stride | host asserts `in_aligned_page_bytes == out_aligned_page_bytes == page_bytes` (true for 2048 and 1088 on both destinations) | misaligned writes or a stride mismatch between source and destination |
| NoC burst (`noc_parameters.h:290`, `NOC_MAX_BURST_SIZE`) | bytes per write | one page ≤ 2048 B, far below the burst | — |
| shard placement | shards per core ≤ 1 for a height/width spec (`buffer.cpp:437-447`) | `num_blocks_this_core ∈ {0, 1}`; host asserts `num_shards <= len(cores)` | — |

### Regimes

Named boundary: **each tile is read where its shard lies (local L1, no NoC) and crosses the NoC exactly once, as the write to its output page** (DRAM banks or L1 banks). That is the minimum: P page writes of `page_bytes`, zero reads over the NoC, zero extra DRAM crossings.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **owner writes whole shard** — every core holding a shard writes its own resident tiles to their output pages | **built** | every input `validate()` accepts (any height/width spec of whole tiles, any grid, either orientation, short last shard included) | `(block_tile_rows, block_tile_cols)` = the core's whole shard; `num_blocks_this_core ∈ {0, 1}` | minimum: input 0 crossings (read in place from the core's own L1 through `cb_input_shard`), output 1 crossing per tile | the per-block fixed cost is one RT-arg decode, one `TensorAccessor` set-up and one write barrier; whole shard = paid once per core. Per-tile cost (one NoC write) is constant in the extent, so the extent is invisible to the split ranking and the barrier count alone sets it — the coarsest block (the whole shard) is the only sensible value |
| gather and write — fewer cores (or one) read remote shards and write pages | rejected — superseded by owner writes | — | — | +1 NoC crossing per tile (P × page_bytes read across the NoC before the write): 2× the bytes; forbidden by spec rule 4 | — |
| rebalance across the full grid — `split_work_to_cores` over all worker cores for occupancy (e.g. p2's 160 tiles on 110 cores) | rejected — superseded by owner writes | — | — | every tile not on its owner crosses twice; forbidden by rule 4. The occupancy it buys is not on offer: the producer's shard spec already fixes where each tile is | — |
| stage through DRAM / an extra full-size buffer | rejected — superseded by owner writes | — | — | +2 DRAM crossings per tile; forbidden by the contract | — |
| owner writes in multi-page runs | rejected — unreachable | — | — | consecutive local tiles map to consecutive output pages, which lie in different banks; pages contiguous in one bank (`p`, `p + num_banks`) are `num_banks` tiles apart in the shard, so no single write covers two pages for any bank count > 1. One write per page is already the minimum transaction count | — |
| owner writes logical rows only — skip tile padding (p2: 1 of 32 rows per tile) | rejected — superseded by owner writes whole shard | — | — | 32× fewer DRAM bytes for p2, but 2 writes of 32 B per tile (face 0 row 0, face 1 row 0) instead of 1: doubles the transaction count of a transaction-bound op, DRAM writes below the 64-B DRAM alignment, a second layout for bfloat8_b exponents, and the contract says the op moves whole tiles | — |

No dependent axis exists, so there is no "split the dependent axis and combine" regime and no residency/streaming dual path. Single built regime → no selection predicate; the page map is pinned by the "other shard spec" golden cases (column-major, short last shards, two core ranges).

### Traffic ranking

| Rank | Candidate split | DRAM crossings (per tensor) | Cross-core (NoC) traffic | Verdict |
|------|-----------------|-----------------------------|--------------------------|---------|
| 1 | owner writes (split = the input's own shards) | input 0; output 1 when DRAM (p2: 327680 B), 0 when L1 | output pages only: P × page_bytes, once (p1: 1114112 B to L1 banks) | chosen — the named-boundary minimum |
| 1 (same bytes) | owner writes, each shard further cut across the core's two DM RISCs | as rank 1 | as rank 1, issued from two RISCs / both NoCs | same bytes; an extent decision, not a split — perf lamp L1 |
| 2 | gather onto fewer cores | as rank 1 | + P × page_bytes reads across the NoC | rejected (rule 4) |
| 2 | rebalance over the full grid | as rank 1 | + up to P × page_bytes moved twice | rejected (rule 4) |
| 3 | DRAM staging | +2 × tensor bytes | as rank 1 | rejected (contract) |
| — | split a dependent axis + combine | — | — | no dependent axis exists |

Operand-reuse check: one input operand; each tile feeds exactly one output page; nothing is reuse-shared under the owner split, so no broadcast regime exists.

Stall-shadow check: the only wait is the final `noc_async_write_barrier` (acks of the core's own writes). Every write of the core is already issued when it starts; nothing independent remains to schedule into it.

### Block schedule

```cpp
// per core (writer kernel); num_blocks_this_core ∈ {0, 1}
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    store_block(block_idx);   // every tile of the resident shard rectangle → its output page
}
drain_writes();               // wait for the acks of every write this core issued
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `store_block` | `block_tile_rows × block_tile_cols` tiles (the whole shard) | the shard in `cb_input_shard` (resident before launch, never modified) | output `TensorAccessor` built once per kernel; one non-posted NoC write per tile (irreducible: one page per bank-contiguous run); no CB handshake (no producer kernel); no LLK init (no compute); no barrier inside the block |
| `drain_writes` | — | — | one `noc_async_write_barrier` per kernel, after the last write (BB-03: nothing in flight when the kernel returns) |

There is no `load_block`: the data is already resident in the core's L1. There is no compute phase and no compute kernel.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 — one writer RISC per core (`num_writer_riscs = 1`) | p1 (PERF FOCUS) issues 10 page writes from one RISC; with the NoC far from saturated the RISC's per-transaction issue cost is the likely bound (catalog `double_buffer/README.md` L92-99: ~115 ns per tile transaction regardless of how many are in flight; `split_reader/README.md` L12-14, L71-72: a second RISC helps only when a DM RISC is the bottleneck, 1.2× at 2 KiB) | `num_writer_riscs = 2`: cut each shard into two blocks along its longer axis (p1: 5 + 5 tile rows; p2: 3 + 2 tile columns), the same writer source on BRISC/NoC1 and NCRISC/NoC0, each with its own barrier. Costs a second kernel binary per core range (BB-17/18): judge on host time of trace replays, not profiled kernel time alone (lesson RB-053) |
| L2 — ascending write order from the shard's first tile | when a core's page count is a multiple of the bank count, every core starts on the same bank (catalog `bank_stagger/README.md` L76-81, ≤ ~1 µs measured). Production: p2 starts at page 5k → DRAM banks 5k mod 8 already differ; p1 starts at page 10k | rotate each core's write order by its index in the walk (start at tile `core_idx mod block_tiles`), same pages, same count |
| L3 — `TensorAccessor::get_noc_addr(page)` per tile | for an L1 destination the bank count is not a power of two, so each page's bank/offset needs a software division on the RISC | walk bank index and bank offset incrementally across the block's pages (`bank += 1`, wrap → `offset += aligned_page`), resolving the NoC address from the bank table |

Overlap lamp: does not apply — no compute, no CB pipeline; the block's writes overlap in the NoC by construction. Grid-synchronization lamp: does not apply as a measurable alternative — no collective exists, and "fewer participating cores" is excluded by rule 4; p2's per-call fixed cost (32 cores × 5 tiles) is addressed by BB-17 (kernel group = the input's own core ranges, 1 for p2) and the host cache (BB-07).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| input shard in L1 of its owner core | TILE pages as stored (Float16_b 2048 B / Bfp8_b 1088 B incl. exponents) | `cb_input_shard`, a CB backed on the sharded buffer (`ttnn.cb_descriptor_from_sharded_tensor`, `ttnn/core/tensor/tensor_utils.cpp:48`), read through `get_read_ptr` (`dataflow_api.h:346`) | zero-copy, no NoC read; never re-read through a `TensorAccessor` |
| local L1 → output page | same bytes, whole page | `noc_async_write_page(page, out_accessor, src_l1)` (`dataflow_api.h:1418`), non-posted, writer RISC (BRISC / NoC1, the default writer config) | DRAM banks (buffer = dram) or L1 banks of the worker grid (buffer = l1); for an L1 page whose bank is on the writing core this is a NoC write to self, still the one crossing |
| completion | — | `noc_async_write_barrier` (`dataflow_api.h:1951`) | before the kernel returns |
| mesh | per-device tensors | `ProgramDescriptor` → SPMD on every chip; output allocated on the mesh device | rule 5: every page written on every chip |

TARGET placement axes as regimes (all **knob-turns** — the same `owner writes whole shard` regime, no scheme change):

| TARGET value | Cut / change | Cost class | What changes |
|--------------|--------------|------------|--------------|
| `in_scheme = width_sharded` (Phase 0) | shards cut tile columns; every core holds all `R` rows | built | — |
| `in_scheme = height_sharded` | shards cut tile rows; every core holds all `C` columns | knob-turn | host plan only (`num_shard_cols = 1`); the kernel is scheme-agnostic |
| `dtype = bfloat8_b` | page bytes 1088 | knob-turn | none: page bytes come from the tensor; rule 7 is met by moving whole pages |
| `buffer = l1` | output banks are L1 | knob-turn | output `TensorAccessorArgs` encode the buffer type; enable the BB-05 L1 guard |

Structural impossibilities: none beyond feature_spec.py's empty INVALID.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block = the whole shard a core holds |
| Grid | the kernel's core ranges = `input_tensor.memory_config().shard_spec.grid`, verbatim (p2: 1 range (0,0)-(7,3); p1: 2 ranges (0,0)-(10,8), (0,9)-(3,9)) — the fewest ranges reachable without moving data (BB-17); cores of the grid past `num_shards` run the kernel with `num_blocks_this_core = 0` |
| Per-core work | `block_tile_rows × block_tile_cols` page writes; production: p2 5 on each of 32 cores; p1 10 on 102 cores, 4 on core (3,9); p2 other spec: 3 on 53 cores, 1 on the last; p1 other spec: 11 on 93 cores, 1 on the last |
| Remainder | per shard via `min` on the host (Axes); tile geometry from `padded_shape` (`ceil` per image) |
| Core count | set by the blocking decision: owner writes is the only split at the named-boundary minimum, so the core set is the input's shard owners — chosen by the traffic ranking, not limited by the tensor shape |
| Regime selection | single built regime; no predicate |

Host plan and cache (BB-07, BB-08, BB-21):

| Item | Decision |
|------|----------|
| Cache key | `(device identity, tuple(shape), tuple(padded_shape), dtype, memory_layout, shard grid ranges in stored order, shard shape, orientation, output buffer type)` — orientation must be in it: `generic_op`'s own hash excludes runtime-arg values (`generic_op_device_operation.cpp:78-146`), and orientation changes only per-core RT args |
| Built once per key | core list, per-core RT args (origins, extents, block count), CT args, kernel core ranges, the `ProgramDescriptor` |
| Per call | (1) allocate the output; (2) bind `cb_input_shard` to this call's input (`cb_descriptor_from_sharded_tensor` on the current tensor — never reuse a `Buffer*` from an earlier call); (3) set the output base address (a common RT arg: the same on every core); (4) dispatch. `generic_op` re-applies every RT arg, common RT arg and buffer-backed CB address from the passed descriptor on a cache hit (`tt_metal/impl/program/program_descriptors.cpp:188-240`). nanobind vector properties (`desc.cbs`, `desc.kernels`) return copies: reassign the whole list |
| Plan report | `describe_plan(input_tensor, memory_config)` returns the regime name and, per core, its role (`writer` / `idle`) and block `(origin_row, origin_col, rows, cols)` |
| Zones | writer kernel: `DeviceZoneScopedN("S2I-STORE-BLOCK")` around the block loop, `DeviceZoneScopedN("S2I-DRAIN-WRITES")` around the barrier; compiled out of the plain build |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_input_shard` | 0 | input aligned page bytes (2048 / 1088) | `shard_tile_rows * shard_tile_cols` (= the sharded buffer's `aligned_size_per_bank`) | backed on the input's sharded buffer (zero-copy, 0 B of op L1); live set spans both block axes (the whole shard is resident at once) | input dtype (Float16_b / Bfp8_b) | none in this program — the shard is resident before launch (written by the producing program) | writer | whole program |

No other CB. Sync: producer push count 0 = consumer wait count 0 — the writer reads `get_read_ptr(cb_input_shard)` without `cb_wait_front` (no kernel ever pushes; a `cb_wait_front` here hangs). Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `store_block` | `block_tile_rows × block_tile_cols` | no (raw NoC write; no dataflow helper writes resident pages to interleaved banks) | `cb_input_shard`, whole shard, resident, read-only | none — interleaved output pages via `TensorAccessor` | `cb_input_shard` unchanged |
| 2 | `drain_writes` | — | no (raw barrier) | — | — | all writes acked |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `store_block` (source address) | raw_api | `get_read_ptr` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:346` | `cb_input_shard` | `cb_input_shard` | — | none |
| `store_block` (destination) | raw_api | `TensorAccessor` (interleaved specialization), `TensorAccessorArgs` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:493-560` | CT: output accessor args; base: output address (common RT) | — | output pages | none |
| `store_block` (write) | raw_api | `noc_async_write_page` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1418` | page id, accessor, `src_l1 = shard_base + local offset`; size = page bytes; non-posted | `cb_input_shard` | output pages | trip counts `block_tile_rows`, `block_tile_cols` (RT) |
| `drain_writes` | raw_api | `noc_async_write_barrier` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1951` | default NoC | — | — | none |
| host: shard CB | raw_api (host) | `ttnn.cb_descriptor_from_sharded_tensor` | `ttnn/core/tensor/tensor_utils.cpp:48` (binding `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:518`) | `cb_index`, `input_tensor` (defaults: grid = shard grid, size = one shard) | — | — | none |
| host: core walk | raw_api (host) | `ttnn.corerange_to_cores` | `tt_metal/common/core_coord.cpp:604` (binding `ttnn/cpp/ttnn-nanobind/tensor.cpp:660`) | input's own `shard_spec.grid`, `None`, `row_wise = orientation == ROW_MAJOR` | — | — | none |

Helpers considered and rejected (every `raw_api` row above):

| Candidate | File:Line of the mismatch | Why it cannot express the operation |
|-----------|---------------------------|-------------------------------------|
| `local_copy_helpers_dataflow.hpp` (`local_addr`, `read_with_state`) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:12-20` | a self-aimed **read** into a local typed buffer; this op's destination is a remote bank's NoC address, i.e. a write, which that header states it cannot target |
| `McastArgs` sender/receiver | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp:16-19` | one sender to a receiver group; each page here has exactly one destination bank — no fan-out, no reuse |
| compute helpers (`tilize`, `untilize`, `eltwise_*`, `reduce`, `matmul_block`) | — | no compute phase exists; a pass of a bfloat8_b tile through unpack/pack recomputes shared exponents, which spec rule 7 forbids |

Gap record: no `kernel_lib` dataflow helper writes a resident L1 rectangle of pages to an interleaved `TensorAccessor`. The raw loop lives inside `store_block`; it does not retain a unit-at-a-time pipeline (one write per page is the transaction minimum for interleaved output).

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| wrong page map (R1, R12) | the op is nothing but the map; orientation, multiple core ranges, short last shards change it | core walk from the input's own grid with `corerange_to_cores` and the orientation flag; one origin formula; nominal in-shard stride; golden "other shard spec" cases pin it |
| a short last width shard with `R > 1` read with the actual width as stride | the buffer keeps nominal strides on the last shard (`buffer.cpp:462-466`) | stride = `shard_tile_cols` (CT), extent = `block_tile_cols` (RT) — two different parameters |
| `cb_wait_front` on `cb_input_shard` | no kernel pushes to it | the writer reads `get_read_ptr` directly; no CB sync calls on it |
| stale buffer pointer or address on a cache hit (R8) | a cached descriptor holding an earlier input's `Buffer*` points at freed memory; orientation-only changes hit `generic_op`'s cache | per call rebind the input CB and the output address; Python key includes orientation (Work Distribution) |
| writes in flight at return (R3, BB-03) | the next program starts right after | non-posted writes and one `noc_async_write_barrier` at the end; posted writes are not used, because a flush only proves departure, not completion |
| bfloat8_b moved as 1024 B or through compute (R5) | the page is 1088 B | transfer size from `buffer_page_size()`; no compute kernel |
| output allocated in a captured trace (BB-16, D7) | the signature has no output argument, and rule 2 requires a new buffer per call | the entry point allocates with a host-side call (no device program); a replay runs only the captured program and allocates nothing. The program is compiled by the first (eager) call, never at replay. The contract's "allocated before capture" can only be met by the model's flow, since the signature takes no output — for the lead |
| p2 dominated by per-call fixed cost (R6) | 5 tiles per core | 1 core range (the input's), one small kernel, no compute kernel, host descriptor cache (BB-07, BB-17, BB-18) |
| mesh (R9) | FLUX runs on 4 chips | SPMD `ProgramDescriptor`; output allocated on the mesh device |
