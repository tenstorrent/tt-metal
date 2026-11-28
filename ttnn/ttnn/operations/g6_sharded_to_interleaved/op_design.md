# Operation Design: g6_sharded_to_interleaved

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no arithmetic) |
| Goal | Copy a height- or width-sharded L1 TILE tensor, bit for bit, into a newly allocated interleaved TILE tensor in the buffer type `memory_config` names (DRAM or L1). One device program per call. Each tile crosses the NoC exactly once: the owning core writes it from its own shard straight to its output page. |
| Math | `out[i] = input_tensor[i]` for every logical index `i`; page `p = a*C + b` of the output holds the input tile at tile row `a`, tile column `b` (`C` = tensor tile columns) |
| References | `eval/golden_tests/g6_sharded_to_interleaved/feature_spec.py` (TARGET / INPUTS / LOOSE_CASES), `eval/golden_tests/g6_sharded_to_interleaved/helpers.py` (exact gate), `eval/op_template.py` (registry model), `tt_metal/impl/buffers/buffer.cpp:425-476` (shard slot layout: `core_page_idx = shard_row * shard_w_pages + shard_col`, padding slots for a short last shard), `tt_metal/common/core_coord.cpp:604-626` (`corerange_to_cores`: ranges in set order, each walked row-wise or column-wise), `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:517-554` (`cb_descriptor_from_sharded_tensor`), `ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:78-120,307-327` (program hash: CT args, core ranges, CB sizes, `custom_program_hash`; RT-arg values are NOT hashed), perf catalog `ttnn/ttnn/operations/examples/master.md` entries `tile_reorder`, `split_reader`, `bank_stagger`, `noc_placement`, `double_buffer` (see Perf lamps) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes (positional) | on device, TILE, HEIGHT_SHARDED or WIDTH_SHARDED in L1, shard shape a whole number of 32x32 tiles, rank 2 or 4 | — | shard geometry → CT (`shard_col_tiles`, `tensor_col_tiles`, page sizes) + RT (per-core rectangle); buffer → sharded CB |
| `memory_config` | `ttnn.MemoryConfig` | yes (positional) | `memory_layout == INTERLEAVED`, buffer type DRAM or L1 | — | selects output buffer → `TensorAccessorArgs(output)` CT + output address RT |

No dtype, layout, compute-config, or memory-format parameter exists. Shard grid, shard shape and orientation are read from `input_tensor.memory_config().shard_spec`.

### Registry model (names match `feature_spec.py` TARGET)

| Item | Value |
|------|-------|
| `INPUT_TAGGERS` | `tag_dtype` → `inputs[0]["dtype"]`; `tag_layout` → `inputs[0]["layout"]`; `tag_in_scheme` → `inputs[0]["in_scheme"]` (`"width_sharded"` / `"height_sharded"`); `tag_buffer` → `inputs[0]["buffer"]` (`"dram"` / `"l1"`). No other tagger. |
| `validate()` | Builds `(scenario,)` from the arguments: `shape`, `dtype`, `layout` off the tensor; `in_scheme` from `memory_layout`; `shard_shape`, `grid` (ranges in `shard_spec.grid.ranges()` order), `orientation` off the shard spec; `buffer` from `memory_config.buffer_type`. Then (1) structural ValueError checks (below), (2) SUPPORTED per axis → `UnsupportedAxisValue`, (3) EXCLUSIONS → `ExcludedCell`. First line of the public entry point. |
| Phase 0 `SUPPORTED` (implementer writes it) | `dtype: [bfloat16]`, `layout: [TILE_LAYOUT]`, `in_scheme: ["width_sharded"]`, `buffer: ["dram"]` |
| `EXCLUSIONS` | none, in every phase |
| INVALID | none (TARGET is TILE only) — not declared or checked by the op |
| Structural impossibilities missed by `/golden-tests` | none |

### Structural validation (ValueError / RuntimeError, before the support check)

| Condition | Error |
|-----------|-------|
| `not ttnn.is_tensor_storage_on_device(input_tensor)` | ValueError |
| `input_tensor.layout != TILE_LAYOUT` | ValueError |
| `memory_layout` not HEIGHT_SHARDED / WIDTH_SHARDED (interleaved, block, ND) | ValueError |
| input `buffer_type != L1` | ValueError |
| `shard_spec.shape[0] % 32 != 0 or shard_spec.shape[1] % 32 != 0` | ValueError |
| rank not in {2, 4} | ValueError |
| `memory_config.memory_layout != INTERLEAVED` | ValueError |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d0, d1)` or `(d0, d1, d2, d3)`, every size ≥ 1 |
| Dtype | bfloat16 (Phase 0), bfloat8_b (refinement) |
| Layout | TILE |
| Memory | L1, HEIGHT_SHARDED or WIDTH_SHARDED, shard shape whole tiles, any grid / range order / orientation, short last shard allowed |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` (per device on a mesh) |
| Dtype | `input_tensor.dtype` |
| Layout | TILE |
| Memory | INTERLEAVED in `memory_config.buffer_type`, newly allocated by `ttnn.allocate_tensor_on_device(input.shape, input.dtype, TILE_LAYOUT, input.device(), memory_config)` on the input's (mesh) device |

## Blocking Model

The op is a pure page permutation. The 2-D page grid is `tensor_row_tiles × tensor_col_tiles`:

| Symbol | Definition (alignment-aware, per image) |
|--------|----------------------------------------|
| `tensor_col_tiles` (`C`) | `padded_shape[-1] / 32` (= `ceil(shape[-1]/32)`) |
| `tensor_row_tiles` | `prod(padded_shape[:-1]) / 32` (= `prod(shape[:-2]) * ceil(shape[-2]/32)`; NOT `floor(prod(shape[:-1])/32)`) |
| `shard_row_tiles` | `shard_spec.shape[0] / 32` |
| `shard_col_tiles` | `shard_spec.shape[1] / 32` — also the **slot row stride** inside a shard |
| `num_shards` | height: `ceil(tensor_row_tiles / shard_row_tiles)`; width: `ceil(tensor_col_tiles / shard_col_tiles)` |
| `shard_cores` | `ttnn.corerange_to_cores(shard_spec.grid, row_wise=(orientation == ROW_MAJOR))[:num_shards]` — the allocator's own order (`buffer.cpp:440`), so shard `k` ↔ core `k` matches the input buffer by construction |

Production: (1, 5120) bf16 width → `C=160`, rows 1, shard 1×5 tiles, 32 shards. (1,128,256,32) bf8b height → `C=1`, rows 1024, shard 10×1, 103 shards, last 4.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `tile_row` (all dims but last, folded per image) | independent — every output page depends on exactly one input tile, no combine | `block_rows` | the core's whole shard rows: height `min(shard_row_tiles, tensor_row_tiles − k*shard_row_tiles)`; width `tensor_row_tiles` | host, computed once per core from the shard spec in the program-descriptor builder → RT arg `block_rows` | **fixed by placement**: owner-computes — the core holding shard `k` writes it (height scheme cuts this axis across `shard_cores`) | knob-turn (lane split, see Perf lamps) |
| `tile_col` (last dim) | independent — same reason | `block_cols` | height: `tensor_col_tiles`; width: `min(shard_col_tiles, tensor_col_tiles − k*shard_col_tiles)` | host → RT arg `block_cols` | fixed by placement (width scheme cuts this axis across `shard_cores`) | knob-turn |
| shard / core axis (`k`) | independent — shards never interact | `num_blocks_this_core` | 1 (one block = the whole resident shard, Rule 2 sharded default) | host constant `BLOCKS_PER_SHARD = 1` → loop bound in kernel RT arg | one shard per core, as the input places it; cores of the grid with `k ≥ num_shards` get `block_rows = block_cols = 0` | none needed (cutting a shard into more blocks buys nothing: no staging buffer, no L1 to bound it) |

Why not other cores: any core that does not own a tile would have to fetch it across the NoC before writing it — a second crossing the spec forbids (Rules: "MUST NOT fetch a tile from another core's shard"). So the participating-core set is the shard grid, decided by the one-crossing traffic ranking, not by the tensor shape; every shard core participates (32 cores for p2, 103 for p1). This satisfies "the number of cores used is the blocking decision's choice".

Both schemes are the SAME block operation: a shard is a rectangle at tile origin `(row0, col0)` with extents `(block_rows, block_cols)`; height sets `row0 = k*shard_row_tiles, col0 = 0`, width sets `row0 = 0, col0 = k*shard_col_tiles`. `in_scheme` therefore changes only host-computed RT args — adding `"height_sharded"` is a SUPPORTED-list edit, not new code. Likewise `buffer` changes only the output `TensorAccessorArgs`, and `dtype` only the page sizes.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_shard` | none — globally allocated on the input's shard buffer (zero-copy), capacity = the shard | whole shard | nothing to pipeline: every tile is already resident before the program starts |
| (in-flight NoC writes) | `writes_per_barrier` | whole block (all `block_rows*block_cols` writes issued, then ONE `noc_async_write_barrier`) | maximum transactions in flight; the only wait is the final barrier required by BB-03 |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Interleaved page placement (consecutive pages → different banks) | one NoC write moves at most one page | each write = exactly one tile page (`output.buffer_page_size()`: 2048 bf16 / 1088 bf8b) | a multi-page write would put page `p+1` into bank of page `p` → wrong output map |
| Shard slot stride | source address of slot `s` | `src = get_read_ptr(cb_input_shard) + s * input.buffer_aligned_page_size()`, `s = r*shard_col_tiles + c` (stride is the FULL shard width even for a short last width shard — padding slots, `buffer.cpp:461-469`) | using `block_cols` as the stride corrupts every short last width shard (e.g. LOOSE `_P2_OTHER_SPEC`, last shard of 1) |
| Grid beyond `num_shards` | cores with `k ≥ num_shards` | RT args `block_rows = block_cols = 0` | indexing past the tensor → writes outside the output buffer |
| uint32 page / slot index | `tensor_row_tiles * C` | host asserts `< 2^32` (structurally always true in L1) | silent wrap |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **owner-writes** (each shard core NoC-writes its own resident tiles to their interleaved pages) | built | every input (single row covers both schemes, both dtypes, both destinations) | `block_rows × block_cols` = the whole shard on core `k` | **minimum**: boundary = NoC. Each tile is read from local L1 (no NoC) and crosses the NoC once, as the write to its page; DRAM crossings: 0 reads, 1 write per tile for a DRAM destination, 0 for L1. | One block per core per call: one kernel launch, one RT-arg fetch, one CB-address fetch, one write barrier — paid once per core per call; per-tile cost is only the NoC write issue. Domain: every height/width sharded TILE input; no other regime applies because any other assignment adds a crossing (rows below). |
| owner-writes, 2 lanes (shard rectangle split between BRISC/NoC1 and NCRISC/NoC0 kernels) | deferred — perf refinement; same regime, two instances of the same kernel with disjoint sub-rectangles. Reachable because RT args already carry `(src_first_slot, first_page, block_rows, block_cols)` per kernel instance. Not Phase 0 because it is only a win if a single RISC is issue-bound (catalog `split_reader`), which is unmeasured at 5–10 tiles/core. | — | half-shard sub-rectangle per lane | minimum (unchanged) | halves per-RISC issue time |
| gather-then-write (move tiles to fewer / other cores, write from there) | rejected — superseded by owner-writes; adds a full NoC crossing per tile and violates the one-crossing rule | — | — | 2 NoC crossings per tile | — |
| stage through an L1 CB / DRAM temp | rejected — dead end: an extra full-size copy, no structure survives | — | — | ≥2 crossings | — |
| reader-side (output-owning core pulls from shard core) | rejected — interleaved pages have no owning Tensix (DRAM banks; L1 banks are passive), and a pull by a third core is a fetch-before-write | — | — | 2 crossings | — |

### Traffic ranking

| Candidate split | NoC crossings per tile | DRAM crossings (DRAM dest) | Fan-out | Rank |
|-----------------|------------------------|----------------------------|---------|------|
| owner-writes (split = the shard split already in L1) | 1 | 1 write | each core → `block_rows*block_cols` distinct banks | **1 (chosen; also the cheapest)** |
| owner-writes, 2 lanes | 1 | 1 write | same, from two NoCs | 1 (same bytes; issue-parallelism lamp) |
| gather to a subset, then write | 2 | 1 write | n:1 then 1:many | 3 |
| re-split across idle cores of the grid | 2 | 1 write | — | 3 |

No dependent axis exists, so "split the dependent axis + combine" has no candidate. Operand-reuse check: the single operand varies along every split axis — nothing is reuse-shared, no broadcast row. Stall-shadow check: the only wait is the final write barrier; all independent work (every write of the block) is issued before it, so nothing remains to schedule into it.

### Block schedule

```cpp
// per participating core, one writer kernel (Phase 0: one lane)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {  // num_blocks_this_core = 1 (0 on idle grid cores)
    store_block(block_idx);   // issue block_rows*block_cols page writes: slot (r*shard_col_tiles + c) -> page first_page + r*C + c
}
drain_writes();               // one noc_async_write_barrier
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_block` | — (none: the shard IS the block, already in L1 before launch; zero-copy CB) | whole shard | 0 |
| `store_block` | `block_rows × block_cols` tiles | whole shard (read-only) | TensorAccessor construction once per kernel; `get_read_ptr(cb_input_shard)` once; NoC write issue once per tile (one page per transaction, mechanism cap) — no per-tile barrier, no per-tile CB handshake |
| `drain_writes` | — | — | one `noc_async_write_barrier` per kernel invocation (BB-03: all writes acked before return) |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Write lanes (catalog `split_reader`) | One RISC issues all 5 (p2) or 10 (p1) writes; at this size the per-call launch dominates, but issue time may be on the critical path for p1 (1088 B pages, 103 cores, L1 destination) | 2 lanes: BRISC/NoC1 + NCRISC/NoC0 kernel instances, each a half sub-rectangle (`writer_lanes` knob, Phase 0 = 1) |
| Issue order / bank stagger (catalog `bank_stagger`) | All cores start their walk at local slot 0 at the same instant; p2 core `k` starts at DRAM page `5k`, p1 core `k` at L1 page `10k` — naturally staggered, but neighbour cores may still collide on banks | rotate the start index within the block by `k % (block_rows*block_cols)` (CT switch) |
| NoC selection (catalog `noc_placement`) | default writer = BRISC on NoC1; for an L1 interleaved destination spread over the whole grid NoC0 routing may disperse better | writer on NoC0 |
| Grid-synchronization lamp | n/a — no collective, barrier or multicast exists | — |
| Overlap lamp | n/a — no compute; the whole block is issued asynchronously before the one barrier | — |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input shard in L1 of core `k` | tiles of the input dtype, slots row-major with stride `shard_col_tiles` | `cb_input_shard` = `ttnn.cb_descriptor_from_sharded_tensor(CB_INPUT_SHARD, input_tensor)` (zero-copy, core ranges = shard grid) | never re-read through a TensorAccessor; read pointer only |
| Shard L1 → output page | same bytes, whole tile | `noc_async_write_page(page, out_accessor, src_l1_addr)` / `noc_async_write(src, get_noc_addr(page, out_accessor), page_size)`, `out_accessor = TensorAccessor(TensorAccessorArgs<...>(), out_addr, out_page_size)` | DRAM or L1 banks decided by the output buffer type through `TensorAccessorArgs(output_tensor)` — the kernel source is identical for both |
| Completion | — | `noc_async_write_barrier()` | before return (BB-03) |

Placement axis (`in_scheme`) as regime view: both `width_sharded` and `height_sharded` are physical realizations of the owner-writes split along an independent axis → **knob-turn** unlocks (RT-arg change only). `buffer: l1` and `dtype: bfloat8_b` are knob-turns too (accessor args / page size). bfloat8_b: the tile is moved as raw bytes (1088 B incl. shared exponents) — never through unpack/pack.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block = the whole shard on one core |
| Grid | the input's shard grid, exactly `shard_spec.grid` (its own few rectangles — BB-17), kernel core ranges = `shard_spec.grid` |
| Per-core work | core `shard_cores[k]`, `k < num_shards`: RT args `(out_addr, first_page = row0*C + col0, block_rows, block_cols, src_first_slot = 0)`; cores of the grid not in `shard_cores[:num_shards]`: `(out_addr, 0, 0, 0, 0)` |
| Remainder | the short last shard: height → `block_rows = tensor_row_tiles − (num_shards−1)*shard_row_tiles`; width → `block_cols = tensor_col_tiles − (num_shards−1)*shard_col_tiles`; slot stride stays `shard_col_tiles` |
| CT args (writer) | `cb_input_shard` index, `shard_col_tiles` (slot row stride), `tensor_col_tiles` (page row stride), `out_page_size`, `in_aligned_page_size`, then `TensorAccessorArgs(output_tensor)` |
| `split_work_to_cores` | not used: placement fixes the assignment |
| Mesh | one `ttnn.ProgramDescriptor` (SPMD) → the same program on every chip; output allocated on the mesh device with the per-device shape; shard buffer address is identical on every chip |
| Program cache (BB-07) | host caches the built descriptor in a dict keyed by `(shape, dtype, in_scheme, shard_shape, grid ranges in order, orientation, out buffer_type)`; per call only the output address RT arg (and the CB's input buffer) is patched. Set `custom_program_hash` from that key: generic_op does not hash RT-arg values (`generic_op_device_operation.cpp:78-100`), and orientation alone changes only RT args (LOOSE `cache_order` case) |
| Regime selection | single regime — no predicate. `last_plan()` reports `"owner_writes"` plus `num_shards` and lanes (BB-08) |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_shard` | 0 | `input.buffer_aligned_page_size()` | `shard_row_tiles * shard_col_tiles` (set by `cb_descriptor_from_sharded_tensor`, globally allocated on the input buffer — 0 new L1) | spans `tile_row` and `tile_col` of the whole shard (the block); host-placed before launch | input dtype | host placement (pre-resident shard; no kernel pushes) | writer | whole program |

No other CB: the write goes from the shard straight to the output page (Rule 3 "pack into destination" — here no intermediate exists at all). No compute kernel, no reader kernel.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `store_block` | `block_rows × block_cols` | no (dataflow NoC API) | `cb_input_shard`, whole shard, resident, read via `get_read_ptr` (no wait/pop) | output tensor pages via TensorAccessor (no CB) | unchanged (input must stay intact) |
| 2 | `drain_writes` | — | no | — | — | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `store_block` | raw_api | `noc_async_write_page` / `noc_async_write` + `TensorAccessor` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1418`, `:876`; `tech_reports/tensor_accessor/tensor_accessor.md` | `(page_id, accessor, src_l1_addr)` | `cb_input_shard` (`get_read_ptr`) | — | RT `block_rows`, `block_cols`, `first_page`, `src_first_slot` |
| `drain_writes` | raw_api | `noc_async_write_barrier` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1951` | — | — | — | — |
| host: shard order | host | `ttnn.corerange_to_cores` | `ttnn/cpp/ttnn-nanobind/tensor.cpp:659-665` | `(grid, row_wise=orientation==ROW_MAJOR)` | — | — | — |
| host: shard CB | host | `ttnn.cb_descriptor_from_sharded_tensor` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:517` | `(CB_INPUT_SHARD, input_tensor)` | — | — | — |

Helpers considered and rejected (for both raw_api rows): every `kernel_lib` helper is a compute helper (`tilize`, `untilize`, `reduce`, `eltwise_*`, `matmul_block`, …) that moves data through unpack → DEST → pack; this op has no compute phase and must keep bfloat8_b bytes (shared exponents) verbatim, which a repack cannot guarantee (spec rule on bfloat8_b). `mcast/kernel/mcast_args.hpp` multicasts one payload to many cores; here each tile goes to exactly one page, so there is no multicast. No dataflow helper for "sharded slot → interleaved page" exists in `ttnn/cpp/ttnn/kernel_lib/`.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Wrong shard ↔ core order | grid has several ranges (p1) and column-major specs (LOOSE) | derive order ONLY via `ttnn.corerange_to_cores(grid, row_wise=...)` — the allocator's function — truncated to `num_shards`; never assume a rectangle or row-major |
| Short last width shard stride | padding slots: slot stride is `shard_col_tiles`, not `block_cols` | stride is a CT arg = `shard_col_tiles`; LOOSE `_P2_OTHER_SPEC` (last shard of 1, column-major) tests it |
| Padded rows: `(1, 5120)` has 1 logical row | `floor(1/32) = 0` tile rows | `tensor_row_tiles` from `padded_shape` (per image, ceil) |
| Program cache replays a wrong rectangle | orientation / range order change only RT args, not hashed | `custom_program_hash` over the full shard-spec key (Work Distribution) |
| Output pre-filled correctly by chance | fresh / dirty buffers hide map errors | every page written every call; input is tile-distinct in the tests; LOOSE dirty-destination case |
| Input must stay intact | sharded CB aliases the input buffer | kernel only reads the CB; no `cb_pop_front`/writes; never deallocate / relabel the input |
| Trace | allocation must not happen at replay | the only allocation is the output in the Python entry point; the device program allocates nothing; replay runs the cached program |
| bfloat8_b refinement | page size 1088; a compute repack could regroup exponents | raw byte copy of `out_page_size`; `in_aligned_page_size` stride for the source |
| L1 budget (BB-05) | L1 destination output pages share the 1347776 B | op adds 0 B of CBs; output L1 per bank ≈ `ceil(1024/num_l1_banks) * 1088` B (≈ 10 KB) — see `l1_ledger.md` |
| Profiling (BB-21) | writes + barrier must be zoned | `DeviceZoneScopedN("S2I_STORE_BLOCK")` around `store_block`, `DeviceZoneScopedN("S2I_DRAIN")` around the barrier; host helper dumps core → role (`writer` / `idle`) |

## Acceptance test scope

`tests/ttnn/unit_tests/operations/g6_sharded_to_interleaved/test_g6_sharded_to_interleaved.py` covers only the production cases and the cases the spec's rules require (BB-22 overrides the generic "≥4 shapes" convention): p2 production, p2 other spec (column-major, short last shard), p2 → L1, p1 production, p1 other spec, plus repeat and input-intact checks. Exact bit equality (TOLERANCES are `("exact", None)`). Cells outside the current SUPPORTED raise a `SupportRefusal` and are reported as xfail.
