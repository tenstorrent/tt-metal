# Operation Design: typecast

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (one format conversion per element, no arithmetic; bound by data movement) |
| Goal | `typecast(input_tensor, dtype)`: a new tensor of the input's shape holding every element converted to `dtype`, produced by exactly one `ttnn.generic_op` dispatch of one Metal 2.0 `ttnn.ProgramSpec` |
| Math | `output[i] = convert_{in_dtype -> dtype}(input[i])`; Phase 0: bf16→fp32 exact widen, fp32→bf16 one round-to-nearest-even, same dtype = copy |
| References | Spec `eval/specs/typecast/spec.py` (authoritative); target `.claude/references/targets/quasar.md`; backend `.claude/references/backends/ttsim.md`; host-program pattern `.claude/references/quasar_op_examples/toy_thread_plan/` (`toy_thread_plan_program_artifacts.py`, `kernels/reader_implicit.cpp`, `kernels/writer_implicit.cpp`, `kernels/compute.cpp`); catalog `ttnn/ttnn/operations/examples/master.md` (`double_buffer`, `compute_block_size` + its reconfig ablation) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | `quasar` | target page |
| Compute grid | 8 × 4 = 32 clusters (confirmed: `device.compute_with_storage_grid_size()` returned `8-4` on this run) | target page + probe |
| L1 per core | 4,194,304 B | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` = 3,963,328 − 65,536 = 3,897,792 B on this target (query returned 3,963,328 on this run) | run-time query, read on this run through `scripts/tt-probe.sh` |
| Host program | `ttnn.ProgramSpec` (Metal 2.0) + `ttnn.generic_op` — `.claude/references/quasar_op_examples/` | target page |
| Threads per core | 4 NEO compute engines (C ∈ {1,2,4}); 6 user DM harts (R + W ≤ 6); on STRIDED rings R | C and W | C. DFB ring ≤ 1,048,560 B compute side; 32 DFBs/cluster | target page *Cluster model / Thread plan* |
| Unavailable here | (1) `ttnn.ProgramDescriptor` / Metal 1 kernels and the legacy C++ program factory — refused at kernel creation. (2) `bfloat8_b` / `bfloat4_b` — no path on this family; the golden suite skips those cells as `unbuildable on quasar`. (3) `UnpackToDest` of a **bfloat16** page into a 32-bit DEST: not used; the eltwise chain's operand reconfig (`DataFormatReconfig::Enabled`): not used on a ring read with `UnpackToDest`; `fpu_math_fidelity` is always set explicitly; uint16 / uint32 are not DFB formats (`dfb.cpp:148`); implicit-sync kernels: tag hand-back required | target page *Not available* |

All worked numbers below are for `quasar` (8×4 grid, 3,963,328 B unreserved). Every run-time predicate uses the query, never these values.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on device; Phase 0: bf16/fp32, TILE, DRAM interleaved, rank 4, tile-aligned | — | tensor binding |
| `dtype` | `ttnn.DataType` | yes (positional or `dtype=`) | TARGET `target_dtype`; Phase 0: `bfloat16`, `float32`. Not a `ttnn.DataType` → `TypeError` (REJECTIONS X1), checked **before** any axis is derived | — | host only: selects ring formats + unpack mode (compile-time through the DFB formats) |

No compute config parameter (spec DEFINITION: "The op has no compute config"). The compute kernel's hardware config is fixed by the op (see Dataflow Strategy).

### Entry point and validate()

| Step | Rule |
|------|------|
| 1 | `if not isinstance(dtype, ttnn.DataType): raise TypeError(...)` (X1) |
| 2 | axes = `{"dtype": input_tensor.dtype, "target_dtype": dtype, "layout": input_tensor.layout, **eval.memory.memory_axes(input_tensor)}` (R7: `memory_config`, `sharding` via `memory_axes`), then `INPUT_TAGGERS = AXES` imported from `eval.specs.typecast.spec` (`alignment`, `rank`) applied to `((tuple(input_tensor.shape),), axes)` |
| 3 | every axis in SUPPORTED (= PHASE0 after the first phase) checked; refusal raises `UnsupportedAxisValue` (a `NotImplementedError`, R5). A sharded input is refused by name through the `sharding` axis |
| 4 | output = `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(input_tensor.shape, dtype, output_layout, input_tensor.memory_config()), device)` — the input's buffer, interleaved (R6); `output_layout` = input layout (TILE in Phase 0; TILE for a bfloat8_b output per spec OUTPUT) |
| 5 | build the ProgramSpec and dispatch once: `ttnn.generic_op([input_tensor, output], spec, run_args, tensor_indices)`. Same-dtype calls go through the same program (never return the input: one dispatch per call, and tt-metal returns a new tensor) |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | any rank (TARGET 2–4); Phase 0 rank 4, H and W multiples of 32 |
| Dtype | TARGET: float32, bfloat16, bfloat8_b, int32, uint32, uint16, uint8; Phase 0: bfloat16, float32 |
| Layout | TILE (Phase 0); ROW_MAJOR deferred (regime table) |
| Memory | interleaved, DRAM (Phase 0) or L1 (TARGET, R4) |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` |
| Dtype | `dtype` |
| Layout | input's layout; TILE when `dtype == bfloat8_b` |
| Memory | input's buffer type, interleaved (`input_tensor.memory_config()`) |

## Blocking Model

The op's index space, in TILE layout, is the tensor's page sequence: `tensor_tiles = output.buffer_num_pages()` pages, which equals `prod(leading dims) · ceil(H/32) · ceil(W/32)` (each image tile-padded independently — the buffer's own page count, so `ceil`-per-image is by construction, never `floor(batch·H/32)`). Every output element depends on exactly one input element at the same position, and the two tensors share one page order, so the op flattens honestly to one axis: the flat tile index. The leading, H-tile and W-tile axes are contiguous in that order and treated identically, so they fold into it.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `tile` (flat page index = leading dims × H-tiles × W-tiles) | independent — each output tile is a function of the same-index input tile only | `block_tiles` (tiles per compute block, per NEO) | `clamp(ceil(core_tiles_max / buffer_depth), 1, block_tiles_cap)`; = 1 for both Phase-0 calls (1 tile per core) and 4 for the largest INPUTS shape (16 tiles per core; the DEST-section cap binds — verifier correction, see Mechanism caps) | one host variable in `create_program_artifacts`; CT arg of compute; ring capacities derived from it | contiguous ranges of `core_tiles` tiles per cluster from `ttnn.split_work_to_cores(grid, tensor_tiles, row_wise=True)`, `start_id` cumulative; all 32 clusters when `tensor_tiles ≥ 32` | knob-turn |
| leading dims (N, C / B, S) | independent — elementwise | folded into `tile` (no separate knob) | folded | — | folded into `tile` ranges | — (a fold is not a decision that later work turns) |
| H-tiles | independent — elementwise | folded into `tile` | folded | — | folded | — |
| W-tiles | independent — elementwise | folded into `tile` | folded | — | folded | — |
| elements within a tile (32×32) | independent — elementwise | not a blocking axis: the tile is the hardware's unit (unpack/pack move whole tiles) | whole tile | hardware | never split | — |

Phase-0 `block_tiles = 1` is the whole per-core share (each Phase-0 call has ≤ 16 tiles over 32 clusters → 1 tile per cluster), not a collapsed knob. Where a core owns more, the rule takes the coarsest block that still leaves `buffer_depth` blocks in flight on the busiest core (overlap lamp, below), clamped by the ring and L1 caps.

Derived quantities (all from the knobs above, never restated):

| Quantity | Expression |
|----------|------------|
| `tensor_tiles` | `output.buffer_num_pages()` |
| `grid` | `device.compute_with_storage_grid_size()` (one variable; the grid-sync lamp shrinks it) |
| `core_tiles`, `start_id` per cluster | `split_work_to_cores(grid, tensor_tiles, row_wise=True)` → groups 1 and 2 |
| `core_tiles_max` | group-1 per-core count |
| `in_page`, `out_page` | `input_tensor.buffer_page_size()`, `output.buffer_page_size()` (2,048 B bf16, 4,096 B fp32) |
| `block_tiles_cap` | `min(dest_tiles, floor(DFB_RING_MAX_BYTES / (C · buffer_depth · max(in_page, out_page))), floor(l1_budget / (C · buffer_depth · (in_page + out_page))))`; `dest_tiles` = 4 (32-bit DEST, half sync) binds on quasar |
| `l1_budget` | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` |
| ring `num_entries` | `buffer_depth · block_tiles · C` (both rings) |
| compute iterations | one helper call over `core_tiles` with `.block_size(block_tiles)` |

### Thread-plan and core knobs

| Knob | Phase 0 value | Source |
|------|---------------|--------|
| `THREAD_PLAN = (R, C, W)` | (1, 1, 1) | one module constant in the program-artifacts file; counts passed as CT args `reader_threads`, `compute_threads`, `writer_threads` |
| `buffer_depth` | 2 | one module constant |
| `l1_reserve_bytes` | 65,536 | one module constant |
| `DFB_RING_MAX_BYTES` | 1,048,560 | one module constant, from the target page (Python has no query for it) |
| `grid` | device query | one variable |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `buffer_depth` | 2 (blocks) | reader fills block b+1 while compute converts block b |
| `cb_output_tiles` | `buffer_depth` | 2 (blocks) | compute packs block b+1 while the writer drains block b |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DFB ring size (≤ 1,048,560 B compute side, target page) | `block_tiles` | `block_tiles ≤ floor(DFB_RING_MAX_BYTES / (C · buffer_depth · max(in_page, out_page)))` = 127 for any pair with an fp32 side, 255 for bf16→bf16 (quasar) | program build refuses the ring, or a ring wraps into neighbouring L1 |
| L1 budget (query) | `block_tiles` | `floor(l1_budget / (C · buffer_depth · (in_page + out_page)))` (237–475 on quasar; never binding before the ring cap here) | allocation failure |
| Ring wrap with block-wise wait (no copy window may straddle the ring end) | ring capacity vs block | capacity is exactly `buffer_depth · block_tiles` per NEO, so every full block starts block-aligned; a ragged (shorter) block is only ever the **last** block of a core, so no later block is misaligned | — |
| DEST section (32-bit DEST, half sync: 4 tiles) | `block_tiles` (verifier correction: the original row said "none") | `block_tiles ≤ dest_tiles` (host `dest_tiles()`; compute `static_assert` against `DEST_AUTO_LIMIT`) | the chain clamps a `PerBlockSize` block to one DEST section and syncs the rings per clamped window, so with a ring of `buffer_depth · block_tiles` entries a window straddles the ring end (`l1_ledger.md`) |
| `UnpackToDest` scope (kernel-wide `UnpackToDestEn`, `tt_metal/jit_build/genfiles.cpp:1221-1228`) | none on extents; it is a per-program choice | set per input dtype (fp32 → UnpackToDest; bf16 → UnpackToSrc) | fp32 via SrcA truncates to 19 bits → fp32→fp32 not exact |

### Regimes

The minimum for every row is "the input crosses DRAM (or L1-interleaved banks) once, the output once".

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `tile_stream_float` — TILE, interleaved, in/out dtype ∈ {bf16, fp32}; reader → `cb_input_tiles` → compute (copy into 32-bit DEST + pack to output format) → `cb_output_tiles` → writer | **built** (Phase 0) | `layout == TILE and dtype ∈ {bf16, fp32} and target_dtype ∈ {bf16, fp32} and sharding == interleaved` (covers DRAM and L1, any rank, any alignment — padded tiles are pages like any other; Phase 0 SUPPORTED is the DRAM / rank-4 / tile-aligned corner of it) | `block_tiles` × 1 tile per NEO | minimum: each input tile read once, each output tile written once; no cross-core traffic | per block: one wait/pop on `cb_input_tiles`, one reserve/push on `cb_output_tiles`, one pipeline fill/drain of unpack→math→pack. Inits (`copy_init`, `typecast_tile_init`, `pack` config) are once per kernel at the intended frequency (chain is uniform → boot-hoisted). Per-tile NoC transactions are fixed by the page size and do not change with the block |
| `tile_stream_int` — the same kernels; the compute call becomes `unary<Typecast<in_df, out_df>>` with explicit formats (the `typecast_tile` SFPU arm runs), per-ring unpack mode (32-bit and narrow-int sources → UnpackToDest) | deferred — outside PHASE0; built **on top of** `tile_stream_float` (identical reader/writer/split; only the compute call and the ring formats change). Open input it must settle first: uint16/uint32 are not DFB formats, and a ring declared as another format changes what the unpacker decodes | `layout == TILE and (dtype or target_dtype) ∈ {int32, uint32, uint16, uint8}` | same | minimum | same, plus the SFPU pass amortized nowhere (per tile, inherent); thread-plan lamp applies (SFPU work per tile) |
| `row_major_stream` — ROW_MAJOR in and out: the reader packs `entry_bytes` of contiguous stick bytes into each ring entry (explicit sync, tagged transfers), compute treats each entry as a pseudo-tile (conversion is position-agnostic and pack writes in unpack order), writer emits sticks | deferred — outside PHASE0; compute is unchanged from `tile_stream_float`, only reader/writer and the page unit change (a knob-turn of the unit, not a new loop nest). bf8b output from RM needs a tilize and is unbuildable on quasar anyway | `layout == ROW_MAJOR` | `block_tiles` pseudo-tiles | minimum (sticks read once, written once); ragged last entry converts padding that is not written | as above, plus fewer stick-gather transactions per entry |
| `same_dtype_dm_copy` — `dtype == input.dtype`: reader → one ring → writer, no compute kernel | deferred — `tile_stream_float` already serves every shape this would, at the same DRAM crossings (minimum); the compute pass it drops is off the critical path of an op bound by data movement. One experiment settles it on a timed backend: model cycles of the two programs at the same shape | `dtype == input_tensor.dtype` | same | minimum (same as built) | — |
| `return_input` — return the input tensor (or a host-side clone) with no dispatch | rejected — violates "exactly one native dispatch per invocation"; tt-metal returns a new tensor (spec DECISIONS 2026-11-21) and R6 allocates a fresh output. Superseded by `tile_stream_float` | — | — | — | — |
| `sharded_in_place` — ring backed on an L1 shard, no NoC read | rejected — outside TARGET (`sharding: ["interleaved"]`); R7 refuses sharded inputs by name. Should TARGET grow, it is a placement knob-turn (zero-copy ring on the shard), the axis being independent | — | whole shard | — | — |
| bfloat8_b endpoints | rejected on this target — no block-float path on quasar (target page *Family facts*); cells skip as `unbuildable on quasar` | — | — | — | — |

Single built regime: there is no dependent axis and no shared operand, so no combine or multicast regime exists; the regime is selected by dtype/layout, not by size, and no size predicate is needed (the per-core working set is bounded by `block_tiles_cap` regardless of tensor size).

### Traffic ranking

| Candidate split | DRAM crossings | Cross-core | Notes | Rank |
|-----------------|----------------|------------|-------|------|
| S1: flat tile index, contiguous range per cluster | input 1×, output 1× | none | balanced to ±1 tile; contiguous ranges keep the RM refinement's stick ranges contiguous | **1 (chosen)** |
| S2: round-robin tiles over clusters (tile i → cluster i mod N) | 1×, 1× | none | same bytes; interleaved pages are bank-scattered anyway, so no bank benefit; tie broken for S1 (simpler start/count args) | 1 (tie) |
| S3: tile-rows (H-tiles × leading) per cluster | 1×, 1× | none | same bytes, coarser unit → imbalance up to `Wt − 1` tiles per cluster, under-fills the grid on short tensors (Phase-0 call 1 has 4 tile-rows) | 3 |
| Split a dependent axis + combine | — | — | no dependent axis exists (elementwise) | n/a |
| Broadcast a reuse-shared operand | — | — | operand-reuse check: the single input varies along every split axis → nothing is re-read by any split | n/a |

Stall-shadow: the only waits are compute on the reader's tiles and the writer on compute's tiles; double-buffered rings already overlap them, and no other work exists to schedule into them.

### Block schedule

Per cluster (one NEO, one reader hart, one writer hart in Phase 0):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // num_blocks_this_core = ceil(core_tiles / block_tiles)
    load_block(block_idx);      // reader: block_tiles tagged reads, pages start_id + block_idx*block_tiles + j
    convert_block(block_idx);   // compute: unpack (Src or Dest per dtype) -> 32-bit DEST -> pack in the output format
    store_block(block_idx);     // writer: block_tiles tagged writes to the same page ids of the output
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `load_block` | `block_tiles` input tiles (last block of a core: `core_tiles − (num_blocks_this_core−1)·block_tiles`) | nothing persists; ring holds ≤ `buffer_depth` blocks | one tagged `async_read` per tile (implicit sync: no reserve/barrier/push); TensorAccessor built once per kernel; tag hand-back (`noc_async_read_set_trid(0, …)`) once at kernel end |
| `convert_block` | `block_tiles` tiles | nothing persists | `compute_kernel_hw_startup` once; chain init (`copy_init` + `typecast_tile_init`, the latter a no-op for these pairs) once per kernel (uniform chain); data-format reconfig **never** (Disabled: one input and one output ring, formats set by startup); one wait/pop + one reserve/push per block |
| `store_block` | `block_tiles` output tiles | nothing persists | one tagged `async_write` per tile; tag hand-back (`noc_async_write_set_trid(0, …)`) once at kernel end |

The compute kernel realizes the whole loop as **one** helper call over `core_tiles` with `.block_size(block_tiles)` and per-block wait/pop/reserve/push (not a per-tile loop in the main schedule). Reader and writer loop per tile because implicit sync is one tagged call per ring entry (target page *Ring protocols*); that is the transport's unit, not a blocking decision.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (block size) | The coarse default would be the whole per-core share in one block, which serializes read → convert → write on each cluster; the chosen default (`ceil(core_tiles_max/buffer_depth)`) keeps two blocks in flight, but the catalog (`double_buffer`: sweet spot 4–8) says a small fixed block may be as good with less L1 | `block_tiles ∈ {whole share, 8, 4, 1}` at a shape with ≥ 32 tiles per cluster, model cycles on `ttsim-perf` |
| Grid synchronization | Phase-0 calls have 8 and 16 tiles: 8/16 clusters each run a full 3-kernel launch for one tile; per-cluster launch cost may dominate | shrink `grid` to engage `ceil(tensor_tiles / k)` clusters for k ∈ {2, 4} |
| Thread plan | (1,1,1) is right for the float pairs (no SFPU work: copy + pack). The integer pairs of `tile_stream_int` run the SFPU typecast per tile | (1,4,1) and (2,2,2) for an int pair; (1,1,1) stays for float pairs |
| Same-dtype compute pass | The compute pass in a same-dtype call does no conversion | `same_dtype_dm_copy` (regime row) — model cycles of the two programs |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM/L1 interleaved input → `cb_input_tiles` | input tiles, ring format = input dtype (Float16_b 2,048 B or Float32 4,096 B) | reader hart: `noc.async_read<NocOptions::TXN_ID>(acc_in, cb_input_tiles, {.page_id = start_id + i}, {})`, implicit sync | one hart (R=1); stride `R` from CT arg; last statement `noc_async_read_set_trid(0, noc.get_noc_id())` |
| `cb_input_tiles` → DEST | bf16 input: **UnpackToSrc** (SrcA, then A2D datacopy into 32-bit DEST — exact widen). fp32 input: **UnpackToDest** (full 32 bits) | compute `ComputeHardwareConfig(enable_32_bit_dest=True, fpu_math_fidelity=LoFi, unpack_modes={"cb_input_tiles": UnpackToDest if input fp32 else UnpackToSrc})` | `enable_32_bit_dest=True` for all four pairs; UnpackMode must be explicit for a 32-bit ring with 32-bit DEST (`compute_hardware_config.hpp:89-115`); fidelity explicit, LoFi because a datacopy has no multiply |
| DEST → `cb_output_tiles` | packer emits the output ring's format: Float32 (exact) or Float16_b (one RNE rounding — see Risks) | `typecast` helper (SFPU step is a compile-time no-op for Float16_b↔Float32 on quasar, `typecast.h:19-23,111`) | `DataFormatReconfig::Disabled` on input and output (unnecessary: one ring each side) |
| `cb_output_tiles` → output | output tiles | writer hart: `noc.async_write<NocOptions::TXN_ID>(cb_output_tiles, acc_out, {}, {.page_id = start_id + i})`, implicit sync | last statement `noc_async_write_set_trid(0, noc.get_noc_id())` |

Placement as a TARGET axis: `memory_config ∈ {DRAM, L1}` interleaved — the TensorAccessor addresses both; L1 is a knob-turn (SUPPORTED widening + output follows the input's buffer, R6). `sharding` has only `interleaved` in TARGET; no physical shard scheme is owed.

Inter-Tensix communication: none in any regime (no dependent or reuse-shared axis).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one tile (= one ring entry) |
| Grid | `device.compute_with_storage_grid_size()` (8×4 on quasar) |
| Per-core work | `ttnn.split_work_to_cores(grid, tensor_tiles, row_wise=True)`: group 1 gets `core_tiles_max`, group 2 `core_tiles_max − 1`; `start_id` = running sum over cores in row-major order. Worked (quasar): `[1,4,32,64]` → 8 tiles → 8 clusters × 1; `[1,1,128,128]` → 16 tiles → 16 clusters × 1; `[1,1,2048,256]` → 512 tiles → 32 × 16, `block_tiles = 4` (DEST cap) |
| Remainder | split_work_to_cores' two groups; within a core the last block is ragged (`BlockTailSync::ValidTiles`); `tensor_tiles` is the buffer's page count (per-image `ceil`) |
| Bound | data movement — per tile the compute is one unpack + datacopy + pack with no SFPU arithmetic (bf16↔fp32 is an SFPU no-op), against 2–4 KiB read and 2–4 KiB written |
| Thread plan | (R, C, W) = (1, 1, 1), host constant `THREAD_PLAN`; counts reach the kernels as CT args and set the reader/writer stride and the compute share (`core_tiles / C` + remainder per `get_my_thread_id()`), so turning it is a one-constant edit (perf lamp) |
| Unit dealt to compute threads | the ring entry itself (a tile): no padding rounds; round-robin dealing may stop anywhere |
| Tests the plan needs | (a) a test reading `num_threads` of reader/compute/writer off the ProgramSpec equals `THREAD_PLAN`; (b) one device session: run another program (e.g. a second `typecast` call, or `ttnn.experimental.quasar` softmax if available) **then** this op, **then** `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)`, and compare both results — catches a missing tag hand-back; (c) at a C > 1 turn of the plan, a test showing every NEO converted its tiles |

Other tests required beyond the acceptance test (regime-pinned and mechanism-pinned):

| Test | Pins |
|------|------|
| all four Phase-0 dtype pairs (acceptance covers) | both unpack-mode variants (bf16 via Src, fp32 via Dest) |
| fp32→bf16 on crafted ties: fp32 values whose low 16 bits are exactly `0x8000` with even and odd bf16 LSB, plus `0x7FFF`/`0x8001` neighbours; compare bit-exactly with `torch.Tensor.to(torch.bfloat16)` | packer rounding is RNE (Risk 1) |
| fp32→fp32 on values with nonzero low 13 mantissa bits | UnpackToDest preserves all 32 bits (no SrcA 19-bit truncation) |
| `[1,1,96,992]` (93 tiles: 29 clusters × 3, 3 × 2 → `block_tiles = 2`, ragged last block) and `[1,1,2048,256]` (16 per cluster, four blocks of 4) | multi-block walk, ragged tail, ring alignment |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `cb_input_tiles` | 0 | `in_page` (input `buffer_page_size()`) | `buffer_depth · block_tiles · C` | spans `tile` → `block_tiles` per block, × depth for overlap, × C per NEO | input dtype (Float16_b / Float32) | reader | compute | reader: implicit (tagged reads); compute: chain wait/pop per block | whole kernel |
| `cb_output_tiles` | 1 | `out_page` (output `buffer_page_size()`) | `buffer_depth · block_tiles · C` | same as above | output dtype (Float16_b / Float32) | compute | writer | compute: chain reserve/push per block; writer: implicit (tagged writes) | whole kernel |

DFB unique ids are the semantic names; STRIDED (default) access pattern. Ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_tiles` tiles | no (dataflow, raw NoC API) | DRAM/L1 tensor via TensorAccessor | `cb_input_tiles`, `block_tiles` entries credited by DM0 ISR | — |
| 2 | `convert_block` | `block_tiles` tiles (whole share in one call, synchronized per block) | yes: `compute_kernel_lib::typecast` | `cb_input_tiles`, waits/pops `block_tiles` per block | `cb_output_tiles`, reserves/pushes `block_tiles` per block | both rings drained of this block |
| 3 | `store_block` | `block_tiles` tiles | no (dataflow, raw NoC API) | `cb_output_tiles` (entries acked by DM0 ISR) | output tensor via TensorAccessor | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| startup | raw_api (mandatory engine init, not a phase) | `compute_kernel_hw_startup(icb, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:113` | `(cb_input_tiles, cb_output_tiles)`, first statement | — | — | none |
| `convert_block` | helper | `compute_kernel_lib::typecast<Input, Output>(IterationShape, in, out)` | decl `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:89-91`; def `convenience.inl:48-57` (derives in/out formats from the rings via `dfb_l1_format`, forwards to `unary<Typecast<in_df,out_df>>` = `CopyTile → Typecast → PackTile`) | `Input = input(cb_input_tiles, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block, DataFormatReconfig::Disabled)` (`chain.hpp:385-392`; Block + PerBlockSize legal per `chain.inl:58-63`); `Output = output(cb_output_tiles, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize, DataFormatReconfig::Disabled)` (`chain.hpp:442-450`); shape `IterationShape::tiles(neo_tiles).block_size(block_tiles)` (`chain.hpp:132,159,168-172`, default `BlockTailSync::ValidTiles`) | `cb_input_tiles` | `cb_output_tiles` | `block_tiles` (`.block_size`), `neo_tiles` (= this NEO's share of `core_tiles`) |
| (inside the helper) | — | `Typecast<InDF,OutDF>` → `typecast_tile_init` / `typecast_tile` | `ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.inl:85-90`; quasar no-op for Float16_b↔Float32 and same format: `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h:19-23,103,111,473-474` | — | — | — | — |
| (inside the helper) | — | `copy_tile` with `UnpackToDestEn` | `tt_metal/hw/inc/api/compute/tile_move_copy.h:130-132`; `UnpackToDestEn` kernel-wide from the per-ring unpack modes `tt_metal/jit_build/genfiles.cpp:1221-1228` | — | — | — | — |
| `load_block` | raw_api | `noc.async_read<NocOptions::TXN_ID>(accessor, ring, {.page_id}, {})`; `noc_async_read_set_trid(0, noc.get_noc_id())` at end | pattern: `.claude/references/quasar_op_examples/toy_thread_plan/kernels/reader_implicit.cpp:26-36` | `TensorAccessor(tensor::input)` | — | `cb_input_tiles` | none (per-entry transport) |
| `store_block` | raw_api | `noc.async_write<NocOptions::TXN_ID>(ring, accessor, {}, {.page_id})`; `noc_async_write_set_trid(0, noc.get_noc_id())` at end | pattern: `.../toy_thread_plan/kernels/writer_implicit.cpp:24-30` | `TensorAccessor(tensor::output)` | `cb_output_tiles` | — | none |
| host | — | `ttnn.experimental.ProgramSpec`, `KernelSpec(num_threads=…)`, `DataflowBufferSpec`, `ComputeHardwareConfig`, `split_work_to_cores`, `generic_op` | pattern: `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py`; `ComputeHardwareConfig` fields `tt_metal/api/tt-metalium/workload/specification/compute_hardware_config.hpp:53,70,86,89-115`, Python kwargs `ttnn/cpp/ttnn-nanobind/program_spec.cpp:546-550` | — | — | — | `block_tiles`, `buffer_depth`, `THREAD_PLAN`, `grid` |

Helpers considered and rejected for the raw DM entries: `dfb_helpers_dataflow.hpp` (tile r/c-dim queries only, `:13-17`) and `local_copy_helpers_dataflow.hpp` (L1-local copies, `:74-132`) do not issue DRAM/L1-interleaved reads or writes to a ring; the kernel library has no dataflow helper for the tagged implicit-sync transfer, and the target page prescribes that form.

The `copy` convenience (`convenience.inl:66-69`) would also serve the four Phase-0 pairs (Typecast is a no-op there), but `typecast` is the one call that stays correct when `tile_stream_int` widens the dtype set, so it is the one used.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| 1. fp32→bf16 rounding mode of the packer | The spec grades `exact`: bit-identical to `torch.to(bfloat16)` (RNE). No Quasar LLK file shows a pack rounding config; the design relies on the packer narrowing Float32 DEST to Float16_b correctly (the path `typecast.h:16-18` documents) | Tie test (Work Distribution). If it fails, add one SFPU element before the pack that rounds the 32-bit value to bf16 RNE in DEST (same block shape, same rings); record as a finding, do not loosen the metric |
| 2. fp32 input through SrcA | SrcA holds 19 bits: fp32→fp32 would not be exact and fp32→bf16 would round twice | `unpack_modes["cb_input_tiles"] = UnpackToDest` when the input is fp32 (mandatory explicit mode with 32-bit DEST) |
| 3. bf16 input with UnpackToDest | — | bf16 input ring stays UnpackToSrc; the A2D copy into 32-bit DEST is exact |
| 4. Chain reconfig with UnpackToDest | — | Disabled on both specs; formats come from `compute_kernel_hw_startup` once |
| 5. Fidelity | set on every compute kernel | `fpu_math_fidelity = LoFi` set explicitly |
| 6. Sticky NoC tag | an implicit-sync kernel leaves a ring's tag in the hart's tag register | tag hand-back as the last statement of reader and writer; device-session test (b) |
| 7. Same-dtype calls | Tempting to return the input; that is zero dispatches and not a new tensor | same program, `Typecast` no-op → copy; result a fresh tensor in the input's buffer |
| 8. Output placement | R6: output in the input's buffer, interleaved | `TensorSpec(..., input_tensor.memory_config())` |
| 9. `dtype` not a DataType (X1) | deriving axes from a string would fail with an unclear error or be refused as "unsupported" | `isinstance` check first, `TypeError` |
| 10. Two page sizes | input and output rings differ in entry size (2,048 vs 4,096 B) | each ring sized from its own tensor's `buffer_page_size()`; the block cap uses the larger page |
| 11. Later integer pairs | uint16/uint32 are not DFB formats; Int8/UInt8/Int16 sources must UnpackToDest (`typecast.h` static_assert) | recorded in `tile_stream_int`'s row as the first thing that refinement settles |

### Structural impossibilities

None beyond the specification's INVALID.
