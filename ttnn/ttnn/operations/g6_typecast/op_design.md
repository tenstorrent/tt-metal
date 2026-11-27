# Operation Design: g6_typecast

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (elementwise format conversion; DRAM/L1-bandwidth-bound streaming) |
| Goal | Convert every value of one tensor to `dtype` in **one** device program, bit-identical to `torch.Tensor.to` (bfloat8_b: within one quantization step of the host rendering), keeping shape, layout and interleaved memory config. |
| Math | `out[i] = conv_{src→dst}(x[i])` for every logical index `i` (rules in the spec; Phase 0: bf16→fp32 exact, bf16→bf8b block-quantize, bf16→int32 trunc toward zero) |
| References | `eval/golden_tests/g6_typecast/feature_spec.py` (TARGET / INPUTS / INVALID / LOOSE_CASES); `eval/golden_tests/g6_typecast/helpers.py` (oracle, TOLERANCES: exact / `bfp8_group`); `eval/op_template.py` (registry model); `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:91` (`typecast` helper); `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h:91` (`typecast_tile`); `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h:198` (fp32→int32 truncation); catalog `ttnn/ttnn/operations/examples/master.md` entries `double_buffer`, `noc_placement`, `compute_block_size`, `width_split`, `bank_stagger`, `read_coalescing` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 1–4, on device, interleaved | — | — (addresses RT) |
| `dtype` | `ttnn.DataType` (positional or keyword) | yes | TARGET `output_dtype` | — | CT (selects CB formats, `fp32_dest_acc_en`, `bfp8_pack_precise`, SFPU arm) |
| `sub_core_grids` | `ttnn.CoreRangeSet` keyword-only | no | `None` only | `None` | — |

Entry-point checks (before `validate()`'s registry checks, inside `validate()` which is the entry point's first line): `dtype` not a `ttnn.DataType` → `ValueError`; `sub_core_grids is not None` → `ValueError`; `len(shape) ∉ {1,2,3,4}` → `ValueError`. Then SUPPORTED per-axis → `UnsupportedAxisValue`, then EXCLUSIONS (empty) → `ExcludedCell`.

### Registry axes (names = `feature_spec.TARGET` keys)

| Axis | Tagger | Value |
|------|--------|-------|
| `dtype` | `tag_dtype` | `scenario["dtype"]` (input dtype) |
| `output_dtype` | `tag_output_dtype` | `scenario["output_dtype"]` (the `dtype` argument) |
| `layout` | `tag_layout` | `scenario["layout"]` |
| `buffer` | `tag_buffer` | `"dram"` / `"l1"` from the input memory config's buffer type |
| `alignment` | `tag_alignment` | W=`shape[-1]`, H=`shape[-2]` (H=1 for rank 1): `"w_non_aligned"` if W%32, elif H%32 `"h_non_aligned"`, else `"tile_aligned"` |
| `rank` | `tag_rank` | `len(shape)` |

`validate()` builds the scenario dict `{input_shape, dtype, output_dtype, layout, buffer}` from `input_tensor` and `dtype` and runs the taggers on `(scenario,)`. Phase 0 SUPPORTED: `dtype=[bfloat16]`, `output_dtype=[float32, bfloat8_b, int32]`, `layout=[TILE]`, `buffer=["dram","l1"]`, `alignment=[all three]`, `rank=[3,4]`. EXCLUSIONS = `[]` (and stays empty in every phase). No size / width / page-size axis exists.

Structural impossibilities: none beyond `feature_spec.INVALID` (`{output_dtype: bfloat8_b, layout: ROW_MAJOR}`).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 1–4 (Phase 0: 3, 4); any H, W (non-tile-aligned handled natively) |
| Dtype | Phase 0: bfloat16 (TARGET adds float32, int32, uint32, uint16) |
| Layout | Phase 0: TILE (TARGET adds ROW_MAJOR) |
| Memory | interleaved, DRAM or L1 |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` (same logical and padded shape) |
| Dtype | `dtype` argument |
| Layout | `input_tensor.layout` |
| Memory | `input_tensor.memory_config()` (interleaved, same buffer type). Allocated with `ttnn.allocate_tensor_on_device(shape, dtype, layout, device, input_tensor.memory_config())`. Tile padding content is unspecified. |

## Blocking Model

### Axes

Tile index space of a TILE tensor: `(batch = prod(shape[:-2]), tile_row ∈ [0, ceil(H/32)), tile_col ∈ [0, ceil(W/32)))`. Interleaved page id is `((b·tensor_tile_rows) + r)·tensor_tile_cols + c` — the three outer axes are contiguous in page order and the op treats them identically, so they fold honestly into one flat axis `tile`. Within a tile, the only structure that matters is the bfloat8_b group (16 consecutive values of one tile row = one face row).

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `tile` (flat: batch × tile_row × tile_col) | independent — each output tile depends only on the same input tile | `block_tiles` (tiles per block = per CB handshake = per reader barrier) | `min(core_tiles_max, block_tiles_l1_cap)` → the **whole per-core assignment** for every production shape | host constant computed once in the program-descriptor builder; passed as one CT arg to reader, compute, writer; every CB page count derives from it | contiguous ranges of `tile` via `ttnn.split_work_to_cores(grid, tensor_tiles, row_wise=True)` over the **full compute grid**; `core_tiles` = RT arg | knob-turn |
| `batch` (folded into `tile`) | independent — no op math spans it | none of its own: folded (contiguous, treated identically) | folded | — | via `tile` | knob-turn |
| `tile_row` (folded into `tile`) | independent — no op math spans rows; H padding rows form their own bf8b groups | folded | folded | — | via `tile` | knob-turn |
| `tile_col` (folded into `tile`) | independent — no op math spans tiles | folded | folded | — | via `tile` | knob-turn |
| `bf8b_group` (16 values of a tile row, inside a tile) | **dependent** (bfloat8_b only) — the shared exponent spans the 16 values | not a knob: always the whole group, inside the atomic tile | whole tile (never cut) | fixed by the format | never split across cores (a tile is the atomic transfer and the packer computes the exponent) | n/a — cutting it is not expressible (tile is the NoC/packer unit) |

Step-4 table (one row per axis, no blanks):

| Axis | Character | Block factor (Phase 0) | Core assignment | Buffer depth | Why |
|------|-----------|------------------------|-----------------|--------------|-----|
| `tile` | independent | `block_tiles = min(core_tiles_max, block_tiles_l1_cap)` | split contiguous across full grid, row-wise | `buffer_depth` knob 2, effective `min(2, max_blocks_per_core)` | elementwise; coarsest block that fits; contiguous ranges = sequential page ids round-robin over banks |
| `batch` | independent | folded into `tile` | via `tile` | via `tile` CBs | contiguous with `tile_row` in page order |
| `tile_row` | independent | folded into `tile` | via `tile` | via `tile` CBs | same; a row-only split strands cores when `tensor_tile_rows` is small (catalog `width_split`) |
| `tile_col` | independent | folded into `tile` | via `tile` | via `tile` CBs | same; a column-only split leaves 1 core busy when W ≤ 32 (Q2185 has `tensor_tile_cols = 1`) |
| `bf8b_group` | dependent (bf8b) | whole group inside the whole tile | not assigned across cores | n/a | sub-tile; the packer owns it; the only design obligation is that padding columns never raise a group's exponent (see `mask_pad_columns`) |

No stage introduces intermediate data: the conversion packs straight into the output CB, so there is no second table to re-run. No core is designated to collect or coordinate — every core is a peer.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `buffer_depth` (shared single source) | `effective_depth = min(buffer_depth=2, max_blocks_per_core)` → 1 for every production shape (one block per core) | read of block k+1 overlaps conversion of block k when a core owns >1 block |
| `cb_output_tiles` | `buffer_depth` (same source) | same | write of block k overlaps conversion of block k+1 |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| L1 grant of the model (BB-05): 1347776 B per core for the op | `block_tiles` | `block_tiles_l1_cap = floor(cb_budget_bytes / (effective_depth · (in_page_bytes + out_page_bytes)))`, `cb_budget_bytes = L1_OP_BUDGET_BYTES − out_l1_bytes_per_bank − L1_SLACK_BYTES` where `out_l1_bytes_per_bank = ceil(out_pages / num_l1_banks) · out_page_bytes` when the output is in L1, else 0; `L1_SLACK_BYTES = 65536`; assert `block_tiles_l1_cap ≥ 1` on host | out-of-memory at launch, or an op that only fits when it owns all of L1 |
| `bf8b_group` exponent is computed by the packer over all 16 values in DEST, padding included | tile content, not extent | when `output_dtype == bfloat8_b` and `W % 16 != 0`, the reader zeroes the input's padding columns (`col ≥ W % 32`) of every tile in the last tile column before `push_back` (CT flag `mask_pad_columns`) | a nonzero padding value sets the shared exponent of a straddling group → logical values off by many steps |
| Int32 / Float32 / UInt32 in DEST need 32-bit DEST (`typecast.h:64`) | none (precision flag) | `fp32_dest_acc_en = output_dtype ∈ {float32, int32, uint32}` (Phase 0: float32, int32) | int32 results truncated to 16-bit DEST width — wrong values |
| DEST tile budget (`DEST_AUTO_LIMIT`, `dest_helpers.hpp:103`) | subblock only — owned by the eltwise chain | none at design level | — |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 `tile_stream`** — interleaved TILE in → compute typecast → interleaved TILE out, contiguous tile ranges over the full grid | **built** | `layout == TILE` (every Phase 0 cell; Phase 0 pairs bf16→{fp32, bf8b, int32}) | `block_tiles` contiguous tiles (Phase 0 = whole per-core range) | **minimum**: input crosses its buffer tier (DRAM or L1 banks) once, output once; no cross-core traffic; boundary = the input/output buffer banks | per block: one reader barrier, one CB reserve/push/wait/pop handshake per CB per kernel, one pipeline fill/drain. Inits + format reconfig are **once per kernel** (one chain call per core covering all `core_tiles`), independent of `block_tiles`. Larger blocks = more reads in flight per barrier (catalog `double_buffer`: batching reads is the dominant single-core lever) |
| **R1+ refinement pairs on `tile_stream`** (fp32→bf16, int32→bf16, uint32→int32, uint16→int32 on TILE) | **deferred** — inputs outside Phase 0 SUPPORTED `dtype`; same regime, a CT conversion arm + `unpack_to_dest_mode = UnpackToDestFp32` on `cb_input_tiles` for 32-bit sources (SRCA would cut fp32/int32 to 19 bits) | `layout == TILE ∧ dtype ≠ bfloat16` | same `block_tiles` | minimum | same as R1 |
| **R2 `rm_pseudo_tile_stream`** — ROW_MAJOR in/out (u32→i32, i32→u32, bf16→u8): reader gathers row pages densely into L1, compute treats each 1024-value run as a "pseudo-tile" (unpack and pack use the same intra-tile permutation, so an elementwise op preserves positions), writer scatters rows back | **deferred** — ROW_MAJOR is not in Phase 0 SUPPORTED; reachable because R1's CBs/kernels are already "page stream → elementwise chain → page stream" with `block_tiles` as the unit: only the reader/writer page geometry changes (rows of `row_bytes` per page, `rows_per_block = block_tiles·1024 / W`) | `layout == ROW_MAJOR` | `block_tiles` pseudo-tiles = `block_tiles·1024` values | logical bytes cross once; **physical** DRAM traffic per page is the bank-aligned page (DRAM alignment ≫ 6 B / 3 B rows for F317/F322), so per-page reads are transaction-bound: the refinement must read bank-contiguous runs of pages (consecutive pages of one bank are adjacent at `aligned_page_size` stride) and compact them in L1; input and output share the page→bank map (same page count), so a core can own a bank-local page run in both | per block: one coalesced bank-run read + one write run; this is the regime where block size matters most (4 194 304 pages in F322) |
| **R3 `dataflow_only_convert`** — RISC-V converts values in the reader, no compute kernel | **rejected** — superseded by R1/R2: the unpacker/packer/SFPU convert a tile at hardware rate, a RISC-V loop is per element and would be compute-bound on every case | — | — | minimum bytes, but compute-bound | — |
| **R4 `host_staged`** (to_layout / pad / typecast-in-DRAM staging) | **rejected** — forbidden by the spec (one dispatch, no host transforms, each tensor crosses DRAM once) | — | — | ≥ 2× minimum | — |
| **R5 split of the dependent `bf8b_group` axis + combine** | **rejected** — the group is 16 values inside one tile; the tile is the atomic NoC page and the packer computes the exponent in one pass; there is nothing to combine | — | — | — | — |

Single built regime ⇒ no selection predicate is needed in Phase 0; R2 will be selected by `layout == ROW_MAJOR` (host-checkable). When R2 is built it needs regime-pinned tests (a TILE case and a ROW_MAJOR case each in the acceptance test).

### Traffic ranking

| Candidate split | DRAM/L1-bank crossings | Cross-core | Occupancy | Verdict |
|-----------------|------------------------|------------|-----------|---------|
| flat contiguous `tile` ranges, full grid, row-wise core order | in ×1, out ×1 | none | `min(tensor_tiles, grid)` cores | **chosen** — minimum traffic, full grid, contiguous page ids hit banks round-robin |
| `tile_row` only | in ×1, out ×1 | none | `≤ batch·tensor_tile_rows` | same bytes; strictly worse occupancy (tiebreak) |
| `tile_col` only | in ×1, out ×1 | none | `≤ tensor_tile_cols` (=1 for the dominant W=12 class) | same bytes; strands the grid — rejected |
| bank-affine assignment (core k reads pages of bank k) | in ×1, out ×1 | none | full grid | same bytes, fewer NoC hops/less bank contention — a **perf lamp** (core-assignment variant), not a regime |
| dependent `bf8b_group` split + combine | — | — | — | not expressible (R5) |

All candidate splits move the same bytes; the decision is made by occupancy and contiguity. Operand-reuse check: the single operand varies along `tile` (the split axis) → nothing reuse-shared, no broadcast row. Stall-shadow check: compute waits on the reader and the writer on compute; there is no work independent of the awaited data except the next block, which `buffer_depth` already schedules into the stall when a core owns >1 block. Nothing else to lift.

### Block schedule

```cpp
// per core: core_tiles = RT arg, block_tiles = CT arg, num_blocks_this_core = ceil(core_tiles / block_tiles)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);      // reader
    convert_block(block_idx);   // compute: unpack src format -> (SFPU arm) -> pack dst format
    store_block(block_idx);     // writer
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_block` | `block_tiles` input pages (last block: `last_block_extent` valid, nominal reserve/push of `block_tiles`) | nothing persists | `block_tiles` async page reads via `TensorAccessor`, **one** `noc_async_read_barrier` per block, one `push_back(block_tiles)`; when `mask_pad_columns`: zero padding columns of last-column tiles after the barrier, before the push |
| `convert_block` | `block_tiles` tiles | nothing persists | `compute_kernel_hw_startup` once per kernel; typecast init + data-format reconfig **once per kernel** (one chain call over `IterationShape::tiles(core_tiles).block_size(block_tiles, BlockTailSync::FullBlock)`); CB wait/pop and reserve/push once per block; DEST acquire/release per helper subblock (helper-owned) |
| `store_block` | `block_tiles` output pages | nothing persists | one `wait_front(block_tiles)`, valid-page async writes, `noc_async_writes_flushed` before `pop_front`, **one** final `noc_async_write_barrier` before return (BB-03) |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap | `block_tiles` = whole per-core range ⇒ one block per core ⇒ read, convert and write are serial on each core (effective depth 1) | `block_tiles = ceil(core_tiles / 4)` (or 8 tiles, catalog `double_buffer` sweet spot) with depth 2, on P39 (`[1,128,256,256]` bf8b L1, ~70 tiles/core) and Q2185 (`[16,4096,12]` fp32, ~16 tiles/core). Catalog `double_buffer` says no gain once the grid is DRAM-bandwidth-bound — measure, do not assume |
| Grid synchronization | decode shapes are 1–16 tiles: full occupancy puts one tile on each of up to 16 cores; launch/binary-write cost per core range may dominate | fewer cores (e.g. 1–4) with more tiles each on Q2191 / Q2216 / Q2227 / Q2229 |
| Core placement / NoC | contiguous row-wise ranges vs bank affinity / staggered start (catalog `noc_placement`, `bank_stagger`) | bank-affine assignment or `start = core % num_banks` stagger on Q2185 and P39 |
| bf16→fp32/bf8b unpack path | the copy goes SRCA → datacopy → DEST; unpack-to-DEST skips the FPU | `unpack_to_dest_mode = UnpackToDestFp32` on `cb_input_tiles` for the no-SFPU arms (bit results unchanged; measure) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| input buffer → `cb_input_tiles` | src dtype tiles (bf16, 2048 B) | reader (NCRISC, NoC0): `TensorAccessor` page reads, `block_tiles` in flight, one barrier per block | works for DRAM and L1 interleaved identically (accessor args carry the buffer type) |
| `mask_pad_columns` (CT flag) | bf16 bytes in L1 | reader writes zeros into the padding columns of last-tile-column tiles (faces 1 and 3 for cols 16–31, faces 0/2 tail for `W%32 < 16`) | only when `output_dtype == bfloat8_b ∧ W % 16 != 0`; never active in production (all bf8b W = 256) |
| `cb_input_tiles` → DEST → `cb_output_tiles` | src → DEST (fp32 if `fp32_dest_acc_en`) → dst | compute: eltwise `typecast` helper; arm chosen from CB formats at compile time: bf16→fp32 packer-only; bf16→bf8b packer-only (`bfp8_pack_precise=True`); bf16→int32 SFPU `calculate_typecast_fp32_to_int32` (truncation toward zero, saturating outside ±2³¹, which the contract excludes) | output CB format = destination dtype; page size = `output_tensor.buffer_page_size()` (fp32/int32 4096 B, bf8b 1088 B) |
| `cb_output_tiles` → output buffer | dst tiles | writer (BRISC, NoC1): `TensorAccessor` page writes; final write barrier | output page ids equal input page ids (same padded shape) |

Physical-shard view of the TARGET placement axis: TARGET's placement axis here is `buffer` (`dram`/`l1`, interleaved only) — both are **knob-turn-free**: the same pages, only the accessor's buffer type differs, so both are built in Phase 0. No sharded value exists in TARGET; if one were added, every cut of `tile` is independent ⇒ a knob-turn (CB backed on the shard via `ttnn.cb_descriptor_from_sharded_tensor`, one block = the whole resident shard, no NoC re-read).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block of `block_tiles` contiguous tiles in flat page order |
| Grid | `device.compute_with_storage_grid_size()` (runtime-queried, never inlined); cores from `ttnn.split_work_to_cores(grid, tensor_tiles, row_wise=True)` → `all_cores`, `core_group_1` (`tiles_per_core_g1`), `core_group_2` (`tiles_per_core_g2`); each a CoreRangeSet of ≤ 2 rectangles (BB-17) |
| Per-core work | RT args `start_tile`, `core_tiles` (g1 or g2 count); CT `block_tiles`, `mask_pad_columns`, geometry for masking (`tensor_tile_cols`, `w_tail = W % 32`) |
| Tile geometry | `tensor_tiles = prod(shape[:-2]) · ceil(H/32) · ceil(W/32)` (per image, `ceil`; rank 3 uses `prod(shape[:-2]) = shape[0]`); must equal `input_tensor.buffer_num_pages()` (host assert). Rank 1/2 (refinement): H := 1 / `shape[0]`, batch := 1 |
| Remainder | `split_work_to_cores` gives g1/g2 differing by one tile; within a core `num_blocks_this_core = ceil(core_tiles / block_tiles)`, `last_block_extent = core_tiles − (num_blocks_this_core − 1)·block_tiles`; push/pop counts stay nominal `block_tiles` (FullBlock), only NoC transfers / math cover valid tiles |
| Kernel groups | one reader, one compute, one writer kernel over `all_cores` (identical CT args; per-core RT args) |
| Plan reporting (BB-08) | the op records the selected plan (`regime="tile_stream"`, `num_cores`, `block_tiles`, `effective_depth`) on a module-level `last_plan` attribute |
| Program cache (BB-07) | the descriptor is built once per key `(shape, padded_shape, dtype, output_dtype, layout, memory_config)`; repeat calls only patch buffer addresses (RT args); every CT-affecting value derives from the key |
| Profiling (BB-21) | each kernel wraps its block loop and each blocking wait in `DeviceZoneScopedN` zones (`READ_BLOCK`, `CONVERT`, `WRITE_BLOCK`); no effect on the plain build |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_tiles` | 0 | `input_tensor.buffer_page_size()` (bf16 tile 2048 B) | `effective_depth · block_tiles` | spans `tile` → `block_tiles` × depth; folded axes via `tile`; `bf8b_group` inside the page | input dtype (Float16_b) | reader | compute | whole kernel, streaming |
| `cb_output_tiles` | 1 | `output_tensor.buffer_page_size()` (fp32/int32 4096 B, bf8b 1088 B) | `effective_depth · block_tiles` | same | output dtype (Float32 / Bfp8_b / Int32) | compute | writer | whole kernel, streaming |

Two CBs is the minimum: the conversion packs straight into the destination CB (Rule 3.1). In-place is impossible because input and output pages differ in format and size.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_tiles` pages | dataflow raw API (`TensorAccessor`) | input buffer | `cb_input_tiles`, `block_tiles` pushed | block visible to compute |
| 2 | `convert_block` | `block_tiles` tiles | helper `compute_kernel_lib::typecast` | `cb_input_tiles`, `block_tiles` waited then popped | `cb_output_tiles`, `block_tiles` reserved then pushed | input slot freed |
| 3 | `store_block` | `block_tiles` pages | dataflow raw API (`TensorAccessor`) | `cb_output_tiles`, `block_tiles` waited then popped | output buffer | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `convert_block` (boot) | raw_api (required boot) | `compute_kernel_hw_startup(cb_input_tiles, cb_output_tiles)` | chain contract `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:24-38` | — | `cb_input_tiles` | `cb_output_tiles` | — |
| `convert_block` | helper | `compute_kernel_lib::typecast<input(cb_input_tiles, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize), output(cb_output_tiles, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>(IterationShape::tiles(core_tiles).block_size(block_tiles, BlockTailSync::FullBlock), in, out)` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:91` (decl), `convenience.inl:49-57` (derives `Typecast<in_df,out_df>` from CB formats), `eltwise/unary/misc.inl:87-89`, policies `chain.hpp:233-242`, `IterationShape` `chain.hpp:146-172` | formats derived from CB formats; arms: `typecast.h:150` (Float16_b→Int32 SFPU), `:158` (Float16_b→Float32 packer), `:315` (Float16_b→Bfp8_b packer) | `cb_input_tiles` | `cb_output_tiles` | `block_tiles` (the `block_size`), `core_tiles` (RT extent) |
| `convert_block` (refinement, bf16→uint8) | helper | `unary<Typecast<Float16_b, UInt8>, ...>` explicit formats (`typecast()` static-asserts on UInt8 CBs, `convenience.inl:52-55`) | `convenience.inl:52-56` | — | — | — | same |
| `load_block` | raw_api | `TensorAccessor` + `noc_async_read_page` / `noc_async_read_barrier`, `cb_reserve_back` / `cb_push_back` | `tech_reports/tensor_accessor/tensor_accessor.md` | `TensorAccessorArgs(input_tensor)` CT args | input buffer | `cb_input_tiles` | `block_tiles` |
| `store_block` | raw_api | `TensorAccessor` + `noc_async_write_page`, `noc_async_writes_flushed`, `noc_async_write_barrier` | same | `TensorAccessorArgs(output_tensor)` | `cb_output_tiles` | output buffer | `block_tiles` |

Helpers considered and rejected for the dataflow kernels: `local_copy_helpers_dataflow.hpp` / `tile_column_dataflow.hpp` — they move pages within L1 / by tile column, not a contiguous interleaved page range from a `TensorAccessor`; the reader/writer here is the plain "batch N reads, one barrier" loop the catalog `double_buffer` entry measures. No dataflow helper owns an interleaved page-range stream.

Compute config (`ttnn.ComputeConfigDescriptor`): `fp32_dest_acc_en = output_dtype ∈ {float32, int32, uint32}`; `bfp8_pack_precise = (output_dtype == bfloat8_b)` (packer source format Float16_b → one rounding to the block format, `tt_metal/jit_build/data_format.cpp:329-331`); `math_fidelity` irrelevant (no FPU math); `math_approx_mode=False`.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| bf8b off by >1 step vs the host rendering | the default packer path (`pack_src_format = Bfp8_b`) quantizes differently from the host; padding values inside a straddling 16-group set its exponent | `bfp8_pack_precise=True`, 16-bit DEST; `mask_pad_columns` zeroes padding columns when `W % 16 != 0` |
| int32 values wrong above 2¹⁶ / sign | Int32 needs 32-bit DEST; the golden includes ±(2³¹−2²³) and 2³⁰ | `fp32_dest_acc_en=True` for int32/float32 outputs; SFPU arm truncates toward zero (`ckernel_sfpu_typecast.h:198-238`) |
| fp32 / int32 / uint32 **sources** (refinement) lose low bits | SRCA holds 19 bits; unpacking a 32-bit source through SRCA cuts mantissa/integer bits | refinement sets `unpack_to_dest_mode[cb_input_tiles] = UnpackToDestFp32` for 32-bit sources; int32→bf16 must be RNE twice (verify LLK arm `typecast.h:142` against the two-rounding witnesses 33685505, 2²⁴+2¹⁶+1) |
| uint32↔int32 must keep all 32 bits (refinement) | any float detour or sign-magnitude handling flips bits | use the integer arm with unpack-to-DEST; acceptance test compares bit patterns |
| page size differs in/out | bf16 2048 B in, fp32 4096 B / bf8b 1088 B out; page counts are equal | each CB/accessor uses its own tensor's `buffer_page_size()`; host asserts `in_pages == out_pages` |
| output placement | the output must equal the input memory config (DRAM or L1) | allocate with `input_tensor.memory_config()`; L1 output share counted in `cb_budget_bytes` |
| transactions in flight at kernel exit (BB-03) | the next program starts immediately | writer ends with `noc_async_write_barrier()`; reader's last block has a read barrier |
| ragged last block | nominal push/pop vs valid tiles | FullBlock lifecycle; NoC transfers and math only on `last_block_extent` tiles |
