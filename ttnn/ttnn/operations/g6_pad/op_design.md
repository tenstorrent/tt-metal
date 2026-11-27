# Operation Design: g6_pad

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel, no arithmetic) |
| Goal | Constant padding at the high end of the trailing dims, bit-exact, as one device program. An extension call writes `c` into only the new rows of the last tile row of each slice, inside the input's storage, and returns a metadata view. Every other call assembles each output page once in L1 and writes it once. |
| Math | `D_i = d_i + h_i`; `out[j] = x[j]` if `j_i < d_i` for every `i`, else `c`, for `0 <= j_i < D_i` |
| References | Contract: `eval/prompts/g6_pad.txt`. Golden: `eval/golden_tests/g6_pad/{feature_spec.py,helpers.py}`. Blocking: `.claude/references/{blocking-model.md,l1-footprint-discipline.md,ttnn-cb-memory-fundamentals.md}`. Catalog (`ttnn/ttnn/operations/examples/master.md`): `noc_placement`, `width_split`, `double_buffer`, `tile_reorder`, `split_reader`, `bank_stagger`. Helpers: `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp`. NoC alignment: `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:375-380`. View path: `ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp:778-787`. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 2..4 on device, interleaved. Phase 0: bfloat16, TILE, DRAM, rank 3/4 | — | buffer address: common RT arg. Shape, dtype, layout, buffer type: program cache key |
| `padding` | sequence of `k` pairs `(low, high)`, positional or keyword | yes | `1 <= k <= r`. Each pair is a 2-element list/tuple of ints with `low == 0` and `high >= 0`. Pair `m` applies to dim `r-k+m` | — | Host-derived. Output shape, the segment table and per-core ranges all follow from the highs, which are part of the cache key |
| `value` | `int` or `float`, positional or keyword | no | any float for float dtypes. Within the integer range for int32/uint32, after truncation | `0.0` | `fill_bits`: common RT arg (32-bit). Not in the cache key |

Canonicalization, done once in the entry point before any check: `highs = [0]*(r-k) + [h for (_, h) in padding]` (length `r`). Every later quantity reads `highs`.

`c` conversion (host, torch semantics): bfloat16 uses `torch.tensor([value], dtype=torch.float32).to(torch.bfloat16)`, i.e. RNE to fp32 then RNE to bf16. The bit pattern goes in the low 16 bits of `fill_bits` (`-2**127` gives `0xFF00`). float32 uses its fp32 bits. int32/uint32 use `int(value)` (truncation), range-checked, then the two's-complement bits.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d_0, …, d_{r-1})`, `r ∈ {2,3,4}` (Phase 0: 3, 4) |
| Dtype | bfloat16 (Phase 0). float32, int32, uint32 are refinements |
| Layout | TILE (Phase 0). ROW_MAJOR is a refinement |
| Memory | DRAM interleaved (Phase 0). L1 interleaved is a refinement. Sharded is refused with `ValueError` |

### Output

| Property | Value |
|----------|-------|
| Shape | `(d_0+h_0, …, d_{r-1}+h_{r-1})` |
| Dtype | input dtype |
| Layout | input layout |
| Memory | input's memory config |
| Storage | `extension_band`: a metadata view of the input buffer (same buffer address, same padded shape). `new_storage_tiles`: a newly allocated buffer, and the input stays unchanged |

## Blocking Model

Tile-geometry symbols (tensor scope). Every formula uses `ceil` and is per slice, because each `(lead…, H, W)` image is tile-padded independently:

| Symbol | Definition |
|--------|------------|
| `elem_bytes`, `page_bytes` | element size; tile page size (`input_tensor.buffer_page_size()`: 2048 B bf16, 4096 B fp32/int32/uint32) |
| `face_row_bytes` | `16 * elem_bytes`; `face_bytes = 256 * elem_bytes` (32×32 tile, 16×16 faces; the host asserts `tile_shape == (32, 32)`) |
| `tensor_slices_in` / `tensor_slices_out` | `prod(d_0..d_{r-3})` / `prod(D_0..D_{r-3})`; 1 for rank 2 |
| `tensor_tile_rows_in` / `_out` | `ceil(d_{r-2}/32)` / `ceil(D_{r-2}/32)` |
| `tensor_tile_cols_in` / `_out` | `ceil(d_{r-1}/32)` / `ceil(D_{r-1}/32)` |
| `row_rem`, `col_rem` | `d_{r-2} mod 32`, `d_{r-1} mod 32` |
| `row_mask_end` | `min(32, D_{r-2} − 32·(tensor_tile_rows_in − 1))`. Rows `[row_rem, row_mask_end)` of the input's last tile row are logical output positions that must hold `c`. The range is empty when `h_{r-2} = 0` or `row_rem = 0` |
| `col_mask_end` | `min(32, D_{r-1} − 32·(tensor_tile_cols_in − 1))`. Same rule for columns `[col_rem, col_mask_end)` |
| `row_segments(a, b)` | Host function returning the byte ranges of tile rows `[a, b)` across all 32 columns. In face 0/1 the upper rows map to `[a·frb, min(b,16)·frb)` and `face_bytes + [a·frb, min(b,16)·frb)`. In face 2/3 the lower rows map to `2·face_bytes + …` and `3·face_bytes + …`. Adjacent ranges are merged. At most 3 ranges when `a > 0` (`frb = face_row_bytes`) |

### Axes

**Regime `extension_band`** (the band = the last tile row of every input slice):

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `slice` (leading dims flattened, extent `tensor_slices_in`) | independent: each slice's band is written without reading anything | folded with `band_tile_col` into `band_unit` (both are walked identically and are contiguous in band order) | — (see `band_unit`) | — | via `band_unit` | knob-turn |
| `band_tile_col` (extent `tensor_tile_cols_in`) | independent: each band tile's new rows are pure `c` writes | folded into `band_unit` | — | — | via `band_unit` | knob-turn |
| `band_unit` = `slice × band_tile_col` (extent `band_units = tensor_slices_in · tensor_tile_cols_in`) | independent | `band_block_tiles` (band tiles per write batch ending in one barrier) | `core_band_tiles` (the whole per-core run = 1 block). The source is a constant L1 buffer, so the block extent costs no L1 | host `make_plan()`, passed as RT arg | contiguous runs of `band_unit` over the full grid, `row_wise=True`, `num_cores = min(grid_cores, ceil(band_units / min_units_per_core))` | knob-turn |
| `tile_row` (extent `tensor_tile_rows_in`) | independent, but the contract pins it: only the last tile row of a slice may be touched | none: the block extent is 1, the last tile row | 1. Contract-mandated (no other tile row may be read or written). Not an L1 choice | the formula `(s·tensor_tile_rows_in + tensor_tile_rows_in − 1)·tensor_tile_cols_in + tc` | not assigned across cores (single row per slice) | none: fixed by contract |
| `row_in_tile` (new rows `[row_rem, row_mask_end)`) | independent | `band_segments = row_segments(row_rem, row_mask_end)` (≤ 3 byte ranges) | the whole new-row range in one write per segment. All 32 columns are written: columns `col_rem..31` of the last tile column are output tile padding, and skipping them would fragment the face rows | host `make_plan()`, passed as common RT args | not split (a sub-tile split across cores is the `band_segment_split` regime, rejected) | none |

**Regime `new_storage_tiles`** (output page space):

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `slice` (extent `tensor_slices_out`) | independent: each output page depends on at most one input page and on `c` | folded into `out_page` | — | — | via `out_page` | knob-turn |
| `tile_row` (extent `tensor_tile_rows_out`) | independent | folded into `out_page` | — | — | via `out_page` | knob-turn |
| `tile_col` (extent `tensor_tile_cols_out`) | independent | folded into `out_page` | — | — | via `out_page` | knob-turn |
| `out_page` = `slice × tile_row × tile_col` in interleaved page order (extent `out_pages`) | independent. No operand is shared between pages, so a rectangular block buys nothing over a run of consecutive pages. A run keeps the core's pages contiguous across the bank rotation | `block_tiles` (input-backed output pages per pipeline pass: one CB reserve/push/wait/pop, one read barrier, one write flush) | `min(max(1, max_core_input_pages), l1_block_cap)` = the whole per-core input-backed run in every production shape (1 page for the Phase 0 shapes, 14 for p1) | host `make_plan()`, single CT arg to reader and writer, and sizes `cb_assembled_tiles` | contiguous runs of `out_page` over the full grid, `row_wise=True`, `num_cores = min(grid_cores, ceil(out_pages / min_units_per_core))` | knob-turn |
| `row_in_tile`, `col_in_tile` (of boundary pages) | independent | mask ranges `row_segments(row_rem, row_mask_end)` and columns `[col_rem, col_mask_end)` | whole ranges, applied once per boundary page | host `make_plan()`, passed as common RT args | inside the page owner | none |

Shared knobs (one source each, in the program-descriptor module):

| Knob | Phase 0 value | Source | Derived from it |
|------|---------------|--------|-----------------|
| `grid` | `device.compute_with_storage_grid_size()` | queried per cache key | `grid_cores`, core ranges |
| `min_units_per_core` | 1 | module constant `MIN_UNITS_PER_CORE` | `num_cores` (both regimes) |
| `buffer_depth` | 2 | module constant `BUFFER_DEPTH` | `eff_depth = min(buffer_depth, max_num_blocks_per_core)`, `l1_block_cap` |
| `L1_OP_BUDGET_BYTES` | 1 347 776 (BB-05) | module constant | `l1_block_cap = floor((L1_OP_BUDGET_BYTES − page_bytes) / (page_bytes · buffer_depth))` |
| `band_block_tiles` | `max core_band_tiles` | `make_plan()` | `num_blocks_this_core = ceil(core_band_tiles / band_block_tiles)` |
| `block_tiles` | see above | `make_plan()` | CB capacity, `num_blocks_this_core = ceil(core_input_pages / block_tiles)` |

No knob is inlined in a kernel: every loop bound is a CT/RT arg derived from these.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_fill_source` (extension_band) | none (a constant buffer, never cycled) | 1 page | nothing to overlap: it is built once and read by every write |
| `cb_assembled_tiles` (new_storage_tiles) | `buffer_depth` → `eff_depth = min(buffer_depth, max_num_blocks_per_core)` | `eff_depth = 1` in every production shape (one block per core) | with ≥ 2 blocks per core, the reader loads block `k+1` while the writer drains block `k`. With one block there is nothing to overlap, so the second slot would be dead L1 |
| `cb_fill_page` (new_storage_tiles) | none | 1 page | constant `c` tile, sourced by every fill-only page write |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| NoC DRAM/L1 write alignment 16 B (`noc_parameters.h:376,380`) | band segment offset and length | Segment offsets and lengths are multiples of `face_row_bytes ≥ 32 B`. All segments read the fill source from offset 0 of a CB (L1-aligned). The host asserts `face_row_bytes % 16 == 0` | misaligned writes corrupt neighbouring bytes or hang |
| NoC DRAM read alignment 64 B (`noc_parameters.h:379`) | input reads | `new_storage_tiles` reads whole tile pages only (page bases are bank-aligned). Never sub-page reads | wrong bytes land in L1 |
| Segment table width | `row_in_tile` ranges | `row_segments(a, b)` with `a > 0` yields ≤ 3 ranges. The table holds 3 `(offset, bytes)` entries plus a count, and the host asserts `count ≤ 3` | out-of-bounds RT-arg reads |
| L1 budget (BB-05) | `block_tiles` | `block_tiles ≤ l1_block_cap`. The footprint is then `block_tiles·eff_depth·page_bytes + page_bytes ≤ 1 347 776` for every shape, so no shape is ever refused | launch failure in the model, where 1 347 776 B is all the op gets |
| CB ring wrap (`ttnn-cb-memory-fundamentals.md` §Ring-Wrap) | `cb_assembled_tiles` push quantum | Every block pushes and pops the nominal `block_tiles`, ragged last block included. Capacity = `block_tiles · eff_depth` | the pointer overshoots `fifo_limit` and silently corrupts neighbouring L1 |
| `fill_l1_range` contract (`l1_helpers.hpp:100-146`) | mask/fill ranges | starts and lengths are multiples of `elem_bytes` (element-aligned by construction) | ASSERT under `--dev`; silent under-fill otherwise |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `extension_band` | built | `layout == TILE and highs[i] == 0 for all i != r-2 and highs[r-2] > 0 and ceil(D_{r-2}/32) == ceil(d_{r-2}/32)` | `band_block_tiles` band tiles × `band_segments` (new rows × 32 cols) | Minimum at the DRAM boundary is every new logical output position written once and nothing read. Actual: 0 B read; `band_units · Σ band_segments` B written. Over the minimum only by columns `col_rem..31` of the last tile column, which are output tile padding written as part of whole face rows | One write barrier per block (Phase 0: once per core). No CB handshakes, no inits. The per-core fixed cost (launch, fill-source build of `fill_source_bytes`) is amortized by tiles per core: the grid-sync lamp |
| `new_storage_tiles` | built | `layout == TILE and not extension_band` | `block_tiles` input-backed output pages, consecutive in page order | Minimum: each input page crosses DRAM once, each output page once, no fill-only page read. Actual = minimum: input-backed pages read once, all output pages written once. Fill-only pages are written from `cb_fill_page` and never read | Per block, intended once each: one CB reserve/push + wait/pop, one read barrier, one write flush. Per core, intended once: the fill-page build and the final write barrier. Phase 0 runs 1 block per core |
| `new_storage_rows` | deferred: ROW_MAJOR is outside the Phase 0 SUPPORTED rectangle and arrives with the layout refinement (its 6 production cases are int32/uint32/bf16 ROW_MAJOR). No Phase 0 cell selects it. It reuses the `new_storage_tiles` skeleton unchanged (reader assembles, writer stores, fill-only pages written from a fill buffer): page = stick, `block_rows` sticks per block, the stick tail `[d_{r-1}·e, D_{r-1}·e)` is filled with `c` in L1, and fill-only sticks come from a one-stick fill buffer | `layout == ROW_MAJOR` | `block_rows` output sticks (cap: 64 B DRAM read alignment, so slot stride = aligned page size) | Minimum (input sticks once, output sticks once) | as `new_storage_tiles`, per stick block |
| `fill_then_copy` | rejected: a second full-tensor DRAM pass, forbidden by the contract. Superseded by `new_storage_tiles` (assemble in L1, write once) | — | — | +1 full output write + copy pass | — |
| `extension_full_tile_rmw` | rejected: reads the band tiles and writes them back whole (2 × band bytes + reads) where `extension_band` writes only the new rows. Superseded by `extension_band`, which is always legal because face-row writes meet the 16 B alignment | — | band tile | +band read, +written rows `0..row_rem-1` and `row_mask_end..31` | — |
| `extension_new_storage_copy` | rejected: forbidden when a slice has more than one tile row; for single-tile-row slices it moves a full read + full write per band tile. Superseded by `extension_band` | — | band tile | +1 read and +1 full write per band tile | — |
| `compute_kernel_assembly` (unpack → DEST → pack to mask) | rejected: a format detour that the bit-exact rule forbids (denormals/int32 through the unpacker), plus an extra CB hop. Superseded by RISC-V masking in L1 | — | — | same DRAM; +2 L1 passes | — |
| `band_segment_split` (one band tile's segments across ≤ 3 cores) | rejected: below the whole-tile granularity floor. ≤ 1.5 KB per core cannot pay the per-core launch and fill-build cost. Superseded by the whole-tile `band_unit` | — | one segment | same bytes | — |
| `fill_page_multicast` | rejected: `c` is a 4-byte RT arg and the fill buffer is generated in-core with zero NoC/DRAM bytes. A multicast would add traffic. Superseded by local generation | — | — | +NoC bytes | — |

**Regime-selection function** (host, exact; also the BB-08 plan report): `select_regime(layout, d, highs)` returns `extension_band` when the extension predicate above holds; `new_storage_tiles` when `layout == TILE`; `new_storage_rows` otherwise. Regime-pinned tests are required: the acceptance test pins `extension_band` (single- and multi-tile-row slices) and `new_storage_tiles` (`h` and `hw`) by shape and asserts the reported regime.

### Traffic ranking

No axis is dependent, so a split-the-dependent-axis-and-combine candidate does not exist. No operand is reuse-shared: each input page feeds exactly one output page, and `c` is a scalar generated per core. Every candidate therefore moves the same DRAM bytes, the ranking ties on bytes, and the tiebreakers decide (contiguous bank rotation, page-granular occupancy, row-wise core order).

| Regime | Candidate split | DRAM crossings (in / out) | Cross-core | Verdict |
|--------|-----------------|---------------------------|------------|---------|
| extension_band | contiguous runs of `band_unit` (slice × band col) | 0 / new rows once | 0 | **chosen**: fills the grid for p2–p4 (1940 units) and for multi-slice shapes (`[16,1,C]`: 16·48 units) alike |
| extension_band | by slice only | 0 / same | 0 | under-fills for every 1-slice shape (p2–p4: 1 slice) |
| extension_band | by band col only (`width_split`) | 0 / same | 0 | under-fills for `[8|16, 1, 12]` (1 col) |
| extension_band | sub-tile segments | 0 / same | 0 | rejected (`band_segment_split`) |
| new_storage_tiles | contiguous runs of `out_page` | in once / out once | 0 | **chosen**: page-granular occupancy, consecutive pages rotate DRAM banks |
| new_storage_tiles | by slice / tile row / tile col, or 2-D page blocks | same | 0 | same bytes. Under-fills (Phase 0 shapes: 1 slice, 8 rows, 4–8 cols) or strides page runs |
| both | broadcast fill buffer by mcast | same + mcast bytes | > 0 | rejected (`fill_page_multicast`) |

### Block schedule

**`extension_band`**: one band-writer kernel per core (RISCV_0 / NoC1):

```cpp
build_fill_source();                                    // once per core: fill_source_bytes of c into cb_fill_source
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    write_band_block(block_idx);                        // band_block_tiles band tiles × band_segments writes
    retire_band_block(block_idx);                       // one noc write barrier per block
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `build_fill_source` | `fill_source_bytes = max(segment bytes)` of `c` at offset 0 | `cb_fill_source` (whole kernel) | once per core |
| `write_band_block` | `band_block_tiles` tiles × ≤ 3 segments; segment `i` goes from `cb_fill_source + 0` to `page_addr(tile) + offset_i` | fill source | no barrier inside; writes self-throttle on NoC command-buffer space |
| `retire_band_block` | — | — | one write barrier per block. Phase 0: once per core, which is also the BB-03 end-of-kernel barrier |

**`new_storage_tiles`**: assembler reader (RISCV_1 / NoC0) and assembler writer (RISCV_0 / NoC1). Each core owns a contiguous run of output pages. Within it, *input-backed* pages flow through `cb_assembled_tiles` in page order, and *fill-only* pages never touch the CB.

```cpp
// reader                                               // writer
for (block_idx = 0; block_idx < num_blocks_this_core;   if (core_fill_pages > 0) {
     ++block_idx) {                                         build_fill_page();      // whole tile of c, once
    load_block(block_idx);                                   store_fill_pages();     // every fill-only page of the run
    mask_block(block_idx);                               }
    hand_off_block(block_idx);                           for (block_idx = 0; block_idx < num_blocks_this_core; ++block_idx)
}                                                            store_block(block_idx);
                                                         retire_writes();            // final write barrier (BB-03)
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `load_block` | `block_tiles` input-backed pages (ragged last block: fewer reads, nominal reserve) | the block's slots in `cb_assembled_tiles` | one `cb_reserve_back(block_tiles)` and one read barrier per block |
| `mask_block` | the block's boundary pages: rows `[row_rem, row_mask_end)` of pages in input tile row `tensor_tile_rows_in − 1`, and columns `[col_rem, col_mask_end)` of pages in input tile col `tensor_tile_cols_in − 1` | same slots | none: RISC-V stores, after the read barrier and before the push |
| `hand_off_block` | `block_tiles` | — | one `cb_push_back(block_tiles)` per block |
| `build_fill_page` | one page of `c` | `cb_fill_page` (whole kernel) | once per core, only if the run has fill-only pages |
| `store_fill_pages` | all fill-only pages of the run | fill page | no wait on the reader (stall shadow, below) |
| `store_block` | `block_tiles` | — | one `cb_wait_front(block_tiles)`, one `noc_async_writes_flushed`, one `cb_pop_front(block_tiles)` per block |
| `retire_writes` | — | — | one write barrier per core |

**Stall-shadow decisions.**
- The writer's only wait is `cb_wait_front` on the reader's block. Fill-only output pages do not depend on the input: in pad the fill region is independent of `x`, so their writes commute with every input-page write. They are issued before the first wait, and the fill-page build overlaps the reader's first reads. This is pure data movement with no floating-point reordering, so no precision gate applies.
- The reader's wait is its read barrier. No work is independent of the data in flight: masking needs the data. With ≥ 2 blocks per core and `eff_depth = 2`, the next block's reads overlap the writer's drain of the previous one. That is the depth knob.
- `extension_band`'s only wait is its final write barrier. Nothing remains to schedule into it.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization (both regimes) | Max participation leaves 1 page per core on the Phase 0 new-storage shapes (32–64 pages) and on short bands (`[1,3844,1536]`: 48 band tiles). Per-core launch and fill-build cost may exceed the work | `min_units_per_core ∈ {2, 4, 8}` (fewer cores, more units each) on cases 64/65 and 247 |
| Overlap (`new_storage_tiles`) | Block = whole run serializes each core's reads before its writes (p1: 14 pages) | `block_tiles = ceil(core_input_pages / {2,4})` with `buffer_depth = 2` (catalog `double_buffer`: no gain once DRAM-bound) |
| Second DM RISC for band writes (`extension_band`) | One RISC issues all ≤ 3 writes per band tile; if issue-bound, NCRISC sits idle (catalog `split_reader`) | Split each core's band run between BRISC/NoC1 and NCRISC/NoC0, each building its own fill source (no shared buffer, no handshake). Measure on p2–p4 |
| Fill build for `c == 0` (`extension_band`) | All 47 production extension calls have `c = 0`, and the RISC-V store loop builds up to 1.5 KB per core | Build `cb_fill_source` with `noc.async_write_zeros` (`tt_metal/hw/inc/api/dataflow/noc.h:849`) when `fill_bits == 0` |
| Bank alignment of runs (both) | A run length that is a multiple of the DRAM bank count starts every core on the same bank (catalog `bank_stagger`) | Rotate each core's walk start by `core_idx mod num_banks` |
| Load balance (`new_storage_tiles`) | A uniform page split hands write-only fill pages to the tail cores. p1 (refinement): 576 of 1536 pages are fill-only, contiguous at the end | Cost-weighted split: an input-backed page counts 2, a fill-only page 1 |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| E0 host view (`extension_band`) | metadata | `out = ttnn.reshape(input_tensor, ttnn.Shape(out_shape), input_tensor.padded_shape)`, which takes the volume-change TILE view branch (`reshape.cpp:778-787`: codegen refuses volume change at `reshape_codegen_supported.cpp:402-403`, then `ttnn::experimental::view`). No program, same buffer | Host guard: `out.buffer_address() == input_tensor.buffer_address()` and `out.padded_shape == input_tensor.padded_shape`, else `RuntimeError`. Never fall back to a copy |
| E1 fill source | input dtype bits in L1 | `fill_l1_range<elem_bytes>` into `cb_fill_source` | `fill_source_bytes` only |
| E2 band writes | raw bytes | `noc_async_write(fill_src, accessor.get_noc_addr(page) + offset_i, bytes_i)` on NoC1 | page = `(s·tensor_tile_rows_in + tensor_tile_rows_in − 1)·tensor_tile_cols_in + tc`. No other tile row is ever addressed |
| E3 dispatch | — | `ttnn.generic_op([out], program)`: single in-place io tensor | returns `out` |
| N1 input page → L1 (`new_storage_tiles`) | tile page, input dtype | reader: `noc_async_read(accessor.get_noc_addr(in_page), slot, page_bytes)` on NoC0, one barrier per block | only input-backed pages |
| N2 mask | bits | `fill_l1_range<elem_bytes>` over the row ranges (`row_segments(row_rem,row_mask_end)`) and per-face-row column ranges | before push; no unpack/pack |
| N3 hand-off | — | `cb_assembled_tiles` reader → writer | nominal `block_tiles` per push/pop |
| N4 output pages → DRAM | tile page | writer: `noc_async_write(slot, out_accessor.get_noc_addr(out_page), page_bytes)` on NoC1 | |
| N5 fill-only pages → DRAM | tile page of `c` | writer: `noc_async_write(cb_fill_page, out_accessor.get_noc_addr(out_page), page_bytes)` | no DRAM read for any fill-only page |

No inter-Tensix communication in any regime, built or deferred. Physical shards are not in TARGET; a sharded input is refused.

### TARGET axes as regime rows

| TARGET value | Regime(s) | Class | What changes |
|--------------|-----------|-------|--------------|
| dtype float32 / int32 / uint32 | extension_band, new_storage_tiles | knob-turn | `elem_bytes = 4`, `page_bytes = 4096`, `fill_l1_range<4>`, 32-bit `fill_bits`. Host `c` conversion (fp32 bits; int truncation + range check) |
| layout ROW_MAJOR | new_storage_rows | new regime on the same two-kernel skeleton (deferred row) | page = stick; tail fill per stick |
| memory_config "l1" | both TILE regimes | knob-turn | the accessor is built over an L1 interleaved buffer; the output takes the input's config |
| alignment "tile_aligned" | new_storage_tiles (an extension call is never tile-aligned: it needs `row_rem > 0`) | knob-turn | mask ranges are empty |
| rank 2 | both TILE regimes | knob-turn | `tensor_slices_* = 1`, no leading radices |
| padded_dims "leading" | new_storage_tiles | knob-turn | the leading-coordinate predicate marks whole fill-only slices (p1) |
| padded_dims "w" | new_storage_tiles | knob-turn | column mask plus fill-only tile columns |

### Host-side contract

| Item | Decision |
|------|----------|
| `validate()` (first line of `g6_pad`) | Step 1 raises `ValueError` for structural argument errors, in this order: sharded input; padding not a list/tuple of 1..r 2-element pairs of ints; `low != 0`; `high < 0`; `value` not int/float; integer `c` out of range. Step 2 builds the scenario dict (`shape`, `padding`, `value`, `dtype`, `layout`, `"dram"/"l1"`), applies `INPUT_TAGGERS`, and gates on `SUPPORTED` then `EXCLUSIONS` (`UnsupportedAxisValue` / `ExcludedCell`). Step 1 precedes step 2 because the golden refusal cases `sharded` (case 64) and `args` (cases 247/253) lie inside SUPPORTED |
| Taggers | `tag_padded_dims` reads only `padding[-1][1]` (dim -1) and `padding[-2][1]` (dim -2, when `k ≥ 2`). Positive means padded. It never indexes by `r`, so a malformed `k > r` scenario still tags. `tag_alignment`: `W % 32 != 0` gives `"w_non_aligned"`; otherwise `H % 32 != 0` gives `"h_non_aligned"`; otherwise `"tile_aligned"` |
| Phase 0 `SUPPORTED` | `dtype: [bfloat16]`, `layout: [TILE]`, `memory_config: ["dram"]`, `alignment: ["w_non_aligned","h_non_aligned"]`, `rank: [3,4]`, `padded_dims: ["h","hw"]`. `EXCLUSIONS = []` |
| Program cache (BB-07) | Python memo `key → (ProgramDescriptor, per-call patch slots)` with `key = (regime, logical shape, padded shape, tuple(highs), dtype, layout, buffer_type, grid)`. On a hit, only common RT args are patched (buffer addresses, `fill_bits`). `custom_program_hash` is set once from `ttnn.compute_program_descriptor_hash`. `value` is never in the key |
| Plan report (BB-08) | `describe_plan(input_tensor, padding) -> dict`, exported from the package (`from ttnn.operations.g6_pad import describe_plan`; the acceptance test pins regimes with it): regime, num_cores, core ranges, units per core (both groups), `block_tiles`/`band_block_tiles`, `eff_depth`, `band_segments`, L1 bytes per core. The entry point keeps the last plan in a module attribute |
| Role dump (BB-21) | `describe_plan` lists each core's role (`band_writer` / `assembler`) and unit range. Kernels wrap every phase and every blocking wait in `DeviceZoneScopedN` zones (`fill_build`, `band_writes`, `write_barrier`, `read_issue`, `read_barrier`, `mask`, `cb_reserve`, `cb_wait`, `block_writes`, `write_flush`) that compile to nothing in the plain build |
| Structural impossibilities | none to add (`INVALID = []` is correct) |

## Work Distribution

| Field | `extension_band` | `new_storage_tiles` |
|-------|------------------|---------------------|
| Work unit | a band tile (`band_unit`) | an output page (`out_page`); `block_tiles` input-backed pages per block |
| Grid | `device.compute_with_storage_grid_size()` | same |
| Split | `ttnn.split_work_to_cores(grid, band_units, row_wise=True)` (catalog `noc_placement`: row-wise spread) | `ttnn.split_work_to_cores(grid, out_pages, row_wise=True)` |
| Per-core work | `core_band_tiles` = g1/g2 counts; RT args `start_unit`, `core_band_tiles` | RT args `start_page`, `core_out_pages`, plus host-computed `core_input_pages` and `core_fill_pages` |
| Remainder | split_work_to_cores' two groups; all formulas `ceil`, per slice | the same. The ragged last block keeps nominal push/pop counts |
| Unit → page | `s = u / tensor_tile_cols_in`, `tc = u % tensor_tile_cols_in` → `(s·tensor_tile_rows_in + tensor_tile_rows_in − 1)·tensor_tile_cols_in + tc` | `tc = p % tensor_tile_cols_out`, `tr = (p / tensor_tile_cols_out) % tensor_tile_rows_out`, `s_out = p / (tensor_tile_rows_out·tensor_tile_cols_out)`. Leading coords come from `s_out` in the output radices `D_lead`. The page is input-backed iff every leading coord `< d_lead`, `tr < tensor_tile_rows_in` and `tc < tensor_tile_cols_in`. Then `s_in` = the coords in radices `d_lead` and `in_page = (s_in·tensor_tile_rows_in + tr)·tensor_tile_cols_in + tc` |
| Core ranges (BB-17) | row-wise groups give ≤ 3 rectangles | same |
| Kernels | band writer only | reader + writer; no compute kernel |

Production geometry. Units/core assume a 110-core grid; `make_plan()` derives them from the queried grid:

| Case | Regime | Units | Units / core | Segments (B) per tile |
|------|--------|-------|--------------|------------------------|
| p2 `[1,1,2,62080]`→32 | extension_band | 1940 | 17–18 | 448, 1472 |
| p3 `[1,1,8,62080]`→16 | extension_band | 1940 | 17–18 | 256, 256 |
| p4 `[1,1,8,62080]`→32 | extension_band | 1940 | 17–18 | 256, 1280 |
| `[1,3844,1536]`→3872 | extension_band | 48 | 1 | 384, 1408 |
| `[16,1,1536]`→32 | extension_band | 768 | 6–7 | 480, 1504 |
| `[1,3844,12]`→3872 | extension_band | 1 | 1 | 384, 1408 |
| `[1,1,196,196]`→256 | new_storage_tiles | 64 pages (49 input-backed, 15 fill-only) | 1 | — |
| `[1,196,128]`→256 | new_storage_tiles | 32 pages (28 input-backed, 4 fill-only) | 1 | — |
| p1 `[5,12,128,128]`→8 (refinement) | new_storage_tiles | 1536 pages (960 / 576) | 13–14 | — |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_fill_source` (extension_band) | 0 | `page_bytes` | 1 | spans `row_in_tile` (live: `fill_source_bytes ≤ page_bytes`); streams `band_unit`, `tile_row` | input dtype | writer (`build_fill_source`) | writer (NoC source) | whole kernel |
| `cb_assembled_tiles` (new_storage_tiles) | 0 | `page_bytes` | `block_tiles · eff_depth` | spans `out_page` → `block_tiles`, ×`eff_depth` | input dtype | reader | writer | per block, whole kernel |
| `cb_fill_page` (new_storage_tiles) | 1 | `page_bytes` | 1 | spans one page of `c`; streams `out_page` | input dtype | writer (`build_fill_page`) | writer (NoC source) | whole kernel |

The two scratch CBs are reserved once and never pushed (`cb_reserve_back(…, 1)` + `get_write_ptr`). Producer and consumer are the same kernel. Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| E1 | `build_fill_source` | `fill_source_bytes` | `fill_l1_range` | — | `cb_fill_source`, 1 (reserved, not pushed) | holds `c` for the whole kernel |
| E2 | `write_band_block` | `band_block_tiles` × `band_segments` | raw NoC | `cb_fill_source` (read as NoC source) | DRAM (input buffer = output view) | unchanged |
| E3 | `retire_band_block` | — | raw NoC | — | — | all writes acked |
| N1 | `load_block` | `block_tiles` | raw NoC + CB | DRAM input | `cb_assembled_tiles`, reserve `block_tiles` | slots hold raw input pages |
| N2 | `mask_block` | boundary pages of the block | `fill_l1_range` | `cb_assembled_tiles` (reserved, not yet pushed) | same slots | padded rows/cols hold `c` |
| N3 | `hand_off_block` | `block_tiles` | CB | — | `cb_assembled_tiles`, push `block_tiles` | visible to writer |
| N4 | `build_fill_page` | 1 page | `fill_l1_range` | — | `cb_fill_page`, 1 (reserved, not pushed) | holds `c` |
| N5 | `store_fill_pages` | all fill-only pages of the run | raw NoC | `cb_fill_page` | DRAM output | unchanged |
| N6 | `store_block` | `block_tiles` | raw NoC + CB | `cb_assembled_tiles`, wait `block_tiles` | DRAM output | popped `block_tiles` after flush |
| N7 | `retire_writes` | — | raw NoC | — | — | all writes acked (BB-03) |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| E1, N2, N4 | helper | `dataflow_kernel_lib::fill_l1_range<val_size>` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:100-101` | `val_size = elem_bytes` (CT); `(start_addr, n_bytes, fill_bits)` | — | scratch / slot | `n_bytes` = segment / range length (derived from `row_segments`, mask ranges) |
| E2, N4–N6 writes | raw_api | `noc_async_write` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:876` | `(src_l1, dst_noc_addr, bytes)` | fill source / slot | DRAM | `band_block_tiles`, `block_tiles` (loop extents) |
| E2, N1, N4–N6 addressing | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | page id + segment offset | — | — | — |
| N1 reads | raw_api | `noc_async_read` | `dataflow_api.h:547` | `(src_noc_addr, slot_l1, page_bytes)` | DRAM | `cb_assembled_tiles` | `block_tiles` |
| E3, N7 | raw_api | `noc_async_write_barrier` | `dataflow_api.h:1951` | — | — | — | once per block / core |
| N6 | raw_api | `noc_async_writes_flushed` | `dataflow_api.h:1980` | — | — | — | once per block |
| N1 | raw_api | `noc_async_read_barrier` | `dataflow_api.h:1921` | — | — | — | once per block |
| N1, N3, N6 | raw_api | `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` / `get_write_ptr` / `get_read_ptr` | `dataflow_api.h:406 / 208 / 479 / 260 / 325 / 346` | quantum `block_tiles` | — | — | `block_tiles` |
| fill (lamp) | raw_api | `Noc::async_write_zeros` | `tt_metal/hw/inc/api/dataflow/noc.h:849` | only on the `c == 0` lamp | — | `cb_fill_source` | — |

**Helpers considered and rejected** (for every `raw_api` row):

| Raw use | Candidate helper | Mismatch (file:line) |
|---------|------------------|----------------------|
| sub-page band writes, page reads and writes | `dataflow_kernel_lib::fill_tile_column` | `tile_column_dataflow.hpp:69-70`: RISC-V stores into an L1 tile column; no NoC transfer to an interleaved tensor |
| same | `local_copy_helpers_dataflow.hpp` (`read_with_state`, `set_read_trid`) | `local_copy_helpers_dataflow.hpp:9,94-95`: self-aimed L1→L1 reads only (trid helpers `:147,158` are the barrier-granularity lamp's tools, not a page mover) |
| same | `mcast/kernel/mcast_args.hpp` | multicast between Tensix cores; this op has no cross-core traffic |
| same | compute-side helpers (`tilize`/`untilize`/`eltwise_*`/`reduce`) | compute kernels unpack and pack, which is the format detour the bit-exact rule forbids; this op has no compute kernel |
| CB handshakes | none: no dataflow helper wraps raw `cb_*` for a reader→writer page hand-off | — |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| The view relies on `ttnn.reshape`'s volume-change TILE branch | `reshape.cpp:760` `this_is_view` is false when the old dim -2 is not a multiple of 32. The view comes only from the `:778-787` branch, which upstream marks as legacy. `ttnn.view`/`ttnn.experimental.view` TT_FATAL on volume change (`data_movement/view/view.cpp:24`, `reshape_common.cpp:58`). The single-shape `ttnn.reshape(t, list)` form resolves to the `SmallVector` overload and hits the same fatal | Always call the two-shape form with `ttnn.Shape(out_shape)` and `input_tensor.padded_shape`. Host guard on buffer address and padded shape. The acceptance test counts programs. If the branch ever goes, add a nanobind overload binding `ttnn::experimental::view(t, logical, padded)` (`experimental/reshape/view.cpp:12`). Never copy |
| Producer leftovers in the input's tile padding (rule 2) | `new_storage_tiles` copies whole tiles. With zero-padded test inputs and `c = 0`, a missing mask passes | The mask covers rows `[row_rem, row_mask_end)` and cols `[col_rem, col_mask_end)` of every boundary page. The acceptance test poisons padding with 1000.0 |
| Touching a non-last tile row (rule 3) | An off-by-one in the band page formula silently rewrites real data rows in multi-tile-row slices | The page formula fixes `tile_row = tensor_tile_rows_in − 1`. The acceptance test diffs the raw padded buffer of every earlier tile row |
| Bit exactness | Any unpack/pack or float conversion breaks int32/uint32/denormals | No compute kernel. Masking and fill are RISC-V integer stores of `fill_bits`. bf16 `c` comes from torch fp32 → bf16 RNE on the host (`-2**127` → `0xFF00`, `0.1` → torch's value) |
| Cache key vs per-call data (BB-07) | The same shape is called with different `value` and padding (loose case 253 → `value=7.0` → 291) | `fill_bits` and addresses are common RT args patched per call. Highs are in the key |
| Ragged blocks and ring wrap | Per-core `core_input_pages` differ, but `block_tiles` is uniform | Nominal `block_tiles` push/pop per block. Capacity = `block_tiles·eff_depth` |
| Race on hand-off (BB-06) | Masking writes slots after the reads land | Order: read barrier → mask → `cb_push_back`. The writer pops only after `noc_async_writes_flushed` |
| Transactions left in flight (BB-03) | The next model program starts immediately | Every writer ends with `noc_async_write_barrier`. The reader ends after its last read barrier |
| Odd column boundary in bf16 | `col_rem` odd puts mask ranges at 2-byte alignment | `fill_l1_range<2>` handles 16-bit head/tail stores; ranges are element-aligned |
| Cores without input-backed pages | Fill-only tail runs (p1, tile row 7 of the 256 cases) | `num_blocks_this_core = 0`: the reader exits and the writer only writes fill pages. CB config stays uniform across the core range |
| Refusals inside SUPPORTED | The golden `sharded` (case 64) and `args` (247, 253) refusals lie in SUPPORTED cells | Structural `ValueError` checks run before the support gate. Taggers never crash on `k > r` |
