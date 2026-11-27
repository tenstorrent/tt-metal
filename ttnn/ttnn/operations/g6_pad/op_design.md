# Operation Design: g6_pad

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no compute kernel; no arithmetic; bit-exact copy + constant fill) |
| Goal | Constant high-end padding of the trailing `k` dims of a rank-2/3/4 tensor, one device program per call. A **tile_padding_extension** call writes `c` into the new rows of the last tile row of each slice **in place** and returns a metadata-only view of the input's storage; every other call assembles each output page once in L1 (input tile + fill patches, or a pure fill page) and writes it once. |
| Math | `D_i = d_i + h_i`; `out[j] = x[j]` if `j_i < d_i ∀i`, else `c` (`c` = `value` converted to dtype by torch: bf16 = RNE of fp32, int = truncation) |
| References | `eval/golden_tests/g6_pad/feature_spec.py` (TARGET / INPUTS / LOOSE_CASES), `eval/golden_tests/g6_pad/helpers.py` (oracle `pytorch_g6_pad`, `storage_class`, exact grading), `eval/op_template.py` (registry model), `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `ttnn/ttnn/operations/examples/master.md` (`noc_placement`, `double_buffer`, `bank_stagger`, `tile_reorder`, `width_split`), `ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation_types.hpp:50-52` (single-tensor in-place `io_tensors`), `ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.hpp:44-46` (cache hash excludes runtime-arg values), `ttnn/core/tensor/tensor_ops.cpp:533-547` (`tt::tt_metal::view`, metadata-only) |

Integrity note: no pre-existing implementation of this op was read; this design is from the spec, the golden suite and first principles.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 2..4, interleaved (DRAM Phase 0; L1 refinement), not sharded | — | buffer address = RT |
| `padding` | sequence of `(low, high)` int pairs | yes (positional or keyword) | `1 ≤ k ≤ r` pairs; `low == 0`; `high ≥ 0`; pairs aligned to the trailing `k` dims | — | shape-derived quantities: CT (structural, in cache key); per-core ranges: RT |
| `value` | `float` or `int` | no (positional or keyword) | any float for float dtypes; int in dtype range for int32 / uint32 | `0.0` | `fill_bits` (32-bit pattern of `c`): **RT** (never CT: value changes per call on one shape, LOOSE case `loose_same_shape_two_values`) |

`fill_bits` is computed on the host with torch: `torch.tensor(value, dtype=torch_dtype)` then reinterpret the bits (`view(torch.int16)` for bf16 → `(b << 16) | b`; `view(torch.int32)` for 32-bit dtypes). `-1.7014118346046923e38` → bf16 `0xFF00` (exactly `-2^127`).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d_0, …, d_{r-1})`, `r ∈ {2,3,4}` (Phase 0: 3, 4) |
| Dtype | bfloat16 (Phase 0); float32, int32, uint32 (refinements) |
| Layout | TILE (Phase 0); ROW_MAJOR (refinement) |
| Memory | DRAM interleaved (Phase 0); L1 interleaved (refinement); sharded → `ValueError` |

### Output

| Property | Value |
|----------|-------|
| Shape | `(d_0 + h_0, …, d_{r-1} + h_{r-1})` |
| Dtype | = input |
| Layout | = input |
| Memory | = input placement. **tile_padding_extension**: the input's own buffer, viewed at the output logical shape with the input's padded shape. **new_storage**: a freshly allocated interleaved tensor with the input's memory config |

## Blocking Model

Everything below is downstream of this section. Semantics: `.claude/references/blocking-model.md`.

The op has **no dependent axis** (no value of the output depends on more than one input element) and **no reuse-shared input operand** (every input element feeds exactly one output position). The only shared quantity is the scalar `c`, which every core materializes locally from an RT arg — zero bytes moved. Hence every split is a knob-turn and the design question is entirely *which pages are touched, how often, and how coarsely the per-core run is handed to the NoC*.

Notation (all from the shapes; host-computed once, single source = the plan dict of the entry point, see Work Distribution):

| Symbol | Definition |
|--------|------------|
| `tile_h = tile_w = 32`, `face = 16` | tile geometry |
| `elem_bytes` | 2 (bf16), 4 (fp32 / int32 / uint32) |
| `row_bytes = face · elem_bytes` | bytes of one face row (bf16: 32 B) |
| `face_bytes = face · face · elem_bytes`, `tile_bytes = 4 · face_bytes` | bf16: 512 B / 2048 B |
| `in_tiles_h = ceil(d_{r-2}/32)`, `in_tiles_w = ceil(d_{r-1}/32)` | per-slice input tile counts (alignment-aware, per image) |
| `out_tiles_h = ceil(D_{r-2}/32)`, `out_tiles_w = ceil(D_{r-1}/32)` | per-slice output tile counts |
| `in_slices = Π_{i<r-2} d_i`, `out_slices = Π_{i<r-2} D_i` | slices (leading dims folded; rank 2 → 1) |
| `edge_row = d_{r-2} mod 32`, `edge_col = d_{r-1} mod 32` | first input-tile-padding row / column of the last input tile row / column (0 = aligned, no patch) |
| `band_row_begin = edge_row`, `band_row_end = D_{r-2} − 32·(in_tiles_h − 1)` | extension only: rows of the last tile row that must become `c` (`1 ≤ begin < end ≤ 32`) |

### Axes

Regime **extension_band** (predicate in Regimes):

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| slice (leading dims, folded) | independent — no output value of one slice depends on another slice | `band_block_tiles` (folded with tile-column into the flat band index `band_idx = slice · in_tiles_w + wt`) | whole per-core band range (one block) | host plan `band_block_tiles = core_band_tiles` | contiguous `band_idx` ranges over the grid (`split_work_to_cores`, `row_wise=True`) | knob-turn |
| tile column `wt` | independent — columns are disjoint | folded into `band_block_tiles` (above) | whole per-core range | same | same flat split | knob-turn |
| tile row `ht` | independent in the math, but **not split and not walked**: the contract forbids touching any tile row but `in_tiles_h − 1` | none (extent fixed at 1 by the contract — the only legal tile row) | 1 (contract-pinned, not an L1 choice) | constant from the contract; `last_tile_row = in_tiles_h − 1` host-derived | not assigned (one row per slice) | — (contract) |
| rows inside the band tile | independent | `[band_row_begin, band_row_end)` — whole row range in one block | all band rows of the tile (coalesced segments, ≤ 3 per tile, 2 in every production case) | host plan `band_segments` (offset, bytes) list, CT | not split (sub-tile: splitting would only add NoC transactions) | — |

Regime **new_storage_tiles**:

| Axis | Character | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|-----------|-------------|---------------|-------------|-----------------|--------------|
| output slice (leading dims, folded) | independent — each output slice is either a copy of one input slice or pure fill | folded into `block_tiles` over the flat output tile index `t = (slice · out_tiles_h + ht) · out_tiles_w + wt` | whole per-core range (one block) unless the L1 cap clamps it | host plan `block_tiles = min(core_max_tiles, block_tiles_cap)` | contiguous `t` ranges over the grid (`split_work_to_cores`, `row_wise=True`) | knob-turn |
| output tile row `ht` | independent | folded into `block_tiles` | whole per-core range | same | same flat split | knob-turn |
| output tile column `wt` | independent | folded into `block_tiles` | whole per-core range | same | same flat split | knob-turn |
| elements inside a tile | independent | whole tile (page = transfer unit) | 1 page | page = tile (NoC/DRAM page) | not split | — |

Folding is legal: the three outer axes are treated identically (each output tile is classified by its own coordinates — copy, copy+patch, or fill-only) and are contiguous in output page order, so the per-core run is one rectangle in flat page space.

Intermediate data: the only data a stage produces is the assembled output page in `cb_input_tiles` (same axes as the output tile index; it inherits the split above — nothing is collected on a designated core, there is no coordinator role, every core does its own pages end to end).

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` (new_storage_tiles) | `input_depth_blocks` | `min(2, num_blocks_max)` → **1** whenever the per-core run is one block (every Phase 0 shape), 2 when the L1 cap splits a run | overlap of block `i+1` reads with block `i` patch+writes; worthless with one block, so not allocated |
| `cb_fill_tile` (both regimes) | none (constant source page, read by NoC writes, never cycled) | 1 page | — |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| NoC write alignment (Blackhole `NOC_DRAM_WRITE_ALIGNMENT_BYTES = 16`, `NOC_L1_WRITE_ALIGNMENT_BYTES = 16`, `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:376,380`) | band segment start/length; source offset in `cb_fill_tile` | segment offsets are multiples of `row_bytes` (≥ 32 B, 16-aligned); source = `fill_base + (dst_offset mod 16)` with `fill_base` 16-aligned → equal mod 16. Every source offset used must stay within `cb_fill_tile` page (segment ≤ `tile_bytes − face_bytes`) | misaligned NoC write: corrupt bytes / hang |
| NoC DRAM read alignment (`NOC_DRAM_READ_ALIGNMENT_BYTES = 64`, same file :379) | new_storage reads | reads are whole tiles to CB page starts (CB pages are 64-aligned since `tile_bytes` is a multiple of 64) | — |
| Column patch granularity (`edge_col` is 2 B granular for bf16) | input-tile-padding columns patch | done by `fill_l1_range` (RISC-V L1 stores, alignment-safe head/tail) in the writer, **not** NoC writes (a NoC transfer cannot start at a 2 B offset) | misaligned NoC write |
| Extension contract | tile rows touched | exactly one tile row per slice (`in_tiles_h − 1`); band rows `[band_row_begin, band_row_end)` only | contract violation (golden `untouched_rows`, storage checks) |
| L1 budget (BB-05: 1347776 B/core) | `block_tiles` | `block_tiles_cap = floor((1347776 − fill_page_bytes) / (input_depth_blocks_if_split · tile_bytes))` (bf16 ≈ 328 tiles, fp32 ≈ 164) | OOM / refused launch |
| Kernel RT-arg count | none — per-core args are a fixed handful (range start, count, fill bits, buffer address) | — | — |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **extension_band** | **built** | `layout == TILE` ∧ `h_{r-2} > 0` ∧ `h_i == 0 ∀ i ≠ r-2` ∧ `ceil(D_{r-2}/32) == ceil(d_{r-2}/32)` (= `helpers.storage_class == "tile_padding_extension"`). 34 of the 40 Phase 0 cases, p2–p4 | `band_block_tiles` consecutive band tiles (= whole per-core range) × `band_segments` | **Minimum** at the DRAM boundary: input read 0 times; written bytes = `in_slices · in_tiles_w · 2 · (band_row_end − band_row_begin) · row_bytes` — only positions that must change (the last tile column's padding columns ride along inside the same face rows, no extra transaction). No allocation, no output page written outside the band | one write barrier per core instead of per tile; fill page materialized once per core; per-core RT-arg / launch cost amortized over `core_band_tiles` |
| **new_storage_tiles** | **built** | `layout == TILE` ∧ ¬extension_band. 6 Phase 0 cases (`[1,L,128]`, `[1,1,L,L]`, L=185/196/207); later p1 and the rank-2 uint32 `[1,L]` cases | `block_tiles` consecutive output tiles (flat index), one block per core in every INPUTS shape | **Minimum**: every input tile crosses DRAM once (read), every output tile once (write); fill-only output tiles are never read (written from `cb_fill_tile`) | one read barrier + one CB handshake per block instead of per tile; fill-page set-up once per core |
| new_storage_rows | deferred — the 6 ROW_MAJOR cases (int32/uint32 `[5,8192]`, `[5,1]`, `[1,1,1,1]`, `[1,1,1,8]`, bf16 `[5,3,2560]`) are outside Phase 0 SUPPORTED (layout ROW_MAJOR, dtypes int32/uint32); they arrive with the layout refinement, which adds a row-page variant of the same reader/writer split | `layout == ROW_MAJOR` | `block_rows` consecutive output rows (page = one row of `D_{r-1} · elem_bytes`) | minimum: each input row read once into the start of a CB page (64-aligned), tail `[d_{r-1}, D_{r-1})` filled by RISC stores in L1, written once; fill-only rows written from a fill row page, never read | one barrier per block; Reachability: same plan dict, same flat-range split, same `cb_input_tiles` slot discipline with page = row |
| extension_rmw (read last tile row, patch in L1, write whole tiles back) | rejected — superseded by extension_band, which writes ≤ the same bytes and reads none (RMW: band read 1× + band written 1× at full tile size) | — | — | 2× tile bytes per band tile vs ≤ 1.5× tile bytes written-only | — |
| extension_copy_to_new_storage | rejected — forbidden by the contract where `in_tiles_h > 1`; where `in_tiles_h == 1` it would read + write every band tile at full size plus allocate, superseded by extension_band on every shape | — | — | 1× read + 1× write of the whole tensor | — |
| fill_then_copy (fill all output, then copy input over it) | rejected — forbidden (second full DRAM pass); new_storage_tiles assembles each page once in L1 | — | — | output written ~2× | — |
| fill_page_broadcast (one core builds the fill page and multicasts it) | rejected — the fill page is generated by RISC stores from one RT arg in < 1 µs locally; a multicast adds NoC traffic and a semaphore rendezvous for zero DRAM saving | — | — | adds `tile_bytes × ncores` NoC bytes | — |
| split a dependent axis + combine | not applicable — the op has no dependent axis (no output value spans more than one input element) | — | — | — | — |

**Regime-selection function** (host, exact; also the plan name reported per BB-08):

```
def select_regime(layout, in_shape, out_shape):
    if layout == ROW_MAJOR: return "new_storage_rows"          # deferred: refused by validate() in Phase 0
    highs = [D - d for D, d in zip(out_shape, in_shape)]
    only_h = highs[-2] > 0 and all(h == 0 for i, h in enumerate(highs) if i != len(highs) - 2)
    same_tile_rows = ceil(in_shape[-2] / 32) == ceil(out_shape[-2] / 32)
    return "extension_band" if (only_h and same_tile_rows) else "new_storage_tiles"
```

Regime-pinned tests are required: the acceptance test parametrizes production cases of both built regimes and pins the regime by its observable storage (extension_band → output buffer address == input buffer address and equal padded shape; new_storage_tiles → a different buffer), independent of the grid size.

### Traffic ranking

| Candidate split | DRAM crossings | Cross-core traffic | Rank |
|-----------------|----------------|--------------------|------|
| extension: flat band-tile range per core (chosen) | 0 reads; band rows written once | none | 1 (minimum) |
| extension: split by slice only | same bytes | none | 2 — same traffic, but a 1-slice tensor (p2–p4: 1 slice × 1940 band tiles) would run on one core (`width_split` catalog entry: wide-short split by width) |
| new_storage: flat output-tile range per core (chosen) | input 1×, output 1× | none | 1 (minimum) |
| new_storage: split by input tile then derive outputs | input 1×, output 1×, but fill-only output tiles have no input owner → second assignment pass needed | none | 2 — same bytes, two work lists |
| new_storage: per-core resident slice + leading-dim fill | same bytes | none | 3 — no axis to reuse; nothing gained by residency |
| dependent-axis split + combine | — | — | n/a (no dependent axis) |

Cheapest-traffic split = the implemented one in both regimes. Occupancy (core count) is a separate decision (Work Distribution) and does not change bytes.

### Block schedule

extension_band (one kernel role, `band_writer`):

```cpp
make_fill_page();                                   // once per core: RISC stores of fill_bits into cb_fill_tile
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // num_blocks_this_core = 1 in Phase 0
    write_band_block(block_idx);                    // for each band tile of the block: issue every band segment
}                                                   // as a NoC write from cb_fill_tile to (page addr + seg offset)
drain_writes();                                     // noc_async_write_barrier once per core (BB-03)
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `make_fill_page` | 1 page | `cb_fill_tile` for the whole kernel | once per core |
| `write_band_block` | `band_block_tiles × band_segments` | `cb_fill_tile` | no barrier inside; address generation via `TensorAccessor` per band tile (`page = slice · in_tiles_h · in_tiles_w + (in_tiles_h−1) · in_tiles_w + wt`) |
| `drain_writes` | — | — | one write barrier per core |

new_storage_tiles (reader + writer):

```cpp
make_fill_page();                                   // writer, once per core
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);       // reader: reserve block_tiles slots; for each source-bearing output tile k of the block,
                                 // async-read input tile into slot k; ONE read barrier; push block_tiles (nominal)
    write_fill_tiles(block_idx); // writer: issue writes of every fill-only output tile of the block from cb_fill_tile
                                 // -- scheduled BEFORE the wait: it does not depend on the input (stall shadow)
    patch_block(block_idx);      // writer: wait block_tiles; in each edge slot, RISC-store fill_bits into rows
                                 // [edge_row, 32) (last input tile row) and/or cols [edge_col, 32) (last input tile col)
    store_block(block_idx);      // writer: async-write each source-bearing slot to its output page; writes_flushed; pop block_tiles
}
drain_writes();                  // writer: noc_async_write_barrier (BB-03); reader: read barrier already per block
```

| Operation | Block shape | Resident across it | Intended frequency |
|-----------|-------------|--------------------|--------------------|
| `make_fill_page` | 1 page | `cb_fill_tile` whole kernel | once per core |
| `load_block` | `block_tiles` slots (slot `k` ↔ k-th output tile of the block; fill-only slots left unwritten) | — | 1 reserve, 1 read barrier, 1 push per block |
| `write_fill_tiles` | fill-only tiles of the block | `cb_fill_tile` | no barrier; runs while `load_block` is in flight |
| `patch_block` | edge slots of the block (≤ 1 edge row segment + 1 edge column per slice band) | slots of the block | 1 wait per block; pure L1 stores, no NoC |
| `store_block` | source-bearing slots of the block | — | 1 `writes_flushed` + 1 pop per block |
| `drain_writes` | — | — | once per core |

Stall-shadow check: the writer's only wait is on `cb_input_tiles`; the fill-only tile writes do not depend on it and are moved into that window (a pure data move — no floating-point order changes, no precision gate needed). In extension_band nothing waits (write-only).

Tile classification (both kernels compute the same predicate from CT args; the host never enumerates tiles):

| Output tile (slice coords `j`, `ht`, `wt`) | Class | Source page |
|---|---|---|
| `j_i < d_i ∀ leading i` ∧ `ht < in_tiles_h` ∧ `wt < in_tiles_w` | source-bearing | `(in_slice(j) · in_tiles_h + ht) · in_tiles_w + wt` (`in_slice` = mixed-radix over input leading dims) |
| otherwise | fill-only | none |
| source-bearing ∧ `ht == in_tiles_h − 1` ∧ `edge_row ≠ 0` | + row patch `[edge_row, 32)` | |
| source-bearing ∧ `wt == in_tiles_w − 1` ∧ `edge_col ≠ 0` | + column patch `[edge_col, 32)` | |

Patching is unconditional on whether `D` grows along that dim: positions past `D` are the output's own tile padding (may hold anything), so one rule covers every case and the poisoned input padding never leaks.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| grid synchronization (extension_band) | p2–p4: 1940 band tiles / ~110–130 cores ≈ 15–18 tiles, 2 small writes each; per-core launch + RT-arg cost may dominate | `min_band_tiles_per_core` ∈ {1 (default), 32, 64} → fewer cores |
| grid synchronization (new_storage_tiles) | `[1,L,128]`: 32 output tiles → 32 cores × 1 tile | `min_tiles_per_core` ∈ {1 (default), 4, 8} |
| overlap (new_storage_tiles) | one block per core serializes reads before patch+writes on a core (p1: ~14 fp32 tiles/core) | `block_tiles` ∈ {4, 8} with `input_depth_blocks = 2` (`double_buffer` catalog: block 4–8, CB = 2·block) |
| band writes on one RISC | extension is pure write-issue; one BRISC may be issue-bound at 2 writes/tile | split each core's band range across both DM RISCs (RISCV_0/NOC1 and RISCV_1/NOC0), each with its own `cb_fill_tile` (one consumer per CB) (`split_reader` catalog entry, mirrored for writes) |
| load balance (new_storage_tiles) | equal output-tile counts give copy cores 2× the bytes of fill-only cores (p1: 960 copy + 576 fill tiles) | cost-weighted split (copy = 2, fill = 1) or two separate even splits (source-bearing list, fill-only list) |
| bank alignment | contiguous ranges whose length is a multiple of the 8 Blackhole DRAM banks start every core on the same bank | `bank_stagger`: rotate each core's walk start by `core mod 8` |
| core placement | `split_work_to_cores` default `row_wise=False` gives column lines | default here is `row_wise=True` (`noc_placement` catalog); measure diagonal on Blackhole |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| `fill_bits` → `cb_fill_tile` | dtype bit pattern, one tile page (bf16: 2048 B) | `fill_l1_range<elem_bytes>` (`l1_helpers.hpp:101`) in the writer | No DRAM read, no compute. Every NoC write sourcing it happens after the stores (same RISC, program order) |
| extension: `cb_fill_tile` → DRAM band rows | raw bytes | `noc_async_write` per band segment (src offset ≡ dst offset mod 16), `TensorAccessor` page address of the last tile row + segment offset | input never read; output = view of input storage |
| new_storage: DRAM input tile → `cb_input_tiles` slot | TILE page, input dtype, bytes unchanged | `noc_async_read` (`TensorAccessor`), reader on RISCV_1 / NOC0 | each input tile read exactly once globally |
| new_storage: slot patch | in-place in L1 | writer `fill_l1_range` of `fill_bits` into rows `[edge_row,32)` and/or columns `[edge_col,32)` (face-aware: rows 0–15 in faces 0/1, 16–31 in faces 2/3; columns 0–15 in faces 0/2, 16–31 in faces 1/3) | bit-exact: no unpack/pack |
| new_storage: slot / fill page → DRAM output | TILE page | `noc_async_write` (`TensorAccessor`), writer on RISCV_0 / NOC1 | every output page written exactly once |

TARGET placement axis: `memory_config ∈ {"dram","l1"}` — both interleaved; `"l1"` is a **knob-turn** (TensorAccessor already abstracts the bank type; the three L1 cases are extension calls). Sharded inputs are refused by contract (no TARGET value), so no physical-shard scheme is designed.

Output storage (extension_band): the entry point builds the output as a metadata-only view `out = view(input_tensor, logical=out_shape, padded=input_tensor.padded_shape)` (C++ `tt::tt_metal::view`, `ttnn/core/tensor/tensor_ops.cpp:533`, tracked as `Tensor::reshape` with no program factory; Python: `ttnn.reshape(input_tensor, out_shape, input_tensor.padded_shape)` takes the volume-mismatch tile view path, `ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.cpp:771-779`, **only if** the routed-codegen check before it does not dispatch — the implementer MUST confirm under `ttnn.graph` capture that the view adds zero program nodes; if it does dispatch, bind/locate a direct `tt::tt_metal::view(logical, padded)` path instead). Then `ttnn.generic_op([out], desc)` — single-tensor in-place `io_tensors` is supported (`generic_op_device_operation_types.hpp:50-52`). The input's logical values are never written.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | extension_band: one band tile (`band_idx = slice · in_tiles_w + wt`, `N_band = in_slices · in_tiles_w`). new_storage_tiles: one output tile (flat `t`, `N_out = out_slices · out_tiles_h · out_tiles_w`) |
| Grid | `device.compute_with_storage_grid_size()` (runtime, never a literal); core count `ncores = min(grid_x·grid_y, ceil(N / min_units_per_core))`, `min_units_per_core` = host knob (Phase 0: 1) |
| Per-core work | `ttnn.split_work_to_cores(grid, N, row_wise=True)` → ≤ 2 core groups (`units_per_core_g1`, `units_per_core_g2`), contiguous unit ranges in core order; RT args per core: `start_unit`, `num_units` |
| Remainder | handled by the two core groups of `split_work_to_cores`; tile counts are `ceil` per image (`in_tiles_h = ceil(d_{r-2}/32)` per slice, never `floor(prod/32)`) |
| Blocks per core | Phase 0: `num_blocks_this_core = ceil(num_units / block_tiles)` = 1 for every INPUTS shape; `last_block_extent = num_units − (num_blocks−1)·block_tiles` (nominal CB push/pop counts keep `block_tiles`) |
| Core ranges (BB-17) | row-wise fill of the first `ncores` cores = ≤ 2 rectangles per group |
| Plan report (BB-08) | entry point keeps `g6_pad.last_plan` (regime name, ncores, block extents); the same dict is the descriptor cache key value |
| Program cache (BB-07) | Python-side cache of the built `ProgramDescriptor` keyed by `(regime, in_shape, in_padded_shape, out_shape, dtype, layout, buffer_type, grid)`; on hit only RT args are patched (`fill_bits`, buffer addresses). `generic_op`'s own hash excludes RT values (`generic_op_device_operation.hpp:44-46`), so all structure (CT args, core ranges, CB sizes) is shape-derived and keyed |

Phase 0 core counts (Blackhole, ~110–130 cores): p2–p4 1940 band tiles → full grid; `[1,3844,1536]` 48 → 48 cores; `[16,1,1536]` 768 → full grid; `[1,3844,12]` 1 → 1 core (one unit of work exists); `[1,1,196,196]→256²` 64 output tiles → 64 cores; `[1,196,128]→256` 32 → 32 cores.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_fill_tile` | 0 | `tile_bytes` | 1 | constant source page; spans no block axis; segments ≤ `tile_bytes − face_bytes` so any 16-aligned source offset fits | input dtype | writer (`band_writer` in extension_band) | writer (NoC source) | whole kernel |
| `cb_input_tiles` | 1 | `tile_bytes` | `input_depth_blocks · block_tiles` | spans the flat output-tile axis of one block (slot k ↔ k-th tile of the block); depth 1 when one block per core | input dtype | reader | writer | per block (new_storage_tiles only) |

Ring-wrap: pushes and pops are always the nominal `block_tiles` (fill-only slots and a short last block are simply not written / not sent), so every cycle closes exactly at capacity.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| E1 | `make_fill_page` | 1 page | `fill_l1_range` | — | `cb_fill_tile`, 1 (written by stores, no push needed: same-kernel scratch) | resident |
| E2 | `write_band_block` | `band_block_tiles × band_segments` | no | `cb_fill_tile`, 1, resident | DRAM (output = input storage) | unchanged |
| E3 | `drain_writes` | — | no | — | — | — |
| N1 | `make_fill_page` | 1 page | `fill_l1_range` | — | `cb_fill_tile`, 1 | resident |
| N2 | `load_block` | `block_tiles` slots | no | DRAM input | `cb_input_tiles`, push `block_tiles` | full block visible to writer |
| N3 | `write_fill_tiles` | fill-only tiles of block | no | `cb_fill_tile`, 1 | DRAM output | unchanged |
| N4 | `patch_block` | edge slots | `fill_l1_range` | `cb_input_tiles`, wait `block_tiles` | same slots, in place | owned by writer until pop |
| N5 | `store_block` | source slots | no | `cb_input_tiles` | DRAM output; pop `block_tiles` | freed |
| N6 | `drain_writes` | — | no | — | — | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| page addressing | raw_api | `TensorAccessor` / `TensorAccessorArgs` | `tech_reports/tensor_accessor/tensor_accessor.md` | CT accessor args of input (and output) | — | — | — |
| E2, N3, N5 | raw_api | `noc_async_write(src_l1, dst_noc_addr, bytes)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:876` | segment offset/bytes (E2), `tile_bytes` (N3/N5) | `cb_fill_tile` / `cb_input_tiles` | — | `band_block_tiles`, `block_tiles` (trip counts) |
| N2 | raw_api | `noc_async_read` + `noc_async_read_barrier` (once per block) | `dataflow_api.h:547`, `:1921` | `tile_bytes` | — | `cb_input_tiles` | `block_tiles` |
| N2/N4/N5 | raw_api | `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front`/`get_write_ptr`/`get_read_ptr` | same | nominal `block_tiles` | `cb_input_tiles` | | `block_tiles` |
| E3, N6 | raw_api | `noc_async_write_barrier` | `dataflow_api.h:1951` | — | — | — | — |
| N5 | raw_api | `noc_async_writes_flushed` before `cb_pop_front` | `dataflow_api.h:1980` | — | `cb_input_tiles` | — | — |
| E1, N4 | helper | `dataflow_kernel_lib::fill_l1_range<elem_bytes>(start_addr, n_bytes, fill_bits)` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:101` | E1: one call over the whole page. N4 row patch: one call per contiguous run — `edge_row < 16`: face 0 rows `[edge_row,16)`, then face 1 rows `[edge_row,16)` + faces 2,3 whole (one run); `edge_row ≥ 16`: face 2 rows `[edge_row−16,16)`, face 3 rows `[edge_row−16,16)`; column patch: one call per face row over columns `[edge_col mod 16 …)` of faces 0/2 (if `edge_col<16`) and whole face rows of faces 1/3, or `[edge_col−16,16)` of faces 1/3 | — | `cb_fill_tile` / `cb_input_tiles` | — |

**Helpers considered and rejected** (there is no compute phase; every operation is a data move):

| Helper | Mismatch | Citation |
|--------|----------|----------|
| eltwise `copy` (`ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:113`) | Would route every tile through unpack → DEST → pack: a format path the Precision rule forbids for int32/uint32/fp32 (a detour through another format) and that adds a compute kernel to a pure move; the edge patch with `c` has no element in the chain vocabulary | `convenience.hpp:113`, `convenience.inl:68` |
| `fill_tile_column<dfb_id, T>` | Works one column at a time with element-sized stores (`ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:70-78`); the column patch is `32 − edge_col` adjacent columns, which `fill_l1_range` covers as one contiguous run per face row with 4-byte bulk stores | `tile_column_dataflow.hpp:70` |
| `tilize_helpers.hpp` / `untilize_helpers.hpp` | No layout change happens: input and output share the layout | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` |
| `mcast/kernel/mcast_args.hpp` | No reuse-shared operand (see regime `fill_page_broadcast`, rejected) | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp` |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Input tile padding copied into logical output | Golden inputs are zero padded and `c = 0` mostly; a whole-tile copy passes except under `poison_padding` | Unconditional row/column patch of every edge slot (N4); extension writes all band rows `[edge_row, band_row_end)` from the fill page |
| View dispatches a program | Golden counts program nodes; `ttnn.reshape` may route to codegen before its view path | Verify zero extra program nodes under `ttnn.graph` capture; build the view before `generic_op([out], …)` |
| Touching other tile rows in extension | `untouched_rows` compares the raw buffer | Page index uses `in_tiles_h − 1` only; segments derive from `[band_row_begin, band_row_end)` only |
| Value in the cache key / CT | Same shape, different `value` replays a stale fill (`loose_same_shape_two_values`) | `fill_bits` is RT only; padding → output shape → part of the Python cache key |
| bf16 fill conversion | `-1.7e38` must be `0xFF00`; `0.1` must match torch RNE | Host converts with torch, never with `struct.pack` truncation |
| Column patch on 2-byte granularity | NoC cannot start a write at a 2 B offset | Patches are RISC stores in L1, face-aware |
| Partial-tile band writes | 32 B granularity; must be 16 B aligned and src≡dst mod 16 | `row_bytes` multiple of 16; source offset from a 16-aligned fill page |
| Writes left in flight at return | Next program hangs (BB-03) | `noc_async_write_barrier` at the end of every writing kernel |
| Fill-only tiles needlessly read | Rule: no page holding only `c` is read from DRAM | Reader skips fill-only slots; nominal push count still `block_tiles` |
| L1 cases (refinement) and other dtypes | `elem_bytes` changes `row_bytes`/`face_bytes`/`tile_bytes` | Every geometry value derived from `elem_bytes` in the plan; no literal 2048/512/32 B in kernels |
| Profiling (BB-21) | Each role's work and its wait need zones | Zones: `make_fill_page`, `write_band_block`, `load_block`, `write_fill_tiles`, `patch_block` (incl. its wait), `store_block`, `drain_writes`; the plan report lists role per core |

### Registry model (for the implementer)

| Item | Value |
|------|-------|
| INPUT_TAGGERS | `tag_alignment`: W%32≠0 → `"w_non_aligned"`; elif H%32≠0 → `"h_non_aligned"`; else `"tile_aligned"` (H, W = input `shape[-2]`, `shape[-1]`). `tag_rank`: `len(shape)`. `tag_padded_dims`: highs aligned to trailing dims; `h[-2]>0 ∧ h[-1]>0` → `"hw"`, only `h[-2]` → `"h"`, only `h[-1]` → `"w"`, else `"leading"`. `tag_dtype`, `tag_layout`, `tag_memory_config` read the scenario keys |
| Phase 0 SUPPORTED | dtype `[bfloat16]`, layout `[TILE_LAYOUT]`, memory_config `["dram"]`, alignment `["w_non_aligned","h_non_aligned"]`, rank `[3,4]`, padded_dims `["h","hw"]` |
| EXCLUSIONS | none |
| validate() order | (1) argument validation → `ValueError` (padding form, low≠0, high<0, k∉[1,r], value type/int range, sharded input); (2) build the scenario dict from the tensor and gate SUPPORTED per axis then EXCLUSIONS (`UnsupportedAxisValue` / `ExcludedCell`). `validate()` is the first line of `g6_pad` |

### Structural impossibilities

None beyond `INVALID = []`.
