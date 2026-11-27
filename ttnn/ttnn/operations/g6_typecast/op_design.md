# Operation Design: g6_typecast

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (elementwise format conversion, data-movement-bound at full grid) |
| Goal | Convert every value of one tensor to `dtype` in **one** device program; the output keeps the input's shape, layout and interleaved memory config (DRAM or L1). Phase 0: bfloat16 TILE → float32 / bfloat8_b / int32, ranks 3–4, tile-aligned and non-aligned. |
| Math | `out[i] = conv(x[i])` for every logical index `i`. Phase 0: `bf16→f32: v` (exact), `bf16→i32: trunc(v)` (toward zero, \|v\| < 2^31), `bf16→bf8b: v` in the block format (16 values of a tile face row share one exponent; within one step `2^(e_max−6)` of the host rendering). Refinement pairs: `f32→bf16: rne16(v)`, `i32→bf16: rne16(rne32(v))`, `u32↔i32: same 32 bits`, `u16→i32: zero-extend`, `bf16→u8: trunc(v)` (0 ≤ v < 256). |
| References | `eval/golden_tests/g6_typecast/{feature_spec.py, helpers.py, test_golden.py, test_regression.py}`; `.claude/references/{blocking-model.md, l1-footprint-discipline.md, ttnn-cb-memory-fundamentals.md, noc-alignment-in-tt-metal.md}`; `ttnn/cpp/ttnn/kernel_lib/eltwise/api/{chain.hpp, convenience.hpp}`; `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h`; `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h`; catalog `ttnn/ttnn/operations/examples/master.md` (`double_buffer`, `noc_placement`, `bank_stagger`, `read_coalescing`, `trid_double_issue`, `tile_reorder`, `compute_block_size`) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | on device; interleaved; rank 1–4; dtype ∈ TARGET `dtype` | — | — |
| `dtype` | `ttnn.DataType` (positional or keyword) | yes | ∈ TARGET `output_dtype` | — | selects CB formats + compute config (program cache key) |
| `sub_core_grids` | `ttnn.CoreRangeSet \| None` (keyword-only) | no | `None` only | `None` | — |

**Validation (in `validate()`, the entry point's first line), in this order:**

| # | Check | Raises |
|---|-------|--------|
| 1 | `isinstance(dtype, ttnn.DataType)` | `ValueError` |
| 2 | `sub_core_grids is None` | `ValueError` |
| 3 | `1 <= len(input_tensor.shape) <= 4` | `ValueError` |
| 4 | input memory layout is `INTERLEAVED` (the contract has no sharded placement) | `ValueError` |
| 5 | build the scenario dict `{"input_shape", "dtype", "output_dtype", "layout", "buffer"}` from `input_tensor` + `dtype`, apply `INPUT_TAGGERS` in declaration order | — |
| 6 | every axis value ∈ `SUPPORTED[axis]` | `UnsupportedAxisValue` (`ttnn.operations._op_contract`) |
| 7 | no `EXCLUSIONS` entry matches (Phase 0: `EXCLUSIONS = []`, stays empty in every phase) | `ExcludedCell` |

`INPUT_TAGGERS` (names and return values match TARGET in `feature_spec.py`): `tag_dtype` → `scenario["dtype"]`; `tag_output_dtype` → `scenario["output_dtype"]`; `tag_layout` → `scenario["layout"]`; `tag_buffer` → `"l1"` if `buffer_type == ttnn.BufferType.L1` else `"dram"`; `tag_alignment` → `"w_non_aligned"` if `W % 32`, else `"h_non_aligned"` if `H % 32`, else `"tile_aligned"` (`W = shape[-1]`, `H = shape[-2]`, `H = 1` for rank 1); `tag_rank` → `len(shape)`.

Phase 0 `SUPPORTED`: `dtype: [bfloat16]`, `output_dtype: [float32, bfloat8_b, int32]`, `layout: [TILE_LAYOUT]`, `buffer: ["dram", "l1"]`, `alignment: ["tile_aligned", "w_non_aligned", "h_non_aligned"]`, `rank: [3, 4]`. No size, width or page-size axis exists or may be added.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 1–4, any H and W (non-multiples of 32 handled natively through tile padding) |
| Dtype | Phase 0: bfloat16 (TARGET adds float32, int32, uint32, uint16) |
| Layout | Phase 0: TILE (TARGET adds ROW_MAJOR) |
| Memory | interleaved, DRAM or L1 |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` (logical); padded shape = the input's tile padding (same page count for TILE) |
| Dtype | `dtype` |
| Layout | `input_tensor.layout` |
| Memory | `input_tensor.memory_config()` (same interleaved layout, same buffer type) — allocated with `ttnn.allocate_tensor_on_device(shape, dtype, layout, device, input_tensor.memory_config())` |

Tile padding of the output is unspecified, except that **padding never sets a bfloat8_b group's exponent** (see Mechanism caps, Key Risks).

## Blocking Model

The op is a parameterized blocking scheme over one flattened, fully independent page axis. Every
decision below is the realization source for Work Distribution, Circular Buffers and Block
Operation Realization; nothing downstream re-decides it.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` (all dims before H: `N·C` at rank 4, `B` at rank 3, 1 at ranks 1–2) | independent — `out[i]` depends on `x[i]` only | folded into `tile` (contiguous in page order, treated identically) → its extent is set through `block_tiles` | folded | `Plan.block_tiles` | through `tile` | knob-turn |
| `tile_row` (`tensor_tile_rows = ceil(H/32)` per image) | independent — same | folded into `tile` | folded | `Plan.block_tiles` | through `tile` | knob-turn |
| `tile_col` (`tensor_tile_cols = ceil(W/32)`) | independent — a bfloat8_b group (16 values of one face row) never leaves its tile | folded into `tile` | folded | `Plan.block_tiles` | through `tile` | knob-turn |
| `tile` (flattened page id in row-major tile order = interleaved page order; extent `tensor_tiles = input.buffer_num_pages()`) | independent — no result spans two tiles | `block_tiles` | `min(ceil(core_tiles_max / pipeline_depth), block_tiles_l1_cap)` — the core's whole assignment resident across the `pipeline_depth` slots, cut into `pipeline_depth` blocks so block *k*'s conversion overlaps block *k+1*'s read (overlap lamp is the stated reason it is not one block); 1 for every core whose assignment is 1 tile | host `_plan()` (one function); passed to all three kernels as a common runtime arg; CB capacities derive from it | contiguous page ranges over `num_cores = min(tensor_tiles, max_cores)` cores, row-wise (`split_work_to_cores(..., row_wise=True)`); per-core walk `first_page + k·page_stride`, `page_stride = 1` | knob-turn (`block_tiles`, `max_cores`, `page_stride` → bank-affine regimes) |
| `tile_elem` (32×32 positions of one tile: 4 faces × 16 rows × 16 cols) | independent per value; the bf8b shared exponent couples the 16 values of one face row inside the packer | whole tile (never split) | 1 tile = 1 page | the page format (a TILE page is one tile) | not assigned across cores: the tile is the unpack/pack unit, the NoC page and the bf8b group carrier; the sub-tile split is a `rejected` regime | none |

Every axis is independent; **there is no dependent axis and no reuse-shared operand** (one input
operand, varying along every split axis), so no combine and no broadcast regime exists.

**Plan-scoped names** (single source: the host `_plan()`; every other quantity derives from these):
`tensor_tiles`, `max_cores` (Phase 0 = `grid.x · grid.y` of `compute_with_storage_grid_size()`),
`num_cores = min(tensor_tiles, max_cores)`, `core_tiles` (this core's share, `ceil` or `floor` of
`tensor_tiles / num_cores`), `core_tiles_max = ceil(tensor_tiles / num_cores)`, `pipeline_depth`
(Phase 0 = 2), `block_tiles`, `num_blocks_this_core = ceil(core_tiles / block_tiles)`,
`last_block_extent = core_tiles − (num_blocks_this_core − 1) · block_tiles`, `page_stride`
(Phase 0 = 1).

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `pipeline_depth` (`Plan.pipeline_depth`) | 2 | reader fills block *k+1* (its DRAM/L1 read latency) while compute converts block *k* |
| `cb_output_tiles` | `pipeline_depth` (same source) | 2 | compute packs block *k+1* while writer drains block *k*; writer's per-block flush overlaps the next conversion |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Model L1 budget, 1,347,776 B/core (BB-05) | `block_tiles` | `block_tiles ≤ block_tiles_l1_cap = floor((L1_OP_BUDGET − out_bank_bytes) / (pipeline_depth · (in_page_bytes + out_page_bytes)))`, `L1_OP_BUDGET = 1,347,776`, `out_bank_bytes = ceil(tensor_tiles / num_l1_banks) · out_page_bytes` when the output is in L1, else 0. If `block_tiles_l1_cap < 1` the host raises `RuntimeError` (L1 budget) — unreachable for every golden INPUTS shape (max 0.27 MB, ledger) | CB region clashes with the model's L1 tensors: refused launch or overwritten model state |
| Core assignment | `block_tiles` | `≤ core_tiles_max` (built into the Phase 0 rule) | wasted L1 only |
| `DataflowBuffer` handshake count is `uint16_t` (`dataflow_buffer.h:243-246`) | `block_tiles` | `≤ 65535`, implied by the L1 cap (≤ ~220 tiles) | truncated count → hang |
| DEST capacity | the DEST group *inside* `convert_block` — **not** `block_tiles` | the chain clamps `block_size · lane_width ≤ DEST_AUTO_LIMIT` at runtime (`chain.hpp:504-506`; `dest_helpers.hpp:89-103`: 4 tiles fp32 / 8 tiles fp16 in half-sync) | n/a (helper-owned) |
| Interleaved page = one NoC transaction | `load_block` / `store_block` granularity | one `async_read` / `async_write` per page; consecutive interleaved pages live in different banks, so a block is never one transfer | a multi-page transfer reads/writes the wrong pages |
| bfloat8_b shared exponent (16 values of one face row) | `tile_elem` (unsplittable); padding inside a straddling group | block unit = whole tile; a group straddles logical/padding only when `output_dtype == bfloat8_b ∧ W % 16 ≠ 0` — no INPUTS case (every bf8b case has W = 256); see Key Risks for the fill | padding with a larger magnitude sets the exponent; logical values lose more than one step |
| 32-bit DEST for 32-bit formats (`typecast.h:64`) | compute config | `fp32_dest_acc_en = True` iff `dtype` or `output_dtype` ∈ {float32, int32, uint32}, or the SFPU stores INT32 (bf16→u8) | int32/float32 pack from 16-bit DEST corrupts silently |
| SrcA carries 19 bits | compute config for 32-bit **sources** (refinement pairs) | `unpack_to_dest_mode[cb_input_tiles] = UnpackToDestFp32` when `dtype` ∈ {float32, int32, uint32} | f32→bf16 rounds from a truncated mantissa (double rounding); int32 loses low bits (violates the all-32-bits rule) |
| Blackhole NoC offset rule: DRAM read ≡ mod 64, DRAM write ≡ mod 16, L1 ≡ mod 16 (`noc-alignment-in-tt-metal.md` §1; `core_config.h:42-43`) | RM page placement in L1 (deferred RM regimes) | RM pages land in alignment-strided staging slots; dense packing is a RISC repack, never a mismatched-offset NoC transfer | silent corruption without watcher |

### Regimes

Named minimum: **each input page crosses its memory tier (DRAM, or the L1 banks) once and each
output page crosses once.**

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `tile_convert` | **built** | `layout == TILE ∧ (dtype, output_dtype) ∉ REINTERPRET_PAIRS` (all 40 Phase 0 cases; later f32→bf16, i32→bf16, u16→i32 with the per-pair compute config) | `block_tiles` consecutive pages of the core's contiguous range | **minimum**: each input tile read once, each output tile written once; tile padding moves with its tile (a TILE page is the whole tile). L1 placement: each page crosses the NoC once between its bank and the converting core (removed by `l1_bank_affine`) | per block: reader 1 reserve + 1 read barrier + 1 push; compute 1 wait + 1 reserve + chain init/reconfig + 1 pop + 1 push; writer 1 wait + 1 flush + 1 pop. Lower bound on block count is the overlap lamp (≥ `pipeline_depth` blocks when `core_tiles ≥ pipeline_depth`) |
| `l1_bank_affine` | **deferred** — the built `tile_convert` already serves every L1 shape (Rule 1b); the assignment needs the L1-bank → logical-core map (host: `ttnn.reports.get_buffer_pages` per-page `core_x/core_y`, `reports.cpp:42-44`, or a device-side scan of the accessor's bank table). Reachable with no kernel change: host sets `first_page = bank`, `page_stride = num_l1_banks` per core | `layout == TILE ∧ buffer == "l1" ∧ pair ∉ REINTERPRET_PAIRS` (takes precedence over `tile_convert` once built) | same, over pages `{bank + k·num_l1_banks}` | same tier crossings, all core-local: removes `tensor_tiles · (in_page + out_page)` cross-core NoC bytes (P39: 24.5 MiB) | same as `tile_convert` |
| `tile_reinterpret` | **deferred** — serves only uint32/int32 sources (`dtype` values outside Phase 0 SUPPORTED; Q2230). Built on `tile_convert`'s reader and writer with the compute kernel and `cb_output_tiles` removed (stepping stone) | `layout == TILE ∧ (dtype, output_dtype) ∈ REINTERPRET_PAIRS = {(uint32, int32), (int32, uint32)}` | `block_tiles` pages | minimum; no compute (the 32 bits are the result) | reader barrier + handshakes per block |
| `rm_convert` | **deferred** — serves only ROW_MAJOR (layout outside Phase 0 SUPPORTED; F317, F322 bf16→u8). Reuses `tile_convert`'s compute kernel unchanged on dense pseudo-tiles | `layout == ROW_MAJOR ∧ pair ∉ REINTERPRET_PAIRS ∧ ¬rm_bank_runs_predicate` | `block_pages` consecutive pages, densely repacked into `ceil(block_pages · W / 1024)` pseudo-tiles (a pseudo-tile = 1024 consecutive values; unpack/pack are inverse layouts, so an elementwise op is position-preserving) | each page crosses DRAM once, but a page ≤ 64 B occupies one 64-B DRAM slot (BH DRAM alignment, `core_config.h:42`): the floor is set by **page count**, not logical bytes (F322: 4,194,304 pages each way). One NoC transaction per page each way | chain init, handshakes and repack setup amortized over `block_pages` |
| `rm_convert_bank_runs` | **deferred** — successor built **on top of** `rm_convert` (same repack + compute); positive reason: it exists only once `rm_convert` exists | `layout == ROW_MAJOR ∧ buffer == "dram" ∧ in_aligned_page_bytes == DRAM alignment (pages ≤ 64 B) ∧ pair ∉ REINTERPRET_PAIRS` | runs of `block_pages` same-bank pages (bank-affine: `page_stride = num_dram_banks`); a run is `block_pages · 64 B` contiguous in one bank | moves whole 64-B slots (F322: 256 MiB each way physical vs 37.7 MB logical) in large transfers instead of 8.4 M single-page transactions; trades transaction count for padding bytes | one read + one write per run instead of per page |
| `rm_reinterpret` | **deferred** — ROW_MAJOR with uint32/int32 sources (outside Phase 0: Q2189…Q2234 u32→i32 single pages ≤ 16,384 B; F321 i32→u32 one value) | `layout == ROW_MAJOR ∧ pair ∈ REINTERPRET_PAIRS` | `block_pages` whole pages; a page larger than `page_chunk_bytes_cap` (64 KiB) is cut into byte chunks (no INPUTS page exceeds 16 KiB) | minimum; no compute | reader barrier + handshakes per block |
| `single_core` | **rejected** — superseded by `tile_convert` at full grid: same bytes, `1/num_cores` the occupancy | — | — | — | — |
| `sub_tile_split` (faces across cores) | **rejected** — superseded by `tile_convert`: 4× the transactions at 512 B (catalog `tile_reorder`: whole-page transfers win); only differs for tensors below `num_cores` tiles, which are one DRAM round trip either way | — | — | — | — |
| `dm_scalar_widen` (bf16→f32 by RISC shifts in the reader) | **rejected** — superseded by `tile_convert`: the packer widens for free; a scalar loop is O(1024) RISC ops per tile on the RISC that issues the NoC reads | — | — | — | — |
| `rm_via_tilize` (tilize rows → convert → untilize) | **rejected** — superseded by `rm_convert`: a 3-value row padded to a 32-wide tile row pushes 10.7× the values through unpack/pack, adds two phases, and any DRAM-staged copy breaks the one-crossing rule | — | — | — | — |

**Regime selection function** (host, exact; Phase 0 has one built row, so `validate()` already
confines calls to `tile_convert`; the function is pinned now so later rows plug in without
re-deciding):

```python
REINTERPRET_PAIRS = {(ttnn.uint32, ttnn.int32), (ttnn.int32, ttnn.uint32)}
def select_regime(layout, dtype, output_dtype, buffer, in_aligned_page_bytes, dram_alignment):
    if (dtype, output_dtype) in REINTERPRET_PAIRS:
        return "tile_reinterpret" if layout == ttnn.TILE_LAYOUT else "rm_reinterpret"
    if layout == ttnn.TILE_LAYOUT:
        return "tile_convert"            # "l1_bank_affine" when buffer == "l1", once built
    if buffer == "dram" and in_aligned_page_bytes == dram_alignment:
        return "rm_convert_bank_runs"    # once built; "rm_convert" until then
    return "rm_convert"
```

Each refinement that builds a second regime adds regime-pinned tests (one production case per
regime, asserting `describe_plan(...)["regime"]`).

### Traffic ranking

| Candidate split | DRAM crossings per tensor | Cross-core NoC bytes | Verdict |
|-----------------|---------------------------|----------------------|---------|
| contiguous page ranges over `min(tensor_tiles, max_cores)` cores, row-wise (**built**) | input 1, output 1 | DRAM placement: none beyond the DRAM transfer itself. L1 placement: `tensor_tiles · (in_page + out_page)` (each page between its bank and its converting core) | primary — for DRAM it **is** the minimum |
| L1 bank-affine, owner computes (L1 placement only) | 0 (L1 tier, core-local) | 0 | cheapest for L1 → `deferred` row `l1_bank_affine` |
| DRAM bank-affine (cores adjacent to their bank) | input 1, output 1 | same bytes, fewer hops | comparable bytes → perf lamp (placement), not a regime |
| sub-tile split | input 1, output 1 | same bytes, 4× transactions | `rejected` |
| single core | input 1, output 1 | same | `rejected` (occupancy) |
| dependent axis split + combine | n/a — no dependent axis exists (`out[i]` depends on `x[i]` only; the bf8b group is intra-tile and packer-internal) | — | no row |

**Operand-reuse check:** the single input varies along the split axis → nothing is re-read; no
broadcast row. **Stall-shadow check:** the reader blocks on its read barrier, compute on the
reader's push, the writer on compute's push. The only work independent of each wait is the next
block's transfer, which `pipeline_depth = 2` already schedules into the window. No algebraic
reorder exists (no combine, no reduction).

### Block schedule

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    const uint32_t block_extent = (block_idx + 1 == num_blocks_this_core) ? last_block_extent : block_tiles;
    load_block(block_idx, block_extent);     // reader
    convert_block(block_idx, block_extent);  // compute
    store_block(block_idx, block_extent);    // writer
}
```

Logical schedule: the three kernels run it asynchronously, pipelined `pipeline_depth` deep.
**Every reserve / push / wait / pop uses the nominal `block_tiles`**, including the ragged last
block; `block_extent` reaches only the NoC transfers and the conversion (`blocking-model.md` §2,
`ttnn-cb-memory-fundamentals.md` Ring-Wrap Invariant).

| Block operation | Kernel | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|--------|-------------|--------------------|-----------------------------------|
| `load_block` | reader (RISCV_1, NoC0) | `block_extent` pages of `in_page_bytes` into `block_tiles` reserved pages | the other `cb_input_tiles` slot (block *k−1* under conversion) | 1 `reserve_back(block_tiles)`, `block_extent` page reads issued back to back, **1** `async_read_barrier`, 1 `push_back(block_tiles)` per block |
| `convert_block` | compute | `block_extent` tiles, DEST groups of ≤ `DEST_AUTO_LIMIT` (helper clamp) | nothing across blocks; both CB slots stream | 1 `wait_front(block_tiles)` + 1 `reserve_back(block_tiles)` per block; chain init + format reconfig once per block (≤ `pipeline_depth` per core under the Phase 0 extent rule; hoisting to once per kernel is a permitted realization); dst-sync once per DEST group; `compute_kernel_hw_startup` once per kernel |
| `store_block` | writer (RISCV_0, NoC1) | `block_extent` pages of `out_page_bytes` | the other `cb_output_tiles` slot (block *k+1* being packed) | 1 `wait_front(block_tiles)`, `block_extent` page writes, 1 `async_writes_flushed` before 1 `pop_front(block_tiles)` per block; **1 `async_write_barrier` per kernel**, before return (BB-03) |

Profiler build (BB-21): each block operation and each blocking wait (`reserve_back`,
`wait_front`, barrier) sits in its own `DeviceZoneScopedN` zone; zones compile out of the plain
build.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap | the default splits each assignment into `pipeline_depth` blocks; the first block's read and the last block's conversion + write are exposed (fill/drain of a half-assignment) | `block_tiles ∈ {4, 8}` fixed (more, smaller blocks; catalog `double_buffer` sweet spot) on Q2185, P31, P39, P50 |
| Grid synchronization | decode shapes (`tensor_tiles ≤ 16`) put one tile on each of `tensor_tiles` cores; per-core launch cost may exceed one tile's DRAM round trip | `max_cores ∈ {tensor_tiles, ceil(tensor_tiles / 4)}` on Q2191, Q2194, Q2200 |
| Bank stagger | `core_tiles` a multiple of the DRAM bank count puts every core's first read on the same bank (catalog `bank_stagger`, 1.04–1.32× on BH) | rotate each core's walk start by its core index within its range |
| DRAM placement | the row-wise contiguous split ignores DRAM-bank proximity (catalog `noc_placement`, `read_coalescing` bank affinity 0.91–1.28×) | `page_stride = num_dram_banks` with cores from `get_optimal_dram_bank_to_logical_worker_assignment` (`device.cpp:811`) |
| DEST group | the compute-bound L1 bf8b cases (P39, P50) pay one dst-sync per DEST group | chain `block_size` 1 vs `DEST_AUTO_LIMIT`; `dst_full_sync_en=True` on P39 |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| input banks → reader | input tile page (bf16: 2048 B) | `TensorAccessor` page reads on NoC0, one per page, one barrier per block | DRAM or L1 interleaved — the accessor resolves the bank; no kernel difference |
| reader → compute | `cb_input_tiles`, input dtype | block-granular CB handshake (`block_tiles` pages) | |
| compute | unpack input → DEST (fp32 when `fp32_dest_acc_en`) → SFPU conversion (int32 only in Phase 0) → pack to the output format | `compute_kernel_lib::typecast` (eltwise chain) | bf16→f32 and bf16→bf8b are unpacker/packer conversions (no SFPU body); bf16→i32 truncates on the SFPU |
| compute → writer | `cb_output_tiles`, output dtype (f32/i32: 4096 B, bf8b: 1088 B) | block-granular CB handshake | |
| writer → output banks | output tile page | `TensorAccessor` page writes on NoC1, flush per block, barrier at kernel end | output in the input's buffer type |

**TARGET axes seen as regimes** (TARGET has no sharded placement; every value is interleaved):

| TARGET axis value | Regime / cost | Unlock class |
|-------------------|---------------|--------------|
| `buffer: "dram"`, `"l1"` | `tile_convert` serves both today (accessor-resolved banks); the cheaper L1 split is `l1_bank_affine` | knob-turn (`first_page`, `page_stride`) |
| `dtype: float32, int32` with `output_dtype: bfloat16`; `uint16 → int32` | `tile_convert` with the per-pair compute config (API Mapping) | knob-turn (CB formats + compute config) |
| `dtype: uint32 ↔ int32` | `tile_reinterpret` / `rm_reinterpret` | new kernel set on the same reader/writer |
| `layout: ROW_MAJOR` (+ `output_dtype: uint32, uint8, bfloat16` there) | `rm_convert`, `rm_convert_bank_runs`, `rm_reinterpret` | scheme-change of the reader/writer (page repack); compute unchanged |
| `rank: 1, 2` | every production rank-1/2 case is ROW_MAJOR; for TILE, rank only changes `tensor_tiles = buffer_num_pages()` | knob-free |

A deferred regime introduces no inter-Tensix communication; every regime is owner-computes with no
semaphores, multicast or combine.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block = `block_tiles` consecutive tile pages (nominal); the last block of a core is `last_block_extent` pages |
| Grid | `device.compute_with_storage_grid_size()` queried at runtime; `max_cores = grid.x · grid.y` (knob; never inlined) |
| Per-core work | `ttnn.split_work_to_cores(grid, tensor_tiles, row_wise=True)` → `num_cores = min(tensor_tiles, max_cores)` cores (≤ 2 rectangles — full rows + one partial row — for BB-17), group 1 with `ceil(tensor_tiles / num_cores)` pages, group 2 with `floor`. Core *c* walks pages `first_page_c + k · page_stride`, `k ∈ [0, core_tiles_c)`, `first_page` = prefix sum of earlier cores' counts, `page_stride = 1` |
| Remainder | split across the two groups; within a core, the ragged last block carries `last_block_extent ∈ [1, block_tiles]` |
| Tile geometry | `tensor_tiles = input.buffer_num_pages()` = `prod(shape[:-2]) · ceil(H/32) · ceil(W/32)` — per image, `ceil` on both, H = 1 at rank 1. Assert `output.buffer_num_pages() == tensor_tiles` on the host |

The same three kernels run on every core; the two groups differ only in runtime args (one
`KernelDescriptor` per kernel over the union of both groups). Representative Phase 0 values at a
13×10 grid (`max_cores = 130`; the real grid is queried):

| Case | Shape | Pair | `tensor_tiles` | `num_cores` | `core_tiles_max` | `block_tiles` | blocks/core |
|------|-------|------|----------------|-------------|------------------|---------------|-------------|
| Q2185 | (16, 4096, 12) | bf16→f32 | 2048 | 130 | 16 | 8 | 2 |
| Q2187 | (1, 1, 3844, 256) | bf16→bf8b | 968 | 130 | 8 (ragged: 7-page cores) | 4 | 2 |
| Q2209 | (1, 1, 1000, 256) | bf16→bf8b | 256 | 130 | 2 | 1 | 2 |
| Q2229 | (1, 1, 32, 128) | bf16→i32 | 4 | 4 | 1 | 1 | 1 |
| Q2227 | (1, 1, 12), L1 | bf16→f32 | 1 | 1 | 1 | 1 | 1 |
| P39 | (1, 128, 256, 256), L1 | bf16→bf8b | 8192 | 130 | 64 (ragged: 63-page cores) | 32 | 2 |
| P50 | (1, 128, 256, 128), L1 | bf16→bf8b | 4096 | 130 | 32 | 16 | 2 |

27 of the 40 Phase 0 cases have `tensor_tiles ≤ 130` → one tile on each of `tensor_tiles` cores.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_tiles` | 0 | `input.buffer_page_size()` (bf16 tile: 2048 B) | `pipeline_depth · block_tiles` | spans `tile` → `block_tiles` per slot × `pipeline_depth` slots; spans `tile_elem` (one whole tile per page); streams over the core's `num_blocks_this_core` | input dtype (bfloat16) | reader | compute | whole kernel |
| `cb_output_tiles` | 1 | `output.buffer_page_size()` (f32/i32: 4096 B, bf8b: 1088 B) | `pipeline_depth · block_tiles` | same axis accounting as `cb_input_tiles` | output dtype | compute | writer | whole kernel |

Two CBs is the minimal inventory: DRAM/L1 → core needs a landing buffer and core → DRAM/L1 needs a
source buffer; an in-place conversion is impossible (input and output page sizes and formats
differ). Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_extent` pages (nominal `block_tiles`) | raw dataflow API (no interleaved page-block helper exists) | input tensor pages `first_page + (block_idx·block_tiles + i)·page_stride` | `cb_input_tiles`, `block_tiles` pushed | `block_tiles` pages credited to compute |
| 2 | `convert_block` | `block_extent` tiles, `IterationShape::tiles(block_extent).block_size(DEST_AUTO_LIMIT)` | `compute_kernel_lib::typecast` | `cb_input_tiles`, `block_tiles` waited (caller), tiles `[0, block_extent)` read by index (`InputTileMapping::Block`), `block_tiles` popped (caller) after the chain | `cb_output_tiles`, `block_tiles` reserved (caller), tiles `[0, block_extent)` packed, `block_tiles` pushed (caller) | input slot freed; output slot credited to writer |
| 3 | `store_block` | `block_extent` pages | raw dataflow API | `cb_output_tiles`, `block_tiles` waited | output tensor pages, same page ids as `load_block` | slot popped after `async_writes_flushed` |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot | raw_api | `compute_kernel_hw_startup(icb, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:113` | `(cb_input_tiles, cb_output_tiles)`, first statement of the compute kernel | — | — | none |
| `convert_block` | helper | `compute_kernel_lib::typecast<Input, Output>(IterationShape, in, out)` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:89-91`, impl `convenience.inl:48-57` (formats derived from the bound CBs via `dfb_l1_format`) | `Input = input(cb_input_tiles, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block)` (`chain.hpp:385-391`); `Output = output(cb_output_tiles, ReservePolicy::None, PushPolicy::None)` (`chain.hpp:442-450`); shape `IterationShape::tiles(block_extent).block_size(DEST_AUTO_LIMIT)` (`chain.hpp:146-173`, clamp `chain.hpp:504-506`); `DataflowBuffer` objects constructed once per kernel | `cb_input_tiles` | `cb_output_tiles` | `block_extent` (from `block_tiles`); the DEST group is the helper's clamp, not a knob |
| `convert_block` (CB sync around the chain) | raw_api | `DataflowBuffer::wait_front / pop_front / reserve_back / push_back` | `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:243-246` | nominal `block_tiles` each | `cb_input_tiles` | `cb_output_tiles` | `block_tiles` |
| `load_block` | raw_api | `Noc::async_read(accessor, dfb, page_bytes, {.page_id}, {.offset_bytes})`, `Noc::async_read_barrier`, `TensorAccessor` | `tt_metal/hw/inc/api/dataflow/noc.h:252`, `noc.h:739`; `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | one read per page into slot offset `i · in_page_bytes`; one barrier per block | input tensor | `cb_input_tiles` | `block_tiles`, `page_stride` |
| `store_block` | raw_api | `Noc::async_write`, `Noc::async_writes_flushed`, `Noc::async_write_barrier` | `noc.h:411`, `noc.h:783`, `noc.h:760` | one write per page; flush per block; barrier once at the end | `cb_output_tiles` | output tensor | `block_tiles`, `page_stride` |

**Helpers considered and rejected** (dataflow raw_api rows):

| Candidate | File:Line | Why it cannot express `load_block` / `store_block` |
|-----------|-----------|--------------------------------------------------|
| `local_copy_helpers_dataflow.hpp` (`read_with_state`, `set_read_trid`) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:74-158` | core-local / fixed-source stateful reads; no interleaved-accessor page walk. Its trid ring is the `trid_double_issue` refinement, not a page reader |
| `tile_column_dataflow.hpp` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:69-92` | fills/copies one column inside an L1 tile; no NoC page transfer |
| `dfb_helpers_dataflow.hpp` | `ttnn/cpp/ttnn/kernel_lib/dfb_helpers_dataflow.hpp:14-18` | tile-dimension queries only |

**Per-pair compute configuration** (one compute kernel source; the pair selects formats and
config at compile time; all entries are part of the program cache key):

| (dtype → output_dtype) | Phase | `fp32_dest_acc_en` | `unpack_to_dest_mode[cb_input_tiles]` | Chain op | Conversion mechanism (File:Line) |
|------------------------|-------|--------------------|---------------------------------------|----------|----------------------------------|
| bf16 → float32 | 0 | True | Default | `typecast` → `Typecast<Float16_b, Float32>` | no SFPU body; DEST holds the exact value, packer emits Float32 (`typecast.h:158-159`) |
| bf16 → bfloat8_b | 0 | False | Default | `typecast` → `Typecast<Float16_b, Bfp8_b>` | no SFPU body; packer quantizes per 16-value face row (`typecast.h:315-316`) |
| bf16 → int32 | 0 | True | Default | `typecast` → `Typecast<Float16_b, Int32>` | `calculate_typecast_fp32_to_int32`: mantissa shifted by `exp − 23`, negated for `in < 0` → truncation toward zero (`typecast.h:150-157`; `ckernel_sfpu_typecast.h:198-241`) |
| float32 → bf16 | refinement | True | UnpackToDestFp32 | `typecast` → `Typecast<Float32, Float16_b>` | RNE: `+0x7FFF + lsb`, mask low 16 (`typecast.h:160-167`; `ckernel_sfpu_typecast.h:273-289`) |
| int32 → bf16 | refinement | True | UnpackToDestFp32 | `typecast` → `Typecast<Int32, Float16_b>` | `SFPCAST` (rne32) then `SFP_STOCH_RND FP32_TO_FP16B` (rne16) (`typecast.h:142-149`; `ckernel_sfpu_typecast.h:139-196`) |
| uint16 → int32 | refinement | True | verify per LLK (UInt16 source) | `typecast` → `Typecast<UInt16, Int32>` | zero-extend (`typecast.h:288-295`) |
| bf16 → uint8 (RM) | refinement | True (SFPU stores INT32) | Default | `unary<Typecast<Float16_b, UInt8>, ...>` — `typecast<>` static_asserts on UInt8 CBs (`convenience.inl:52-55`) | `calculate_typecast_fp32_to_uint8`: floor of \|v\|, negate, `& 0xFF` (`typecast.h:380-390`; `ckernel_sfpu_typecast.h:953-990`) |
| uint32 ↔ int32 | refinement | — | — | none (`*_reinterpret` regimes) | the definition is the same 32 bits; DM-only transfer |

Math fidelity is irrelevant (datacopy + integer SFPU); keep the default. No caller-visible
compute config exists.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| bf16 → int32 rounds instead of truncating | a rounding cast passes every integer-valued input; golden witnesses −2.75, −0.5, 0.5, 2.75, 127.5 (helpers.py, test_regression.py) fail by one | use `Typecast<Float16_b, Int32>` only (shift-based truncation, `ckernel_sfpu_typecast.h:198-241`); never a `round`/`SFPCAST`-based path |
| 32-bit output from 16-bit DEST | Int32/Float32 pack or INT32 SFPU store with `fp32_dest_acc_en=False` corrupts silently | `fp32_dest_acc_en` derived from the pair (table above); in the cache key |
| bf8b group exponent set by padding | the packer computes one exponent per 16 values of a face row, padding included; logical precision drops beyond the 1-step gate | groups straddle padding only when `output_dtype == bfloat8_b ∧ W % 16 ≠ 0`: no INPUTS case (all bf8b cases have W = 256; H padding fills whole rows = whole groups). When a refinement meets it: `load_block` zero-fills columns `[W % 32, next multiple of 16)` of the straddling group in last-column tiles before `push_back` (one fill row + matched-offset NoC copies, `noc-alignment-in-tt-metal.md` rule 5) |
| bf8b rounding differs from the host rendering | the device packer and `ttnn.from_torch(..., bfloat8_b)` may round differently; the gate is 1 step of the group's logical maximum | expected within 1 step; if a group misses, set `bfp8_pack_precise=True` (`ComputeConfigDescriptor`, `program_descriptors.cpp:651`) — a cache-key field |
| Output page size ≠ input page size | 2048 → 4096 / 1088 B; reusing input metadata sizes `cb_output_tiles` and the writer wrong | `cb_output_tiles` page size and writer page bytes come from the **output** tensor |
| Ragged cores (Q2187: 7- and 8-page cores; P39: 63- and 64-page cores) | a `min(block_tiles, remaining)` count reaching `push_back` breaks the ring cycle (silent corruption in release) | nominal `block_tiles` on every reserve/push/wait/pop; `last_block_extent` only in NoC transfers and `IterationShape` |
| Program-cache hit with stale buffer addresses (BB-07) | `generic_op` hashes the descriptor without runtime-arg values (`generic_op_device_operation.cpp:79-140`); a memoized descriptor with old addresses writes into a freed buffer | memoize per cache key; on a hit rewrite only the input/output address runtime args; acceptance test runs one shape with three different tensors |
| L1-placed output + CBs exceed the model budget (BB-05) | the op's L1 = its CBs + its L1 output share; the model holds the rest | `block_tiles_l1_cap` includes `out_bank_bytes`; max over INPUTS ≈ 0.27 MB (ledger) |
| Refinement: 32-bit sources through SrcA | SrcA carries 19 bits: f32→bf16 double-rounds, int32 loses low bits — passes small-integer inputs only | `UnpackToDestFp32` on `cb_input_tiles` for float32/int32/uint32 sources |
| Refinement: uint32 ↔ int32 via DEST | DEST integer handling (sign-magnitude fixups exist in the SFPU typecasts) can alter bits | `*_reinterpret` regimes move the 32 bits without compute; the bits are the conversion, not a format copy |
| Refinement: ROW_MAJOR 3-value rows (F322, perf focus p1) | a 6-B page occupies a 64-B DRAM slot on BH; the logical-byte roofline (81.92 µs) is not the physical floor; NoC offset rules forbid dense landing | `rm_convert` repacks from aligned staging; `rm_convert_bank_runs` coalesces per bank; quantify the physical floor with `/perf-ceiling-dm` before tuning |

## Host Contract

| Item | Decision |
|------|----------|
| Dispatch | exactly one `ttnn.generic_op([input_tensor, output_tensor], program_descriptor)` per call; no other device op, no host reads of tensor values, no host-side layout/pad/slice/typecast (BB-01, BB-02) |
| Entry point order | `validate()` → allocate output (input's shape, `dtype`, input's layout, input's `memory_config()`) → look up / build the plan and descriptor → patch address runtime args → `generic_op` |
| Plan | one host function `_plan(input_tensor, output_dtype) -> Plan` (regime, `num_cores`, core ranges, `pipeline_depth`, `block_tiles`, `page_stride`, per-core `first_page`/`core_tiles`/`num_blocks_this_core`/`last_block_extent`, CB bytes). Every CB size and runtime arg is derived from it. Knob overrides (for perf sweeps) are `Plan` fields and enter the cache key |
| Program cache (BB-07, BB-16) | key = (logical shape, padded shape, `dtype`, `output_dtype`, layout, buffer type, device id, compute grid, knob overrides). Hit: reuse the stored `ProgramDescriptor`, rewrite the input/output buffer-address runtime args only, set `custom_program_hash` from the key. No device allocation other than the output and no kernel build after the first call of a key |
| Plan reporting (BB-08, BB-21) | `describe_plan(input_tensor, dtype) -> dict` returns the selected regime, `num_cores`, core ranges, `block_tiles`, `pipeline_depth`, CB bytes, and the per-core role (every core: reader + compute + writer) |
| Completion (BB-03) | writer ends with `async_write_barrier`; reader's last block ends with its read barrier |
| Structural impossibilities | none beyond `feature_spec.py` INVALID (`bfloat8_b` × `ROW_MAJOR`) |
| Index axes | none (no `dim` parameter) |
