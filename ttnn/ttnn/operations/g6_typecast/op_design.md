# Operation Design: g6_typecast

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (elementwise format conversion); data-movement-bound at every production size above a few tiles |
| Goal | Convert every value of one interleaved tensor to `dtype` in **one** device program, bit-identical to `torch.Tensor.to` (bfloat8_b: within one quantization step of each 16-value group), keeping shape, layout and memory placement. |
| Math | `out[i] = conv_{src→dst}(x[i])` for every logical index `i`; positions in tile padding are unspecified |
| Phase 0 scope | `bfloat16` TILE input → `float32` / `bfloat8_b` / `int32`, DRAM or L1 interleaved, ranks 3–4, any H/W alignment (40 of 65 production cases) |
| References | `eval/golden_tests/g6_typecast/feature_spec.py` (TARGET / INPUTS / INVALID); `eval/golden_tests/g6_typecast/helpers.py` (oracle, `TOLERANCES`); `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp`; `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h`; `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h`; catalog `ttnn/ttnn/operations/examples/{double_buffer,noc_placement,bank_stagger,trid_double_issue,compute_block_size,read_coalescing,tile_reorder}`; `.claude/references/noc-alignment-in-tt-metal.md` |

### Conversion rule per dtype pair (all TARGET pairs)

| Pair (src → dst) | Phase | Device mechanism | Exactness argument (file:line) |
|------------------|-------|------------------|--------------------------------|
| bfloat16 → float32 | 0 | unpack bf16 → 32-bit DEST, pack Float32; no SFPU op | widening is exact; `typecast.h:158` (no SFPU kernel, handled by packer) |
| bfloat16 → bfloat8_b | 0 | unpack bf16 → DEST, packer emits Bfp8_b (shared exponent per 16-value face row) | `typecast.h:315` (handled by packer); graded within one step of the host rendering |
| bfloat16 → int32 | 0 | unpack → 32-bit DEST, SFPU `calculate_typecast_fp32_to_int32`, pack Int32 | mantissa shifted by `exp-23` (right shift for `exp<23`) then negated for `in<0` → truncation toward zero: `ckernel_sfpu_typecast.h:197-241` |
| float32 → bfloat16 | ref. | unpack-to-DEST fp32, SFPU `calculate_typecast_fp32_to_fp16b`, pack Float16_b | `+0x7FFF + lsb, &0xFFFF0000` = RNE: `ckernel_sfpu_typecast.h:272-288` |
| int32 → bfloat16 | ref. | unpack-to-DEST Int32, SFPU `calculate_typecast_int32_to_fp16b`, pack Float16_b | `SFPCAST` (int→fp32) then `SFP_STOCH_RND FP32_TO_FP16B` = two roundings: `ckernel_sfpu_typecast.h:138-195` |
| uint16 → int32 | ref. | unpack-to-DEST, SFPU `calculate_typecast_uint16_to_uint32`, pack Int32 | zero extension (`& 0xFFFF`): `ckernel_sfpu_typecast.h:501-541` |
| bfloat16 → uint8 | ref. | 32-bit DEST, SFPU `calculate_typecast_fp32_to_uint8`, pack UInt8 | `floor(|x|)` then `& 0xFF`; domain `[0,256)` → trunc: `ckernel_sfpu_typecast.h:952-991` |
| uint32 → int32, int32 → uint32 | ref. | **no compute**: the 32 bits are moved unchanged (`*_bitcopy` regimes) | the definition *is* bit identity ("the same 32 bits, read as dst"); `typecast.h` has no Int32↔UInt32 pair, and routing it through DEST/packer would risk a sign-magnitude reinterpretation. This is not "copying bytes between formats" in the prohibited sense — no value changes under this definition. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 1–4, interleaved DRAM or L1, TILE or ROW_MAJOR (Phase 0: TILE, rank 3–4, `bfloat16`) | — | addresses → common RT args; shape → per-core RT args + CT `block_tiles` |
| `dtype` | `ttnn.DataType` (positional or keyword) | yes | TARGET `output_dtype` (Phase 0: `float32`, `bfloat8_b`, `int32`) | — | selects CB data formats + compute config (program identity) |
| `sub_core_grids` | `ttnn.CoreRangeSet` or `None` (keyword-only) | no | `None` only | `None` | — |

### Entry point and validation order

| Step | Check | Raises |
|------|-------|--------|
| 1 | `dtype` is not a `ttnn.DataType` | `ValueError` |
| 2 | `sub_core_grids is not None` | `ValueError` |
| 3 | `len(input_tensor.shape)` outside 1–4 | `ValueError` |
| 4 | build scenario `{"input_shape", "dtype", "output_dtype", "layout", "buffer"}` from the tensor + `dtype`; apply `INPUT_TAGGERS` in the order `dtype, output_dtype, layout, buffer, alignment, rank`; check every axis against `SUPPORTED` | `UnsupportedAxisValue` |
| 5 | `EXCLUSIONS` (stays empty — every TARGET cell is a production case) | `ExcludedCell` |

`validate()` is the first line of `g6_typecast()`. Axis names and values are exactly TARGET's (`feature_spec.py`); `tag_buffer` returns `"l1"` iff `memory_config().buffer_type == ttnn.BufferType.L1`, else `"dram"`; `tag_alignment` uses `H = 1` for rank 1. Steps 1–3 are contract errors (not support refusals) and run before the registry gate. Each message names the offending argument verbatim — `dtype`, `sub_core_grids`, `rank` — because the acceptance test matches those words.

Module exports (`ttnn/ttnn/operations/g6_typecast/__init__.py`): `g6_typecast`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate`, `describe_plan`. `describe_plan(input_tensor, dtype) -> dict` returns the selected plan without dispatching (BB-08, BB-21): keys `"regime"` (one of the regime names below), `"num_cores"`, `"block_tiles"`, `"core_roles"` (`{(x, y): "reader+compute+writer"}` for `tile_convert`; `"reader+writer"` for the bitcopy regimes).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 1–4, any H and W (Phase 0: rank 3–4) |
| Dtype | TARGET `dtype` (Phase 0: `bfloat16`) |
| Layout | TILE or ROW_MAJOR (Phase 0: TILE) |
| Memory | interleaved, DRAM or L1 |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` |
| Dtype | `dtype` |
| Layout | `input_tensor.layout` |
| Memory | `input_tensor.memory_config()` (same memory layout and buffer type) |

Allocated by `ttnn.allocate_tensor_on_device(input_tensor.shape, dtype, input_tensor.layout, device, input_tensor.memory_config())`. No host-side transform of input or output (no `to_layout`/`tilize`/`pad`/`slice`/`reshape`/`typecast`), and no ttnn op other than the single `ttnn.generic_op` dispatch.

## Blocking Model

The op is a pure page-to-page map: output page `p` is the conversion of input page `p`, because both tensors have the same shape and layout, so the TILE layout pads both identically and their page orders coincide. No output element depends on another element except inside a bfloat8_b group, which never leaves one tile row. Every axis is independent; the op is a parameterized streaming scheme over the flattened page axis.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` (all dims before the last two, folded; `prod(shape[:-2])`, 1 for rank ≤ 2) | independent — `out[i]` reads only `x[i]` | folded into `tile_page` (outermost factor of the page index) | whole extent folded, no separate knob — the op treats every image identically and images are page-contiguous | `tile_page`'s `block_tiles` | spread through `tile_page` ranges (a range may cross image boundaries) | knob-turn |
| `tile_row` (`ceil(H/32)` per image, `H=1` for rank 1) | independent — no cross-row dependency | folded into `tile_page` (middle factor) | folded, as above | `block_tiles` | through `tile_page` | knob-turn |
| `tile_col` (`ceil(W/32)` per image) | independent — no cross-column dependency (a bf8b group is 16 columns *inside* one tile) | folded into `tile_page` (innermost factor) | folded, as above | `block_tiles` | through `tile_page` | knob-turn |
| **`tile_page`** — flattened page index `p ∈ [0, tensor_pages)`, `tensor_pages = input_tensor.buffer_num_pages() = prod(shape[:-2]) · ceil(H/32) · ceil(W/32)` | independent — union of the three axes above, all independent, contiguous in buffer page order | `block_tiles` (pages per block) | `min(core_tiles_g1, BLOCK_TILES_CAP)`, `BLOCK_TILES_CAP = 8` (overlap lamp, below) | host constant `BLOCK_TILES_CAP` in the program-descriptor module → `block_tiles` computed once on host → CT arg to all three kernels and the CB capacities | contiguous page ranges, `split_work_to_cores(grid, tensor_pages, row_wise=True)` over the full compute grid | knob-turn (cap, core count, page stride) |
| `tile_elements` (32×32 elements of one tile; contains the bf8b 16-value face-row groups) | independent per element, except a **dependent** 16-value group for bf8b output (shared exponent spans the group) | 1 whole tile (fixed) | 1 tile — the transfer unit (one TensorAccessor page), the unpack/pack unit, and the bf8b exponent scope; splitting a tile buys nothing at any production shape (the smallest has 1 tile, which is launch-latency-bound) | tile format (mechanism cap) | never split | none (rejected regime `sub_tile_split`) |
| `rm_row` (ROW_MAJOR only: one page per row, `prod(shape[:-1])` pages) | independent — elementwise | `block_rows` (pages per block) | not built in Phase 0 (ROW_MAJOR ∉ SUPPORTED); defined by regimes `rm_convert` / `rm_bitcopy` | host constant `BLOCK_ROWS_CAP` (refinement) | contiguous page ranges via `split_work_to_cores` (same code path as `tile_page`) | new regime, same topology |
| `rm_width` (ROW_MAJOR only: `W` values in one page) | independent — elementwise | whole page | not built in Phase 0; a page is one NoC transfer and stays whole (`W ≤ 4096` in INPUTS) | page size (mechanism cap) | not split across cores — single-page tensors (`(1, N)` RM cases, `(1,)`) run on one core, as the rule for ≤ 1 page allows | none |

Intermediate data: the only data a stage produces is the converted tile, which has exactly the axes of its input page and streams in the same order on the same core. There is no collection or coordinator stage, no core that waits for others, and no stage left on a designated core.

**Operand-reuse check:** one input operand; it varies along `tile_page`, the split axis, so no core re-reads bytes another core reads. Nothing is reuse-shared; no broadcast row exists.

**Stall-shadow check:** the waits are the reader's read barrier, compute's wait for an input block, and the writer's wait for an output block plus its write barrier. Buffer depth 2 already fills each wait: while the reader waits on block `k`'s reads, compute converts `k−1` and the writer drains `k−2`. No other work remains to move into a stall, and since no algebraic reordering applies, floating-point order cannot change.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `IN_DEPTH` (host constant) | 2 | the reader lands block `k+1` while compute converts block `k` |
| `cb_output_tiles` | `OUT_DEPTH` (host constant) | 2 | the writer drains block `k` while compute packs block `k+1` |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| CB ring wraps only on an exact `fifo_limit` hit | push/pop quantum vs capacity `DEPTH × block_tiles` | every non-final handshake on both CBs, in all three kernels, is exactly `block_tiles`; the ragged block is always the **last** handshake of the program on that CB (`last_block_extent`), so nothing follows it that could wrap | pointers miss the ring end → pages overwritten / garbage read |
| DEST capacity `DEST_AUTO_LIMIT` = 4 tiles (32-bit DEST, half-sync) / 8 (16-bit) — `dest_helpers.hpp:89-103` | the DEST group inside `convert_block`, **not** `block_tiles` | the eltwise chain clamps its `IterationShape::block_size` to `DEST_AUTO_LIMIT` at runtime (`chain.inl:3347`); pass `dest_group = DEST_AUTO_LIMIT` | none (helper clamps) |
| Typecast LLK: Float32/Int32/UInt32 endpoints need 32-bit DEST (`typecast.h:64`) | — | `fp32_dest_acc_en = in ∈ {float32, int32, uint32, uint16} or out ∈ {float32, int32, uint32, uint8}` (Phase 0: on for →float32 and →int32, off for →bfloat8_b) | int32 / float32 outputs read from 16-bit DEST → wrong values |
| SrcA path is 19 bits wide (refinement inputs) | — | `unpack_to_dest_mode[cb_input_tiles] = UnpackToDestFp32` iff input ∈ {float32, int32, uint32, uint16}; Phase 0 (bf16 input) keeps the default | low mantissa / integer bits dropped before conversion → wrong RNE ties, wrong int32 bits |
| One NoC transaction = one TensorAccessor page | per-transaction bytes = page size (bf16 2048, f32/int32 4096, bf8b 1088) | whole pages only | n/a |
| bf8b shared exponent covers 16 consecutive columns of one tile row | bf8b output with `W % 16 ≠ 0` (the boundary group mixes logical and padding columns) | none built — no TARGET cell reaches it (every bf8b INPUTS shape has `W = 256`); see Key Risks | coarser quantization if the input's padding columns are non-zero |
| `DataflowBuffer::{reserve_back,push_back,wait_front,pop_front}` take `uint16_t` counts (`dataflow_buffer.h:243-246`) | `block_tiles` | `BLOCK_TILES_CAP = 8` | silent wrap of the count |

### Regimes

Selection predicate, evaluated by `select_regime(layout, in_dtype, out_dtype)` on host (shape never selects a regime). `BITCOPY_PAIRS = {(uint32, int32), (int32, uint32)}`.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `tile_convert` | **built** (Phase 0: bf16 → f32 / bf8b / int32; the refinement pairs f32→bf16, i32→bf16 and u16→i32 land in this same regime by setting new CB formats and compute config) | `layout == TILE and (in, out) ∉ BITCOPY_PAIRS` | `block_tiles` consecutive pages of `tile_page` | **minimum**: each input page crosses its memory boundary (DRAM bank, or the NoC port of the L1 bank's core) exactly once into exactly one worker, and each output page crosses once out; no re-read, no spill, no inter-core traffic. Tile padding is moved because it is part of the layout (W = 12 → 12/32 of each tile row is logical); that is the tensor's own byte count, not an excess. | per block: 1 read barrier + 1 write barrier + 4 CB handshakes (reader reserve/push, writer wait/pop) and compute's 2 waits/2 pushes, intended **once per block**. Inits and format reconfig are **once per kernel**; DEST acquire/commit/release happens once per DEST group. More pages per barrier keep more reads in flight (catalog `double_buffer`: plateau at 4–8). |
| `tile_bitcopy` | deferred — Phase 0 SUPPORTED has no 32-bit integer input, and the regime's only TILE production cell is Q2230 (u32 → i32). Reachable: the same split, reader and writer with one CB (`cb_input_tiles`, reader → writer) and no compute kernel or `cb_output_tiles` | `layout == TILE and (in, out) ∈ BITCOPY_PAIRS` | `block_tiles` pages | minimum (each tensor crosses DRAM once) | barriers + handshakes per block, as above, minus compute |
| `rm_convert` (stepping stone: per-page slots) | deferred — ROW_MAJOR is outside Phase 0 SUPPORTED by contract (F317/F322 bf16→u8 are refinement cells). Reachable: same three-kernel topology and the same `split_work_to_cores` code, applied over `rm_row`. Only the reader/writer address walk (page → slot) and the CB formats differ | `layout == ROW_MAJOR and (in, out) ∉ BITCOPY_PAIRS` | `block_rows` pages laid into fixed-stride slots of tile-sized CB pages. Slot stride `S_in` is a multiple of the 64 B DRAM read alignment (BH), so every page lands at a matching offset. Compute converts whole pseudo-tiles: unpack and pack apply the same face mapping, so element `k` of the input page buffer becomes element `k` of the output page buffer, and no tilize/untilize is needed. The output slot stride `S_out = S_in · out_elem / in_elem` must be a multiple of 16 B (BH DRAM-write / L1 alignment) | bytes: minimum (each page read once, written once). **Transactions above the byte minimum are structural:** one NoC read per input page and one write per output page, 4,194,304 each for F322. Compute also converts the slot padding (`S_in / (W·in_elem)` = 64/6 ≈ 10.7× the logical elements for F322) | per block: barriers + handshakes; the per-page transaction cost is **not** amortized by the block |
| `rm_convert_bank_run` | deferred — successor of `rm_convert`, which it is layered on (same slots and compute). Reason: ROW_MAJOR is outside Phase 0, and it can only be measured on top of the stepping stone | same as `rm_convert`, selected by a host flag once measured | as `rm_convert`, but the reader reads a run of `R` consecutive slots of one DRAM bank per transaction (pages `b, b+NB, b+2·NB, …` are contiguous in bank `b` at `aligned_page_size` stride) | bytes: the alignment padding of each 64 B DRAM slot is read too (the tensor's own allocation); read transactions drop `R×`. Writes stay per page unless `S_out` equals the output's aligned page size | read transaction count ÷ `R` |
| `rm_bitcopy` | deferred — ROW_MAJOR is outside Phase 0. Production: U32→I32 `(1, N)` (one page each) and I32→U32 `(1,)`. Reachable: the `tile_bitcopy` kernels with RM page size and aligned stride | `layout == ROW_MAJOR and (in, out) ∈ BITCOPY_PAIRS` | `block_rows` pages | minimum | barriers + handshakes per block |
| `tile_convert_bank_local` | deferred — the cheapest-traffic split for **L1-placed** tensors: each core takes the pages of the L1 bank it hosts (page stride `NB_L1`), so reads and writes stay core-local. Positive reason: the built `tile_convert` already serves every L1 shape (P22/P31/P39/P46/P50) with the same kernels. The single delta is the NoC-mesh crossing of both tensors, which competes with compute only when per-core NoC throughput, not unpack/pack, is the bound — on P39 both sit at ~63 tiles per core. Reachable: Phase 0's reader/writer walk pages as `first_page + i · page_stride` with `page_stride` an RT arg (Phase 0 value 1); this regime passes `page_stride = NB_L1` and assigns bank-owner cores | `buffer == "l1"` and a host flag | `block_tiles` pages, strided by `NB_L1` | removes all mesh traffic: P39 24.5 MiB of mesh crossings → 0 (each core touches only its own bank) | as `tile_convert` |
| `tile_convert_single_core` | rejected — superseded by `tile_convert` with `num_cores = 1`; a single-core scheme is not a design | — | — | same bytes, 1/G the parallelism | — |
| `dram_staged_layout_conversion` (tilize/untilize or pad through a DRAM copy) | rejected — dead end: each tensor crosses DRAM 2–3 times, the rules forbid it, and `rm_convert` replaces it rather than building on it | — | — | ≥ 2× the minimum per tensor | — |
| `sub_tile_split` (faces of one tile across cores) | rejected — the tile is the transfer, unpack and bf8b exponent unit. Catalog `tile_reorder` shows whole-tile transfers at least as fast as face transfers, and the 1-tile shapes are launch-bound, not bandwidth-bound | — | — | same bytes, 4× the transactions | — |

Regime-pinned tests: each refinement that builds a regime adds `describe_plan(...)["regime"]` assertions on its production cells. Phase 0's acceptance test pins `tile_convert`.

### Traffic ranking

| Candidate split | DRAM crossings per tensor | Cross-core (mesh) traffic | Fan-out | Verdict |
|-----------------|---------------------------|---------------------------|---------|---------|
| contiguous `tile_page` ranges over the full grid (chosen) | 1 in, 1 out | DRAM placement: none. L1 placement: each page crosses the mesh once from its bank core | 1 reader per page | chosen: minimum DRAM bytes, finest balance (±1 page per core) |
| whole `tile_row` ranges | 1 / 1 | same | 1 | same bytes; worse balance (a `(1,1,32,128)` tensor has one tile row → one core) |
| `tile_col` (width) ranges | 1 / 1 | same | 1 | same bytes; strided page walk per core, no gain — every axis is independent |
| bank-local pages (L1 placement) | 0 DRAM (tensors in L1) | **none** | 1 | cheapest for L1; deferred row `tile_convert_bank_local` |
| DRAM-bank-affine pages (DRAM placement) | 1 / 1 | none | 1 | same bytes, different NoC paths — perf lamp (catalog `read_coalescing` affinity: 0.91–1.28×) |
| split a dependent axis + combine | — | — | — | no dependent axis exists (the bf8b group is intra-tile, owned by the packer) |

All DRAM-placed splits move the same bytes, so for DRAM placement the choice is the one that fills the grid with the finest balance. Occupancy is the tiebreaker, not the reason.

### Block schedule

```cpp
// per core: first_page, core_tiles (RT); block_tiles (CT)
// num_blocks_this_core = ceil(core_tiles / block_tiles)
// last_block_extent    = core_tiles - (num_blocks_this_core - 1) * block_tiles
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    // block_extent = (block_idx + 1 == num_blocks_this_core) ? last_block_extent : block_tiles
    load_block(block_idx);      // reader
    convert_block(block_idx);   // compute
    store_block(block_idx);     // writer
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `load_block` | `block_extent` pages `first_page + (block_idx·block_tiles + j)·page_stride`, `j < block_extent` | the other block slot of `cb_input_tiles` (still being converted) | 1 `reserve_back(block_extent)`, `block_extent` async page reads, **1** read barrier, 1 `push_back(block_extent)` per block |
| `convert_block` | `block_extent` tiles, walked internally in DEST groups of `dest_group = DEST_AUTO_LIMIT` | nothing (no state crosses blocks) | input wait / output reserve **once per block** (`block_extent`), pop / push once per block; `compute_kernel_hw_startup` once per kernel; copy/typecast init and data-format reconfig **once per kernel** (hoisted; formats never change); DEST acquire/commit/wait/release once per DEST group |
| `store_block` | `block_extent` pages to the same page ids in the output | the other block slot of `cb_output_tiles` | 1 `wait_front(block_extent)`, `block_extent` async page writes, 1 write completion wait, 1 `pop_front(block_extent)` per block; **final `noc_async_write_barrier()` before return** (BB-03) |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Overlap** — `BLOCK_TILES_CAP = 8`, below the whole per-core assignment | The bound default (one block = all of a core's pages, up to 63 on P39 and 16 on Q2185) serializes read → convert → write on each core and leaves DRAM idle during the convert gap. Catalog `double_buffer`: 4–8 pages per barrier with a 2× CB reach the single-core NoC limit and bigger blocks add nothing. `trid_double_issue`: at a large CB the one-block loop is within 3% of the best | `BLOCK_TILES_CAP ∈ {4, 8, 16, 32, ∞ (whole core_tiles)}` × `DEPTH ∈ {2, 3}` on Q2185 (DRAM, f32), P39 (L1, bf8b), Q2187 (DRAM, bf8b) |
| **Grid synchronization** — every page spread to its own core for small tensors | No collective exists, so the only per-core cost is launch plus kernel start. Decode shapes (1–16 pages) run 1 page per core | cap active cores at `ceil(tensor_pages / 4)` for `tensor_pages ≤ 16` (Q2191, Q2216, Q2229) vs full spread |
| Bank stagger | Contiguous ranges whose `first_page % NB_DRAM` coincide start every core on the same DRAM bank (catalog `bank_stagger`, BH height split: 1.05–1.32×) | rotate each core's block walk start by `core_idx % num_blocks_this_core` |
| NoC placement | `row_wise=True` lays partial-grid spreads along the DRAM-facing axis (catalog `noc_placement`: column line 2.9× slower) | `row_wise=False` on the decode shapes |
| Barrier granularity | one drain-all read barrier per block idles the NoC at every block boundary | transaction-id ring (`set_read_trid` / `async_read_barrier_with_trid`, `local_copy_helpers_dataflow.hpp:147,158`), depth 3–4 |
| Init hoisting | if `convert_block` is realized as one chain call per block with `InitReconfigOwner::Chain`, inits repeat per block (catalog `compute_block_size`: ~1.6 µs per extra pass on WH for multi-phase chains; this chain has one phase) | `eltwise_chain<InitReconfigOwner::Caller>` with the setup emitted once, vs per-block init |

### Catalog evidence per knob

| Knob | Catalog entry (`ttnn/ttnn/operations/examples/…`) | What it set |
|------|---------------------------------------------------|-------------|
| `BLOCK_TILES_CAP = 8`, reads/writes per barrier | `double_buffer/README.md` | batching plateaus at 4–8 pages per barrier; bigger blocks only cost L1; no gain once DRAM-bound |
| `IN_DEPTH = OUT_DEPTH = 2` | `double_buffer/README.md` | depth 2 overlaps the reader (NoC0) with the writer (NoC1); depth 1 caps throughput |
| core assignment `row_wise=True`, reader NoC0 / writer NoC1 | `noc_placement/README.md` | row spreads beat column spreads; the default NoC split is the fast one |
| whole-tile transfers, no sub-tile split | `tile_reorder/README.md` | whole-page transactions are at least as fast as faces on a DRAM-bound move |
| bank stagger, trid ring, bank affinity (lamps only) | `bank_stagger`, `trid_double_issue`, `read_coalescing` | measured 1.0–1.3× effects that depend on shape; candidates to measure, not defaults |
| init once per kernel | `compute_block_size/README.md` | per-call init/reconfig is the fixed cost a block amortizes |
| RM per-page transaction cost (deferred `rm_convert`) | `double_buffer` (≈115 ns per tile transaction per core), `split_reader`, `read_coalescing` | per-transaction issue cost, not bandwidth, dominates 6 B pages; coalescing and a split reader are the levers |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| input pages (DRAM or L1 interleaved) → `cb_input_tiles` | input tile format (Phase 0: Float16_b, 2048 B) | reader on NCRISC / NoC0: `TensorAccessor` page reads, `block_extent` reads then one barrier | the L1 placement is the same code: `TensorAccessor` resolves the remote L1 bank |
| `cb_input_tiles` → DEST | DEST Float32 (→f32, →int32) or Float16_b (→bf8b) | unpacker via SrcA (bf16 is exact in SrcA), copy to DEST | 32-bit inputs (refinement) unpack straight to DEST (`UnpackToDestFp32`) |
| DEST → DEST | — | SFPU `typecast_tile<in_df, out_df>` (no-op for →f32 and →bf8b; fp32→int32 for →int32) | `typecast.h:91` |
| DEST → `cb_output_tiles` | output format: Float32 4096 B / Bfp8_b 1088 B / Int32 4096 B | packer | bf8b exponent sharing is done by the packer, per 16-value face row |
| `cb_output_tiles` → output pages | output tile format | writer on BRISC / NoC1: `TensorAccessor` page writes, `block_extent` writes then one completion wait; final barrier | same page ids as the input pages |

Placement axis (`buffer`): TARGET has only interleaved placements (`dram`, `l1`). Both run the same regime rows; going from one to the other is a placement change resolved by `TensorAccessor`, not a new algorithm. No regime, built or deferred, communicates between Tensix cores, so there is no Tensix-to-Tensix contract to reserve. No physical shard exists in TARGET.

### TARGET − SUPPORTED: how each value unlocks

| Axis value | Production cells | Regime | Unlock class | What changes |
|------------|------------------|--------|--------------|--------------|
| `dtype=float32` (→ bfloat16, TILE) | F315, F316, F318–F320, F323, F324 | `tile_convert` | knob-turn | `cb_input_tiles` format Float32; `unpack_to_dest_mode=UnpackToDestFp32`; `fp32_dest_acc_en=1`; SFPU fp32→fp16b |
| `dtype=int32` (→ bfloat16, TILE) | Q2228 | `tile_convert` | knob-turn | Int32 input, unpack-to-DEST, SFPU int32→fp16b |
| `dtype=uint16` (→ int32, TILE) | Q2231 | `tile_convert` | knob-turn | UInt16 input, unpack-to-DEST, SFPU uint16→uint32 |
| `dtype=uint32` (→ int32) | Q2230 (TILE), Q2189 … Q2234 (RM) | `tile_bitcopy`, `rm_bitcopy` | new regime, same split/reader/writer | drop the compute kernel and `cb_output_tiles` |
| `dtype=int32` (→ uint32, RM) | F321 | `rm_bitcopy` | new regime | as above, one 4 B page |
| `layout=ROW_MAJOR` (with `output_dtype` bfloat16 / uint32 / uint8) | F317, F322 (bf16 → uint8); RM bitcopy cells | `rm_convert`, `rm_bitcopy` | new regime, same topology | page → slot walk, pseudo-tile CB formats, `block_rows` knob |
| `output_dtype=uint8` | F317, F322 | `rm_convert` | with ROW_MAJOR | `unary<Typecast<Float16_b, UInt8>>`, `fp32_dest_acc_en=1` |
| `rank=1, 2` | F321 (rank 1), RM `(1, N)` cells (rank 2) | any | knob-turn for TILE (the page count already folds rank); arrives with the RM regimes for the production cells | `SUPPORTED["rank"]` |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block = `block_extent ≤ block_tiles` consecutive pages of `tile_page` |
| Grid | `grid = device.compute_with_storage_grid_size()` (the full worker grid; `sub_core_grids` is `None` only); `G = grid.x · grid.y` |
| Split | `(num_cores, all_cores, core_group_1, core_group_2, core_tiles_g1, core_tiles_g2) = ttnn.split_work_to_cores(grid, tensor_pages, row_wise=True)`; `num_cores = min(tensor_pages, G)` |
| Core order and ranges | `cores = ttnn.corerange_to_cores(core_group_1, row_wise=True) + ttnn.corerange_to_cores(core_group_2, row_wise=True)`; core `i` gets `core_tiles_i` (g1 or g2 value) pages starting at `first_page_i = Σ_{j<i} core_tiles_j`; ranges are disjoint and cover `[0, tensor_pages)` |
| Per-core RT args | reader: `(first_page, core_tiles, page_stride=1)`; writer: same; compute: `(core_tiles)`. Buffer addresses are **common** RT args (`input.buffer_address()` for the reader, `output.buffer_address()` for the writer), so a program-cache hit patches 1 value per kernel |
| CT args | `block_tiles`, CB ids, `TensorAccessorArgs` (input for reader, output for writer); compute: `block_tiles` |
| Kernel groups | one `KernelDescriptor` per role over `all_cores`. Per-core counts are RT args, so group-1 and group-2 cores share one binary. With `row_wise=True`, `all_cores` is ≤ 2 rectangles (full rows + one partial row) (BB-17) |
| Remainder | `tensor_pages = input_tensor.buffer_num_pages()` already is `prod(shape[:-2]) · ceil(H/32) · ceil(W/32)` with per-image padding; the host asserts `output_tensor.buffer_num_pages() == tensor_pages`. Never derive pages as `floor(volume / 1024)` |
| Program cache (BB-07, BB-16) | the Python module caches the built `ProgramDescriptor` keyed by `(device id, tuple(shape), tuple(padded_shape), in dtype, out dtype, layout, memory_config, grid)`. On a hit it patches only the two common-RT-arg addresses and calls `ttnn.generic_op([input, output], descriptor)`. The device-side hash (`ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:78-100`) excludes RT values, so patched addresses replay the cached program |
| Profiling (BB-21) | named zones per block op and per blocking wait: reader `TC-RD-WAIT` (reserve), `TC-RD-ISSUE`, `TC-RD-BARRIER`; compute `TC-CV-WAIT` (an explicit `wait_front(block_extent)` issued only under `#if defined(PROFILE_KERNEL)`, so the chain's own wait returns at once), `TC-CV-BLOCK`; writer `TC-WR-WAIT`, `TC-WR-ISSUE`, `TC-WR-BARRIER`. `describe_plan()` dumps each core's role. The plain build is unchanged |

Example extents: Q2185 `(16, 4096, 12)` → 2048 pages → `ceil(2048/G)` pages per core (≈ 16 at G ≈ 130) → 2 blocks of ≤ 8. P39 `(1, 128, 256, 256)` → 8192 pages → ≈ 63 per core → 8 blocks, the last one ragged. Q2227 `(1, 1, 12)` → 1 page → 1 core.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_tiles` | 0 | `input_tensor.buffer_page_size()` (bf16 tile: 2048 B) | `IN_DEPTH · block_tiles` | spans `tile_page` (`block_tiles`) × depth; streams over the rest of the core's range; `tile_elements` = 1 whole tile per page | input dtype (Phase 0 `bfloat16` → Float16_b) | reader | compute | whole program |
| `cb_output_tiles` | 1 | `output_tensor.buffer_page_size()` (f32/int32 4096 B, bf8b 1088 B) | `OUT_DEPTH · block_tiles` | same as above | output dtype (`float32` / `bfloat8_b` / `int32`) | compute | writer | whole program |

Two CBs is the minimum inventory for a converting regime: the formats and page sizes differ (2048 vs 4096 / 1088 B), so an in-place transform is impossible, and depth-2 pipelining keeps both live at once. The bitcopy regimes use `cb_input_tiles` alone (reader → writer). Full ledger: `l1_ledger.md`.

Compute config: `ComputeConfigDescriptor(fp32_dest_acc_en = <per Mechanism caps>, math_approx_mode = False, dst_full_sync_en = False)`. `math_fidelity` is left at its default because no FPU math runs; `bfp8_pack_precise` stays at its default (False) unless the bf8b group check fails (Key Risks). `unpack_to_dest_mode` is default in Phase 0.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_extent` pages | raw dataflow API (no kernel_lib helper reads interleaved tile pages; see API Mapping) | — (DRAM / L1 pages via `TensorAccessor`) | `cb_input_tiles`, `block_extent` pushed | `block_extent` pages visible to compute |
| 2 | `convert_block` | `block_extent` tiles; DEST groups of `dest_group` | yes: `compute_kernel_lib::typecast` → `eltwise_chain(CopyTile, Typecast<in_df,out_df>, PackTile)` | `cb_input_tiles`, wait `block_extent` (Upfront), popped `block_extent` at end | `cb_output_tiles`, reserve `block_extent` (Upfront), push `block_extent` at end | input slot freed for `load_block(k+2)`; output visible to writer |
| 3 | `store_block` | `block_extent` pages | raw dataflow API | `cb_output_tiles`, wait `block_extent`, popped `block_extent` after the writes complete | — (output pages) | output slot freed |

CB sync per block: reader pushes `block_extent` = compute waits `block_extent`; compute pushes `block_extent` = writer waits `block_extent`. Over a core: `Σ block_extent = core_tiles` on every side.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| host split | host | `ttnn.split_work_to_cores(grid_size, units_to_divide, row_wise)` | `ttnn/cpp/ttnn-nanobind/operations/core.cpp:470-477` | `(grid, tensor_pages, row_wise=True)` | — | — | core count (grid) |
| host core order | host | `ttnn.corerange_to_cores(crs, row_wise=True)` | `ttnn/cpp/ttnn-nanobind/tensor.cpp:660` | per group | — | — | — |
| `load_block` | raw_api | `TensorAccessor(args, addr, page_size)`, `noc_async_read(acc.get_noc_addr(page), l1, page_size)`, `noc_async_read_barrier()`, `DataflowBuffer::reserve_back/push_back/get_write_ptr` | pattern: `ttnn/ttnn/operations/examples/double_buffer/kernels/db_reader.cpp:36-54`; `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:243-246,407` | `TensorAccessorArgs<ct_base>()`; page = `first_page + (block_idx·block_tiles + j)·page_stride` | — | `cb_input_tiles` | `block_tiles` (CT), `page_stride` (RT) |
| `convert_block` setup | helper prerequisite | `compute_kernel_hw_startup(icb, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:113` | `(cb_input_tiles, cb_output_tiles)`, first statement of `kernel_main` | — | — | — |
| `convert_block` | helper | `compute_kernel_lib::typecast<Input, Output>(shape, in, out)` → `unary<Typecast<in_df,out_df>>` → `eltwise_chain(CopyTile, Typecast, PackTile)` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:89-91`, `convenience.inl:48-57`; `eltwise/unary/misc.hpp:43-45`, `misc.inl:85-90`; `eltwise/api/chain.hpp:620` | `Input = input(cb_input_tiles, WaitPolicy::Upfront, PopPolicy::AtEnd, InputTileMapping::Block)` (`chain.hpp:385`); `Output = output(cb_output_tiles, ReservePolicy::Upfront, PushPolicy::AtEnd)` (`chain.hpp:442`); `shape = IterationShape::tiles(block_extent).block_size(dest_group)` (`chain.hpp:146`). `in_df`/`out_df` are derived from the CB formats (`dfb_helpers_compute.inl:107-114`) | `cb_input_tiles` | `cb_output_tiles` | `block_extent` (from `block_tiles`); `dest_group` is the chain's internal DEST grouping, clamped to `DEST_AUTO_LIMIT` |
| `convert_block` init hoist (optional realization) | helper | `eltwise_chain<InitReconfigOwner::Caller>(...)` | `eltwise/api/chain.hpp:189-192,620` | caller emits the chain's one-time setup once before the block loop | same | same | — |
| `convert_block` math | LLK (inside helper) | `typecast_tile<IN,OUT>` / `typecast_tile_init<IN,OUT>` | `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h:91,150,158,315,463` | Float16_b→{Float32, Int32, Bfp8_b} | — | — | — |
| `store_block` | raw_api | `noc_async_write(l1, acc.get_noc_addr(page), page_size)`, `noc_async_write_barrier()`, `DataflowBuffer::wait_front/pop_front/get_read_ptr` | pattern: `ttnn/ttnn/operations/examples/double_buffer/kernels/db_writer.cpp:29-41`; `dataflow_buffer.h:243-246,408` | same page walk as the reader, over the output accessor | `cb_output_tiles` | — | `block_tiles`, `page_stride` |

**Helpers considered and rejected for the `raw_api` rows (`load_block`, `store_block`):**

| Candidate | File:Line | Why it cannot express the shape |
|-----------|-----------|---------------------------------|
| `read_sticks_for_tilize` / `write_sticks_after_untilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:87-94,129-134` | they move ROW_MAJOR sticks to feed a tilize / drain an untilize; the TILE regime moves whole tile pages and does no tilize |
| `local_addr` / `set_read_state` / `read_with_state` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57,75,95` | self-aimed L1 → L1 copies only (the source is this core's own L1); interleaved pages live in DRAM banks or remote L1 banks |
| `McastArgs` / `ttnn.Mcast` | `ttnn/cpp/ttnn/kernel_lib/mcast/` | no operand is shared between cores, so there is nothing to multicast |

**Compute-helper gotchas:** the default `InputTileMapping` in `input(...)` is `Scalar` (`chain.hpp:382`). With an upfront or blocked lifecycle, every lane would then read tile 0, so `InputTileMapping::Block` is mandatory. The convenience `typecast` static-asserts against UInt8 CBs (`convenience.inl:53-55`), so the refinement's bf16 → uint8 uses `unary<Typecast<Float16_b, UInt8>, Input, Output>` with explicit formats.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| int32 / float32 output from a 16-bit DEST | `typecast.h:64` requires 32-bit DEST for Int32/Float32 endpoints; without it the SFPU stores INT32 into a 16-bit DEST | `fp32_dest_acc_en` derived from the pair (Mechanism caps); on for →f32 and →int32 in Phase 0 |
| Output CB page size taken from the input | bf8b tile = 1088 B and f32 tile = 4096 B ≠ bf16 2048 B; using the input size corrupts every page after the first | `cb_output_tiles` page size = `output_tensor.buffer_page_size()`; the writer's accessor uses the output tensor |
| Ragged last block mismatched across kernels | if one side pushes the nominal `block_tiles` and the other waits `last_block_extent`, the program hangs or leaves the ring inconsistent | all three kernels derive `num_blocks_this_core` / `last_block_extent` from the same `core_tiles` (RT) and `block_tiles` (CT). Every non-final handshake is `block_tiles`; the final one is `last_block_extent` and is the last handshake on that CB |
| Page ranges built in a different core order than the RT-arg assignment | overlapping or missing pages, which pass for shapes where `tensor_pages ≤ G` and fail on the large ones | one ordered list `cores` (g1 then g2, row-wise) drives both `first_page` accumulation and RT-arg placement |
| `InputTileMapping` left at the default `Scalar` | every DEST lane of a group converts tile 0 of the block | pass `InputTileMapping::Block` (API Mapping) |
| bf8b group exponent at a `W % 16 ≠ 0` boundary | the packer's shared exponent covers the 16 columns of a face row, padding included. With non-zero input padding (producer-dependent) the logical values of that group are quantized more coarsely than the host rendering | not reachable by any TARGET cell (all bf8b INPUTS have `W = 256`); tensors from `from_torch` / ttnn tilize carry zero padding. If a production bf8b shape with `W % 16 ≠ 0` appears, the reader zeroes columns `[W % 16, 16)` of the boundary face rows of each last-tile-column page before `push_back` |
| bf8b packer rounding vs. the host rendering | the packer's mantissa rounding may differ from `ttnn.from_torch`'s | grading allows one quantization step per group; if it is exceeded, set `bfp8_pack_precise=True` in the compute config |
| 32-bit inputs through SrcA (refinement) | SrcA keeps 19 bits, so f32 → bf16 rounds from a truncated mantissa and int32 / uint32 lose bits | `unpack_to_dest_mode[cb_input_tiles] = UnpackToDestFp32` for f32 / int32 / uint32 / uint16 inputs; validate with the golden witnesses (exact ties `1+2^-8`; `33685505 → 2^25`) |
| u32 ↔ i32 through compute | no LLK pair exists; the packer may reinterpret the sign | the `*_bitcopy` regimes never touch DEST |
| Program rebuilt every call | Python descriptor construction per call costs milliseconds on 100+ cores (BB-07) | module-level descriptor cache with only common-RT address patches on a hit; the acceptance test asserts the device program-cache entry count is stable across repeated calls |
| Writes in flight at kernel exit | the next program in the model hangs (BB-03) | the writer ends with `noc_async_write_barrier()`; the reader's last block ends with a read barrier before `push_back` |
| Refinement rectangle admits non-production pairs | once `dtype` and `output_dtype` grow, the SUPPORTED rectangle includes pairs no INPUTS cell exercises (e.g. f32 → int32), and EXCLUSIONS must stay empty | the program factory derives formats, `fp32_dest_acc_en` and `unpack_to_dest_mode` generically from `(in, out)`; every pair the LLK `typecast_tile` lists then works through the same `tile_convert` path |
