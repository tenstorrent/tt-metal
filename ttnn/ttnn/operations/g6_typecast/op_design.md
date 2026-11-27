# Operation Design: g6_typecast

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (elementwise format conversion; one device program) |
| Goal | Convert every value of one interleaved tensor to `dtype`, value-for-value equal to `torch.Tensor.to(dtype)` (bfloat8_b: within one quantization step of the host rendering), keeping shape, layout and memory placement, in exactly one `ttnn.generic_op` dispatch. |
| Math | `out[i] = conv(x[i])` for every logical index `i`; `conv` per (source, destination) pair: see *Conversion Pair Table*. Phase 0: `bf16→fp32: v` (exact), `bf16→bfp8_b: v in the block format` (≤ 1 step of its 16-value group), `bf16→int32: trunc(v)` (toward zero). |
| References | `eval/golden_tests/g6_typecast/feature_spec.py` (TARGET / INPUTS / INVALID / LOOSE_CASES), `eval/golden_tests/g6_typecast/helpers.py` (oracle, `TOLERANCES`, checks), `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp`, `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h`, `tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h`, `ttnn/ttnn/operations/examples/master.md` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 1–4 (Phase 0: 3, 4); interleaved DRAM or L1; layout TILE (Phase 0) | — | buffer address RT; shape → per-core RT counts |
| `dtype` | `ttnn.DataType` (positional or keyword) | yes | TARGET `output_dtype` (Phase 0: `float32`, `bfloat8_b`, `int32`) | — | CT (CB data formats, compute config) |
| `sub_core_grids` | `ttnn.CoreRangeSet \| None` (keyword-only) | no | `None` only | `None` | — (any other value raises `ValueError`) |

Derived host knobs (single source each, all in the program-descriptor builder): `block_tiles`, `in_buffer_depth`, `out_buffer_depth`, `core_grid`, `min_tiles_per_core`, `page_stride`, `fp32_dest_acc_en`, `bfp8_pack_precise`, `unpack_to_dest_mode` — defined in the Blocking Model and the Conversion Pair Table.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | any rank 1–4; H, W need not be multiples of 32 (Phase 0: rank 3, 4) |
| Dtype | TARGET `dtype` (Phase 0: `bfloat16`) |
| Layout | TILE (Phase 0); ROW_MAJOR in refinement (regimes R2/R3) |
| Memory | interleaved, buffer type DRAM or L1 (both Phase 0). Sharded memory layouts are outside the contract → `ValueError` |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` (same logical shape; same padded tile grid in TILE layout) |
| Dtype | `dtype` |
| Layout | `input_tensor.layout` |
| Memory | `input_tensor.memory_config()` (interleaved, same buffer type) — allocated with `ttnn.allocate_tensor_on_device(input.shape, dtype, input.layout, device, input.memory_config())` |

Tile padding of the output is unspecified (positions beyond the logical shape are not part of the result).

## Blocking Model

The first design section. Everything below it is downstream of these decisions.
Semantics: `.claude/references/blocking-model.md`.

The op is a per-element map with no reuse and no reduction. In TILE layout the input and the output have the
same padded tile grid, and tile page `p` of the output is `conv` applied to tile page `p` of the input
(unpack and pack preserve each element's position inside the tile image). So the leading axes fold exactly into
one **page** axis in the tensor's own page order (contiguous, identical treatment, no per-row state). A block is a
run of `block_tiles` consecutive pages owned by one core.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` = ∏ `shape[:-2]` (1 for rank ≤ 2) | independent — no output value reads another image | folded into `page`: carried by `block_tiles` | blocks may span image boundaries (no per-image state exists) | — (folded) | via `page` | knob-turn |
| `tile_row` = ⌈H/32⌉ per image (H = 1 for rank 1) | independent — elementwise | folded into `page`: carried by `block_tiles` | blocks may span tile rows | — (folded) | via `page` | knob-turn |
| `tile_col` = ⌈W/32⌉ | independent — elementwise (W = 12 → one tile column; a width split would idle the grid) | folded into `page`: carried by `block_tiles` | blocks may span tile columns | — (folded) | via `page` | knob-turn |
| `page` = `batch × tile_row × tile_col` (TILE page order) | independent — page `p` out depends only on page `p` in | `block_tiles` | `min(dest_capacity_tiles, max_core_tiles)`: **4** for 32-bit-DEST pairs (`bf16→fp32`, `bf16→int32`), **8** for `bf16→bfp8_b` (16-bit DEST); smaller only when a core owns fewer pages | host `_block_tiles()` in the descriptor builder → one CT arg `block_tiles` to reader, compute and writer; CB capacities derived from it | contiguous page ranges, `ttnn.split_work_to_cores(core_grid, tensor_tiles, row_wise=True)` over the full grid; per-core `(start_page, core_tiles, page_stride)` RT args | knob-turn (larger blocks, more cores, bank-local walk R4 = RT-arg change) |
| `elem` = 32×32 inside one tile | independent per element, **except** `bfp8_b` output: dependent inside a 16-value face-row group (shared exponent) — resolved entirely by the packer | `tile` (fixed) | **1 tile** — the tile is the indivisible unit of the NoC page, of unpack/pack, and of the bfp8_b group geometry; it is never split | hardware unit | not split | none |

`tensor_tiles = batch · ⌈H/32⌉ · ⌈W/32⌉` (per-image ceil; never `floor(batch·H/32)`), cross-checked against
`input_tensor.buffer_num_pages()` and `output.buffer_num_pages()` (must be equal). `max_core_tiles` = the largest
per-core count `split_work_to_cores` returns. `dest_capacity_tiles` = `DEST_AUTO_LIMIT / lane_width` of the chain
(`dest_helpers.hpp:89-103`, half-sync: 4 with `fp32_dest_acc_en`, 8 without; `Typecast<…, Dst::D0>` has lane width 1).

**Every stage's own data.** The scheme has one stage per core (read → convert → write); its only produced data is
the output pages, which carry the same `page` axis and are written by the core that converted them. There is no
intermediate tensor, no collector and no coordinator core — every active core does an equal (±1 page) share of
every phase.

**Operand-reuse check.** One input operand; it varies along `page`, the split axis. Nothing is re-read by more
than one core → no reuse-shared operand, no broadcast row needed.

**Stall-shadow check.** The waits are: reader on its read barrier (DRAM/L1 round trip), compute on
`cb_input_tiles`, writer on `cb_output_tiles` and on its final write barrier. The only work independent of a
pending read is *the next block's reads* — filled by `in_buffer_depth = 2` (reader runs one block ahead) and,
beyond that, by the trid/batched-barrier lamp. Compute has nothing independent of its input (every output tile
depends on exactly its own input tile); no algebraic reordering applies (a per-element map has nothing to commute
past). No floating-point order is changed anywhere.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `in_buffer_depth` (host constant) | 2 | reader fills block k+1 (reads in flight) while compute converts block k |
| `cb_output_tiles` | `out_buffer_depth` (host constant) | 2 | compute packs block k+1 while the writer drains block k |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `eltwise_chain` per-block lifecycle (`WaitPolicy/PopPolicy::PerBlockSize`, `ReservePolicy/PushPolicy::PerBlockSize`) maps one CB handshake to one DEST acquire in which **all `block_tiles` tiles are live at once** (`chain.hpp:497-509`; clamp at `chain.inl:3339-3350`) | `block_tiles ≤ chain_max_block_v = DEST_AUTO_LIMIT / lane_width` = 4 (32-bit DEST) / 8 (16-bit DEST) | host: `block_tiles = min(dest_capacity_tiles, max_core_tiles)`; compute kernel: `static_assert(block_tiles <= chain_max_block_v<…>)` | debug: `ASSERT` in the chain (FullBlock); release: the chain silently clamps its block, so its sync quantum drops below the reader/writer quantum → `cb_wait_front` never satisfied (hang) or ring pointer overrun |
| 32-bit DEST for any 32-bit endpoint (`typecast.h:64`: "For input/output to be UInt32, Int32, or Float32, Dest must be in 32 bit mode") | sets `fp32_dest_acc_en`, which halves `dest_capacity_tiles` | `fp32_dest_acc_en` derived per pair (Conversion Pair Table); `block_tiles` re-derived from it | int32 result stored into 16-bit DEST loses its high bits (wrong integers) |
| Tile atomicity (NoC page = tile; unpack/pack unit; bfp8_b group = 16 values of one face row) | `elem` extent = 1 whole tile; page-axis granularity = 1 tile | no sub-tile split anywhere | a split tile misaddresses faces and breaks bfp8_b groups |
| bfp8_b group content: the shared exponent is the max over the 16 values the packer sees, padding included | not an extent cap — a content requirement on the last group of a row when `W % 16 ≠ 0` | Phase 0: unreachable (every bfp8_b production case has `W = 256`, so no group mixes logical and padding values); regime R1m below | padding written non-zero by an upstream producer would set a logical group's exponent |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 `tile_stream_convert`** | **built** (Phase 0 pairs `bf16→{fp32, bfp8_b, int32}`; every other non-bit-identity TILE pair of the Conversion Pair Table is a compile-time knob-turn of the same kernels) | `layout == TILE` and `(dtype, output_dtype) ∉ BIT_IDENTITY_PAIRS` and not R1m | `block_tiles` consecutive pages of `page` | **minimum**, named boundary = the tensor's interleaved bank (DRAM bank or L1 bank): each input page crosses it once, each output page once. Whole padded tiles move because the page *is* the tile (W = 12 → 12/32 of each tile is logical; structurally unavoidable). No cross-core traffic. | per block: 1 reader barrier, 1 reserve/push + 1 wait/pop per CB, 1 DEST acquire/commit/wait/release, 1 writer flush. Inits, format reconfig and `compute_kernel_hw_startup` are once per **core** (one helper call spans all blocks), not per block. Domain: all 50 TILE production cases except Q2230 (`uint32→int32`, R2); 40 of them are Phase 0. No other regime applies there: no dependent axis (no combine), no shared operand (no mcast), interleaved pages (no resident shard to consume in place). |
| R1m `tile_stream_convert + pad-column zeroing` | deferred — unreachable for this op's production shapes: all 17 bfp8_b cases have `W = 256` (a multiple of 16), so no 16-value group mixes logical and padding values | `output_dtype == bfloat8_b` and `W % 16 ≠ 0` | as R1 | as R1 (zeroing is core-local) | as R1. Reachable: the reader zeroes columns `≥ W mod 32` of the last-tile-column pages in `cb_input_tiles` before `cb_push_back` (CT flag); compute and writer unchanged |
| R2 `bit_identity_copy` | deferred — inputs `uint32` / `int32` are outside Phase 0 SUPPORTED (contract phase plan), and R1 cannot serve these pairs: the LLK typecast table has no `UInt32↔Int32` entry (`typecast.h:29-62`) and the contract's rule is "the same 32 bits, read as dst", so no engine may touch the values | `(dtype, output_dtype) ∈ BIT_IDENTITY_PAIRS = {(uint32,int32), (int32,uint32)}` ∪ identity pairs `X→X` if they ever enter the SUPPORTED cross product | `block_pages` consecutive pages | **minimum**, no compute engine at all | 1 barrier + 1 handshake per block. Serves 14 production cases (Q2230 TILE; 12 × `uint32→int32` RM rank 2 single page; F321 RM rank 1). Reachable: same reader/writer page walk and work split; compute kernel and `cb_output_tiles` dropped; one CB reader→writer, output page size = input page size |
| R3 `rm_stream_convert` | deferred — ROW_MAJOR is outside Phase 0 SUPPORTED (contract phase plan) | `layout == ROW_MAJOR` and `(dtype, output_dtype) ∉ BIT_IDENTITY_PAIRS` | `block_pages` consecutive RM pages whose payloads are packed densely into whole 1024-element pseudo-tiles in `cb_input_tiles` | minimum against the bank boundary is **one aligned slot per page per tensor** (`aligned_page_size = round_up(page_bytes, alignment)`), not `page_bytes`; for F322 (4,194,304 pages of 6 B in / 3 B out) per-page NoC transactions make the reader/writer **issue-bound**, so the lever is transaction count (bank-contiguous slot runs: input page `i` and output page `i` share bank `i mod num_banks` and slot `i div num_banks`), not bytes | fewer reader/writer issue rounds per page; fixed compute init still once per core. Serves F317, F322 (`bf16→uint8`, perf focus p1/p2). Reachable: the compute block op is unchanged — elementwise + order-preserving unpack/pack means a dense run of RM values *is* a valid tile image; the work split is the same `split_work_to_cores` over pages; only reader/writer addressing changes |
| R4 `l1_bank_local` | deferred — no host binding exposes the L1 bank→core map (`ttnn/cpp/ttnn-nanobind/device.cpp:205-214` exposes only `MemoryView.num_banks`), and R1 already serves all six L1 production cases (P22, P31, P39, P46, P50, Q2227) at one boundary crossing per page per tensor | `buffer == "l1"` | `block_tiles` pages of the core's own bank: `page = bank_id + k · num_l1_banks` | cheapest tier: every page read and written in the core's **own** L1 bank (core-local), zero mesh traversal; crossings per tensor unchanged (1×) | as R1. Reachable: R1's page walk is `(start_page, page_stride, core_pages)` RT args with Phase 0 `page_stride = 1`; R4 sets `start_page = bank_id`, `page_stride = num_l1_banks`, and puts one core per bank — no kernel restructure |
| split dependent axis + cross-core combine | rejected — no axis is dependent; the bfp8_b exponent group lives inside one tile face row and the tile is atomic. Superseded by R1 | — | — | — | — |
| broadcast a shared operand (mcast) | rejected — single input operand, varying along the split; nothing is reuse-shared. Superseded by R1 | — | — | — | — |

**Regime-selection function** (host, `_select_regime(layout, dtype, output_dtype, buffer, W)`): Phase 0 always
returns `R1` (every Phase 0 cell satisfies R1's predicate; R1m's predicate is false for every Phase 0 INPUT).
When a second regime is built, the selector is the predicate column above, evaluated in the order
R2 → R3 → R1m → R4 → R1, and the acceptance/golden tests must pin each regime with a case whose predicate selects it.

### Traffic ranking

| Candidate split | Crossings of the bank boundary per tensor | Cross-core traffic | Occupancy | Verdict |
|-----------------|-------------------------------------------|--------------------|-----------|---------|
| A. contiguous `page` ranges over the full grid (R1) | in 1×, out 1× | none | `min(tensor_tiles, G)` cores, ±1 page balance | **chosen** |
| B. by `batch` (one image per core) | 1×, 1× | none | ≤ `batch` cores (Q2187, P-cases: 1–128; Q2229: 1) | rejected — same bytes, under-fills the grid |
| C. by `tile_col` (width split, catalog `width_split`) | 1×, 1× | none | ≤ ⌈W/32⌉ cores (W = 12 → 1 core) | rejected — same bytes, idles the grid on the dominant shape |
| D. by `tile_row` | 1×, 1× | none | ≤ `batch·⌈H/32⌉` | subsumed by A (A is its exact flattening, with finer balance) |
| E. L1 bank-local (R4), `buffer == l1` only | 1×, 1×, but core-local tier (no mesh hops) | none | one core per L1 bank | **cheapest for L1 placement**; deferred (R4 row) |
| F. DRAM-bank-affine walk, `buffer == dram` | 1×, 1× (shorter NoC paths only) | none | ≤ G | ties A on crossings; NoC-path effect is a measurement (catalog `read_coalescing`: affinity 0.91–1.28×, regresses on long cases) — not a regime |
| G. split a dependent axis + combine | n/a — no dependent axis | — | — | rejected |

Split chosen from the ranking: for DRAM placement A, E and F tie on crossings and A has the best balance; for L1
placement E is cheaper and is the `deferred` R4 row. Occupancy was not used to rank; it only orders A against B/C/D,
which tie on bytes.

### Block schedule

```cpp
// per active core: core_tiles pages starting at start_page (stride page_stride, Phase 0 = 1)
// num_blocks_this_core = ceil(core_tiles / block_tiles); last_block_extent = core_tiles - (num_blocks_this_core - 1) * block_tiles
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);      // reader
    convert_block(block_idx);   // compute
    store_block(block_idx);     // writer
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_block` | `block_tiles` pages; the last block of a core reads only `last_block_extent` pages but reserves/pushes the nominal `block_tiles` | nothing but the page cursor | per block: 1 `cb_reserve_back(block_tiles)`, `block_extent` page reads issued back-to-back, 1 read barrier, 1 `cb_push_back(block_tiles)` |
| `convert_block` | `block_tiles` tiles (`IterationShape::tiles(core_tiles).block_size(block_tiles, BlockTailSync::FullBlock)` — the helper executes only valid tiles of the last block but synchronizes the nominal `block_tiles`) | unpack/math/pack configuration and the Typecast SFPU init state (set once per core) | per core: 1 `compute_kernel_hw_startup`, 1 copy/typecast init + data-format reconfig (**one helper call spans every block of the core**); per block: 1 wait/pop on `cb_input_tiles`, 1 reserve/push on `cb_output_tiles`, 1 DEST acquire/commit/wait/release |
| `store_block` | `block_tiles` pages; the last block writes only `last_block_extent` pages but waits/pops the nominal `block_tiles` | nothing but the page cursor | per block: 1 `cb_wait_front(block_tiles)`, `block_extent` page writes, 1 writes-flushed, 1 `cb_pop_front(block_tiles)`; per core: 1 final write barrier (BB-03) |

No multicast, no semaphores, no cross-core barriers in any built regime.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (block / depth) | `block_tiles` sits at the helper's DEST cap (4 / 8). On DRAM cases at full grid the per-core bandwidth share needs few bytes in flight, but on few-core shapes or L1 cases 8–16 KiB per barrier may leave the NoC idle | (a) `in/out_buffer_depth` 3–4; (b) a data-movement block decoupled from DEST: 16–32 pages per handshake with the helper's `WaitPolicy::Upfront/PopPolicy::AtEnd` + `ReservePolicy::Upfront/PushPolicy::AtEnd` window and its internal DEST subblocking (one helper call per block); (c) whole assignment as one block, depth 1 (expected worse: serializes read/convert/write) |
| Grid synchronization / occupancy | Full participation (`min_tiles_per_core = 1`) gives the 1–16-page shapes (Q2191, Q2198, Q2227, Q2229, …) one-block pipelines on many cores. No collective exists, so only per-core startup and fill/drain are at stake | `min_tiles_per_core ∈ {block_tiles, 2·block_tiles}` (fewer cores, ≥ 1–2 full blocks each) on the small-shape cases |
| Bank stagger (catalog `bank_stagger`) | Contiguous ranges whose length is a multiple of the DRAM bank count start every core on the same bank (Blackhole: 8 banks) | rotate each core's walk start by `core_idx mod num_dram_banks` inside its own range (wrap) |
| Batched barrier / trids (catalog `trid_double_issue`) | The reader stops issuing during each block's barrier | issue block k+1 before retiring block k with `dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid` (`local_copy_helpers_dataflow.hpp:136-158`), depth ≥ 2 |
| bfp8_b pack mode | `bfp8_pack_precise = True` packs from a Float16_b source with one rounding; the non-precise path converts to a Bfp8_b pack source first (`tt_metal/jit_build/data_format.cpp:273-276, 329-333`) and may be faster | `bfp8_pack_precise = False` — measure speed **and** the 1-step group gate on all 17 bfp8_b cases (a second truncation can exceed one step) |
| DEST width for `bf16→fp32` | The LLK doc requires 32-bit DEST for a Float32 endpoint, which halves the block cap to 4; bf16 ⊂ fp32, so a 16-bit DEST + packer widening is also exact | `fp32_dest_acc_en = False` for `bf16→fp32` (`block_tiles` cap 8) — measure speed and bit-exactness on the 22 fp32 cases |

### Catalog entries that set the knobs

| Entry (`ttnn/ttnn/operations/examples/master.md`) | Knob | How |
|---------------------------------------------------|------|-----|
| `double_buffer` | `block_tiles`, buffer depths | 4–8 reads per barrier + 2× CB is the measured sweet spot; no gain once DRAM-bound at grid scale |
| `noc_placement` | core assignment | `split_work_to_cores(..., row_wise=True)`; reader on NoC0, writer on NoC1 (descriptor defaults) |
| `compute_block_size` | `convert_block` frequency | one helper call per core: init and format reconfig paid once, not per block |
| `tile_reorder` | transfer unit | whole-page transfers, batched barriers |
| `width_split` | traffic ranking | a width split needs many tile columns; W = 12 has one |
| `trid_double_issue`, `bank_stagger` | lamps | above |
| `split_reader`, `read_coalescing` | R3 (deferred) | tiny RM pages make a data-movement RISC-V issue-bound; coalescing is invalid across interleaved pages — slot-run reads per bank are the applicable form |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input pages (DRAM or L1 interleaved) | input dtype tile page (`bfloat16`: 2048 B) | reader (NCRISC, NoC0): `TensorAccessor` page reads, `block_extent` reads per barrier, walk `start_page + k·page_stride` | `TensorAccessorArgs(input)` CT args; DRAM and L1 placement identical to the kernel |
| `cb_input_tiles` | Float16_b tiles | CB, `in_buffer_depth × block_tiles` pages | |
| unpack → DEST | Float16_b → DEST (32-bit when `fp32_dest_acc_en`, else 16-bit) | `compute_kernel_lib::typecast` → `CopyTile` | `bf16→fp32`, `bf16→bfp8_b`: no SFPU work (`typecast.h:158-159, 315-316`) |
| SFPU | DEST fp32 → DEST int32 | `Typecast<Float16_b, Int32>` → `calculate_typecast_fp32_to_int32` (`ckernel_sfpu_typecast.h:197-241`): mantissa shifted by `exp − 23` (truncation toward zero), two's-complement negate for `v < 0`; saturates only for `|v| ≥ 2³¹` (outside the contract domain) | `bf16→int32` only |
| pack | DEST → output format | packer: Float32 / Int32 directly; Bfp8_b with `bfp8_pack_precise = True` (single rounding, shared exponent from the 16 values of a face row) | |
| `cb_output_tiles` | output dtype tiles (`float32` / `int32`: 4096 B, `bfloat8_b`: 1088 B) | CB, `out_buffer_depth × block_tiles` pages | |
| Output pages | output dtype tile page | writer (BRISC, NoC1): `TensorAccessor` page writes, flush per block, one write barrier at kernel end | `TensorAccessorArgs(output)` CT args |

No Tensix-to-Tensix communication in any built or deferred regime (R4 is core-local; R2/R3 are reader→writer).

**TARGET placement axis (`buffer`).** `dram` and `l1` are the same regime R1 — only the bank the
`TensorAccessor` resolves differs — both built in Phase 0. TARGET has no sharded placement; were one added, every
cut of this op is independent (knob-turn): consume the shard in place through a CB backed on the sharded buffer
(`ttnn.cb_descriptor_from_sharded_tensor`), one block = the resident shard clamped to `dest_capacity_tiles` per
handshake — never re-read it over the NoC.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block = `block_tiles` consecutive tile pages; assignment granularity = one page |
| Grid | `core_grid` host knob, default `device.compute_with_storage_grid_size()` (G = grid.x · grid.y); `sub_core_grids` must be `None` |
| `tensor_tiles` | `batch · ⌈H/32⌉ · ⌈W/32⌉` (per-image ceil; H = 1 for rank 1); asserted equal to `input.buffer_num_pages()` and `output.buffer_num_pages()` |
| Split | `ttnn.split_work_to_cores(core_grid, tensor_tiles, row_wise=True)` → `num_cores = min(tensor_tiles, G)`, `core_group_1` with `units_g1` pages, `core_group_2` with `units_g2 = units_g1 − 1` (or empty) |
| Per-core RT values | walk cores in `ttnn.grid_to_cores(num_cores, grid.x, grid.y, row_wise=True)` order, group 1 first: `start_page` = running sum of earlier cores' `core_tiles`; `core_tiles` ∈ {`units_g1`, `units_g2`}; `page_stride = 1` |
| Per-core blocks | `num_blocks_this_core = ⌈core_tiles / block_tiles⌉`; `last_block_extent = core_tiles − (num_blocks_this_core − 1)·block_tiles` |
| Remainder | ragged last block: nominal CB counts (`block_tiles`) on all three kernels; only NoC transfers and compute narrow to `last_block_extent`. Each program launch is a single pass, so the CB pointers reset at the next launch |
| Kernel groups | one kernel descriptor per role on `all_cores` (per-core counts are RT args) → ≤ 2 rectangular core ranges (full rows + one partial row) for every shape (BB-17) |

Example `tensor_tiles` (Phase 0): Q2185 2048, Q2186 1024, Q2190 512, Q2188 121, Q2191 16, Q2227 1, Q2229 4,
Q2187 968, Q2197 8, P39 8192, P31 2048, P50 4096. Single regime in Phase 0, so no regime-pinned tests are needed
yet (see the regime-selection function).

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_tiles` | 0 | `input.buffer_page_size()` (bf16 tile: 2048 B) | `in_buffer_depth × block_tiles` | spans `page` by `block_tiles` (compute holds one whole block, reader fills the next); `batch`/`tile_row`/`tile_col` folded into `page`; `elem`: one whole tile per page | input dtype (Phase 0: Float16_b) | reader | compute | whole kernel |
| `cb_output_tiles` | 1 | `output.buffer_page_size()` (fp32/int32: 4096 B, bfp8_b: 1088 B) | `out_buffer_depth × block_tiles` | spans `page` by `block_tiles` (writer drains one block while compute packs the next); folded axes as above | output dtype (Float32 / Bfp8_b / Int32) | compute | writer | whole kernel |

Full ledger, data-movement budget and audits: `l1_ledger.md` (same directory).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_tiles` pages (`last_block_extent` valid on the last block) | no — raw dataflow (no kernel_lib interleaved page-stream helper; see API Mapping) | input tensor pages `[start_page + block_idx·block_tiles·page_stride, …)` | `cb_input_tiles`, `block_tiles` pushed | `block_tiles` pages visible to compute |
| 2 | `convert_block` | `block_tiles` tiles | yes — `compute_kernel_lib::typecast`, **one call covering all `num_blocks_this_core` blocks** | `cb_input_tiles`, `block_tiles` waited + popped per block | `cb_output_tiles`, `block_tiles` reserved + pushed per block | input block freed; output block visible to writer |
| 3 | `store_block` | `block_tiles` pages (`last_block_extent` valid on the last block) | no — raw dataflow | `cb_output_tiles`, `block_tiles` waited + popped | output tensor pages, same indices as `load_block` | after the last block: write barrier, nothing in flight |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| (compute boot) | raw_api | `compute_kernel_hw_startup(icb, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:113` | `(cb_input_tiles, cb_output_tiles)`, first statement of the compute kernel | `cb_input_tiles` | `cb_output_tiles` | none |
| `convert_block` | helper | `compute_kernel_lib::typecast<Input, Output>(IterationShape, in, out)` → `unary<Typecast<in_df, out_df>, …>` → `eltwise_chain(CopyTile, Typecast, PackTile)` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:89-91`, `convenience.inl:48-57`; `Typecast` struct `eltwise/unary/misc.inl:85-90`; `IterationShape` `eltwise/api/chain.hpp:146-173`; policies `chain.hpp:233-242`; `input()` `chain.hpp:385-391`; `output()` `chain.hpp:442-450` | `Input = input(cb_input_tiles, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block)` (Block mapping is required for per-block indexing, `chain.inl:57-63`); `Output = output(cb_output_tiles, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)`; shape `IterationShape::tiles(core_tiles).block_size(block_tiles, BlockTailSync::FullBlock)`. Formats derived from the CBs (`dfb_l1_format`, `convenience.inl:50-51`) | `cb_input_tiles` | `cb_output_tiles` | `block_tiles` (block size); `core_tiles` is the per-core walk length |
| `convert_block` (SFPU, int32 only) | helper (inside `Typecast`) | `typecast_tile<Float16_b, Int32>` / `typecast_tile_init` | `tt_metal/hw/inc/api/compute/eltwise_unary/typecast.h:150-157, 462-534`; kernel `ckernel_sfpu_typecast.h:197-241` (Blackhole) | needs `fp32_dest_acc_en = True` (`typecast.h:64`) | — | — | none |
| `load_block` | raw_api | `cb_reserve_back`, `get_write_ptr`, `noc_async_read_page<TensorAccessor>`, `noc_async_read_barrier`, `cb_push_back`; `TensorAccessor` from `TensorAccessorArgs` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:406, 325, 1243, 1921, 208`; `tt_metal/hw/inc/api/tensor/tensor_accessor.h:71`, `tensor_accessor_args.h:21` | page id `start_page + (block_idx·block_tiles + j)·page_stride`, `j < block_extent`; L1 dest `get_write_ptr + j·page_size` | — | `cb_input_tiles` | `block_tiles` (reserve/push count, reads per barrier) |
| `store_block` | raw_api | `cb_wait_front`, `get_read_ptr`, `noc_async_write_page<TensorAccessor>`, `noc_async_writes_flushed`, `cb_pop_front`, final `noc_async_write_barrier` | `dataflow_api.h:479, 346, 1418, 1980, 260, 1951` | same page ids as `load_block` | `cb_output_tiles` | — | `block_tiles` (wait/pop count, writes per flush) |
| host work split | raw_api | `ttnn.split_work_to_cores(grid_size, units, row_wise)`, `ttnn.grid_to_cores` | `ttnn/cpp/ttnn-nanobind/operations/core.cpp:470-503` | `(core_grid, tensor_tiles, row_wise=True)` | — | — | core count (via `core_grid`, `min_tiles_per_core`) |
| host compute config | raw_api | `ttnn.ComputeConfigDescriptor(fp32_dest_acc_en, dst_full_sync_en=False, unpack_to_dest_mode, bfp8_pack_precise)` | `tt_metal/api/tt-metalium/program_descriptors.hpp:103-113` | per pair, Conversion Pair Table | — | — | `fp32_dest_acc_en` sets the `block_tiles` cap |
| host program cache | raw_api | `ProgramDescriptor.custom_program_hash` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:1140-1143`; consumed `ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp:78-80` | hash of the full cache key (see Host Contract) | — | — | — |

**Helpers considered and rejected (dataflow `load_block` / `store_block`).** `ttnn/cpp/ttnn/kernel_lib` has no
helper that streams interleaved tensor pages into or out of a CB: `dfb_helpers_dataflow.hpp:14-18` (tile-dimension
queries only), `local_copy_helpers_dataflow.hpp:46-158` (self-aimed L1→L1 reads and trid controls — used only by the
trid lamp), `tile_column_dataflow.hpp:36-92` (one tile column fill/copy), `l1_helpers.hpp:25-101` (L1 pointers,
zero/fill). The block operations are realized with the raw page APIs above; the gap a helper would close is
"`read_pages_to_dfb(accessor, dfb, first_page, count, stride)` / `write_pages_from_dfb(...)` with one barrier per call".

**Compute: no raw-API fallback.** The whole conversion is the `typecast` convenience helper. For a `UInt8` CB
(refinement `bf16→uint8`) the convenience form `static_assert`s (`convenience.inl:52-55`); use
`unary<Typecast<Float16_b, UInt8>, Input, Output>` (`convenience.hpp:86-87`) — same helper family, same knobs.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| bfp8_b beyond one step of the host rendering | Without precise packing the pack source is Bfp8_b (`data_format.cpp:273-276, 329-333`): a value can be truncated at its own exponent and again when aligned to the group exponent, up to ~1.5 steps from the exact value and ~2 steps from a rounded host value | `bfp8_pack_precise = True` for every bfp8_b destination (single rounding from a Float16_b pack source); `fp32_dest_acc_en = False` for `bf16→bfp8_b` |
| int32 written into 16-bit DEST | `calculate_typecast_fp32_to_int32` stores INT32 (`ckernel_sfpu_typecast.h:239`); the LLK requires 32-bit DEST (`typecast.h:64`) | `fp32_dest_acc_en = True` for every pair with a 32-bit endpoint; `block_tiles` cap re-derived (4) |
| Chain silently shrinks its block | Release builds clamp `block_size` to `chain_max_block_v` (`chain.inl:3345-3349`) — compute then syncs fewer pages than reader/writer push/pop | host clamp `block_tiles ≤ dest_capacity_tiles`; `static_assert` in the compute kernel |
| Ragged last block | `core_tiles` is rarely a multiple of `block_tiles` | nominal counts everywhere: reader pushes `block_tiles` after reading `last_block_extent` pages; helper uses `BlockTailSync::FullBlock` (syncs `block_tiles`, computes only valid tiles); writer waits/pops `block_tiles`, writes `last_block_extent` |
| Output page `p` ≠ input page `p` | Only holds because input and output share the logical shape and TILE layout (same per-image padding) | allocate output with the input's shape/layout/memory config; assert equal `buffer_num_pages()` on host |
| Tile count from `floor` | Non-aligned H (3844, 1000, 77, 31, 33, …) and W = 12 are the dominant production shapes | per-image `⌈H/32⌉·⌈W/32⌉` only |
| In-flight writes at kernel exit | Another program starts right after the op (BB-03) | writer ends with `noc_async_write_barrier`; reader barriers every block before pushing |
| Incomplete program-cache key | Same shape with a different `dtype` shares every kernel source and could hash alike (BB-07) | `custom_program_hash` from the full key, including output dtype, shape and grid; the acceptance test runs one shape with two destinations |
| 32-bit sources lose bits in srcA (refinement) | srcA holds TF32: `fp32→bf16` RNE, `int32→bf16`, `uint32→*` read wrong bits if unpacked through srcA; small-integer / short-mantissa tests still pass | `unpack_to_dest_mode[cb_input_tiles] = UnpackToDestFp32` for every 32-bit source (Conversion Pair Table); test with full-range values |
| UInt32↔Int32 through DEST (refinement) | no LLK entry; packer integer handling could alter bits | regime R2 (dataflow-only bit copy) |
| L1 placement footprint (BB-05) | the op's L1 per core = CBs + the output's share of the L1 banks | CBs ≤ 50 KiB (Phase 0); P39 output share ≈ `8192·1088 B / num_l1_banks`; total ≪ 1,347,776 B; shape-independent CB footprint, so no shape can exceed it |

## Conversion Pair Table (TARGET)

`validate()` accepts the cross product `SUPPORTED["dtype"] × SUPPORTED["output_dtype"]` and EXCLUSIONS must stay
empty, so **every pair in that cross product must be implemented when its input dtype enters SUPPORTED** — not
only the production pairs. Phase 0's cross product is exactly its three production pairs. LLK references:
`typecast.h` (Blackhole branch, lines 122-456) and `ckernel_sfpu_typecast.h` (Blackhole).

| Source → destination | Regime | Mechanism (LLK) | `fp32_dest_acc_en` | `unpack_to_dest_mode[cb_input_tiles]` | `bfp8_pack_precise` | Production cases |
|----------------------|--------|-----------------|--------------------|---------------------------------------|---------------------|------------------|
| bf16 → fp32 | R1 | unpack/pack only (`typecast.h:158-159`) | True | Default | — | 22 (Phase 0) |
| bf16 → bfp8_b | R1 (R1m if `W % 16 ≠ 0`) | pack only (`typecast.h:315-316`) | False | Default | True | 17 (Phase 0) |
| bf16 → int32 | R1 | SFPU `fp32_to_int32`, trunc (`typecast.h:150-157`) | True | Default | — | Q2229 (Phase 0) |
| bf16 → uint8 | R3 (RM) / R1 | SFPU `fp32_to_uint8`: floor of `|v|`, low byte (`typecast.h:381-391`; sfpu `:952-991`); explicit `unary<Typecast<Float16_b, UInt8>>` | True | Default | — | F317, F322 |
| bf16 → uint32 | R1 / R3 | SFPU `fp32_to_uint32` (`typecast.h:232-239`) | True | Default | — | — |
| bf16 → bf16 | R2 | identity bit copy | — | — | — | — |
| fp32 → bf16 | R1 | SFPU `fp32_to_fp16b`, RNE (`typecast.h:160-167`; sfpu `:272-288`) | True | UnpackToDestFp32 | — | F315, F316, F318, F319, F320, F323, F324 |
| fp32 → fp32 | R2 | identity bit copy | — | — | — | — |
| fp32 → bfp8_b | R1 | pack only (`typecast.h:319-320`) | True | UnpackToDestFp32 | True | — |
| fp32 → int32 / uint32 / uint8 | R1 / R3 | SFPU `fp32_to_int32` / `fp32_to_uint32` / `fp32_to_uint8` (`typecast.h:184-191, 248-255, 381-391`) | True | UnpackToDestFp32 | — | — |
| int32 → bf16 | R1 | SFPU `int32_to_fp16b`: int→fp32 cast (rne32) then RNE to bf16 (rne16) (`typecast.h:142-149`; sfpu `:138-195`) | True | UnpackToDestFp32 | — | Q2228 |
| int32 → fp32 / bfp8_b | R1 | SFPU `int32_to_fp32` / `int32_to_fp16b` + bfp8 pack (`typecast.h:192-199, 224-231`) | True | UnpackToDestFp32 | True for bfp8_b | — |
| int32 → uint32 | R2 | bit identity | — | — | — | F321 (RM) |
| int32 → int32 | R2 | identity | — | — | — | — |
| int32 → uint8 | R1 / R3 | SFPU `uint_to_uint8`: low byte (`typecast.h:392-401`) | True | UnpackToDestFp32 | — | — |
| uint32 → int32 | R2 | bit identity | — | — | — | Q2230 (TILE) + 12 RM |
| uint32 → fp32 / bf16 / bfp8_b / uint8 | R1 / R3 | SFPU `uint32_to_fp32` / `uint32_to_fp16b` / `uint_to_uint8` (`typecast.h:240-247, 256-263, 272-279, 392-401`) | True | UnpackToDestFp32 | True for bfp8_b | — |
| uint32 → uint32 | R2 | identity | — | — | — | — |
| uint16 → int32 / uint32 | R1 | SFPU `uint16_to_uint32`, zero-extend (`typecast.h:280-296`; sfpu `:501-541`) | True | verify on device (UInt16 source must reach DEST unchanged) | — | Q2231 |
| uint16 → fp32 / bf16 / bfp8_b / uint8 | R1 / R3 | SFPU `uint16_to_fp32` / `uint16_to_fp16b` / `uint_to_uint8` (`typecast.h:134-141, 176-183, 208-215, 392-401`) | True | as above | True for bfp8_b | — |

(`bfloat8_b` destination with ROW_MAJOR is INVALID in `feature_spec.py` and never reaches the op.)

## Registry Model and Host Contract

| Item | Specification |
|------|---------------|
| `INPUT_TAGGERS` (this order) | `tag_dtype` → `scenario["dtype"]`; `tag_output_dtype` → `scenario["output_dtype"]`; `tag_layout` → `scenario["layout"]`; `tag_buffer` → `scenario["buffer"]`; `tag_alignment` → W = `input_shape[-1]`, H = `input_shape[-2]` (1 for rank 1): `"w_non_aligned"` if `W % 32`, elif `H % 32` `"h_non_aligned"`, else `"tile_aligned"`; `tag_rank` → `len(input_shape)`. Each takes `(inputs, axes)` with `inputs = (scenario,)` |
| `SUPPORTED` (Phase 0) | `dtype: [bfloat16]`, `output_dtype: [float32, bfloat8_b, int32]`, `layout: [TILE_LAYOUT]`, `buffer: ["dram", "l1"]`, `alignment: ["tile_aligned", "w_non_aligned", "h_non_aligned"]`, `rank: [3, 4]` |
| `EXCLUSIONS` | `[]` — every phase (no size / width / page-size axis, ever) |
| `validate(input_tensor, dtype, *, sub_core_grids=None)` order | Each `ValueError` message names the offending argument — the acceptance test matches `dtype`, `sub_core_grids`, `rank` (case-insensitive for `rank`). 1. `isinstance(dtype, ttnn.DataType)` else `ValueError`; 2. `sub_core_grids is None` else `ValueError`; 3. `1 ≤ rank ≤ 4` else `ValueError`; 4. interleaved memory layout else `ValueError`; 5. build `scenario = {"input_shape": tuple(shape), "dtype": input.dtype, "output_dtype": dtype, "layout": input.layout, "buffer": "l1" if buffer_type == L1 else "dram"}` and apply the taggers in order; 6. SUPPORTED per axis → `UnsupportedAxisValue`; 7. EXCLUSIONS → `ExcludedCell` |
| Entry point | `g6_typecast(input_tensor, dtype, *, sub_core_grids=None)`: first line `validate(...)`; allocate the output (input's shape, layout, memory config; `dtype`); fetch or build the descriptor; one `ttnn.generic_op([input_tensor, output_tensor], descriptor)`. No `to_layout` / `tilize` / `pad` / `slice` / reshape / existing typecast or copy op, ever |
| Program-descriptor cache (BB-07, BB-16) | module-level dict keyed by `(tuple(input.shape), input.dtype, dtype, input.layout, buffer_type, memory_layout, core_grid.x, core_grid.y)`; a hit reuses the stored `ProgramDescriptor` and rewrites only the input/output buffer-address RT args; `custom_program_hash` = hash of the key, so a different key can never replay another program and the device side skips the descriptor walk. No kernel build and no device allocation besides the output tensor happen after the first call for a key |
| Plan report (BB-08, BB-21) | `describe_plan(input_tensor, dtype) -> dict`: regime, `core_grid`, `num_cores`, core ranges, `block_tiles`, buffer depths, `fp32_dest_acc_en`, `bfp8_pack_precise`, per-core `(start_page, core_tiles, page_stride)`, and the role per core (`reader`, `compute`, `writer` on every active core; none elsewhere). Computed from the same builder the entry point uses |
| Profiler zones (BB-21) | zone build only (no change to the plain build): reader — reserve wait, read issue + barrier; compute — the helper call; writer — data wait, write issue + flush, final barrier. Each kernel's work and blocking waits fall inside a named zone |
| Determinism (BB-06) | no semaphores, no cross-core state; every hand-over is a CB credit issued after the data landed (read barrier before push, pack before push) |

Structural impossibilities beyond `feature_spec.py` INVALID: none found.
