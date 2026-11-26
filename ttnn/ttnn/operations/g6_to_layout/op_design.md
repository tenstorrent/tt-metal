# Operation Design: g6_to_layout

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (layout conversion; the only value transform is the bf16 <-> bf8b storage conversion of later refinements) |
| Goal | Move every logical value from TILE order to ROW_MAJOR order (Phase 0), and later ROW_MAJOR -> TILE, in ONE device program: each live input byte crosses its memory tier once, each output byte once; a bfloat16 output is bit-identical to its input. |
| Math | `y[n, h, w] = x[n, h, w]` for `n < N, h < H, w < W`; `shape(y) == shape(x)`; `N = prod(shape[:-2])` (1 for rank 2) |
| Storage orders | TILE: tile `(n, h//32, w//32)`, face `2*((h%32)//16) + (w%32)//16`, face row `h%16`, column `w%16` (face = 16 rows x 32 B, face row = 32 B for bf16). ROW_MAJOR: page `n*H + h` holds the `W` values of that row (`W*2` B, stored at an aligned stride of `align(W*2, 64)` in DRAM, `align(W*2, 16)` in L1 on Blackhole). |
| References | `eval/golden_tests/g6_to_layout/{feature_spec.py,helpers.py}`; `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp` + `.inl`; `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`; `tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_defs.h:20-41`; `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_common.h:54-59`; `.claude/references/{blocking-model.md,l1-footprint-discipline.md,noc-alignment-in-tt-metal.md}`; perf catalog `ttnn/ttnn/operations/examples/master.md` (entries cited below); `l1_ledger.md` (this directory) |

### The decision everything else hangs on: the reshuffle must not interpret the bits

Nothing is computed, but tile order -> row order is a 32-B-granular reshuffle that has to happen
in L1 (a NoC transfer cannot stride). Three actors can do it: the Tensix compute engine
(unpack -> MOVA2D -> pack_untilize), a data-movement RISC issuing 32-B local NoC copies, or
DRAM-side 32-B gathers. The compute engine is the cheapest by far (one LLK call per tile-row
segment, the data-movement RISCs stay free for DRAM traffic), but its float path is lossy:

| Fact | Source |
|------|--------|
| While `ALU_ACC_CTRL_Zero_Flag_disabled_src` is 0, MOVA2D flushes every SrcA datum whose low 8 bits (the exponent of the float Src layout) are 0 -> bf16 subnormals AND `-0.0` become `+0`. | `ckernel_defs.h:20-25` |
| The flag follows the operand formats at `hw_configure` / every `reconfig_data_format_srca`: it is 1 (preserve) only when SrcA or SrcB is `UInt16`. | `ckernel_defs.h:38-41`, `llk_math_common.h:54-59`, `cmath_common.h:96-106` |
| The datacopy used by `pack_untilize` does not override it on the 16-bit path: its only flag writes sit in the 32-bit unpack-to-dest broadcast branch and the fp32-DEST UInt16 broadcast branch. | `llk_math_eltwise_unary_datacopy.h:100-101`, `:296-301` |
| `UInt16` "shares the Src bit layout"; with the flag disabled, "16-bit integer data survives the move into Dst". | `ckernel_defs.h:26-31` |

**Decision:** the bf16 -> bf16 path declares both of its circular buffers `UInt16`. The tile bytes
are unchanged (2048 B per 32x32 tile); the compute engine untilizes them as 16-bit integers, so
no float conversion, flush or NaN canonicalization can touch them. The reader and writer move raw
bytes and never look at the format. A `Float16_b` declaration of the same CBs is a rejected regime
(R8). If silicon disproves the UInt16 transparency (the special-pattern tests are the gate), the
fallback is the deferred regime R5 with the identical CB inventory.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes (positional) | rank 2-4; Phase 0: bf16, TILE, interleaved DRAM or L1 | — | buffer address: RT (reader); geometry: host plan |
| `layout` | `ttnn.Layout` | yes (positional) | `ROW_MAJOR_LAYOUT` or `TILE_LAYOUT`, never the input's layout | — | host (regime selection) |
| `dtype` | `ttnn.DataType` | keyword-only | bf16 (Phase 0); bf8b only with `layout == TILE_LAYOUT` (refinement) | `None`: input dtype, except bf8b input -> ROW_MAJOR gives bf16 | host (output allocation, CB formats) |
| `memory_config` | `ttnn.MemoryConfig` | keyword-only | interleaved, DRAM or L1 | `None`: the input's placement | host (output allocation) -> writer accessor CT args |

No `compute_kernel_config`: no arithmetic, and the one format conversion (refinements) has no precision knob.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d0, ..., H, W)`, rank 2, 3 or 4; any `H >= 1`, `W >= 1` (non-multiples of 32 included); leading dims and `W` unbounded |
| Dtype | Phase 0: `bfloat16`. TARGET adds `bfloat8_b` (TILE inputs only) |
| Layout | Phase 0: `TILE_LAYOUT`. TARGET adds `ROW_MAJOR_LAYOUT` |
| Memory | Phase 0: `INTERLEAVED`, DRAM or L1. TARGET adds `HEIGHT_SHARDED`, `BLOCK_SHARDED` (L1) |
| Tile padding | content arbitrary (the producer leaves anything there); never read as values |

### Output

| Property | Value |
|----------|-------|
| Shape | `input_tensor.shape` |
| Dtype | `dtype` if given; else bf16 when the input is bf8b and `layout == ROW_MAJOR_LAYOUT`; else the input dtype |
| Layout | `layout` |
| Memory | `INTERLEAVED`; buffer type of `memory_config` if given, else the input's buffer type |
| Padding (TILE output, refinement) | free for bf16; for bf8b, padding columns inside a 16-value exponent group that holds logical values must be 0 |

## Host Contract (registry model)

| Item | Specification |
|------|---------------|
| Scenario dict | `validate()` builds `{"shape": tuple(input.shape), "dtype": input.dtype, "layout": input.layout, "memory_layout": mc.memory_layout, "buffer": mc.buffer_type, "shard_spec": <grid ranges / shard shape / orientation or None>, "target_layout": layout, "out_dtype": dtype, "memory_config": memory_config}` with `mc = input.memory_config()` — the same dict `eval/golden_tests/g6_to_layout/axes.py:classify_call` builds — and applies `INPUT_TAGGERS` to `(scenario,)` |
| `INPUT_TAGGERS` (order) | `dtype` -> `scenario["dtype"]`; `layout` -> `scenario["layout"]`; `output_dtype` -> `out_dtype` if not None, else bf16 if (`dtype == bf8b` and `target_layout == ROW_MAJOR`), else `dtype`; `memory_layout` -> `scenario["memory_layout"]`; `alignment` -> `"w_non_aligned"` if `W % 32`, elif `H % 32` -> `"h_non_aligned"`, else `"tile_aligned"` (`H = shape[-2]`, `W = shape[-1]`); `rank` -> `len(shape)` |
| `SUPPORTED` (Phase 0) | `dtype: [bf16]`, `layout: [TILE]`, `output_dtype: [bf16]`, `memory_layout: [INTERLEAVED]`, `alignment: ["tile_aligned", "w_non_aligned", "h_non_aligned"]`, `rank: [2, 3, 4]` |
| `EXCLUSIONS` | `[]` — and it stays free of production cells in every refinement |
| `validate()` order | (1) `ValueError`: `layout` not in {RM, TILE} or equal to `input.layout`; rank not in 2..4; `dtype == bf8b` with `layout == ROW_MAJOR`. (2) per-axis `SUPPORTED` -> `UnsupportedAxisValue`. (3) `EXCLUSIONS` -> `ExcludedCell`. (4) `ValueError`: `memory_config` sharded; `memory_config is None` while the input is sharded. Step (4) follows step (2) as the spec requires. Every `ValueError` / refusal message contains `g6_to_layout` (the acceptance test matches on it). |
| Entry point | `validate(...)` is the first line; then `select_plan(...)` (memoized), output allocation (`ttnn.allocate_tensor_on_device`, positional args), the memoized program descriptor with buffer addresses patched into the runtime args, and exactly one `ttnn.generic_op([input, output], descriptor)` |
| Forbidden on host | `ttnn.to_layout`, `tilize`, `untilize`, `pad`, `slice`, `typecast`, any second device op, any host read of values |
| Plan report (BB-08) | `select_plan(input_tensor, layout, *, dtype=None, memory_config=None) -> Plan` is exported from the op module; `Plan` carries `regime`, `grid`, `cores_used`, `num_col_chunks`, `segment_width`, `block_segments`, `tiles_depth`, `rows_depth`, the two core groups with their segment counts, and `core_roles` (core -> reader/compute/writer or idle) for the zone build (BB-21). It is a pure function of shape, dtypes, layouts, placements and grid — never of measured free L1 |
| Cache key (BB-07) | `(shape, input dtype, input layout, input memory_config incl. buffer type and shard spec, output dtype, output layout, output memory_config, grid size)`; a hit reuses the plan and the descriptor and patches only the two buffer addresses |
| Structural impossibilities | none beyond `feature_spec.py:INVALID` |

## Blocking Model

The op has no dependent axis and no reuse-shared operand: every output byte is one input byte.
So the work split costs no combine and no broadcast, every candidate split moves the same bytes,
and the remaining questions are (a) the transfer shape each split produces, (b) occupancy, and
(c) the block extent — which only sets fixed per-block costs and L1.

### Axes

Work unit: a **segment** = one tile-row `r` x `segment_width` consecutive tile-columns starting at
column chunk `j`. Segment index `u = r * num_col_chunks + j`. A **block** = `block_segments`
consecutive segments of one core. Every block operation acts on segments of the identical shape
`1 x segment_width` tiles, so the block is a `block_segments x segment_width` rectangle in
(segment, tile-in-segment) index space; when `num_col_chunks == 1` it is literally
`block_segments` tile-rows x `tensor_w_tiles` tiles.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `n` — folded leading dims (`tensor_n = prod(shape[:-2])`) | independent — no value crosses images | folded with `ht` into the tile-row index `r = n * tensor_h_tiles + ht`; extent along it = `block_segments` | `block_segments` (row below) | host plan | spread with `ht` (contiguous segment ranges) | knob-turn |
| `ht` — tile-rows of one image (`tensor_h_tiles = ceil(H/32)`, per image) | independent — rows are copied, never combined | `block_segments` (segments per block) | coarsest that fits: `min(segments_per_core_max, floor(B_cb / (rows_depth * segment_width * (tile_bytes_in + tile_bytes_rows))))` (depths equal, see ledger) | `Plan.block_segments` (host), passed to reader + writer | contiguous ranges of segment indices over the first `cores_used` cores in row-major order (`ttnn.split_work_to_cores(grid, num_segments, row_wise=True)`) — the full grid from Phase 0 | knob-turn |
| `wt` — tile-columns (`tensor_w_tiles = ceil(W/32)`) | independent — chunks of one row land at disjoint byte offsets of the same output pages; no combine | `segment_width` (tiles per segment) | `ceil(tensor_w_tiles / c)` with `c` = the smallest column-chunk count whose per-core load is within `balance_slack` of ideal (Work Distribution), clamped to `segment_width_max` | `Plan.segment_width` (host) -> compute CT arg (untilize template param) and reader/writer args | chunks spread together with `ht` through the segment index | knob-turn |
| `i` — row within a tile (32) | independent | not split: the tile is the input transfer unit; extent fixed by the TILE format (`tile_h = 32`). Live rows `rows_live = min(32, H - ht*32)` are a per-segment runtime extent, not a knob | 32 (format) | TILE format constant | none (inside one tile) | none — format-fixed |
| `c` — column within a tile (32 = two faces) | independent | not split; `tile_w = 32` fixed by the format. Live columns `cols_live = min(32, W - wt*32)` are a per-tile runtime extent | 32 (format) | TILE format constant | none (inside one tile) | none — format-fixed |

The intermediate the scheme produces (untilized rows in `cb_rows`) has the same `(segment, row,
column)` axes, the same characters and the same assignment: the core that untilizes a segment
writes it. No core collects, coordinates or hands out work; no stage computes on one core while
others wait.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_tiles` | `tiles_depth` | 2 | the reader issues block k+1's DRAM reads while compute untilizes block k |
| `cb_rows` | `rows_depth` | 2 | compute untilizes block k+1 while the writer drains block k |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| MOVA2D FlushDenormals (`ckernel_defs.h:20-41`, `llk_math_common.h:54-59`) | not an extent: a datapath cap on the CB data format | both CBs of the bf16 -> bf16 path are `UInt16` | subnormals and `-0.0` silently become `+0` |
| Untilize helper: DEST capacity (`dest_helpers.hpp:89-99`: 8 tiles, half-sync 16-bit) | `segment_width` above 8 runs the block-based path, sub-blocks = largest divisor of `segment_width` <= 8 (`untilize_helpers.inl:69-86`) | none needed for correctness (any width works); see perf lamp L5 | a prime `segment_width > 8` runs 1-tile sub-blocks: correct, slower |
| Untilize helper: fast path gate (`untilize_helpers.inl:23-53`) | `UInt16` is not on the BH fast-untilize path | accepted: standard `pack_untilize` | — |
| Untilize helper: output CB (`untilize_helpers.inl:164`, `:210`) | output must not be block-float; output CB >= `segment_width` pages | `cb_rows` is `UInt16` with `rows_depth * block_segments * segment_width` pages | assert / hang |
| `DataflowBuffer` quanta are `uint16_t` (`dataflow_buffer.h:243-246`) | `block_segments * segment_width <= 65535` | implied by L1: at most 658 pages of 2048 B fit in 1347776 B | counter wraps |
| CB ring wrap (`ttnn-cb-memory-fundamentals.md:85`) | per-push page count | non-terminal pushes / pops are the nominal `block_segments * segment_width`; only a core's terminal block is shorter; compute's per-segment quanta (`segment_width`, or its sub-block) divide the capacity | pointer walks past `fifo_limit`, silent L1 corruption |
| BH DRAM read alignment: local and NoC address equal mod 64 (`noc-alignment-in-tt-metal.md:43`, `core_config.h:42`) | live-region read ranges | every read range starts at a face boundary (offset 0/512/1024/1536 of a 2048-B tile page) and lands at the same offset of a 2048-B slot of a CB whose base is 64-B aligned | silent corruption (watcher stops it) |
| NoC write alignment: equal mod 16 for DRAM and L1 writes (same reference) | row-write offsets | source row stride `segment_width * 64` and destination offset `j * segment_width * 64` are multiples of 64; sizes are arbitrary (6 B rows legal) | silent corruption |
| L1 budget, 1347776 B per core, counted with the L1 output's own pages (BB-05) | `block_segments`, `segment_width` | `segment_width <= segment_width_max = floor(B_cb / (tiles_depth*tile_bytes_in + rows_depth*tile_bytes_rows))`; `block_segments` per the Axes row; host refuses (RuntimeError) a non-golden shape whose `B_cb` is below one segment of width 1 | launch failure beside the model's tensors |

### Regimes

Minimum, named boundary: each live input byte crosses DRAM (or, for an L1 input, the NoC) once
at the 32-B face-row granule, each output byte crosses once.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 `untilize_interleaved`** | **built** (Phase 0; all 56 production cases of the class, every alignment and rank) | `layout == TILE` and `memory_layout == INTERLEAVED` and `dtype == output_dtype == bf16` | `block_segments x segment_width` tiles | minimum: a tile's live face rows are read once (dead faces and tile-padding rows never read: `H = 1` reads 64 B of 2048, `W <= 16` reads faces 0 and 2 only), each output row written once (`W*2` B, one write per row per column chunk). Above the logical-byte minimum only by the 32-B granule (W = 3 uses 6 B of each 32-B face row) — structurally unreachable: no DRAM access is narrower | per block: one DRAM read barrier + one `cb_tiles` reserve/push (reader), one writes-flushed + one `cb_rows` pop (writer). Untilize init/uninit and `compute_kernel_hw_startup` are once per core and independent of the block. No regime-added terms (no combine, no shared operand) |
| R2 `untilize_interleaved_bf8b` | deferred — dtype axis (bf8b is not in Phase 0 `SUPPORTED["dtype"]`); production cases 29, 43 (L1 input) | `layout == TILE` and `INTERLEAVED` and `dtype == bf8b` (output bf16 by the Output rule) | as R1 | minimum at whole-tile granule (1088 B; the shared-exponent section precedes the mantissa faces, so live-face reads do not apply; production bf8b shapes are tile-aligned) | as R1. Reachable: same three kernels; `cb_tiles` page format `Bfp8_b` (1088 B), `cb_rows` `Float16_b`, untilize does a real conversion (BH fast-untilize eligible); the reader's read rule switches to whole tiles |
| R3 `untilize_sharded` | deferred — memory_layout axis; production cases 20 (BLOCK), 37, 48 (HEIGHT), all bf8b TILE in L1 -> bf16 RM L1 interleaved | `layout == TILE` and `memory_layout in {HEIGHT_SHARDED, BLOCK_SHARDED}` | the whole live shard: `shard_live_rows/32` segments x `shard_cols/32` tiles (default one block; split only under L1 pressure) | minimum: the input never crosses the NoC — `cb_tiles` is backed on the shard (`ttnn.cb_descriptor_from_sharded_tensor`), only the live tile-rows of the last shard are pushed / read; output rows once | one block = one handshake per shard. Reachable: compute and writer unchanged; the reader only pushes the live shard pages; core assignment = the caller's shard grid; row ownership: shard `s` (HEIGHT) owns 2-D rows `[s*shard_rows, ...)`, BLOCK shard `(y, x)` owns tile-column `x` of its rows (writes at byte offset `x*shard_cols*2` of shared pages — disjoint, no combine) |
| R4 `tilize_interleaved` | deferred — layout axis (ROW_MAJOR input: 19 production cases incl. case 16 / Q3 with bf8b output and the L1 Panoptic cases) | `layout == ROW_MAJOR` and `INTERLEAVED` | `block_segments x segment_width`, same split rule | minimum: each live input row read once per column chunk (`seg_bytes` from page `n*H + h`), each output tile written once (2048 B bf16 / 1088 B bf8b, padding included as tile storage) | as R1 with roles mirrored. Reachable: reader fills a row-major staging CB at row stride `segment_width*64`; compute `tilize` helper (`UInt16` both sides for bf16 -> bf16; `Float16_b` -> `Bfp8_b` packer conversion for a bf8b output, after the reader zeroes columns `[W, round_up(W, 16))` of every live row so padding never joins a live exponent group); writer writes whole tiles |
| R5 `untilize_dataflow_reshuffle` | deferred — fallback for R1's `untilize_block` if silicon disproves UInt16 transparency. Positive reason: R1 serves every shape this regime serves and spends no data-movement issue slots on the reshuffle, where R5 spends one 32-B local NoC read per live face row (64 per full tile) on a data-movement RISC | same as R1 | as R1 | as R1 (local L1 -> L1 traffic added, no extra DRAM bytes) | as R1. Reachable: identical CB inventory and layouts (`cb_tiles` -> `cb_rows`); the compute kernel is replaced by self-aimed 32-B reads (`local_copy_helpers_dataflow.hpp:75,95` `set_read_state<32>` / `read_with_state`) on the writer RISC |
| R6 `untilize_narrow_direct` | deferred — R1 serves these shapes (W <= 16: p1, p4). Positive reason: there the writer issues 32 row writes per tile while compute issues one `pack_untilize` per tile, so compute is not on the critical path; R6 would remove `cb_rows` (L1) and nothing on the critical path | `W <= 16` and R1's predicate | as R1 | as R1 | as R1. Reachable: the writer reads row `i` from face row `i%16` of face `2*(i//16)` in `cb_tiles` instead of from `cb_rows`; reader unchanged |
| R7 `untilize_bank_coalesced_writes` | deferred — R1 serves these shapes. Positive reason: writing a bank's run of rows as one NoC write writes the whole aligned page slot, 64 B per row in DRAM where the row is `W*2` B (W = 3: 64 B vs one masked 32-B burst) — more DRAM write traffic on shapes that are already DRAM-bound | narrow rows (`align(W*2, 64) > W*2`) and R1's predicate | as R1 | above minimum by `align(W*2,64)/(W*2)` on writes | trades `rows` write transactions for `num_dram_banks` per block. Reachable: `store_block` only — rows gathered into bank-major slots (local NoC reads) before one write per bank per block |
| R8 `untilize_float_compute` | rejected — lossy: with `Float16_b` CBs MOVA2D flushes zero-exponent datums (subnormals, `-0.0`) (`ckernel_defs.h:20-25`, `llk_math_common.h:54-59`). Superseded by R1, the same datapath with `UInt16` CB formats | — | — | — | — |
| R9 `untilize_dram_face_gather` | rejected — dead end: reads every 32-B face row from DRAM straight into row order, 64 DRAM transactions of 32 B per full tile instead of one 2-KB read. Superseded by R1 on every shape | — | — | 64x the DRAM transaction count | — |
| R10 `staged_two_pass` | rejected — the contract forbids an intermediate device tensor, a second DRAM pass or host staging | — | — | 2x | — |
| R11 `dependent_axis_split_combine` | rejected — no dependent axis exists (each output byte depends on one input byte), so there is nothing to combine | — | — | — | — |

**Selection function** (host, exact; once every row is built):

```
if layout == TILE_LAYOUT:
    if memory_layout in (HEIGHT_SHARDED, BLOCK_SHARDED): R3
    elif dtype == bfloat8_b:                              R2
    else:                                                 R1
else:                                                     R4
```

No predicate depends on the grid; the column-chunk count inside R1 does. Regime-pinned tests
therefore cover R1 with `num_col_chunks == 1` and with `num_col_chunks > 1` on any grid:
`(1, 1024, 1024, 3)` has 32768 tile-rows (always 1 chunk), `(16, 1, 2560)` and `(1, 131072)`
have 16 and 1 tile-rows (always > 1 chunk on a grid of more than 16 cores).

### Traffic ranking

| Candidate split | DRAM crossings (input / output) | Cross-core | Write transactions | Rank |
|-----------------|--------------------------------|------------|--------------------|------|
| Tile-rows only (`num_col_chunks = 1`) | 1 / 1 | none | 1 per output row (`W*2` B) | 1 — cheapest; chosen whenever it balances |
| Tile-rows x column chunks (`num_col_chunks = c > 1`) | 1 / 1 | none | `c` per output row (`segment_width*64` B each, last one shorter) | 2 — same bytes, `c - 1` extra writes per row; chosen only when the rows alone cannot fill the grid within `balance_slack` |
| Tile-columns only | 1 / 1 | none | `G` per output row | 3 — maximal fragmentation; reached by rank 2 only at `R = 1` |
| Dependent axis + combine | n/a | n/a | n/a | does not exist |

Bytes per tier are identical for every split (no shared operand, no halo), so occupancy and
traffic do not conflict; the ranking is by write fragmentation, and the chosen split is the
least-fragmented one that balances.

**Operand-reuse check.** One input; it varies along every split axis; no operand is re-read by
any core. **Stall-shadow check.** The only waits are producer -> consumer hand-offs (compute waits
on the reader's DRAM barrier, the writer on compute); nothing is independent of the data being
waited for except the neighbouring block, which `tiles_depth = rows_depth = 2` already schedules
into the stall. No algebraic reorder applies (a copy has no floating-point order).

**Catalog evidence used for the knobs.**

| Catalog entry | Knob it sets |
|---------------|--------------|
| `noc_placement` | `split_work_to_cores(..., row_wise=True)`; reader on NoC0 (RISCV_1), writer on NoC1 (RISCV_0) |
| `width_split` | column chunks when tile-rows cannot fill the grid (`(1, 131072)`, `(32, 5120)`, `(16, 1, 2560)`) |
| `double_buffer`, `trid_double_issue` | a block of reads then ONE barrier; depth 2; at a large CB the one-barrier-per-block loop needs no trids |
| `tile_reorder` | whole-page coalesced reads (one 2-KB read per fully live tile), never sub-tile pieces when the tile is live |
| `compute_block_size` | one untilize helper call per core (init once), `NoReconfigure` (the format never changes) |
| `bank_stagger`, `split_reader` | perf lamps L4, L3 |

### Block schedule

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);      // reader
    untilize_block(block_idx);  // compute
    store_block(block_idx);     // writer
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_block` | `block_segments x segment_width` tiles; per tile only the live-region ranges (rule below) | the block in `cb_tiles` until compute pops it; the previous block may still be in `cb_tiles` (depth 2) | one `reserve_back`, one read barrier, one `push_back` per block; accessor setup once per kernel |
| `untilize_block` | `block_segments` segments of `1 x segment_width` tiles each | nothing beyond the segment in DEST | `compute_kernel_hw_startup` once per core; untilize init/uninit once per core (a single helper call over all segments of the core, `num_blocks = segments_this_core`); per-segment wait/reserve/pop/push are the helper's own |
| `store_block` | `block_segments` segments x `rows_live` rows x `seg_bytes` | the block in `cb_rows` until all its writes have left L1 | one writes-flushed + one `pop_front` per block (waits are cumulative per segment so writing starts as soon as a segment is packed); one `noc_async_write_barrier` at kernel end (BB-03) |

**Live-region read rule** (`load_block`, per tile; bf16 face = 512 B, face row = 32 B):
`top = min(rows_live, 16)`, `bot = rows_live - top`, `right = cols_live > 16`. Ranges in the tile
page: `[0, top*32)`; if `right`: `[512, 512 + top*32)`; if `bot > 0`: `[1024, 1024 + bot*32)`; if
`right and bot > 0`: `[1536, 1536 + bot*32)`. Adjacent ranges merge (a fully live tile is one
`[0, 2048)` read). Each range lands at the same offset of the tile's slot. Bytes outside the
ranges in the slot are stale L1; the writer never emits them.

**Per-segment geometry** (decoded once per segment, never per row): `r = u / num_col_chunks`,
`j = u % num_col_chunks`, `n = r / tensor_h_tiles`, `ht = r % tensor_h_tiles`,
`rows_live = min(32, H - ht*32)`, first tile `r*tensor_w_tiles + j*segment_width`,
`tiles_live = min(segment_width, tensor_w_tiles - j*segment_width)`,
`seg_bytes = min(segment_width*64, W*2 - j*segment_width*64)`, first output page `n*H + ht*32`,
byte offset in the page `j*segment_width*64`. Tiles of the segment past `tiles_live` are not read;
compute untilizes their stale slots; the writer's `seg_bytes` excludes them.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 overlap | `block_segments` is the coarsest that fits; small per-core assignments (e.g. `(1, 4096, 2560)`: at most 5 segments per core) run as ONE block, serializing read -> untilize -> write on each core | cap `block_segments` so every core has >= 4 blocks; measure on `(1, 4096, 2560)` and `(16, 4096, 2560)` |
| L2 grid synchronization | tiny tensors (16-240 tiles: `(1, 512)`, `(1, 64, 512)`, `(1, 96, 2560)`) spread over up to `G` cores leave 1-3 tiles per core; launch and pipeline fill dominate | a floor of >= 4 tiles per core (fewer cores) |
| L3 writer issue-bound on narrow rows | `W <= 32` (p1, p4, `(1, 64, 128, 32)`): 32 row writes per tile on one RISC; `TensorAccessor` page math per row | (a) stateful single-packet writes with incremental bank/offset; (b) split row writes across both data-movement RISCs (catalog `split_reader`): compute alternates segments between two output CBs, one consumed by each RISC — one consumer per CB |
| L4 bank stagger | every core starts its tile walk at tile `r0*tensor_w_tiles` and its row writes at page `n*H + ht*32`; with `tensor_w_tiles` and 32 multiples of the DRAM bank count all cores hit the same bank in lockstep | rotate each core's start tile / start row by `core_idx % num_dram_banks` (catalog `bank_stagger`) |
| L5 untilize sub-block | `segment_width > 8` with no divisor in 4..8 runs 1- or 2-tile sub-blocks | round `segment_width` up to a multiple of 8 when > 8 (stale tail tiles cost compute only) |
| L6 balance slack | `balance_slack = 1/8` trades up to 12.5% load imbalance for fewer column chunks (fewer, larger writes) | `balance_slack` 0 and 1/4 on `(16, 128, 2560)`, `(1, 4096, 2560)` |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| input tiles (DRAM or L1 interleaved) -> `cb_tiles` | TILE bytes; CB declared `UInt16` | reader (RISCV_1, NoC0): `TensorAccessor::get_noc_addr(tile_id, range_offset)` + `noc_async_read` per live range; one barrier per block | an L1 input is read over the NoC from its banks the same way |
| `cb_tiles` -> DEST -> `cb_rows` | `UInt16` both sides (bit reinterpretation of bf16) | compute: `compute_kernel_lib::untilize` (unpack -> MOVA2D with the zero flag preserved -> `pack_untilize`) | `fp32_dest_acc_en = False`, `dst_full_sync_en = False` |
| `cb_rows` -> output pages (DRAM or L1 interleaved) | ROW_MAJOR bytes | writer (RISCV_0, NoC1): `noc_async_write` of `seg_bytes` per live row to `get_noc_addr(n*H + ht*32 + i, j*segment_width*64)` | stride in `cb_rows` = `segment_width*64` B per row |

**TARGET axes as regimes.**

| TARGET value beyond Phase 0 | Axis it cuts / changes | Class | Regime |
|-----------------------------|------------------------|-------|--------|
| `memory_layout: HEIGHT_SHARDED` | tile-rows (`n`, `ht`): independent | knob-turn (placement: CB on the shard, no reads) | R3 |
| `memory_layout: BLOCK_SHARDED` | tile-rows and tile-columns: both independent; column shards write disjoint offsets of shared pages | knob-turn (placement) | R3 |
| `dtype: bfloat8_b` | none — datapath of `untilize_block` (real Bfp8_b -> bf16 conversion) and CB formats | knob-turn in format, same schedule | R2 |
| `layout: ROW_MAJOR` | none — mirrored block operations (`load_rows`, `tilize_block`, `store_tiles`) on the same segment split | new block operations, same blocking | R4 |
| `output_dtype: bfloat8_b` | none — `tilize_block` packs `Bfp8_b`; reader zero-fills live groups' padding | format | R4 |

No regime, built or deferred, needs Tensix-to-Tensix communication.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a segment (1 tile-row x `segment_width` tiles); a block is `block_segments` consecutive segments of one core |
| Grid | full compute grid `G = grid.x * grid.y` (`device.compute_with_storage_grid_size()`), a runtime value; cores used `cores_used = min(G, num_segments)`, the first ones in row-major order |
| Per-core work | `ttnn.split_work_to_cores(grid, num_segments, row_wise=True)`: group 1 gets `ceil(num_segments / cores_used)` contiguous segments, group 2 one fewer; each core gets `(start_segment, num_segments_this_core)` |
| Remainder | segment ranges are contiguous in `u`; ragged chunk widths, partial tile-rows and partial tiles are per-segment runtime extents (`tiles_live`, `rows_live`, `seg_bytes`), never a different program |
| Balance slack | `balance_slack = 1/8` (host constant, `Plan` field): the largest per-core load excess over `ideal` accepted to keep fewer column chunks |
| Tile geometry | per image: `tensor_h_tiles = ceil(H/32)`, `tensor_w_tiles = ceil(W/32)`, `tensor_tile_rows = N * tensor_h_tiles`; never `floor(N*H/32)` |
| Core ranges (BB-17) | one kernel set on the union of used cores: at most a block of full grid rows plus one partial row |

**Plan algorithm** (host, memoized):

```
B_cb            = 1347776 - out_l1_bank_bytes - cb_align_slack          # ledger symbol table
seg_width_max   = floor(B_cb / (tiles_depth*tile_bytes_in + rows_depth*tile_bytes_rows))
ideal           = ceil(tensor_tile_rows * tensor_w_tiles / G)
for c in 1 .. tensor_w_tiles:                                           # smallest c wins
    w = ceil(tensor_w_tiles / c);  if w > seg_width_max: continue
    chunks = ceil(tensor_w_tiles / w);  U = tensor_tile_rows * chunks
    if ceil(U / G) * w <= ideal * (1 + balance_slack): break
segment_width = w;  num_col_chunks = chunks;  num_segments = U
block_segments = max(1, min(ceil(U / min(G, U)),
                     floor(B_cb / (segment_width * (tiles_depth*tile_bytes_in + rows_depth*tile_bytes_rows)))))
```

`c = tensor_w_tiles` always satisfies the balance test (its load is `ideal`), so the loop ends.
Illustrative only, for `G = 110`: `(16, 4096, 2560)` -> `c = 1`, `segment_width = 80`,
`block_segments = 2`; `(16, 128, 2560)` -> `c = 5`, `segment_width = 16`; `(1, 2048, 2048, 3)` ->
`c = 1`, `segment_width = 1`, `block_segments = 164`; `(1, 131072)` -> `c = 98`, `segment_width = 42`,
one segment on each of 98 cores; `(1, 4096, 2560)` -> `c = 4`, `segment_width = 20`, at most 5
segments (one block) per core.

**Kernel configuration.** Compute: `ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=False,
dst_full_sync_en=False)`. Reader: `ReaderConfigDescriptor()` (RISCV_1 / NoC0). Writer:
`WriterConfigDescriptor()` (RISCV_0 / NoC1). Profiler zones (BB-21): every block operation and
every blocking wait (`reserve_back`, `wait_front`, read barrier, writes-flushed, final barrier) in
its own `DeviceZoneScopedN`; nothing changes in the plain build.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_tiles` | 0 | `tile_bytes_in` = 2048 (32x32 x 2 B) | `tiles_depth * block_segments * segment_width` | live set spans `ht` (`block_segments`), `wt` (`segment_width`), `i`/`c` (whole tile slot); streams over the core's segments | `UInt16` (bit view of bf16) | reader | compute | whole program |
| `cb_rows` | 1 | `tile_bytes_rows` = 2048 (32 rows x 64 B per tile column) | `rows_depth * block_segments * segment_width` | live set spans `ht` (`block_segments`), `wt` (`segment_width`), `i`/`c`; streams over the core's segments | `UInt16` | compute | writer | whole program |

Two CBs is the minimum: the untilize helper cannot run in place (`untilize_helpers.inl:153-154`),
and the writer cannot emit rows from tile order without 32-B fragmentation (R6 is the narrow-W
exception, deferred). Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_segments x segment_width` tiles (live ranges only) | no (raw NoC; no tile-reading dataflow helper exists) | input tensor (accessor) | `cb_tiles`, `block_segments*segment_width` pages reserved and pushed once (terminal block: its actual segments) | block visible to compute |
| 2 | `untilize_block` | `block_segments x (1 x segment_width)` | yes — `compute_kernel_lib::untilize` | `cb_tiles`, `segment_width` pages waited/popped per segment | `cb_rows`, `segment_width` pages reserved/pushed per segment | segment k of the block packed as 32 rows x `segment_width*64` B |
| 3 | `store_block` | `block_segments` x `rows_live` rows x `seg_bytes` | no (raw NoC; helper rejected below) | `cb_rows`, waited cumulatively per segment, popped once per block after writes-flushed | output tensor (accessor) | block slots free for compute |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| (compute setup) | helper prerequisite | `compute_kernel_hw_startup(cb_tiles, cb_rows)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:113` | 2-arg form, srcA = srcB = `cb_tiles` (`UInt16` -> zero flag preserved, `llk_math_common.h:54-59`) | `cb_tiles` | `cb_rows` | — |
| `untilize_block` | helper | `compute_kernel_lib::untilize<segment_width, cb_tiles, cb_rows, InitUninitMode::InitAndUninit, WaitMode::WaitBlock, ReconfigureRegisterDatatypeMode::NoReconfigure>(in, out, segments_this_core)` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:163` (impl `untilize_helpers.inl:151-282`) | one call per core over all its segments | `cb_tiles` | `cb_rows` | `block_width_tiles = segment_width` (CT); `num_blocks = segments_this_core` (RT) |
| `load_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)`; `noc_async_read`; `noc_async_read_barrier`; `DataflowBuffer::reserve_back` / `push_back` / `get_write_ptr` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165`; `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547`, `:1921`; `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:243-244`, `:407` | per live range: `(tile_id, range_offset, range_bytes)` | input tensor | `cb_tiles` | reserve/push count `block_segments*segment_width` |
| `store_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)`; `noc_async_write`; `noc_async_writes_flushed`; `noc_async_write_barrier`; `DataflowBuffer::wait_front` / `pop_front` / `get_read_ptr` | `tensor_accessor.h:165`; `dataflow_api.h:876`, `:1980`, `:1951`; `dataflow_buffer.h:245-246`, `:408` | per live row: `(page n*H + ht*32 + i, offset j*segment_width*64, seg_bytes)` | `cb_rows` | output tensor | pop count `block_segments*segment_width` |

**Helpers considered and rejected.**

| Raw block operation | Candidate helper | Mismatch (file:line) | Why it cannot express this block |
|---------------------|------------------|----------------------|----------------------------------|
| `load_block` | `dataflow_kernel_lib::read_sticks_for_tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:87-93` | reads row-major sticks for a tilize; this operation reads tile pages (and sub-tile live ranges). No tile-page reading helper exists in `kernel_lib`. Gap that would close it: a tile-page block reader taking a per-tile live-range mask |
| `store_block` | `dataflow_kernel_lib::write_sticks_after_untilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.inl:224-238` | (1) page index `start_page + block*32 + row` (`:230`) assumes rows of consecutive tile-rows are consecutive pages — wrong for multi-image `h_non_aligned` shapes (`(16, 1, 2560)`: tile-row `n` maps to page `n*H`); (2) the row stride is re-derived from `row_bytes` (`:207-208`), not the nominal `segment_width*64`, so a ragged last column chunk reads at the wrong stride; (3) a full `noc_async_write_barrier` per tile-row (`:235`) instead of one flush per block. Gap: per-tile-row page base + explicit row stride + caller-owned sync |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| A float-format CB makes the copy lossy | MOVA2D flushes zero-exponent datums for `Float16_b` operands; random inputs almost never hold subnormals or `-0.0`, so every randn cell still passes | both CBs `UInt16`; the acceptance test and loose cases inject `-0.0`, subnormals, infinities and a NaN payload |
| UInt16 transparency is documented, not yet observed on this silicon | the LLK comment is the evidence; nobody has run this exact path with special patterns | the special-pattern test is the gate; on failure switch `untilize_block` to R5 (same CBs, no other change) |
| Per-image H padding | page of row `i` of tile-row `r` is `n*H + ht*32 + i`, not `r*32 + i`; only multi-image `h_non_aligned` shapes (`(16, 1, 2560)`, `(8, 1, 2560)`) expose it | decode `r -> (n, ht)` per segment |
| Poisoned input padding | the producer leaves anything in tile padding; host-made tensors hold zeros, hiding a read of padding | the reader reads live face rows only; the writer writes `seg_bytes` of live rows only — padding has no path to the output |
| Ragged last column chunk | `segment_width` is a template param; the last chunk holds `tiles_live < segment_width` tiles | nominal page counts everywhere; stale slots are untilized and never written |
| Coarse blocks fill L1 | `block_segments` targets the whole 1347776-B budget; a miscount fails at launch only beside the model's tensors | budget is exact (`l1_ledger.md`), includes the L1 output's own pages per bank and CB alignment slack; plan never reads free L1 |
| p1 target below the physical floor | p1 `(1, 2048, 2048, 3)` must read >= 131072 tiles x 1024 B = 134 MB (live columns sit in faces 0 and 2; DRAM granule 32 B) and issue 4,194,304 separate 6-B row writes. 134 MB alone takes >= 262 us at Blackhole's ~512 GB/s nominal peak — above the lead's 147.46 us target (computed from 25 MB of logical bytes) | build the floor-reaching scheme (live-face reads, one write per row, full grid); report the target as unreachable rather than chase it; L3/L4 are the levers for the write-issue side |
| Kernels left with NoC transactions in flight (BB-03) | the next program starts immediately | reader ends on a read barrier, writer on `noc_async_write_barrier` |
| CB base alignment for DRAM reads | BH DRAM reads require local and NoC addresses equal mod 64; partial-tile ranges start at face offsets | ranges start at multiples of 64 inside 2048-B slots; verify once under `--dev` (watcher) |
| Rebuilding the descriptor per call (BB-07) | `generic_op` callers build descriptors in Python | memoize plan + descriptor on the cache key; patch buffer addresses only |
| bf8b refinements (R2, R3) | `Bfp8_b` SrcA keeps the flush flag at 0, so a `-0.0` the host keeps in its bf8b rendering could come back `+0` | the case-29 special-pattern loose case gates R2; if it fails, the refinement needs a datacopy that applies the preserve flag (`cmath_common.h:108-112`) |
