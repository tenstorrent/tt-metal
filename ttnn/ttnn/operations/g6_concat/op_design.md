# Operation Design: g6_concat

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement. No compute kernel: nothing is computed, and no value passes through unpack/DEST/pack. |
| Goal | Join K ≥ 1 interleaved device tensors of rank 3 or 4 along one dim into a newly allocated tensor, bit for bit, in **one** device program. Every logical input element crosses its buffer once and every logical output element is written once. |
| Math | `d = dim + r if dim < 0 else dim`; `o_k = n_0 + … + n_{k-1}`; `out[i_0, …, o_k + j, …, i_{r-1}] = x_k[i_0, …, j, …, i_{r-1}]`, `0 ≤ j < n_k` |
| References | Task spec (contract; wins on conflict) · `eval/golden_tests/g6_concat/feature_spec.py` (TARGET / INPUTS / INVALID / LOOSE_CASES) · `eval/golden_tests/g6_concat/helpers.py` (oracle `pytorch_g6_concat`, bit-exact grading, `fill_implicit_tile_padding` poisoning) · `eval/golden_tests/g6_concat/test_regression.py` (repeat / cache-key / trace replay) · `eval/op_template.py` (registry model) · `.claude/references/blocking-model.md` · `.claude/references/l1-footprint-discipline.md` · `.claude/references/noc-alignment-in-tt-metal.md` · catalog `ttnn/ttnn/operations/examples/master.md` (entries listed under *Catalog evidence per knob*) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `tensors` | `list[ttnn.Tensor]`, positional | yes | K ≥ 1 and K ≤ `K_max` (the common-RT-arg cap, see Mechanism caps). All on one device, rank 3 or 4, same rank, dtype and layout, equal sizes in every dim except `d`, interleaved (never sharded). | — | Host. Shapes go into common RT args; the placement class goes into the CT accessor args. |
| `dim` | `int`, positional | no | `[-r, r-1]`. Canonicalized to `d` **before** any tagging or gating, so `dim=-1` and `dim=3` (rank 4) are the same call and the same cell. | `0` | Host. Selects the regime and page map, and derives the common RT args. |
| `memory_config` | `ttnn.MemoryConfig`, keyword-only | no | `None` or an interleaved config. Its buffer is DRAM (Phase 0) or L1 (TARGET). | `None`, which means `ttnn.DRAM_MEMORY_CONFIG` | Host. Sets the output allocation and the writer's accessor CT args. |

No `compute_kernel_config`, no `out=`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | Rank 3 or 4. All inputs equal except at `d`. `n_k ≥ 1`. The second-to-last dim is any size; its tile padding rows are **not data**, and no output value may depend on them. |
| Dtype | One dtype shared by all inputs: `bfloat16` (Phase 0), `float32`, `bfloat8_b` (TARGET). |
| Layout | One layout shared by all inputs: `TILE` (Phase 0), `ROW_MAJOR` (TARGET). |
| Memory | Interleaved. Each input is read where it lives: DRAM (Phase 0), or L1 / mixed (TARGET). |

### Output

| Property | Value |
|----------|-------|
| Shape | `x_0.shape` with `N = Σ n_k` at `d` |
| Dtype | the inputs' dtype |
| Layout | the inputs' layout |
| Memory | Interleaved in the buffer `memory_config` names; DRAM interleaved when `memory_config is None`, including when the inputs are in L1. Allocated with `ttnn.allocate_tensor_on_device(Shape, dtype, layout, device, mem_cfg)`. |
| Padding | Unspecified. Writers move only the valid-row prefix of each output tile, so tile-padding rows keep whatever DRAM held. |

### Support contract (registry)

Axis names match `TARGET` in `feature_spec.py`. `validate()` builds the scenario dict `{"shapes", "dim", "dtype", "layout", "input_buffers", "memory_config"}` from the call, exactly as `eval/golden_tests/g6_concat/axes.py::scenario_of_call` does.

| Tagger | Returns | Rule |
|--------|---------|------|
| `tag_dtype` | `ttnn.bfloat16 \| ttnn.float32 \| ttnn.bfloat8_b` | `scenario["dtype"]` (shared) |
| `tag_layout` | `ttnn.TILE_LAYOUT \| ttnn.ROW_MAJOR_LAYOUT` | `scenario["layout"]` (shared) |
| `tag_concat_dim` | `"outer" \| "h" \| "w"` | `d = dim % r` → `"w"` if `d == r-1`, `"h"` if `d == r-2`, else `"outer"` |
| `tag_alignment` | `"tile_aligned" \| "w_non_aligned" \| "h_non_aligned"` | `"w_non_aligned"` if any `shape[-1] % 32`; else `"h_non_aligned"` if any `shape[-2] % 32`; else `"tile_aligned"` |
| `tag_input_buffer` | `"dram" \| "l1" \| "mixed"` | set of `scenario["input_buffers"]` |
| `tag_output_buffer` | `"dram" \| "l1"` | `"l1"` iff `scenario["memory_config"] == "l1"` |

| Declaration | Phase 0 value |
|-------------|---------------|
| `SUPPORTED` | `dtype: [bfloat16]`, `layout: [TILE_LAYOUT]`, `concat_dim: ["outer", "h", "w"]`, `alignment: ["tile_aligned", "h_non_aligned"]`, `input_buffer: ["dram"]`, `output_buffer: ["dram"]` |
| `EXCLUSIONS` | `[]`. Rows of several inputs in one tile (`h` × `h_non_aligned`) are **in** Phase 0; never exclude them. |
| `validate()` order | (1) Structural errors raise `ValueError`: empty list; rank ∉ {3, 4}; rank / dtype / layout differ between inputs; `dim ∉ [-r, r-1]`; sizes differ outside `d`; a sharded input or a sharded `memory_config`; `K > K_max`. (2) Canonicalize `dim`, then build the axes with the taggers in declaration order. (3) SUPPORTED → `UnsupportedAxisValue`. (4) EXCLUSIONS → `ExcludedCell`. INVALID is never checked. `g6_concat()` calls `validate()` as its first line. |
| Plan report (BB-08, BB-21) | `select_plan(tensors, dim=0, *, memory_config=None) -> dict` is exported from `ttnn.operations.g6_concat`. The entry point calls this same function, so the report is the plan the call ran. Keys: `"regime"` (`"page_copy"` / `"row_gather"`), `"num_cores"`, `"block_pages"`, `"buffer_depth"`, `"output_pages"`, `"roles"` (logical core → `"mover"`, i.e. the reader + writer pair). The plan depends only on shapes, dtype, layout, placements and grid, never on free L1. A production shape therefore gets the same plan wherever it is called. |

Structural impossibilities beyond `INVALID`: none found.

## Blocking Model

The op computes nothing. A block is therefore a set of output pages that one core moves in one uninterrupted pipeline pass: one reserve/push on the reader, one wait/pop on the writer, and the regime's barriers once per block.

### Tensor axes and their characters

Shown for rank 4 `[D0, D1, H, W]`; rank 3 drops `D1`. "Tile" units use `ceil(·/32)` per image.

| Tensor axis | Character (+ one-clause reason) | Folds into blocking axis |
|-------------|--------------------------------|--------------------------|
| `lead`: dims `< d` | **independent**: `out[a, …]` reads only `x_k[a, …]` | page_copy: `out_page`. row_gather: `out_tile_row` (image index) |
| `seam`: dim `d` | **independent**: output position `o_k + j` copies position `j` of input `k`. Segmenting by `k` is an address map, not a combine. | page_copy: `out_page` (K contiguous segments per page-matrix row). row_gather: the rows of `out_tile_row` |
| `inner`: dims strictly between `d` and `H` (only when `d ≤ r-3`) | **independent**: same reason as `lead` | `out_page` |
| `tile_row`: `H` in tiles | **independent**: a tile row's values do not depend on any other tile row | page_copy: `out_page`. row_gather: it is the seam, giving `out_tile_row` |
| `tile_col`: `W` in tiles | **independent**: same reason | page_copy: `out_page`. row_gather: `tile_col` |
| `in_tile_row`: the 32 rows inside a tile | **independent**: each output row has exactly one source row | never split. Transfers move a page's valid-row prefix; row_gather places rows at their destination. |

There is **no dependent axis**, so no combine regime exists. There is **no reuse-shared operand**; see *Operand-reuse check*.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `out_page` (page_copy): the output's page index in storage order, `pm_row · pm_cols + pm_col` (page matrix in *Work Distribution*) | independent: each output page has exactly one source page, so every page is moved identically and the whole tile-index space folds honestly into one axis | `block_pages` | `max(1, min(ceil(core_out_pages_max / buffer_depth), cap_l1))`; `cap_l1 = floor(L1_OP_BUDGET / (buffer_depth · page_bytes))` (bf16: 329; fp32: 164). | Computed once in the host plan, passed as a CT arg to reader and writer. CB capacity and every loop bound derive from it. | Contiguous `out_page` ranges, `split_work_to_cores(grid, P, row_wise=True)` over `num_cores = min(grid_cores, ceil(P / min_pages_per_core))` cores. The grid is `device.compute_with_storage_grid_size()`, a runtime parameter. | knob-turn |
| `out_tile_row` (row_gather): image × output tile row | independent: an output tile row's content comes from input rows selected by its own row range only | `block_tile_rows` | **1.** A block's placement map (its run list) belongs to exactly one output tile row. Every production row_gather output has exactly one tile row per image (Qwen 79/89/143: `H_out` = 16/2/8), so 1 is the whole extent on every row_gather geometry in INPUTS. | Regime constant (blocks are cut at tile-row ends). | Same contiguous `out_page` ranges as page_copy, so output tile rows *are* split across cores. Exact row-range reads mean a source tile shared by two cores is read as two disjoint row ranges and never duplicated. | knob-turn (multi-row block = one run list per row under one set of barriers) |
| `tile_col` (row_gather) | independent: columns never mix | `block_pages` (the same single source) | `max(1, min(ceil(core_out_pages_max / buffer_depth), cap_l1_rg, out_tile_cols))`; `cap_l1_rg = floor((L1_OP_BUDGET − s·dram_read_align) / ((buffer_depth + s) · page_bytes))`, `s ∈ {0,1}` = parity stage present (bf16 on BH: 219) | host plan → CT | as above | knob-turn |
| `in_tile_row` (both) | independent | none: the tile is the format's addressing unit | 32 (whole tile; valid-row trimming narrows *bytes*, not the block) | tile format | not split | — |

Knob sources (single source each):

| Knob | Defined once in | Threaded as |
|------|-----------------|-------------|
| `buffer_depth` | host constant `BUFFER_DEPTH = 2` | CT arg; CB capacity = `buffer_depth · block_pages` |
| `block_pages` | host plan function (formula above) | CT arg to reader and writer; CB capacities; block cutting in both kernels |
| `min_pages_per_core` | host constant `MIN_PAGES_PER_CORE = 1` | sets `num_cores` |
| `L1_OP_BUDGET` | host constant `1347776` (model leaves this per core). When the output is in L1 (TARGET), the output's per-core L1 share is subtracted. | `cap_l1` |
| grid | `device.compute_with_storage_grid_size()` | `split_work_to_cores` |

Interleaved default, as two ordered steps: (1) spread the output pages over the full grid; (2) take the coarsest block that fits within a core's assignment. The `ceil(core_out_pages_max / buffer_depth)` term is that bound. A block equal to the whole assignment would turn the depth-2 ring into single buffering (read fully serialized against write, the overlap lamp), so the coarsest block that keeps the `buffer_depth` knob meaningful is `assignment / buffer_depth`. The L1 cap binds only on very large per-core assignments.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_moved_pages` | `buffer_depth` | 2 | The reader fills block `b+1` (DRAM reads on NoC0) while the writer drains block `b` (DRAM writes on NoC1), so the two streams overlap per core. |
| `cb_parity_stage` (row_gather only) | — (one block) | 1 | The stage is reader-private and is emptied inside the block (phase-2 barrier) before the next block's phase 1 writes it. A second stage buys nothing unless the reader issues the next block's reads before draining this one (trid lamp). |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| NoC DRAM-read congruence: `(L1 dst % A) == (DRAM src % A)`, `A = NOC_DRAM_READ_ALIGNMENT_BYTES` (64 on BH, 32 on WH; `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`). Size may be any byte count. | Destination row of a row_gather read: a bf16 face row is 32 B, so on BH a run whose `(src_row − dst_row)` is odd lands at a non-congruent offset. | Each run gets a class `c = (src_in_tile_off − dst_in_tile_off) mod A`. `c = 0` is read straight into the output slot. `c ≠ 0` is read into `cb_parity_stage` at base `≡ slot_base + c (mod A)`, then L1→L1 copied. For bf16 on BH the only nonzero class is 32. fp32 (64 B face row) and bf16 on WH have none: no stage, `s = 0`. | Silently corrupted rows (the watcher stops the device only under `--dev`). |
| CB base alignment. CBs are allocated at the DRAM alignment (`tt_metal/impl/program/program.cpp:2505-2509`, `allocate_dfb(..., get_alignment(BufferType::DRAM))`), and `page_bytes % 64 == 0`. | — | Every `cb_moved_pages` slot base is `≡ 0 (mod A)`, so class 0 is exactly "direct". The reader computes classes from the actual `get_write_ptr` values and never from an assumed base. | A hard-coded class table breaks on an arch whose CB alignment differs. |
| L1→L1 copy congruence mod 16 (`NOC_L1_READ_ALIGNMENT_BYTES`, `noc_parameters.h:375`) | Stage→slot spans | Spans are whole 32 B face rows, and stage and slot bases differ by 32, so both are `≡ 0 mod 16`. | — |
| DRAM-write congruence mod 16 (`noc_parameters.h:380`) | Writer's valid-row pieces | Pieces sit at the same in-tile offset in the slot and in the output page, so they are congruent by construction. | — |
| Valid-row trimming needs `face_row_bytes % 16 == 0` (bf16 32 B, fp32 64 B). In `bfloat8_b` each face row has a 1 B shared exponent in a separate 64 B block. | Transfer size per page | `bfloat8_b` (TARGET) moves whole pages only (`vr` clamped to 32). bf8b row_gather is a `deferred` regime. | Exponent bytes are lost or misplaced: wrong values. |
| Row_gather block within one output tile row | `block_pages` (row_gather) | `≤ out_tile_cols`. Blocks are cut at tile-row ends (ragged blocks there). | One run list would be applied to columns of another tile row: wrong rows. |
| Common runtime args per kernel. The portable floor is `max_runtime_args = 341` (`tt_metal/api/tt-metalium/kernel_types.hpp:49`); the per-input table is 2 words per input plus a fixed header. | K | `K_max = (341 − header_words) // 2` (≥ 160). The host raises `ValueError` beyond it. Production K ≤ 16. | Launch refused, or args truncated. |
| CB ring-wrap (`ttnn-cb-memory-fundamentals.md` §CB Ring-Wrap Invariant) | push/pop quanta | Every push and pop is the nominal `block_pages`. Ragged blocks narrow only the NoC transfers. Capacity = `buffer_depth · block_pages`. | Pointer overshoot into neighbouring L1: silent corruption on the second launch. |
| L1 budget, BB-05 (`L1_OP_BUDGET` = 1347776 B per core) | `block_pages` | `cap_l1` / `cap_l1_rg` (closed form, no search). | Launch failure under model L1 pressure only. |

### Regimes

Named boundary for the minimum: **each logical input row segment crosses DRAM once, and each logical output row segment crosses DRAM once.** A full tile counts as 32 rows; a page's tile-padding rows are not data.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `page_copy` | **built** | `not row_gather` | `block_pages` consecutive `out_page`s (crosses segment and page-matrix-row boundaries freely) | **Minimum.** Each output page has one source page. Reader and writer move that page's valid-row prefix `vr(p)` (one transfer when `vr = 32`; ≤ 3 when `16 ≤ vr < 32`; 2 when `vr < 16`). Inputs cross DRAM once, output once. No cross-core traffic. Covers every Phase 0 INPUTS cell except Qwen 79/89/143. RM outer/h (TARGET) is this regime with page = stick: a knob-turn. | Per block, intended once each: reader `cb_reserve_back` + one `noc_async_read_barrier` + `cb_push_back`; writer `cb_wait_front` + one `noc_async_writes_flushed` + `cb_pop_front`. No inits, reconfigs, multicast or semaphores (no compute). Pipeline fill/drain is paid once per core. |
| `row_gather` | **built** | `layout == TILE and d == r-2 and any(shape_k[d] % 32 != 0 for k in 0..K-2)`, i.e. some input offset `o_k` (k ≥ 1) is not tile-aligned | 1 output tile row × `block_pages` tile columns | **Minimum on DRAM**: each run reads exactly its rows (never padding) and each output page is written as its valid-row prefix. **Plus a core-local term**: off-parity rows cross L1 once more (stage → slot span copy). Qwen 79: 1.99 MB in + 1.99 MB out DRAM, 1.86 MB L1→L1. | Per block: the run list is derived once (not per tile), **3 barriers** (stage reads, span copies, direct reads) and 1 push. The writer pays as in page_copy. Wider blocks amortize the 3 barriers over more columns. |
| `whole_tile_staging` row gather | **rejected**: superseded by `row_gather`'s exact-row reads | — | — | Reads every source tile whole, padding included. Qwen 79: 63.6 MB in vs 1.99 MB (32× the input bytes). | — |
| `per_run_source_staging` | **rejected**: superseded by `row_gather`'s parity staging | — | — | Same DRAM bytes as `row_gather`, but a source-image tile per run and one copy per run piece. Qwen 79: 32 local copies and a barrier per run per tile, vs 2 span copies per tile and 3 barriers per block. A dead end: its staging layout (source image) is not the one the better scheme uses (destination image). | — |
| `single_core` | **rejected**: superseded by the full-grid `out_page` split | — | — | Same bytes; 1/num_cores of the bandwidth. | — |
| `copy_then_fixup` (two programs, or host pad/slice/to_layout) | **rejected**: forbidden by the one-dispatch rule (BB-01/BB-02); superseded by `row_gather` inside the one program | — | — | ≥ 2× crossings of the output. | — |
| `column_gather` (W not tile-aligned: tile columns of several inputs share an output tile) | **deferred**: `"w_non_aligned"` is not a TARGET value and no production input has it; `validate()` refuses it via SUPPORTED | `d == r-1 and any(shape_k[-1] % 32 != 0 for k < K-1)` | 1 output tile × face columns | Column moves are 2 B-granular in bf16, which is below the NoC's 16 B L1 congruence, so they need RISC-V packing. The run-list / stage / slot structure of `row_gather` carries over with columns for rows. | — |
| `rm_w_stick_splice` (ROW_MAJOR, `d == r-1`) | **deferred**: no INPUTS cell (every RM production call joins along outer or h, which is `page_copy` with page = stick) | `layout == ROW_MAJOR and d == r-1` | `block_pages` output sticks | Each output stick is K input sticks written at byte offsets `Σ W_k'·elem`. That needs `(W_k·elem) % 16 == 0` for direct DRAM writes, otherwise a bounce. Same reader/writer split; per-stick piece list instead of per-page. | — |
| `bf8b_row_gather` | **deferred**: no INPUTS cell (every bf8b production call joins along `w`, i.e. `page_copy` on whole pages) | `dtype == bfloat8_b and row_gather` | as `row_gather` | 1 B exponents per face row need RISC-V byte moves. 16 B mantissa rows on BH give 3 parity classes, so 3 stages. Same run list. | — |

Selection is the exact host predicate in the table (`select_plan`). **Regime-pinned tests** are required and present in the acceptance test: Qwen 79/89 (`row_gather`), a mixed-height rank-4 h-concat (`row_gather`), h-concat with only the *last* input misaligned (`page_copy`: the predicate edge), FLUX 21, Qwen 109 and Qwen 81 (`page_copy`). The predicate does not depend on the grid, so no regime is grid-dependent.

### Traffic ranking

Counted per tier, for the candidates the op admits. No dependent axis exists, so there is no combine candidate.

| Candidate split | DRAM crossings | Cross-core | Occupancy on INPUTS | Verdict |
|-----------------|----------------|------------|---------------------|---------|
| **Contiguous `out_page` ranges, valid-row transfers** | in ×1, out ×1, logical rows only | 0 | `min(grid, P)`. All production shapes with `P ≥ grid` fill the grid, including the Qwen 79/160 single tile row × 1940 columns, which only a width-capable split fills (catalog `width_split`). | **chosen**: minimum bytes, full grid, and the writer gets one contiguous page run. |
| Input-major ranges (each core a range of one input's pages) | same | 0 | same | Ties on bytes. Loses the tiebreak: row_gather's output tile needs rows of up to 16 inputs on one core, and output-major keeps that local. |
| One core per input | same | 0 | ≤ K ≤ 16 | rejected: under-fills the grid |
| Page-matrix rows only (`lead`/`tile_row`) | same | 0 | `pm_rows` = 1 for every dim-0 case | rejected: occupancy |
| Seam / columns only | same | 0 | `pm_cols` = 8 for Qwen 151 | rejected: occupancy |
| Whole-page transfers (no valid-row trimming) | in ×1, out ×1, **padding included** | 0 | full | rejected: Qwen 160 moves 3.97 MB each way vs 124 KB (32×); Qwen 80 2× |
| Whole-tile staging for row_gather | input **padding** ×1 (32× bytes on Qwen 79) | 0 | full | rejected (regime row) |
| Bank-affine page sets (a core owns same-bank pages; m pages per NoC transaction) | same bytes, fewer and larger transactions | 0 | full | not a byte difference. Perf lamp. |

### Operand-reuse check and stall-shadow check

| Check | Finding |
|-------|---------|
| Operand reuse, for each (input, `out_page` split) pair | Input `k` varies along the split axis: each of its pages (row ranges, in row_gather) is read by exactly one block on one core. No input is re-read by several cores, so there is no broadcast regime. |
| Stall: reader `noc_async_read_barrier` (both regimes) | Independent work that could run in the shadow is the next block's reads, but the barrier drains all reads. The depth-2 ring already places the *writer's* block `b` into this window. Filling it on the reader needs transaction-scoped barriers: trid lamp. |
| Stall: row_gather phase-1 barrier (waiting for off-parity stage reads) | Phase-3 direct reads are independent of phase-1 *data*, but **not** of phase 2: the span copy writes whole face spans and would overwrite earlier-landed direct rows. Lifting phase 3 into the phase-1 shadow is legal only if phase 2 copies exactly the off-parity row runs (disjoint bytes). Phase 0 keeps span copies (2 copies per tile on Qwen 79, vs 16 for exact runs). The reorder is perf lamp L6. No floating-point order is involved (nothing is computed). |
| Stall: writer `cb_wait_front` | Nothing on the writer is independent of the reader's data. The depth-2 ring is the overlap. |

### Block schedule

page_copy (reader and writer realize their halves; adjacent blocks pipeline through `cb_moved_pages`):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    gather_pages_block(block_idx);   // reader: block_extent output pages ← their source pages, valid-row prefixes
    store_pages_block(block_idx);    // writer: block_extent pages → output pages, valid-row prefixes
}
```

row_gather:

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    derive_runs_block(block_idx);         // reader: run list of this block's output tile row
    stage_offparity_rows_block(block_idx); // reader: off-parity runs DRAM → cb_parity_stage, at destination rows
    merge_stage_spans_block(block_idx);    // reader: per face, one L1→L1 span copy stage → output slot
    read_onparity_rows_block(block_idx);   // reader: on-parity runs DRAM → output slot, at destination rows; push
    store_pages_block(block_idx);          // writer: identical to page_copy
}
```

| Block operation | Block shape | Resident across it | Intended fixed-cost frequency |
|-----------------|-------------|--------------------|-------------------------------|
| `gather_pages_block` | `block_extent ≤ block_pages` output pages | one `cb_moved_pages` slot (block_pages pages) | 1 reserve, 1 read barrier, 1 push per block. Source accessor rebuilt only when the walk crosses into another input's segment. |
| `derive_runs_block` | 1 output tile row (`vr` rows) | nothing (recomputed on the fly per phase; no per-run storage) | once per block (integer walk over `row_offset[]`) |
| `stage_offparity_rows_block` | off-parity runs × `block_extent` columns | stage (one tile image per column) | 1 read barrier per block (skipped when the tile row has no off-parity run) |
| `merge_stage_spans_block` | ≤ 4 face spans × `block_extent` columns | stage + reserved output slot | 1 read barrier per block (skipped as above) |
| `read_onparity_rows_block` | on-parity runs × `block_extent` columns | reserved output slot | 1 read barrier + 1 push per block |
| `store_pages_block` | `block_extent` pages | one filled slot | 1 wait, 1 `noc_async_writes_flushed`, 1 pop per block. **One `noc_async_write_barrier` at kernel end** (BB-03). |

Profiler zones (BB-21; `DeviceZoneScopedN`, no-op outside the zone build): each block operation above is one zone, and each blocking wait (`cb_reserve_back`, `cb_wait_front`, every barrier, the final write barrier) is its own zone. Zone names are prefixed `g6c_`.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 overlap (large assignment) | On Qwen 18 / 17 / 19 the L1 cap binds: a 656–658 KB slot means a long read-only fill and write-only drain per core. | `block_pages` = 32 (bf16) / 16 (fp32) with `buffer_depth` = 4, on LOOSE p1–p4 |
| L2 overlap (small assignment) | With `core_out_pages ≤ 2·buffer_depth`, each block is 1–4 pages, so latency dominates. | `buffer_depth` = 3–4 with `block_pages = ceil(core_out_pages/4)` on Qwen 151 / 109 |
| L3 grid synchronization | For `P ≈ grid` or `P < grid` (FLUX 23: 8 pages; Qwen 74 / 139), one page per core pays a full launch for one transfer. | `min_pages_per_core` = 4–8 (fewer cores) on FLUX 23, Qwen 74, 108 |
| L4 bank stagger (`bank_stagger`) | Contiguous ranges with `core_out_pages % num_dram_banks == 0` (e.g. 768 at 128 cores) start every core on the same bank. | Start each core's walk at `core_idx % num_banks` inside its range, with wrap, on Qwen 17 / 18 |
| L5 transaction size (`read_coalescing`, `tile_reorder`) | 2–4 KB page transactions; same-bank pages `p, p+NB, …` are contiguous in the bank. | Bank-affine page sets, m pages per NoC transaction, on Qwen 18 / 17 |
| L6 row_gather placement | Span copies force the phase-1 → 2 → 3 ordering (3 barriers per block). | Exact off-parity run copies, with phases 1 and 3 issued together (2 barriers), on Qwen 79 |
| L7 reader issue rate (`split_reader`, `trid_double_issue`) | row_gather issues 2 DRAM reads per 1-row run per tile (32 per output tile on Qwen 79), so NCRISC may be issue-bound. | Move phase-2 copies to BRISC, or trid-scoped barriers (depth 3–4) so phase 1 of block b+1 overlaps phase 3 of block b |

### Catalog evidence per knob

| Knob / decision | Catalog entry | What it set |
|-----------------|---------------|-------------|
| `buffer_depth = 2`, many transactions per barrier, never read-one/barrier | `double_buffer` | Batched reads per barrier plus a double-buffered CB. Once DRAM-bound at many cores, the block size stops mattering (WH, 64 cores at DRAM peak). This is why the L1 cap binding on large shapes is performance-neutral, and why the L1 lamp is a measurement, not a redesign. |
| Whole-page transfers where `vr = 32`, read on NoC0 / write on NoC1 | `tile_reorder`, `noc_placement` | One NoC transaction per whole tile; default reader/writer NoC assignment |
| `split_work_to_cores(..., row_wise=True)`; split over all of `out_page`, not tile rows | `noc_placement`, `width_split` | Row placement for partial grids; 1-tile-row × 1940-column outputs need a width-capable split |
| Lamps L4, L5, L7 | `bank_stagger`, `read_coalescing`, `split_reader`, `trid_double_issue` | Alternatives to measure, not Phase 0 machinery (each says when it does not help) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input DRAM → reader L1 (page_copy) | Raw page bytes of the input dtype, never interpreted | `TensorAccessor(in_args, in_addr[k], page_bytes).get_noc_addr(q, piece_off)` + `noc_async_read` on NoC0 into the `cb_moved_pages` slot at the same in-tile offset | Pieces = the valid-row prefix of `vr(p)` (*Work Distribution*). `vr(source) = vr(output page)` holds for every page_copy page: w and outer keep H, and in h page_copy only the last input's last tile row is partial. |
| Input DRAM → reader L1 (row_gather) | Raw face-row segments | Same call with a run piece's source offset; the destination is the slot (class 0) or the stage (class ≠ 0), always at the **destination** in-tile offset | Exact run rows. Never a padding row, and never a row twice (two output tile rows that share a source tile read disjoint row ranges, on one core or two). |
| Stage → slot (row_gather) | Raw bytes | Self-aimed NoC read: `noc_async_read(get_noc_addr(stage_addr), slot_addr, span_bytes)` (or `dataflow_kernel_lib::local_addr`) | ≤ 4 spans per tile (one per face that holds off-parity rows: rows `[min, max]` of that face). Garbage it copies onto on-parity rows is overwritten by phase 3. Garbage on rows `≥ vr` lands in padding. |
| Reader → writer | `cb_moved_pages`, `block_pages` pages per push | `cb_reserve_back` / `cb_push_back` ↔ `cb_wait_front` / `cb_pop_front` | Nominal counts on every push and pop |
| Writer L1 → output | Valid-row prefix pieces of each page | `TensorAccessor(out_args, out_addr, page_bytes).get_noc_addr(p, piece_off)` + `noc_async_write` on NoC1 | `noc_async_writes_flushed` before each pop, `noc_async_write_barrier` before return |

No Tensix-to-Tensix traffic, no multicast, no semaphores; no deferred regime needs any.

**TARGET axes as blocking-scheme changes**:

| TARGET value beyond Phase 0 | Class | What changes |
|-----------------------------|-------|--------------|
| `dtype: float32` | knob-turn | `page_bytes` 4096, `face_row_bytes` 64. Every run is class 0, so there is no stage (`s = 0`). |
| `dtype: bfloat8_b` | knob-turn (page_copy) | `page_bytes` 1088. Whole pages only (`vr` clamped to 32). bf8b row_gather is a deferred regime (no cell). |
| `layout: ROW_MAJOR` | knob-turn (page_copy, outer and h) | Page = stick (`buffer_aligned_page_size`); the page matrix counts sticks; no `vr`. RM `w` is a deferred regime (no cell). |
| `input_buffer: l1`, `mixed` | knob-turn | Two input accessor CT arg sets (DRAM, L1) plus a per-input class flag in the common RT table. L1 sources use congruence `A = 16`, so row_gather runs from L1 inputs are all class 0. |
| `output_buffer: l1` | knob-turn | Writer accessor CT args for L1. `L1_OP_BUDGET` minus the output's per-core share `ceil(P / num_l1_banks) · aligned_page_bytes`. |

None of these is sharding: inputs and output are always interleaved.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | One output page; a block is `block_extent ≤ block_pages` consecutive pages (row_gather: cut at output tile-row ends). |
| Grid | `device.compute_with_storage_grid_size()` (runtime). `num_cores = min(grid_x·grid_y, ceil(P / min_pages_per_core))`. |
| Split | `split_work_to_cores(grid, P, row_wise=True)` → `core_group_1` (`core_out_pages_max` each) / `core_group_2` (one fewer). Cores are walked in the same row-wise order (`ttnn.corerange_to_cores(all_cores, row_wise=True)`), and core `i` gets `[p0_i, p0_i + core_out_pages_i)`. Kernels and CBs span `all_cores`, which is one rectangle on a full grid and ≤ 2 rectangles otherwise (BB-17). |
| Per-core work | `num_blocks_this_core` blocks from `core_out_pages_i` and the cutting rule. Both kernels derive the identical block sequence from the same per-core RT args. |
| Remainder | Group-1/group-2 imbalance ≤ 1 page. The last block of a core (and, in row_gather, the last block of each tile row) is ragged: nominal push/pop, narrowed transfers. |

Geometry is alignment-aware: `ceil`, per image. With `r` = rank, `d` canonical, `Ht = ceil(H/32)`, `Wt = W/32` (W is a multiple of 32 on every TARGET cell), and `img = Π_{i<r-2} shape[i]`:

| Quantity | page_copy, `d == r-1` (w) | page_copy, `d == r-2` (h, aligned offsets) | page_copy, `d ≤ r-3` (outer) | row_gather |
|----------|---------------------------|--------------------------------------------|------------------------------|-----------|
| `pm_rows` | `img · Ht` | `img` | `Π_{i<d} shape[i]` | — |
| `seg_pages[k]` | `Wt_k` | `Ht_k · Wt` | `n_k · Π_{d<i<r-2} shape[i] · Ht · Wt` | — |
| `pm_cols` | `Σ seg_pages[k]` | same | same | — |
| source of output page `p` | `a = p / pm_cols`, `c = p % pm_cols`; `k` with `seg_start[k] ≤ c < seg_start[k] + seg_pages[k]`; `q = a · seg_pages[k] + (c − seg_start[k])` | same | same | runs (below) |
| `P` | `pm_rows · pm_cols` | same | same | `img · out_tile_rows_per_image · out_tile_cols` |

Output-page valid rows, used by both kernels in both regimes: `out_tile_rows_per_image = ceil(H_out/32)`, `out_rows_tail = H_out % 32`, `out_tile_cols = W_out/32`. `vr(p) = out_rows_tail` if `((p / out_tile_cols) % out_tile_rows_per_image) == out_tile_rows_per_image − 1 and out_rows_tail != 0`, else 32. A valid-row prefix of `v` rows is moved as: `v = 32` → `[0, page_bytes)`; `16 ≤ v < 32` → `[0, 2·face_bytes)` plus face 2 and face 3 rows `[0, v−16)`; `v < 16` → face 0 and face 1 rows `[0, v)`. Zero-length pieces (e.g. faces 2/3 at `v = 16`) are skipped. Here `face_bytes = 16 · face_row_bytes`, and face `f` starts at `f · face_bytes`.

Row_gather runs for output tile row `t` of image `a` (output rows `R ∈ [32t, 32t + vr)`): for each input `k` with `[row_offset[k], row_offset[k] + in_rows[k])` intersecting that range, the intersection is split where the source row `j = R − row_offset[k]` crosses a multiple of 32 (source tile row `ht = j/32`) and where `j % 32` or `R % 32` crosses 16. Each piece is `(k, ht, j0, r0, L)` with a constant face band on both sides, moved as 2 reads per column (left and right face). Source page = `(a · in_tile_rows[k] + ht) · out_tile_cols + w`. Class = `((j0 − r0)·face_row_bytes + (src_face − dst_face)·face_bytes) mod A`; face terms are multiples of 512, so only the row term matters.

Host-derived quantities (single source = host plan; shape quantities go in common RT args so one binary serves many shapes, knobs in CT args):

| Quantity | Where |
|----------|-------|
| `block_pages`, `buffer_depth`, `page_bytes`, `face_row_bytes`, `dram_read_align`, cut-at-tile-rows flag, accessor CT args (input class, output) | CT |
| K; per input `{base address, seg_pages[k]}` (page_copy) or `{base address, in_rows[k]}` (row_gather; `row_offset`, `in_tile_rows` derived in-kernel); `pm_cols`; `out_tile_cols`, `out_tile_rows_per_image`, `out_rows_tail`; output base address | common RT. The only RT args that change between calls are addresses. |
| `p0_i`, `core_out_pages_i` | per-core RT. No addresses here. |

**Program cache (BB-07).** A module-level dict keyed by `(grid, tuple(shapes), d, dtype, layout, tuple(input buffer types), output buffer type)` caches the plan, the per-core `RuntimeArgs`, CT args, CB descriptors and a `custom_program_hash` (blake2b of `("g6_concat", key)`). A call rebuilds only the thin `KernelDescriptor` / `ProgramDescriptor` wrappers with fresh common RT args (addresses) and sets `ProgramDescriptor.custom_program_hash` (`ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:1138-1143`). On a hit, C++ copies the new runtime args into the cached program (`tt_metal/impl/program/program_descriptors.cpp:188-213`) and requires unchanged arg counts (`validate_cached_runtime_arg_counts`). Every count is a function of the key. Pattern: `models/tt_dit/models/vae/minimax_h3/unpatchify_minimax_h3.py:13-136`. Under trace capture the output is allocated at capture; replays run no host code (BB-16).

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_moved_pages` | 0 | `page_bytes` (`buffer_page_size()` of the output: 2048 bf16) | `buffer_depth · block_pages` | Spans `out_page` / `tile_col` at `block_pages`; streams over blocks (window `buffer_depth`). Spans `in_tile_row` (whole tile per page). | input dtype (raw payload, never through DEST) | reader | writer | whole program, both regimes |
| `cb_parity_stage` | 1 | `dram_read_align` (64 B granules; never pushed or popped) | `(block_pages · page_bytes + dram_read_align) / dram_read_align`, allocated only when `row_gather and face_row_bytes < dram_read_align`, else absent | Spans `tile_col` at `block_pages` (one destination-image tile per column) and `in_tile_row` (rows at destination positions). `+dram_read_align` is the class-offset slack (base used = CB base + class). | input dtype bytes (raw) | reader | reader (private scratch, no CB handshake) | row_gather, phases 1–2 of each block |

Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `gather_pages_block` | `block_extent` pages | raw NoC + TensorAccessor (no dataflow page-copy helper exists) | — (DRAM inputs) | `cb_moved_pages`: reserve `block_pages`, fill `block_extent` valid-row prefixes, push `block_pages` | slot owned by writer |
| 2 | `derive_runs_block` | 1 output tile row | in-kernel integer walk | — | — | — |
| 3 | `stage_offparity_rows_block` | off-parity runs × `block_extent` | raw NoC + TensorAccessor | — (DRAM inputs) | `cb_parity_stage` (scratch): rows at destination positions | stage full for this block |
| 4 | `merge_stage_spans_block` | ≤ 4 face spans × `block_extent` | `dataflow_kernel_lib::local_addr` + self-aimed read | `cb_parity_stage` (scratch) | `cb_moved_pages` slot (reserved at block start, `block_pages`) | stage reusable after the barrier |
| 5 | `read_onparity_rows_block` | on-parity runs × `block_extent` | raw NoC + TensorAccessor | — (DRAM inputs) | `cb_moved_pages`: push `block_pages` | slot owned by writer |
| 6 | `store_pages_block` | `block_extent` pages | raw NoC + TensorAccessor | `cb_moved_pages`: wait `block_pages`, pop `block_pages` after flush | — (DRAM output) | slot free |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| 1, 3, 5 | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)`; `offset` is a byte offset into the page | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` (interleaved specialization ~:554); ctor :95-100 | `TensorAccessorArgs<CTA>()` from `ttnn.TensorAccessorArgs(t).get_compile_time_args()`; chained with `next_compile_time_args_offset()` (`tensor_accessor_args.h:198`) | — | — | — |
| 1, 3, 4, 5 | raw_api | `noc_async_read(src_noc_addr, dst_l1, size)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | size = piece bytes (any count; congruent offsets) | — | `cb_moved_pages` / `cb_parity_stage` | block_extent (count of calls) |
| 1, 3, 4, 5 | raw_api | `noc_async_read_barrier()` | `dataflow_api.h:1921` | once per phase per block | — | — | — |
| 4 | helper | `dataflow_kernel_lib::local_addr(addr, noc_id)` (self-aimed source); alternatively `get_noc_addr(addr)` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57`; `tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:272` | span bytes | `cb_parity_stage` | `cb_moved_pages` | — |
| 1, 5 / 6 | raw_api | `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`, `get_write_ptr` / `get_read_ptr` | `dataflow_api.h:406 / 208 / 479 / 260 / 325 / 346` | `block_pages` (nominal) | `cb_moved_pages` | `cb_moved_pages` | `block_pages` |
| 6 | raw_api | `noc_async_write(src_l1, dst_noc_addr, size)` | `dataflow_api.h:876` | piece bytes | `cb_moved_pages` | — | block_extent |
| 6 | raw_api | `noc_async_writes_flushed()` / `noc_async_write_barrier()` | `dataflow_api.h:1980 / 1951` | flushed per block; barrier once at kernel end | — | — | — |
| all | raw_api | `get_arg_val` / `get_common_arg_val` | `dataflow_api.h:148 / 168` | per-core range / common table | — | — | — |
| all | raw_api | `DeviceZoneScopedN(name)` | `tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012` (no-op :1047) | zone per block op and per wait | — | — | — |
| host | — | `ttnn.split_work_to_cores(grid, units, row_wise)` | `ttnn/cpp/ttnn-nanobind/operations/core.cpp:470-475` | `row_wise=True` | — | — | `min_pages_per_core` (via units / num_cores) |
| host | — | `KernelDescriptor.common_runtime_args`, `ProgramDescriptor.custom_program_hash` | `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:851-854, 1140-1143` | addresses only in common args | — | — | — |

**Helpers considered and rejected** (every `raw_api` row):

| Candidate | File:Line | Why it cannot express the block operation |
|-----------|-----------|-------------------------------------------|
| `compute_kernel_lib::copy` (eltwise convenience) | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:113` | Moves tiles through unpack → DEST → pack. The op must not convert any value (NaN payloads, subnormals, fp32 low bits); a compute pass is a conversion risk and an extra L1 crossing, with nothing to compute. No compute kernel exists. |
| `tilize` / `untilize` helpers | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp`, `untilize_helpers.hpp` | Input and output are both TILE, so there is no layout change. Using them would be the forbidden layout round trip. |
| `read_sticks_for_tilize` / `write_sticks_after_untilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:88 / :130` | Stick-to-tilize staging only. No page-to-page or partial-page copy. |
| `copy_tile_column` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:92` | Moves one element **column** with RISC-V stores. row_gather moves face-**row** spans, which are contiguous and NoC-copyable. |
| `set_read_state` / `read_with_state` | `local_copy_helpers_dataflow.hpp:75 / :95` | `transfer_size` is a template constant (single packet). Span and piece sizes vary per run at runtime. |
| `set_read_trid` / `async_read_barrier_with_trid` | `local_copy_helpers_dataflow.hpp:147 / :158` | Applicable; deferred to lamp L7 (Phase 0 uses one drain-all barrier per phase). |
| Any dataflow page-copy helper | `ttnn/cpp/ttnn/kernel_lib/` (dataflow headers: `local_copy_helpers_dataflow.hpp`, `l1_helpers.hpp`, `tilize_helpers_dataflow.hpp`, `reduce_helpers_dataflow.hpp`, `tile_column_dataflow.hpp`, `index_tile_dataflow.hpp`, `dfb_helpers_dataflow.hpp`) | None exists. Writing the block operation with TensorAccessor + `noc_async_read/write` closes the gap. |
| `ttnn.Mcast` / `McastArgs` | `ttnn/cpp/ttnn/kernel_lib/mcast/` | No reuse-shared operand (*Operand-reuse check*). |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Padding leaking into logical rows | Golden inputs are zero-padded, so a padding row copied into a logical row passes the default cells. LOOSE cases poison padding with 1000.0 / NaN. | No transfer ever reads an input row `≥ in_rows[k]` (row_gather runs) and no page_copy piece reaches past `vr(p)`. The acceptance test poisons padding on w (31 rows), Qwen 79 and the mixed-height h case. |
| Off-parity rows silently misplaced | On BH a bf16 face row (32 B) is half the 64 B DRAM-read congruence. A direct read of an odd-shift run lands 32 B off, with no error outside `--dev`. | Class per run computed from real addresses; stage at `CB base + class`; class 0 asserted direct only because CBs are DRAM-aligned (`program.cpp:2505-2509`). Qwen 79 has 8 of 16 inputs off-parity. |
| Span copy overwriting direct rows | Phase 2 copies whole face spans that include on-parity rows. | Strict order: phase 1 → barrier → phase 2 → barrier → phase 3 → barrier → push. Reordering is lamp L6 and requires exact-run copies. |
| Reader and writer disagree on block boundaries | Nominal pushes with ragged extents; row_gather cuts at tile-row ends but page_copy does not. | One cutting rule, from the same per-core RT args plus the CT cut flag, in both kernels; `vr(p)` from the same common RT geometry. |
| Ring overshoot on ragged blocks | `push_back(last_block_extent)` against capacity `buffer_depth · block_pages` corrupts neighbouring L1 on the second launch. | Always push and pop `block_pages`. |
| Stale program on a cache hit | The hash ignores RT values. A changed per-core arg count with the same key raises "Runtime args size mismatch"; missing addresses replay old buffers. | Key covers every count-changing quantity. Addresses only in common RT args, refreshed every call. Acceptance test calls twice with new tensors and checks values and cache-entry count. |
| Transactions in flight at kernel exit (BB-03) | `noc_async_writes_flushed` only guarantees the source L1 was read, not that DRAM was written. | Writer ends with `noc_async_write_barrier()`. Reader's last action is a read barrier (and the stage/slot copies are barriered). |
| `vr` mismatch between source and output page in page_copy | Holds only because w and outer keep H, and h page_copy only arises when offsets are tile-aligned. | The page_copy predicate is exactly `not row_gather`. Any h-concat with a misaligned non-last input goes to row_gather; the predicate edge is pinned by test. |
| `dim` gated on its literal | `dim=-1` vs `3` are the same cell. | Canonicalize before tagging; `select_plan` and the taggers take the same `d`. |
| Wide 1-tile-row outputs serialized | Qwen 79 / 160 are 1 × 1940 tiles. A tile-row split would use one core. | Flat `out_page` split, verified by the acceptance test's full-grid assertion on Qwen 79 / 160. |
| bf16 special values | NaN payloads, ±inf, −0.0 and subnormals must survive. | Pure byte moves, no compute kernel. The acceptance test covers it with special values. |
