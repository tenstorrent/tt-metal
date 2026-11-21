# Operation Requirements: scatter

## Definition
- **Formula**: `out = input`; for every source coordinate c (ascending along `dim`): `out[c with c[dim] := index[c]] = src[c]`. Rank 2, `dim` 0: `out[index[i][j]][j] = src[i][j]`. Where several source elements are sent to one place, any one of them stands, whole (R2). Values are moved, never converted (R1).
- **PyTorch Reference**: `torch.scatter(input, dim, index, src)`; the graded reference is `reference()` in `eval/specs/scatter/spec.py` (float64, slices along `dim` in ascending order).
- **Import Path**: `from ttnn.operations.scatter import scatter`
- **Function Signature**: `scatter(input: ttnn.Tensor, dim: int, index: ttnn.Tensor, src: ttnn.Tensor) -> ttnn.Tensor` (`index` INT32 ROW_MAJOR; `src` of `input`'s dtype and layout; output allocated from `input.spec`, R7)

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Backend (this run)**: Quasar on the ttsim simulator, which has no clock. Every perf refinement is gated on a **structural count** read off the op by a test or probe (backend page, *Performance measurement*); its gain in time is unconfirmed until a timed backend (`ttsim-perf`) or hardware measures it, and no simulator timing is reported as a speedup.
> **One dispatch (mandatory for every refinement)**: each public call runs exactly one `ttnn.generic_op`. No `ttnn.to_layout` / `tilize` / `untilize` / `to_memory_config` around the kernel, not even as a partial-tick fallback: it would be a second dispatch.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED memory_config / sharding**: [DRAM] / [interleaved]
- **SUPPORTED shape-derived axes**: dim ∈ {-2}, index_form ∈ {lines}, alignment ∈ {tile_aligned, h_non_aligned}, rank ∈ {2}
- **Regime**: `resident_line_block`. A width split (face-columns, `split_work_to_cores(row_wise=True)`) across all clusters puts the whole output line of each block in L1. One DM kernel runs with thread plan (6, 0, 6): its 6 harts read, scatter and write. There is no compute kernel and no dataflow buffer.
- **Cores**: up to all 32 clusters (`min(grid, face-columns)`)
- **Compute config**: none (the op only moves bytes; DEFINITION)
- **Golden baseline**: 12 supported_pass (9 cells + 3 loose cases) + 4 rejections, 223 xfail_expected, 0 loud (`verifier_report.json`)
- **Precision**: bitwise exact on 5 shapes plus the model shape [2766, 2560] / Hs 2752 (PCC 1, 0 ULP)

### [ ] Refinement 1 — Lines along the rows (dim -1) and widths that are not tile-aligned

**Goal**: add -1 to `SUPPORTED["dim"]` (the design's deferred regime `row_lines`) and add "w_non_aligned" to `SUPPORTED["alignment"]`, for both line orientations (`dim` -2 and -1), at rank 2, bfloat16 TILE DRAM.
- **`row_lines`.** A line is a row. The unit is a face-row (16 rows × the whole W, resident). The rows are the independent axis and are split across clusters. Index pieces are whole contiguous row segments. The kernel structure is the same as `resident_line_block`, with rows and columns exchanged in the element offset and in the transfer shapes.
- **`w_non_aligned`.** For `dim` -2, each index piece is clipped to the real columns of its page (`W·4` B), so no read runs past the page. For `dim` -1, a position bound is `W` (the real line), never the padded tile width. Tile-padding columns are copied along and never receive a source element (R6).

Cells this unlocks: `32x50_dim0_16x50`, `128x100_dim-2_64x100`, `32x256_dim1_32x64`, `128x512_dim-1_128x512`, `32x17_dim1_32x9`, `17x128_dim-1_17x32`.

**Implementation skill**: /interleaved-parallel

**Verifier notes**:
- Hardest generality first: two of the three remaining regimes are new line orientations, and Refinements 2 and 4 extend whatever structure lands here. `row_lines` is a **scheme variant**, not a knob-turn. It must ship performant: it fills the grid with the face-row split, batches both dataflow halves (one read barrier per chunk, one write barrier per block), and keeps a single source for each block knob (`choose_blocking` extended with the row regime's footprint). This is the Phase 0 bar.
- The thread plan carries over: the same `SCATTER_THREADS` kernel, with the threads-ran probe (`probe_skip_thread`) and the one-session test extended to a `dim` -1 cell.
- `create_program_artifacts` checks `index_page_bytes == width·4` (verifier, Phase 0). For `dim` -1 the index's width is the source width, not `W`: generalize the check, don't delete it.
- The design's mechanism cap applies to `dim` -2 index pieces: starts stay at `face_col·64` B, which is DRAM-aligned. Only the size is clipped.

**Done when**: the six cells above pass in the golden suite with the cells and tests of Phase 0 still green; `verify_supported` loud categories stay 0; the threads-ran test passes on a `dim` -1 cell.

### [ ] Refinement 2 — Lines across images (dim -3, -4) and ranks 3 and 4

**Goal**: add 3 and 4 to `SUPPORTED["rank"]`, and add -3 and -4 to `SUPPORTED["dim"]` (the design's deferred regime `cross_image_lines`).
- **Ranks 3 and 4 with `dim` -2 and -1** are the `batch` knob-turn. Work units become (image, face-column) / (image, face-row) pairs over the same split. The `dim` -2 kernel already does this: `test_images_along_rows` passes through the program directly. The `dim` -1 regime from Refinement 1 folds images the same way.
- **`dim` -3 / -4.** A line runs across images at a fixed (tile-row, face-column). The unit is a face-column of one tile position, and the slab is that tile position across the `dim` extent. Only the page-id formula and the unit enumeration change; the element move stays the same.

Cells this unlocks: the 14 rank-3/4 INPUTS entries, e.g. `4x128x512_dim-2_4x64x512`, `4x32x64_dim2_4x32x16`, `4x32x64_dim0_2x32x64`, `2x8x32x64_dim1_2x4x32x64`, `4x2x32x64_dim0_2x2x32x64`, `2x3x32x47_dim-4_1x3x32x47`.

**Implementation skill**: /interleaved-parallel

**Verifier notes**:
- **Depends on Refinement 1.** Rank-3/4 cells with `dim` -1 need `row_lines`, and `2x100x47`, `1x1x17x50` and `2x3x32x47` need `w_non_aligned`.
- `cross_image_lines` is a scheme variant: it keeps the minimum traffic (each tensor crosses DRAM once) and adds no cross-core combine. Split the (tile-row, face-column) positions across the full grid.
- Leading dims can be small (`dim` -4 with N=2..4). The slab is then tiny, and the work units are positions, not images, so the grid still fills.

**Done when**: every rank-3 and rank-4 bfloat16 TILE DRAM cell of the golden suite passes, earlier cells stay green, loud categories 0.

### [ ] Refinement 3 — Speed up the traced call and the model shape (structural)

**Type**: perf

**Goal**: no `LOOSE_CASES` entry is perf-flagged, so the target is the op's real callers, both already in SUPPORTED: the Qwen3-VL traced call `[70, 512]` / Hs 64 (INPUTS, and the `index_all_zeros` loose case), and the model's shape `[2766, 2560]` / Hs 2752 (GUIDANCE, "2766 and 2752 rows of 2560"), bfloat16 TILE DRAM, `dim` -2. Both are bitwise exact today. The design states the op as bound by DM-hart work per element. The verifier measured these structural counts (probe in `verification_report.md`):

| Count | Traced call | Model shape | Lever (`ttnn/ttnn/operations/examples/master.md`) |
|---|---|---|---|
| L2 line-maintenance writes per cluster per block (`scoped_lock` acquire + release, every thread over the **whole** slab and both stages) | 6 × 2 × 224 lines ≈ 2.7 K | 6 × 2 × 30,368 ≈ **364 K**, against 220 K element moves | maintain each line once per cluster: deal the ranges across threads, and invalidate the read-only stages without a flush on release |
| 512 B face transfers per call (tiles split between clusters) | 512 (every tile) | **16,640** (half of all tile transfers) | `tile_reorder`, `width_split`: whole-tile (tile-column) units when `tensor_tile_cols ≥ clusters` |
| index bytes per transaction | 64 B | 320 B | coarser width units (same lever) |
| writes on NoC1 | 0 % (one kernel, reader config) | 0 % | `noc_placement`: reads NoC0 / writes NoC1 |

Pick the levers. The first two rows are the large ones on the model shape. No SUPPORTED change.

**Verifier notes**:
- **L2 maintenance (lever 1).** Dealing it across threads is correct only if an L2 range invalidate or flush also covers the other harts' private L1 D$ lines. `risc_common.h` says L2 flush probes the L1 D$. Establish the invalidate side from the tt-metal source or the simulator source (`$TTSIM_DIR`, read-only) before relying on it. ttsim may not model the caches at all, so a green test does not prove this lever.
- **Tile-column units (lever 2)** halve the units on the traced call (16 tile-columns: 16 clusters instead of 32). The table's occupancy rule applies: take whole tiles where `tensor_tile_cols ≥ clusters`, keep face-columns below that, and record both counts.
- Thread-plan alternatives (`SCATTER_THREADS` ∈ {1, 2, 4, 6}) need model cycles. That backend is not this run's: leave the plan as is.

**Done when**: a test or probe reads off each count the refinement turned, before and after, on both target shapes, and the changelog records them. On the model shape: face transfers 16,640 → 0 if lever 2 is taken, and L2 maintenance writes ≤ 1 × (slab + stage lines) per block if lever 1 is taken. The golden suite and the precision baseline stay bitwise exact. No count regresses across the guard set: one representative per regime built by then (`dim` -2, -1, -3/-4) × alignment bucket. The entry and the outcome state that the gain in time is unconfirmed on this backend.

### [ ] Refinement 4 — Row-major layout, float32 and tensors in L1

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]` for `input`, `src` and the output together (all four `dim` regimes; the design's deferred regime `row_major_lines`). Add `ttnn.float32` to `SUPPORTED["dtype"]`. Add "L1" to `SUPPORTED["memory_config"]`.
- **ROW_MAJOR** is native in-kernel row-major data access: slab pieces of row segments, element offset `row·block_cols + c`, no tile math.
- **float32** is the element type `uint32_t`, with `tile_bytes`, `face_bytes` and the RM row piece bytes all derived from the tensor's page size. No value passes through a bfloat16 buffer (R1, checked by the float32 cells).
- **L1**: tensors in L1 are read through the same accessors. The output follows `input.spec` (R7, checked by HOOKS["check"] in every L1 cell). The scratchpad budget must hold beside L1-resident tensors.

**Implementation skill**: /memory-layouts, /numeric-formats-metal, /interleaved-parallel

**Verifier notes**:
- Placed after the line regimes (difficulty tier "layouts", below the regime changes). RM must extend every `dim` regime, so it lands once over the final structure. float32 and L1 are knob-turns, bundled here rather than standing alone.
- `/numeric-formats-metal` applies only to its dtype surface: this op has no `compute_kernel_config` (DEFINITION) and no DEST.
- **Mechanism cap (design, `row_major_lines`).** A face-column of an RM row is 32 B (bf16), below the 64 B DRAM alignment. RM blocks need `block_width_faces ≥ 2`, or an even `block_width_faces·elem_bytes·16 ≥ 64`, plus 64 B-aligned starts. The split's unit must respect that.
- No `ttnn.to_layout` wrapper, not even as a partial-tick fallback: it breaks the one-dispatch rule. If RM for one regime cannot land, ship `[~]` with the remaining regime as `Refinement 4b`.

**Done when**: every cell of the golden suite passes (TARGET = SUPPORTED); loud categories 0; the float32 cells are exact (`legacy_default`: relative RMS 0); the L1 cells' output is in L1 (R7); the one-session test runs on an RM and an L1 cell.

### [ ] Refinement 5 — Per-element DM cost and overlap on the model shape (structural)

**Type**: perf

**Goal**: the design's bound is per-element work on the DM harts: an index load, a bounds compare, offset arithmetic, and a 2 B load and store. Phase 0's inner loop recomputes the full tile offset of both the source and the target element for every element (`tile_elem` ×2). It also scatters only after the whole block has landed: on the model shape, one ~1.9 MB read per cluster, then the scatter, then the write, with no overlap between them. Reduce the DM work per moved element and overlap the reads with the scatter on `[2766, 2560]` / Hs 2752 (bfloat16, both layouts once Refinement 4 lands), using the patterns in `ttnn/ttnn/operations/examples/master.md` (`double_buffer` / `trid_double_issue` for keeping reads in flight under a scatter). Candidates: hoist the source pointer (contiguous along a face row) and the per-row index pointer; precompute the per-position target row base; `src_stage_depth = 2` (stage ping-pong on trids 1/2, the kernel's static_assert relaxed) with a forced chunk so chunk k+1's reads fly during chunk k's scatter; the design's lamp "threads own disjoint face-columns" where a cluster has ≥ `SCATTER_THREADS` face-columns, which removes 2 `sync_threads` per chunk. No SUPPORTED change.

**Verifier notes**: run this after Refinement 4 so the RM path's inner loop is tuned in the same pass. The counts are read off without a clock: instructions in `scatter_chunk`'s innermost loop body, from the disassembly of the compiled kernel ELF in the JIT cache; `sync_threads` per block; and source chunks whose reads are issued before the previous chunk's scatter ends.

**Done when**: inner-loop instructions per moved element fall on the model shape, for TILE and ROW_MAJOR (before/after in the changelog). Read/scatter overlap is ≥ 1 chunk in flight on a forced multi-chunk run of the model shape, if that lever is taken. The golden suite stays green and bitwise exact. No count regresses across the guard set (one representative per regime × layout × placement). The gain in time is stated as unconfirmed on this backend.
