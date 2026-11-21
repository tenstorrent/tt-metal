# Operation Requirements: reshape

The specification is `eval/specs/reshape/spec.py`; it governs wherever this file and it differ. Target: `quasar` on the
ttsim simulator (no clock): every perf refinement below is gated on a **structural count**, not on device-ns, and its
gain in time stays unconfirmed until a timed backend or hardware measures it (`.claude/references/backends/ttsim.md`,
*Performance measurement*).

## Definition
- **Formula**: `flat(output)[k] = flat(input)[k]` for every k (row-major order, last dimension fastest; tile padding is not an element). One `-1` in `shape` takes the size the volume leaves (R2).
- **PyTorch Reference**: `torch.reshape(input_tensor, tuple(shape))` (the spec's `reference` runs it in float64; every numeric entry is `exact`).
- **Import Path**: `from ttnn.operations.reshape import reshape`
- **Function Signature**: `reshape(input_tensor: ttnn.Tensor, shape) -> ttnn.Tensor` (`shape`: list or tuple of ints, at most one `-1`). No compute config (DEFINITION: the op computes nothing). Output: the input's dtype and layout, a new interleaved buffer of the input's buffer type (R7).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **One dispatch (mandatory, every refinement)**: each call stays exactly one `ttnn.generic_op` of one `ttnn.ProgramSpec`. No refinement may sequence ttnn ops (`ttnn.to_layout`, `ttnn.tilize`, `ttnn.untilize`, `ttnn.typecast`, …) around the kernel: the run's integrity rules forbid it, so the usual manipulation-op partial-tick escape hatch does **not** exist for this op.
> **Session rule (every refinement that adds or changes a kernel)**: before ticking, run the new path once after another program and once before `ttnn.to_layout(output, …)` in one device session (target page, *Ring protocols*), as `test_reshape_regimes.py::test_session_after_other_programs` / `test_a_tt_metal_program_follows` do for the Phase 0 regimes.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE, ROW_MAJOR]
- **SUPPORTED memory_config / sharding**: [DRAM, L1] / [interleaved] (a sharded input is refused by name, R8)
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}; changes ∈ {leading, h}; rank ∈ {1, 3, 4}
- **Regimes**: `page_identity` (row-major with W kept, or tile layout when the tiles hold the same rows) — reader → `dfb_pages` → writer, implicit sync; `tile_row_regroup` (tile layout, H changes, not both H multiples of 32) — reader (explicit, whole-tile window) → relayout DM kernel (face-row runs) → writer (implicit).
- **Cores**: full grid (32 clusters on quasar) in both regimes; thread plan (R, relayout, W) = (1, 1, 1), no NEO (no compute kernel; data-movement bound).
- **Compute config**: none (the op computes nothing).
- **Golden baseline**: 124 / 644 passed (121 `supported_pass` incl. the `shape_as_list` loose case, 3 rejection tests), 520 `xfail_expected`, 0 `supported_fail` / `xpass_drift` / `xfail_wrong_mode` (per verifier CLI).
- **Precision**: bit-exact on every measured shape (PCC 1.0, max abs err 0, relative RMS 0, ULP p99 0).

**Queue shape.** `TARGET − SUPPORTED` = `changes` {w}, `dtype` {float32, int32, uint32}, `rank` {2}. No loose case is perf-flagged, so the perf targets below are free-selected regions that are already in SUPPORTED: the anchor rule adds nothing to Refinements 1 and 2. The generality refinements are therefore ordered hardest first. The two `changes = w` scheme-changes come first, and the dtype + rank gate widening comes last, so that it extends every regime once.

### [ ] Refinement 1 — `changes = w` in row-major layout (regime `w_regroup_rm`)

**Goal**: add `"w"` to `SUPPORTED["changes"]` by building the deferred regime `w_regroup_rm` of `op_design.md` (Regimes table): row-major, `W_in != W_out`. Output row j is elements `[j·W_out, (j+1)·W_out)` of the flat stream, cut at byte offsets that are neither page- nor 16 B-aligned. Topology: reader → input-page window ring (explicit sync, whole input pages) → relayout DM kernel copying byte runs into output pages → writer (implicit sync), i.e. the `tile_row_regroup` topology with a byte-run walker in place of the face-row walker. Until Refinement 2 lands, add `{"changes": "w", "layout": ttnn.TILE_LAYOUT}` to `EXCLUSIONS` (a structural capability gap: the tile path does not exist yet). Cells moved: the 16 bf16 row-major `changes=w` cells (8 INPUTS entries × DRAM/L1) go from `xfail_expected` to `supported_pass`.

**Verifier notes**: scheme-change, so it stands alone. The new topology is the work. Classify the axes the way Phase 0 does: output rows are independent, and adjacent output-row blocks share at most one boundary input page. Spread the output pages over the full grid from the first build, with a block factor and window depth as host parameters (a single source; no literal loop bounds; no ring sized to a whole-op dimension), and update `l1_ledger.md` and the data-movement budget for the new ring. Derive every byte count from `buffer_page_size()` and the element size of the tensor's dtype, never a bf16 literal, so that Refinement 4 is only a gate change. The relayout kernel must not divide (Quasar XIP DM build): the host computes the start cursors. Rows that are not 16 B-aligned at both ends need a sub-word copy (byte or halfword) on the CPU path, or an aligned word copy with masked edges. Keep the row-major page cap (`NOC_MAX_BURST_SIZE`) for both W's. Add regime-pinned tests (non-aligned W both sides; `(1,32,128)→(4096,)`; `(2048,)→(32,64)`; a forced multi-block run) and the session tests above.

**Done when**: every bf16 row-major `changes=w` cell of the golden suite passes, every Phase 0 cell still passes, `verify_supported` shows 0 `supported_fail` / `xpass_drift` / `xfail_wrong_mode`, and the regime-pinned and session tests pass.

### [ ] Refinement 2 — `changes = w` in tile layout (regime `w_regroup_tile`)

**Goal**: complete `SUPPORTED["changes"]` value `w` for `ttnn.TILE_LAYOUT` by building the deferred regime `w_regroup_tile` (`op_design.md` Regimes table), and delete the `{"changes": "w", "layout": TILE}` `EXCLUSIONS` entry Refinement 1 added. Elements move inside face-rows at element granularity (2 or 4 B), so runs split additionally at the 16-element face-column boundary of source and destination, and the output tiles are padded for the new shape (R4: input padding never becomes an output element). Cells moved: the 16 bf16 tile-layout `changes=w` cells go from `xfail_expected` (excluded) to `supported_pass`.

**Verifier notes**: a scheme-change of its own: the same reader → window → relayout → writer topology as `tile_row_regroup`, with a window spanning the input tile-rows *and* tile-columns that feed a block, and an element-run walker. It comes after Refinement 1, not before, for narrative only (the RM byte-run walker is the simpler instance of the same flat-stream walk); neither strictly needs the other. Rejected in the design and still rejected here: a NEO `untilize → DM regroup → tilize` pipeline inside the one program is allowed by the one-dispatch rule, but it costs two compute passes and still needs the DM regroup. Re-propose it only with a structural count showing it moves fewer bytes. Same knob, DRY, ledger, no-divide and element-size rules as Refinement 1. Regime-pinned tests: `(1,1,32,64)→(1,1,64,32)`, `(1,1,17,50)→(1,1,25,34)` (non-aligned in and out), `(4,128,47)→(4,47,128)`, `(1,1,32,8192)→(1,8,32,1024)` (wide: the window must fit; the fit search shrinks the block, never refuses an INPUTS entry), plus the session tests.

**Done when**: every bf16 `changes=w` cell (both layouts, DRAM and L1) passes, `EXCLUSIONS` is empty again, every earlier cell still passes, and the loud categories are 0.

### [ ] Refinement 3 — Coalesce regime A's page transfers (row-major traced calls)

**Type**: perf

**Goal**: `page_identity` moves one page per NoC transaction on each side. For the YOLOv8 row-major calls, which are most of the traced calls, that is a 288 B transaction per row: `(1,1,1600,144)→(1,1600,144)` RM DRAM issues 1,600 reads + 1,600 writes, and `(1,1,400,144)→(1,400,144)` issues 400 + 400. Input and output have identical per-bank layouts (same page size, page count and buffer type, both interleaved from bank 0), so a page's within-bank offset is the same in both buffers. A core can therefore move a run of k consecutive same-bank pages (contiguous at the aligned page stride) as one transfer each way. This is the design's "Transaction size (regime A)" perf lamp; levers in `ttnn/ttnn/operations/examples/master.md`: `read_coalescing` (contiguity-gated wide reads), `double_buffer` (bytes in flight), `noc_placement` (keep `row_wise=True`). Co-tune `BLOCK_BYTES_TARGET` (currently 32 KiB, a design choice not backed by measurement) with the coalesced entry size. The work split may become bank-affine. No SUPPORTED change.

**Done when**: the structural count **NoC transactions per side, summed over cores** (read off the built plan by a test) falls from 1,600 to ≤ 200 on `(1,1,1600,144)→(1,1600,144)` RM DRAM, and from 64 to ≤ 64 (no regression) on the wide-page `(1,64,8400)→(1,4,16,8400)` RM DRAM. Results stay bit-exact. The golden suite is green. There is no regression on the config-spanning guard set (one representative per kernel path × layout × placement: RM DRAM, RM L1 with a 94 B page `(4,128,47)`, TILE DRAM `(1,1,2048,256)→(1,4,512,256)`, TILE L1 rank-1 `(512,)`). The session tests pass. The changelog records before/after counts and states that the time gain is unconfirmed on ttsim.

### [ ] Refinement 4 — float32, int32, uint32 and rank 2

**Goal**: add `ttnn.float32`, `ttnn.int32`, `ttnn.uint32` to `SUPPORTED["dtype"]` and `2` to `SUPPORTED["rank"]`, in every regime that exists by then (`page_identity`, `tile_row_regroup`, `w_regroup_rm`, `w_regroup_tile`). Cells moved: the 8 bf16 rank-2 cells and the 480 float32 / int32 / uint32 cells (TARGET × INPUTS) go from `xfail_expected` to `supported_pass`.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: the skill's dtype half applies. Its `compute_kernel_config` / DEST / intermediate-CB half does not: the op has no compute kernel and DEFINITION says it takes no compute config, so do not expose one. Rule R1 is the binding constraint: no value passes through another dtype, so ring `data_format_metadata` stays the tensor's dtype and every copy is a byte copy. The pass condition is zero kernel changes: face-row and page bytes already derive from `buffer_page_size()` (fp32 / int32 tile = 4,096 B, face row 64 B; widest RM page 8,400 × 4 = 33,600 B < `NOC_MAX_BURST_SIZE`), provided Refinements 1 and 2 kept element size derived. Rank 2 is a gate change only (`H = shape[-2]` already covers it). It is last because it is the cheapest tier: a dtype extends the structure the two scheme-changes build, so it is not landed before them. If a cell fails out of the box, fix it. Do not exclude it: an exact op has no precision lever to defer to.

**Done when**: every golden cell of TARGET × INPUTS passes (`xfail_expected` = 0), the loud categories are 0, and an int32 cell with values outside the bfloat16-exact range (rule R1's check) passes bit-exact.

### [ ] Refinement 5 — Regime B: no boundary re-reads, read/regroup overlap, batched output handshakes

**Type**: perf

**Goal**: three cheap levers on `tile_row_regroup`, the only traced call tt-metal runs a program for (`(1,64,400)→(1,4,16,400)` TILE L1), measured on the regroup shapes of INPUTS. No SUPPORTED change.
1. **Boundary re-reads** (perf lamp "Grid synchronization / duplication"). The balanced output-tile-row split re-reads the input tile-row that straddles each row-group boundary. Input tiles read / minimum is currently 28 / 16 = 1.75 on `(2,1,100,47)→(2,4,25,47)`, 128 / 104 = 1.23 on `(1,1,400,256)→(1,20,20,256)`, 224 / 208 on `(1,1,400,512)→(1,20,20,512)` and 12 / 8 on `(1,1,50,128)→(1,5,10,128)`. Choose row-group boundaries that are aware of input tile-row ends, or trade row groups for column blocks, so that the engaged-cluster count does not drop.
2. **Overlap** (perf lamp "Overlap"). Every regroup cell runs one block per core today, so the whole window is read before any regroup starts. Where a core holds ≥ 2 tile columns (e.g. `(1,64,8400)→(1,4,16,8400)`: 8–9 per core), two column blocks with window depth 2 cost no traffic.
3. **Output handshakes**. The relayout does `reserve_back(1)` / `push_back(1)` per output tile. Batch them per output tile-row: `wb` tiles, or `output_quantum`, keeping the ring contiguous.

Catalog: `ttnn/ttnn/operations/examples/master.md` (`double_buffer`, `width_split`, `tile_reorder`).

**Done when**: (1) `plan.input_tiles_read()` equals the minimum (`tensor_in_tile_rows × tensor_w_tiles`) on every regroup shape of INPUTS, or exceeds it only where a test-recorded reason shows that the grid cannot otherwise be filled; (2) blocks per core ≥ 2 with window depth 2 on `(1,64,8400)→(1,4,16,8400)`; (3) output-ring reserve/push pairs per output tile ≤ 1 / min(`wb`, `output_quantum`). All three are read off the plan by a test. Results stay bit-exact. The golden suite is green. There is no regression on the guard set (one regroup shape per placement and alignment: `(1,64,400)` L1, `(1,40,40,512)` DRAM, `(2,1,100,47)` L1, `(2,3,5,64)` DRAM), and the session tests pass. The changelog states that the time gain is unconfirmed on ttsim.

### [ ] Refinement 6 — The relayout copy engine (regimes B and both W regimes)

**Type**: perf

**Goal**: the relayout DM kernel copies every output byte through the uncached CPU alias, one 32-bit load and one store at a time, and it is the per-byte hot path of `tile_row_regroup`, `w_regroup_rm` and `w_regroup_tile`. Levers, from the design's "Relayout copy engine" and "Thread plan" perf lamps and the `split_reader` entry of `ttnn/ttnn/operations/examples/master.md`: (a) self-aimed NoC reads (`dataflow_kernel_lib::local_addr`, `local_copy_helpers_dataflow.hpp`) from the window into the output entry. These need a public NoC address of the window's read pointer, so check `dataflow_buffer.h` at apply time; it was private at Phase 0. (b) Two relayout harts splitting a block's output tile-rows, within R + W + relayout ≤ 6 user DM harts. The window ring's STRIDED dealing means each hart needs its own window, or the window must move to a non-strided access pattern. This is the ⭐⭐⭐ restructure of the queue, a whole phase on its own. No SUPPORTED change.

**Done when**: on `(1,40,40,512)→(1,1,1600,512)` TILE DRAM and on one W-regime shape, the structural count **CPU-copied bytes per output tile** falls to 0 (lever a), or **relayout harts at work** rises from 1 to 2 with each moving half the output tiles, shown by a test where a missing hart leaves tiles unwritten (lever b). Results stay bit-exact. The golden suite is green. There is no regression on Refinement 5's guard set. The session tests pass. The changelog states that the time gain is unconfirmed on ttsim.
