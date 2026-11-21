# Verification Report: reshape

Specification: `eval/specs/reshape/spec.py` (sha256 `9b9d2096…`). Target `quasar` on the ttsim simulator (32 clusters, no clock). Verifier CLI output: `verifier_report.json` (copied beside this file from the golden results directory).

## Code Review

**Fixed in this pass**
- **Tile geometry had two sources (DRY).** The relayout DM kernel (`kernels/reshape_compute.cpp`) restated `TILE_ROWS = 32` / `FACE_ROWS = 16` as literals, while the host held its own `TILE_ROWS` and validated the tensor's tile shape against it. The host now defines `TILE_ROWS`, `FACE_ROWS = TILE_ROWS // 2` and `FACE_ROWS_PER_TILE = FACES_PER_TILE × FACE_ROWS` once and passes `tile_rows` / `face_rows` to the relayout as compile-time arguments. A change of the validated tile shape now lands in one place.
- **R8 had no test the op carries.** The spec's R8 check is "a sharded input is a test the op carries itself" (the generic runner builds interleaved tensors only). I added `test_reshape_regimes.py::test_sharded_input_refused_by_name[height|width]`: an L1 height- or width-sharded input is refused with `UnsupportedAxisValue` naming `sharding`.
- **Precision baseline added**: `test_reshape_precision_baseline.py` (below).

**Reviewed, no change needed**
- Kernels use `void kernel_main()`, `api/dataflow/...` includes, `TensorAccessor`, and the Metal 2.0 `DataflowBuffer` / `Noc` API. Only a `ttnn.ProgramSpec` is built: no Metal 1 descriptor, no legacy device op.
- **Ring sync counts match.** `dfb_pages`: one entry per page on both sides. `dfb_input_window`: `window_entries` pushed and waited/popped once per block on both sides, at the nominal count, so the window never wraps. `dfb_output_tiles`: one entry per tile on both sides.
- **Sticky transaction tag.** The implicit-sync reader (regime A) and writer (both regimes) end with `noc_async_*_set_trid(0, …)`. The explicit regime-B reader tags every transfer `{.trid = 0}`. The relayout issues no NoC transfer. The session tests (`test_session_after_other_programs`, `test_a_tt_metal_program_follows`) run reshape after `ttnn.to_layout` and before `ttnn.to_layout` in one device session, for both regimes, and pass.
- **No division in DM kernels** (Quasar XIP build): every cursor is host-computed and advanced by increments; `face_row_bytes / 4` is a compile-time constant.
- **Run walk correctness**: the run table splits at input tile ends, input plane ends (so input padding rows are never copied into logical rows, R4) and the 16-row face halves of source and destination; ≤ 32 runs per output tile-row (each run ≥ 1 row).
- **Helpers.** The op has no compute phase. No `kernel_lib` dataflow helper expresses an interleaved page copy into or out of a DFB, or a byte-offset copy inside a DFB window whose NoC address is private. The design's "helpers considered and rejected" table holds. The multicast helper does not apply, because there is no cross-core traffic.
- **Kernel file name.** The middle DM kernel is named `reshape_compute.cpp` although it is a data-movement kernel. The task prompt fixed that file name, so I left it, and the file header says why.

**Deferred to refinements** (architectural, not in-pass fixes): regime A per-page transactions (Refinement 3), regime B boundary re-reads / no read-regroup overlap / per-tile output handshakes (Refinement 5), the CPU copy engine of the relayout (Refinement 6).

## Specification Rules

| Rule | Applies | Status |
|------|---------|--------|
| R1 exact, no dtype detour | yes | Every copy is a byte copy; ring formats are the tensor dtype; baseline bit-exact. The int32 check re-arms with Refinement 4. |
| R2 resolve `-1` before the support check | yes | `validate()` calls `resolve_shape` first (X2/X3 raised there); the taggers get the call's shape after the checks, as the spec's AXES expect. `changes=leading` cells with `-1` pass. |
| R3 list or tuple | yes | `shape_as_list` loose case passes. |
| R4 tile padding never becomes an element | yes | The run walk skips input padding rows at plane ends; the output padding rows are unwritten (content unspecified by DECISIONS). `TILE`+`h` cells pass. |
| R5 DRAM or L1 interleaved | yes | All `memory_config=L1` cells pass. |
| R6 refuse with `NotImplementedError` | yes | 520 `xfail_expected`, 0 `xfail_wrong_mode`. |
| R7 output placed like the input | yes (review) | Output allocated from `input_tensor.memory_config()`. A probe read `output.memory_config()`: `INTERLEAVED` / `L1` for an L1 input and `INTERLEAVED` / `DRAM` for a DRAM input. `_check` in `test_reshape_regimes.py` asserts the buffer type on every regime test. |
| R8 memory axes from `eval.memory.memory_axes`, sharded refused by name | yes (review) | `_axes()` uses `memory_axes`; the new sharded test passes. |
| X1–X3 | yes | Golden `test_rejection[X1..X3]` pass; `test_rejections_and_shape_forms` too. |

GUIDANCE: followed. Row-major and tile identity cases copy page for page, the H-changing tile case runs a program, an L1 input is read through the accessor, and the output is allocated in the input's buffer type. The view the GUIDANCE mentions is excluded by the run's one-dispatch rule (the design's `view` regime is rejected); a copy is also correct per DEFINITION.

## Registry Conformance
- `INPUT_TAGGERS = AXES` imported from the specification; no op-local axis function.
- `SUPPORTED` equals `PHASE0` axis for axis (the lint confirms); it covers every axis the kernel gates on (dtype, layout, memory_config, sharding, changes, alignment, rank).
- `EXCLUSIONS = []`. `validate()` checks the shape (X1–X3), then SUPPORTED per axis (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`), then the mechanism caps (tile shape 32×32; row-major page ≤ one NoC burst), all `NotImplementedError` subclasses.
- The public `reshape()` calls `validate()` first. The op file declares no `INVALID`.
- No SUPPORTED auto-fix was needed (no `xpass_drift`).
- **INVALID audit (specification)**: `INVALID = []`, which is right. TARGET has no block-float dtype, so the canonical bf8b + ROW_MAJOR entry has nothing to cover. The target page skips bf8b/bf4b on Quasar by itself. There is no weight tensor (no cross-tensor coupling, no norm canonicalization). `python3 -m eval.validate_contract reshape --phase0`: PASSED, 0 errors, 5 UNRESOLVED open items in the spec (governs, special values, freedoms, performance bar and measure). None of them blocks.

## Design Conformance
- **Algorithm / topology**: as designed. Two regimes, selected by the exact host function in `op_design.md`, one `ttnn.generic_op` each. Regime A is reader → `dfb_pages` → writer. Regime B is reader (explicit window) → relayout DM → writer (implicit).
- **Work distribution**: fills the machine. Regime A uses `split_work_to_cores(grid, pages, row_wise=True)` over the queried grid. Regime B uses columns first, then row groups capped at `tensor_in_tile_rows`: 32 clusters on `(1,40,40,512)` and `(1,64,8400)`, 26 on `(1,64,400)`. `test_engaged_clusters` reads the `ProgramSpec` target nodes for both regimes. Both dataflow halves are batched: regime A has no per-page barrier on either side (implicit sync), and regime B has one barrier per window block on the read side and implicit writes.
- **Thread plan**: `THREAD_PLAN = {"reader_threads": 1, "relayout_threads": 1, "writer_threads": 1}`, one named host constant passed as CT args; the kernels stride by it. The design states the op as data-movement bound (no compute kernel, so no NEO), which the target page's *Choosing the plan* sanctions. The test that every thread ran is `test_thread_plan_page_identity[R2M1W2, R3M1W1, R1M1W3]` / `test_thread_plan_regroup_writer[R1M1W2]`: a missing thread leaves pages unwritten. The session tests are present and pass. This is the ttsim backend, which has no timing model, so the plan could not be measured against alternatives. The regime-B relayout as the CPU-copy hot spot is queued as Refinement 6.
- **Blocking-model fidelity**: every block knob is a parameter with one host source. Regime A: `BLOCK_BYTES_TARGET` → `block_pages` (CT) → ring size. Regime B: `block_tile_cols`, `block_out_tile_rows`, `row_groups` (host knobs, test-overridable) → `n_in_cap`, `window_entries`, `input_window_depth`, `output_quantum` (CT), with the kernels deriving `tile_row_stride_bytes` from them. No CB scales unconditionally with a whole-op dimension. The window spans `n_in_cap × block_tile_cols` and is bounded by the fit predicate. Expression check:
  - Regime A has no per-unit synchronization.
  - Regime B reads, waits and pops the window once per block.
  - The output ring is acquired one tile at a time. It is a streaming window by design (it spans no block axis), so this is not an expression failure; batching it is lever 3 of Refinement 5.
  - The tile/row loops inside the relayout are internal to one block-scoped phase.
- **Half-turned split**: none. In regime A each core's block is `min(32 KiB of pages, core_pages)`. In regime B each core's block is its whole assignment.

## L1 Ledger Audit
- **Currency**: the three rows (`dfb_pages`, `dfb_input_window`, `dfb_output_tiles`) match the three `DataflowBufferSpec`s, and the size expressions match the code. That includes regime A's round-up of `block_pages` to `lcm(R, W)`, which the implementation notes record.
- **Capacity vs live set**:
  - `dfb_pages` = 2 × block, double-buffered: justified.
  - `dfb_input_window` exceeds its live set only for ragged blocks (nominal push/pop for contiguity): justified by a mechanism reason.
  - `dfb_output_tiles` = 2 × `output_quantum` against a live set of 1 tile being assembled plus tiles in flight. This is justified as writer batching, and Refinement 5's handshake batching will use that capacity.
  - Axis accounting is consistent: the window spans both block axes and scales with both, and the output ring streams.
  - No buffer flagged.
- **Page format vs DEST**: not applicable. There is no compute kernel and no DEST; pages hold the tensor's own bytes (R1).
- **Disjoint lifetimes**: none. The regimes are separate programs, and within regime B the window and output ring are concurrent, as the ledger justifies.
- **Bounds**: `page_bytes` (≤ `NOC_MAX_BURST_SIZE`, refused above), `n_in_cap` (≤ 34 at one output tile-row; exact host maximum) and `budget` (a run-time query) are all bounded. The totals are closed-form.
- **Block-size defaults**:
  - Regime B holds the default: its block is the whole per-core assignment, the coarsest that fits on every INPUTS entry.
  - Regime A departs from it: its block is 32 KiB of pages, not the whole core range. The design gives a bytes-in-flight reason, but no measurement backs it (ttsim cannot time). I folded the co-tune of `BLOCK_BYTES_TARGET` into Refinement 3.
- **Data-movement budget**: present and consistent with the code. A plan count confirms it: `plan.input_tiles_read()` equals the minimum on `(1,40,40,512)`, `(1,64,400)` and `(1,64,8400)`. It exceeds the minimum where row groups straddle input tile-rows: 28/16 on `(2,1,100,47)`, 128/104 on `(1,1,400,256)`, 12/8 on `(1,1,50,128)`, 224/208 on `(1,1,400,512)`. The cheapest-traffic split (`tile_col` only) is reachable as the `row_groups = 1` knob. The ledger records it with a delta and its reason (an explicit trade of occupancy, capped below 2× the minimum). Occupancy alone is not a sufficient reason, so lever 1 of Refinement 5 removes the re-reads without giving up clusters. The view (zero traffic) is rejected by the one-dispatch rule.
- **Per-core footprint**:
  - Regime A: `2 × block_pages × entry_bytes ≤ 2 × max(32 KiB, entry_bytes)`.
  - Regime B: `depth × n_in_cap × block_tile_cols × page_bytes + 2 × output_quantum × page_bytes`, at most 86 KiB on PHASE0 inputs, against a 2.9 MB budget.
- **Ledger findings**: no in-place fix needed. The two folded items above go into Refinements 3 and 5.

## Precision Baseline

`tests/ttnn/unit_tests/operations/reshape/test_reshape_precision_baseline.py`, bfloat16, seed 0, standard normal inputs:

| Shape → new shape | Layout / memory | Regime | PCC | Max Abs Err | Mean Abs Err | Relative RMS | ULP p99 | got/true ratio (p5 / median / p95) |
|---|---|---|---|---|---|---|---|---|
| (1,64,400) → (1,4,16,400) | TILE / L1 | tile_row_regroup | 1.0 | 0 | 0 | 0 | 0 | 1.0 / 1.0 / 1.0 |
| (1,1,1600,144) → (1,1600,144) | RM / DRAM | page_identity | 1.0 | 0 | 0 | 0 | 0 | 1.0 / 1.0 / 1.0 |
| (1,40,40,512) → (1,1,1600,512) | TILE / DRAM | tile_row_regroup | 1.0 | 0 | 0 | 0 | 0 | 1.0 / 1.0 / 1.0 |
| (1,64,8400) → (1,4,16,8400) | RM / DRAM | page_identity | 1.0 | 0 | 0 | 0 | 0 | 1.0 / 1.0 / 1.0 |

**Assessment**: bit-exact (`torch.equal`) on every shape, which is what a pure data-movement op must give (R1). There is no scale signature: every ratio is exactly 1.0.
**Recommended tolerances**: exact (every `NUMERIC_RESOLVED` entry is `exact`; legacy default (0.9999, 0.0) holds).

## Verifier CLI Summary
- supported_pass: 121 (120 PHASE0 cells + the `shape_as_list` loose case)
- xfail_expected: 520. These cells are covered: 32 with `changes=w` only (Refinements 1 and 2), 8 with `rank=2` only (Refinement 4), and 480 with a float32 / int32 / uint32 dtype (Refinement 4; 96 of those also need Refinements 1 and 2).
- invalid_skipped: 0 (`INVALID = []`)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 3 (the X1–X3 rejection tests, which pass)
- Golden runner: 124 / 644 passed, 0 failed, 0 hangs.
- Queue check (`--queue op_requirements.md`): 0 gaps for the 5 values of TARGET − SUPPORTED.

## Recommendations
- **Queue order**: there is no perf-flagged loose case, so the first perf target (regime A coalescing on the YOLOv8 row-major calls) is already supported, and Refinements 1 and 2 are ordered by difficulty: the two `changes=w` scheme-changes come first, and the dtype + rank widening comes last so that it extends every regime once. Refinements 1 and 2 must keep element size derived from the dtype; otherwise Refinement 4 stops being a gate change.
- **No manipulation-op fallback exists for this op**: the one-dispatch rule forbids wrapping the kernel in `ttnn.to_layout` / `ttnn.untilize` / … The queue states this so that no partial tick reaches for one.
- **Output padding content**: the regroup leaves output padding rows unwritten. That is allowed (DECISIONS, `pad_value=None`), but a downstream consumer that assumes zero padding would see stale ring bytes. If a caller ever needs zero padding, it is a spec change, not a fix.
- **Regime A small tensors** spread one page per cluster (e.g. `(1,1,4,8400)` RM: 4 pages on 4 clusters). The engaged-cluster minimum is a perf lamp, and Refinement 3's bank-affine split is the place to revisit it.
- **Unmeasured on this backend**: every perf claim in the queue is a structural count. The thread plan alternatives (two relayout harts; (2, 0, 2) for small row-major pages) need a timed backend to rank; Refinement 6 carries the relayout one.
