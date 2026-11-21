# Verification Report: permute

Target: Quasar (8×4 clusters) on the ttsim simulator (no clock: every perf statement below is a structural count, never a time). Specification: `eval/specs/permute/spec.py` (sha256 `aadc3d32275a…`). Specification lint (`python3 -m eval.validate_contract permute --phase0`): **PASSED**, 0 errors. Its 5 open items are the spec's own UNRESOLVED fields: `governs`, `special_values`, `freedoms`, `PERFORMANCE.bar` and `PERFORMANCE.measure`.

## Code Review

Fixed in place:
- **`normalize_dims` accepted non-integer `dims` entries silently.** `int(x)` turned `1.5` into `1`. It now uses `operator.index`, which refuses a float entry with `ValueError` as `torch.permute` does. Lists and tuples of ints (R3) and negative entries (R2) are unchanged. (`permute.py`)
- **`l1_ledger.md` named the wrong gate twice.** It said `validate()` refuses an RM row over the ring cap and a staged box that cannot fit even one thread. Both refusals are raised by the program builders (`build_page_gather()`, `plan_staged_gather()`) with `NotImplementedError`, Python-side and before dispatch. The ledger now says so. No INPUTS entry reaches either cap.

Reviewed and found conforming (no change needed):
- **Specification rules.** R1: all three regimes move bits only. The NEO transpose is a bf16 unpack → datacopy → pack at 16-bit DEST, measured bit-exact below. R2/R3/X1–X3: `normalize_dims` runs before the support check. R4: `staged_gather` dereferences only logical source coordinates and writes 0 to TILE output padding; the page regimes carry the input's padding along, so no input padding lands on a logical output element. R5/R7: `_output_spec` allocates interleaved in the input's buffer type (`test_output_placed_like_input`). R6/R8: `memory_axes()` plus `validate()`. No rule has a `structure` part.
- **One dispatch per call.** `permute` → `_permute_into` → exactly one `ttnn.generic_op` on one of three `ProgramSpec`s. There is no manipulation-op wrapper, and the Metal 1 `ProgramDescriptor` is not used.
- **Kernels.** All four use `void kernel_main()`, `api/...` includes, `TensorAccessor` and Metal 2.0 `DataflowBuffer` / `Scratchpad`. The ring protocols follow the target page:
  - `page_gather` and `tile_transpose` DM ends use implicit sync with one tagged call per entry, and hand the tag back as their last statement;
  - `staged_gather` tags every transfer with trid 0 and has no rings;
  - compute uses explicit wait/pop and reserve/push, with counts matched per window (`n` in, `n` out).
- **DM caches.** `invalidate_l2_cache_range` runs on staging after each read barrier, and `flush_l2_cache_range` on each slot before its write. `cache.h` documents that the L2 flush probes the L1 D$, so the pattern is complete. Per-thread regions are 64-B aligned and disjoint.
- **Helper usage.** The transpose uses the raw compute API, which is justified: the eltwise chain has no transpose element, and `copy_tile` refuses transposes on Quasar (`tile_move_copy.h`). The dataflow helpers do not fit (design table "Helpers considered and rejected"). There is no multicast and no broadcast in this op.

## Registry Conformance

- `INPUT_TAGGERS = AXES` is imported from the specification, and no axis function is redefined. `SUPPORTED` equals `PHASE0` axis for axis: dtype, layout, memory_config, sharding, permutation, alignment and rank. `EXCLUSIONS = []`. `validate()` checks SUPPORTED per axis (`UnsupportedAxisValue`) and then EXCLUSIONS (`ExcludedCell`), and is the first line of `permute()`. The op file declares no `INVALID`.
- No SUPPORTED auto-fix was needed: `xpass_drift` = 0.
- **INVALID audit (specification).** `INVALID = []`, which is correct here. TARGET has no block-float dtype, so the canonical bf8b + ROW_MAJOR entry has nothing to exclude. Quasar's bf8b / bf4b have no path anyway, and the target page skips them without INVALID entries. There is no weight tensor, so no canonicalization cells apply, and there is no cross-tensor coupling.

## Design Conformance

- **Algorithm, pipeline, regimes.** `select_regime` is pinned exactly as the design writes it. All three regimes are built as designed: odometer page sources, `transpose_block` windows, and box gather → shuffle → store. The regime-pinned tests cover every regime × layout × DRAM/L1.
- **Work distribution.** Each regime fills the grid: `split_work_to_cores(row_wise=True)` over output pages or boxes, and `staged_gather` engages 6 harts per cluster. Both dataflow halves are batched: implicit-sync streams on the page regimes; on `staged_gather`, one read barrier per box and one `async_writes_flushed` per 8 writes.
- **Thread plan (read off the host program).** One constant, `THREAD_PLANS`, holds the plans: `page_gather` (1,0,1), `tile_transpose` (1,1,1), `staged_gather` `GATHER_THREADS` = 6 (clamped by L1 fit). Each count reaches its kernel as a CT arg (`reader_threads`, `writer_threads`, `compute_threads`, `gather_threads`), and the loops are sized from it, not from `get_num_threads()`. The (1,·,1) page-regime plans are the design's stated choice for a data-movement bound, with its reason: zero compute per page, or one FPU datacopy per 2 KiB tile. That is acceptable per the target page's *Choosing the plan*. The required tests are present and pass:
  - every thread ran: `test_every_gather_thread_ran` (192 boxes over 32 × 6) and `test_every_ring_thread_ran`;
  - device sessions: `test_regimes_share_a_device_session`, and `test_a_tt_metal_program_follows`, which runs `to_layout` after the op and then the op again;
  - live knobs: plans (2,0,2), (3,0,3), (2,2,2), (1,4,1) and (4,4,2) all run exact.

  Measuring the plan against the design's alternatives needs a timing model. ttsim has none, and no `pipeline_logs/model_cycles.jsonl` exists for this run, so no plan comparison was taken. This is a finding for a ttsim-perf or hardware run, not a refinement.
- **Blocking-model fidelity.** Every knob is one named host constant, or is derived by `choose_gather_block()`. `PAGE_RING_DEPTH`, `TRANSPOSE_BLOCK_TILES` and `TRANSPOSE_RING_BLOCKS` size the rings; the block size reaches compute as a CT arg and is clamped there by `DEST_AUTO_LIMIT`. `OUT_SLOTS`, `GATHER_THREADS` and `gather_extent[0..3]` are passed as CT or RT args. No knob is restated as a literal. No ring scales with a whole-op dimension: `cb_pages` RM entries are one row, guarded by the ring-cap refusal. The schedule reads as blocks:
  - compute acquires, commits and packs one DEST window of `block_tiles`;
  - the page-regime DM ends have no per-unit completion boundary under implicit sync;
  - `staged_gather` has one barrier and one invalidate per box, and its per-element loops are internal to the custom shuffle block operation.
- **Deviation (advisory).** `staged_gather` issues its writes on NoC0 as well: one `Noc` from the reader config. The design lists this as a perf lamp, and Refinement 2 targets it.

## L1 Ledger Audit

- **Currency.** All four buffers (`cb_pages`, `cb_input_tiles`, `cb_output_tiles`, `scratch_gather` staging + slots) have rows, and each size expression matches the builder. `cb_pages` holds `page_ring_depth × max(R,W)` entries of `align_up(page_bytes, buffer alignment)`. The transpose rings hold `ring_blocks × block_tiles × C` tiles. The scratchpad holds `threads × (align64(staging) + out_slots × align64(slot))`. Two wording errors were fixed (Code Review).
- **Capacity vs live set.** Every capacity equals its live set: the in-flight window, the double-buffered DEST window, or the whole box for staging. Axis accounting is stated per axis. Staging spans all four box axes and scales with `e_j`; slots span only the output's in-page axes. No over- or under-sized buffer was found.
- **Page format vs DEST width.** The rings carry the input dtype (bf16) at `enable_32_bit_dest=False`, so neither the wide-page nor the truncated-page direction applies. The scratchpad carries no format metadata; it is DM only.
- **Disjoint lifetimes.** Each regime is its own program, so its buffers never coexist with another regime's. Within one program every pair is either concurrent (pipelined) or one allocation with a stated reason (staging + slots). All justifications are present.
- **Bounds.** Every symbol is bounded with its predicate: ring cap on RM `s[3]`, `budget` from the device query, and the thread clamp. The total is closed-form per regime.
- **Data-movement budget.** Present, and consistent with the code: 1 input and 1 output crossing in every regime, with no cross-core traffic. The probe below confirms that the boxes partition the input. The cheapest-traffic split, output-major, is implemented; the deferred regime rows change L1 fit or the shuffle stage, not bytes moved.
- **Block-size defaults.** The interleaved splits spread work units over the full grid, then coarsen while there are ≥ 2 boxes per worker and the box still fits (`choose_gather_block` step 4). There is no sharded path.
- **Per-core footprint.**
  - `page_gather`: `depth·max(R,W)·entry`, which scales with W in RM (134,400 B at W = 8400).
  - `tile_transpose`: `2·ring_blocks·block_tiles·C·tile` = 32 KiB.
  - `staged_gather`: `threads·(staging(e) + out_slots·slot(e)) ≤ budget`. The worst PHASE0 case is 2.1 MB in one thread; the traced calls use ≤ 6 × 30 KB.

## Precision Baseline

`tests/ttnn/unit_tests/operations/permute/test_permute_precision_baseline.py`, bf16, randn input, seed 0:

| Shape | dims | Layout | Regime | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio median [p5, p95] |
|-------|------|--------|--------|-----|-------------|--------------|------------------|---------|----------------------------------|
| (1,400,144) | (0,2,1) | TILE | tile_transpose | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (1,4,16,8400) | (0,1,3,2) | RM | staged_gather | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (2,32,64,128) | (0,2,1,3) | TILE | staged_gather | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (4,8,32,256) | (1,0,2,3) | RM | page_gather | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |
| (2,4,128,512) | (0,1,3,2) | TILE | tile_transpose | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 [1.0, 1.0] |

**Assessment**: bit-exact in every regime and layout, as R1 requires. There is no scale signature: the ratio is exactly 1.0 everywhere.
**Recommended tolerances**: exact. Every cell resolves to `exact` in `NUMERIC_RESOLVED`, and the legacy default (0.9999, 0.0) is met only by an exact copy.

## Verifier CLI Summary

Golden suite: `eval/eval_test_runner.sh eval/golden_tests/permute/` (8 sim workers). The result is 136/644 passed, 0 failed, 0 hangs. A copy of the report is at `verifier_report.json` beside this file.

- supported_pass: 133: the 132 PHASE0 cells plus the loose case `dims_as_list` (R3). The 3 / 3 rejection tests X1–X3 also pass; they carry no axes.
- xfail_expected: 508. By missing value: float32, int32 and uint32 at rank 3/4 have 132 each; rank 2 has 28 each for bf16, float32, int32 and uint32. All 508 are named by Refinement 1, and `--queue` reports 0 gaps.
- invalid_skipped: 0 (INVALID is empty)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0

Unit tests: `tests/ttnn/unit_tests/operations/permute/` 70 / 70: acceptance 13, plan / session / knob 52, precision baseline 5.

## Structural counts of the traced YOLOv8 calls (probe 2026-11-21)

| Call | Regime | Box | Boxes / clusters | Reads × payload | Writes × payload |
|------|--------|-----|------------------|-----------------|------------------|
| RM `[1,16,20,20]` `(0,2,3,1)` | staged | (1,16,1,20) | 20 / 20 | 320 × 40 B | 400 × 32 B |
| RM `[1,4,16,8400]` `(0,1,3,2)` | staged | (1,1,16,192) | 176 / 32 | 2,816 × 384 B | **33,600 × 32 B** |
| RM `[1,4,16,400]` `(0,1,3,2)` | staged | (1,1,16,32) | 52 / 32 | 832 × 64 B | 1,600 × 32 B |
| RM `[1,4,8400,1]` `(0,3,1,2)` | staged | (1,1,192,1) | 176 / 32 | **33,600 × 2 B** | 176 × 384 B |
| TILE `[1,4,400,1]` `(0,3,1,2)` | staged | (1,4,32,1) | 13 / 13 | 52 tiles | 13 tiles |
| TILE `[2,32,64,128]` `(0,2,1,3)` (not traced) | staged | (1,32,32,32) | **16 / 16** (16 of 192 harts) | 512 tiles | 512 tiles |

The two bold RM figures are each page's own transfer, the layout minimum. Those calls are transaction-bound (Refinement 2). The last row shows the whole-tile quantum capping occupancy (Refinement 4).

## Recommendations

- **Queue.** Refinement 1 (32-bit dtypes + rank 2) holds all of the remaining generality. It is followed by three perf refinements, all gated on structural counts: NoC split and write depth for the transaction-bound YOLOv8 RM gathers, the shuffle's per-element cost, and TILE gather occupancy. The 2:1 cadence collapses to "all perf" once generality is exhausted after Refinement 1.
- **No timed result exists for this op.** Every plan and knob choice (the (1,1,1) page plans, `PAGE_RING_DEPTH` 8, `GATHER_THREADS` 6, `OUT_SLOTS` 8) rests on the design's argument. A ttsim-perf or hardware run should time the design's perf lamps; that is a finding for such a run, not a queue entry here.
- **Rows at 2 B are tiny.** RM `[1,4,8400,1]` reads 33,600 input pages of 2 B each, each into a 64-B staging slot (32× L1 inflation), at one NoC transaction per page. This is the layout minimum for interleaved RM input. A bank-aware coalesced read, which reads a run of one bank's consecutive page slots in one transfer, would cut transactions about 32× at the cost of reading the padding. It is out of scope for a pass that keeps the accessor abstraction; note it for a future perf phase if Refinement 2's counts leave this shape transaction-bound.
- **`staged_gather_subtile`** (the design's deferred row) is unreachable from INPUTS even at 4 B. Refinement 4 adopts its face-granular quantum for occupancy, which also closes the 4-MiB both-leading TILE fit case for 32-bit dtypes.
- **NaN / Inf movement** is UNRESOLVED in the spec (`NUMERIC["special_values"]`). The DM regimes move bits, so it would hold there. Whether the bf16 FPU datacopy in `tile_transpose` preserves NaN payloads is unchecked, and no cell tests it.
