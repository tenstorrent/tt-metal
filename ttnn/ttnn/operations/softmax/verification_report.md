# Verification Report: softmax

Target: Quasar (8 × 4 = 32 clusters, 4 NEOs + 6 user DM cores each) on ttsim (no clock, slow dispatch).
Specification: `eval/specs/softmax/spec.py` (sha256 `654aac567f0b…`). Design: `op_design.md`. Ledger: `l1_ledger.md`.
Golden results: `pipeline_logs/verifier_report.json` (copy of the CLI output for `/tmp/softmax_golden_v2`).

## Code Review

Fixed in this pass:

1. **Rejection X1 raised the wrong refusal type: 38 `xfail_wrong_mode` cells.** `validate()` raised a plain `ValueError` for an in-range `dim` that canonicalises outside {−1, −2}. tt-metal's translated tests call `dim=-3` and `dim=0` on rank 3 (`test_softmax__softmax`, `test_softmax_dtypes`). Those cells are outside SUPPORTED, and the harness recognises a refusal only by the typed `SupportRefusal`. The op now raises `UnsupportedAxisValue`, a `NotImplementedError` and one of the types X1 allows. A `dim` outside `[-ndim, ndim)` is malformed and stays a `ValueError`. Tests: `test_rejects_non_reducible_dim` now pins `SupportRefusal`; `test_rejects_out_of_range_dim` is new.
2. **The resident-fit refusals are typed.** `select_regime` / `select_blocking` raised a bare `NotImplementedError`. They now raise `SupportRefusal`, so a resource refusal reads as a deliberate refusal rather than an accident.
3. **Dead constant.** Removed `NEOS = 4` from `softmax_program_artifacts.py`; the plan's C is `THREAD_PLAN[1]` only.
4. **L1 ledger, `cb_output_tiles` row.** The row was untrue. It claimed the live set was "one DEST section's packed tiles (`out_chunk_tiles` = 8)", but at fp32 half-sync DEST `normalize_block` packs 1 tile (masked) or 2 tiles per section. The row now states the real per-push live set and why the capacity is larger (sized to the full-sync section bound, so the DEST-sync perf lamp needs no ring change; ≤ 128 KB per cluster). Tightening it is folded into Refinement 6, which changes the section size.

Reviewed, no change needed:
- **Kernel syntax and APIs.** `void kernel_main()`, `api/...` include paths, `TensorAccessor`, Metal 2.0 `ProgramSpec`, one `ttnn.generic_op` per call (one native dispatch).
- **CB sync.** Push = wait = pop on every ring:
  - `cb_input_tiles`: `bc·Ht` per block on both sides, implicit on the reader;
  - `cb_h_mask`: 1/1/1;
  - stat rings: `bc` per block;
  - `cb_output_tiles`: `n` per section on compute, one implicit entry per write on the writer.
- **Helper usage.** Raw SFPU in all three compute block operations. The design's "Helpers considered and rejected" table justifies it (reduce helper's indexed policies unsafe under a thread plan, `llk_unpack_AB_reduce_api.h:122-125`; SFPU reduce path static-asserted out; chain has no DEST-accumulating SFPU or row-broadcast DEST reuse; `minmax.hpp` does not compile), and the kernel head repeats it. The mask is built with `fill_l1_range` (helper). No multicast or semaphore in Phase 0, so the mcast-helper check does not apply.
- **Broadcasts.** `m` and `1/s` live in row 0 of one Float32 tile per column and are applied with `sfpu_*_bcast_row`; no full tile of repeated data is filled anywhere.

Deferred to refinements (architectural): none from code review. The ledger finding above is folded into Refinement 6.

### Specification rules

| Rule | Status | Evidence |
|------|--------|----------|
| R1 stable form | met | max fold + `sfpu_sub_bcast_row` before every `exp`; `large_magnitude_bfloat16` passes (PCC 0.9999993) |
| R2 canonical `dim` before the support check | met | `validate()` canonicalises first; `test_positive_dim_alias` (`dim=2`) |
| R3 float32 + 16-bit DEST refused | met | explicit refusal after the SUPPORTED gate (float32 is refused earlier by `dtype` today) |
| R4 one `default_compute_kernel_config()` | met | exported; `None` resolves through it only; lint `default_factory` passes |
| R5 no gate on fidelity/approx | met | loose case `yolo_dfl_l1_lofi` passes; `test_lofi_approx_masked` |
| R6 DRAM or L1 interleaved | met | L1 cells pass |
| R7 refusal type | met | every outside-SUPPORTED cell is `xfail_expected` (after fix 1) |
| R8 output in the input's buffer | met | `allocate_tensor_on_device(input.spec)`; `test_output_in_input_buffer[DRAM/L1]` |
| R9 `memory_axes()` gate | met | `**memory_axes(input_tensor)` in `validate()` |
| R10 device compute config | met | read by attribute, descriptor rebuilt; `test_device_compute_kernel_config` with `init_device_compute_kernel_config` |
| X1 | met | `test_rejection[X1]` passes; translated `dim=-3` cells now `xfail_expected` |

GUIDANCE is followed throughout:
- `fpu_math_fidelity` is set explicitly;
- no FPU multiply runs at HiFi4, so the ttsim fidelity stop is never met;
- the H-padding rows enter neither `m` nor `s`;
- an L1 input is read through its accessor;
- DEST is sized with `DEST_AUTO_LIMIT`.

## Registry Conformance

- **Declarations.** `INPUT_TAGGERS = AXES` is imported from the specification, with no local axis function. `SUPPORTED` equals `PHASE0` exactly (8 axes) and `EXCLUSIONS = []`. `validate()` runs in this order: `dim` canonicalisation (R2, X1) → axes (spec taggers + `memory_axes`, R9) → SUPPORTED → EXCLUSIONS → R3 → resident-regime predicate. `softmax()` calls `validate()` as its first statement. The op file declares no `INVALID`.
- **Lint.** `python3 -m eval.validate_contract softmax --phase0`: PASSED, 0 errors (6 UNRESOLVED items belong to the specification: numeric class, special values, freedoms, perf bar; not defects).
- **SUPPORTED drift.** None; there was no XPASS evidence.
- **INVALID audit (specification).** The one entry is `{dtype: bfloat8_b, layout: ROW_MAJOR}`. It concerns a single tensor, is structural (a block format with no blocks), and is the canonical entry. No cross-tensor entries. Not a norm-like op, so no weight canonicalisation applies. bfloat8_b as a whole has no path on Quasar; the harness adds it to the invalid set as target-unsupported.

## Design Conformance

- **Algorithm.** Matches. Column-resident max → exp-sum (+ `recip`) → normalize with the `exp` recomputed. Stats are Float32 tiles (row 0). The additive `NEG_BIG` mask goes on the last tile-row only.
- **Pipeline topology.** Matches. Reader (implicit-sync tagged reads + explicit-sync mask ring) → compute (3 block ops) → writer (implicit-sync tagged writes). Both DM kernels end by handing the transaction tag back.
- **Parallelization.** Fills the machine. Tile-column units go out in whole rounds of C to `split_work_to_cores(..., row_wise=True)`, which puts `[1,4,16,8400]` on 32 of 32 clusters × 4 NEOs, 8–9 columns each. The reader and writer issue one tagged transfer per entry with no per-unit barrier (implicit sync on Quasar), so neither dataflow half dribbles.
- **Thread plan.** Read off the host program: `THREAD_PLAN = (2, 4, 2)` is one constant, its counts reach every kernel as compile-time args, and the kernels size loops from them plus `get_my_thread_id()`. Rings are counted per NEO × C. The input and output rings use implicit sync, the mask ring explicit sync with no NoC transfer. Tests:
  - `test_every_neo_ran`: two calls in one session, and asserts `num_threads` = {reader 2, compute 4, writer 2};
  - `test_thread_plans`: (1,1,1), (1,2,1), (2,4,2), (4,4,2);
  - `test_device_session_neighbours`: after a device-side tilize, before `ttnn.to_layout`.

  All pass. The design states the op is compute-bound, which justifies C = 4. **Plan versus its alternatives: not measured.** This backend (plain ttsim) has no timing model, so the comparison the design names, (1,1,1) / (2,4,2) / (4,4,2), waits for `ttsim-perf` or hardware. There is no `pipeline_logs/model_cycles.jsonl` reference figure for this run.
- **Blocking-model fidelity.** Every knob is a parameter with one source:
  - `block_cols` and `in_depth` come from `select_blocking()` (CT arg `block_cols`; `in_depth` sizes the ring);
  - `OUT_DEPTH`, `OUT_CHUNK_MAX_TILES`, `IN_DEPTH_MULTI` and `THREAD_PLAN` are host constants;
  - `Ht` and `Wt` are CT args derived once from `padded_shape`.

  No CB unconditionally scales with a whole-op dimension. `cb_input_tiles` holds `block_cols · Ht` and is guarded by the resident predicate, which is the sanctioned predicate-guarded residency; the streaming and split fallbacks are the queued refinements. The knob is live: `test_block_cols_knob[1, 3]` and `test_multi_block_ragged_last_block` (blocks 2,2,2,2,1 at `in_depth = 2`) both pass.
- **Expression.** Each compute block operation sets its inits and `pack_init` once per block and holds the block's input live (`cb_input_tiles` is popped only at block end). The per-column DEST sections sit inside one block-scoped phase and add no per-unit ring handshake (stat rings take one push per column inside the block, waited once per block). The reader and writer have no per-unit barrier. No expression failure.
- **Axis accounting.** `cb_input_tiles` spans `cols` and `rows`, and its capacity scales with both. The stat rings span `cols` and collapse `rows`. `cb_output_tiles` and `cb_h_mask` are streaming or constant. Consistent.

## L1 Ledger Audit

- **Currency.** All 5 rings declared in `create_program_artifacts` have rows, and their size expressions match the code. `cb_h_mask` is always allocated (1 · C), as the row says. The `cb_output_tiles` live set was corrected (Code Review, fix 4).
- **Capacity vs live set.**
  - `cb_input_tiles`: live set = capacity, ×`in_depth` only on multi-block shapes, with the reason stated.
  - Stat rings: capacity = live set (every column's `m` and `1/s` are re-read by `normalize_block`).
  - `cb_output_tiles`: over its per-push live set, now with a stated reason; tightening it is folded into Refinement 6.
  - No collapsed extent.
- **Page format vs DEST width.** Float32 stat pages with `fp32_dest_acc_en = True` are correct (16-bit pages would truncate `m` and `1/s`). The bf16 input and output pages are the dataflow payload and the output ABI. No mismatch in either direction.
- **Disjoint lifetimes.** Every pair overlaps or is justified. `cb_col_max` and `cb_col_recip` are both live in `normalize_block`; `cb_h_mask` lives for the whole kernel; `cb_output_tiles` overlaps both stat rings.
- **Bounds and closed form.** Every symbol (`C`, `Ht`, `block_cols`, `in_depth`, `tile_in`, `has_mask`, the ring cap, the budget) has a bound and its predicate. Total per cluster: `C · tile_in · (in_depth·block_cols·Ht + OUT_DEPTH·min(block_cols·Ht, 8) + 1) + C · 4096 · 2 · block_cols`. The input term scales with `block_cols × Ht × in_depth`, the stat term with `block_cols`, and the output window is bounded. On Quasar the per-ring cap binds before the 3,963,328 B budget.
- **Data-movement budget.** Present and consistent with the code: input 1×, output 1× (plus at most C − 1 recomputed columns per op), mask and stats 0 DRAM, 0 cross-core. The cheapest-traffic split is the one implemented. `h_split_combine` is a `deferred` regime row with a positive reason (the built regime serves every shape it would serve in Phase 0), and it is queued as Refinement 2 for the shapes that now need it.
- **Block-size defaults.** Held: the units are spread over the full grid, and the coarsest block (each NEO's whole share as one block, `in_depth = 1`) is taken whenever it fits, which covers every Phase 0 INPUTS shape.

## Precision Baseline

`tests/ttnn/unit_tests/operations/softmax/test_softmax_precision_baseline.py` (bf16 input, fp32 DEST, HiFi4, `dim=-2`, seed 42, standard normal × scale; reference `torch.softmax` in float64). ULP = bfloat16 ULP distance from the reference rounded once to bf16. Ratio = got / true over elements with |true| > 1e-6.

| Shape (memory, scale) | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | ULP p99 / max | Ratio p5 / median / p95 |
|-------|-----|-------------|--------------|------------------|---------------|-------------------------|
| (1,4,16,400) L1, ×1 | 0.999997 | 1.64e-3 | 9.1e-5 | 0.00238 | 1 / 1 | 0.9970 / 0.9997 / 1.0023 |
| (1,4,16,8400) DRAM, ×1 | 0.999997 | 2.43e-3 | 9.0e-5 | 0.00237 | 1 / 1 | 0.9970 / 0.9996 / 1.0023 |
| (1,1,64,128) DRAM, ×1 | 0.999998 | 5.74e-4 | 2.2e-5 | 0.00224 | 1 / 1 | 0.9970 / 0.9997 / 1.0023 |
| (2,4,128,512) DRAM, ×1 | 0.999998 | 1.21e-3 | 1.1e-5 | 0.00219 | 1 / 1 | 0.9970 / 0.9996 / 1.0023 |
| (1,1,2048,256) DRAM, ×1 | 0.999998 | 7.6e-5 | 1.0e-6 | 0.00216 | 1 / 1 | 0.9970 / 0.9996 / 1.0023 |
| (1,4,16,400) DRAM, ×10 | 0.999999 | 2.44e-3 | 6.3e-5 | 0.00125 | 1 / 1 | 0.9971 / 0.9998 / 1.0022 |

**Assessment**: every element is within 1 bf16 ULP of the correctly rounded result (median 0 ULP). The ratio spread is centred on 1.0, with a width of about one bf16 half-ULP (±0.23 %): ordinary output rounding, no scale or structural bug. The fp32 DEST path is close to exact; the output's bf16 rounding is the only visible error. The golden numeric cases agree: `large_magnitude_bfloat16` PCC 0.9999993 / rel-RMS 0.0012, `uniform_bfloat16` 0.99998 / 0.0063, loose LoFi+approx case 0.999993 / 0.0036.

**Recommended tolerances**: defaults hold. `NUMERIC["legacy_tolerances"][(bfloat16, True)] = (0.99, 0.05)` has more than 20× margin on rel-RMS.

## Verifier CLI Summary

Run: `eval/eval_test_runner.sh eval/golden_tests/softmax/ /tmp/softmax_golden_v2` (2,191 tests, 8 sim workers, 0 failed, 0 hangs), then `python3 -m eval.verify_supported … --output …/verifier_report.json`.

- supported_pass: 72 (42 PHASE0 matrix cells + loose case + 4 bf16 numeric regressions + translated `dim=-2` bf16 tests)
- xfail_expected: 1468 (was 1430 before fix 1)
- invalid_skipped: 592
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0 (was 38 before fix 1)
- invalid_unexpected: 57. Harness-side, not an op defect: bfloat8_b cells of the translated tt-metal tests (`test_softmax_stable_with_program_cache` 32, `test_large_fill_softmax` 16, `test_softmax_bfloat8_dims` 6, `test_softmax_mix_precision` 3). The verifier counts bf8b as invalid on Quasar (target-unsupported), but the translated test file does not apply the `unbuildable on quasar` skip, so the op's correct bf8b refusal shows as xfail rather than skipped.
- no_axes_found: 2 (`test_rejection[X1]` passed; translated `test_specific_tensor_combination` skipped)
- queue coverage (`--queue op_requirements.md`): 7 values in TARGET − SUPPORTED, 0 gaps

## Recommendations

- **Queue order and why** (details in `op_requirements.md`):
  - R1 `dim=-1` + ranks 2/3 (knob-turn of the reduce direction);
  - R2 reduced-axis split + combine (scheme-change; it depends on R1 and clears the 16 wide-row cells R1 leaves failing);
  - R3 perf on the YOLOv8 `Ht = 1` columns;
  - R4 ROW_MAJOR (tilize/untilize in-kernel; the `ttnn.to_layout` wrapper is unavailable because of the one-dispatch rule);
  - R5 float32 + `fp32_dest_acc_en=False`;
  - R6 perf on multi-tile reduced extents.

  bfloat8_b is omitted: no path on Quasar.
- **Wide rows with `dim=-1`.** bf16 rows of `Wt = 128` miss the ring cap by exactly 16 B at C = 4 (128 × 2,048 × 4 = 1,048,576 vs 1,048,560). Any `Wt ≥ 128` row needs Refinement 2's split. float32 halves every resident cap (Ht or Wt ≤ 63 at C = 4).
- **Mcast on Quasar under `ProgramSpec`.** Not demonstrated in `quasar_op_examples/`. Refinement 2 must confirm it, or fall back to unicast + semaphores, before building on it.
- **Thread-plan measurement.** It waits for a timed backend (`ttsim-perf`): (1,1,1), (2,4,2) and (4,4,2) are one constant apart.
- **Stat-ring precision.** No concern at bf16 today: `m` reloads through SrcA (tf32) identically for both of its readers, so the truncation cancels; `1/s` loses ≤ 2⁻¹¹. For float32 (Refinement 5), re-check against (0.999, 0.01).
- **Harness.** The translated test file should apply the target-unsupported skip to bfloat8_b cells, so they land in `invalid_skipped` instead of `invalid_unexpected`.
