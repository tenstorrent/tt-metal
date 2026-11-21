# Verification Report: subtract

Target: Quasar (8×4 clusters) on the ttsim simulator. Specification: `eval/specs/subtract/spec.py` (sha256 `e6de1456…`). Phase 0 = regime R0 `tile_stream_same_shape` of `op_design.md`.

## Code Review

**Fixed:**
- **`split_tiles` duplicated the work split** (`subtract_program_artifacts.py`). It had one branch for "fewer clusters engaged than the grid" and one for the full grid. Both computed the same row-major contiguous assignment through two different code paths, so a change to one could drift from the other. I collapsed them into one path: pick the `engaged = min(grid, ceil(tiles / MIN_TILES_PER_CORE))` clusters row-wise, then deal tiles contiguously, with the first `tiles mod engaged` clusters taking one more. This is the same assignment `split_work_to_cores(grid, tiles)` gives on the full grid. Added `test_split_tiles_partitions_range`, a host test checking that every tile is covered exactly once, contiguous, with the engaged count right and balanced within 1, for 8 tile counts × `min_tiles_per_core` ∈ {1, 4, 8}. The `MIN_TILES_PER_CORE` knob (a perf lamp) is now covered for values above 1.

**Checked and conformant (nothing to change):**
- **Specification rules**:
  - R1 and R2: `EXCLUSIONS` pre-armed; the cells are refused today by the SUPPORTED check.
  - R3: the default lives only in `default_compute_kernel_config()`, and `None` resolves through it. The lint passes.
  - R4: `validate()` reads neither `math_fidelity` nor `math_approx_mode`. Both are passed to `ComputeHardwareConfig`.
  - R5: no broadcast in Phase 0.
  - R6: one pack from FP32 DEST into the output dtype.
  - R7: the op runs in DRAM and in L1.
  - R8: `UnsupportedAxisValue` / `ExcludedCell` are `NotImplementedError`.
  - R9: output in `a.memory_config()`. Tested by `test_l1_output_placement`.
  - R10: `eval.memory.memory_axes` of each operand; operands in different buffers are refused by name.
  - X1: `ValueError` from the broadcast check; the golden `test_rejection[X1]` passes.
- **Design conformance**:
  - Algorithm and pipeline match R0.
  - The flat output-tile range is split over the **full grid** (`row_wise=True`, contiguous; only clusters with work are in `target_nodes`).
  - One `sub` helper call per NEO over its whole share. Chain init and reconfig are paid once per kernel, and the DEST/ring handshake runs once per `sync_tiles` = 4.
  - Reader and writer both use implicit-sync tagged transfers with no barrier in the loop, so neither half dribbles. Each hands the tag back as its last statement.
- **Thread plan**: read off the host program, it is `THREAD_PLAN = (1, 1, 1)`, one named constant, passed as compile-time args to all three kernels and as `num_threads` on each `KernelSpec`. The design states the op as **bound by data movement** (one FPU op per tile over DRAM tensors), which is the target page's case for (1,1,1). Rings are sized per NEO (`2·sync_tiles·C`). The kernels are plan-agnostic, and the "every thread ran" tests cover (2,2,2), (1,4,1), (4,4,2) and (2,4,2) × `sync_tiles` ∈ {1, 3, 4} on a ragged 263-tile shape. The one-session test (tt-metal tilize → subtract ×3 → tt-metal untilize) passes. Comparing the plan against its alternatives needs a timing model, and this backend has none (`pipeline_logs/model_cycles.jsonl` absent). The toy measurement in `quasar_op_examples/README.md` (same program shape, DM-bound multiply: 84,562 vs 85,938 model cycles for (1,1,1) vs (4,4,2)) supports the choice.
- **Blocking-model fidelity**:
  - Every knob is defined once on the host: `THREAD_PLAN`, `SYNC_TILES` (clamped by `dest_tiles()`, the same table as `get_dest_limit()`), `RING_DEPTH_QUANTA`, `MIN_TILES_PER_CORE`. Ring depth, compute CT args and the split are all derived from them.
  - No ring scales with an op dimension.
  - The per-core block is the whole range, so the split is not half-turned. Reader/writer issue one tagged transfer per tile with no per-unit barrier, which is the protocol the target page prescribes, not an expression failure.
- **Helper usage**: `compute_kernel_lib::sub` with `PerBlockSize` policies and `Block` mapping. No dataflow helper wraps Quasar implicit-sync tagged accessor reads, so the raw calls follow the target page's protocol.
- **Correctness**:
  - Push/pop counts balance: the helper pops what the reader filled; implicit-sync rings are credited by DM0.
  - `TensorAccessor` is used throughout, `void kernel_main()`, `api/...` include paths.
  - Library gaps honoured: `fpu_math_fidelity` is always set, and bf16 rings use `UnpackToSrc`.

**Deferred to refinements:** none from review; all remaining work is generality or perf (see `op_requirements.md`).

## Registry Conformance

- `INPUT_TAGGERS = AXES`, imported from the specification. No local axis function.
- `SUPPORTED` == `PHASE0`, axis for axis (the lint confirms). Every gated axis is present: dtype, fp32_dest_acc_en, layout, memory_config, sharding, alignment, rank, broadcast, output_dtype.
- `EXCLUSIONS` = R1 and R2 cell dicts. `validate()` checks SUPPORTED per axis, then EXCLUSIONS, and is the first statement of `subtract()`.
- The op file declares no `INVALID`.
- No auto-fixes to SUPPORTED were needed (xpass_drift = 0).
- **INVALID audit (specification)**: `INVALID = []`. This is correct for this rectangle: bf8b is not in TARGET (Quasar has no block floats, family fact), the op has no weight operands, and nothing is structurally impossible. R1/R2 are lossy-but-legal refusals, which are RULES and not INVALID.
- `python3 -m eval.validate_contract subtract --phase0`: PASSED, 0 errors. The 5 unresolved items are the spec's own `UNRESOLVED` markers (governance, special values, freedoms, perf bar/measure); none blocks this phase.

## L1 Ledger Audit

- **Currency**: three rings (`cb_minuend_tiles`, `cb_subtrahend_tiles`, `cb_difference_tiles`), each `num_entries = RING_DEPTH_QUANTA · sync · C` = 8 entries of `buffer_page_size()`. That matches the ledger's `2 · sync_tiles · C`. No undeclared or stale rows.
- **Capacity vs live set**: each ring has 2× its live quantum, for double buffering (mechanism reason stated). Axis accounting is consistent: `tile_id` per cluster is streamed, and `sync_tiles` per NEO is spanned with capacity scaling with it. No collapsed or inflated extent.
- **Page format vs DEST width**: the input rings are bf16 dataflow payloads that never pass through DEST (fine). The output ring is bf16 under 32-bit DEST, justified by the output ABI and R6 (one rounding into the requested dtype). It is not an intermediate, so there is no truncation finding.
- **Disjoint lifetimes**: none. All three rings are live for the whole kernel and are concurrent (pipelined). Sharing is justified in each row.
- **Bounds / closed form**: `L1 = 6 · sync_tiles · C · page_bytes`. Every symbol is bounded with its establishing predicate. Phase 0: 49,152 B (1.2 % of the 3,963,328 B unreserved budget). Worst case over all knobs: 1.5 MiB, with each ring under the 1,048,560 B cap. Constant in the op's shape; no op dimension in any capacity.
- **Block-size defaults**: held. Interleaved: the full grid first, then the coarsest block (the whole per-core range). The quantum is the DEST width.
- **Data-movement budget**: present and consistent with the code. `a`, `b` and `out` each cross once, with 0 B cross-core. The cheapest-traffic split is the implemented one, and the ranking table shows every split moves the same bytes (no shared operand). The broadcast-regime traffic (R1 → R1b) is recorded in the design and is Refinement 3.
- Findings: none.

## Precision Baseline

`tests/ttnn/unit_tests/operations/subtract/test_subtract_precision_baseline.py`: bf16 TILE operands drawn standard normal, default config (HiFi4, fp32 DEST). The reference is float64 on the stored inputs. DRAM and L1 give identical numbers.

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP (bf16) | Elements > 0 ULP | got/true ratio p5 / median / p95 |
|-------|-----|-------------|--------------|------------------|----------------|------------------|---------------------------|
| (1,1,32,64) | 0.9999984 | 0.0156 | 0.00143 | 0.00179 | 0 | 0 | 0.9972 / 1.0000 / 1.0029 |
| (1,1,17,50) | 0.9999983 | 0.0156 | 0.00162 | 0.00186 | 0 | 0 | 0.9971 / 1.0000 / 1.0028 |
| (1,2,8400) | 0.9999984 | 0.0156 | 0.00147 | 0.00181 | 0 | 0 | 0.9971 / 1.0000 / 1.0028 |
| (2,512,1024) | 0.9999984 | 0.0156 | 0.00147 | 0.00180 | 1 | 10 / 1,048,576 | 0.9972 / 1.0000 / 1.0028 |

**Assessment**:
- The output is the correctly rounded bf16 difference on all but 10 of 1 M elements, which are 1 ULP off. That fits round-to-nearest ties resolved differently by the packer than by torch's RNE. All of the absolute and RMS error is the output's own bf16 rounding against a float64 reference.
- The ratio is centred exactly on 1.0 with a symmetric ±2⁻⁹ spread, so there is no scale or structural bug.
- Golden suite: min PCC 0.9999982, max relative RMS 0.0019, `ulp_p99` = 0 on all 78 graded cells, including the three numeric regression distributions (small ×0.001, large ×1000, uniform [0,1)).

**Recommended tolerances**: defaults hold. Grading is by the spec's resolved numeric entry: 1 ULP for bf16 → bf16, met with margin.

## Verifier CLI Summary

Golden run: `eval/eval_test_runner.sh eval/golden_tests/subtract/` (8 sim workers), then `python3 -m eval.verify_supported … --output verifier_report.json` (copy in this directory).
- supported_pass: 78 (72 Phase 0 cells + 6 numeric regression cases)
- xfail_expected: 1,512. Every one maps to a refinement: broadcast → R1, ROW_MAJOR → R2, float32 / fp32_dest_acc_en=False / output float32 → R4.
- invalid_skipped: 0 (`INVALID = []`)
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- other: `test_rejection[X1]` passed, and `test_op_loose` was skipped (empty `LOOSE_CASES`). These are the 2 "no axes" entries.
- Queue coverage (`--queue op_requirements.md`): 9 TARGET − SUPPORTED values, 0 queue gaps.

## Recommendations

- **Queue order** (hardest first, with the one true dependency): R1 broadcast (both orders) → R2 ROW_MAJOR (its broadcast path needs R1) → R3 perf (resident broadcast operand) → R4 precision bundle → R5 perf (per-regime thread plan, reconfig off).
- No `LOOSE_CASES` entry is perf-flagged, so the perf regions are chosen from structural headroom.
- **No headroom on R0 itself on this backend**: R0 already moves the minimum bytes over the full grid with implicit sync at the toy-measured floor. So no perf refinement targets the R0 cells. Two lamps from the design that need a timing model and are not queued: small-tensor grid engagement (`MIN_TILES_PER_CORE` > 1 for tensors of a few tiles) and `SYNC_TILES` ∈ {1, 4, 8}. Measure them on `ttsim-perf` or hardware before filing.
- **One-dispatch constraint**: the usual `ttnn.to_layout` partial-tick fallback for ROW_MAJOR (and any host-side expand for broadcast) is forbidden in this run. R2 names the native partial (RM × broadcast = none) as its escape hatch instead.
- **Unverified on Quasar**: the FPU in-tile broadcast (`BroadcastDim::Col/Row/Scalar`) has not been exercised by a pipeline op on this family. R1 asks for a C = 1 probe first and names the SFPU broadcast fallback.
- **Precision risk for R4**: float32 → bf16 output is graded at 0 ULP (provisional). The Phase 0 path shows 10 in 10⁶ tie cases at 1 ULP. If that pattern recurs where the grade is 0 ULP, keep the cells failing as `numerical-precision` and name them in a follow-up, not in EXCLUSIONS.
- Specification observation (not a defect): `first_operand` also tags calls where both operands need an in-tile broadcast. `INPUTS` has none, and R1 asks for those to be handled or explicitly refused.
