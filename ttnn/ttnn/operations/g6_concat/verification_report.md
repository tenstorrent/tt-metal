# Verification Report: g6_concat

Blackhole p150b, 11 × 10 = 110-core grid (`TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=10,9`), 2026-11-27.
Artifacts: `pipeline_logs/phases/verifier/` (`verifier_report.json`, `test_results.json`, `golden_results.txt`).

## Code Review

The implementation matches the design on every binding dimension. Review fixes applied in place:

| # | Finding | Fix |
|---|---------|-----|
| 1 | **Program-cache key omitted the blocking knobs (BB-07).** The key carried `MAX_CORES` but not `BLOCK_BYTES_CAP`, `OUTPUT_PAGES_DEPTH` or `TRIM_ROWS_MAX`. Each of those changes the program (CB sizes, `block_pages` common RT arg, trim CT arg). A perf sweep that turns a knob in one process would have replayed the stale program. | `g6_concat_program_descriptor.blocking_knobs()` returns every knob, and `_cache_key` includes it. The knobs stay defined once, in the descriptor module. |
| 2 | Kernel comments pointed at host symbols that do not exist (`PAGE_READER_COMMON_ARGS`, `GATHER_READER_COMMON_ARGS`, `WRITER_COMMON_ARGS`). The arg layout is duplicated across languages, so the cross-reference has to resolve. | Comments now name `_reader_common_args()` / `PAGE_READER_COMMON_FIRST_INPUT`, `GATHER_READER_COMMON_FIRST_INPUT` and `_writer_common_args()` / `WRITER_COMMON_OUTPUT_BASE`. |
| 3 | **The test directory did not run.** `test_g6_concat_bench.py` imported a missing `select_plan`. `test_g6_concat_perf.py` read plan keys that do not exist (`total_pages`, `l1_total_bytes`). `test_g6_concat_core_sweep.py` and `test_g6_concat_perf_probe.py` duplicated cases of `test_g6_concat_device_perf.py`. `tools/zones.py` was empty. | All five removed. `test_g6_concat_device_perf.py` is now the single device-time harness: p1–p4, Qwen 79, Qwen 25, Qwen 80 and the Qwen 79 core-scaling sweep. A case outside SUPPORTED skips, so p1 (fp32) runs as soon as R1 lands. |

Checked and found conformant, no change:
- **Algorithm and regimes.** `page_copy` and `row_gather` follow the design's address maps. `select_regime` is exact (only the last input may be unaligned in `page_copy`). `row_gather` reads only logical rows and splits pieces at row 16 on both sides. A misaligned piece goes through the staging mirror at `dst_off + shift`, and the place phase builds its local source with `local_addr`.
- **Work distribution (performance conformance).** `split_work_to_cores(grid, P_out, row_wise=True)` gives contiguous `out_page` ranges on `min(P_out, 110)` cores, at most 2 rectangles (BB-17). Every golden cell with ≥ 110 pages ran on 110 cores. Reads issue on NOC0 and writes on NOC1.
- **Blocking-model fidelity.** `block_pages` is computed once on the host from `BLOCK_BYTES_CAP` and the most-loaded core's count. It reaches both kernels as one common RT arg, and the CB capacity is `OUTPUT_PAGES_DEPTH · block_pages · page_bytes`. No CB scales with a whole-op dimension. No block value is restated as a literal.
- **Expression.** Reader: one reserve, all reads of the block, one barrier, one push. Writer: one wait, all writes of the block, one flush, one pop. Neither side synchronizes per page. A ragged last block keeps the nominal push/pop counts.
- **Correctness.** Push and wait counts match (`block_pages` on both sides). The kernels use `TensorAccessor`, `void kernel_main()` and the `api/dataflow/...` includes. The writer ends with `noc_async_write_barrier` and the reader with a read barrier (BB-03).
- **Helpers.** The design's rejections hold: the compute copy helpers round-trip through DEST, which the Precision rule forbids, and `set_read_state` needs a compile-time size. `local_addr` is used where it applies. Nothing is multicast because there is no shared operand.
- **Profiler zones and plan report (BB-21/BB-08).** All six zones are present, and `g6_concat_plan` reports the regime, cores, block, CB bytes, L1 total and per-core role. The profiler build compiles and runs (used for the p1 measurement below).
- The page reader builds its `TensorAccessor` per page rather than rebasing it when the input changes. That is a code-shape choice: an interleaved accessor is two words, and the loop is DRAM-bound. Not a finding.

### Prompt rules (`eval/prompts/g6_concat.txt` `## Rules`)

Every rule is a MUST. Status of each rule whose condition applies in Phase 0:
- **Read once, write once, one program, no staging tensor.** Holds. The core-local L1→L1 hop of `row_gather` is the allowed "inside the kernels' own L1" case.
- **Bit-identical output.** Holds. The acceptance special-values tests compare against `torch.cat` bit for bit and pass. The golden special-values cell fails because of a harness oracle defect (see the Verifier CLI Summary).
- **`h_non_aligned` handled natively, padding never reaches logical rows.** Holds; the poisoned-padding tests (1000.0 and NaN) pass.
- **h join with unaligned offsets inside one program, not refused.** Holds (Qwen 79/89/143).
- **dim canonicalized before the gate.** Holds.
- **Core count is the blocking decision.** Holds.
- **BB-01…BB-22.** Hold. BB-07 was fixed above. BB-16 trace replay passes outside the eval runner's perf capture.

No soft (prefer/consider/avoid) rules exist, so there are no advisories.

## Registry Conformance

- `INPUT_TAGGERS` (6 axes, all `(inputs, axes)`), `SUPPORTED`, `EXCLUSIONS = []` and `validate()` are present and wired correctly. `validate()` runs the structural `ValueError`/`RuntimeError` checks, then the SUPPORTED per-axis gate (`UnsupportedAxisValue`), then EXCLUSIONS (`ExcludedCell`). `g6_concat()` calls `validate()` first. The op file does **not** declare `INVALID`.
- Taggers canonicalize `dim` before deriving `concat_dim`. The scenario dict is built exactly as `axes.py:scenario_of_call` builds it.
- No XPASS drift, so SUPPORTED was not auto-fixed.
- **INVALID audit** (`feature_spec.py`). There is one entry, `{bfloat8_b, ROW_MAJOR}`: the canonical one, and a single-tensor coupling, since all inputs and the output share dtype and layout. It passes all three sanity rules. The op is not norm-like, so the no-weight rule does not apply. No issue.
- `tag_alignment` can return `"w_non_aligned"`, which is not in TARGET (no production cell, BB-22). `validate()` refuses it with `UnsupportedAxisValue`. Documented omission: no refinement.

## L1 Ledger Audit

- **Currency.** `cb_output_pages` (`2 · block_pages · page_bytes`) and `cb_gather_staging` (`(block_pages + 1) · page_bytes`, only when `row_gather` ∧ a DRAM input ∧ `rb mod A_dram ≠ 0`) match `build_plan` / `build_program` exactly. The Qwen 79 row (`n1 = 18`, 112,640 B) matches the plan.
- **Capacity vs live set.** `cb_output_pages` holds two blocks, which is the double buffer; the reason is stated. `cb_gather_staging` uses page-sized mirror slots, of which only the misaligned pieces are written (about 25% of a slot on Qwen 79). The stated reason is that the mirror at `dst_off + shift` keeps one offset formula for both phases, and the slots are bounded by `block_pages + 1 ≤ 65` pages. Accepted. In the under direction, both rows span `out_page` with capacity scaling in `block_pages`; no collapsed extent.
- **Page format vs DEST.** No compute kernel and no DEST. Both CBs carry dataflow-only payloads in the tensor's dtype. Clean.
- **Disjoint lifetime.** The two CBs are live concurrently (place reads staging while writing output slots), and the ledger says so. Clean.
- **Bounds and closed form.** Every symbol is bounded; `out_l1_per_bank` is already listed for the L1-output refinement. The total is closed-form.
- **Block-size defaults.** Occupancy holds (full grid). The coarsest-block step departs from the default. `block_pages = min(n1, 128 KiB / page_bytes)` caps p1 at 32 fp32 pages, while ~164 pages at depth 2 would fit the 1,347,776 B budget. The cap is justified by the catalog's `double_buffer` / `trid_double_issue` plateau, which was measured on bf16 single-core copies, not on this op. On bf16 p2 the cap costs ≤ 2.5% (p2 runs at 1.025× the DRAM roofline). On fp32 p1 it is a plausible cause of the 1.081× gap. **Folded into Refinement 3** (the `BLOCK_BYTES_CAP` sweep on p1).
- **Per-core footprint.** `L1_op = OUTPUT_PAGES_DEPTH · block_pages · page_bytes + [staging] · (block_pages + 1) · page_bytes (+ out_l1_per_bank after R1)`. The first two terms scale with `block_pages` and depth; the third scales with `P_out`. Worst case 395,264 B (bf16 `row_gather` at the cap), well under 1,347,776 B.
- **Data-movement budget.** Present and consistent: 1× DRAM per input and 1× for the output, no cross-core traffic, and only logical rows in `row_gather` plus the core-local hop. The cheapest-traffic split (output-page ranges) is the one implemented.

## Precision Baseline

`tests/ttnn/unit_tests/operations/g6_concat/test_g6_concat_precision_baseline.py`, bfloat16 TILE DRAM:

| Shape | Regime | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err | Max ULP | got/true ratio p5 / p50 / p95 |
|-------|--------|-----|-------------|--------------|------------------|---------|-------------------------------|
| 2 × (1,1,1,128), dim −1 | page_copy, trimmed v=1 | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |
| (1,6,31,64)+(1,6,31,192), dim −1 | page_copy, 31-row images | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |
| 16 × (1,1,1,62080), dim 2 | row_gather | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |
| 8 × (1,4096,1536), dim 0 | page_copy, multi-block | 1.0 | 0.0 | 0.0 | 0.0 | 0 | 1.0 / 1.0 / 1.0 |

**Assessment:** bit-exact. The ratio spread is exactly 1, so there is no scale bug and no noise. The special-value acceptance tests (NaN payload, ±inf, −0.0, ±subnormal) also pass against `torch.cat`.
**Recommended tolerances:** exact for every dtype. The op moves bytes and nothing passes through DEST. bfloat8_b is bit-exact too, because every bfloat8_b production cell joins tile-aligned widths and groups move whole.

## Verifier CLI Summary

Golden run `pipeline_logs/phases/verifier/`: 176 tests = 156 INPUTS cells + 10 LOOSE_CASES + 10 regression tests.

- supported_pass: **79**
- xfail_expected: **86** (83 INPUTS + 3 loose: p1 fp32, fp32 special values, L1-input poisoned padding)
- invalid_skipped: 0 (no INPUTS cell is bf8_b + ROW_MAJOR)
- supported_fail: **1**. This is a **harness oracle defect**, not an op failure and not an OOM/precision cell, so it is escalated rather than queued.
- xpass_drift: 0
- xfail_wrong_mode: 0
- no_axes_found: 10 (regression tests). 7 pass under the eval runner and 3 fail; all 3 are harness defects (below). 9 of 10 pass under plain `run_safe_pytest.sh`.

**Escalation: two golden-harness defects. Neither is fixable in the op, and neither may be "fixed" there.**

1. **The bf16 NaN-payload oracle is wrong.** This hits `test_op_loose[1x6x31x64+1x6x31x192_d-1 …]` (special values) and `test_regression.py::test_special_values`. `helpers.pytorch_g6_concat` round-trips through float32: `torch.cat([t.to(float32)…]).to(bfloat16)`. torch's float32→bfloat16 conversion canonicalizes NaN, so bf16 `0x7FC1` becomes `0xFFFF`. On device (probe `probes/probe_010.py`) the op returns `0x7FC1` at every special position. That is bit-identical to `torch.cat(xs, -1)` and differs from the oracle only there (4 elements). The spec requires NaN payloads to be preserved, so the op is right. Fix belongs in `eval/golden_tests/g6_concat/helpers.py`: build the expected tensor with `torch.cat` directly on the input dtype (lossless for bf16/fp32), or concatenate bit views. fp32 is unaffected because there `.to(float32)` is a no-op. **Do not** change the op to canonicalize NaN, and **do not** EXCLUDE the cell: it is the Qwen 109 production cell, which passes on random data.
2. **The eval-runner perf capture breaks under trace.** This hits `test_trace_replay[qwen79]` and `[qwen151]` under `eval_test_runner.sh` only. `eval/profiling.py:301` unpacks 3 values while `_COLLECTOR.capture` returns a 2-tuple on its early-exit paths (`return None, None`). The device-close timeout follows. Stock `ttnn.add` and `ttnn.concat` in the same capture/replay test reproduce both errors, so the defect is in the harness. Both tests pass under plain `run_safe_pytest.sh`, including replay on new input values (BB-16).

I did not edit the harness (`eval/` is the `tt_ops_code_gen` submodule). Both items need the golden-suite owner.

## Performance (device kernel ns, 110 cores)

| Case | g6_concat | Bytes moved | Achieved | Lead roofline (460.8 GB/s) | Ratio |
|------|-----------|-------------|----------|----------------------------|-------|
| p2 Qwen 17 bf16 outer | 895.6 µs | 402.65 MB | 449.6 GB/s | 873.8 µs | 1.025× |
| p3 Qwen 19 bf16 outer | 897.0 µs | 402.65 MB | 448.9 GB/s | 873.8 µs | 1.027× |
| p4 Qwen 23 bf16 outer | 452.2 µs | 201.33 MB | 445.2 GB/s | 436.9 µs | 1.035× |
| **p1 Qwen 18 fp32 outer** (`validate()` bypassed in a measurement-only test, bit-exact) | **1889 µs** | 805.31 MB | 426.3 GB/s | 1747.6 µs | **1.081×** |
| p1, stock `ttnn.concat` | 1950 µs | | | | 1.116× |
| Qwen 79 row_gather | 125.6 µs | ~4.0 MB | ~32 GB/s | ~8.6 µs | ~14.6× |

p1 already meets its LOOSE_CASES goal (< 1948.52 µs and ≤ 1.35× roofline) on today's kernels: it is 3.1% faster than stock. The residual 8% gap to roofline is Refinement 3.

## Recommendations

- **Queue shape.** R1 (dtype + interleaved L1 placement) → R2 (ROW_MAJOR) → R3 (perf, p1). TARGET − SUPPORTED is six values in two natural units. A numeric-only R1 would be about 10× smaller than the initial build (fp32 needs zero kernel change, measured), so it absorbs placement. Placement is the unit the bf8_b production cells need anyway: every Panoptic cell is bf8_b with L1/mixed inputs and an L1 output.
- **Qwen 79 (`row_gather`) is the op's largest headroom, but no perf refinement is filed for it.** Only p1 carries the `attention: PERF FOCUS` flag, and the shape-selection rule pins every perf refinement to a flagged shape. Qwen 79 runs ~14× off its byte bound and is issue-bound: about 576 reads of 32 B plus about 288 local copies per core on NCRISC, against about 18 writes on BRISC. The design's lamps cover it: *Gather issue balance* (move `gather_place_block` to the writer, `split_reader`) and *Gather fetch/place overlap*. The core-scaling sweep in `test_g6_concat_device_perf.py` classifies the bound before anyone touches it. If the perf focus is ever widened, this is the next target.
- **Grid-synchronization lamp.** FLUX 22/23 (8 pages) and Qwen 104 (16) run one page per core. Those cells are ≤ 32 tiles, so `perf_rules.txt` forbids measuring them. No action.
- **R1 must build the L1-input path of `row_gather`.** The gather reader fixes `input_read_alignment = NOC_DRAM_READ_ALIGNMENT_BYTES` at compile time. With all-L1 bf16 inputs, `has_staging` is false and the `static_assert` fires, which is a compile error. No golden cell hits it, but SUPPORTED would claim `l1 × h × h_non_aligned` after R1. Details are in the refinement entry.
