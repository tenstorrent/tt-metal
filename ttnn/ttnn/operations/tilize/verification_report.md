# Verification Report: tilize

Date: 2026-07-17
Verifier pass over the registry-model `tilize` op (ROW_MAJOR → TILE layout conversion).

## Summary

The op is in good shape and the implementer already grew `SUPPORTED` well beyond the
Phase-0 baseline (multi-core, all rank 2/3/4, all interleaved buffer directions,
fp32 + bf8b output). All acceptance tests pass (35/35), the golden suite is clean on
every loud verifier category (`supported_fail=0`, `xpass_drift=0`,
`xfail_wrong_mode=0`), and bf16/fp32 tilize is bit-exact.

The remaining gap toward `feature_spec.TARGET` is exactly two features: **uint32
integer passthrough** and **sharded I/O** (legacy_2d HEIGHT/WIDTH/BLOCK + nd). Both
are filed as refinements.

## Code Review

**Fixed:**
- **Golden test helper bug — `eval/golden_tests/tilize/helpers.py:71`.** `_check_grid_bounds`
  read `core_range.end_coord`, which does not exist on `ttnn.CoreRange` (the attribute
  is `.end`). This raised `AttributeError` during test *setup* (`build_mem_config`) for
  every sharded scenario — 35 golden cells crashed before ever reaching the op, so they
  were mis-categorized as `xfail_wrong_mode` (a loud failure) instead of the clean
  `xfail_expected` refusal they should be. Changed `.end_coord` → `.end`. This is a pure
  mechanical test-infra fix (not a spec change) and it moved all 35 cells to
  `xfail_expected`, zeroing the loud category. The previous implementer noted this issue
  in `changelog.md` but declined to fix it ("golden tests are ground truth"); it is an
  authoring typo in a test helper, not a spec decision, so it was fixed here to make the
  golden run honest.

**Reviewed — no change needed:**
- **fp32 `Fp32Mode::Lossless` (kernel deviates from `op_design.md`, which said `Fast`).**
  This is a *correct* and well-documented deviation. tilize is a **terminal** op: the
  tiled output goes straight to DRAM/L1 with no downstream FPU consumer, so the design's
  "FPU re-truncates to tf32 anyway" rationale does not hold. The golden oracle for fp32
  is `comp_equal` (exact). `Fast` (tf32 truncation) would fail the exact fp32 identity;
  `Lossless` + `UnpackToDestFp32` (set on `cb_rm_in` in the descriptor) +
  `fp32_dest_acc_en=true` gives bit-exact fp32 (confirmed: max_abs = 0 in the precision
  baseline, and all fp32 golden cells pass exact). The kernel comment documents the
  reasoning. Keep.
- **Registry wiring.** `INPUT_TAGGERS` (all with the `(inputs, axes)` signature),
  `SUPPORTED`, `EXCLUSIONS`, `validate()` all present and correctly ordered
  (SUPPORTED per-axis → EXCLUSIONS cell-level, both raising `_op_contract` refusals).
  `validate()` is the first line of the public entry point. The op file does **not**
  declare `INVALID` (correct — it lives in `feature_spec.py`).
- **Kernel hygiene.** All three kernels use `void kernel_main()`, include
  `api/dataflow/dataflow_api.h` (not the bare path), and use `TensorAccessor` (not the
  deprecated `InterleavedAddrGen`). CB push/wait counts match (`Wt_chunk` per block on
  both CBs) — no sync mismatch, no hangs observed.
- **Writer raw `noc_async_write` loop.** The design's "helpers considered and rejected"
  analysis is correct — there is no tile-page writer helper in the dataflow kernel lib
  (`write_sticks_after_untilize` emits RM sticks and would destroy the tile layout). The
  batched raw loop (one barrier per `Wt_chunk` block) is the canonical example-op pattern.
- **Wide-W CB bounding.** The reader chunks W to a constant (`WT_CHUNK_MAX=8`) via
  `byte_offset_within_page`, so the CB footprint is bounded by a constant, not `Wt`.
  Correct per the design.

**Advisory (not blocking, not a refinement — perf micro-opt):**
- **Compute init/uninit fusion across the W-chunk loop.** `tilize_compute.cpp` calls
  `tilize<...InitAndUninit>` once per chunk, re-running `tilize_init`/`tilize_uninit` on
  every chunk. For wide W (`num_chunks > 1`, e.g. the perf bench W=2048 ⇒ 8 chunks) this
  is redundant work between chunks whose CB formats do not change. The helper supports
  `InitOnly`/`Neither`/`UninitOnly` fusion (example 6 in `tilize_helpers.hpp`). The golden
  cells (W ≤ 512 ⇒ `num_chunks` = 1–2) barely exercise this, so it is left as an advisory
  tied to the perf-overlap work rather than fixed here — the loop index is a runtime
  variable, so fusing requires special-casing first/middle/last chunk, and this belongs
  with the R2-style read/compute/write overlap tuning, not a correctness pass.

## Registry Conformance

- **Confirmed present and wired:** `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate()`.
  `validate()` gates SUPPORTED then EXCLUSIONS and is called first in `tilize()`.
- **Confirmed the op file does NOT declare `INVALID`** (it is sourced from
  `feature_spec.py`).
- **No drift auto-fixes needed.** `xpass_drift = 0` — every axis value the op accepts
  (multi-core, all buffers, all ranks, fp32, bf8b output) is claimed in `SUPPORTED` and
  passes. The implementer's SUPPORTED block is honest.

### INVALID audit (`eval/golden_tests/tilize/feature_spec.py`)

`INVALID` declares the 5 int↔float cast crosses (uint32↔{bf16,fp32,bf8b}). Well-formed
against the three sanity rules:
- **Single-tensor coupling:** `dtype` (input value format) × `output_dtype` (output value
  format) describe the same logical tensor's value across the op — not two different
  tensors. An int↔float "cast" is value reinterpretation, a structurally different op,
  not value preservation. Legitimate INVALID (not a "not-yet-implemented" EXCLUSION).
- **Universe-must-change:** correct — int↔float is out of tilize's value-preserving contract.
- **Canonical bf8b+ROW_MAJOR:** correctly *absent*, and the reason is documented in the
  file: bf8b is never an INPUT dtype here (the input is always ROW_MAJOR; block-float has
  no row-major form). bf8b only appears as a TILE output, which is legal. So there is no
  bf8b+ROW_MAJOR cell to prune.
- **No cross-tensor-axis coupling; no norm-like weight axes** (N/A for a layout op).

No changes recommended to `INVALID`.

## Precision Baseline

Measured via `tests/ttnn/unit_tests/operations/tilize/test_tilize_precision_baseline.py`
(4 shapes × {bf16→bf16, fp32→fp32, bf16→bf8b}).

| Shape | Cast | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|------|-----|-------------|--------------|------------------|
| (1,1,32,32)   | bf16→bf16 | 1.0 (exact) | 0.0 | 0.0 | 0.0 |
| (1,1,64,128)  | bf16→bf16 | 1.0 (exact) | 0.0 | 0.0 | 0.0 |
| (2,3,128,256) | bf16→bf16 | 1.0 (exact) | 0.0 | 0.0 | 0.0 |
| (1,1,512,512) | bf16→bf16 | 1.0 (exact) | 0.0 | 0.0 | 0.0 |
| (1,1,32,32)   | fp32→fp32 | 1.0 (exact) | 0.0 | 0.0 | 0.0 |
| (1,1,64,128)  | fp32→fp32 | 1.0 (exact) | 0.0 | 0.0 | 0.0 |
| (2,3,128,256) | fp32→fp32 | 1.0 (exact) | 0.0 | 0.0 | 0.0 |
| (1,1,512,512) | fp32→fp32 | 1.0 (exact) | 0.0 | 0.0 | 0.0 |
| (1,1,32,32)   | bf16→bf8b | ≥0.99 | 2.34e-02 | 7.45e-03 | 9.70e-03 |
| (1,1,64,128)  | bf16→bf8b | ≥0.99 | 2.93e-02 | 7.13e-03 | 9.37e-03 |
| (2,3,128,256) | bf16→bf8b | ≥0.99 | 4.69e-02 | 7.07e-03 | 9.32e-03 |
| (1,1,512,512) | bf16→bf8b | ≥0.99 | 4.69e-02 | 7.08e-03 | 9.32e-03 |

**Assessment:** tilize does no arithmetic, so value-exact dtypes (bf16, fp32) are
bit-identical (max_abs = 0) — exactly as required. The only loss is the value-preserving
cast into bfloat8_b at pack time, which is the expected block-float quantization
(rel-RMS ~1e-2, PCC well above 0.99).

**Recommended tolerances:** bf16/fp32 identity: exact (`comp_equal`) or PCC ≥ 0.9999.
bf8b cast: PCC ≥ 0.99, atol ≈ 0.05.

## Verifier CLI Summary

From `verifier_results/verifier_report.json` (post helper-fix run, `/tmp/tilize_results2`):

- supported_pass:      36
- xfail_expected:      41  (35 sharded + 6 uint32→uint32 — the refinement queue)
- invalid_skipped:     55  (int↔float casts)
- supported_fail:       0  ✓ (must be 0 to ship)
- xpass_drift:          0  ✓ (must be 0 to ship)
- xfail_wrong_mode:     0  ✓ (must be 0 to ship, was 35 before the helper fix)
- supported_marked_xfail: 0 ✓
- no_axes_found:      144  (non-registry extra files: `test_golden_main_tests.py`,
                            `test_regression.py` — see note below)

**`no_axes_found` note.** These 144 come from `test_golden_main_tests.py` (the hidden
main-branch grader) and `test_regression.py`, which are not registry-driven `test_op`
cases and carry no axes for the CLI to map. Their 72 "failures" are all the op correctly
refusing sharded input with `UnsupportedAxisValue` (a `NotImplementedError`) — verified
by traceback. They are not a loud-category signal; they flip to passing once the sharding
refinement lands.

## TARGET − SUPPORTED gap (drives the refinement queue)

| Axis | TARGET | SUPPORTED | Missing | Disposition |
|------|--------|-----------|---------|-------------|
| dtype | bf16, fp32, uint32 | bf16, fp32 | **uint32** | Refinement 1 |
| output_dtype | bf16, fp32, bf8b, uint32 | bf16, fp32, bf8b | **uint32** | Refinement 1 |
| use_multicore | False, True | False, True | — | complete |
| shard_api | none, legacy_2d, nd | none | **legacy_2d, nd** | Refinement 2 |
| out_scheme | interleaved, HEIGHT, WIDTH, BLOCK, nd | interleaved | **HEIGHT, WIDTH, BLOCK, nd** | Refinement 2 |
| buffer | 4 dirs | 4 dirs | — | complete |
| rank | 2, 3, 4 | 2, 3, 4 | — | complete |

Every `xfail_expected` cell maps to one of the two refinements (uint32↔uint32; sharded
schemes), or is a cross of both (uint32 + sharded — needs both). No orphan gap cells.
int↔float crosses are `INVALID`, not queue entries.

## Recommendations

- **Refinement order:** uint32 (R1) first — it is interleaved-only and independent; the
  4 uint32+sharded crossed cells only pass after *both* refinements. Sharding (R2) is the
  larger memory-config change and goes last per the memory-pressure ordering rule. See
  `op_requirements.md`.
- **Perf continuation (report-level, not queue entries — no clean measured off-roofline
  gap with a named unaddressed lever):** the prompt frames this run as performance-first.
  Multi-core (prompt R1) and depth-2 CBs (prompt R2 partial) are already implemented; the
  changelog records a wall-clock R0 bracket (single 918 µs / multi 276 µs vs ~58 µs DRAM
  floor) but flags those numbers as dispatch-overhead-laden, not clean device-kernel
  durations. Before any perf lever can be filed as a refinement, the queued formal R0
  work is needed: stub-compute ablation to confirm DM-bound, tt-npe DRAM-util/congestion
  pin, and a median Tracy device-kernel duration (`/perf-measure`, `/perf-roofline-dm`).
  The **sharded-output zero-copy write** (the big perf lever) rides on Refinement 2 —
  when it lands, re-target the roofline (the write becomes an L1 loopback, not DRAM).
- **When sharding lands (Refinement 2):** `validate()._shard_api_of` currently returns
  `"legacy_2d"` for *any* sharded config, including nd — fix it to distinguish nd (return
  `"nd"`) so the axis matches the tagger; and add the design's `EXCLUSIONS` for
  single-core + sharded (sharding is inherently multi-core). Both are called out in the
  refinement entry.
