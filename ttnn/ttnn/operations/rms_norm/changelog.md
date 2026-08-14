# Changelog: rms_norm

## Phase 0 — Core Implementation

- **Date**: 2026-08-13
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  Native `ProgramDescriptor` op with a 2D core partition — `num_row_groups` rectangles × `num_hidden_slices`
  cores per rectangle — including a real cross-core combine on the dependent (hidden) axis
  (gather-to-root + `Mcast2D` broadcast of the finalized `rsqrt`). TILE and ROW_MAJOR are both handled
  natively in the kernels (tilize/untilize staging, no host-side transform), non-tile-aligned H and W
  natively (W-mask in compute for TILE, reader tail zero-fill for ROW_MAJOR), and the `1/W` factor uses the
  true element count exactly once, after the combine.
- **SUPPORTED at Phase 0**: dtype=[bfloat16, float32], fp32_dest_acc_en=[True],
  layout=[TILE, ROW_MAJOR], alignment=[tile_aligned, w_non_aligned, h_non_aligned], rank=[2, 3, 4],
  gamma_mode=[gamma, no_gamma], gamma_dtype=[bfloat16, float32, "none"],
  gamma_layout=[TILE, ROW_MAJOR, "none"], memory_layout=[INTERLEAVED]. `EXCLUSIONS = []`.
- **Accuracy achieved**: bf16 PCC=0.999997, max_abs_err=0.051, mean_abs_err=1.2e-03, rel_rms_err=2.4e-03;
  fp32 PCC=0.999999, max_abs_err=0.026, mean_abs_err=9.0e-04, rel_rms_err=1.6e-03
  (measured over 40 cells = 5 shapes × 2 dtypes × 2 layouts × gamma/no-gamma via
  `test_rms_norm_precision_baseline.py`). Scale check: `median(got/true)` = 0.9998 (bf16) / 0.9986 (fp32),
  width- and alignment-independent ⇒ no padded-denominator / scale bug.
- **Golden suite at Phase 0**: 737 / 737 supported cells passing; 6174 xfail_expected, 33900 invalid_skipped;
  supported_fail = xpass_drift = xfail_wrong_mode = 0 (per `verifier_report.json`, results dir
  `/tmp/rms_verify2`). Full run: `PASSED=752 FAILED=0 ERRORS=0 SKIPPED=33900 HANGS=0 TOTAL=40828`.
- **Issues encountered** (all fixed in this verification pass, no drift fixes were needed):
  1. ROW_MAJOR reader/writer compared a *stick* counter against the tiles-per-barrier knob `DM_CHUNK_TILES`,
     barriering ~8× more often than the TILE path; both now derive `RM_CHUNK_STICKS` from the same knob.
  2. `cb_w_mask` was allocated on every program; now created only when `mask_enabled`
     (needs a compile-time gate in the reader because `prepare_reduce_mask` `static_assert`s on the CB format).
  3. `IN_CB_DEPTH` looked like a free knob but is load-bearing at 1 (the in-place rewrite of x needs
     `write_ptr == read_ptr`); now asserted with the reason.
  4. Added `static_assert` on the ROW_MAJOR staging stick-pitch alignment.
  5. `l1_ledger.md` was stale in five places against the shipped CB set; brought current.
- **Tests added**: `test_rms_norm_precision_baseline.py` (40 cells, incl. the got/true ratio scale-bug
  detector). Pre-existing: `test_rms_norm.py` (immutable acceptance, 107), `test_rms_norm_debug.py`,
  `test_rms_norm_perf.py` (device-ns harness + the two knob sweeps).

## Refinement 1 — Numerical configurability expansion (unlocks the perf target's config)

- **Date**: 2026-08-14
- **What was done**: grew the precision surface to the whole TARGET rectangle, with **zero kernel
  changes** — the pass condition of `/numeric-formats-metal` held (compute is fully helper-based:
  `tilize` / `eltwise_chain` / `sum_of_squares` / `reduce` / `untilize`; no hardcoded sizes or
  data-format constants in any kernel).
  - `SUPPORTED["fp32_dest_acc_en"] = [True, False]`. The axis only selects the DEST accumulation
    width; the four stat CBs (`cb_sq_partials`, `cb_gathered_partials`, `cb_rms_bcast`,
    `cb_rms_recip`) stay `float32` regardless of input dtype **and** regardless of this axis —
    Phase 0 already pinned them that way (`stat_tile = ttnn.tile_size(ttnn.float32)`), so the
    measured accuracy requirement is met by construction, not by a new branch.
  - `ttnn.bfloat8_b` added to `SUPPORTED["dtype"]` and to `SUPPORTED["gamma_dtype"]`. CB formats
    were already derived from `input_tensor.dtype` / `gamma.dtype` / `output_tensor.dtype`, so the
    tile size (1088 B) and the `Bfp8_b` page format fall out of the existing derivation.
  - `EXCLUSIONS = [{"dtype": ttnn.float32, "fp32_dest_acc_en": False}]` — now expressible for the
    first time, and a permanent refusal, not a refinement candidate.
  - **The one host-side fix**: `Tensor.element_size()` raises `ValueError: datum for bfp2, bfp4,
    bfp8 is invalid` for block-float formats (`tt_backend_api_types.hpp:83`), which broke every
    `bfloat8_b` cell at descriptor-build time. Added `_elem_bytes()` in the program descriptor,
    which substitutes 1 for block-float. The `*_ELEM_BYTES` compile-time args it feeds are consumed
    **only** by the ROW_MAJOR stick paths, and a block-float tensor is necessarily TILE-layout
    (`{bfloat8_b, ROW_MAJOR}` is INVALID in `feature_spec.py`), so the substituted value is never
    dereferenced — it only has to keep the reader/writer `RM_STICK_PITCH % 16 == 0` static_assert
    (which is evaluated on *every* program, not just ROW_MAJOR ones) true.
  - `l1_ledger.md` symbol table brought current: `tile_bytes ∈ {1088, 2048, 4096}`, with the
    block-float datum-size note. No CB was added, resized, merged or deleted; the footprint
    expression and the data-movement budget are unchanged (`bfloat8_b` is the *smallest* tile, so
    it only shrinks the footprint and the `float32`-everywhere fit predicate stays a conservative
    upper bound).
- **Accuracy achieved** (device-measured, `test_rms_norm_precision_matrix.py`, 112 passed /
  36 skipped):
  - **The perf target's exact config** — bf16 / HiFi2 / `fp32_dest_acc_en=False` / TILE / bf16 TILE
    gamma, `(1,1,32,W)` for W ∈ {1024, 2304, 5120, 7168}: PCC **0.99998** at W=7168
    (rel-RMS 0.0138), against the `_perf_case` soft gate of 0.9995. Refinement 3 can now be
    specified at its real config instead of an `fp32_dest_acc_en=True` proxy.
  - `fp32_dest_acc_en=False` at bf16, across all 8 matrix shapes × 2 distributions × 4 fidelities:
    PCC ≥ 0.995 gate held everywhere including LoFi.
  - `bfloat8_b`: PCC ≥ 0.9998, rel-RMS ≤ 0.020 across input × {bf16, fp32, bfloat8_b, no} gamma ×
    both accumulation widths × shapes up to `(1,1,32,8192)` — against the golden bf8b tolerance of
    0.99 PCC / 0.10 rel-RMS, i.e. **~40× margin on rel-RMS**. The verifier's flagged hazard (the
    in-place double rewrite of `cb_input_tiles` re-quantizes `x·r` to block-float before the gamma
    multiply) is real but far below tolerance, so the fused in-place path was **kept** and no cell
    was excluded for it.
  - No regression: `test_rms_norm.py` (107) + `test_rms_norm_precision_baseline.py` +
    `test_rms_norm_debug.py` = 175 passed.
- **Golden test progress** (targeted slices, not the full suite — the harness re-runs everything):
  - `pad_poison` group: **6/6 interleaved passing** (all 6 were xfail at Phase 0). The
    padding-in-the-denominator guard holds at `fp32_dest_acc_en=False`.
  - `perf` group: **8/8 interleaved passing** (all were xfail at Phase 0), including the decisive
    `(1,1,32,7168)` case and all four prefill shapes.
  - `resilience` group: **86/86 interleaved passing** (all were xfail at Phase 0).
  - Cartesian `bfloat8_b` cells: **450/450 passing**. 5-shape full cartesian slice (all
    dtype × fp32_dest_acc_en × layout × gamma combos): **288/288 passing**.
  - `eval/golden_tests/rms_norm/test_regression.py`: 15/15.
  - Every remaining xfail in those groups is a `*_SHARDED` `memory_layout` — Refinement 2's scope.
- **Issues encountered**: one, the `element_size()` block-float raise described above. It presented
  as a clean `ValueError` at descriptor build (a free failure), not as a hang or a numerical
  mismatch. Neither of the two hazards the verifier flagged cost anything: `DEST_AUTO_LIMIT`'s
  4→8 doubling is handled entirely inside the helpers (no number is hardcoded anywhere in the op),
  and the `bfloat8_b` in-place re-quantization stayed ~40× inside tolerance.
- **Tests added**: `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_matrix.py` —
  the authoritative precision characterization mandated by `/numeric-formats-metal` §10, in three
  parts: `test_rms_norm_precision_matrix` (the gated axes dtype × `fp32_dest_acc_en`, over 8 shapes
  × 2 input distributions), `test_rms_norm_precision_fidelity` (the ungated `math_fidelity` axis,
  all 4 values, on 2 representative shapes), and `test_rms_norm_perf_target_config` (Refinement 3's
  exact config pinned as a named cell at its tighter 0.9995 gate, so a later perf phase cannot
  silently substitute an `fp32_dest_acc_en=True` proxy). Skips are exactly the op's refusal
  surface: `{float32, False}` (EXCLUSIONS) and `bfloat8_b` on a non-tile-aligned shape (INVALID).
