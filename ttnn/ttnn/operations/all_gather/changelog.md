# Changelog: all_gather

## Phase 0 — Core Implementation

- **Date**: 2026-06-25
- **What was done**: Initial implementation via the incremental pipeline
  (planner → implementer → verifier). Self-contained Python CCL op
  (`ttnn.generic_op` + `ttnn.MeshProgramDescriptor`) with newly-authored fabric
  ring/line dataflow kernels (per-device forward + backward worker cores, each a
  reader (NCRISC) + writer (BRISC)). Bidirectional store-and-forward on a 1-D
  line; pure byte movement (identity gather, no compute). Fabric egress through
  the safety-by-construction `ccl_helpers_dataflow.hpp` helper; cross-device
  ordering via one cached op-internal `GlobalSemaphore` (counting atomic-inc +
  reader-side cache-reuse reset).
- **SUPPORTED at Phase 0**: dtype=[bfloat16, float32], layout=[TILE, ROW_MAJOR],
  topology=[Linear], gather_dim=[-4] (≡ gather_dim 0, page-contiguous concat),
  alignment=[tile_aligned, non_tile_aligned]. Memory: interleaved (DRAM/L1);
  sharded rejected by validate().
- **Accuracy achieved**: PCC=1.0, max_abs_err=0, mean_abs_err=0, rms_err=0 —
  bit-for-bit identity copy (measured on 4 shapes × {bf16, f32} via
  `test_all_gather_precision_baseline.py`, run on the WH sim).
- **Golden suite at Phase 0 (hybrid)**: 384 cells = 32 supported_pass + 288
  xfail_expected + 64 INVALID-skipped (`bf8b × ROW_MAJOR`). Loud categories
  (supported_fail / xpass_drift / xfail_wrong_mode) all 0. Categories computed
  with the harness logic (`eval.feature_matrix`); SUPPORTED-rectangle cells
  **observed passing on the WH sim**. See
  `eval/results/all_gather/verifier_report.json`.
- **On-device verification (PASSED)**: ran on the deterministic
  `wh_t3k_allmmio_all_gather` sim (`(1,8)` line, `FABRIC_1D`,
  `scripts/run_multidevice_sim_pytest.py --op all_gather`):
  - acceptance: **22/22 passed** (bf16/f32 × TILE/RM × 5 shard shapes,
    gather_dim=0, Linear) + program-cache (2-call) + output_tensor paths.
  - precision: **8/8 passed**, zero error.
  - aggregate exit 0. The cross-device gather actually executed and PCC asserted
    — not code review alone. (Contrast `point_to_point`, whose BH sim was
    fabric-blocked; the WH all-MMIO sim for all_gather works.)
- **Issues encountered / fixes applied**:
  - **Code-review fix**: simplified the `validate()` page-alignment gate
    (`page % 16 != 0 and page != 16` → `page % 16 != 0`; the `and page != 16`
    clause was vacuous) and documented *why* the check is load-bearing — the
    fabric writer sends `align(page_size, l1_alignment)` bytes per page while the
    output TensorAccessor spaces pages by the raw `page_size`, so a non-16-aligned
    page would overrun the next output page. No behavior change.
  - **Registry conformance**: already correct as shipped — INPUT_TAGGERS
    (`alignment` + `tag_alignment`), SUPPORTED (all gated axes), EXCLUSIONS (empty),
    validate() (first line, correct order, negative gather_dim canonicalization),
    no INVALID in the op file. (The `alignment`-axis bug that had to be fixed
    during `point_to_point` verification was not present here.)
  - **Design deviation (benign, no action)**: the writer omits the design's
    optional Phase-1 startup barrier (`arm_multicast_inc`) and relies on the
    counting semaphore + persistent output for ordering, plus the reader's
    end-reset for cache re-arm. This is a sound simplification — it avoids the
    helper's shared-sem-header footgun and is empirically correct (all 30 sim
    cases pass, including the program-cache reuse test that the barrier+reset
    dance exists to protect). See `verification_report.md` → Design Conformance.
- **Tests added**:
  - `tests/ttnn/unit_tests/operations/all_gather/test_all_gather.py` (acceptance —
    pre-existing immutable spec; confirmed green on the sim).
  - `tests/ttnn/unit_tests/operations/all_gather/test_all_gather_precision_baseline.py`
    (new — PCC + max/mean abs + relative RMS over 4 shapes × {bf16, f32}).
- **Refinement queue**: 3 refinements filed in `op_requirements.md` —
  R1 bfloat8_b dtype, R2 gather_dim != 0 (strided concat addressing), R3 Ring
  topology. Every `TARGET − SUPPORTED` pair is covered; `bf8b × ROW_MAJOR` is
  INVALID. No implementation-skill applies (CCL fabric axis expansions are
  outside the current skill inventory) — all three are verifier-authored.

## Refinement 1 — bfloat8_b dtype

- **Date**: 2026-07-02
- **What was done**: Added `ttnn.bfloat8_b` to `SUPPORTED["dtype"]`. **No kernel
  or program-descriptor change** — all_gather copies physical pages verbatim
  (never tilizes), so a bf8b TILE page (a 1088 B block-float tile) is gathered
  bit-for-bit exactly like a bf16/f32 tile. The `data_format = input.dtype`
  descriptor path already flowed bf8b through the CBs and the fabric writer; the
  16B-page invariant in `validate()` holds (1088 % 16 == 0). `validate()` is
  generic (iterates SUPPORTED), so no gate change was needed beyond the list.
- **Accuracy achieved**: bf8b PCC ≥ 0.99 (golden tolerance 0.999 also passed).
  Precision baseline (`test_all_gather_precision_baseline.py`, WH sim): bf8b
  `max_abs = 0.03125`, `mean_abs ≈ 0.006`, `rel_rms ≈ 0.0077` on shards
  {(1,1,64,128),(1,1,96,64),(1,1,256,256),(1,1,32,32)} — this is purely the
  shared-exponent quantization applied at `from_torch`; the gather itself adds
  zero error. bf16/f32 remain bit-exact (`max_abs = mean_abs = rel_rms = 0`),
  confirming no regression.
- **Golden progress (WH sim)**: the 8 `bf8b × TILE × Linear × gather_dim=-4`
  golden cells (one per `feature_spec.INPUTS` shape) moved from xfail-strict to
  **PASSING** (8/8, 119.78s). The remaining 56 non-INVALID bf8b cells stay xfail
  (multi-gap: `gather_dim ∈ {-3,-2,-1}` needs R2, `Ring` needs R3); the 64
  `bf8b × ROW_MAJOR` cells stay INVALID-skipped. supported_pass 32 → 40.
- **Issues encountered**: None. Trivial as the verifier predicted; no EXCLUSIONS
  added.
- **Tests added/updated**: extended
  `tests/ttnn/unit_tests/operations/all_gather/test_all_gather_precision_baseline.py`
  — added `ttnn.bfloat8_b` to `DTYPES` + `PCC[bfloat8_b] = 0.99`, and taught
  `_torch_dtype` to reference bf8b in torch.bfloat16 (no native torch bf8b). Full
  suite 12/12 green on the WH sim.
