# Changelog: multigammaln

## Phase 0 — Core Implementation

- **Date**: 2026-05-11
- **What was done**: initial implementation via the incremental pipeline
  (planner → implementer → verifier). The verifier identified a missing
  per-CB `unpack_to_dest_mode=UnpackToDestFp32` setting that was causing
  inputs near integer boundaries (e.g., `a ≈ 0.500`) to produce `+inf` in
  sub-phase B. The fix was added to the program descriptor; all 18 acceptance
  tests, the 4 precision-baseline tests, and the 5 extended tests now pass
  (27/27 total).
- **Accuracy achieved**: PCC ≈ 0.9999993 across the baseline shapes; max abs
  error ≈ 0.054; mean abs error ≈ 3.9e-3; relative RMS ≈ 6.3e-4. Tolerances
  in the acceptance test are `rtol=0.05`, `atol=0.2` (the multi-step compute
  band); the achieved precision exceeds those by 1–2 orders of magnitude.
- **Issues encountered**:
  - The `sfpu_chain` framework discards op instances (default-constructs the
    chain from the type list), so `AddScalar`'s scalar member cannot be
    preserved. Sub-phase B is implemented with raw APIs as a result and the
    framework fix is logged as Refinement 4 in `op_requirements.md`.
  - Inputs near `a = 0.500` initially produced `+inf` in sub-phase B. Root
    cause was `UnpackToDestMode` defaulting to the SrcA/SrcB path, causing
    TF32 truncation of intermediate fp32 reloads. Setting
    `UnpackToDestFp32` on every fp32 CB fixed the issue.
- **Tests added**:
  - `test_multigammaln.py` (acceptance — provided to the implementer).
  - `test_multigammaln_precision_baseline.py` (new — PCC, abs error,
    relative RMS across 4 shapes).
  - `test_multigammaln_extended.py` (new — output memory-config,
    large-magnitude Stirling, domain-boundary strip, multi-core stress).
