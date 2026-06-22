# Changelog: point_to_point

## Phase 0 — Core Implementation
- **Date**: 2026-06-22
- **What was done**: Initial implementation via the incremental pipeline (planner → implementer
  → verifier). Self-contained Python CCL data-movement op built on `ttnn.generic_op` +
  `ttnn.MeshProgramDescriptor` with newly-authored sender/receiver dataflow kernels
  (`reader_send` / `writer_send` / `reader_receive` / `writer_receive`). Pure cross-chip byte
  copy over the Tenstorrent fabric (1-D unicast, single worker core, single link); 2-party
  `GlobalSemaphore` handshake with cache-reuse reset. Does **not** wrap the bound C++
  `ttnn.point_to_point` (used as a correctness reference only).
- **SUPPORTED at Phase 0** (== TARGET on every axis):
  dtype=[bfloat16, float32, bfloat8_b, uint16, int32, uint32], layout=[TILE, ROW_MAJOR],
  topology=[Linear, Ring], alignment=[tile_aligned, non_tile_aligned]; EXCLUSIONS=[].
- **Accuracy achieved**: pure byte copy → analytically bit-exact (PCC=1.0, max_abs_err=0,
  mean_abs_err=0, rms_err=0) for every dtype. **Not yet measured on hardware** — the precision
  baseline (`test_point_to_point_precision_baseline.py`) requires ≥ 2 mesh devices and skipped
  on this 1-device machine. Table to be populated when ≥2-device hardware is available.
- **Golden suite at Phase 0**: **0 / 0 cells** (per `verifier_report.json`, `total: 0`). The
  golden suite could not run: (a) `eval/golden_tests/conftest.py` imports the missing
  `ttnn.operations._op_contract` (collection aborts for all ops), (b) the `point_to_point`
  golden dir has only `feature_spec.py` — no `test_golden.py` harness wiring, and (c) the
  transfer needs ≥ 2 devices. This is a *no-signal* run, not a clean run.
- **Issues encountered / fixes applied this pass**:
  - **FIX (pipeline-blocking)**: `__init__.py` now re-exports `SUPPORTED` (lazy, via package
    `__getattr__`) / `EXCLUSIONS` / `INPUT_TAGGERS` / `validate` from the implementation
    submodule. Before this, the registry declarations were invisible on the package, so
    `eval.verify_supported` crashed (`missing required 'SUPPORTED'`) and the golden harness's
    `from ttnn.operations.point_to_point import ...SUPPORTED` would have failed too.
  - **FIX (cleanup)**: `reader_receive.cpp` `packet_l1_addr` retyped `uint64_t → uint32_t` to
    match `get_write_ptr()`'s return type and its `uint32_t` use sites.
  - No drift fixes (no XPASS evidence; nothing ran). No SUPPORTED changes (already == TARGET).
- **Tests added**:
  - `test_point_to_point_extended.py` — integer dtypes (uint16/int32/uint32, bit-exact) +
    non-tile-aligned shapes; the two SUPPORTED axis values the acceptance test omits.
  - `test_point_to_point_precision_baseline.py` — PCC + max/mean abs error + relative RMS
    across 4 shapes × 3 dtypes (copy-fidelity vs the sent shard).
  - Both collect cleanly and skip on < 2 devices (multi-device `mesh_device` fixture).
- **Verifier artifacts**: `verifier_report.json` (this directory), `verification_report.md`,
  `op_requirements.md`.
