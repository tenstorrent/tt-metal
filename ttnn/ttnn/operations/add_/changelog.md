# add_ changelog

## Phase 0 — `sharded_resident` (ttnn-implementer, 2026-11-21)

Built as op_design.md specifies; no binding element simplified.

- Op file `add_.py`: `INPUT_TAGGERS = AXES` (spec), `SUPPORTED` = `PHASE0`, `EXCLUSIONS = []`, `validate()`:
  X1 (`ValueError`), R7 (dtype / layout / `eval.memory.memory_axes` placement of both operands), the axis gate,
  then the shard-spec checks of the regime (same grid / shape / orientation, whole tile rows, full padded
  width). `default_compute_kernel_config()` is the only home of the default (R2); fidelity / approx mode are
  never gated (R3). One `ttnn.generic_op` per call; returns `input_tensor_a` itself (R4).
- Host `add__program_artifacts.py` (`ttnn.ProgramSpec`, named after the Quasar reference examples'
  `*_program_artifacts.py`): three rings borrowed from the shards (`cb_a_shard`←a, `cb_b_shard`←b,
  `cb_result_shard`←a), `THREAD_PLAN = (1, 4, 1)` clamped so C | shard_tiles and C ≤ busiest core's tiles
  (R, W = gcd with C). Cores whose shard holds no tile get no kernels (R8).
- Kernels: reader / writer post and acknowledge credits per NEO share (explicit sync, rings opted out of
  implicit sync, no NoC transfer, no tag ever set). Compute: one `compute_kernel_lib::add` call per NEO
  (`PerBlockSize`, `block_size(neo_block_tiles)`, `PackRelu::Zero` iff relu); `tag_threads` test probe.

Structural counts (ttsim has no timing; any time gain is unconfirmed until a timed backend measures it):
- DRAM bytes: 0. NoC bytes: 0. Op-owned L1: 0 B (all rings borrowed).
- Compute threads at work per core: 4 (2 on a 2-tile shard). Helper calls per NEO per op: 1.
- DM credit calls per core: 2·C (reader), C (writer).

Findings:
- The packer ReLU on 32-bit DEST works on ttsim (the `all_negative` regression case passes).
- A borrowed ring of 1,605,632 B (above the target page's 1,048,560 B per-ring figure) is accepted and correct
  (`test_add__plan.py::test_ring_above_one_mib`), so `staged_shard` stays deferred.
- Two rings borrowed from the same tensor (`cb_a_shard`, `cb_result_shard`) are accepted by the host.

Tests: acceptance 4/4; `test_add__plan.py` 12/12 (every NEO ran, helper at C = 1/2/4, one device session
before / after tt-metal programs, ResNet-50 grids with partly filled and empty shards, >1 MiB ring); golden
suite 23 passed, 850 xfailed, 289 skipped, 0 failed.

## Phase 0 — Verification (ttnn-verifier, 2026-11-21)

- **SUPPORTED at Phase 0**: dtype=[bfloat16], fp32_dest_acc_en=[True], layout=[TILE], memory_config=[L1],
  sharding=[height], activation=["relu"], alignment=[tile_aligned, h_non_aligned], rank=[4], broadcast=[none]
  (= the specification's PHASE0; `validate_contract --phase0` passes).
- **Accuracy achieved**: PCC ≥ 0.9999974, max_abs_err = 0.0156, mean_abs_err ≈ 7.4e-4, relative RMS ≈ 0.0022,
  bf16 ULP p99 = 0 / max = 1; got/true ratio median 1.0000 (measured on 4 shapes via
  `test_add__precision_baseline.py`).
- **Golden suite at Phase 0**: 22 / 22 supported cells passing (14 PHASE0 + 8 numeric regression), 850
  xfail_expected, 288 invalid_skipped, X1 rejection passing; 0 supported_fail / xpass_drift / xfail_wrong_mode;
  queue_gaps 0 (`verifier_report.json`).
- **Issues encountered**: none in the kernels or the contract. Code-review fix: the compute-config default and
  the activation are now resolved once (`_resolve_config()`, axes returned by `validate()`), so `add_()` no longer
  duplicates them.
- **Probes** (verifier, `tests/ttnn/unit_tests/operations/add_/probes/`): probe_002 shows activation None,
  w_non_aligned and ranks 2/3 already correct on the sharded path (SUPPORTED widened in the probe only); probe_003
  shows the L1-interleaved page→bank→core map (32 banks, one per worker core) that Refinement 3 builds on.
- **Tests added**: `test_add__precision_baseline.py`.
- **Queue**: `op_requirements.md`. Refinement 1 is the interleaved DRAM/L1 regime plus the gate-only axes;
  Refinement 2 adds float32 and the 16-bit accumulator; Refinement 3 is perf (structural gate), a bank-local deal
  for L1-interleaved operands.
