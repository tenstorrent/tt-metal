# Claude review round 8 — 2026-09-09 — SATISFIED

Fresh Claude Opus 5 / high-effort session `897d360e-89e5-467f-8060-14ee34aeec8b`
reviewed `f90f03425d8` and returned **SATISFIED**: no required changes remain.
The complete review is preserved in
`generated/reduce_migration_reviews/round_08_20260909/review.md`.

It verified R7.1 from the regenerated install script rather than from the source
list alone, confirming that `build/ttnn/cpp/ttnn/operations/normalization/cmake_install.cmake`
no longer references the missing header and that neither planning header is
exported. It also confirmed the tree is now configured after the rename, so the
blind spot that hid R7.1 is closed.

Build-system, install and packaging correctness was added to the review scope
this round, since R7.1 showed that defect class is invisible to incremental
builds. The audit found no second defect: every path in every `sources.cmake`
under `ttnn/` and `tt_metal/` exists, all six files the branch adds are covered
by an install or file-set rule, all five deletions are clean, and every include
in the 171 touched kernel files was cross-checked against the install file sets.

The regenerated reports were verified by re-running the generator's projection
in memory against the committed manifest: `sanity_test_suite.md`, `.html` and
`sanity_kernel_coverage.csv` reproduce byte-identically. All six CSV row changes
were re-verified against source, including `S090`'s reduction to factory `F085`.
Recorded test evidence was re-parsed independently, including all 63 JUnit files
from the post-rebuild sanity run.

## Non-blocking items addressed in this round

- The full unit-suite derived reports still carried base-inventory numbers with
  nothing pointing at the delta. `unit_test_suite.md` and `unit_test_suite.html`
  now state that the migration added 16 test definitions and 408 collected cases,
  that the runner's manifest holds 1,000 definitions and 18,860 known cases, and
  that `unit_test_suite.json`'s `migration_regressions` block is authoritative
  and is what `scripts/run_reduce_migration_tests.py` reads. Regenerating the
  full-suite md/html/csv bodies from the manifest, as was done for the sanity
  set, remains open.
- `check_counting_block` now validates the manifest's own `counting` block
  against recomputed totals, including `covered_by_kind` and `cases_by_lane`, so
  that summary can no longer drift silently.
- Group/kernel membership is now checked, and the asymmetry the review noted is
  reported rather than asserted away. `SM076` and `SM077` legitimately exercise
  kernels whose primary case is `SM011`, so requiring symmetry would have been
  wrong; the mislabel was the actual defect. The md and HTML now say "Kernels"
  and mark any entry whose primary case lies elsewhere, e.g.
  `SM077 … Kernels: S090 (primary case SM011), DF044 (primary case SM011)`. The
  validator still fails on an unknown kernel id or a listed kernel with no
  primary case.

## Non-blocking items deliberately left open

- The ignored `build_Release/libexec/tt-metalium/` install tree is stale and now
  also incomplete: 41 installed kernel files still call the retired dataflow
  helpers, and the tree lacks the new `attention/compute/softmax_reduce.hpp`, so
  a runtime pointed there would fail to compile the migrated softmax kernels
  rather than silently run old ones. JIT resolves against `TT_METAL_HOME`, so no
  recorded run is affected. It is a local build artifact; refresh or delete it
  before pointing anything at it.
- `sanity_kernel_coverage.csv` lists five kernel paths that no longer exist at
  HEAD (`S066`, `S067`, `S089`, `S092`, `DF014`). Expected for a baseline-pinned
  inventory: each gap reason is precisely why those files were deleted.
- Pre-existing and untouched: the Welford post reader's padded `Wt` versus
  compute's `tiles_per_core_y` in the 2D path; the RMSNorm `use_2d_core_grid`
  dispatch follow-up; `wh_generate_reduce_scaler` with no callers;
  `transformer/CMakeLists.txt` globbing only `*.hpp` for the SDPA kernel
  directories. Out of inventory scope by construction: the local
  `generate_reduce_scaler` in `models/demos/deepseek_v3_b1/kernel_includes/tt_metal/dm_utils.hpp:12`,
  a different helper family that was never inventoried.

## Remaining work

The review loop is complete. The full prepared regression — 178 groups, 1,000
definitions, 18,860 known cases — is the last outstanding user requirement and
has never been started. Its clean-checkpoint driver is
`generated/reduce_migration_reviews/full_regression_20260909/run_full_suite.py`.

Hardware limits are unchanged and cap what that run can prove: this host is a
two-chip Wormhole N300, so the Quasar, Blackhole, Galaxy, T3K and 1x4-fabric
lanes remain unexecuted. Round 8 added a source-level arithmetic proof of the
Quasar CTA layout, which is not a substitute for compiling or running it.
