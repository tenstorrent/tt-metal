# Claude review round 7 — 2026-09-09

Fresh Claude Opus 5 / high-effort session reviewed `26b2b030a5d` and returned
**CHANGES REQUIRED**. The complete review is preserved in
`generated/reduce_migration_reviews/round_07_20260909/review.md`. It verified
R6.1 as correct and complete, re-derived the migration scope from source, and
found one build-system regression that every recorded build was structurally
unable to detect.

## R7.1: orphaned softmax planning header in the module manifest

Round 5's name-collision fix renamed the host planning header in `97689309d9d`:

    softmax/device/softmax_reduce.hpp  →  softmax/device/softmax_reduce_plans.hpp

`ttnn/cpp/ttnn/operations/normalization/sources.cmake` kept the pre-rename path
inside `TTNN_OP_NORMALIZATION_API_HEADERS`, which
`normalization/CMakeLists.txt:33` consumes as a `FILE_SET api TYPE HEADERS`.
File-set headers are target sources, so the missing file breaks configure, and
the generated install script breaks `cmake --install` and wheel packaging.

Both failures were reproduced before the fix:

    CMake Error at ttnn/cpp/ttnn/operations/normalization/CMakeLists.txt:33 (target_sources):
      Cannot find source file:
        .../normalization/softmax/device/softmax_reduce.hpp

    CMake Error at build_Release/ttnn/cpp/ttnn/operations/normalization/cmake_install.cmake:54 (file):
      file INSTALL cannot find ".../softmax/device/softmax_reduce.hpp": No such file or directory.

The install failure came from the install script already generated in the tree,
so it was reachable without reconfiguring; the configure failure required
re-running CMake, which is why no recorded build hit it. `build` is a symlink to
`build_Release`, configured on 2026-09-07 and last generated 2026-09-08, both
before the 2026-09-09 rename. `sources.cmake` was not edited after the rename and
the header is in no `GLOB_RECURSE` set, so ninja never regenerated. No pytest or
gtest evidence touches configure or install.

The entry now follows its `groupnorm/device/groupnorm_reduce_plans.hpp` sibling
at `sources.cmake:12`: listed in `TTNN_OP_NORMALIZATION_SRCS`, not exported. Both
are host-only planning headers included solely by same-directory factories
(`softmax_program_factory_attention_optimized.cpp:6`,
`..._sharded.cpp:6`), so neither belongs in the exported API file set. This also
keeps them out of `INTERFACE_HEADER_SETS_TO_VERIFY api`, which would otherwise
compile them standalone.

## Other review notes

- The three derived sanity reports had drifted from the manifest, and the drift
  ran in both directions. `sanity_test_suite.md`, `sanity_test_suite.html` and
  `sanity_kernel_coverage.csv` still reported 74 cases / 129 covered kernels and
  listed `DF001` as a gap, while `sanity_test_suite.json` had 77 cases,
  130 covered and `DF001` covered by `SM075`. In the other direction the CSV
  carried the current `SM006`, `SM022`, `SM023` and `SM028` selections while the
  manifest's `kernels[]` copies still held superseded text.
- `scripts/generate_reduce_migration_sanity_reports.py` now projects all three
  artifacts from the manifest, so they cannot drift again. It refuses to run when
  a `kernels[]` entry disagrees with its group, or when a lane has no documented
  requirement, and requires collection evidence covering every manifest group.
- Corrected inside the manifest, from the authoritative group entries: eight
  stale `kernels[]` selections and seven stale `test_evidence` values, a missing
  `SM075` `test_evidence`, and a stray `"; "` prefix on 69 kernel evidence
  strings. Regenerating then changed six CSV rows, all corrections: `DF001` to
  covered by `SM075`; `S080`/`DF036` to the current `SM006` Python selection;
  `S090` to factory `F085` alone, since round 6 removed the RMSNorm branch from
  the Welford factory and `F086` no longer references
  `rmsnorm_post_allgather_metal2.cpp`; and unnecessary CSV quoting on
  `DP010`/`DP012`.
- Not changed: the untracked `build_Release/libexec/tt-metalium/` install tree
  still holds 33 pre-migration kernel copies that call the retired dataflow
  helpers. JIT resolves against `TT_METAL_HOME`, so every recorded run is
  unaffected, but a runtime pointed at that install tree would load stale
  kernels. It is an ignored local build artifact; refresh or remove it before
  pointing anything at it.
- Confirmed unchanged pre-existing observations, restated by the review and not
  migration debt: the Welford post reader's padded `Wt` versus the compute
  kernel's `tiles_per_core_y` in the 2D path, the RMSNorm `use_2d_core_grid`
  dispatch follow-up, and `wh_generate_reduce_scaler` having no callers at either
  baseline or HEAD.

## Validation

- Configure and install were both exercised, in the failing and fixed states:
  `cmake -S . -B build` fails without the fix and passes with it;
  `cmake --install build --component ttnn-dev --prefix <scratch>` fails without
  it and completes with it, installing `softmax_device_operation.hpp`,
  `softmax_operation_types.hpp` and `softmax.hpp` and, as intended, neither
  planning header.
- Native build passed after reconfigure: `cmake --build build --target ttnn
  unit_tests_ttnn --parallel 8`. The reconfigure invalidated the test PCH, so
  `unit_tests_ttnn` was fully rebuilt and both targets relinked.
- The five sanity groups covering the two factories that include the renamed
  header passed **5/5**, with no skips or failures
  (`generated/test_reports/reduce-r71-softmax`). Command:
  `python3 scripts/run_reduce_migration_sanity.py --factory F088 --factory F089`.
  That is SM013, SM014, SM047, SM048 and SM049.
- Full sanity collection at this checkpoint: **77 of 77 groups, 77 cases, zero
  failed groups** (`generated/test_reports/reduce-r71-collect`), plus the four
  architecture-template selections collected exactly once under
  `--tt-arch=blackhole` (`generated/test_reports/reduce-r71-collect-bh`).
  `sanity_test_collection.json` was rebuilt from those two runs. Its carried-over
  host-runner checks are labelled with their provenance: verified at
  `ba5eff432b7`, with the runner, plugin and gtest adapter unchanged since.
- Pre-commit and `git diff --check` passed for the manifest fix, the generator
  and all four regenerated artifacts.

Counts are unchanged by this round: **178 full groups, 1,000 definitions, 18,860
known cases; 77 sanity cases, 63 available on N300**; kernel-entry coverage
130/144, now reported consistently by all four artifacts.

A fresh review and the full prepared regression remain due. The hardware coverage
limits recorded in the prior review reports are unchanged: this host is a
two-chip Wormhole N300, and Quasar, Blackhole, Galaxy, T3K and 1x4-fabric
numerical coverage remains unavailable.
