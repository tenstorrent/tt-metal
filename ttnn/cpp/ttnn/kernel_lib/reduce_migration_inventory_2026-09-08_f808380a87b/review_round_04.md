# Claude review round 4 — 2026-09-09

Fresh Claude Opus 5 / high-effort session `4b52d738-f23e-4c1c-baed-6093ea8f9392`
reviewed production commit `11d81c6fdea` against `ba5eff432b7` and returned
**CHANGES REQUIRED**. The complete original review is preserved in
`generated/reduce_migration_reviews/round_04_20260909/review.md`.

The reviewer confirmed migration coverage, the round-1 fixes, the groupnorm
follow-up, unchanged upstream tolerances, and the completed 61/61 N300 sanity
results. It required two further corrections and raised four optional notes.

## Corrections

| Finding | Resolution |
| --- | --- |
| R4.1: SFPU reduction configured the output mask inside each output loop. | Move the invariant setup immediately after the auxiliary wait, before the dimension dispatch. The existing final clear remains. Every pack in the call targets the same output buffer; no performance improvement is claimed without measurements. |
| R4.2: Moreh dot selected NONE although BF8 input and its new BF16 auxiliary have different formats. | Select INPUT for every dot call, letting the helper configure the correct operand order before native W reduction. Multiplication still restores both original input formats before the next tile. |
| O1: Reader comments described the obsolete independent-column plan and Welford flag. | Describe the planned DEST column groups and explicit reduce_output_tiles argument. Remove unused Quasar reader locals while preserving serialized argument positions. |
| O2: Host and device softmax headers shared a basename. | Rename the host header to softmax_reduce_plans.hpp and update both host includes. The installed device header keeps its name. |
| O3: Quasar's positional reader hardcoded one column without documenting its caller restriction. | State that only the Quasar Welford factory uses this reader, requiring a single running mean/M2 state. |
| O4: FP32 parameter-gradient bounds lacked reported absolute error. | Log maximum absolute error separately for dgamma and dbeta, together with the destination-accumulation mode. All assertions and tolerances remain unchanged. |

## Validation

- Pre-commit checks passed for all changed files.
- Native build passed: `cmake --build build --target ttnn unit_tests_ttnn --parallel 8`.
  Log: `/tmp/reduce-review4-fixes-build-20260909.log`. Refreshed the runtime
  Python extension atomically after completion.
- Requested T052/T173/T175/T178 checks passed: **186 passed, 4 upstream skips**
  (`reduce-migration-kuz604bw`). Counts: dot 21 passed/4 BF8 skips, narrow SFPU
  12 passed, complete helper matrix 111 passed, height grouping 42 passed.
  Command: `python3 scripts/run_reduce_migration_tests.py --group T052 --group T173 --group T175 --group T178`.
- All four Moreh layernorm backward boundary cases passed unchanged
  (`reduce-migration-kw3an9hl`). The log now records FP32-destination maximum
  absolute errors of 0.4824/0.1770 and 0.5477/0.2993 for dgamma/dbeta at the two
  shapes. BF16 retains the previously accepted relative-L2 check.
  Command: `python3 scripts/run_reduce_migration_tests.py --group T172 -- -k test_moreh_layer_norm_backward_reduce_boundaries`.
- The complete available N300 sanity suite passed **61/61**, with no skips or
  failures (`reduce-migration-y7lizu94`). Command:
  `python3 scripts/run_reduce_migration_sanity.py --lane common --lane wormhole --lane wormhole-n300`.
  Runner counts agree with JUnit for all recorded checks.
- Another fresh review and the full prepared regression remain due. The existing
  BF8 Moreh dot cases are skipped upstream; T052 does not prove BF8 numerics.
- Blackhole, Quasar and larger-mesh hardware limitations remain as recorded in
  the previous review reports; no mock result is counted as a numerical pass.
