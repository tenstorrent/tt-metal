# Claude review round 6 — 2026-09-09

Fresh Claude Opus 5 / high-effort session `193f4621-cbb7-45e5-ac28-50b0c0a821f2`
reviewed `3911616a707` and returned **CHANGES REQUIRED**. The complete review is
preserved in `generated/reduce_migration_reviews/round_06_20260909/review.md`.
It verified R5.1, repeated the inverse factory/reader audit, and found one missing
unpack-mode condition for independently typed input and statistics tensors.

## R6.1: RMSNorm post-all-gather auxiliary format

The planned RMSNorm auxiliary dtype follows the statistics tensor. The factory
still configured `POST_REDUCE`'s unpack mode only when the original input was
FP32. Consequently BF16 input with FP32 statistics produced an FP32 auxiliary
buffer without the explicit unpack mode required by the program-spec validator.

Reproduced on N300 at the unchanged review checkpoint: **6 passed, 2 failed**
across four BF16/FP32 input/statistics dtype combinations and one/four statistics
tiles. Only BF16 input / FP32 statistics failed, with:

> Compute kernel 'post_compute' consumes FP32 DFB 'post_reduce' with enable_32_bit_dest=true, but provides no unpack_modes entry for this DFB.

Source, exact command and JUnit are preserved under
`generated/reduce_migration_reviews/mixed_stats_repro_20260909/`; console log:
`/tmp/reduce-review6-mixed-stats-repro-20260909.log`.

The factory now gates `POST_REDUCE`'s `UnpackToSrc` mode on its actual
`scaler_data_format`, separately from `POST_INPUT`. The standard LayerNorm branch
continues to follow its own input-derived scalar format.

Added an optional statistics dtype to the existing T036 helper without changing
its default. Eight new cases cover both mixed BF16/FP32 directions, RMSNorm and
LayerNorm, and one/four statistics pairs. They retain the existing numerical
bound and determinism wrapper and assert one program-cache entry. SM077 selects
the smallest BF16-input/FP32-statistics RMSNorm case; SM011 retains the pre/post
pipeline. Current counts: **178 full groups / 1,000 definitions / 18,860 known
cases; 77 sanity cases, including 63 available on N300**. Kernel-entry coverage
is unchanged.

## Other review notes

- Removed the stale logical-width scaler comment from the Welford post factory.
- Removed its unused auxiliary buffer, declaration, sizing constant and reader
  self-loop. `USE_WELFORD` removes every reader reference to `dfb::reduce`, and
  the compute kernel never bound that buffer.
- The Welford factory now explicitly requires LayerNorm and selects only its
  Welford compute kernel. Removed the unreachable RMSNorm branches and associated
  unpack-mode comments. Public validation already rejected RMSNorm+Welford.
- The review's run-twice documentation concern is a false positive. The test
  imports `ttnn_layer_norm_post_all_gather` from `utility_functions.py`; that
  function at line 144 delegates to `_run_twice` at line 80, which calls the op
  twice and asserts `torch.equal` on both outputs. The RMSNorm wrapper at line 152
  does the same. The repeatability statements in the journal and round-5 report
  were correct; this source reference makes the indirect assertion explicit.
- Deferred pre-existing behavior: the Welford post reader uses padded `Wt` while
  the compute kernel uses `tiles_per_core_y` in the 2D path, unchanged from the
  baseline. The separately documented RMSNorm 2D dispatch issue also remains
  outside this migration. The unused `wh_generate_reduce_scaler` had no callers
  at either baseline or HEAD. These observations do not identify new migration
  regressions.

## Validation

- Native build passed: `cmake --build build --target ttnn unit_tests_ttnn --parallel 8`.
  Log: `/tmp/reduce-review6-fixes-build-20260909.log`; refreshed the runtime
  extension from this successful build before tests.
- Pre-commit passed for both factories, the Python test and both manifests.
- All eight new mixed-dtype cases passed (`reduce-migration-uusolvow`). Command:
  `python3 scripts/run_reduce_migration_tests.py --group T036 -- -k test_post_all_gather_mixed_stats_dtype_with_program_cache`.
- Complete T036/T038/T159 checks passed **152 cases with 106 upstream skips**
  (`reduce-migration-ybtk53_l`). This includes every existing/new T036 case,
  all four RMSNorm pipeline cases and all 33 enabled C++ normalization cases.
  Command: `python3 scripts/run_reduce_migration_tests.py --group T036 --group T038 --group T159`.
  The skips are 102 existing Python dtype/L1 guards and four disabled C++ cases;
  none was added or changed for this migration.
- The original reproduction also passed **8/8** after the fix, with no changes
  to its source. Post-fix evidence: `mixed_stats_repro_20260909/results_after_fix.xml`
  and `/tmp/reduce-review6-mixed-stats-after-fix-20260909.log`. This additionally
  covers the no-weight path and both same-dtype controls.
- The expanded N300 sanity passed **63/63**, with no failures, errors or skips
  (`reduce-migration-5txgugv1`), including SM076 and SM077. Command:
  `python3 scripts/run_reduce_migration_sanity.py --lane common --lane wormhole --lane wormhole-n300`.
  JUnit agrees with the runner's counts for every post-fix check.

A fresh review and the full prepared regression remain required. The hardware
coverage limits recorded in the prior review reports remain unchanged.
