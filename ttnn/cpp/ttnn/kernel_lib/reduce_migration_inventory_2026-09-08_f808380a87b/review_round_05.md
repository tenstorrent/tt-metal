# Claude review round 5 — 2026-09-09

Fresh Claude Opus 5 / high-effort session `ada77aba-169c-46d8-b497-1d0bcfd814d3`
reviewed commit `97689309d9d` and returned **CHANGES REQUIRED**. The original
review is preserved in `generated/reduce_migration_reviews/round_05_20260909/review.md`.
It verified both round-4 fixes, all four cleanups and the completed 61-case
sanity run, then found one missed shared-reader factory during its inverse
caller audit.

## R5.1: distributed Welford post-all-gather reader

The shared post-all-gather reader decoded `ReduceAuxiliaryArgs<0>` unconditionally.
The standard factory supplied the recipe, but the Welford factory supplied only
named arguments and tensor accessors. Consequently the reader decoded a tensor
accessor's first word as the auxiliary header and failed the nonzero-tile-count
static assertion. The compute kernel's lack of a scaler dependency did not make
this reader instantiation safe.

Reproduced at `97689309d9d` with the existing disabled C++ test, explicitly opted
in through the safe runner:

```
python3 scripts/run_reduce_migration_tests.py --group T159 --include-disabled-gtests -- -k DistributedLayerNormPostWelford
```

`reduce-migration-6w24bcee/T159.log` reports the NCRISC compile failure at
`reduce_plan_args.hpp:105`: `Reduction auxiliary recipe must contain at least one tile`.
This is separate from the test's documented upstream numerical problem.

The Welford reader now defines `USE_WELFORD`, and the shared reader guards the
unused auxiliary initialization with `#ifndef USE_WELFORD`, matching the pre
path. Its unused self-loop binding remains. Removed the obsolete named
`reduce_factor` reader argument from both factories and its reader declaration;
the standard factory still computes its host-planned scalar normally.

Added a direct single-device numerical/cache test with correctly constructed
[mean, variance] statistics. It covers BF16/FP32, one/four statistics pairs, and
small/batched tensors: eight cases in T036. SM076 selects the smallest case to
cover this shared reader's Welford factory variant. SM011's non-Welford RMS test
no longer claims F086. The full manifest is now **178 groups / 999 definitions /
18,852 known cases**; sanity is **76 cases / 62 N300-applicable**, with unchanged
kernel-entry coverage.

## Other review notes

- Removed the tautological second clause from groupnorm's final-row contract;
  retained `last_rows <= normal_rows`.
- Documented in the FP32 gradient test that both relative and absolute terms
  apply, with a fixed seed and logged maxima. No bound or assertion changed.
- PR note for the earlier DiT Welford change: the reader previously pushed a
  scaler into compute-owned transpose scratch. Removing that push changes the
  data consumed by the Welford calculation, correcting the ownership conflict
  identified from source. The resulting numerical behavior on Galaxy remains
  **unverified**; it must not be described as a measured numerical improvement.
- Deferred pre-existing follow-up: RMSNorm with `use_2d_core_grid=True` selects
  the layernorm 2D kernel via `layernorm_pre_all_gather_device_operation.cpp` and
  `layernorm_pre_all_gather_program_factory.cpp`. The retired RMSNorm 2D file was
  unreferenced already at the baseline. Its deletion does not introduce this
  dispatch behavior; an explicit validation guard or active RMSNorm 2D dispatch
  belongs in the follow-up identified by the reviewer.

## Validation

- Native build passed: `cmake --build build --target ttnn unit_tests_ttnn --parallel 8`.
  Final log: `/tmp/reduce-review5-welford-build-v3-20260909.log`. The v2 cleanup
  build caught an unused `logical_W` variable, which was removed before this
  successful rebuild. Refreshed the runtime extension from the successful build.
- All eight new Welford post-all-gather cases passed, including the explicit
  one-entry program-cache assertion (`reduce-migration-yedaglif`).
  Command: `python3 scripts/run_reduce_migration_tests.py --group T036 -- -k test_layer_norm_post_all_gather_welford_with_program_cache`.
- The existing T036 cases passed: **99 passed, 102 upstream skips**
  (`reduce-migration-5fcqrnv3`). The new eight cases had already passed separately;
  together these cover all 209 current T036 cases. Command:
  `python3 scripts/run_reduce_migration_tests.py --group T036 -- -k 'not test_layer_norm_post_all_gather_welford_with_program_cache'`.
  Skips preserve the existing FP32 destination, gamma/beta dtype and L1-capacity
  guards; none were added for this migration.
- Both existing Welford pre-all-gather tests passed all **10 cases**
  (`reduce-migration-1y4ikhq1`). Command:
  `python3 scripts/run_reduce_migration_tests.py --group T037 -- -k 'test_layernorm_pre_all_gather_welford_residual or test_layernorm_pre_all_gather_welford_fp32_precision'`.
- The expanded N300 sanity suite passed **62/62**, without failures, errors or
  skips (`reduce-migration-839b7bv3`), including SM076. Command:
  `python3 scripts/run_reduce_migration_sanity.py --lane common --lane wormhole --lane wormhole-n300`.
  JUnit agrees with the runner's counts for all post-fix checks.
A fresh review and the full prepared regression remain required. Hardware
limitations remain unchanged.
