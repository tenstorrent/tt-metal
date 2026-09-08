# Reduce migration progress

Migration base: `f808380a87b320e24457c600cc79b05d7a0b8f73`.
Branch: `malimpic/migrate-to-host-reduce-helpers`.

The original inventory and test manifests describe the base revision. This
separate journal records implementation and validation without rewriting that
audit.

## Environment and baseline

- Committed the inventory and regression runners in `ba5eff432b7`.
- `.github/scripts/copilot-build.sh --configure-only` could not run because
  Docker is unavailable. No dependencies were installed.
- The native toolchain is available. Baseline
  `cmake --build build --target ttnn unit_tests_ttnn --parallel 8` passed.
- `tt-smi -ls` reports an N300 with two Wormhole chips. Common, Wormhole, and
  N300 test lanes can run here. Other architecture/topology lanes still need
  their supported environments; their results must be reported separately.
- Claude reviews have not started yet.

## Implementation checkpoints

- SSM single-tile sum (S052, DF021): migrated the factory's compute descriptor,
  reader's auxiliary recipe, and compute call. The native host build passed.
  `python3 scripts/run_reduce_migration_sanity.py --group SM019` passed (1/1),
  with fresh device compilation; results in
  `generated/test_reports/reduce-migration-e0axez3j/summary.json`.
- Added Metal 2 buffer binding support without changing planner-selected call
  behavior, and optional fused post operations to `reduce<Call>`.
- Moreh dot (S057, DF025): host-planned seed/middle/final calls handle the last
  partial tile; removed the reader's obsolete manual mask. SM021 passed (1/1).
  Full module T052 passed 21 enabled cases; four upstream BF8 cases skipped.
  Results: `generated/test_reports/reduce-migration-npvcsg59/summary.json`.
- Planned fused post-operation validation: the existing average/post-op test
  passed, plus eight new cases testing final-only callbacks across three-call
  sequences with both algorithms, H/W dimensions, and BF16/FP32 accumulation.
  Logs: `/tmp/reduce-planned-post-op-20260908-r2.log` and
  `/tmp/reduce-planned-final-post-op-20260908-r2.log`.
  This exposed and fixed a pre-existing example bug: a batched column output's
  logical planning width differs from its physical one-core shard width.
- Moreh height mean/sum (S065/S078, DF027/DF034): one planned call replaces full
  tiles plus masked-tail accumulation. Removed each factory's mask, accumulator,
  and masked-input buffers. Native build and SM032/SM043 passed (2/2), results
  `generated/test_reports/reduce-migration-bgr8odj8/summary.json`.
- Sanity selection correction needed: SM033/SM034 used `p=2.5`, which dispatches
  to abs-pow and sum, bypassing the claimed norm factories. Use `p=0` instead.

## Remaining work

- Migrate all remaining compute and dataflow helper callers in the inventory,
  with their factories and auxiliary allocations.
- Simplify obsolete manual tail masking and cross-tile add accumulation where
  supported by the planner, preserving fused operations and stream ordering.
- Exercise the sanity cases throughout and commit meaningful phases.
- Obtain satisfied fresh-context Claude Opus 5 reviews at high effort; record
  and address every review concern.
- Run the full prepared regression suite after review and resolve failures.
