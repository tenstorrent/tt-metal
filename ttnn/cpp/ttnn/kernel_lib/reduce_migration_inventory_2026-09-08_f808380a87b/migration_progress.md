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
- In progress: Metal 2 buffer binding support and planned fused post operations.

## Remaining work

- Migrate all remaining compute and dataflow helper callers in the inventory,
  with their factories and auxiliary allocations.
- Simplify obsolete manual tail masking and cross-tile add accumulation where
  supported by the planner, preserving fused operations and stream ordering.
- Exercise the sanity cases throughout and commit meaningful phases.
- Obtain satisfied fresh-context Claude Opus 5 reviews at high effort; record
  and address every review concern.
- Run the full prepared regression suite after review and resolve failures.
