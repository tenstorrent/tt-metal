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
- Claude reviews have not started yet. An availability check succeeded with
  `claude --model claude-opus-5 --effort high --no-session-persistence`; the
  result's model usage confirms `claude-opus-5`. This was not a code review.

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
- Corrected SM033/SM034 to use `p=0`: the old `p=2.5` cases dispatched to
  abs-pow and sum, bypassing the claimed norm factories. The two corrected
  cases passed with migrated kernels; results
  `generated/test_reports/reduce-migration-a76yepiy/summary.json`.
- Full Moreh mean/sum groups T056/T058 passed: 197 passed, 133 upstream skips
  across 330 collected cases. Results:
  `generated/test_reports/reduce-migration-4d4e9vf_/summary.json`.
- Moreh norm H/W (S068/S069, DP014/DP015): transformed inputs now use planned
  reductions and final-only fused negation. Bounded resident blocks keep a
  full block with the partial tail so all calls can use AccumulateViaAdd;
  this avoids the BF16 precision loss found with a tiny last ReduceTile call.
  Removed manual tile-add/max accumulation and mask/reduced-result buffers.
  Native build, both corrected sanity cases, and 60 new block-boundary cases
  passed. Log: `/tmp/reduce-moreh-norm-blocks-20260908-r2.log`.
- Added the 60 norm regressions and nine planned callback checks to the full
  manifest (T057/T170): three additional definitions and 69 cases beyond the
  original base inventory's count.
- Moreh vector/scalar bias gradients (S063/S064, DF026/DP013): migrated both
  factories, readers, and compute kernels. The vector path combines complete
  batches and handles each partial height through a planned call. The scalar
  HW path retains its two-dimensional input mask, which the current one-axis
  partial recipes cannot represent. Both use planned accumulation descriptors.
  Native build and SM030/SM031 passed. Full group T055 passed 173 enabled cases
  with 168 upstream skips (341 collected); results
  `generated/test_reports/reduce-migration-15umv7i2/summary.json`.
- Moreh softmax forward/backward H/W, small/large (S070–S077, DF028–DF033,
  DP016/DP017): all eight factories now pass host plans and auxiliary recipes.
  Forward MAX covers the full logical axis in one call. Large forward sums and
  non-log gradients use bounded transformed blocks and planned accumulation;
  large log-softmax gradients reduce the incoming dy stream directly. Removed
  obsolete mask/staging buffers; small forward kernels retain their output
  padding mask. Native build and all eight sanity cases passed.
  Full groups T059–T062 passed 367 cases with 96 upstream BF8 skips (463 total),
  including all 162 existing ULP cases. Results:
  `generated/test_reports/reduce-migration-o97h1hu_/summary.json`.
- Added T171 with 108 forward/backward boundary cases covering H/W, small/large,
  softmax/softmin/log-softmax, partial padding poisoned with 42, block transitions,
  repeated accumulation and BF16/FP32 destination formats. The sweep uses SMALL
  shapes within its existing 512-KB gate and retains 1025-element cases for LARGE.
  Its PyTorch autograd reference uses BF16 outputs matching the device inputs;
  near-zero log-softmax gradients allow one BF16 ULP at unit intermediate scale.
  All 108 passed; results in
  `generated/test_reports/reduce-migration-h6zf0w0c/summary.json`.
- Moreh gradient clipping step 1 (S056, DP002): per-input host plans describe
  bounded transformed blocks and cross-call accumulation, replacing the manual
  tile-add loop. Retained the two-dimensional source mask around the power
  transform. Native build, SM020, and all 20 full T051 cases passed. Results:
  `generated/test_reports/reduce-migration-h6zf0w0c/summary.json`.
- Shared Moreh layer/group normalization forward (S058/S059, DP003/DP004,
  DP008/DP009): mean and variance now use bounded planned reductions, replacing
  the handwritten tile-add loops. W tails use planner masks; HW retains the
  two-dimensional source masks. Removed the squared-tile scratch buffer.
  Native build and all four sanity cases passed. Existing layer-norm forward
  T054 cases passed (22 passed, 23 upstream skips); results in
  `generated/test_reports/reduce-migration-0br1k4us/summary.json`.
- Added T172 direct normalization boundary coverage: 24 group-norm and 16
  layer-norm cases, all passed. These check output, mean and reciprocal standard
  deviation with poisoned padding, small/large paths, affine parameters, and
  both destination accumulation formats. They exposed two issues fixed here:
  group-norm block sizes exceeded FP32 destination capacity; width-reduction
  statistics need a full tile transpose before the writer consumes them.
  Results: `generated/test_reports/reduce-migration-7tbxlkvi/summary.json`
  and `/tmp/reduce-moreh-layer-norm-boundaries-20260908-r2.log`.
- Replaced SM022/SM023's unconditionally skipped group-norm tests with exact
  cases from T172. Both now execute and pass; results in
  `generated/test_reports/reduce-migration-yj0k8o8w/summary.json`. Original
  full-suite skips remain visible. The full runner now contains six added test
  definitions and 217 added parameterized cases beyond the base inventory.

## Remaining work

- Migrate all remaining compute and dataflow helper callers in the inventory,
  with their factories and auxiliary allocations.
- Simplify obsolete manual tail masking and cross-tile add accumulation where
  supported by the planner, preserving fused operations and stream ordering.
- Exercise the sanity cases throughout and commit meaningful phases.
- Obtain satisfied fresh-context Claude Opus 5 reviews at high effort; record
  and address every review concern.
- Run the full prepared regression suite after review and resolve failures.
