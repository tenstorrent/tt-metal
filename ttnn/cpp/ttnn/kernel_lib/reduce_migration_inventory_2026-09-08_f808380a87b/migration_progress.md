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
- Shared Moreh layer/group normalization input gradients (S060/S061,
  DP005/DP006/DP010/DP011): both small and large paths now reduce transformed
  dy and y*dy blocks through host plans, replacing their manual tile-add loops.
  Factories account for bounded resident blocks and planned auxiliary formats;
  readers materialize the shared recipe. Retained masks around fused transforms.
  Native build and all four backward sanity cases passed (SM024/25/28/29).
  Full T053/T054 backward selections passed (86 passed, 75 upstream skips); results in
  `generated/test_reports/reduce-migration-qz31vai7/summary.json`.
- Shared Moreh layer/group normalization parameter gradients
  (S062, DP007/DP012). Both SUM reductions now use planned transformed blocks;
  the layer-norm path that only sums across batches retains its elementwise
  accumulation because it does not reduce inside tiles. Group-norm parameter
  reductions need FP32 destination and partial buffers to preserve cancellation
  over long HW reductions; an existing 500x500 case caught BF16 accumulation
  error and passes with the wider path.
- Corrected SM028 to width normalization with affine gradients: its old case
  reached the parameter factory but bypassed the actual reduce branch. The
  corrected case passed; results
  `generated/test_reports/reduce-migration-3dky0k8w/summary.json`.
- Added four layer-norm gradient boundary cases to T172. The full-precision
  elementwise assertion was too strict for the BF16 destination option: the
  legacy parameter kernel also fails it (maximum failing delta 1.8145 versus
  1.3659 after migration). Its standalone baseline run is preserved in
  `generated/test_reports/reduce-migration-efk7jb53/summary.json`. The legacy
  source was restored only for that comparison and then replaced by the
  migrated source. BF16 parameter gradients use a 2% relative L2 error bound;
  dx and FP32 parameter gradients retain elementwise checks. Wider L1-only
  partials and larger blocks did not improve BF16 results and were reverted.
  Native build and final T053/T054/T172 backward selections passed: 90 passed,
  75 upstream skips across 165 collected cases. Results:
  `generated/test_reports/reduce-migration-49t6sox0/summary.json`.
  Full runner additions now total seven definitions and 221 parameterized cases.

## Remaining work

- Migrate all remaining compute and dataflow helper callers in the inventory,
  with their factories and auxiliary allocations.
- Simplify obsolete manual tail masking and cross-tile add accumulation where
  supported by the planner, preserving fused operations and stream ordering.
- Exercise the sanity cases throughout and commit meaningful phases.
- Obtain satisfied fresh-context Claude Opus 5 reviews at high effort; record
  and address every review concern.
- Run the full prepared regression suite after review and resolve failures.

## MoE and sampling

Both factories now plan and serialize their MAX/SUM calls and the auxiliary tile recipes. Compute binds those calls to its existing intermediate buffers; dataflow materializes the planned auxiliary tiles. Top-k masking, temperature scaling, and the Tensix synchronization workaround remain in place.

Validation: `cmake --build build --target ttnn unit_tests_ttnn --parallel 8` passed. Sanity SM004/SM017 passed (2 cases; `reduce-migration-eikab75a`). Full groups T018/T019 passed all 28 cases (`reduce-migration-ihxzaarz`). Tests ran through `run_safe_pytest.sh` on Wormhole N300.

## Generic tiled, sharded, and row-major reductions

The W, H, and single-core HW factories now serialize their reduce calls and auxiliary recipes. The H readers stream independent columns in the order described by the plan. Dense row-major compute keeps its tilization and identity padding, then uses planned seed/repeat/final calls; per-tile consumption keeps circular-buffer pointers balanced across short chunks. Existing external post-scaling and raw fused-negate kernels are retained. Welford's shared readers receive an explicit auxiliary recipe.

Fixed the host planner's SFPU H output-slot calculation: reserve the work register before limiting by tensor width. Added 12 numerical regression cases for one/three columns, two batches, INT32/accurate FLOAT32, and SUM/MAX/MIN (T173).

Validation: native build passed (`cmake --build build --target ttnn unit_tests_ttnn --parallel 8`). All five generic sanity selections passed (`reduce-migration-peko34_v`). Full T003/T012/T160 completed 641 cases: **589 passed, 52 upstream skips**, no failures (`reduce-migration-2272a4vn`). All 12 new SFPU cases passed (`reduce-migration-u_niosd1`).

## DeepSeek grouped gate

The normalization SUM now uses a host-planned call with the selected-expert count as its logical width. The writer creates the plan's auxiliary tiles, including the partial-tile recipe. Existing gather, sorting, epsilon, and route scaling remain unchanged.

Validation: native build passed, SM018 passed (`reduce-migration-0f0kavqf`), and full T022 passed all 6 cases (`reduce-migration-_3qcre6w`).

## Attention and general softmax

Migrated the attention MAX/SUM calls, their five readers, and both attention factories. The large FP32 path uses planned cross-call accumulation; the BF16 path keeps each pass reduced from zero and combines pass results in an SFPU callback. Carrying a large BF16 sum through either native per-tile reduction or an unreduced cross-call add lost small contributions, caught by the wide softmax tests. Added an optional sequence-planner algorithm selection so this numerical requirement is planned together with its physical auxiliary recipe; extended the host regression and Python binding (T174).

Also migrated the four general H/W factories which reuse the Moreh compute/readers. The Falcon sanity test had an obsolete model configuration lookup; it now specifies the small sharded softmax configuration it exercises.

Validation: native build passed. All five attention sanity selections passed (`wfu3dlkz`, with corrected SM049 in `fvr4t6tk`). Full T034/T035/T046/T047/T048 completed **677 passed, 1 upstream skip** across 678 cases: general and ULP results `5hw4h99z`, interleaved/sharded nightly results `6ow2r_al`, and final wide/partial BF16 correction `7l8rdcz7`. The host planner regression T174 passed (`w961j4ud`). All device runs used the safe wrapper. The local Python binding was refreshed from `build/ttnn/_ttnn.so` after its signature changed.
- Output padding contract: AccumulateViaAdd now programs the reduced-output
  pack mask after finalization and callbacks, while intermediate accumulation
  tiles remain unmasked. The SFPU width mask clears the right faces explicitly
  on wide tiles. Native reduce_tile retains its existing mask setup. Added an
  exact-zero padding assertion to the existing complete helper matrix; T175
  passed all 111 cases (`reduce-migration-f40qn6dj`). The full runner now
  includes that matrix and the repeated-input-CB test (two definitions).
- Interleaved layernorm readers (DF037/DF038/DF039): host auxiliary recipes
  replace the old full/partial scaler calls. The raw compute reduction stays
  unchanged. SM007/SM008/SM046 passed (`reduce-migration-s80bxp5k`).
- Distributed LN/RMS pre/post all-gather (S087/S088/S090/S091,
  DF043/DF044/DF045): factories serialize calls for the actual logical widths
  and statistics divisors; compute uses reduce<Call>. The shared Welford reader
  bypasses auxiliary generation and its scratch buffer is a compute self-loop.
  All three sanity cases passed (`reduce-migration-s9pzm_81`). Full T037 passed
  142/142 including poisoned-padding and FP32 precision cases
  (`reduce-migration-6430fkj5`); T036 passed 99 with 102 upstream skips and T038
  passed four (`reduce-migration-fk3d5mea`, excluding its superseded T037 result).
- Sharded layernorm (S083/S085, DF040/DF041/DF042): host descriptors cover full
  and tail shards and preserve the existing cross-core reduction protocol.
  Explicitly reconfigure the BF16 epsilon operand after the now-FP32 auxiliary
  buffer; otherwise the old implicit format assumption corrupts variance.
  SM009/SM044/SM045 passed with temporary diagnostics removed
  (`reduce-migration-6f4vbtkm`). Distributed uneven/two-stage boundary selection
  T026 passed 24 cases (`reduce-migration-ugdiw030`).
- Groupnorm (S079/S080, DF035/DF036): all three factories now provide local and
  global calls plus auxiliary recipes. Full and tail blocks are independent
  local reductions; existing two-dimensional group/padding masks stay in place.
  Sharded mean reduction replaces the manual tile-times-one accumulation, and
  removes its ones CB. Variance retains its fused square-and-accumulate to avoid
  an additional full-group L1 allocation, followed by a planned HW reduction.
  SM005/SM006 passed (`reduce-migration-8yd95ysb`, `reduce-migration-6f4vbtkm`).
  T028 selected padding/configuration checks passed 113/113
  (`reduce-migration-14wrg17z`); T029 selected interleaved/padding checks passed
  8/8 (`reduce-migration-hn10c64d`). Native build command for all normalization
  changes: `cmake --build build --target ttnn unit_tests_ttnn --parallel 8`.
  Build logs: `/tmp/reduce-gn-interleaved-build-20260908.log` and
  `/tmp/reduce-gn-sharded-build-20260908.log`.
- Additional completed checks: T031 uneven, row-major, and two-stage sharded
  widths passed 46/46; together with T026 this is 70/70
  (`reduce-migration-ugdiw030`). T160/T170/T173 passed 48/48 across C++ reduction
  smoke tests, final-only callbacks, and narrow SFPU cases
  (`reduce-migration-5zis0zdd`).

## Fused distributed RMSNorm

The pre-all-gather kernel now squares each block and uses a planned reduction sequence instead of manually accumulating one L1 tile. The planner selects a common algorithm for full blocks and short tails. Post-all-gather uses a planned SUM with the full logical width/device divisor. Both readers materialize the host auxiliary recipes; epsilon remains BF16 independently of the statistics format.

Validation: native build passed. SM053 passed (`reduce-migration-7_26p8ry`). Full T070 passed **52 cases with 2 upstream skips** (`reduce-migration-lyl_rblk`), including all four odd-width cases and FP32 statistics/rope. An earlier forced-Add attempt rejected short tails at planning time; automatic sequence planning corrected those failures.

## RMS all-gather

The compute and writer now use separate planned local, first-stage, second-stage, and post-gather scalars. First-stage worker partials remain unscaled in the two-stage path; only the final worker normalizes their sum. The existing fused square accumulation and cross-core protocol remain intact.

Validation: native build passed (`/tmp/reduce-rms-allgather-build-v2-20260908.log`). SM069 passed (`reduce-migration-96kgms4k`). The earlier incorrect two-stage scalar assignment failed PCC and was corrected before this commit. Multi-device topologies beyond the local N300 remain unverified.

## Inline reduction examples

All helper-based inline examples now serialize host calls and auxiliary recipes. The examples retain their explicitly compared manual/FPU/SFPU benchmark variants. Compute fusion describes the aliased sharded input without consuming it and attaches the reciprocal callback to the planned reduction.

Validation: SM055/SM056/SM057 passed (`reduce-migration-96kgms4k`). Full T074/T075/T076 passed all 7 collected cases, including device performance variants (`reduce-migration-n4ve55eh`). These Python/inline-kernel changes required no native build; device JIT compilation ran through the safe wrapper.

## Toy variance and Python Metal2 descriptors

The interleaved variance implementation uses a compressed planned seed/repeat/final sequence with a separate accumulator buffer and a final-only sqrt callback. The sharded implementation describes its local aliased input and the centered-square buffer as independent planned reductions. Both readers use host auxiliary recipes. Exposed the existing KernelAdvancedOptions.compile_time_varargs field to Python so ProgramSpec factories can serialize descriptors.

Validation: native build passed (`/tmp/reduce-toy-variance-build-20260908.log`); SM058 and SM059 passed (`reduce-migration-9dm1kocp`). Full T079 remains scheduled to check wider blocks and BF16 accumulation.

## SDPA auxiliary recipes and DiT Welford scratch

All standard SDPA and decode writers now materialize explicit host auxiliary recipes. Their raw compute kernels retain their existing reduction algorithms. The experimental ring writer advances its fabric argument offset past the serialized recipe. Removed an obsolete reader-produced scaler from DiT Welford: that buffer is compute-owned transpose scratch.

Validation: native build passed (`/tmp/reduce-sdpa-auxiliary-build-20260908.log`). SM054/SM060/SM061/SM062 all passed (`reduce-migration-htyz2h8l`). Sparse Blackhole and large-ring topologies cannot be exercised on this N300.

## KDA and indexer score

Both KDA factories now plan their normalization reductions and auxiliary buffers. Recurrence keeps its independent all-ones matrix constant and gets a separate reduction auxiliary buffer. Indexer score's fallback MAX uses a planned call; its specialized batched MAX path remains intact. Both regular and ring factories serialize compatible compute/reader arguments, including the non-pooling configuration.

Validation: native builds passed (`/tmp/reduce-kda-build-20260908.log`, `/tmp/reduce-indexer-build-v2-20260908.log`). Device compilation and numerical behavior are unverified here because their tests require Blackhole.

## Remaining fused collective reductions

Attn-res gather softmax now binds a planned SUM for both local statistics reductions. DiT fused distributed RMSNorm uses planned local/post-gather calls and host auxiliary recipes. It retains fused square/L1 accumulation to fit its existing pipeline, and retains the packed-statistics add/transpose branch with its documented GMPOOL packer workaround.

Validation: native build passed (`/tmp/reduce-ccl-final-build-20260908.log`). Device checks remain unverified: attn-res requires a Blackhole 2x4 mesh and DiT fused tests require Galaxy.

## Quasar reductions and SDPA

Quasar W/H/HW and dense row-major factories now serialize planned calls and auxiliary recipes. The H reader streams one complete column per planned batch. Welford's shared readers receive a harmless explicit zero recipe, and the three Metal2 SDPA writers receive their identity recipe through varargs. Removed the Quasar sharded H reader, which has no factory references (the H factory rejects width sharding).

Validation: the native build compiled the Quasar factory objects and linked successfully (`/tmp/reduce-quasar-build-20260908.log`). On-device Quasar compilation/numerics cannot be verified on Wormhole. Full toy variance T079 also passed all 41 cases (`reduce-migration-if_f43wa`).
