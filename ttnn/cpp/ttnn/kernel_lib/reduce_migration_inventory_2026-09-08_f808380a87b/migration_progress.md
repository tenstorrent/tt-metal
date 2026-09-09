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
- The first fresh-context Claude Opus 5/high review started on 2026-09-08;
  its transcript is under `generated/reduce_migration_reviews/round_01_20260908`.
  An earlier availability check succeeded with
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

- Consumer migration is complete; continue targeted checks for review fixes.
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
  T026 passed 14 cases with 10 upstream skips (`reduce-migration-ugdiw030`).
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
  widths passed 46/46; together with T026 this is 60 passed and 10 upstream skips
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

Quasar W/H/HW and dense row-major factories now serialize planned calls and auxiliary recipes. The initial H adapter streamed one column per batch; review round 1 restores grouped column accumulators with the existing per-tile FIFO (see review_round_01.md). Welford's shared readers receive a harmless explicit zero recipe, and the three Metal2 SDPA writers receive their identity recipe through varargs. Removed the Quasar sharded H reader, which has no factory references (the H factory rejects width sharding).

Validation: the native build compiled the Quasar factory objects and linked successfully (`/tmp/reduce-quasar-build-20260908.log`). On-device Quasar compilation/numerics cannot be verified on Wormhole. Full toy variance T079 also passed all 41 cases (`reduce-migration-if_f43wa`).

## UDM and unused kernel retirement

The interleaved and both sharded UDM readers materialize host-provided FirstRow identity recipes. Their two C++ test builders serialize the descriptors. Removed four unused legacy compute files: RMSNorm pre-all-gather 2D and post-all-gather, and the superseded Moreh norm H/W copies. Active Moreh factories use the ord_other copies, and active RMSNorm factories use their Metal2 versions. The source-composition-only test now reads the active fused RMSNorm kernel. Its RMS/LN compilation case T176 passed (`reduce-migration-uieyhz42`).

Validation: native build passed (`/tmp/reduce-udm-bge-build-20260908.log`), including the UDM test builder objects. UDM's 1x4 fabric device tests cannot run on this two-chip N300. The original 218 candidate consumer/factory files now have only two unchanged host factories: DiT Welford and layernorm Welford, whose shared readers no longer consume the obsolete helper and whose scratch ownership remains valid.

## BGE model-local SDPA and final caller audit

The BGE writer now consumes an explicit host auxiliary recipe. Added typed Python constructors for auxiliary tile/plan records, so raw-compute Python factories can serialize their physical recipe directly. New T177 covers four combinations: standard/streaming compute and full/partially masked KV. These checks exposed two pre-existing model-local writer compile errors: an obsolete window-mask signature and an unconditional static_assert inside a discarded non-template branch. Both were corrected without enabling the gated KV alias mode.

Validation: all four T177 cases passed (`reduce-migration-lilo65k8`). Added its smallest case as SM075, closing the original DF001 coverage gap. Together with T176, the current full manifest adds 13 definitions / 350 cases beyond the original inventory; the current sanity manifest selects 75 cases. Earlier T177 collection and compile failures are superseded by this passing run. The new constructors built in `/tmp/reduce-udm-bge-build-20260908.log`, and the local extension was refreshed atomically before tests.

Final static scan: no calls to any of the three old dataflow helper APIs remain in ttnn, tests, or models; no old explicit compute reduce overload remains in consumer kernels. The legacy compute overload remains in the library as the backend of reduce<Call>, as requested. Five unreferenced kernel files were removed in total, including the Quasar sharded reader. The original inventory reports remain unchanged; the executable manifests and this journal record migration additions.

## Complete N300 sanity checkpoint

`python3 scripts/run_reduce_migration_sanity.py --lane common --lane wormhole --lane wormhole-n300` passed **all 61 selected cases**, with no skips/failures (`reduce-migration-6tq5gd6d`). The remaining 14 sanity cases require unavailable Quasar, Blackhole, T3K/Galaxy, or 1x4 fabric environments. Additional T072 `-k n300` passed all four combinations of fused residual addition and FP32 destination accumulation (`reduce-migration-l8ewz2id`). These are targeted/pre-review checks; the full prepared regression has not yet run.

The static caller audit at source commit `7d4229a0b76` is saved in `generated/reduce_migration_reviews/caller_audit_20260908_7d4229a0b76.json`. It accounts for all 220 candidates including the two optional library files, with five retired kernels and two unchanged Welford factories. It found no remaining old consumer calls.

## First Claude review and corrections

A fresh Claude Opus 5 session at high effort reviewed the complete branch and
returned CHANGES REQUIRED. The findings and their resolutions are recorded in
`review_round_01.md`; the original review is preserved under
`generated/reduce_migration_reviews/round_01_20260908/`.

Fixed runtime installation of three device headers, H reduction reader/compute
grouping for full-sync and sharded negate paths, explicit format-reconfiguration
modes, and worker-role flags for auxiliary selection. Added host shape contracts,
hoisted output-mask setup, corrected the unused row-mask alias, and restored the
Falcon test's BF8 attention mask. T178 adds 42 cases spanning column grouping,
full-sync, partial heights, repeated width-sharded batches, and float/integer paths.
The full manifest now contains 178 groups, 998 definitions, and 18,844 known cases.

Validation: native build passed; an isolated runtime install contains the three
headers. The complete available sanity suite passed **61/61**
(`reduce-migration-mvkrxglb`). Targeted review checks passed **219 cases**, with
**10 upstream skips**: T178 (42), T175 (111), T026/T031 (60 plus 10 skips),
T072 (4), T152 (1), and T176 (1). Corrected earlier journal entries that had
mistakenly counted the T026 skips as passes.

Blackhole mock JIT checks compiled KDA, indexer, sparse SDPA, the helper matrix,
attn-res, and DiT fused kernels without compiler errors. These checks provide
compilation evidence only. Quasar mock attempts stopped before compilation on
existing setup/factory restrictions; Quasar and non-N300 numerical coverage
remain unavailable. Exact logs and mock-plugin limitations are in the review
resolution report. A fresh second review and the full regression run remain due.

## Groupnorm format follow-up — 2026-09-09

Additional sharded groupnorm checks exposed an error in the first review's
blanket native-NONE correction: the new local mean follows masking and needs to
restore its intermediate/auxiliary unpack formats. Small groups selected native
reduction and failed, while Add groups configured their operands and passed.
The host now requests INPUT for that first native call. See
`review_followup_groupnorm_2026-09-09.md` for the diagnosis and exact command.

At `1005cb6d975` the existing T028 selection had 25 failures and 26 passes. After
the fix, the same 51 cases passed with no changes to their checks or tolerances
(`reduce-migration-n9526jxz`). Native build passed
(`/tmp/reduce-groupnorm-format-fix-build-20260909.log`). SM005/SM006 passed
(`reduce-migration-n1bwzklv`). SM006 now uses the small BF8-mask/FP32-destination
case that catches the regression; the original C++ case remains in the full
suite. Sanity remains 75 cases, now 60 Python and 15 C++.

Review round 2 was interrupted without a verdict; round 3 was stopped when these
independent tests exposed the regression. Both incomplete transcripts are
preserved. Another fresh Opus 5/high review and the full regression remain due.

Additional Quasar evidence: a real one-chip Wormhole run of the existing ResNet
global-pooling test passed through the migrated Quasar H factory (49 logical
spatial values, 2048 channels). Its first two-chip attempt failed only at tensor
readback because the test supplies no mesh composer. A temporary one-chip fixture
selection resolved that test setup issue; no numerical check changed. See
`generated/reduce_migration_reviews/quasar_h_wormhole_validation_20260908.md`.
Quasar mock host planning also succeeded, but the runtime explicitly bypasses JIT
compilation on Quasar mock devices (`tt_metal/impl/program/program.cpp`); that
mock result supplies no Quasar compilation or numerical evidence.

Post-fix complete N300 sanity: **61/61 passed**, with no skips or failures
(`reduce-migration-iy7ny4u1`, source commit `11d81c6fdea`). The JUnit results agree
with the runner's counts. A fresh Opus 5/high review (round 4, session
`4b52d738-f23e-4c1c-baed-6093ea8f9392`) is running; the full regression remains due.
Clarified the SM006 description: its BF16 input uses BF16 L1 intermediates and
FP32 destination accumulation; the BF8-mask transition is what the test guards.

## Fourth Claude review and corrections

Fresh Opus 5/high round 4 completed and returned CHANGES REQUIRED. It confirmed
migration coverage, the earlier fixes, the groupnorm correction and the passing
N300 checkpoint, then requested two additional fixes. SFPU reduction now sets
its invariant output mask once per call, and Moreh dot requests INPUT so its
planned auxiliary format does not depend on the previous multiply's unpack
state. Addressed all four optional notes: corrected reader comments, removed
unused reader locals without changing argument positions, distinguished the host
softmax header name, and logged parameter-gradient maximum absolute errors.
See `review_round_04.md` for the complete resolution table and original review.

Native build and pre-commit checks passed. T052/T173/T175/T178 passed **186 cases
with 4 upstream BF8 skips** (`reduce-migration-kuz604bw`); all four gradient
boundary checks passed (`reduce-migration-kw3an9hl`). The complete available
sanity suite then passed **61/61**, without skips or failures
(`reduce-migration-y7lizu94`). Test bodies retain all numerical checks and
thresholds. Another fresh review and the full 178-group regression remain due.

## Fifth Claude review: Welford shared-reader correction

Fresh Opus 5/high round 5 verified the earlier corrections and found one missed
host/kernel contract. The distributed post-all-gather reader always decoded an
auxiliary recipe, but the Welford factory had not supplied one. The existing
explicitly enabled disabled C++ case reproduced the reader's zero-tile-count
compile assertion (`reduce-migration-6w24bcee`). The Welford factory now defines
USE_WELFORD and the shared reader skips its unused auxiliary initialization,
matching the pre-all-gather path. Removed the dead reader scalar argument from
both post factories. This corrects the earlier audit's Welford exemption: only
the DiT Welford host factory remains unchanged among the inventory candidates.

Eight new single-device post-Welford tests construct the required mean/variance
statistics directly, check numerical output twice and assert one program-cache
entry. All eight passed (`reduce-migration-yedaglif`). Existing T036 cases passed
99 with 102 upstream skips (`reduce-migration-5fcqrnv3`), and both pre-Welford
checks passed all 10 cases (`reduce-migration-1y4ikhq1`). Native build passed in
`/tmp/reduce-review5-welford-build-v3-20260909.log`; its earlier cleanup build's
unused-variable error was fixed before these tests. The expanded N300 sanity
then passed **62/62**, with no skips or failures (`reduce-migration-839b7bv3`).

SM076 adds the shared reader's Welford variant; SM011 now accurately credits its
non-Welford factories. Current counts: **178 full groups, 999 definitions, 18,852
known cases; 76 sanity cases (61 Python/15 C++), 62 available on N300**. Kernel
entry coverage is unchanged. See `review_round_05.md` for all resolutions,
including the documented pre-existing RMSNorm 2D dispatch follow-up and the
unverified DiT Welford numerical impact. No existing test tolerance or skip was
weakened. A fresh review and the full prepared regression remain due.

## Sixth Claude review: mixed statistics dtype

Fresh Opus 5/high round 6 verified the Welford reader correction and identified
one missed format condition in the standard post-all-gather factory. RMSNorm's
new auxiliary format follows the statistics tensor, while its unpack-mode gate
still followed the input tensor. BF16 input with FP32 statistics consequently
failed program-spec validation. An independent N300 reproduction at the clean
review checkpoint produced 6 passes and 2 such failures; evidence is preserved
under `generated/reduce_migration_reviews/mixed_stats_repro_20260909/`.

The unpack mode now follows the actual auxiliary format. Added eight T036 cases
covering both mixed BF16/FP32 directions, both norms and one/four statistics
pairs, without changing the helper's default dtype or numerical checks. SM077
adds the smallest failing RMSNorm configuration. Also removed the Welford
factory's dead auxiliary allocation and stale comments, enforced its LayerNorm
contract locally and removed the unreachable RMSNorm compute selection.

The review's run-twice documentation concern was a false positive: the imported
wrapper in `utility_functions.py:144` calls `_run_twice` at line 80, which executes
the op twice and asserts exact output equality at line 83. See
`review_round_06.md` for all resolutions and deferred pre-existing observations.

Native build and pre-commit passed. All eight new mixed-dtype tests passed
(`reduce-migration-uusolvow`), followed by complete T036/T038/T159 checks:
**152 passed, 106 upstream skips**, with no failures/errors
(`reduce-migration-ybtk53_l`). Current manifests contain **178 full groups, 1,000
definitions, 18,860 known cases; 77 sanity cases, 63 available on N300**. Kernel
entry coverage remains 130/144. The unchanged original reproduction then passed
8/8, including no-weight cases and both same-dtype controls. The expanded N300
sanity passed **63/63** without skips/failures (`reduce-migration-5txgugv1`).
JUnit agrees with every post-fix result. A fresh review and the full prepared
regression remain due.

## Seventh Claude review: module manifest and derived reports

Fresh Opus 5/high round 7 verified R6.1 as correct and complete and found one
build-system regression: `sources.cmake` still listed the pre-rename
`softmax/device/softmax_reduce.hpp` in the exported API header set, so a clean
configure and every `cmake --install`/packaging build failed. Only incremental
ninja builds in a tree configured before the rename were recorded, which could
not detect it. Both failures were reproduced, then fixed by listing
`softmax_reduce_plans.hpp` in `TTNN_OP_NORMALIZATION_SRCS`, matching its
groupnorm sibling.

The round also exposed drift between the sanity manifest and its three derived
reports, in both directions: md/html/csv still reported 74 cases / 129 covered
kernels and `DF001` as a gap, while the manifest's own `kernels[]` copies held
superseded `SM006`/`SM022`/`SM023`/`SM028` selections that the CSV had current.
`scripts/generate_reduce_migration_sanity_reports.py` now projects md, html and
csv from the manifest and refuses to run on internal disagreement, so the class
of drift is closed. Sixteen stale manifest fields and a `"; "` evidence prefix on
69 kernel entries were corrected from the authoritative group entries; the six
resulting CSV row changes are all corrections. See `review_round_07.md`.

Configure, install, native build, pre-commit and `git diff --check` all passed.
The five sanity groups for the two factories including the renamed header passed
**5/5** (`reduce-r71-softmax`), and full sanity collection passed **77/77 groups
with zero failures** (`reduce-r71-collect`), plus four architecture-template
selections under `--tt-arch=blackhole` (`reduce-r71-collect-bh`). Counts are
unchanged: **178 full groups, 1,000 definitions, 18,860 known cases; 77 sanity
cases, 63 available on N300**; kernel-entry coverage 130/144, now consistent
across all four artifacts. A fresh review and the full prepared regression
remain due.

## Eighth Claude review: SATISFIED

Fresh Opus 5/high round 8 reviewed `f90f03425d8` and returned **SATISFIED** with
no required changes. It verified R7.1 from the regenerated install script, not
just the source list, and confirmed the build tree is now configured after the
rename, closing the blind spot that hid it. Build/install/packaging correctness
was added to the review scope and found no second defect across every
`sources.cmake` under `ttnn/` and `tt_metal/`, all six added files, all five
deletions and every include in the 171 touched kernel files. The regenerated
sanity reports reproduce byte-identically from the committed manifest.

Addressed its non-blocking items: `unit_test_suite.md`/`.html` now point at the
`migration_regressions` delta (16 definitions, 408 cases) and name the JSON as
authoritative; the generator now validates the manifest's `counting` block and
group/kernel membership. The `SM076`/`SM077` membership asymmetry was a mislabel
rather than a data defect — those groups do exercise kernels whose primary case
is `SM011` — so the reports now say "Kernels" and mark non-primary entries.
Left open deliberately: regenerating the full-suite report bodies, and the stale
ignored `build_Release/libexec/` install tree. See `review_round_08.md`.

The review loop is complete. The full prepared regression — 178 groups, 1,000
definitions, 18,860 known cases — is the last outstanding requirement.
