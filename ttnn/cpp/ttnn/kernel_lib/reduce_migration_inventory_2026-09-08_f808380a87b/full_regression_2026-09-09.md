# Full prepared regression — 2026-09-09

First and only execution of the complete prepared suite, at committed checkpoint
`d73b7563716` on a two-chip Wormhole N300. Driver:
`generated/reduce_migration_reviews/full_regression_20260909/run_full_suite.py`,
which refuses to start on a dirty worktree, records the checkpoint, and
redirects the fidelity reports so the committed copies stay untouched.

    python3 scripts/run_reduce_migration_tests.py

Duration 3h48m (14:59:51Z to 18:48:39Z). Runner exit code 1, from the seven
failed groups below.

## Totals

| Metric | Value |
|---|---:|
| Groups requested / completed | 178 / 178 |
| Definitions selected | 914 Python functions + 86 C++ cases |
| Cases collected | 18,860 |
| Passed | 12,553 |
| Skipped | 6,210 |
| Failures | 95 |
| Errors | 2 |
| Failed groups | 7 |

The collected total matches the manifest's derived 18,860 exactly, confirming
that figure by execution rather than derivation for the first time. Every group
produced complete JUnit XML. The two committed fidelity reports were verified
byte-identical afterwards (`original_reports_unchanged.json`).

The 6,210 skips are the suite's pre-existing upstream skip/xfail marks and
hardware guards, plus the four upstream-disabled C++ cases. Rounds 5 through 8
separately confirmed that this migration introduced no new skips and relaxed no
tolerances.

## Failure analysis: no migration regressions

All 97 failing cases in seven groups are attributable to hardware that this host
does not have, or to one upstream limitation unrelated to reductions. Each was
verified rather than assumed.

**T119 — 58 cases — `tests/nightly/blackhole/sdpa/test_scaled_dot_product_attention_sprint.py`.**
Blackhole/Galaxy SDPA sweep. 54 cases fail in
`sdpa_program_factory.cpp:450`: "Provided grid must not contain more cores than
the device. Got 110 cores, expected at most 64 cores." Two more hit the 300s
pytest timeout, and two error on setup with "Real-time profiler must be active
for SDPA perf checks", which this build is not. Wrong architecture and a
profiler build requirement.

**T162 — 14 cases — and T163 — 6 cases — UDM fabric reduction gtests.**
`MeshDevice1x4Fabric2DUDMFixture` and `MeshDevice2x4Fabric2DUDMFixture` require
four and eight devices. Each case dies with SIGSEGV (status -11) immediately
after topology discovery, inside fixture setup, before any kernel is dispatched.
Verified not migration-related by running an unrelated, branch-untouched test
under the same fixture:

    ./build/test/ttnn/unit_tests_ttnn_udm \
      --gtest_filter='MeshDevice1x4Fabric2DUDMFixture.TestMeshWidthShardedCopy2D_Small'

That copy test, which performs no reduction, segfaults identically on this host.
The fixture crashing rather than skipping cleanly on an undersized cluster is a
pre-existing rough edge worth reporting upstream, but it is not a defect in this
branch.

**T151 — 12 cases — `models/demos/vision/classification/resnet50/quasar/tests/ops/test_reduce_sum_mean.py`.**
`pytensor.cpp:301`: "Can't convert a tensor distributed on MeshShape([1, 2])
mesh to row-major logical tensor. Supply a mesh composer to concatenate
multi-device shards." The test assumes a single chip while the N300 fixture opens
both. This is the same setup issue already recorded in
`generated/reduce_migration_reviews/quasar_h_wormhole_validation_20260908.md`,
where the migrated Quasar H factory passed once a one-chip mesh was selected.
The failure is in test readback, after the kernels ran.

**T166 — 4 cases — `models/demos/falcon7b_common/tests/unit_tests/test_falcon_matmuls_and_bmms_with_mixed_precision.py::test_softmax`.**
The only failing group on hardware this host has. Despite the name, the failure
is not in softmax: the sharded softmax calls all succeed, and the test then
fails at line 874 on `ttnn.multiply(tt_input, tt_scalar, ...)` with
`binary_ng_device_operation.cpp:255: Invalid subtile broadcast type`, where
`tt_scalar` has shape `[1, 1, 32, 32]`. `seq_len_32` passes because the shapes
match exactly and no broadcast is needed. Reproduced standalone with no migrated
code in the path:

    ttnn.multiply(<[1,1,64,64] bf16 tile>, <[1,1,32,32] bf16 tile>)   -> raises
    ttnn.multiply(<[1,1,128,128] bf16 tile>, <[1,1,32,32] bf16 tile>) -> raises

Both raise the identical `Invalid subtile broadcast type`. The branch modifies
neither this test (`git diff` empty) nor `ttnn/cpp/ttnn/operations/eltwise/binary_ng/`
(`git diff` empty). This is an upstream binary_ng broadcast limitation.

**T106 — 2 cases — `models/demos/deepseek_v3/tests/fused_op_unit_tests/mla/test_flash_mla_deepseek.py`.**
`work_split.cpp:98`: "Target number of cores 64 is greater than total number of
available cores 56." A Wormhole compute grid provides 56 cores.

**T168 — 1 case — `models/demos/minimax_m3/tests/unit/test_msa_prefill_vs_ref.py`.**
`indexer_score_device_operation.cpp:312`: "indexer_score is only supported on
Blackhole, got wormhole_b0."

## What this run does and does not establish

Established: the full prepared suite runs end to end at this checkpoint; the
18,860 case count is real rather than derived; 12,553 cases pass; and no failure
traces to the reduce-helper migration.

Not established, unchanged from every prior round: the Quasar, Blackhole,
Galaxy, T3K and 1x4-fabric lanes have no numerical coverage on this host. The
Quasar W/H/HW/RM factories and SDPA writers, KDA, indexer, attn-res, DiT fused
RMSNorm, DiT Welford — whose consumed data this branch intentionally changes —
sparse SDPA and the UDM fabric tests remain reviewed statically only. Running
this suite on a Blackhole, Galaxy, T3K or 4-device fabric host would convert
most of the seven failed groups into real coverage, and is the obvious next step
if that hardware becomes available.
