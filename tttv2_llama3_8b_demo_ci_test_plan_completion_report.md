# TTTv2 Llama 3.1 8B Demo and CI Completion Report

Date: 2026-07-29

Branch: `gongyu/tttv2_llama8b_perf_parity`

Base implementation commit: `b6f63635f6c8c7f370dbd197916aa535439f8d5f`

Initial rollout implementation commit:
`829e9844fdfc35a8dd378cd9047baa85dbd03375`

Corrective rollout implementation commit:
`efac78758b6248208b5e7bb502d7f961e89b08cb`

## Executive Summary

The local implementation and validation work in
`tttv2_llama3_8b_demo_ci_test_plan.md` is complete. The change provides the
requested TTTv2 Llama 3.1 8B demo matrix, production data-parallel execution,
measured N150 and T3K target rows, strict benchmark-artifact enforcement, and
Wormhole/Galaxy CI registrations. Existing Blackhole TTTv1 coverage remains
unchanged.

All planned local hardware configurations passed:

- N150 primary, performance, accuracy, host-sampling, on-device-sampling, and
  supported capacity cells.
- N300 DP-2.
- T3K primary, DP-4, DP-8, and supported `batch-32-ci` cells.
- Galaxy DP-4, DP-8, DP-16, and DP-32.

Every Galaxy DP factor passed from an empty per-topology tensor cache and then
passed warm. All warm runs used the production 1000-second timeout. The final
N150 and T3K benchmark archives also pass strict target validation with zero
hard failures and zero missing/TODO entries.

The first three `All Model Tests` rollout workflows were dispatched on
2026-07-29 against the exact pushed implementation SHA
`829e9844fdfc35a8dd378cd9047baa85dbd03375`. All three passed preflight and
Harbor availability checks. N150 then exposed a CI-only read-only tensor-cache
failure before model execution. T3K and Galaxy were cancelled because a
corrective implementation commit requires fresh exact-SHA runs. The failure,
fix, and superseded run IDs are recorded below.

Fresh N150, T3K, and Galaxy runs were dispatched against corrective SHA
`efac78758b6248208b5e7bb502d7f961e89b08cb`. At the user's direction, all
three were cancelled while build jobs were running. No corrective hardware
job started, and these cancellations do not prove final CI success.

## Completion Status

| Area | Status | Evidence |
| --- | --- | --- |
| Demo case matrix | Complete | 20 exact nodes collected |
| Shared timing and determinism helpers | Complete | Focused and runtime contract tests pass |
| Hybrid sampling and readback | Complete | Host prefill and device top-k decode tests pass |
| Production DP composition | Complete | N300, T3K, and Galaxy hardware matrix passes |
| Galaxy fabric topology | Implemented; full rerun blocked | Shape-aware fixture policy and DP-8 lane all-gather pass; `/data` NFS unavailable for Llama rerun |
| Strict benchmark dimensions | Complete | Resolver and validator tests pass |
| Measured N150/T3K targets | Complete | Three-sample rows pass strict artifact replay |
| N150 artifacts | Complete | 34 JSON files; 0 hard failures; 0 missing/TODO |
| T3K artifacts | Complete | 39 JSON files; 0 hard failures; 0 missing/TODO |
| Cold/warm local hardware | Complete | Every enabled topology has retained evidence |
| Wormhole/Galaxy CI YAML | Complete | Exact node IDs and required artifact policy registered |
| Final software suite | Complete | 684 passed, 1 expected module skip; 20 nodes collected |
| Final-SHA workflow dispatches | Cancelled by request | Corrective exact-SHA runs `30466788086`, `30466800497`, and `30466811568` are terminal `cancelled` |

## Implementation Summary

### Demo and Runtime

`models/common/tests/demos/llama3_8b/demo.py` defines both `performance` and
`accuracy` profiles for:

- `token-accuracy`
- `batch-1`
- `batch-32`
- `batch-32-ci`
- `eval-32`
- `ci-b1-DP-2`
- `ci-b1-DP-4`
- `ci-b1-DP-8`
- `ci-b1-DP-16`
- `ci-b1-DP-32`

N150 intentionally skips `batch-32-ci`; T3K supports and measures it. The DP
cases construct lane-local public `Llama3Executor` objects and compose them
through the production `LaneGroupExecutor`.

`models/common/tests/demos/run_helpers.py` preserves host argmax behavior,
supports separate prefill and decode sampling parameters, brackets profiler
events, excludes first-token compile effects from steady-state decode metrics,
and synchronizes every lane mesh before timing boundaries.

Multi-device prefill uses host argmax while decode can use device top-k. This
avoids unsupported multi-device sampling during prefill without changing the
requested decode mode.

### Galaxy Correctness

Galaxy DP-4 alone is not sufficient to validate the fabric and CCL policy:
two-device lanes can pass configurations that fail on the four-device lanes
used by DP-8. Controlled DP-4/DP-8 reversions established that Ring fabric on
the `(4, 8)` Galaxy parent cannot route the Linear collectives required by its
four-device children. The correction belongs in the generic fixture rather
than the Llama demo:

- one-row parents with at least eight devices use `FABRIC_1D_RING`;
- multi-row parents, including Galaxy `(4, 8)`, use `FABRIC_1D`;
- explicit test-provided fabric configuration still takes precedence;
- distributed RMSNorm selects its topology through `default_topology`;
- Galaxy opens as the native `(4, 8)` parent mesh;
- DP-4/8/16/32 lane shapes are `(1, 8)`, `(1, 4)`, `(1, 2)`, and `(1, 1)`.

With hardcoded Ring in distributed RMSNorm, Galaxy DP-8 failed routing its
RMSNorm statistics from D0 to D12, so topology-aware RMSNorm remains. Further
experiments localized the later sampling failure to Ring fabric on a DP-8
`(1, 4)` lane. After the fixture correction selected `FABRIC_1D`, the same
synchronous Linear all-gather completed and returned the expected tensor on
all four lane devices. Full Llama DP-8/DP-4 reruns remain pending because the
NFS server backing `/data/umales/.cache/huggingface` became unreachable before
model collection.

Tensor cache filenames retain the device identity. A diagnostic attempt to
reuse tensor binaries across transient lane IDs stalled when a later lane
loaded a tensor serialized for the first lane. The final validation therefore
keeps the established device fingerprint and materializes one cache per lane.

Galaxy DP-8 exposes the four-device name `N150x4`. The TTTv2 adaptor now gives
it the legacy-compatible 4K prefill chunk default and an explicit trace-length
policy. DP-16 and DP-32 continue to use the existing `N300` and `N150`
policies.

### Targets and Artifact Enforcement

Targets match on the exact optional dimensions:

- `sampling_mode`
- `optimization_profile`
- `workload`

The N150 and T3K rows were derived from repeated measurements on the intended
SKU and workload. N150 `batch-32-ci` remains explicitly inactive because its
2048-token capacity is unsupported. T3K `batch-32-ci` is active for both host
and on-device top-k sampling.

The validator now fails `--strict-missing` when no complete benchmark JSON is
present. The model e2e workflow requires benchmark data only for matrix entries
marked `benchmark_data_required`; optional jobs retain warning-only behavior.
The N150 and T3K primary Llama entries are marked required.

### CI Registration

`tests/pipeline_reorg/models_e2e_tests.yaml` registers:

- N150 primary TTTv2 nodes.
- T3K primary TTTv2 nodes.
- T3K DP-4 and DP-8.
- Galaxy DP-4, DP-8, DP-16, and DP-32.

The exact primary nodes are token accuracy, batch-32, and eval-32. Existing
Blackhole TTTv1 commands and unrelated model jobs were not migrated.

## Hardware Results

All hardware used the local offline checkpoint
`/tmp/Llama-3.1-8B-Instruct`, `CI=true`, the requested mesh identity, and
sequential access to the 32-chip Wormhole Galaxy.

### N150

- Token accuracy cold and warm passed: 89.3% top-1, 97.9% top-5.
- Both optimization profiles passed for batch-1 and batch-32.
- Host and on-device top-k sampling passed for every gated performance cell.
- Exact primary token-accuracy, batch-32, and eval-32 nodes passed.
- `batch-32-ci` produced the intended explicit skip.
- Final archive: `/tmp/tttv2_llama3_8b_artifacts/wh_n150_20260729`.

Measured active target values:

| Profile | Sampling | Workload | Decode t/s/u | TTFT ms |
| --- | --- | --- | ---: | ---: |
| performance | host | batch-1 | 26.1 | 110.3 |
| performance | top-k | batch-1 | 30.0 | 100.4 |
| performance | host | batch-32 | 23.6 | 38.0 |
| performance | top-k | batch-32 | 27.1 | 39.4 |
| accuracy | host | batch-1 | 23.8 | 135.5 |
| accuracy | top-k | batch-1 | 27.0 | 125.1 |
| accuracy | host | batch-32 | 21.7 | 43.0 |
| accuracy | top-k | batch-32 | 24.7 | 44.4 |

### N300

- Exact DP-2 cold run passed in 276.71 seconds.
- Exact DP-2 warm run passed in 28.80 seconds.

### T3K

- Token accuracy passed: 90.2% top-1, 97.5% top-5.
- Both profiles, both sampling modes, batch-1, batch-32, and
  `batch-32-ci` passed.
- Exact primary token-accuracy, batch-32, and eval-32 nodes passed.
- All 12 gated performance configurations passed three samples each.
- DP-4 cold/warm passed in 594.38/102.15 seconds.
- DP-8 cold/resumed and warm passed in 704.64/159.90 seconds.
- Final archive:
  `/tmp/tttv2_llama3_8b_artifacts/wh_llmbox_perf_20260729`.

### Galaxy

Final cache:
`/tmp/tttv2_llama3_8b_final_galaxy_linear_v3`.

Retained logs:
`/tmp/tttv2_llama3_8b_artifacts/wh_galaxy_perf_20260729`.

| DP | Lane topology | Cold seconds | Warm seconds | Warm timeout |
| ---: | --- | ---: | ---: | ---: |
| 4 | T3K, 1x8 | 582.85 | 81.81 | 1000 |
| 8 | N150x4, 1x4 | 1333.00 | 132.27 | 1000 |
| 16 | N300, 1x2 | 2306.90 | 130.27 | 1000 |
| 32 | N150, 1x1 | 5348.66 | 349.20 | 1000 |

The cold duration grows with the number of lane-specific serialized tensor
sets. Warm execution for every factor is comfortably within the production
timeout.

## Strict Artifact Validation

T3K command:

```bash
python3 .github/scripts/utils/validate_perf_targets.py \
  --path-profile repo-root \
  --sku wh_llmbox_perf \
  --strict-missing
```

Result:

```text
Validation completed: 39 benchmark file(s), 0 hard failures, 0 missing/TODO entries
```

N150 replay used the same command with `--sku wh_n150` after placing the
archived JSON files under `generated/benchmark_data`.

Result:

```text
Validation completed: 34 benchmark file(s), 0 hard failures, 0 missing/TODO entries
```

Transcripts are retained as `strict_validation.log` in each artifact archive.
Warnings about unrelated repository model/SKU target gaps are informational
and do not affect these Llama artifact results.

## Software Verification

The final verification sequence produced:

| Suite | Result |
| --- | --- |
| Helpers, target resolver, validator | 45 passed |
| Demo collection | 20 nodes collected |
| Lane-group and Llama integration | 32 passed, 1 expected module skip |
| Adaptor, Galaxy demo, trace-region regression | 555 passed |
| Lazy-weight unit/integration | 52 passed |
| Total executed tests | 684 passed, 1 expected skip |

Additional checks:

- `git diff --check` passes.
- Black passes for every changed Python file.
- YAML target and pipeline files are exercised by the resolver, validator, and
  trace-region suites.
- The Black Python 3.10 parser warning is environmental; formatting completed
  successfully.

The consolidated software transcript is:
`/tmp/tttv2_llama3_8b_final_software_validation.log`.

## Workflow Rollout

GitHub CLI device authentication completed as `gwangTT`. Immediately before
dispatch, both local `HEAD` and
`refs/heads/gongyu/tttv2_llama8b_perf_parity` on `origin` resolved to:

```text
829e9844fdfc35a8dd378cd9047baa85dbd03375
```

The final rollout uses `All Model Tests`. Every dispatch sets
`run-unit-tests=false`, `run-e2e-tests=true`, `run-sweep-tests=false`,
`tier-3=false`, `platform=Ubuntu 22.04`, `build-type=Release`,
`enable-lto=false`, and `mlperf-read-only=true`.

| Dispatch | Tiers | SKU | Model filter | Run | Created (UTC) |
| --- | --- | --- | --- | --- | --- |
| N150 | Tier 1 | `wh_n150` | `llama3.1-8b` | [30462837262](https://github.com/tenstorrent/tt-metal/actions/runs/30462837262) | 2026-07-29 14:50:05 |
| T3K | Tier 2 | `wh_llmbox_perf` | `llama3.1-8b` | [30462859524](https://github.com/tenstorrent/tt-metal/actions/runs/30462859524) | 2026-07-29 14:50:20 |
| Galaxy | Tier 1 | `wh_galaxy_perf` | `llama3.1-8b-dp-galaxy` | [30462879456](https://github.com/tenstorrent/tt-metal/actions/runs/30462879456) | 2026-07-29 14:50:34 |

GitHub reports `headSha` as the complete rollout SHA above for all three runs.
Their rendered run names confirm the requested tier, test type, model filter,
and SKU. The T3K substring filter intentionally selects both the primary and
DP model entries.

Initial execution evidence:

- all three `preflight` jobs passed;
- all three `check-harbor / Check Harbor` jobs passed;
- platform parsing and artifact metadata passed;
- Python 3.10 wheel builds passed;
- N150 and T3K Release builds passed;
- N150 selected exactly
  `Llama 3.1-8B e2e tests (Wormhole N150 TTTv2) [wh_n150]`;
- T3K selected exactly the primary T3K and Wormhole data-parallel suites.

### Initial Rollout Failure and Correction

N150 run `30462837262` completed with failure. Its first exact node was:

```text
models/common/tests/demos/llama3_8b/demo.py::test_llama3_8b[
  wormhole_b0-performance-device_params0-token-accuracy-N150]
```

The node failed before model execution and before producing benchmark data:

```text
OSError: [Errno 30] Read-only file system:
'/mnt/MLPerf/huggingface/tt_cache/tttv2/meta-llama/Llama-3.1-8B-Instruct'
```

The adaptor appended the `N150` device directory to the required canonical
`TT_CACHE_PATH`. That directory was absent, and `mlperf-read-only=true`
correctly prevented creating it. The strict artifact step then also failed
because the aborted test produced no `generated/benchmark_data` directory.

The correction preserves both rollout requirements:

- the canonical MLPerf `TT_CACHE_PATH` remains configured;
- the MLPerf mount remains read-only.

`_model_cache_path` now uses the configured device cache when it exists or can
be created. If creation fails with `EROFS`, `EACCES`, or `EPERM`, it uses a
writable job-local path under
`/tmp/tttv2_model_cache/<model>/<device>`. The fallback persists across the
three pytest commands in one CI job, so later commands reuse tensors generated
by the first cold-cache command. Unexpected filesystem errors still propagate.

Focused regression coverage proves both the normal configured-cache path and
the read-only fallback path. The adaptor suite passes with 7 tests.

T3K run `30462859524` and Galaxy run `30462879456` were cancelled after this
failure because they target the superseded SHA. Neither cancellation is
treated as validation evidence. The corrective runs dispatched afterward are
recorded separately below.

### Corrective Rollout Cancellation

The cache correction and report were committed and pushed as:

```text
efac78758b6248208b5e7bb502d7f961e89b08cb
```

Before redispatch, local `HEAD` and the remote branch both resolved to that
exact SHA. Three fresh runs used the same required inputs as the initial
dispatches:

| Dispatch | Run | Created (UTC) | Final conclusion |
| --- | --- | --- | --- |
| N150 | [30466788086](https://github.com/tenstorrent/tt-metal/actions/runs/30466788086) | 2026-07-29 15:37:46 | Cancelled |
| T3K | [30466800497](https://github.com/tenstorrent/tt-metal/actions/runs/30466800497) | 2026-07-29 15:37:55 | Cancelled |
| Galaxy | [30466811568](https://github.com/tenstorrent/tt-metal/actions/runs/30466811568) | 2026-07-29 15:38:03 | Cancelled |

GitHub reports
`headSha=efac78758b6248208b5e7bb502d7f961e89b08cb` for every corrective
run. All three preflight, Harbor, platform parsing, and artifact metadata jobs
passed. Their wheel and Release build jobs were cancelled at approximately
15:49 UTC. No model hardware job had started, so these runs provide no
corrective hardware, exact-node execution, benchmark-artifact, or aggregate
success evidence.

The monitor process was stopped, cancellation was requested for all three
runs, and each run was polled until GitHub reported terminal
`status=completed` and `conclusion=cancelled`. There are no active rollout
runs.

Each run must satisfy:

1. `headSha` equals the committed and pushed rollout SHA.
2. Only intended SKU/model jobs are selected.
3. Every YAML command resolves to one exact pytest node.
4. Every selected node and aggregate workflow concludes successfully.
5. No CI-affecting edit follows the passing final-SHA runs.

Status: the initial rollout diagnosed and corrected a CI-only cache-mount
failure. The corrective exact-SHA runs were cancelled by request before
hardware execution. Aggregate success, selected hardware jobs, exact pytest
node execution, and CI artifact validation therefore remain unproven.

## Replacement-Scoped Revision and Local Validation

After the cancelled rollout, the implementation was revised in the uncommitted
worktree to enforce the clarified objective: replace existing TTTv1 coverage
where TTTv2 can provide the same coverage, without adding CI cases or
redesigning target resolution.

### CI and Demo Scope Cleanup

The primary Wormhole registration now uses one shared command branch for N150
and T3K. The branch exports only the device selected by the SKU and runs these
three exact TTTv2 replacement selectors:

```text
performance and token-accuracy-repeat_batch-1-prefetcher-off
performance and eval-32-repeat_batch-3-prefetcher-off-perf-report-off
performance and eval-32-repeat_batch-1-prefetcher-off-perf-report-on
```

The pytest IDs make the legacy coverage dimensions explicit:

- the default three-repeat TTTv1 eval comparison maps to
  `repeat_batch-3`, `prefetcher-off`, `perf-report-off`;
- the one-repeat TTTv1 performance run maps to
  `repeat_batch-1`, `prefetcher-off`, `perf-report-on`;
- TTTv2 identifies `prefetcher-off` because it has no TTTv1 DRAM
  prefetcher;
- the legacy `--use_hf_rope` invocation remains on TTTv1 because native
  HF-layout RoPE is not yet supported by TTTv2.

Existing data-parallel registrations were replaced at their existing factors:
T3K DP-4/DP-8 and Galaxy DP-4/DP-8/DP-16/DP-32. No new primary, DP, or release
test case was added. Release registrations mirror the existing N150 token,
N150 repeat-three, and Galaxy DP coverage.

The demo implementation was simplified around the shared eval workload:

- both eval cases use
  `models/tt_transformers/demo/sample_prompts/eval_repeat_prompts_batch32.json`;
- repeat count and performance reporting remain pytest parameters instead of
  separate hard-coded workload functions;
- prompt rotation and generated-token comparison stay inline in the eval
  runner;
- redundant wrappers and the proposed standalone
  `models/common/tests/demos/llama3_8b/test_demo.py` were removed;
- `models/model_targets.yaml` and target-resolution behavior remain unchanged.

The only trace-allocation change retained is the existing
`llama3.1-8b/wh_llmbox_perf` value changing from 50 MB to 60 MB, based on a
measured T3K TTTv2 trace footprint of 53,764,096 bytes. The corresponding
legacy-SKU assertion was updated by one line; no new trace test was added.

### Traced-Prefill Correctness Fix

The exact N150 repeat-three evaluation initially failed deterministically for
original users 0 and 1:

```text
AssertionError: eval-32 generated token IDs differed for users [0, 1, 0, 1]
```

The same exact TTTv1 N150 comparison passed in 223.66 seconds. Diagnostics
showed that rotation and KV-slot routing were correct and that the failure
depended on which of the two padded-1024 prompts replayed last.

The root cause was output lifetime in `TracedExecutor.prefill_forward`.
Requests with the same trace signature reuse one persistent trace output
buffer, but the executor first replayed every prepared request into a tuple and
only then assembled/read the results. Later replays therefore overwrote output
referenced by earlier results. Repeat zero and repeat two ended with the same
long prompt, while repeat one reversed the two long prompts, exactly explaining
the two-user mismatch.

The executor now passes a lazy iterable to `PrefillRuntime.assemble`, causing
each trace replay to be read and released before the next replay can overwrite
the shared buffer. `PrefillRuntime.assemble` accepts an iterable and validates
the sampling contract per consumed result. A focused regression test models
two requests sharing one trace output and proves that the first output is
consumed before the second replay mutates it.

Focused traced-prefill execution/runtime validation passes:

```text
96 passed
```

### Static, Collection, and Software Results

The following checks passed against the revised uncommitted implementation:

- Black formatting over all changed Python files;
- Python bytecode compilation over all changed Python files;
- staged and unstaged `git diff --check`;
- YAML parsing for model and release registries;
- model matrix expansion (`65` generated jobs);
- tier-1 and tier-2 time-budget checks;
- exact collection for N150, T3K, and Galaxy;
- each primary selector selects one node and the combined expression selects
  exactly three nodes per primary Wormhole SKU;
- all exact T3K DP, Galaxy DP, N150 release, and Galaxy release nodes exist.

The final focused implementation command completed with:

```text
206 passed, 218 hardware/topology skips, 771 deselected
```

The earlier implementation-only run passed all 327 applicable tests after
supplying `HF_MODEL=/tmp/Llama-3.1-8B-Instruct` to the two RMSNorm cases that
require model weights. The trace-region suite produced 535 passes and one
known unrelated baseline failure: the Gemma 3 27B test expects 30,000,000
while the committed YAML contains 30,100,000. No unrelated Gemma correction
was included.

### N150 Hardware Results

Hardware ran sequentially with:

```text
HF_MODEL=/tmp/Llama-3.1-8B-Instruct
HF_HUB_OFFLINE=1
SAMPLING_MODE=on_device_topk
PIPELINE_READBACK=1
```

All N150 replacement nodes pass after the traced-prefill correction:

| Replacement node | Result | Duration |
| --- | --- | ---: |
| token accuracy, repeat 1, prefetcher off | Passed | 37.99s |
| eval-32, repeat 3, prefetcher off, report off | Passed | 78.77s |
| eval-32, repeat 1, prefetcher off, report on | Passed | 35.01s |

The token-accuracy run measured top-1 accuracy of 89.3% and top-5 accuracy of
97.9%. The one-repeat eval run measured 115.7 ms TTFT, 24.6 tok/s/user, and
788.0 aggregate tok/s. Centralized artifact validation was not applied locally
because the local checkpoint path does not resolve to the canonical CI target
alias; that remains a GitHub-only check.

### Initial Multi-Device Blocker

T3K and Galaxy validation could not initially reach model setup. Repeated T3K
attempts failed while opening the mesh, including after
`tt-smi -glx_reset_auto`. Diagnostics included:

```text
Fabric Router Sync: Timeout after 10000 ms on Device 17
Fabric Router Sync: Timeout after 10000 ms on Device 24
Timed out waiting for ETH heartbeat ... ETH core e7-6 ... Stuck at 0xabcd935c
```

These failures occurred before model execution and are host-fabric failures,
not test assertions. No test or reset process was left running.

### 2026-07-30 Galaxy Recovery and Complete Retry

The blocked matrix was retried using the requested Galaxy reset order. No
hardware tests ran concurrently.

First, the supported warm reset was issued:

```text
tt-smi -r
```

The command reset all 32 PCI devices, reinitialized the boards, completed
topology discovery, and exited successfully. The first T3K token-accuracy
retry nevertheless failed while opening the mesh with the same device 17
fabric-router synchronization timeout. Because the warm reset did not recover
the fabric, the documented Galaxy fallback was then issued:

```text
tt-smi -glx_reset
```

The fallback successfully executed the IPMI tray reset, rediscovered all 32
chips, issued post-reset handling, and reported that all 32 boards were
reinitialized.

After the fallback, every previously blocked exact node was invoked separately.
Each pytest invocation collected exactly its intended node. All nine nodes
reported `ERROR` in the `mesh_device` fixture before model construction,
weight loading, trace capture, prefill, or decode:

| SKU | Exact workload | Result | First timed-out device | Log |
| --- | --- | --- | ---: | --- |
| T3K | token accuracy | Setup error | 17 | `/tmp/tttv2_validation_t3k_token_20260730_glxreset.log` |
| T3K | eval-32 repeat 3, report off | Setup error | 16 | `/tmp/tttv2_validation_t3k_eval_repeat3_20260730.log` |
| T3K | eval-32 repeat 1, report on | Setup error | 16 | `/tmp/tttv2_validation_t3k_eval_repeat1_20260730.log` |
| T3K | DP-4 | Setup error | 16 | `/tmp/tttv2_validation_t3k_dp4_20260730.log` |
| T3K | DP-8 | Setup error | 16 | `/tmp/tttv2_validation_t3k_dp8_20260730.log` |
| Galaxy | DP-4 | Setup error | 16 | `/tmp/tttv2_validation_galaxy_dp4_20260730.log` |
| Galaxy | DP-8 | Setup error | 3 | `/tmp/tttv2_validation_galaxy_dp8_20260730.log` |
| Galaxy | DP-16 | Setup error | 0 | `/tmp/tttv2_validation_galaxy_dp16_20260730.log` |
| Galaxy | DP-32 | Setup error | 0 | `/tmp/tttv2_validation_galaxy_dp32_20260730.log` |

The common failure is:

```text
Fabric Router Sync: Timeout after 10000 ms
expected status 0xa2b2c2d2 (LOCAL_HANDSHAKE_COMPLETE)
```

The timed-out device changes across invocations, demonstrating that this is
not isolated to one pytest parameter or one model topology. Both reset methods
complete, but neither restores a fabric capable of opening the required T3K or
Galaxy mesh. The complete retry therefore adds no model-execution evidence;
the previously passing N150 results remain the only local hardware evidence
for the replacement.

### 2026-07-30 Different-Host Scope Check

The smoke sequence was repeated on host `UF-EV-B6-GWH02` with the offline
checkpoint in the host's shared Hugging Face cache and a writable tensor cache
under `/tmp/tttv2_llama3_8b_cache_gwh02`. No hardware tests ran concurrently.

| SKU | Exact workload | Result | Duration | Log |
| --- | --- | --- | ---: | --- |
| T3K | token accuracy | Setup error on Device 17 fabric sync | 23.56s | `/tmp/tttv2_validation_t3k_token_20260730_gwh02.log` |
| N150 | token accuracy | Passed; 89.3% top-1, 97.9% top-5 | 247.27s | `/tmp/tttv2_validation_n150_token_canonical_20260730_gwh02.log` |
| N300 | DP-2 | Passed | 317.23s | `/tmp/tttv2_validation_n300_dp2_20260730_gwh02.log` |

The N150 and N300 results reached model construction and execution. N300 DP-2
also completed lane construction, trace capture, and on-device top-k sampling.
The T3K node again failed in the `mesh_device` fixture before model
construction:

```text
Fabric Router Sync: Timeout after 10000 ms on Device 17
expected status 0xa2b2c2d2 (LOCAL_HANDSHAKE_COMPLETE)
```

This narrows the observed failure: it does not reproduce on N150 or N300 DP-2
on the same 32-chip host. At this point the evidence was consistent with a
T3K-submesh issue, but native Galaxy DP execution was still required to test
that hypothesis.

### 2026-07-30 Native Galaxy Follow-Up

The four exact Galaxy nodes were then run separately and sequentially on the
same host:

| Exact workload | Result | First timed-out device | Duration | Log |
| --- | --- | ---: | ---: | --- |
| Galaxy DP-4 | Setup error | 16 | 20.21s | `/tmp/tttv2_validation_galaxy_dp4_20260730_gwh02.log` |
| Galaxy DP-8 | Setup error | 3 | 15.25s | `/tmp/tttv2_validation_galaxy_dp8_20260730_gwh02.log` |
| Galaxy DP-16 | Setup error | 0 | 15.30s | `/tmp/tttv2_validation_galaxy_dp16_20260730_gwh02.log` |
| Galaxy DP-32 | Setup error | 0 | 15.27s | `/tmp/tttv2_validation_galaxy_dp32_20260730_gwh02.log` |

Every node collected exactly once and failed in the `mesh_device` fixture
while opening the native 32-chip Galaxy parent mesh. The common error remained:

```text
Fabric Router Sync: Timeout after 10000 ms
expected status 0xa2b2c2d2 (LOCAL_HANDSHAKE_COMPLETE)
```

No node reached lane construction, model construction, weight loading, trace
capture, prefill, or decode. These results reject the tentative
T3K-submesh-only explanation: the native Galaxy parent mesh also cannot
initialize. Together with the passing N150 and N300 DP-2 runs, the evidence
localizes the blocker to the larger Galaxy fabric path rather than the model
implementation or all multi-device execution.

### 2026-07-30 Galaxy Reset Retry

The native Galaxy matrix was retried sequentially on `UF-EV-B6-GWH02`. The
supported warm reset completed, but the first DP-4 smoke test failed in
control-plane auto-discovery because the logical 8x4 mesh could not be mapped
onto the discovered physical topology:

```text
Graph specified in MGD could not fit in the discovered physical topology
```

The fallback Galaxy reset was then issued:

```text
tt-smi -glx_reset
```

It completed the IPMI tray reset, rediscovered the topology, and reinitialized
all 32 boards. After this reset, every exact native Galaxy node passed:

| Exact workload | Final result | Duration | Log |
| --- | --- | ---: | --- |
| Galaxy DP-4 | Passed | 671.01s | `/tmp/tttv2_validation_galaxy_dp4_20260730_gwh02_glxreset.log` |
| Galaxy DP-8 | Passed | 338.16s | `/tmp/tttv2_validation_galaxy_dp8_20260730_gwh02_glxreset_retry.log` |
| Galaxy DP-16 | Passed | 421.98s | `/tmp/tttv2_validation_galaxy_dp16_20260730_gwh02_glxreset_retry.log` |
| Galaxy DP-32 | Passed | 1264.97s | `/tmp/tttv2_validation_galaxy_dp32_20260730_gwh02_glxreset_retry.log` |

DP-8, DP-16, and DP-32 first exceeded pytest's 1000s, 2000s, and 4000s
timeouts respectively while populating an empty per-device tensor cache. No
hardware error occurred in those attempts, and device teardown completed
cleanly. Their retries resumed from the populated cache and passed. DP-32
constructed all 32 replicas, captured the prefill and decode traces, executed
the 32-user workload, and closed all devices cleanly.

This supersedes the pre-reset native Galaxy conclusion above:
`tt-smi -glx_reset` cleared the Galaxy parent-fabric block, and Galaxy
DP-4/8/16/32 validation is complete. The warm reset alone was insufficient.

The exact T3K token-accuracy smoke node was then retried after the successful
fallback reset. It initially failed in the root `mesh_device` fixture before
model construction:

```text
Fabric Router Sync: Timeout after 10000 ms on Device 17
expected status 0xa2b2c2d2 (LOCAL_HANDSHAKE_COMPLETE)
```

That run completed with one setup error in 15.11 seconds and closed the
devices cleanly. Its log is
`/tmp/tttv2_validation_t3k_token_20260730_gwh02_post_glxreset.log`. This
showed that directly opening the 1x8 subset still failed even though opening
the full Galaxy parent succeeded.

The demo was then corrected to use `ttnn_mesh_device`, whose Galaxy handling
opens the full parent mesh before deriving a requested child submesh. The
fixture now also resolves trace-region memory from the requested logical mesh,
and the demo passes that concrete size to the fixture. After a full Galaxy
reset cleared kernels left by the failed direct-subset attempt, the same T3K
token-accuracy workload passed:

| Workload | Result | Accuracy | Duration | Log |
| --- | --- | --- | ---: | --- |
| T3K token accuracy, parent-first 1x8 child with explicit trace size | Passed | 89.8% top-1, 98.0% top-5 | 78.01s | `/tmp/tttv2_validation_t3k_token_20260730_gwh02_explicit_trace_size.log` |
| T3K token accuracy after dead fixture-param removal | Passed | 89.8% top-1, 98.0% top-5 | 77.66s | `/tmp/tttv2_validation_t3k_token_20260730_gwh02_no_device_params.log` |
| T3K token accuracy with fixture-default `FABRIC_1D_RING` | Passed | 89.8% top-1, 98.0% top-5 | 126.80s | `/tmp/tttv2_validation_t3k_token_20260730_gwh02_default_fabric.log` |
| T3K token accuracy with Ring fabric and baseline Ring CCL | Passed | 89.8% top-1, 98.0% top-5 | 90.75s | `/tmp/tttv2_validation_t3k_token_20260730_gwh02_ring_fabric_ring_ccl.log` |
| T3K DP-4, four 1x2 lanes | Passed | Four non-empty lane outputs | 134.59s | `/tmp/tttv2_validation_t3k_dp4_20260730_gwh02_parent_first.log` |
| T3K DP-8, eight 1x1 lanes | Passed | Eight non-empty lane outputs | 176.13s | `/tmp/tttv2_validation_t3k_dp8_20260730_gwh02_parent_first.log` |
| Native Galaxy DP-4 with fixture-default `FABRIC_1D_RING` | Passed | Four non-empty lane outputs | 135.80s | `/tmp/tttv2_validation_galaxy_dp4_20260730_gwh02_default_fabric.log` |
| Native Galaxy DP-4 with Ring fabric and baseline Ring CCL | Passed | Four non-empty lane outputs | 87.62s | `/tmp/tttv2_validation_galaxy_dp4_20260730_gwh02_ring_fabric_ring_ccl.log` |
| T3K DP-4 with hardcoded Ring RMSNorm | Passed | Four non-empty lane outputs | 168.14s | `/tmp/tttv2_validation_t3k_dp4_20260731_gwh02_baseline_rmsnorm_ring.log` |
| Galaxy DP-8 with hardcoded Ring RMSNorm | Failed as expected | RMSNorm all-gather had no route from D0 to D12 | 27.43s | `/tmp/tttv2_validation_galaxy_dp8_20260731_gwh02_baseline_rmsnorm_ring.log` |
| Galaxy DP-8 with topology-aware RMSNorm and fixture-default Ring fabric | Failed as expected | Sampling all-gather had no route from D0 to D12 | 41.68s | `/tmp/tttv2_validation_galaxy_dp8_20260731_gwh02_restored_topology_aware_rmsnorm.log` |
| Galaxy DP-8 with explicit `FABRIC_1D` and topology-aware RMSNorm | Passed | Eight non-empty lane outputs | 130.59s | `/tmp/tttv2_validation_galaxy_dp8_20260731_gwh02_fabric1d_topology_aware_rmsnorm.log` |
| Galaxy DP-4 with explicit `FABRIC_1D` and baseline Galaxy Ring CCL | Failed as expected | Galaxy Ring CCL route failure | 15.95s | `/tmp/tttv2_validation_galaxy_dp4_20260731_gwh02_fabric1d_baseline_tt_ccl.log` |
| Galaxy DP-4 with explicit `FABRIC_1D`, Linear CCL, and topology-aware RMSNorm | Passed | Four non-empty lane outputs | 79.26s | `/tmp/tttv2_validation_galaxy_dp4_20260731_gwh02_fabric1d_linear_ccl_restored.log` |
| Galaxy DP-8 lane all-gather with shape-aware fixture default | Passed | Fixture selected `FABRIC_1D`; synchronous Linear gather matched on all four devices | 9.68s | `/tmp/tttv2_validation_fixture_fabric_dp8_allgather_20260731.log` |

The passing runs confirm:

- fabric initialization on all 32 parent devices before child creation;
- creation of the requested 1x8 T3K child mesh;
- T3K tensor-parallel model construction and teacher-forcing accuracy;
- DP-4 construction of four two-device lanes;
- DP-8 construction of eight single-device lanes;
- prefill and decode trace capture for every DP lane;
- eight-device coordinated execution with on-device top-k sampling; and
- clean child/parent device teardown after every invocation.

The Llama demo resolves its trace-region memory from the requested logical SKU
and passes the resulting `trace_region_size` explicitly to
`ttnn_mesh_device`; the generic fixture remains free of model-specific trace
policy. The now-unused `device_params` parametrization and `_trace_model_key`
sentinel were removed. Exact selectors in `models_e2e_tests.yaml` and
`release_tests.yaml` were updated to the resulting node IDs. A follow-up
cleanup inlined the one-use trace model key and changed the architecture
fixture from an unused test argument to an explicit `usefixtures` dependency.
All eight unique exact CI selectors resolve across N150, T3K, and TG
collection. Python compilation, Black, YAML parsing, and all 31 pipeline
matrix tests pass. A focused AST audit of the modified demo reports no unused
imports or function arguments.

The initial parent-first T3K and native Galaxy DP-4 smoke tests passed with
fixture-default `FABRIC_1D_RING`. Those two-device-lane smokes were useful but
not representative of Galaxy DP-8's four-device lanes. The controlled DP-8
reversion exposed route failures first in hardcoded-Ring RMSNorm and then,
after restoring topology-aware RMSNorm, in sampling under Ring fabric.

The final correction retains topology-aware distributed RMSNorm and makes the
generic fixture default depend on the requested logical mesh, rather than the
physical parent that the fixture must open. A requested `1x8` T3K child
therefore initializes its full Galaxy parent with Ring routes; a requested
native `4x8` Galaxy workload initializes the same parent with `FABRIC_1D`.
Generic CCL topology selection follows the same distinction: T3K uses Ring and
Galaxy submeshes use Linear. Focused policy tests cover single-device, small
line, T3K Ring, multi-row Galaxy fabric, and Galaxy Linear CCL cases. All 11
focused cases pass.

### 2026-07-31 Final Full-Matrix Rerun

When the external `/data` model cache became unavailable, the complete gated
`meta-llama/Llama-3.1-8B-Instruct` snapshot was downloaded to:

`/home/gwang/.cache/huggingface/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659`

The snapshot contains four safetensor shards totaling 16,060,556,376 bytes,
has no broken symlinks, and resolves config and tokenizer data in offline
mode. The full local matrix was then rerun against that snapshot:

| SKU | Exact case | Result | Pytest duration / evidence | Log |
| --- | --- | --- | --- | --- |
| N150 | Token accuracy | Passed | 33.55s call; 89.3% top-1, 97.9% top-5 | `/tmp/tttv2_full_matrix_n150_20260731.log` |
| N150 | Eval-32 repeat 3, report off | Passed | 121.93s call | `/tmp/tttv2_full_matrix_n150_20260731.log` |
| N150 | Eval-32 repeat 1, report on | Passed | 28.61s call; TTFT 115.8ms, 24.6 tok/s/u | `/tmp/tttv2_full_matrix_n150_20260731.log` |
| N300 | DP-2 | Passed | 32.90s session | `/tmp/tttv2_full_matrix_n300_dp2_20260731.log` |
| T3K | Token accuracy | Passed | 74.53s session; 89.8% top-1, 98.0% top-5 | `/tmp/tttv2_full_matrix_t3k_smoke_retry_20260731.log` |
| T3K | Eval-32 repeat 3, report off | Passed | Completed before 04:57:13 UTC | `/tmp/tttv2_full_matrix_t3k_remaining_20260731.log` |
| T3K | Eval-32 repeat 1, report on | Passed | TTFT 50.2ms, 61.1 tok/s/u | `/tmp/tttv2_full_matrix_t3k_remaining_20260731.log` |
| T3K | DP-4 | Passed | 104.33s session | `/tmp/tttv2_full_matrix_t3k_dp4_20260731.log` |
| T3K | DP-8 | Passed | 187.73s session | `/tmp/tttv2_full_matrix_t3k_dp8_20260731.log` |
| Galaxy | DP-4 | Passed | 79.06s session | `/tmp/tttv2_full_matrix_galaxy_dp4_linear_ccl_20260731.log` |
| Galaxy | DP-8 | Passed | 133.31s session | `/tmp/tttv2_full_matrix_galaxy_dp8_20260731.log` |
| Galaxy | DP-16 | Passed | 133.85s session | `/tmp/tttv2_full_matrix_galaxy_dp16_20260731.log` |
| Galaxy | DP-32 | Passed | 351.44s session | `/tmp/tttv2_full_matrix_galaxy_dp32_20260731.log` |

Overall result: **13 of 13 required local hardware nodes passed** with the
final working-tree code.

### 2026-07-31 N150 Release Performance Reporting Regression

The initial release-pipeline replacement included N150 token accuracy and the
three-repeat eval correctness node with performance reporting disabled, but it
omitted the separate one-repeat eval node that produces the release performance
artifact. This was a direct coverage regression from the TTTv1 release command;
the model-E2E pipeline already preserved both eval roles correctly.

`tests/pipeline_reorg/release_tests.yaml` now runs all three N150 roles:

- token accuracy;
- eval-32 repeat 3 with performance reporting off; and
- eval-32 repeat 1 with performance reporting on.

The restored exact node was collected and run locally with `CI=true`. It passed
in 35.04 seconds and reported 115.6ms TTFT, 24.6 tok/s/user, 788.0 aggregate
tok/s, and 40.61ms decode latency. The CI path created
`generated/benchmark_data/partial_run_2026-07-31T11:30:43+0000.pkl`, confirming
that the release performance artifact source is restored. Full run evidence is
in `/tmp/tttv2_release_n150_perf_report_20260731.log`.

### 2026-07-31 DP Performance Collection

The earlier full-matrix DP runs validated correctness, but the DP smoke helper
discarded the `run_perf_benchmark()` result and therefore emitted no benchmark
metrics. The helper now attaches a `BenchmarkProfiler` and passes the completed
result through the same performance reporting and CI artifact path used by the
eval cases. Generated-text logging is disabled for DP because each lane has its
own tokenizer and output stream.

All seven DP nodes were rerun as isolated pytest processes against the local
model snapshot. Each passed and reported the following performance:

| SKU | DP | TTFT | tok/s/user | Aggregate tok/s | Decode latency | Pytest duration | Log |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| N300 | 2 | 111.9ms | 30.0 | 60.0 | 33.35ms | 33.10s | `/tmp/tttv2_dp_perf_n300_dp2_20260731.log` |
| T3K | 4 | 77.3ms | 44.0 | 175.8 | 22.75ms | 104.41s | `/tmp/tttv2_dp_perf_t3k_dp4_20260731.log` |
| T3K | 8 | 109.1ms | 27.9 | 223.3 | 35.82ms | 168.27s | `/tmp/tttv2_dp_perf_t3k_dp8_20260731.log` |
| Galaxy | 4 | 47.9ms | 60.7 | 242.9 | 16.47ms | 79.33s | `/tmp/tttv2_dp_perf_galaxy_dp4_20260731.log` |
| Galaxy | 8 | 54.5ms | 53.1 | 424.9 | 18.83ms | 135.93s | `/tmp/tttv2_dp_perf_galaxy_dp8_20260731.log` |
| Galaxy | 16 | 76.2ms | 39.5 | 632.7 | 25.29ms | 163.22s | `/tmp/tttv2_dp_perf_galaxy_dp16_20260731.log` |
| Galaxy | 32 | 108.9ms | 22.2 | 711.4 | 44.98ms | 383.66s | `/tmp/tttv2_dp_perf_galaxy_dp32_20260731.log` |

DP performance collection result: **7 of 7 nodes passed and emitted metrics**.
The Galaxy DP-32 node used a 900-second pytest timeout because initialization
of 32 independent model replicas legitimately exceeds the repository's default
300-second timeout; it completed in 383.66 seconds.

#### DP telemetry parity refinement

DP performance remains **telemetry-only**, matching TTTv1. TTTv1 called its
performance verifier only for `ci-32`; the `ci-b1-DP-*` cases logged metrics,
emitted CI artifacts, and passed or failed on functional execution. TTTv2 DP
therefore deliberately calls `_report_performance(..., expected={})`: the
`performance` string selects the model optimization profile and does not turn
the DP smoke into a performance gate. The code documents this distinction
explicitly.

The TTTv2 artifact path was also brought closer to TTTv1 telemetry and identity
parity:

- every steady-state decode iteration is emitted as `time_to_token_N`;
- named token checkpoints and `avg_decode_time_first_128` are emitted;
- Hugging Face hub-cache snapshot paths resolve to canonical model identity
  instead of a snapshot hash;
- `data_parallel` and `tensor_parallel` are explicit configuration dimensions;
- input length is the actual maximum prompt length rather than configured model
  capacity; and
- output length is the requested decode count, matching TTTv1's dimension.

DP correctness validation also precedes telemetry emission, matching TTTv1's
ordering. Lane count, non-empty output for every lane, and special-token checks
all complete before `_report_performance()` can write a CI benchmark partial.
Consequently, a failed DP job cannot leave a new artifact for workflow steps
guarded by `if: !cancelled()` to process.

An N300 DP-2 run with `CI=true` and `HF_MODEL` set to the local snapshot path
passed in 32.80 seconds. Its final artifact records the TTTv1-compatible base
identity `llama-3.1-8b`, preserves `Llama-3.1-8B-Instruct` separately as
`model_variant`, and records batch size 2, DP=2, TP=1, input length 115, output
length 200, 199 per-token measurements, token-1 and token-128 checkpoints, and
the first-128 average. The per-token series uses observed iteration timings
rather than copies of the aggregate mean (160 distinct values when rounded to
microsecond precision). All 206 measurements have null targets, directly
confirming that the DP artifact is telemetry-only. Evidence is in
`/tmp/tttv2_dp2_ci_base_model_identity_20260731.log` and
`generated/benchmark_data/partial_run_2026-07-31T12:00:50+0000.pkl`.

The same base/variant identity is used for token-accuracy artifacts. A scope
error found during review initially referenced `model_variant` in
`_measure_teacher_forcing_accuracy()` even though it had only been defined in
its caller. The identity is now resolved inside the artifact-producing
function, while the caller discards the unneeded variant. The exact N150 token
accuracy node passed under `CI=true` in 38.45 seconds (89.3% top-1, 97.9%
top-5) and emitted a `demo_accuracy` partial with base model
`llama-3.1-8b`, variant `Llama-3.1-8B-Instruct`, and 512-token input/output
dimensions. Evidence is in
`/tmp/tttv2_n150_ci_accuracy_model_variant_scope_final_20260731.log` and
`generated/benchmark_data/partial_run_2026-07-31T12:20:08+0000.pkl`.

The rerun also supplied two useful negative controls:

- Choosing `FABRIC_1D` from the opened `4x8` parent for a requested T3K child
  failed with no route from D0 to D28. Selecting from the requested `1x8`
  topology changed the parent to Ring and the same token-accuracy node passed.
- With a fresh Galaxy reset and native `FABRIC_1D`, forcing generic 8-device
  Galaxy CCL operations to Ring failed DP-4 at layer-0 setup with the same
  D0-to-D28 route error. Changing only Galaxy CCL selection to Linear made the
  exact node pass in 79.06 seconds.

The eval and DP cases are separate CI nodes. An artificial combined T3K pytest
process allowed both eval nodes to pass, but transitioning from their retained
trace/event state directly into DP-4 triggered an event-ID assertion. Running
DP-4 and DP-8 in their normal isolated-node processes passed and closed the
parent cleanly.

Final implementation and verification status:

- `ttnn_mesh_device` chooses its default fabric from the requested logical
  mesh: one-row meshes with at least eight devices use Ring; multi-row meshes
  use `FABRIC_1D`.
- The fixture continues to open the full physical parent before deriving a
  requested child submesh.
- Generic CCL topology selects Ring only for an eight-device T3K and falls
  back to Linear for multi-device Galaxy submeshes.
- Topology-aware RMSNorm remains in place so its collectives use the same
  generic CCL policy.
- DP smoke nodes report TTFT, per-user and aggregate throughput, and decode
  latency, and retain CI benchmark artifact generation when `CI=true`.
- All 11 focused fixture/CCL policy cases pass.
- Python compilation, Black formatting checks, and `git diff --check` pass.

The first parent-first attempt was made immediately after a failed
direct-subset invocation and encountered stale Ethernet dispatch kernels,
followed by a full-parent Device 16 fabric timeout. The fixture reported that
attempt as skipped. A full `tt-smi -glx_reset` reinitialized all 32 boards; the
token-accuracy, DP-4, and DP-8 parent-first runs then passed sequentially
without another topology, fabric, device, or timeout failure. This sequence
localizes the former T3K failure to direct subset-only fabric initialization,
not the T3K model or DP paths.

### 2026-07-31 Final CI-Mode Double-Check

The complete 13-node hardware matrix was rerun once more from the final working
tree, with `CI=true`, offline local model weights, and each CI node in its own
pytest process. The machine received a full `tt-smi -glx_reset` before the
N150/N300 group, before the T3K group, and before the native Galaxy group. T3K
used the fixture-selected Ring fabric/CCL policy; the multi-row Galaxy parent
used `FABRIC_1D` with Linear model CCL operations. **All 13 nodes passed.**

| SKU | Case | Result | Observed result | Wall time | Log |
| --- | --- | --- | --- | ---: | --- |
| N150 | Token accuracy | Passed | top-1 89.3%, top-5 97.9% | 35.02s | `/tmp/tttv2_final_matrix_n150_accuracy_20260731.log` |
| N150 | Eval-32, repeat 3, report off | Passed | correctness-only; no artifact | 78.53s | `/tmp/tttv2_final_matrix_n150_eval_repeat3_20260731.log` |
| N150 | Eval-32, repeat 1, report on | Passed | TTFT 115.8ms; 24.6 tok/s/u; 788.0 tok/s | 35.10s | `/tmp/tttv2_final_matrix_n150_eval_report_20260731.log` |
| N300 | DP-2 | Passed | TTFT 112.1ms; 30.0 tok/s/u; 60.0 tok/s | 33.11s | `/tmp/tttv2_final_matrix_n300_dp2_20260731.log` |
| T3K | Token accuracy | Passed | top-1 89.8%, top-5 98.0% | 75.84s | `/tmp/tttv2_final_matrix_t3k_accuracy_20260731.log` |
| T3K | Eval-32, repeat 3, report off | Passed | correctness-only; no artifact | 51.15s | `/tmp/tttv2_final_matrix_t3k_eval_repeat3_20260731.log` |
| T3K | Eval-32, repeat 1, report on | Passed | TTFT 53.4ms; 57.4 tok/s/u; 1837.8 tok/s | 22.30s | `/tmp/tttv2_final_matrix_t3k_eval_report_20260731.log` |
| T3K | DP-4 | Passed | TTFT 77.6ms; 43.9 tok/s/u; 175.7 tok/s | 107.71s | `/tmp/tttv2_final_matrix_t3k_dp4_20260731.log` |
| T3K | DP-8 | Passed | TTFT 109.4ms; 27.9 tok/s/u; 223.3 tok/s | 168.59s | `/tmp/tttv2_final_matrix_t3k_dp8_20260731.log` |
| Galaxy | DP-4 | Passed | TTFT 48.3ms; 60.8 tok/s/u; 243.0 tok/s | 79.34s | `/tmp/tttv2_final_matrix_galaxy_dp4_20260731.log` |
| Galaxy | DP-8 | Passed | TTFT 54.8ms; 53.1 tok/s/u; 425.0 tok/s | 133.70s | `/tmp/tttv2_final_matrix_galaxy_dp8_20260731.log` |
| Galaxy | DP-16 | Passed | TTFT 76.0ms; 39.3 tok/s/u; 628.0 tok/s | 129.75s | `/tmp/tttv2_final_matrix_galaxy_dp16_20260731.log` |
| Galaxy | DP-32 | Passed | TTFT 108.4ms; 22.4 tok/s/u; 718.3 tok/s | 356.81s | `/tmp/tttv2_final_matrix_galaxy_dp32_20260731.log` |

The benchmark audit matched the intended reporting policy exactly: all 11
reporting nodes emitted one artifact each, while the two repeat-3/report-off
nodes emitted none. Every emitted artifact used canonical
`ml_model_name=llama-3.1-8b`, retained
`model_variant=Llama-3.1-8B-Instruct`, and had null performance targets. Every
performance artifact included explicit DP and TP dimensions. Input dimensions
were the actual encoded prompt lengths (115 for DP workloads and 712 for the
eval workload), not model capacity; requested output dimensions were 2048 for
DP-4/8, 200 for DP-2/16/32 and eval, and 512 for accuracy.

Final non-hardware verification also passed:

- Python compilation passed for the changed demo/runtime and topology modules.
- Black left all seven checked Python files unchanged.
- The 21 focused runtime-helper and mesh-policy tests passed.
- All 98 CI Hugging Face trace-region-resolution cases passed.
- Both pipeline YAML files parsed, and the release pipeline contains all three
  N150 roles: accuracy, three-repeat correctness/report-off, and one-repeat
  performance/report-on.
- `git diff --check` passed.

### Final Local Hardware Matrix

| SKU / topology | Exact coverage | Final status |
| --- | --- | --- |
| N150 | Token accuracy; eval-32 repeat 3; eval-32 repeat 1 with report | Passed |
| N300 | DP-2 | Passed |
| T3K child of Galaxy | Token accuracy; eval-32 repeat 3; eval-32 repeat 1 with report; DP-4; DP-8 | Passed |
| Native Galaxy | DP-4, DP-8, DP-16, DP-32 | Passed |

No required local hardware test remains.

## Remaining Work

No local hardware validation remains, and no GitHub Actions run is active.
The remaining release workflow is:

1. After the uncommitted replacement is reviewed and intentionally committed,
   dispatch fresh N150, T3K, and Galaxy GitHub runs against that final SHA.
2. Verify selected jobs, exact pytest nodes, benchmark artifacts, and aggregate
   conclusions.
