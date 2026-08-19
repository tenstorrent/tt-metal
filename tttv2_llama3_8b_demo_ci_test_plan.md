# TTTv2 Llama 3.1 8B CI Replacement Plan

## Goal

Replace the existing Wormhole TTTv1 Llama 3.1 8B CI invocations with TTTv2
invocations while preserving the existing test coverage. This is a replacement,
not a CI expansion or a redesign of benchmark target resolution.

Blackhole remains on TTTv1. The legacy Wormhole `--use_hf_rope` invocation also
remains on TTTv1 until TTTv2 supports native HF-layout RoPE.

## Replacement Mapping

The primary Wormhole registration preserves the existing four test legs:

1. TTTv1 token matching becomes TTTv2 `token-accuracy`.
2. TTTv1 `ci-eval-32` with the default three repeats and no performance report
   becomes TTTv2 `eval-32`, `repeat_batch-3`, `prefetcher-off`,
   `perf-report-off`.
3. TTTv1 `ci-eval-32` with one repeat and performance reporting becomes TTTv2
   `eval-32`, `repeat_batch-1`, `prefetcher-off`, `perf-report-on`.
4. TTTv1 token matching with `--use_hf_rope` remains unchanged.

TTTv2 has no TTTv1 DRAM prefetcher. Both TTTv2 eval cases therefore identify
`prefetcher-off`; this is the only unsupported implementation switch dropped
from the replacement.

The eval cases use the same 32-prompt corpus as TTTv1:

```text
models/tt_transformers/demo/sample_prompts/eval_repeat_prompts_batch32.json
```

The three-repeat case compares generated token IDs after rotating and restoring
prompt order. The one-repeat case runs the same workload and emits the
performance report.

The repeated evaluation exposed an existing traced-prefill result-lifetime
bug: requests that shared a trace also shared its persistent output buffer, but
the executor deferred reading every result until all replays completed. The
replacement consumes each traced replay while iterating the prepared requests,
before the next replay can overwrite the shared output. A focused executor
regression test covers this ordering.

## CI Registrations

### Primary

`tests/pipeline_reorg/models_e2e_tests.yaml` uses one shared Wormhole command
branch:

- `wh_n150` sets `MESH_DEVICE=N150`;
- `wh_llmbox_perf` sets `MESH_DEVICE=T3K`.

Each branch selects exactly these TTTv2 pytest IDs:

```text
token-accuracy-repeat_batch-1-prefetcher-off
eval-32-repeat_batch-3-prefetcher-off-perf-report-off
eval-32-repeat_batch-1-prefetcher-off-perf-report-on
```

The commands use pytest keyword selectors so repeat count, prefetcher state,
and performance reporting state remain visible in the registry.

### Existing Data Parallel Coverage

Existing TTTv1 DP registrations are replaced, not expanded:

- T3K: DP-4 and DP-8;
- Galaxy: DP-4, DP-8, DP-16, and DP-32.

The TTTv2 demo composes production `Llama3Executor` instances through
`LaneGroupExecutor`. The generic mesh fixture selects Ring fabric for a
requested one-row mesh with at least eight devices and `FABRIC_1D` for a
requested multi-row mesh such as Galaxy `(4, 8)`. The fixture still opens the
full physical parent before creating a child, but the requested workload
topology determines which routes the parent must provide. Generic CCL topology
selection matches that policy: T3K uses Ring, while Galaxy submeshes use
Linear.

### Release Mirrors

`tests/pipeline_reorg/release_tests.yaml` mirrors:

- N150 token accuracy;
- N150 three-repeat eval without performance reporting;
- the existing Galaxy DP factors.

No new release test case is added.

## Target and Trace Scope

`models/model_targets.yaml`, the target resolver, and the benchmark validator
remain unchanged. Existing model/SKU/batch/sequence targets remain
authoritative.

T3K TTTv2 trace capture measured 53,764,096 bytes, so
`models/model_trace_region_sizes.yaml` raises only the existing
`llama3.1-8b/wh_llmbox_perf` allocation from 50 MB to 60 MB. The corresponding
legacy-SKU assertion changes from 50 MB to 60 MB. No dedicated trace test or
unrelated trace expectation is added.

## Non-Goals

- No new primary CI workload.
- No sampling/profile/workload dimensions in centralized targets.
- No new `models/model_targets.yaml` entries or aliases.
- No strict benchmark-artifact workflow policy change.
- No new standalone `llama3_8b/test_demo.py`.
- No Blackhole migration.
- No TTTv2 no-op prefetcher or HF RoPE compatibility flag.
- No plan or completion-report commit.

## Verification Plan

GitHub Actions dispatches are intentionally excluded from the current
validation pass.

### 1. Static and Software Validation

Run formatting, syntax, and whitespace checks over changed Python files:

```bash
python3 -m black --fast --check <changed-python-files>
python3 -m py_compile <changed-python-files>
git diff --check
```

Run focused software suites:

```bash
python_env/bin/pytest -q \
  models/common/tests/demos/test_run_helpers.py \
  models/common/tests/test_llama3_8b_hf_adaptor.py \
  models/common/tests/llm_runtime/test_lane_group.py \
  models/common/tests/llm_runtime/test_execution.py \
  models/common/tests/llm_runtime/test_prefill_runtime.py \
  models/common/tests/llm_runtime/test_llama3_8b_integration.py \
  models/common/tests/llm_runtime/test_llama3_8b_model_contract.py \
  models/common/tests/modules/rmsnorm/test_rmsnorm_1d.py \
  models/tt_transformers/tests/test_trace_region_sizes.py
```

The unrelated pre-existing Gemma 3 27B hard-coded trace expectation may fail:
the test expects 30,000,000 while the committed YAML contains 30,100,000. Do
not include a Gemma correction in this replacement.

### 2. Collection and Registry Validation

Collect the demo for `N150`, `T3K`, and `TG`. Verify:

- each primary keyword command selects exactly one test;
- the combined primary expression selects exactly three tests per Wormhole SKU;
- every exact DP node in the model and release registries exists;
- the N150 release exact nodes exist;
- YAML parses and model matrix expansion succeeds;
- existing tier time budgets pass.

### 3. Local Hardware Validation

Use the local Llama 3.1 8B checkpoint and a writable TTTv2 tensor cache. Set:

```bash
HF_HUB_OFFLINE=1
SAMPLING_MODE=on_device_topk
PIPELINE_READBACK=1
```

Run hardware sequentially because the TT device cannot be shared.

#### N150

Run the exact primary replacement selectors:

```text
performance and token-accuracy-repeat_batch-1-prefetcher-off
performance and eval-32-repeat_batch-3-prefetcher-off-perf-report-off
performance and eval-32-repeat_batch-1-prefetcher-off-perf-report-on
```

#### T3K

Run the same three primary selectors, then exact DP-4 and DP-8 nodes.

#### Galaxy

Run exact DP-4, DP-8, DP-16, and DP-32 nodes.

The release entries duplicate N150 and Galaxy nodes already exercised above
and therefore do not require a second hardware execution.

### 4. Artifact Validation

When local runs use `CI=true` and emit benchmark JSON, validate the generated
artifacts against the unchanged targets:

```bash
python3 .github/scripts/utils/validate_perf_targets.py \
  --path-profile repo-root \
  --sku <wh_n150-or-wh_llmbox_perf> \
  --strict-missing
```

If the local checkpoint path cannot resolve the canonical model alias under
`CI=true`, run the exact hardware workload without CI artifact enforcement and
record that artifact validation remains a GitHub-only check. Do not add aliases
or target dimensions solely for local execution.

### 5. GitHub-Only Follow-Up

Not run in the current pass:

- All Model Tests dispatch for N150;
- All Model Tests dispatch for T3K primary plus DP;
- All Model Tests dispatch for Galaxy DP;
- final-SHA job selection and benchmark-artifact inspection.

## Current Local Results

Validated on 2026-07-29 without GitHub Actions:

- formatting, syntax, whitespace, YAML parsing, matrix expansion, collection,
  and tier time-budget checks pass;
- the final focused implementation run completed with `206 passed` and `218`
  hardware/topology skips after providing the local checkpoint to RMSNorm;
- traced-prefill execution/runtime tests pass (`96 passed`);
- trace-region tests have one known unrelated baseline failure: Gemma 3 27B
  expects 30,000,000 while the committed YAML contains 30,100,000;
- N150 token accuracy passes (`37.99s`);
- N150 eval-32 repeat-three, performance-report-off passes (`78.77s`);
- N150 eval-32 repeat-one, performance-report-on passes (`35.01s`);
- the legacy TTTv1 N150 repeat-three comparison also passes (`223.66s`),
  confirming equivalent legacy coverage.

T3K execution is currently blocked by the host's T3K-submesh/fabric health,
not by a test assertion. Earlier T3K and native Galaxy attempts failed before
model setup. Diagnostics included:

```text
Fabric Router Sync: Timeout after 10000 ms on Device 17
Fabric Router Sync: Timeout after 10000 ms on Device 24
Timed out waiting for ETH heartbeat ... ETH core e7-6 ... Stuck at 0xabcd935c
```

On host `UF-EV-B6-GWH02`, `tt-smi -r` did not recover the native Galaxy mesh:
the first DP-4 smoke retry failed during logical-to-physical topology mapping.
The fallback `tt-smi -glx_reset` did recover it. After that reset, the exact
Galaxy DP-4, DP-8, DP-16, and DP-32 nodes all passed sequentially, including
model construction, trace capture, and execution. The larger cold-cache runs
needed timeout retries, but those retries resumed from the populated cache and
passed without another fabric or device failure.

The exact T3K token-accuracy smoke test initially still failed after the
successful fallback Galaxy reset because the demo used the root
`mesh_device` fixture, which directly opened only the requested 1x8 subset.
The demo now uses the parent-first `ttnn_mesh_device` fixture: on Galaxy it
opens the full 4x8 parent and derives the requested 1x8 child mesh. With this
fixture correction, the exact T3K token-accuracy node passed with 89.8% top-1
and 98.0% top-5 accuracy. After moving trace-size resolution entirely into the
demo, the refined implementation passed the same node in 78.01 seconds. The
exact T3K DP-4 and DP-8 nodes
then passed in 134.59 and 176.13 seconds respectively. Both constructed all
lanes, captured prefill/decode traces, executed on-device top-k sampling, and
closed the full parent cleanly. Native Galaxy DP validation also remains
unblocked.

On 2026-07-30, a retry on host `UF-EV-B6-GWH02` narrowed the failure scope:

- the exact N150 token-accuracy node passed with 89.3% top-1 and 97.9% top-5
  accuracy;
- the exact N300 DP-2 node passed, including two-lane model construction,
  trace capture, and on-device top-k sampling;
- the exact T3K token-accuracy node still failed in `mesh_device` setup with a
  Device 17 fabric-router synchronization timeout.

These results show that single-device and N300 DP-2 execution are healthy on
this host. Before the chassis reset, the exact Galaxy DP-4/8/16/32 nodes also
failed while opening the native 32-chip parent mesh, with first timeouts on
Devices 16, 3, 0, and 0 respectively. After `tt-smi -glx_reset`, all four
native Galaxy nodes passed. The subsequent parent-first fixture correction
also made the T3K smoke pass. The combined evidence identifies direct
subset-only fabric initialization as the failure mode rather than the native
Galaxy parent mesh, T3K model path, or model implementation.

All required local hardware nodes are now complete. No additional local
N150, N300, T3K, or Galaxy test remains for this plan.

## Acceptance Criteria

1. Software suites pass except for explicitly identified unrelated baseline
   failures.
2. Every replacement selector and exact release/DP node collects exactly as
   registered.
3. N150 and T3K primary hardware cases pass.
4. T3K DP-4/DP-8 and Galaxy DP-4/8/16/32 pass.
5. No hardware is used concurrently.
6. No target-schema, unrelated trace, or additional CI-test change is
   introduced during verification.
