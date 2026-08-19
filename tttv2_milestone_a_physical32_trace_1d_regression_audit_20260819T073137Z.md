# Milestone A Physical-32 Trace and 1D Regression Audit

Date: 2026-08-19 UTC

## Goal and Constraints

Audit the Milestone A physical-32 prefill trace gate and the existing 1D/default-runtime
regression requirements. This audit did not run TT hardware and did not edit production files.

## Executive Finding

Milestone A's host policy contract is implemented and focused coverage passes, but there is no
existing pytest node that captures and replays a physical-batch-32 prefill trace on a Wormhole
Galaxy `(8, 4)` mesh. The current Attention2D Galaxy hardware test performs eager single-row
prefill against a cache sized for 32 users; it does not exercise the existing `CONCAT_32` recipe
and contains no trace API call. The physical-32 trace gate is therefore a real missing test and
qualification result, not a stale failure.

The complete existing 1D module suite currently collects 2,006 selected tests. A prior serialized
run was stopped after 497.05 seconds at `96 passed, 2 failed, 24 skipped`; the two failures are the
two parametrizations of one host-only Attention1D fake test. The full default-runtime suite was
run before and after the Milestone A runtime diff and retained the same two host-only failures.
All four failures are stale test expectations or incomplete fakes, but they remain required test
fixes because the exit gate requires green suites, not merely no additional failures.

## Plan Requirements

The controlling requirements are:

- `tttv2_2d_modules_plan.md:460-476`: host planning must cover active rows 15/16/31/32,
  padding, row/slot order, cached and over-2048 fallback, sampling, physical-32 program/trace
  signatures, unchanged default policy, and all pre-existing planner tests. Physical-32 traces
  must qualify at sequence 128 first and then through 2048.
- `tttv2_2d_modules_plan.md:597-612`: every module must pass WH `(8, 4)` decode/prefill,
  preserve unchanged 1D implementations, pass the existing 1D module suite, and preserve every
  pre-existing default-runtime test and expectation.

There is a scope tension in the plan: Milestone A contains reusable modules and common runtime but
no reconstructed Galaxy model/executor, while common `TraceCompiler` capture is graph/executor
oriented. A T3K full-model trace is useful reference coverage but cannot qualify the Galaxy
physical-32 gate. The least invasive Milestone A solution is a test-only WH Galaxy harness that
binds `PrefillRuntime`/trace lifecycle to the existing Attention2D concat-32 recipe, or a direct
module trace test if the intended gate is explicitly narrowed to module graph capture/replay.

## Existing Physical-32 Host Coverage

Focused command run during this audit:

```bash
pytest -q \
  models/common/tests/llm_runtime/test_prefill_runtime.py::test_omitted_batched_policy_preserves_only_legacy_contiguous_q128_behavior \
  models/common/tests/llm_runtime/test_prefill_runtime.py::test_omitted_batched_policy_keeps_mixed_cache_hits_sequential \
  models/common/tests/llm_runtime/test_prefill_runtime.py::test_omitted_batched_policy_keeps_legacy_padded_partial_trace_eligible \
  models/common/tests/llm_runtime/test_prefill_runtime.py::test_fixed_physical_32_policy_enforces_active_row_floor \
  models/common/tests/llm_runtime/test_prefill_runtime.py::test_fixed_physical_32_policy_preserves_metadata_and_sanitizes_padding \
  models/common/tests/llm_runtime/test_prefill_runtime.py::test_fixed_physical_32_policy_keeps_cached_and_long_sequences_sequential \
  models/common/tests/llm_runtime/test_prefill_runtime.py::test_fixed_physical_32_policy_keeps_slot_stable_sampling_on_safe_batched_output \
  models/common/tests/llm_runtime/test_prefill_runtime.py::test_fixed_physical_32_policy_rejects_sampling_without_batched_output \
  models/common/tests/llm_runtime/test_prefill_runtime.py::test_fixed_physical_32_signatures_ignore_active_row_count \
  models/common/tests/llm_runtime/test_prefill_runtime.py::test_batched_prefill_accepts_arbitrary_uniform_length_buckets_through_chunk_limit \
  models/common/tests/llm_runtime/test_prefill_runtime.py::test_batched_prefill_strict_token_budget_boundary
```

Result: `20 passed in 3.08s`.

This proves policy and signature behavior on the host, including physical batch 32 and sequence
length 2048. It does not allocate a device trace, capture a Galaxy graph, replay it, verify output
PCC, or verify KV-cache PCC.

## Missing Physical-32 Hardware Test

The closest current node is:

```bash
pytest -q models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py::test_attention_2d_wh_galaxy_decode_and_prefill_repeat
```

It is not the trace gate:

- prefill calls use `PrefillMetadata(sequence_length, user_ids=(0,))`;
- only eager `module.prefill_forward` is invoked;
- no `begin_trace_capture`, `end_trace_capture`, `execute_trace`, `TraceCompiler`, or
  `TracedExecutor` call appears in any `*wh_galaxy.py` module test;
- Attention2D production supports concat-32 only when `user_ids` has exactly 32 entries, and host
  test `test_concat32_contiguous_cache_fill_slices_each_physical_row` covers only mocked cache
  behavior.

Required test-only qualification should prove, in this order:

1. physical rows `(0..31)`, concat-32 input geometry, page-table padding/slot mapping, and the
   `PrefillRowMode.CONCAT_32` recipe at sequence 128;
2. eager compile followed by trace capture, at least two replays with refreshed runtime inputs,
   output PCC `>= 0.99`, KV-cache PCC `>= 0.99`, stable trace identity, and clean release;
3. repeat for the model geometry not covered by the first case;
4. add supported padded lengths through 2048. Based on current runtime trace fixtures the intended
   bounded family is most plausibly `(128, 1024, 2048)`, but the plan should name this tuple
   explicitly before implementing the hardware test because current Attention2D Galaxy resources
   are only provisioned for 128 and 2048.

Estimated serialized runtime after a test exists:

- sequence 128 bring-up, one model: 2-5 minutes cold;
- second model at 128: 2-5 minutes;
- 1024 and 2048 expansion, both models: 8-20 minutes total;
- full repeated trace matrix with teardown margin: approximately 15-35 minutes.

These are planning estimates, not measurements. Run each case as a separate pytest process so a
fault does not contaminate later cases. Use `tt-smi -r` only after a hang/fault or failed teardown.

## Default-Runtime Regression Status

Exact command:

```bash
pytest -q models/common/tests/llm_runtime
```

Recorded pre-change baseline: `1017 passed, 2 failed, 1 skipped in 201.83s`.

Recorded post-change result: `1027 passed, 2 failed, 1 skipped in 200.88s`.

Current focused reproduction: both stale nodes still fail in 1.40 seconds:

```bash
pytest -q \
  models/common/tests/llm_runtime/test_config.py::test_executor_config_has_exact_static_policy_owners_and_is_frozen \
  models/common/tests/llm_runtime/test_execution.py::test_traced_prefill_compile_does_not_interpret_request_eligibility
```

Classification and required test-only fixes:

- `test_executor_config_has_exact_static_policy_owners_and_is_frozen` is stale because the expected
  exact field list omits the already-present
  `allow_batched_prefill_with_device_sampling_for_diagnostics`. Update the expected contract only
  after confirming that field remains intended product-owned policy.
- `test_traced_prefill_compile_does_not_interpret_request_eligibility` uses an old
  `SimpleNamespace` capture-plan fake without `prime` and `release_prime_output`. Update the fake to
  the current capture-plan protocol. No production fallback should be added for an incomplete fake.

After those test fixes, rerun the two nodes, then `test_prefill_runtime.py`, then the complete
1,029-test collected runtime suite. Expected wall time is about 3.5 minutes.

## Existing 1D Module Regression Status

The exact suite is:

```bash
pytest -q \
  models/common/tests/modules/attention/test_attention_1d.py \
  models/common/tests/modules/embedding/test_embedding_1d.py \
  models/common/tests/modules/lm_head/test_lm_head_1d.py \
  models/common/tests/modules/mlp/test_mlp_1d.py \
  models/common/tests/modules/rmsnorm/test_rmsnorm_1d.py \
  models/common/tests/modules/rope/test_rope_1d.py \
  models/common/tests/modules/sampling/test_penalties_1d.py \
  models/common/tests/modules/sampling/test_sampling_1d.py
```

Current selected collection by file:

| File | Selected tests |
| --- | ---: |
| Attention1D | 516 |
| Embedding1D | 129 |
| LMHead1D | 40 |
| MLP1D | 242 |
| RMSNorm1D | 281 |
| RotarySetup1D | 264 |
| Penalties1D | 394 |
| Sampling1D | 140 |
| Total | 2,006 |

The two prior failures reproduce without hardware:

```bash
pytest -q \
  models/common/tests/modules/attention/test_attention_1d.py::test_attention_prefill_selects_scalar_or_tensor_chunk_start_api
```

Result: both `[False]` and `[True]` fail before device execution because the test's `cfg` fake lacks
`li_qkv_prefill_compute_kernel_cfg`; related kernel fields were placed on an unused `arch_cfg`
object. This is a stale test fake, not a Milestone A regression and not evidence of a production
Attention1D defect. Repair the fake in the test only, then rerun both parametrizations before
reserving hardware.

No changed `models/common/modules/**/*_1d.py` file appears in the working tree. That satisfies the
structural half of the 1D gate but does not replace the full test run.

## Recommended Serialized Verification Sequence

1. Host preflight, no hardware: repair the four stale tests, run their focused nodes, run the 20
   physical-32/default-policy nodes above, and run `pytest -q models/common/tests/llm_runtime`.
2. Structural preflight: confirm `git diff --name-only -- 'models/common/modules/**/*_1d.py'` is
   empty and run `git diff --check`.
3. Implement the missing test-only physical-32 trace gate. First collect it and run host-side
   setup/contract tests without opening hardware.
4. Hardware lane A: run one Llama-or-Qwen sequence-128 physical-32 trace node in a fresh process.
   Require capture, two replays, output/KV PCC, and clean teardown.
5. Hardware lane B: run the second model at sequence 128. Do not expand lengths until both 128
   cases pass.
6. Hardware lane C: run the explicitly resolved padded sequence family through 2048, one node per
   process. Reset only after contamination, not between clean cases.
7. Hardware lane D: run the 1D suite one file at a time in the table order, preserving per-file
   JUnit logs. Attention first exposes the largest and historically slowest surface; follow with
   MLP/RMSNorm, then smaller embedding/LM-head/RoPE, then penalties/sampling.
8. Final proof: rerun the full default-runtime suite after all test-only fixture corrections and
   summarize physical-32 trace IDs, replay counts, PCC, KV PCC, teardown, 1D totals, skips, and
   durations in the Milestone A work log/status ledger.

Projected 1D hardware time from the incomplete measured run is at least 2.9 hours at the observed
average, with long-context and per-test timeout tails making 4-8 hours a safer reservation. Keep
the files serialized; parallel pytest workers would violate the single-Galaxy constraint and make
failure recovery ambiguous.

## Exit-Gate Classification

- Physical-32 host planning/signature contract: **green** (`20 passed`).
- Default behavior equivalence: **no new failures**, but **not exit-gate green** until two stale
  runtime tests are repaired and the full suite passes.
- Unchanged 1D implementation files: **green structurally**.
- Existing 1D suite: **incomplete**; two stale host fake failures require test fixes and the full
  2,006-case serialized run remains outstanding.
- WH physical-32 prefill trace: **missing test and missing hardware result**.
