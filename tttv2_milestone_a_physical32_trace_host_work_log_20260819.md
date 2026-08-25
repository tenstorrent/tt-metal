# Milestone A Physical-32 Trace Host Coverage Work Log

Date: 2026-08-19 UTC

## Goal

Close the Milestone A physical batch 32 Wormhole trace host-coverage gap with the narrowest
meaningful host-only runtime contract tests. Do not run TT hardware tests and do not edit RMSNorm,
attention, or Galaxy resource implementation files.

## Checkpoint 1: Scope and Existing Coverage

- Inspected `tttv2_2d_modules_plan.md`, especially planning and trace coverage lines 460-476.
- Confirmed existing host tests cover the physical-32 policy matrix, metadata padding, fallback
  behavior, sampling safety, and signature equality across active row counts.
- Identified the remaining host gap: no test connected the physical-32 `PrefillRuntime` capture
  plan to the real `ProgramCompiler` and `TraceCompiler`, then reused that captured artifact while
  refreshing active rows and slot order.
- Chose a test-only lifecycle contract over production changes. The test uses mocked TTNN trace
  submission and does not open or reset hardware.

## Checkpoint 2: Implementation

- Added a parameterized lifecycle test for padded sequence lengths 128, 1024, and 2048.
- Each case captures one physical-batch-32 trace from 16 active rows, then replays the same program
  and trace identity for 31 and 32 active rows with reversed slot order.
- The contract verifies one capture, two replay submissions, request-data refresh into the same
  persistent inputs, physical-row padding, stable program/trace keys, and prefill replay counters.
- Production runtime and excluded module/resource implementation files remain untouched.

## Checkpoint 3: First Focused Run

- Command: `pytest -q models/common/tests/llm_runtime/test_prefill_runtime.py::test_physical_32_trace_capture_replays_refreshed_active_rows_and_slots`
- Result: 3 failed before capture in 1.03 seconds.
- Cause: the test patched `runtime.input_stager`, but the intentionally narrow ownership path is
  `runtime.trace.hooks.input_stager`.
- Corrected only the test collaborator path; no production change was needed.

## Checkpoint 4: Lifecycle Contract Green

- Re-ran the focused lifecycle node after correcting the collaborator path.
- Result: `3 passed in 0.86s`.
- Verified host capture signatures for padded sequence lengths 128, 1024, and 2048, each with
  `padded_batch_size=32` and two successful replay submissions.

## Checkpoint 5: Physical-32 Regression Set Green

- Ran the new lifecycle cases with the surrounding fixed physical-32 policy, default-policy,
  supported-length, and token-budget host contracts.
- Result: `23 passed, 121 deselected in 3.52s`.
- This verifies the new trace lifecycle contract without weakening the existing planner and
  default-policy expectations.

## Checkpoint 6: Full File and Scope Verification

- Ran the complete prefill runtime host suite: `144 passed in 21.89s`.
- `git diff --check -- models/common/tests/llm_runtime/test_prefill_runtime.py` passed.
- Added an explicit cleanup assertion for release of the mocked trace ID.
- Confirmed this goal changed only the prefill runtime test and this dedicated work log. Existing
  dirty RMSNorm, attention, and Galaxy resource paths were not edited by this goal.
- No TT hardware test, `tt-smi`, or device reset command was run.

## Checkpoint 7: Final Focused Verification

- Re-ran all three lifecycle cases after adding the cleanup-release assertion.
- Result: `3 passed in 0.84s`.
- Final targeted `git diff --check` passed.
