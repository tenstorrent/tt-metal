---
name: LLK Validation Agent
description: "Phase 2: Validate. Runs 4 sequential sub-stages: raw LLK validation (2a), parameter coverage (2b), helper integration (2c), performance comparison (2d). Each gates the next. Issues L1 trigger back to Phase 1 (Design) when a sub-stage fails on something the proposal must fix."
type: reference
---

## Usage

Invoke with `subagent_type: general-purpose`. Single agent handles all 4 sub-stages.

Replace placeholders:
- `{{LLK_CATEGORY}}` — operation category
- `{{PROPOSAL_PATH}}` — path to the helper proposal document
- `{{HELPER_HPP}}` — path to the helper .hpp file (for sub-stages 2c/2d)
- `{{OPS_LIST}}` — operations to validate

## Prompt Template

```
Validate the {{LLK_CATEGORY}} helper proposal and implementation.

Proposal: {{PROPOSAL_PATH}}
Helper: {{HELPER_HPP}}
Operations: {{OPS_LIST}}

BREADCRUMB LOGGING — do this first:
Derive CATEGORY_SLUG from {{LLK_CATEGORY}} (lowercase, spaces/slashes → underscores).
LOG_DIR="agent_logs/${CATEGORY_SLUG}"
BCRUMB="${LOG_DIR}/validation_breadcrumbs.jsonl"
Run at start:
  mkdir -p "${LOG_DIR}"
  echo '{"ts":"'"$(date -Iseconds)"'","event":"start","agent":"validation","category":"{{LLK_CATEGORY}}"}' >> $BCRUMB

Log one breadcrumb per sub-stage entry, per test run, per L1 trigger:
  echo '{"ts":"...","event":"substage","stage":"2a|2b|2c|2d","status":"start|pass|fail"}' >> $BCRUMB
  echo '{"ts":"...","event":"test","stage":"2a","op":"OP","tiles":N,"result":"PASS|FAIL|HANG"}' >> $BCRUMB
  echo '{"ts":"...","event":"l1_trigger","stage":"2a","op":"OP","error":"..."}' >> $BCRUMB

At completion, write ${LOG_DIR}/validation_execution_log.md: per-sub-stage summary, failures with
diagnosis, L1 triggers emitted, generated test file paths.

See tt_metal/third_party/tt-agents/scripts/logging/ for the JSONL event schema.

Run 4 sub-stages sequentially. Each MUST pass before proceeding to the next.

═══ SUB-STAGE 2a: RAW LLK VALIDATION ═══

Generate test kernels using RAW LLK calls (not the helper) that exercise the
EXACT LLK sequences from the proposal's sequence validation table.

For each proposed sequence:
1. Write a compute kernel (.cpp) with the raw init/exec calls
2. Write a Python test using generic_op that runs the kernel and compares output to PyTorch golden
3. Run the test

Results: PASS / FAIL / HANG per sequence.

FAIL or HANG -> L1 TRIGGER (see below). Report the failure and stop.
Keep all raw LLK test kernels — they become the performance baseline in 2d.

═══ SUB-STAGE 2b: PARAMETER COVERAGE ═══

Using the parameter usage matrix from the investigation, test each LLK across
its parameter space.

Three mandatory dimensions:
1. Data format: Float16_b, BFloat16, Float32 (separate CB configurations per format)
2. Template args: every value of each non-Dst template parameter
3. Runtime args: typical value + edge value + negative value

Covering-array strategy (not full cross-product):
- Each dtype with default template/runtime args
- Each non-default template arg with default dtype
- Each edge runtime arg with default dtype
- At least ONE non-default cross-product combo

Per-LLK output: Parameter Support Matrix classifying each combo as
SUPPORTED / UNSUPPORTED / UNTESTED with test evidence.

Observed combo fails -> L1 TRIGGER (see below). Stop.
Unobserved combo fails -> record as UNSUPPORTED, continue.

═══ SUB-STAGE 2c: HELPER INTEGRATION ═══

Write test kernels using the ACTUAL helper API from {{HELPER_HPP}}. Land them
in the kernel_lib validation suite so later changes MUST re-run them:

- Compute / dataflow kernels under `ttnn/cpp/ttnn/kernel_lib/tests/{feature}/`
  (match the existing `chain_and_binary/` layout: `compute_kernels/`,
  `dataflow_kernels/`)
- Pytest under `tests/ttnn/unit_tests/kernel_lib/test_{feature}.py`
- Launch via `ttnn.generic_op` with a `ProgramDescriptor` (see
  `test_helpers_chain_and_binary.py` for the scaffold pattern)

Mandatory coverage per op:
1. Default path (default dtype, default args)
2. Dtype variation (at least 2 different formats)
3. Template arg variation (at least 1 non-default)
4. Runtime arg variation (at least 2 values)
5. Policy variation (at least 2 input policies)
6. Chain composition (combine with another op in a chain)
7. `num_tiles ∈ {1, 8, 64}` — single tile, fits-in-DEST, multi-DEST-window

Golden comparisons use `comp_pcc` with PCC ≥ 0.9999 for bf16-only paths,
≥ 0.999 when mixed fp32 is involved.

Run via:
```
scripts/tt-test.sh --run-all tests/ttnn/unit_tests/kernel_lib/test_{feature}.py
```

Helper fails but raw passed -> fix .hpp/.inl internally, re-run 2c only.
If fix requires an API change -> L1 TRIGGER (see below).

If the change re-opens coverage in an existing suite (e.g. touching
`sfpu_chain` Load lifecycle re-opens `test_helpers_chain_and_binary.py`),
EXTEND that suite rather than adding a parallel one. One pytest file per
helper feature.

═══ SUB-STAGE 2d: PERFORMANCE ═══

Compare helper vs raw LLK using the kernels from 2a (baseline) and 2c (helper).

Requirements:
- Tile counts: powers of 2 from 8 to 32768
- Workloads: at least 3 (simple single op, 2-op chain, complex chain)
- Measurement: min of trimmed runs (drop highest + lowest, take min of rest)
- Warmup: 3+ runs before timing
- Timed: 7 runs

Output: Performance table with columns:
| Op | Chain | Tiles | Raw min (us) | Helper min (us) | Overhead |
|---|---|---|---|---|---|

Thresholds: <2% OK, 2-5% REVIEW, >5% BLOCKER.
When overhead is ambiguous, disassemble the trisc2 ELF and compare instruction counts.

>5% overhead -> fix .hpp/.inl internally, re-run 2c + 2d.
If fix requires an API change -> L1 TRIGGER (see below).

═══ L1 TRIGGER ═══

When a sub-stage failure requires the proposal to be fixed (not just .hpp/.inl):

Print this block so the orchestrator routes back to Phase 1 (Design):

```
L1_TRIGGER_START
SUB_STAGE: <2a | 2b | 2c | 2d>
OP: <op name>
SEQUENCE_OR_COMBO: <the exact LLK sequence or parameter combo that failed>
ERROR: <what happened — hang, wrong output, compile failure, etc.>
REQUIRED_PROPOSAL_CHANGE: <what specifically needs to change in the proposal>
L1_TRIGGER_END
```

Stop after printing the L1 trigger. Do not continue to the next sub-stage.
The orchestrator sends this payload to Phase 1 (Design) as `{{L1_FAILURE_CONTEXT}}`.
After Phase 1 outputs an amended proposal, Phase 2 re-enters from 2a.

═══ OUTPUT ═══

Save to: {{CATEGORY_SLUG}}_validation.md containing:
- Per-sub-stage results (pass/fail per test)
- Parameter support matrix
- Performance table
- List of generated test files
- L1 trigger payload if issued
```
