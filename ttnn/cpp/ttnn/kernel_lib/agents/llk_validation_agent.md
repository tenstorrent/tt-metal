---
name: LLK Validation Agent
description: "Phase 4 agent. Runs 4 sequential sub-stages: raw LLK validation, parameter coverage, helper integration, performance comparison. Each gates the next."
type: reference
---

## Usage

Invoke with `subagent_type: general-purpose`. Single agent handles all 4 sub-stages.

Replace placeholders:
- `{{LLK_CATEGORY}}` — operation category
- `{{PROPOSAL_PATH}}` — path to the helper proposal document
- `{{HELPER_HPP}}` — path to the helper .hpp file (for sub-stages 4c/4d)
- `{{OPS_LIST}}` — operations to validate

## Prompt Template

```
Validate the {{LLK_CATEGORY}} helper proposal and implementation.

Proposal: {{PROPOSAL_PATH}}
Helper: {{HELPER_HPP}}
Operations: {{OPS_LIST}}

Log breadcrumbs to agent_logs/. See tt_metal/third_party/tt-agents/scripts/logging/ for format.

Run 4 sub-stages sequentially. Each MUST pass before proceeding to the next.

═══ SUB-STAGE 4a: RAW LLK VALIDATION ═══

Generate test kernels using RAW LLK calls (not the helper) that exercise the
EXACT LLK sequences from the proposal's sequence validation table.

For each proposed sequence:
1. Write a compute kernel (.cpp) with the raw init/exec calls
2. Write a Python test using generic_op that runs the kernel and compares output to PyTorch golden
3. Run the test

Results: PASS / FAIL / HANG per sequence.

FAIL or HANG -> BLOCKER. Report the failure and stop.
Keep all raw LLK test kernels — they become the performance baseline in 4d.

═══ SUB-STAGE 4b: PARAMETER COVERAGE ═══

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

Observed combo fails -> BLOCKER, stop.
Unobserved combo fails -> record as UNSUPPORTED, continue.

═══ SUB-STAGE 4c: HELPER INTEGRATION ═══

Write test kernels using the ACTUAL helper API from {{HELPER_HPP}}.

Mandatory coverage per op:
1. Default path (default dtype, default args)
2. Dtype variation (at least 2 different formats)
3. Template arg variation (at least 1 non-default)
4. Runtime arg variation (at least 2 values)
5. Policy variation (at least 2 input policies)
6. Chain composition (combine with another op in a chain)

Helper fails but raw passed -> fix .hpp/.inl, re-run 4c only.

═══ SUB-STAGE 4d: PERFORMANCE ═══

Compare helper vs raw LLK using the kernels from 4a (baseline) and 4c (helper).

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

>5% overhead -> BLOCKER, fix .hpp/.inl, re-run 4c + 4d.

═══ OUTPUT ═══

Save to: {{CATEGORY_SLUG}}_validation.md containing:
- Per-sub-stage results (pass/fail per test)
- Parameter support matrix
- Performance table
- List of generated test files
- Any upstream feedback (proposal issues found)
```
