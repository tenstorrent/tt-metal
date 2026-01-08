# Agent Execution Log Template

This template is used by TTNN operation agents to write structured execution logs. Load this file at the end of your execution, fill in the sections based on your breadcrumbs and final state, and write to `{operation_path}/agent_logs/{agent_name}_execution_log.md`.

---

## How to Use This Template

1. **Read your breadcrumbs file**: `{operation_path}/agent_logs/{agent_name}_breadcrumbs.jsonl`
2. **Fill in each section** using data from breadcrumbs and your execution memory
3. **Include agent-specific sections** (see bottom of template)
4. **Write the completed log** to `{operation_path}/agent_logs/{agent_name}_execution_log.md`

Replace all `{placeholders}` with actual values. Delete sections that don't apply.

---

# Agent Execution Log: {agent_name}

## Metadata
| Field | Value |
|-------|-------|
| Operation | `{operation_name}` |
| Agent | `{agent_name}` |
| Stages | {stages_covered} |
| Input | `{input_file_paths}` |
| Predecessor | {predecessor_agent_name or "N/A (first in pipeline)"} |
| Final Status | {SUCCESS / PARTIAL / FAILED} |
| Total Attempts | {sum of all stage attempts} |

---

## 1. Input Interpretation

### Spec/Input Fields Extracted

List all fields you extracted from the input (spec, design doc, etc.) with confidence levels.

| Field | Value | Confidence | Notes |
|-------|-------|------------|-------|
| {field_name} | {extracted_value} | {HIGH/MEDIUM/LOW} | {any ambiguity or inference needed} |

**Confidence Levels**:
- **HIGH**: Explicitly stated in input, no interpretation needed
- **MEDIUM**: Required some inference or combining multiple sources
- **LOW**: Significant guesswork, input was ambiguous or missing

### Interpretation Issues

{Describe any fields that were:
- Missing from the input
- Ambiguously specified
- Required significant inference
- Contradicted other parts of the input

If none, write "None - input was clear and complete."}

### Upstream Feedback

{Suggestions for the agent that produced your input. This helps improve the pipeline.}

| Target Agent | Issue | Suggestion | Severity |
|--------------|-------|------------|----------|
| {agent_name} | {what was wrong or suboptimal} | {how to improve} | {HIGH/MEDIUM/LOW} |

{If no issues, write "None - upstream output was well-formed."}

---

## 2. Execution Timeline

{Document each significant phase/stage of your work. For each stage, document all attempts until success or giving up.}

### {Stage/Phase Name}

#### Attempt 1: {brief_description}
| Field | Value |
|-------|-------|
| Action | {what you did} |
| Expected | {what should have happened} |
| Actual | {what actually happened} |
| Result | {PASS / FAIL} |

{If FAIL, add:}
- **Error Type**: {build_error / test_timeout / test_fail / wrong_output / hang / other}
- **Error Summary**: {1-2 sentence description}
- **Root Cause Hypothesis**: H1: {your hypothesis about the cause}
- **Evidence**: {what led you to this hypothesis}
- **Recovery Action**: {what you did to fix it}

#### Attempt 2: {brief_description}
{Repeat format until PASS or max attempts reached}

---

## 3. Recovery Summary

### Error Recovery Table

| # | Stage | Error Type | Root Cause (Hypothesis) | Recovery Action | Resolved? |
|---|-------|------------|-------------------------|-----------------|-----------|
| 1 | {stage} | {type} | {H#: description} | {action taken} | {YES/NO} |

### Attempts Per Stage

| Stage | Attempts | Final Result |
|-------|----------|--------------|
| {stage_name} | {count} | {PASS / FAIL / SKIPPED} |

### Unresolved Issues

{List any issues that could NOT be resolved. Explain why and what would be needed to resolve them.}

{If all issues resolved, write "All issues were resolved."}

---

## 4. Deviations from Instructions

{Document any cases where you deviated from your agent instructions.}

| What | Why | Impact |
|------|-----|--------|
| {what you did differently} | {justification} | {effect on output or downstream agents} |

{If no deviations, write "None - followed all instructions as specified."}

---

## 5. Artifacts

### Files Created

| Path | Purpose |
|------|---------|
| `{relative_path}` | {what this file does} |

### Files Modified

| Path | Changes |
|------|---------|
| `{relative_path}` | {summary of modifications} |

---

## 6. Handoff Notes

### For Next Agent: {next_agent_name}

{Critical information the next agent in the pipeline needs to know.}

**Key Configuration**:
- {important setting or decision}
- {another important item}

**Special Considerations**:
- {anything unusual about this operation}
- {edge cases or limitations}

**Known Limitations**:
- {what doesn't work or wasn't implemented}

{If this is the final agent (e.g., kernel-writer completing Stage 7), write:
"N/A - This is the final stage. Operation is complete."}

---

## 7. Instruction Improvement Recommendations

{Based on your execution, suggest improvements to your own agent instructions.}

### Recommendation 1: {short_title}
- **Observed**: {what happened during execution}
- **Frequency**: {once / multiple times / every time}
- **Current Instruction**: {what the instructions say now, if relevant}
- **Suggested Change**: {specific modification to instructions}
- **Rationale**: {why this would help}
- **Confidence**: {HIGH / MEDIUM / LOW}

{Add more recommendations as needed. If none, write "None - instructions were sufficient for this operation."}

---

## 8. Raw Logs

{Include relevant raw output for debugging. Use collapsible sections to avoid clutter.}

<details>
<summary>Build Output</summary>

```
{Paste truncated build output here, especially error messages}
```

</details>

<details>
<summary>Test Output</summary>

```
{Paste test output, especially failures}
```

</details>

<details>
<summary>Debug Output</summary>

```
{Any DPRINT output, watcher logs, or other debug info}
```

</details>

---

# Agent-Specific Sections

{Include the section(s) relevant to your agent type.}

---

## For ttnn-operation-scaffolder

Insert after Section 2:

### 2a. Script Execution Log

| Script | Arguments | Result | Output Summary |
|--------|-----------|--------|----------------|
| generate_files.py | {args} | {SUCCESS/FAIL} | {brief output} |
| integrate_build.py | {args} | {SUCCESS/FAIL} | {brief output} |
| verify_scaffolding.sh | {args} | {SUCCESS/FAIL} | {checks passed} |

### JSON Config Validation

| Check | Result | Notes |
|-------|--------|-------|
| JSON syntax valid | {PASS/FAIL} | |
| All required fields present | {PASS/FAIL} | {missing fields if any} |
| C++ expressions valid | {PASS/FAIL} | {issues found} |
| Schema validation | {PASS/FAIL} | |

### Spec Parsing Decisions

| Spec Field | Parsed Value | Inference Required? |
|------------|--------------|---------------------|
| operation_name | {value} | {YES/NO} |
| parameters | {list} | {YES/NO - describe inference} |
| validations | {count} conditions | {YES/NO} |

---

## For ttnn-factory-builder

Insert after Section 2:

### 2a. Circular Buffer Configuration

| CB ID | Index | Page Size | Num Pages | Data Type | Purpose | Source |
|-------|-------|-----------|-----------|-----------|---------|--------|
| cb_in | c_0 | {bytes} | {count} | {dtype} | {purpose} | {Spec table / Inferred} |
| cb_out | c_16 | {bytes} | {count} | {dtype} | {purpose} | {Spec table / Inferred} |

### CB Synchronization Verification

{CRITICAL: Verify push/pop balance to prevent hangs}

| CB | Producer Kernel | Push Operation | Consumer Kernel | Pop Operation | Balanced? |
|----|-----------------|----------------|-----------------|---------------|-----------|
| c_0 | Reader | cb_push_back(c_0, {N}) x {M} | Compute | cb_pop_front(c_0, {N}) x {M} | {YES/NO} |
| c_16 | Compute | cb_push_back(c_16, {N}) x {M} | Writer | cb_pop_front(c_16, {N}) x {M} | {YES/NO} |

**Total tiles through pipeline**:
- Reader pushes: {total} tiles to cb_in
- Compute pops: {total} tiles from cb_in
- Compute pushes: {total} tiles to cb_out
- Writer pops: {total} tiles from cb_out

### Work Distribution

| Parameter | Value | Source |
|-----------|-------|--------|
| Core grid | {rows} x {cols} | {Spec / Calculated} |
| Total work units | {count} | |
| Work per core | {count or formula} | |
| Remainder handling | {strategy} | |

### Stub Kernel Summary

| Kernel | File | Purpose | CB In | CB Out |
|--------|------|---------|-------|--------|
| Reader | {path} | {description} | N/A | {cb_ids} |
| Compute | {path} | Passthrough (copy_tile) | {cb_ids} | {cb_ids} |
| Writer | {path} | {description} | {cb_ids} | N/A |

---

## For ttnn-kernel-writer

Insert after Section 2:

### 2a. Design Document Compliance

{CRITICAL: You must follow the design document exactly}

#### Helper Usage Compliance

| Phase | Design Directive | Your Implementation | Compliant? |
|-------|------------------|---------------------|------------|
| {phase_name} | {USE HELPER: X() / NO HELPER} | {what you wrote} | {YES/NO} |

#### Redundant CB Operation Check

{Verify you did NOT add CB operations around helper calls}

| Helper Used | Wrapper CB Ops Present? | Status |
|-------------|-------------------------|--------|
| compute_kernel_lib::tilize() | {YES/NO} | {CLEAN / VIOLATION} |
| compute_kernel_lib::reduce<...>() | {YES/NO} | {CLEAN / VIOLATION} |
| compute_kernel_lib::untilize<...>() | {YES/NO} | {CLEAN / VIOLATION} |

{If any VIOLATION, explain why and whether it was intentional deviation}

### Stage 7 Correctness Test Results

| Test Case | Input Shape | Reference | Tolerance | Result | Notes |
|-----------|-------------|-----------|-----------|--------|-------|
| {test_name} | {shape} | {PyTorch / manual calc} | rtol={}, atol={} | {PASS/FAIL} | |

### Numerical Debugging (if applicable)

{If you encountered wrong output values, document the debugging process}

| Symptom | Investigation | Root Cause | Fix |
|---------|---------------|------------|-----|
| {e.g., values 10x smaller} | {what you checked} | {what was wrong} | {what you changed} |

---

# Checklist Before Submitting Log

- [ ] All `{placeholders}` replaced with actual values
- [ ] Metadata section complete with final status
- [ ] All attempts documented in Execution Timeline
- [ ] Recovery Summary table populated
- [ ] Upstream Feedback included (even if "None")
- [ ] Instruction Improvement Recommendations included (even if "None")
- [ ] Agent-specific sections included
- [ ] Raw logs added for any failures
- [ ] File saved to correct location: `{operation_path}/agent_logs/{agent_name}_execution_log.md`
