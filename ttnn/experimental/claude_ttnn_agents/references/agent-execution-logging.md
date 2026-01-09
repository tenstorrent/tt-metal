# Agent Execution Logging Reference

This document provides consolidated logging instructions for TTNN operation agents. Logging is **OPTIONAL** and should only be enabled when instructed by the main agent.

---

## When to Enable Logging

Enable logging when the main agent (or user) includes one of these phrases:
- "with execution logging"
- "enable logging"
- "enable detailed logging"
- "with breadcrumbs"

If none of these are present, **skip all logging steps**.

---

## Quick Reference

| Script | Purpose | Location |
|--------|---------|----------|
| `init_breadcrumbs.sh` | Initialize breadcrumbs file | `.claude/scripts/logging/` |
| `append_breadcrumb.sh` | Append event to breadcrumbs | `.claude/scripts/logging/` |
| `agent-log-template.md` | Final log template | `.claude/references/` |

---

## Breadcrumb Initialization

At the **start of execution**, after identifying the operation path:

```bash
mkdir -p {operation_path}/agent_logs
.claude/scripts/logging/init_breadcrumbs.sh \
  {operation_path} \
  {agent_name} \
  {operation_name} \
  "{predecessor_agent}" \
  "{input_file_path}"
```

**Parameters:**
- `operation_path`: e.g., `ttnn/cpp/ttnn/operations/reduction/my_op`
- `agent_name`: e.g., `ttnn-operation-scaffolder`
- `operation_name`: e.g., `my_op`
- `predecessor_agent`: e.g., `ttnn-operation-planner` or `""` if first agent
- `input_file_path`: e.g., `my_op_spec.md`

---

## Appending Breadcrumbs

Use this pattern at key decision points:

```bash
.claude/scripts/logging/append_breadcrumb.sh {operation_path} {agent_name} '{json}'
```

---

## Common Event Types

These events are used by all agents:

### input_parse
After extracting fields from input:
```json
{"event":"input_parse","field":"operation_name","value":"my_op","confidence":"HIGH"}
{"event":"input_parse","field":"parameters","value":"[memory_config]","confidence":"MEDIUM","notes":"Inferred from context"}
```

### action
Before executing an action:
```json
{"event":"action","type":"build","command":"./build_metal.sh -b Debug"}
{"event":"action","type":"script_run","script":"generate_files.py","args":["--force"]}
{"event":"action","type":"test","command":"pytest test_file.py"}
```

### result
After an action completes:
```json
{"event":"result","type":"build","success":true}
{"event":"result","type":"build","success":false,"error":"missing semicolon","error_type":"build_error"}
{"event":"result","type":"script_run","script":"generate_files.py","success":true,"output":"Created 12 files"}
```

### hypothesis
When forming a hypothesis about an error:
```json
{"event":"hypothesis","id":"H1","description":"Missing () on method call","confidence":"HIGH","evidence":"Error at line 45"}
```

### recovery
When attempting to fix an issue:
```json
{"event":"recovery","hypothesis_id":"H1","action":"Added () to method call","file":"device/my_op.cpp"}
```

### test
After running a test:
```json
{"event":"test","stage":1,"file":"test_stage1.py","result":"PASS"}
{"event":"test","stage":2,"file":"test_stage2.py","result":"FAIL","error_summary":"Wrong error message"}
```

### deviation
When deviating from instructions:
```json
{"event":"deviation","what":"Manually edited generated file","why":"Script had incorrect path","impact":"May need re-run"}
```

### upstream_feedback
Issues with input from predecessor agent:
```json
{"event":"upstream_feedback","target_agent":"ttnn-operation-planner","issue":"Validation used property syntax","suggestion":"Use method syntax","severity":"MEDIUM"}
```

### complete
At end of execution:
```json
{"event":"complete","final_status":"SUCCESS","stages_completed":[1,2,3]}
```

---

## Agent-Specific Event Types

### ttnn-operation-scaffolder

No additional event types beyond common ones. Uses `action/result` with `type":"script_run"`.

### ttnn-factory-builder

| Event | Purpose | Example |
|-------|---------|---------|
| `cb_config` | CB configuration decisions | `{"event":"cb_config","cb_id":"c_0","page_size":2048,"num_pages":2,"purpose":"input"}` |
| `work_distribution` | Work split calculations | `{"event":"work_distribution","grid":"8x8","total_tiles":256,"tiles_per_core":4}` |
| `tdd_cycle` | TDD phase tracking | `{"event":"tdd_cycle","stage":4,"phase":"RED","result":"FAIL","expected":true}` |
| `cb_audit` | CB sync verification | `{"event":"cb_audit","cb_id":"c_0","producer":"reader","push_count":"N","consumer":"compute","pop_count":"N","balanced":true}` |
| `cb_sync_summary` | Final CB balance check | `{"event":"cb_sync_summary","total_cbs":2,"all_balanced":true}` |
| `hang_debug` | Debugging hangs | `{"event":"hang_debug","symptom":"timeout","diagnosis":"CB sync mismatch"}` |

### ttnn-kernel-writer

| Event | Purpose | Example |
|-------|---------|---------|
| `design_compliance` | Design doc adherence | `{"event":"design_compliance","phase":"tilize","directive":"USE HELPER","implementation":"compute_kernel_lib::tilize()","compliant":true}` |
| `cb_wrapper_check` | Redundant CB op check | `{"event":"cb_wrapper_check","helper":"tilize","has_wrapper_cb_ops":false,"status":"CLEAN"}` |
| `correctness_test` | Test case results | `{"event":"correctness_test","test_name":"test_basic","expected":"0.07","actual":"0.07","pass":true}` |
| `numerical_debug` | Debugging wrong values | `{"event":"numerical_debug","symptom":"values 10x smaller","finding":"scaler format wrong"}` |
| `design_compliance_summary` | Final compliance check | `{"event":"design_compliance_summary","total_phases":5,"all_compliant":true}` |

---

## Finalization: Writing the Execution Log

**MANDATORY if logging enabled**: Before reporting completion, write the structured execution log.

### Step 1: Read the Log Template
```
Read: .claude/references/agent-log-template.md
```

### Step 2: Read Your Breadcrumbs
```
Read: {operation_path}/agent_logs/{agent_name}_breadcrumbs.jsonl
```

### Step 3: Write the Structured Log

Write to: `{operation_path}/agent_logs/{agent_name}_execution_log.md`

Include ALL sections from the template, with special attention to:
- **Section 1**: Input interpretation and upstream feedback
- **Section 2**: Execution timeline with all attempts
- **Section 2a**: Agent-specific sections (see below)
- **Section 3**: Recovery summary
- **Section 6**: Handoff notes for next agent
- **Section 7**: Instruction improvement recommendations

---

## Agent-Specific Log Sections (Section 2a)

### For ttnn-operation-scaffolder

**Script Execution Log:**
| Script | Arguments | Result | Output Summary |
|--------|-----------|--------|----------------|
| generate_files.py | {args} | {SUCCESS/FAIL} | {brief output} |
| integrate_build.py | {args} | {SUCCESS/FAIL} | {brief output} |
| verify_scaffolding.sh | {args} | {SUCCESS/FAIL} | {checks passed} |

### For ttnn-factory-builder

**CB Configuration Audit:**
| CB ID | Index | Page Size | Num Pages | Purpose | Source |
|-------|-------|-----------|-----------|---------|--------|
| cb_in | c_0 | {bytes} | {count} | {purpose} | {Spec/Inferred} |

**CB Sync Verification (CRITICAL):**
| CB | Producer | Push Operation | Consumer | Pop Operation | Balanced? |
|----|----------|----------------|----------|---------------|-----------|
| c_0 | Reader | cb_push_back x N | Compute | cb_pop_front x N | YES/NO |

**Work Distribution:**
| Parameter | Value | Source |
|-----------|-------|--------|
| Core grid | {rows x cols} | {Spec/Calculated} |
| Total work units | {count} | |

### For ttnn-kernel-writer

**Design Compliance:**
| Phase | Design Directive | Implementation | Compliant? |
|-------|------------------|----------------|------------|
| {phase} | USE HELPER / NO HELPER | {what you wrote} | YES/NO |

**Redundant CB Operation Check:**
| Helper Used | Wrapper CB Ops Present? | Status |
|-------------|-------------------------|--------|
| compute_kernel_lib::X() | YES/NO | CLEAN/VIOLATION |

**Correctness Test Results:**
| Test Case | Input Shape | Tolerance | Result |
|-----------|-------------|-----------|--------|
| {name} | {shape} | rtol/atol | PASS/FAIL |

---

## Example Breadcrumb Sequences

### Successful Run (scaffolder)
```jsonl
{"ts":"...","event":"start","agent":"ttnn-operation-scaffolder","operation":"my_op"}
{"ts":"...","event":"input_parse","field":"operation_name","value":"my_op","confidence":"HIGH"}
{"ts":"...","event":"action","type":"script_run","script":"generate_files.py","args":["--force"]}
{"ts":"...","event":"result","type":"script_run","script":"generate_files.py","success":true}
{"ts":"...","event":"action","type":"build","command":"./build_metal.sh -b Debug"}
{"ts":"...","event":"result","type":"build","success":true}
{"ts":"...","event":"test","stage":1,"result":"PASS"}
{"ts":"...","event":"test","stage":2,"result":"PASS"}
{"ts":"...","event":"test","stage":3,"result":"PASS"}
{"ts":"...","event":"complete","final_status":"SUCCESS","stages_completed":[1,2,3]}
```

### Run with Error Recovery (factory-builder)
```jsonl
{"ts":"...","event":"start","agent":"ttnn-factory-builder","operation":"reduce_op"}
{"ts":"...","event":"cb_config","cb_id":"c_0","page_size":2048,"num_pages":2,"purpose":"input"}
{"ts":"...","event":"tdd_cycle","stage":6,"phase":"GREEN_ATTEMPT","result":"TIMEOUT"}
{"ts":"...","event":"hang_debug","symptom":"timeout","diagnosis":"CB sync mismatch"}
{"ts":"...","event":"hypothesis","id":"H1","description":"Compute pops fewer tiles than reader pushes","confidence":"HIGH"}
{"ts":"...","event":"recovery","hypothesis_id":"H1","action":"Fixed compute to consume all inputs","file":"compute.cpp"}
{"ts":"...","event":"tdd_cycle","stage":6,"phase":"GREEN","result":"PASS"}
{"ts":"...","event":"complete","final_status":"SUCCESS","stages_completed":[4,5,6]}
```

---

## Confidence Levels

Use these for `input_parse` confidence:
- **HIGH**: Explicitly stated in input, no interpretation needed
- **MEDIUM**: Required some inference or combining sources
- **LOW**: Significant guesswork, input was ambiguous

---

## Checklist Before Completing (if logging enabled)

- [ ] Breadcrumbs initialized at start
- [ ] Key events logged during execution
- [ ] Breadcrumbs file read before writing log
- [ ] Log template read and followed
- [ ] All sections filled in execution log
- [ ] Agent-specific sections included
- [ ] Log written to correct path
