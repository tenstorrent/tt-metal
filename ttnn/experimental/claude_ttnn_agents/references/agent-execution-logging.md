# Agent Execution Logging Reference

This document provides consolidated logging and version control instructions for TTNN operation agents.

---

## Two Independent Systems

| System | When Active | Purpose |
|--------|-------------|---------|
| **Git Commits** | **ALWAYS** | Capture actual code state at checkpoints |
| **Breadcrumbs** | Only if orchestrator enables | Detailed event logging for debugging |

**Git is ALWAYS required.** Breadcrumbs are conditional.

---

# PART 1: GIT PROTOCOL (ALWAYS REQUIRED)

Git commits are **MANDATORY** for all agents, regardless of breadcrumb settings. This captures actual code state, not just claims about what was done.

## Why Git is Required

Logs describe intent. Git captures reality. When debugging:
- Logs might say "CB sync balanced: true" but code has wrong values
- Git diff shows exactly what values are in the code
- Git enables rollback, bisect, and attribution

## When to Commit

Commits are required at these points (in order of priority):

### MUST Commit (Required)
1. **Stage completion**: After each numbered stage passes its tests
2. **Before handoff**: Before passing work to next agent
3. **Successful build**: After any `./build_metal.sh` that succeeds

### SHOULD Commit (Strongly Recommended)
4. **Major decision implemented**: After implementing significant code changes
5. **After fixing a bug**: When recovering from an error
6. **Before risky changes**: Create checkpoint before attempting uncertain modifications

## Commit Message Format

Every commit MUST follow this format:

```
[{agent-name}] {stage-or-phase}: {concise-description}

{bullet-list-of-key-changes}

operation: {operation_name}
build: {PASSED|SKIPPED|N/A}
tests: {PASSED|FAILED|SKIPPED}
```

### Examples

**Stage completion commit:**
```
[ttnn-factory-builder] stage 4-6: CB config and stub kernels

- Configured 5 circular buffers (c_0, c_1, c_2, c_3, c_16)
- Created reader/compute/writer stub kernels
- Single-core work distribution

operation: reduce_avg_w_rm
build: PASSED
tests: stage4=PASS, stage5=PASS, stage6=PASS
```

**Build fix commit:**
```
[ttnn-operation-scaffolder] build-fix: launch_on_device -> launch

- Fixed API call in device_operation.cpp line 116
- Changed ttnn::device_operation::detail::launch_on_device to ttnn::device_operation::launch

operation: reduce_avg_w_rm
build: PASSED
tests: SKIPPED
```

**Kernel implementation commit:**
```
[ttnn-kernel-writer] stage 7: implement tilize+reduce+untilize kernels

- Reader: TensorAccessor for input sticks, generate_reduce_scaler
- Compute: tilize_helpers, reduce_helpers, binary_op_helpers, untilize_helpers
- Writer: TensorAccessor for output sticks
- Fixed CB c_0 config: page_size=tile_size, num_pages=Wt

operation: reduce_avg_w_rm
build: PASSED
tests: stage7=9/9 PASS
```

## Git Commands

### Standard Commit Sequence

```bash
# 1. Stage all changes
git add -A

# 2. Create commit with proper format (use heredoc for multiline)
git commit -m "$(cat <<'EOF'
[{agent-name}] {stage}: {description}

- Key change 1
- Key change 2

operation: {op_name}
build: {status}
tests: {status}
EOF
)"
```

### Checking What Changed

Before committing, verify your changes:
```bash
# See what files changed
git status

# See actual code changes
git diff

# See staged changes
git diff --cached
```

## File Type Awareness (CRITICAL)

Different file types have different build requirements:

| File Location | Type | Rebuild Required? |
|---------------|------|-------------------|
| `device/kernels/**/*.cpp` | Kernel | NO (runtime compile) |
| `*.cpp` outside kernels/ | Host | **YES** |
| `*.hpp` | Header | **YES** if included by host code |
| `*.py` | Python | NO |

**CRITICAL**: If you modify ANY host files (program_factory.cpp, device_operation.cpp, etc.), you MUST:
1. Run `./build_metal.sh -b Debug`
2. Verify build succeeds
3. THEN run tests
4. THEN commit

Tests against stale builds produce FALSE RESULTS.

## Git Commit Checklist

Before each commit:
- [ ] All intended changes are staged (`git status`)
- [ ] Commit message follows format with agent name
- [ ] If host files modified: build was run and passed
- [ ] Commit message includes build/test status

---

# PART 2: BREADCRUMBS (CONDITIONAL)

Breadcrumb logging is **CONDITIONAL** based on orchestrator instructions.

## Determining Breadcrumb Status

Check your invocation prompt for these phrases:
- "with execution logging" → **ENABLED**
- "enable logging" → **ENABLED**
- "enable detailed logging" → **ENABLED**
- "with breadcrumbs" → **ENABLED**
- None of the above → **DISABLED**

**If ENABLED**: You MUST follow ALL breadcrumb steps below. This is not optional.
**If DISABLED**: Skip all breadcrumb steps. Do not create breadcrumb files.

---

## Quick Reference (If Breadcrumbs Enabled)

| Script | Purpose | Location |
|--------|---------|----------|
| `init_breadcrumbs.sh` | Initialize breadcrumbs file | `.claude/scripts/logging/` |
| `append_breadcrumb.sh` | Append event to breadcrumbs | `.claude/scripts/logging/` |
| `agent-log-template.md` | Final log template | `.claude/references/` |

---

## Breadcrumb Initialization (If Enabled)

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
- `operation_path`: The operation directory. Path depends on workflow:
  - Generic op workflow: `ttnn/ttnn/operations/{operation_name}` (see `ttnn-generic-op-workflow.md`)
  - Standard C++ workflow: `ttnn/cpp/ttnn/operations/{category}/{operation_name}`
- `agent_name`: e.g., `ttnn-operation-scaffolder`, `ttnn-generic-op-builder`
- `operation_name`: e.g., `my_op`
- `predecessor_agent`: e.g., `ttnn-operation-planner` or `""` if first agent
- `input_file_path`: e.g., `my_op_spec.md`

---

## Appending Breadcrumbs (If Enabled)

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
{"event":"action","type":"git_commit","message":"[agent] stage X: description"}
```

### result
After an action completes:
```json
{"event":"result","type":"build","success":true}
{"event":"result","type":"build","success":false,"error":"missing semicolon","error_type":"build_error"}
{"event":"result","type":"script_run","script":"generate_files.py","success":true,"output":"Created 12 files"}
{"event":"result","type":"git_commit","success":true,"commit_sha":"abc1234"}
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
{"event":"complete","final_status":"SUCCESS","stages_completed":[1,2,3],"final_commit":"abc1234"}
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
| `host_file_modified` | Track host file changes | `{"event":"host_file_modified","file":"program_factory.cpp","build_required":true}` |

---

## Finalization: Writing the Execution Log (If Breadcrumbs Enabled)

**MANDATORY if breadcrumbs enabled**: Before reporting completion, write the structured execution log.

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
- **Section 8**: Git commit history for this agent

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

**Host Files Modified:**
| File | Build Required | Build Ran | Build Result |
|------|----------------|-----------|--------------|
| {path} | YES/NO | YES/NO | PASS/FAIL/N/A |

---

## Example Breadcrumb Sequences (If Enabled)

### Successful Run (scaffolder)
```jsonl
{"ts":"...","event":"start","agent":"ttnn-operation-scaffolder","operation":"my_op"}
{"ts":"...","event":"input_parse","field":"operation_name","value":"my_op","confidence":"HIGH"}
{"ts":"...","event":"action","type":"script_run","script":"generate_files.py","args":["--force"]}
{"ts":"...","event":"result","type":"script_run","script":"generate_files.py","success":true}
{"ts":"...","event":"action","type":"build","command":"./build_metal.sh -b Debug"}
{"ts":"...","event":"result","type":"build","success":true}
{"ts":"...","event":"action","type":"git_commit","message":"[ttnn-operation-scaffolder] stage 1-3: scaffolding complete"}
{"ts":"...","event":"result","type":"git_commit","success":true,"commit_sha":"abc1234"}
{"ts":"...","event":"test","stage":1,"result":"PASS"}
{"ts":"...","event":"test","stage":2,"result":"PASS"}
{"ts":"...","event":"test","stage":3,"result":"PASS"}
{"ts":"...","event":"complete","final_status":"SUCCESS","stages_completed":[1,2,3],"final_commit":"abc1234"}
```

### Run with Error Recovery (factory-builder)
```jsonl
{"ts":"...","event":"start","agent":"ttnn-factory-builder","operation":"reduce_op"}
{"ts":"...","event":"cb_config","cb_id":"c_0","page_size":2048,"num_pages":2,"purpose":"input"}
{"ts":"...","event":"action","type":"build","command":"./build_metal.sh -b Debug"}
{"ts":"...","event":"result","type":"build","success":true}
{"ts":"...","event":"action","type":"git_commit","message":"[ttnn-factory-builder] stage 5: initial CB config"}
{"ts":"...","event":"result","type":"git_commit","success":true,"commit_sha":"def5678"}
{"ts":"...","event":"tdd_cycle","stage":6,"phase":"GREEN_ATTEMPT","result":"TIMEOUT"}
{"ts":"...","event":"hang_debug","symptom":"timeout","diagnosis":"CB sync mismatch"}
{"ts":"...","event":"hypothesis","id":"H1","description":"Compute pops fewer tiles than reader pushes","confidence":"HIGH"}
{"ts":"...","event":"recovery","hypothesis_id":"H1","action":"Fixed compute to consume all inputs","file":"compute.cpp"}
{"ts":"...","event":"action","type":"git_commit","message":"[ttnn-factory-builder] fix: CB sync for shape-changing op"}
{"ts":"...","event":"result","type":"git_commit","success":true,"commit_sha":"ghi9012"}
{"ts":"...","event":"tdd_cycle","stage":6,"phase":"GREEN","result":"PASS"}
{"ts":"...","event":"complete","final_status":"SUCCESS","stages_completed":[4,5,6],"final_commit":"ghi9012"}
```

---

## Confidence Levels

Use these for `input_parse` confidence:
- **HIGH**: Explicitly stated in input, no interpretation needed
- **MEDIUM**: Required some inference or combining sources
- **LOW**: Significant guesswork, input was ambiguous

---

## Checklist Before Completing

### Git (ALWAYS Required)
- [ ] All code changes committed
- [ ] Commit messages follow format with agent name
- [ ] If host files modified: build ran before final commit
- [ ] Final commit SHA recorded

### Breadcrumbs (If Enabled)
- [ ] Breadcrumbs initialized at start
- [ ] Key events logged during execution
- [ ] Git commits logged as events
- [ ] Breadcrumbs file read before writing log
- [ ] Log template read and followed
- [ ] All sections filled in execution log
- [ ] Agent-specific sections included
- [ ] Log written to correct path
