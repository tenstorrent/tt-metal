# Breadcrumb Logging - ttnn-kernel-writer

**If breadcrumb logging is ENABLED, this document is MANDATORY reading.**

First read: `.claude/references/logging/common.md`

---

## Agent Identity

- **Agent name**: `ttnn-kernel-writer`
- **Predecessor**: `ttnn-kernel-designer`
- **Stages owned**: 7

---

## Logging Philosophy for Kernel Writer

The kernel-writer agent often encounters the most challenging debugging scenarios:
- Hangs from CB synchronization issues
- Numerical errors from incorrect helper usage
- Compile errors from template mismatches

**Every debugging step MUST be logged.** This is not optional. The logging trail:
1. Prevents repeated failed approaches
2. Enables pattern recognition across operations
3. Provides handoff context if interrupted
4. Improves instructions based on common failures

---

## Mandatory Events (MUST Log These)

### phase_start
Log at the very beginning of implementation:
```json
{"event":"phase_start","phase":"implementation","design_doc":"path/to/kernel_design.md"}
```

### design_parsed
After reading and understanding the design document:
```json
{"event":"design_parsed","phases_count":5,"helpers_required":["tilize","reduce","untilize"],"raw_phases":["reader","writer"]}
```

### kernel_implemented
After implementing each kernel:
```json
{"event":"kernel_implemented","kernel":"reader","approach":"TensorAccessor for input sticks, cb_push_back per tile"}
{"event":"kernel_implemented","kernel":"compute","approach":"tilize_helpers -> reduce_helpers -> untilize_helpers"}
{"event":"kernel_implemented","kernel":"writer","approach":"TensorAccessor for output sticks, cb_pop_front per tile"}
```

### test_run
**Log after EVERY test run** (passes AND failures):
```json
{"event":"test_run","test":"test_stage7_basic","result":"PASS","duration_s":2.3}
{"event":"test_run","test":"test_stage7_basic","result":"FAIL","failure_type":"numerical","details":"expected 0.07, got 0.7"}
{"event":"test_run","test":"test_stage7_basic","result":"HANG","failure_type":"timeout","timeout_s":30}
```

### design_compliance_summary
**CRITICAL - Log before completing:**
```json
{"event":"design_compliance_summary","total_phases":5,"all_compliant":true,"test_runs":8,"failures_debugged":2}
```

---

## Debugging Events (MUST Log When Debugging)

### hang_detected
When a test times out:
```json
{"event":"hang_detected","test":"test_basic","symptom":"timeout after 30s","suspected_cause":"CB sync"}
```

### numerical_error
When test fails with wrong values:
```json
{"event":"numerical_error","test":"test_basic","expected":"0.07","actual":"0.7","ratio":"10x","pattern":"all values 10x larger than expected"}
```

### compile_error
When kernel fails to compile:
```json
{"event":"compile_error","file":"compute.cpp","error":"no matching function for call to reduce","line":42}
```

### hypothesis
**CRITICAL**: Log every hypothesis you form:
```json
{"event":"hypothesis","id":"H1","description":"CB c_0 push/pop mismatch - compute pops fewer tiles than reader pushes","confidence":"HIGH","evidence":"reduce operation with Wt>1 consumes multiple inputs per output"}
```

```json
{"event":"hypothesis","id":"H2","description":"Wrong PoolType - using SUM instead of AVG","confidence":"MEDIUM","evidence":"values are exactly 10x larger for 10-element reduce"}
```

**Confidence levels:**
- **HIGH**: Strong evidence, likely correct
- **MEDIUM**: Reasonable theory, needs verification
- **LOW**: Speculative, worth checking

### investigation
Log what you're checking to test a hypothesis:
```json
{"event":"investigation","hypothesis_id":"H1","action":"counting CB ops in reader kernel","finding":"reader pushes Wt=10 tiles per block"}
{"event":"investigation","hypothesis_id":"H1","action":"counting CB ops in compute kernel","finding":"compute pops 1 tile per iteration, Wt iterations = 10 pops - MATCHES"}
{"event":"investigation","hypothesis_id":"H1","action":"checking compute output pushes","finding":"compute pushes 1 tile per block to c_16"}
```

### fix_attempt
Log every code change you make:
```json
{"event":"fix_attempt","hypothesis_id":"H1","change":"modified compute loop to pop Wt tiles before producing 1 output tile","files_modified":["device/kernels/compute/reduce_compute.cpp"]}
```

```json
{"event":"fix_attempt","issue":"compile_error","change":"added missing PoolType::AVG template argument","files_modified":["device/kernels/compute/reduce_compute.cpp"]}
```

### fix_result
Log the outcome of every fix:
```json
{"event":"fix_result","hypothesis_id":"H1","success":true,"test_now":"PASS"}
```

```json
{"event":"fix_result","hypothesis_id":"H1","success":false,"new_symptom":"still hangs, different location"}
```

```json
{"event":"fix_result","hypothesis_id":"H2","success":true,"test_now":"numerical match within tolerance"}
```

---

## Optional Events (Log When Relevant)

### design_compliance
Design document adherence per phase:
```json
{"event":"design_compliance","phase":"tilize","directive":"USE HELPER","implementation":"compute_kernel_lib::tilize()","compliant":true}
```

### cb_wrapper_check
Redundant CB operation check:
```json
{"event":"cb_wrapper_check","helper":"tilize","has_wrapper_cb_ops":false,"status":"CLEAN"}
```

### cb_verification
CB configuration verification:
```json
{"event":"cb_verification","cbs_checked":["c_0","c_1","c_16"],"all_match":true}
```

### cb_sync_verified
CB synchronization audit:
```json
{"event":"cb_sync_verified","all_balanced":true,"cb_counts":{"c_0":{"push":10,"pop":10},"c_16":{"push":1,"pop":1}}}
```

### host_file_modified
Track host file changes:
```json
{"event":"host_file_modified","file":"program_factory.cpp","build_required":true}
```

---

## Example Debugging Session (Full Breadcrumb Trail)

This shows what a complete debugging trail looks like:

```jsonl
{"ts":"10:00:01","event":"phase_start","phase":"implementation","design_doc":"reduce_avg_w_rm/kernel_design.md"}
{"ts":"10:00:15","event":"design_parsed","phases_count":5,"helpers_required":["tilize","reduce","untilize"],"raw_phases":["reader","writer"]}
{"ts":"10:05:00","event":"kernel_implemented","kernel":"reader","approach":"TensorAccessor, push Wt tiles per block"}
{"ts":"10:10:00","event":"kernel_implemented","kernel":"compute","approach":"tilize->reduce->untilize sequence"}
{"ts":"10:15:00","event":"kernel_implemented","kernel":"writer","approach":"TensorAccessor, pop 1 tile per block"}
{"ts":"10:16:00","event":"test_run","test":"test_stage7","result":"HANG","failure_type":"timeout","timeout_s":30}
{"ts":"10:16:30","event":"hang_detected","test":"test_stage7","symptom":"timeout after 30s","suspected_cause":"CB sync"}
{"ts":"10:17:00","event":"hypothesis","id":"H1","description":"compute pops fewer tiles than reader pushes","confidence":"HIGH","evidence":"shape-changing op"}
{"ts":"10:18:00","event":"investigation","hypothesis_id":"H1","action":"counting reader pushes","finding":"pushes Wt=10 tiles"}
{"ts":"10:19:00","event":"investigation","hypothesis_id":"H1","action":"counting compute pops","finding":"pops 1 tile per helper call, but helper loops internally"}
{"ts":"10:20:00","event":"investigation","hypothesis_id":"H1","action":"reading reduce_helpers header","finding":"helper pops internally based on ReduceInputBlockShape"}
{"ts":"10:21:00","event":"hypothesis","id":"H2","description":"ReduceInputBlockShape wrong - not consuming all Wt tiles","confidence":"HIGH","evidence":"helper API docs"}
{"ts":"10:22:00","event":"fix_attempt","hypothesis_id":"H2","change":"ReduceInputBlockShape::of(1, Wt, NC) instead of ::of(1, 1, NC)","files_modified":["compute.cpp"]}
{"ts":"10:23:00","event":"test_run","test":"test_stage7","result":"FAIL","failure_type":"numerical","details":"expected 0.07, got 0.7"}
{"ts":"10:23:30","event":"fix_result","hypothesis_id":"H2","success":false,"new_symptom":"no hang but wrong values"}
{"ts":"10:24:00","event":"numerical_error","test":"test_stage7","expected":"0.07","actual":"0.7","ratio":"10x","pattern":"exactly 10x"}
{"ts":"10:25:00","event":"hypothesis","id":"H3","description":"using SUM instead of AVG","confidence":"HIGH","evidence":"10x matches element count"}
{"ts":"10:26:00","event":"fix_attempt","hypothesis_id":"H3","change":"PoolType::AVG instead of PoolType::SUM","files_modified":["compute.cpp"]}
{"ts":"10:27:00","event":"test_run","test":"test_stage7","result":"PASS","duration_s":2.1}
{"ts":"10:27:30","event":"fix_result","hypothesis_id":"H3","success":true,"test_now":"PASS"}
{"ts":"10:28:00","event":"design_compliance_summary","total_phases":5,"all_compliant":true,"test_runs":3,"failures_debugged":2}
```

---

## Agent-Specific Log Section (2a)

Include after Section 2 in your execution log:

### 2a. Design Document Compliance

#### Helper Usage Compliance

| Phase | Design Directive | Your Implementation | Compliant? |
|-------|------------------|---------------------|------------|
| {phase} | USE HELPER: X() | {what you wrote} | YES/NO |
| {phase} | NO HELPER | {raw calls used} | YES/NO |

#### Redundant CB Operation Check

| Helper Used | Wrapper CB Ops Present? | Status |
|-------------|-------------------------|--------|
| compute_kernel_lib::tilize() | YES/NO | CLEAN/VIOLATION |
| compute_kernel_lib::reduce<...>() | YES/NO | CLEAN/VIOLATION |

### 2b. Test Run Summary

| Run # | Test | Result | Failure Type | Duration |
|-------|------|--------|--------------|----------|
| 1 | test_stage7 | HANG | timeout | 30s |
| 2 | test_stage7 | FAIL | numerical | 2.1s |
| 3 | test_stage7 | PASS | - | 2.1s |

### 2c. Debugging Trail

| Hypothesis ID | Description | Confidence | Result |
|---------------|-------------|------------|--------|
| H1 | CB sync mismatch | HIGH | Disproved |
| H2 | ReduceInputBlockShape wrong | HIGH | Partial fix |
| H3 | SUM instead of AVG | HIGH | Fixed issue |

#### Detailed Debugging Cycle for Each Failure

**Failure 1: Hang**
- Symptom: Timeout after 30s
- Hypotheses tested: H1, H2
- Root cause: ReduceInputBlockShape not consuming all input tiles
- Fix: `ReduceInputBlockShape::of(1, Wt, NC)`

**Failure 2: Numerical Error**
- Symptom: Values 10x larger than expected
- Hypotheses tested: H3
- Root cause: Using PoolType::SUM instead of PoolType::AVG
- Fix: Changed template parameter

### 2d. Stage 7 Final Test Results

| Test Case | Input Shape | Tolerance | Result | Notes |
|-----------|-------------|-----------|--------|-------|
| test_basic | [1,1,1,64] | rtol=1e-2 | PASS | |
| test_larger | [1,1,32,128] | rtol=1e-2 | PASS | |

### 2e. Host Files Modified

| File | Build Required | Build Ran | Build Result |
|------|----------------|-----------|--------------|
| {path} | YES/NO | YES/NO | PASS/FAIL/N/A |
