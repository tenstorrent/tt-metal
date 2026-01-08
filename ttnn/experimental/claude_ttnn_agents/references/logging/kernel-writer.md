# Breadcrumb Logging - ttnn-kernel-writer

**Only read this file if breadcrumb logging is ENABLED.**

First read: `.claude/references/logging/common.md`

---

## Agent Identity

- **Agent name**: `ttnn-kernel-writer`
- **Predecessor**: `ttnn-kernel-designer`
- **Stages owned**: 7

---

## Agent-Specific Events

### design_compliance
Design document adherence:
```json
{"event":"design_compliance","phase":"tilize","directive":"USE HELPER","implementation":"compute_kernel_lib::tilize()","compliant":true}
```

### cb_wrapper_check
Redundant CB operation check:
```json
{"event":"cb_wrapper_check","helper":"tilize","has_wrapper_cb_ops":false,"status":"CLEAN"}
```

### correctness_test
Test case results:
```json
{"event":"correctness_test","test_name":"test_basic","expected":"0.07","actual":"0.07","pass":true}
```

### numerical_debug
Debugging wrong values:
```json
{"event":"numerical_debug","symptom":"values 10x smaller","finding":"scaler format wrong"}
```

### design_compliance_summary
Final compliance check (**CRITICAL - log before completing**):
```json
{"event":"design_compliance_summary","total_phases":5,"all_compliant":true}
```

### host_file_modified
Track host file changes:
```json
{"event":"host_file_modified","file":"program_factory.cpp","build_required":true}
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

### Stage 7 Correctness Test Results

| Test Case | Input Shape | Tolerance | Result | Notes |
|-----------|-------------|-----------|--------|-------|
| {test} | {shape} | rtol={}, atol={} | PASS/FAIL | |

### Numerical Debugging (if applicable)

| Symptom | Investigation | Root Cause | Fix |
|---------|---------------|------------|-----|
| {symptom} | {what checked} | {cause} | {fix} |

### Host Files Modified

| File | Build Required | Build Ran | Build Result |
|------|----------------|-----------|--------------|
| {path} | YES/NO | YES/NO | PASS/FAIL/N/A |
