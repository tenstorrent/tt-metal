# Breadcrumb Logging - ttnn-generic-op-builder

**Only read this file if breadcrumb logging is ENABLED.**

First read: `.claude/references/logging/common.md`

---

## Agent Identity

- **Agent name**: `ttnn-generic-op-builder`
- **Predecessor**: `ttnn-operation-planner`
- **Operation path**: `ttnn/ttnn/operations/{operation_name}` (see `ttnn-generic-op-workflow.md`)

---

## Agent-Specific Events

### cb_config
CB configuration decisions:
```json
{"event":"cb_config","cb_id":0,"page_size":2048,"num_pages":2,"purpose":"input"}
```

### work_distribution
Work split calculations:
```json
{"event":"work_distribution","grid":"8x8","total_tiles":256,"tiles_per_core":4}
```

### file_created
File creation tracking:
```json
{"event":"file_created","file":"op/my_op.py","type":"entry_point"}
{"event":"file_created","file":"op/kernels/my_op_reader.cpp","type":"kernel_stub"}
```

### cb_audit
CB sync verification:
```json
{"event":"cb_audit","cb_id":0,"producer":"reader","push_count":"N","consumer":"compute","pop_count":"N","balanced":true}
```

### cb_sync_summary
Final CB balance check (**CRITICAL - log before completing**):
```json
{"event":"cb_sync_summary","total_cbs":2,"all_balanced":true}
```

### test_run
Test execution (no build required):
```json
{"event":"test_run","test":"test_my_op.py","result":"PASS"}
{"event":"test_run","test":"test_my_op.py","result":"FAIL","error":"shape mismatch"}
```

### hang_debug
When debugging hangs:
```json
{"event":"hang_debug","symptom":"timeout","diagnosis":"CB sync mismatch"}
```

---

## Agent-Specific Log Section (2a)

Include after Section 2 in your execution log:

### 2a. Circular Buffer Configuration

| CB ID | Page Size | Num Pages | Data Format | Purpose |
|-------|-----------|-----------|-------------|---------|
| 0 | {bytes} | {count} | {format} | input |
| 16 | {bytes} | {count} | {format} | output |

### CB Synchronization Verification (CRITICAL)

| CB | Producer | Push Operation | Consumer | Pop Operation | Balanced? |
|----|----------|----------------|----------|---------------|-----------|
| 0 | Reader | cb_push_back x N | Compute | cb_pop_front x N | YES/NO |
| 16 | Compute | cb_push_back x M | Writer | cb_pop_front x M | YES/NO |

### Work Distribution

| Parameter | Value | Source |
|-----------|-------|--------|
| Core grid | {rows x cols} | Spec/Calculated |
| Total work units | {count} | |
| Work per core | {formula} | |

### Files Created

| File | Type | Purpose |
|------|------|---------|
| op/{op_name}.py | Entry point | Output allocation, generic_op call |
| op/{op_name}_program_descriptor.py | Program descriptor | CB config, kernel setup, runtime args |
| op/kernels/{op_name}_reader.cpp | Kernel stub | Data movement DRAM → L1 |
| op/kernels/{op_name}_compute.cpp | Kernel stub | FPU/SFPU operations |
| op/kernels/{op_name}_writer.cpp | Kernel stub | Data movement L1 → DRAM |
| tests/test_{op_name}.py | Test | PyTorch reference comparison |

### Test Results

| Test | Result | Notes |
|------|--------|-------|
| Stub compiles | PASS/FAIL | Kernels compile at runtime |
| generic_op executes | PASS/FAIL | No hang |
| Output shape correct | PASS/FAIL | Shape matches expected |
