# Breadcrumb Logging - ttnn-operation-scaffolder

**Only read this file if breadcrumb logging is ENABLED.**

First read: `.claude/references/logging/common.md`

---

## Agent Identity

- **Agent name**: `ttnn-operation-scaffolder`
- **Predecessor**: `ttnn-operation-planner` (or empty if first in pipeline)
- **Stages owned**: 1, 2, 3

---

## Agent-Specific Events

Use standard `action/result` with `type: "script_run"`:

```json
{"event":"action","type":"script_run","script":"generate_files.py","args":["--force"]}
{"event":"result","type":"script_run","script":"generate_files.py","success":true,"output":"Created 12 files"}
```

---

## Agent-Specific Log Section (2a)

Include after Section 2 in your execution log:

### 2a. Script Execution Log

| Script | Arguments | Result | Output Summary |
|--------|-----------|--------|----------------|
| generate_files.py | {args} | SUCCESS/FAIL | {brief output} |
| integrate_build.py | {args} | SUCCESS/FAIL | {brief output} |
| verify_scaffolding.sh | {args} | SUCCESS/FAIL | {checks passed} |

### JSON Config Validation

| Check | Result | Notes |
|-------|--------|-------|
| JSON syntax valid | PASS/FAIL | |
| All required fields present | PASS/FAIL | {missing fields if any} |
| C++ expressions valid | PASS/FAIL | {issues found} |

### Spec Parsing Decisions

| Spec Field | Parsed Value | Inference Required? |
|------------|--------------|---------------------|
| operation_name | {value} | YES/NO |
| parameters | {list} | YES/NO |
| validations | {count} | YES/NO |
