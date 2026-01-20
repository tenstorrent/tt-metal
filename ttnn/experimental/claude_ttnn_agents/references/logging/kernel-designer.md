# Breadcrumb Logging - ttnn-kernel-designer

**Only read this file if breadcrumb logging is ENABLED.**

First read: `.claude/references/logging/common.md`

---

## Agent Identity

- **Agent name**: `ttnn-kernel-designer`
- **Predecessor**: `ttnn-factory-builder`
- **Produces**: Kernel Design Document

---

## Agent-Specific Events

Use standard `action/result` events for:
- File reads (helper headers, spec files)
- Design decisions

### helper_analysis
After analyzing a helper header:
```json
{"event":"helper_analysis","file":"tilize_helpers.hpp","applicable":true,"functions":["tilize()"]}
```

### design_decision
Key design choices:
```json
{"event":"design_decision","phase":"reduce","choice":"USE HELPER","helper":"reduce<AVG, REDUCE_ROW>","rationale":"Helper handles CB ops internally"}
```

---

## Agent-Specific Log Section (2a)

Include after Section 2 in your execution log:

### 2a. Helper Library Analysis

| Helper File | Read? | Applicable? | Functions Identified |
|-------------|-------|-------------|---------------------|
| tilize_helpers.hpp | YES/NO | YES/NO | {list or "N/A"} |
| untilize_helpers.hpp | YES/NO | YES/NO | {list or "N/A"} |
| reduce_helpers.hpp | YES/NO | YES/NO | {list or "N/A"} |
| binary_op_helpers.hpp | YES/NO | YES/NO | {list or "N/A"} |
| dest_helpers.hpp | YES/NO | YES/NO | {list or "N/A"} |

### Phase-to-Helper Mapping

| Phase | Implementation Approach | Rationale |
|-------|------------------------|-----------|
| {phase} | USE HELPER: X() | {why} |
| {phase} | NO HELPER: raw calls | {why no helper available} |

### Encapsulation Notes

For phases marked "USE HELPER", documented that helpers handle:
- [ ] CB wait/pop/reserve/push
- [ ] DST register management
- [ ] Init/uninit sequences
