# Breadcrumb Logging - Common Reference

This file contains shared breadcrumb logging concepts used by all TTNN agents.

**Only read this file if breadcrumb logging is ENABLED.**

---

## Quick Start

### 1. Initialize breadcrumbs at agent start:
```bash
mkdir -p {operation_path}/agent_logs
.claude/scripts/logging/init_breadcrumbs.sh \
  {operation_path} \
  {agent_name} \
  {operation_name} \
  "{predecessor_agent}" \
  "{input_file_path}"
```

### 2. Append events at key decision points:
```bash
.claude/scripts/logging/append_breadcrumb.sh {operation_path} {agent_name} '{json}'
```

### 3. At completion, write execution log:
- Read `.claude/references/logging/{agent_name}.md` for agent-specific sections
- Read `{operation_path}/agent_logs/{agent_name}_breadcrumbs.jsonl`
- Write structured log to `{operation_path}/agent_logs/{agent_name}_execution_log.md`

---

## Common Event Types

Use these JSON patterns with `append_breadcrumb.sh`:

### input_parse
After extracting fields from input:
```json
{"event":"input_parse","field":"operation_name","value":"my_op","confidence":"HIGH"}
```

### action
Before executing an action:
```json
{"event":"action","type":"build","command":"./build_metal.sh -b Debug"}
{"event":"action","type":"test","command":"pytest test_file.py"}
{"event":"action","type":"git_commit","message":"[agent] stage X: description"}
```

### result
After an action completes:
```json
{"event":"result","type":"build","success":true}
{"event":"result","type":"build","success":false,"error":"missing semicolon"}
```

### hypothesis
When diagnosing an error:
```json
{"event":"hypothesis","id":"H1","description":"Missing () on method call","confidence":"HIGH"}
```

### recovery
When fixing an issue:
```json
{"event":"recovery","hypothesis_id":"H1","action":"Added () to method call","file":"device/my_op.cpp"}
```

### test
After running a test:
```json
{"event":"test","stage":1,"result":"PASS"}
{"event":"test","stage":2,"result":"FAIL","error_summary":"Wrong error message"}
```

### complete
At end of execution:
```json
{"event":"complete","final_status":"SUCCESS","stages_completed":[1,2,3],"final_commit":"abc1234"}
```

---

## Confidence Levels

Use for `input_parse` confidence:
- **HIGH**: Explicitly stated in input, no interpretation needed
- **MEDIUM**: Required some inference or combining sources
- **LOW**: Significant guesswork, input was ambiguous

---

## Execution Log Template

The structured log should include these sections:

1. **Metadata**: Operation, agent, stages, status
2. **Input Interpretation**: Extracted fields with confidence
3. **Execution Timeline**: Per-stage attempts
4. **Recovery Summary**: Error table, attempts per stage
5. **Deviations**: Any instruction deviations
6. **Artifacts**: Files created/modified
7. **Handoff Notes**: For next agent
8. **Instruction Recommendations**: Based on observations

Plus **agent-specific sections** defined in each agent's logging file.

---

## Checklist Before Completing

- [ ] Breadcrumbs initialized at start
- [ ] Key events logged during execution
- [ ] Git commits logged as events
- [ ] Breadcrumbs file read before writing log
- [ ] All sections filled in execution log
- [ ] Agent-specific sections included
- [ ] Log written to correct path
