# Agent Logging Mechanism

## Overview

Agent breadcrumb logging uses a **SubagentStart hook** to inject logging instructions into every subagent's context automatically. The orchestrator controls whether logging is enabled by creating or omitting a signal file.

## How It Works

```
Orchestrator                    SubagentStart Hook              Subagent
─────────────                   ──────────────────              ────────
1. Creates signal file ──────►
2. Launches agent      ──────► 3. Hook fires
                                4. Reads signal file
                                5. Injects additionalContext ──► 6. Agent sees
                                   with breadcrumb path            "BREADCRUMBS ENABLED"
                                                                   in its context
                                                                7. Agent writes
                                                                   breadcrumbs
```

## Components

### Signal File

**Path**: `.claude/active_logging.json`

**Format**:
```json
{"operation_path": "ttnn/ttnn/operations/{op_name}"}
```

**Created by**: The orchestrator, before launching any agents.
**Deleted by**: The orchestrator, after the pipeline completes.

### SubagentStart Hook

**Script**: `.claude/scripts/logging/inject-logging-context.sh`

**Registered in**: `.claude/settings.local.json` under `hooks.SubagentStart`

**Behavior**:
1. Fires whenever any subagent starts
2. Reads `.claude/active_logging.json` from the repo root
3. If the file exists, outputs JSON with `additionalContext` containing:
   - The breadcrumb file path for this specific agent
   - Instructions to use `append_breadcrumb.sh`
   - What events to log
4. If the file does not exist, exits silently (logging disabled)

The `additionalContext` is injected directly into the agent's context by Claude Code — it is not prompt text that the agent might ignore.

### Breadcrumb Helper

**Script**: `.claude/scripts/logging/append_breadcrumb.sh`

**Usage** (by agents):
```bash
.claude/scripts/logging/append_breadcrumb.sh "{operation_path}" "{agent_name}" '{"event":"...", "details":"..."}'
```

## Orchestrator Instructions

### Enabling Logging

Before launching the first agent in a pipeline, create the signal file:

```bash
mkdir -p .claude && echo '{"operation_path": "ttnn/ttnn/operations/my_op"}' > .claude/active_logging.json
```

All subsequent subagents will automatically receive breadcrumb instructions.

### Disabling Logging

Simply do not create the signal file. Or, if it was created earlier, delete it:

```bash
rm -f .claude/active_logging.json
```

### Cleanup After Pipeline

After all agents complete:

```bash
rm -f .claude/active_logging.json
```

## What Agents Receive

When logging is enabled, each agent's context includes text like:

```
BREADCRUMBS ENABLED — You MUST write breadcrumbs to: ttnn/ttnn/operations/my_op/agent_logs/{agent_name}_breadcrumbs.jsonl

Use the append_breadcrumb.sh helper:
  .claude/scripts/logging/append_breadcrumb.sh "ttnn/ttnn/operations/my_op" "{agent_name}" '{"event":"...", "details":"..."}'

Log after each significant action: file reads, design decisions, test runs (pass/fail/hang), debugging hypotheses, and fixes applied.
```

## Why This Approach

| Approach | Reliability | Problem |
|----------|-------------|---------|
| Prompt text ("enable logging") | Low | Agents sometimes ignore prompt instructions |
| Script check (`check_logging_enabled.sh`) | Low | Agents skip Step 0 or script checks wrong file |
| **SubagentStart hook + additionalContext** | **High** | **Injected by infrastructure, not parse-dependent** |

The hook fires for every subagent launch. The `additionalContext` appears in the agent's context window alongside system instructions — it cannot be skipped or ignored the way a prompt phrase can.
