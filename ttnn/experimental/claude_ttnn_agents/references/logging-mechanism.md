# Agent Logging Mechanism

## Overview

Agent breadcrumb logging uses a **SubagentStart hook** to inject logging instructions into every subagent's context automatically. Logging is controlled by the presence of a single signal file.

## Quick Start

```bash
# Enable logging (persists across runs):
touch .claude/active_logging

# Disable logging:
rm -f .claude/active_logging
```

That's it. When the file exists, every subagent gets breadcrumb instructions injected into its context. When it doesn't, logging is silent.

## How It Works

```
.claude/active_logging exists?
        │
        ├── NO  → SubagentStart hook exits silently → agent runs without logging
        │
        └── YES → SubagentStart hook injects additionalContext:
                   "BREADCRUMBS ENABLED — write to {op_path}/agent_logs/{agent}_breadcrumbs.jsonl"
                   → agent sees this in its context and writes breadcrumbs
```

## Components

### Signal File

**Path**: `.claude/active_logging`

**Format**: Empty file — just needs to exist. No JSON, no content.

**Created by**: The user (manually) or the orchestrator.

### SubagentStart Hook

**Script**: `.claude/scripts/logging/inject-logging-context.sh`

**Registered in**: `.claude/settings.local.json` under `hooks.SubagentStart`

**Behavior**:
1. Fires whenever any subagent starts
2. Checks if `.claude/active_logging` exists in the repo root
3. If it exists, outputs JSON with `additionalContext` containing breadcrumb instructions
4. If it doesn't exist, exits silently (logging disabled)

The `additionalContext` is injected directly into the agent's context by Claude Code — it is not prompt text that the agent might ignore.

### Breadcrumb Helper

**Script**: `.claude/scripts/logging/append_breadcrumb.sh`

**Usage** (by agents):
```bash
.claude/scripts/logging/append_breadcrumb.sh "{operation_path}" "{agent_name}" '{"event":"...", "details":"..."}'
```

## What Agents Receive

When logging is enabled, each agent's context includes:

```
BREADCRUMBS ENABLED — You MUST write breadcrumbs to {operation_path}/agent_logs/{agent_name}_breadcrumbs.jsonl
where {operation_path} is the operation directory from your prompt.

Use the append_breadcrumb.sh helper:
  .claude/scripts/logging/append_breadcrumb.sh "{operation_path}" "{agent_name}" '{"event":"...", "details":"..."}'

Log after each significant action: file reads, design decisions, test runs (pass/fail/hang),
debugging hypotheses, and fixes applied.
```

The agent derives `{operation_path}` from its own prompt (which always contains the operation path).

## Why This Approach

| Approach | Reliability | Problem |
|----------|-------------|---------|
| Prompt text ("enable logging") | Low | Agents sometimes ignore prompt instructions |
| Script check (`check_logging_enabled.sh`) | Low | Agents skip Step 0 or script checks wrong file |
| **SubagentStart hook + additionalContext** | **High** | **Injected by infrastructure, not parse-dependent** |

The hook fires for every subagent launch. The `additionalContext` appears in the agent's context window alongside system instructions — it cannot be skipped or ignored the way a prompt phrase can.
