#!/bin/bash
# inject-logging-context.sh — SubagentStart hook
#
# When a subagent starts, this hook checks if logging is enabled for the
# current session. If so, it injects breadcrumb instructions directly into
# the agent's context via the SubagentStart additionalContext mechanism.
#
# Signal file: .claude/active_logging.json (created by orchestrator)
# Format:      {"operation_path": "ttnn/ttnn/operations/{op_name}"}
#
# See .claude/references/logging-mechanism.md for full documentation.
#
# Exit codes:
#   0 — always (hook must never block agent startup)

set -euo pipefail

INPUT=$(cat)
AGENT_TYPE=$(echo "$INPUT" | jq -r '.agent_type // empty' 2>/dev/null)
CWD=$(echo "$INPUT" | jq -r '.cwd // empty' 2>/dev/null)

# Find repo root from cwd
REPO_ROOT="${CWD}"
while [[ "$REPO_ROOT" != "/" && ! -d "$REPO_ROOT/.git" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
if [[ ! -d "$REPO_ROOT/.git" ]]; then
    exit 0
fi

SIGNAL_FILE="$REPO_ROOT/.claude/active_logging.json"

# No signal file → logging disabled → exit silently
if [[ ! -f "$SIGNAL_FILE" ]]; then
    exit 0
fi

# Read operation path from signal file
OP_PATH=$(jq -r '.operation_path // empty' "$SIGNAL_FILE" 2>/dev/null)
if [[ -z "$OP_PATH" ]]; then
    exit 0
fi

# Ensure agent_logs directory exists
BREADCRUMB_DIR="$REPO_ROOT/$OP_PATH/agent_logs"
mkdir -p "$BREADCRUMB_DIR" 2>/dev/null

# Normalize agent type for filename (lowercase, hyphens)
AGENT_NAME=$(echo "$AGENT_TYPE" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
BREADCRUMB_FILE="$BREADCRUMB_DIR/${AGENT_NAME}_breadcrumbs.jsonl"

# Inject logging instructions into the agent's context
jq -n \
    --arg breadcrumb_file "$OP_PATH/agent_logs/${AGENT_NAME}_breadcrumbs.jsonl" \
    --arg op_path "$OP_PATH" \
    --arg agent "$AGENT_NAME" \
    '{
        hookSpecificOutput: {
            hookEventName: "SubagentStart",
            additionalContext: ("BREADCRUMBS ENABLED — You MUST write breadcrumbs to: " + $breadcrumb_file + "\n\nUse the append_breadcrumb.sh helper:\n```bash\n.claude/scripts/logging/append_breadcrumb.sh \"" + $op_path + "\" \"" + $agent + "\" \u0027{\"event\":\"...\",\"details\":\"...\"}\u0027\n```\n\nLog after each significant action: file reads, design decisions, test runs (pass/fail/hang), debugging hypotheses, and fixes applied.")
        }
    }'

exit 0
