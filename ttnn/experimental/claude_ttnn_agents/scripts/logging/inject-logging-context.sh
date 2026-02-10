#!/bin/bash
# inject-logging-context.sh — SubagentStart hook
#
# When a subagent starts, this hook checks if breadcrumb logging is enabled.
# If so, it injects logging instructions directly into the agent's context
# via the SubagentStart additionalContext mechanism.
#
# Signal file: .claude/active_logging (just needs to exist, no content required)
#
# Enable:  touch .claude/active_logging
# Disable: rm -f .claude/active_logging
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

SIGNAL_FILE="$REPO_ROOT/.claude/active_logging"

# No signal file → logging disabled → exit silently
if [[ ! -f "$SIGNAL_FILE" ]]; then
    exit 0
fi

# Normalize agent type for filename (lowercase, hyphens)
AGENT_NAME=$(echo "$AGENT_TYPE" | tr '[:upper:]' '[:lower:]' | tr ' ' '-')

# Inject logging instructions into the agent's context
jq -n \
    --arg agent "$AGENT_NAME" \
    '{
        hookSpecificOutput: {
            hookEventName: "SubagentStart",
            additionalContext: ("BREADCRUMBS ENABLED — You MUST write breadcrumbs to {operation_path}/agent_logs/" + $agent + "_breadcrumbs.jsonl where {operation_path} is the operation directory from your prompt.\n\nUse the append_breadcrumb.sh helper:\n```bash\n.claude/scripts/logging/append_breadcrumb.sh \"{operation_path}\" \"" + $agent + "\" \u0027{\"event\":\"...\",\"details\":\"...\"}\u0027\n```\n\nLog after each significant action: file reads, design decisions, test runs (pass/fail/hang), debugging hypotheses, and fixes applied.")
        }
    }'

exit 0
