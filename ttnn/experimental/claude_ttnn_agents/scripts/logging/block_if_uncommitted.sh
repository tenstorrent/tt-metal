#!/bin/bash
# block_if_uncommitted.sh - One-shot reminder for agents to commit before stopping
#
# Blocks ONCE if the agent has uncommitted changes, then allows completion
# on subsequent attempts. The agent is smart enough to commit — this is
# just a nudge in case it forgot.
#
# Usage: block_if_uncommitted.sh <agent_name>
#
# Exit codes:
#   0 - Allow agent to stop
#   2 - Block (first time only) with a reminder to commit

AGENT_NAME="${1:-unknown-agent}"
INPUT=$(cat)

# Extract session_id for per-session state tracking
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty' 2>/dev/null)
STATE_FILE="/tmp/claude_commit_reminder_${AGENT_NAME}_${SESSION_ID:-unknown}"

# Already reminded once — let it go
if [[ -f "$STATE_FILE" ]]; then
    exit 0
fi

# Find repo root
REPO_ROOT="$PWD"
while [[ "$REPO_ROOT" != "/" && ! -d "$REPO_ROOT/.git" ]]; do
    REPO_ROOT="$(dirname "$REPO_ROOT")"
done
if [[ ! -d "$REPO_ROOT/.git" ]]; then
    exit 0
fi

cd "$REPO_ROOT"

# Check if there are any uncommitted changes (staged, unstaged, or untracked in op dirs)
HAS_STAGED=false
HAS_MODIFIED=false
HAS_UNTRACKED=false

git diff --cached --quiet 2>/dev/null || HAS_STAGED=true
git diff --quiet 2>/dev/null || HAS_MODIFIED=true
UNTRACKED=$(git status --porcelain 2>/dev/null | grep "^??" | grep -E "(ttnn/cpp/ttnn/operations|ttnn/ttnn/operations)" || true)
[[ -n "$UNTRACKED" ]] && HAS_UNTRACKED=true

if [[ "$HAS_STAGED" == "false" && "$HAS_MODIFIED" == "false" && "$HAS_UNTRACKED" == "false" ]]; then
    # Nothing to commit — allow
    exit 0
fi

# First block — mark state and remind
touch "$STATE_FILE"

cat >&2 << EOF
REMINDER: You have uncommitted changes. Please commit your work before finishing.

- Stage ONLY files you created/modified (not files from parallel agents)
- Commit with: git commit -m "[${AGENT_NAME}] <description>"
EOF

exit 2
