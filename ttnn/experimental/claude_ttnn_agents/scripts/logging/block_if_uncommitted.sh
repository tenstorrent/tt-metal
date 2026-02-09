#!/bin/bash
# block_if_uncommitted.sh - Blocks agent completion if uncommitted changes exist
#
# This script is a BLOCKING hook (exit code 2 = block).
# It enforces that agents commit their work before completing.
#
# Usage: block_if_uncommitted.sh <agent_name>
#
# When blocked, the agent receives the stderr message and gets another turn
# to address the issue (commit changes), then Stop is attempted again.

AGENT_NAME="${1:-unknown-agent}"

# Read hook input from stdin to check stop_hook_active.
# If this hook already blocked once, let the agent through to avoid
# infinite loops (especially with parallel agents whose files we can't commit).
INPUT=$(cat)
if [ "$(echo "$INPUT" | jq -r '.stop_hook_active // false')" = "true" ]; then
    exit 0
fi

# Find repo root
find_repo_root() {
    local dir="$PWD"
    while [[ "$dir" != "/" ]]; do
        if [[ -d "$dir/.git" ]]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    echo ""
    return 1
}

REPO_ROOT=$(find_repo_root)
if [[ -z "$REPO_ROOT" ]]; then
    # Not in a git repo, allow completion
    exit 0
fi

cd "$REPO_ROOT"

# Check for staged changes
if ! git diff --cached --quiet; then
    STAGED_FILES=$(git diff --cached --name-only | head -10)
    STAGED_COUNT=$(git diff --cached --name-only | wc -l)

    cat >&2 << EOF
BLOCKED: You have staged but uncommitted changes.

Staged files (showing first 10 of $STAGED_COUNT):
$STAGED_FILES

ACTION REQUIRED:
1. Commit ONLY the files YOU worked on: git commit -m "[${AGENT_NAME}] ..."
2. Do NOT commit files from other agents running in parallel
3. If you see files you didn't modify, unstage them: git restore --staged <file>

After committing your changes, you can complete.
EOF
    exit 2
fi

# Check for unstaged modifications
if ! git diff --quiet; then
    MODIFIED_FILES=$(git diff --name-only | head -10)
    MODIFIED_COUNT=$(git diff --name-only | wc -l)

    cat >&2 << EOF
BLOCKED: You have modified but uncommitted files.

Modified files (showing first 10 of $MODIFIED_COUNT):
$MODIFIED_FILES

ACTION REQUIRED:
1. Stage ONLY the files YOU worked on: git add <specific-files>
2. Do NOT use 'git add -A' or 'git add .' - this may steal commits from parallel agents
3. Commit with: git commit -m "[${AGENT_NAME}] <description>"

Example for ${AGENT_NAME}:
  git add ttnn/cpp/ttnn/operations/<category>/<your_op>/device/kernels/*.cpp
  git commit -m "[${AGENT_NAME}] stage X: <what you did>"

After committing your changes, you can complete.
EOF
    exit 2
fi

# Check for untracked files in typical operation directories
UNTRACKED=$(git status --porcelain | grep "^??" | grep -E "(ttnn/cpp/ttnn/operations|ttnn/ttnn/operations)" | head -10 || true)
UNTRACKED_COUNT=$(git status --porcelain | grep "^??" | grep -E "(ttnn/cpp/ttnn/operations|ttnn/ttnn/operations)" | wc -l || echo "0")

if [[ -n "$UNTRACKED" ]]; then
    cat >&2 << EOF
BLOCKED: You have untracked files in operation directories.

Untracked files (showing first 10 of $UNTRACKED_COUNT):
$UNTRACKED

ACTION REQUIRED:
1. Stage ONLY the files YOU created: git add <specific-files>
2. Do NOT use 'git add -A' or 'git add .' - this may steal commits from parallel agents
3. Commit with: git commit -m "[${AGENT_NAME}] <description>"

If these files were created by another agent, leave them alone and they will commit them.

After committing your changes, you can complete.
EOF
    exit 2
fi

# All clean - allow completion
exit 0
