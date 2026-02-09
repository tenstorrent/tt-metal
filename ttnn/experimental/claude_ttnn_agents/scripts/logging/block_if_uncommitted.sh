#!/bin/bash
# block_if_uncommitted.sh - Blocks agent completion if uncommitted changes exist
#
# This script is a BLOCKING hook (exit code 2 = block).
# It enforces that agents commit their work before completing.
#
# If manifest files exist (from track_agent_file.sh PostToolUse hook),
# the script auto-unstages any files NOT in the agent's manifest to prevent
# parallel agents from committing each other's files.
#
# Usage: block_if_uncommitted.sh <agent_name>
#
# When blocked, the agent receives the stderr message and gets another turn
# to address the issue (commit changes), then Stop is attempted again.

AGENT_NAME="${1:-unknown-agent}"

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

# ========== MANIFEST-BASED AUTO-UNSTAGE ==========
# Find all manifest files for the current session in any operation's agent_logs/
# Each parallel agent gets a unique manifest keyed by session_id
auto_unstage_foreign_files() {
    # Collect this agent's manifest entries (there may be multiple manifest files
    # if the agent worked across multiple operation directories)
    local my_manifests
    my_manifests=$(find . -path "*/agent_logs/manifest_*.txt" -newer /proc/1/cmdline 2>/dev/null || true)

    if [[ -z "$my_manifests" ]]; then
        # No manifests found — no auto-unstage protection available
        return
    fi

    # Build a combined list of files this session owns
    local my_files_tmp
    my_files_tmp=$(mktemp)
    for manifest in $my_manifests; do
        cat "$manifest" >> "$my_files_tmp" 2>/dev/null
    done
    sort -u -o "$my_files_tmp" "$my_files_tmp"

    if [[ ! -s "$my_files_tmp" ]]; then
        rm -f "$my_files_tmp"
        return
    fi

    # Check staged files against manifest
    local staged_files
    staged_files=$(git diff --cached --name-only 2>/dev/null)
    if [[ -z "$staged_files" ]]; then
        rm -f "$my_files_tmp"
        return
    fi

    local unstaged_any=false
    while IFS= read -r staged_file; do
        if ! grep -qxF "$staged_file" "$my_files_tmp" 2>/dev/null; then
            # This file is NOT in our manifest — unstage it
            git restore --staged "$staged_file" 2>/dev/null
            echo "AUTO-UNSTAGED (not yours): $staged_file" >&2
            unstaged_any=true
        fi
    done <<< "$staged_files"

    if [[ "$unstaged_any" == "true" ]]; then
        echo "" >&2
        echo "Files from other parallel agents were auto-unstaged." >&2
        echo "Only files you created/modified remain staged." >&2
        echo "" >&2
    fi

    rm -f "$my_files_tmp"
}

# Run auto-unstage before checking git state
auto_unstage_foreign_files

# ========== STANDARD CHECKS ==========

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

# Check for unstaged modifications — only block on files in our manifest
MODIFIED_FILES=$(git diff --name-only)
if [[ -n "$MODIFIED_FILES" ]]; then
    # If we have a manifest, only care about our own modified files
    my_manifests=$(find . -path "*/agent_logs/manifest_*.txt" -newer /proc/1/cmdline 2>/dev/null || true)
    if [[ -n "$my_manifests" ]]; then
        my_files_tmp=$(mktemp)
        for manifest in $my_manifests; do
            cat "$manifest" >> "$my_files_tmp" 2>/dev/null
        done
        sort -u -o "$my_files_tmp" "$my_files_tmp"

        OUR_MODIFIED=""
        while IFS= read -r mod_file; do
            if grep -qxF "$mod_file" "$my_files_tmp" 2>/dev/null; then
                OUR_MODIFIED="${OUR_MODIFIED}${mod_file}\n"
            fi
        done <<< "$MODIFIED_FILES"
        rm -f "$my_files_tmp"

        if [[ -n "$OUR_MODIFIED" ]]; then
            cat >&2 << EOF
BLOCKED: You have modified but uncommitted files.

Your modified files:
$(echo -e "$OUR_MODIFIED" | head -10)

ACTION REQUIRED:
1. Stage your files: git add <specific-files>
2. Commit with: git commit -m "[${AGENT_NAME}] <description>"

After committing your changes, you can complete.
EOF
            exit 2
        fi
        # Other agents' modified files exist but not ours — allow
    else
        # No manifest — fall back to blocking on all modified files
        MODIFIED_COUNT=$(echo "$MODIFIED_FILES" | wc -l)
        cat >&2 << EOF
BLOCKED: You have modified but uncommitted files.

Modified files (showing first 10 of $MODIFIED_COUNT):
$(echo "$MODIFIED_FILES" | head -10)

ACTION REQUIRED:
1. Stage ONLY the files YOU worked on: git add <specific-files>
2. Do NOT use 'git add -A' or 'git add .' - this may steal commits from parallel agents
3. Commit with: git commit -m "[${AGENT_NAME}] <description>"

After committing your changes, you can complete.
EOF
        exit 2
    fi
fi

# Check for untracked files in operation directories — only block on our own
UNTRACKED=$(git status --porcelain | grep "^??" | grep -E "(ttnn/cpp/ttnn/operations|ttnn/ttnn/operations)" || true)
if [[ -n "$UNTRACKED" ]]; then
    my_manifests=$(find . -path "*/agent_logs/manifest_*.txt" -newer /proc/1/cmdline 2>/dev/null || true)
    if [[ -n "$my_manifests" ]]; then
        my_files_tmp=$(mktemp)
        for manifest in $my_manifests; do
            cat "$manifest" >> "$my_files_tmp" 2>/dev/null
        done
        sort -u -o "$my_files_tmp" "$my_files_tmp"

        OUR_UNTRACKED=""
        while IFS= read -r line; do
            untracked_file=$(echo "$line" | sed 's/^?? //')
            if grep -qxF "$untracked_file" "$my_files_tmp" 2>/dev/null; then
                OUR_UNTRACKED="${OUR_UNTRACKED}${untracked_file}\n"
            fi
        done <<< "$UNTRACKED"
        rm -f "$my_files_tmp"

        if [[ -n "$OUR_UNTRACKED" ]]; then
            cat >&2 << EOF
BLOCKED: You have untracked files that you created.

Your untracked files:
$(echo -e "$OUR_UNTRACKED" | head -10)

ACTION REQUIRED:
1. Stage your files: git add <specific-files>
2. Commit with: git commit -m "[${AGENT_NAME}] <description>"

After committing your changes, you can complete.
EOF
            exit 2
        fi
        # Other agents' untracked files — not our problem
    else
        UNTRACKED_DISPLAY=$(echo "$UNTRACKED" | head -10)
        UNTRACKED_COUNT=$(echo "$UNTRACKED" | wc -l)
        cat >&2 << EOF
BLOCKED: You have untracked files in operation directories.

Untracked files (showing first 10 of $UNTRACKED_COUNT):
$UNTRACKED_DISPLAY

ACTION REQUIRED:
1. Stage ONLY the files YOU created: git add <specific-files>
2. Do NOT use 'git add -A' or 'git add .' - this may steal commits from parallel agents
3. Commit with: git commit -m "[${AGENT_NAME}] <description>"

If these files were created by another agent, leave them alone and they will commit them.

After committing your changes, you can complete.
EOF
        exit 2
    fi
fi

# All clean - allow completion
exit 0
