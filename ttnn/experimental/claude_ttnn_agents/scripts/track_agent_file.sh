#!/bin/bash
# track_agent_file.sh - PostToolUse hook for Write|Edit tools
#
# Reads tool_input.file_path from stdin JSON and appends it to the agent's
# manifest file. Used by block_if_uncommitted.sh to enforce that agents
# only commit files they actually created/modified.
#
# Requires: jq
# Exit 0 always (tracking failure should never block the agent's work)

INPUT=$(cat)

FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty' 2>/dev/null)
if [[ -z "$FILE_PATH" ]]; then
    exit 0
fi

# Determine manifest location from the file path
# Look for the nearest agent_logs/ directory in the operation path
OP_DIR=""
if [[ "$FILE_PATH" == *"/ttnn/ttnn/operations/"* ]]; then
    # Extract: .../ttnn/ttnn/operations/{op_name}/...
    OP_DIR=$(echo "$FILE_PATH" | sed 's|\(.*ttnn/ttnn/operations/[^/]*\)/.*|\1|')
elif [[ "$FILE_PATH" == *"/ttnn/cpp/ttnn/operations/"* ]]; then
    OP_DIR=$(echo "$FILE_PATH" | sed 's|\(.*ttnn/cpp/ttnn/operations/[^/]*/[^/]*\)/.*|\1|')
fi

if [[ -z "$OP_DIR" || ! -d "$OP_DIR" ]]; then
    exit 0
fi

MANIFEST_DIR="$OP_DIR/agent_logs"
mkdir -p "$MANIFEST_DIR" 2>/dev/null

# Use session_id as a unique key so parallel agents get separate manifests
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty' 2>/dev/null)
if [[ -z "$SESSION_ID" ]]; then
    exit 0
fi

MANIFEST="$MANIFEST_DIR/manifest_${SESSION_ID}.txt"

# Make path relative to repo root for git comparison
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
REL_PATH="${FILE_PATH#$REPO_ROOT/}"

# Append if not already tracked (dedup)
grep -qxF "$REL_PATH" "$MANIFEST" 2>/dev/null || echo "$REL_PATH" >> "$MANIFEST"

exit 0
