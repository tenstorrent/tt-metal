#!/bin/bash
# Append a breadcrumb entry to the agent's breadcrumb file
# Usage: append_breadcrumb.sh <operation_path> <agent_name> <event_json>
#
# Example (standard C++ workflow):
#   append_breadcrumb.sh ttnn/cpp/ttnn/operations/reduction/my_op ttnn-operation-scaffolder '{"event":"action","type":"build"}'
#
# Example (generic_op workflow - see ttnn-generic-op-workflow.md for canonical path):
#   append_breadcrumb.sh ttnn/ttnn/operations/my_op ttnn-generic-op-builder '{"event":"action","type":"test"}'
#
# The script automatically adds timestamp and ensures the log directory exists.

set -e

OPERATION_PATH="$1"
AGENT_NAME="$2"
EVENT_JSON="$3"

if [[ -z "$OPERATION_PATH" || -z "$AGENT_NAME" || -z "$EVENT_JSON" ]]; then
    echo "Usage: append_breadcrumb.sh <operation_path> <agent_name> <event_json>" >&2
    exit 1
fi

# Ensure agent_logs directory exists
LOG_DIR="${OPERATION_PATH}/agent_logs"
mkdir -p "$LOG_DIR"

BREADCRUMB_FILE="${LOG_DIR}/${AGENT_NAME}_breadcrumbs.jsonl"

# Add timestamp to the event JSON
TIMESTAMP=$(date -Iseconds)

# Use jq if available for proper JSON handling, otherwise use sed
if command -v jq &> /dev/null; then
    FULL_EVENT=$(echo "$EVENT_JSON" | jq -c --arg ts "$TIMESTAMP" '. + {ts: $ts}')
else
    # Fallback: prepend timestamp to JSON (assumes EVENT_JSON starts with {)
    FULL_EVENT=$(echo "$EVENT_JSON" | sed "s/^{/{\"ts\":\"${TIMESTAMP}\",/")
fi

# Append to breadcrumb file
echo "$FULL_EVENT" >> "$BREADCRUMB_FILE"

echo "Breadcrumb appended to $BREADCRUMB_FILE"
