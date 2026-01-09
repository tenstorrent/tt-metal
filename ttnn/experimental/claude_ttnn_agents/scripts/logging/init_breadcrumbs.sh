#!/bin/bash
# Initialize breadcrumb file for an agent run
# Usage: init_breadcrumbs.sh <operation_path> <agent_name> <operation_name> [predecessor_agent] [input_files...]
#
# Example:
#   init_breadcrumbs.sh ttnn/cpp/ttnn/operations/reduction/my_op ttnn-operation-scaffolder my_op "" spec.md
#   init_breadcrumbs.sh ttnn/cpp/ttnn/operations/reduction/my_op ttnn-factory-builder my_op ttnn-operation-scaffolder spec.md

set -e

OPERATION_PATH="$1"
AGENT_NAME="$2"
OPERATION_NAME="$3"
PREDECESSOR="$4"
shift 4
INPUT_FILES=("$@")

if [[ -z "$OPERATION_PATH" || -z "$AGENT_NAME" || -z "$OPERATION_NAME" ]]; then
    echo "Usage: init_breadcrumbs.sh <operation_path> <agent_name> <operation_name> [predecessor] [input_files...]" >&2
    exit 1
fi

# Ensure agent_logs directory exists
LOG_DIR="${OPERATION_PATH}/agent_logs"
mkdir -p "$LOG_DIR"

BREADCRUMB_FILE="${LOG_DIR}/${AGENT_NAME}_breadcrumbs.jsonl"

# Build input_files JSON array
INPUT_FILES_JSON="["
FIRST=true
for f in "${INPUT_FILES[@]}"; do
    if [[ -n "$f" ]]; then
        if $FIRST; then
            INPUT_FILES_JSON+="\"$f\""
            FIRST=false
        else
            INPUT_FILES_JSON+=",\"$f\""
        fi
    fi
done
INPUT_FILES_JSON+="]"

# Build the start event
TIMESTAMP=$(date -Iseconds)

if [[ -n "$PREDECESSOR" ]]; then
    START_EVENT="{\"ts\":\"${TIMESTAMP}\",\"event\":\"start\",\"agent\":\"${AGENT_NAME}\",\"operation\":\"${OPERATION_NAME}\",\"input_files\":${INPUT_FILES_JSON},\"predecessor_agent\":\"${PREDECESSOR}\"}"
else
    START_EVENT="{\"ts\":\"${TIMESTAMP}\",\"event\":\"start\",\"agent\":\"${AGENT_NAME}\",\"operation\":\"${OPERATION_NAME}\",\"input_files\":${INPUT_FILES_JSON}}"
fi

# Write to breadcrumb file (overwrite if exists from previous run)
echo "$START_EVENT" > "$BREADCRUMB_FILE"

echo "Initialized breadcrumbs at $BREADCRUMB_FILE"
