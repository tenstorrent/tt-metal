#!/bin/bash
# PostToolUseFailure hook for kernel-writer: classify error and remind agent to log
INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

[[ "$COMMAND" == *dev-test.sh* || "$COMMAND" == *pytest* ]] || exit 0

ERROR=$(echo "$INPUT" | jq -r '.error // empty')

# Hang: dev-test.sh exits 2, or triage log was generated
if [[ "$ERROR" == *"status code 2"* || "$ERROR" == *"exit code 2"* ]] || [[ -s /tmp/dev-test-triage.log ]]; then
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "PostToolUseFailure",
      additionalContext: "HANG DETECTED. You MUST: 1) If logging is enabled, log a hang_detected breadcrumb with the test path and CB state from triage. 2) Read the triage callstacks and watcher log above — identify which RISC-V is stuck and on which CB. 3) If logging is enabled, log a hypothesis breadcrumb. 4) Cross-check against the design CB Sync Summary table before changing code."
    }
  }'
else
  # Normal failure — numeric error, compile error, assert, etc.
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "PostToolUseFailure",
      additionalContext: "Test FAILED. You MUST: 1) If logging is enabled, log a test_run breadcrumb (status=fail, include error summary). 2) Classify: numerical_error (wrong values), compile_error, or assert failure. 3) If logging is enabled, log a hypothesis breadcrumb BEFORE making any code changes."
    }
  }'
fi
