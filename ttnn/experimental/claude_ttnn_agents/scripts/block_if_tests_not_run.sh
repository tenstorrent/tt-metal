#!/bin/bash
# block_if_tests_not_run.sh - Blocks agent completion if test files exist but weren't run
#
# Checks for test_*.py files in ttnn/ttnn/operations/ that were created/modified
# but have no corresponding __pycache__ entry (meaning pytest never imported them).
#
# Exit codes:
#   0 - Tests were run (or no test files found)
#   2 - Tests exist but were not run (BLOCK)

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# Find test files that are new or modified (untracked or changed)
TEST_FILES=$(git status --porcelain -- 'ttnn/ttnn/operations/*/test_*.py' 2>/dev/null | awk '{print $NF}')

if [[ -z "$TEST_FILES" ]]; then
    # No new/modified test files — nothing to check
    exit 0
fi

for test_file in $TEST_FILES; do
    dir=$(dirname "$test_file")
    basename=$(basename "$test_file" .py)

    # Check if __pycache__ contains a compiled version of this test file
    if ! ls "$dir/__pycache__/${basename}."cpython-*.pyc &>/dev/null; then
        cat >&2 << EOF
BLOCKED: You created/modified $test_file but never ran it.

ACTION REQUIRED:
Run your tests before completing:
  .claude/scripts/dev-test.sh $test_file

This catches CB misconfigurations, DataFormat errors, and allocation failures
before downstream agents have to deal with them.
EOF
        exit 2
    fi
done

exit 0
