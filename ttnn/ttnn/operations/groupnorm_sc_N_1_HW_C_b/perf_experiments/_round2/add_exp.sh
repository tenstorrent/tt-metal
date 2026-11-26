#!/usr/bin/env bash
# git-add an experiment dir's durable artifacts: sources, diffs, small results / summaries (no raw logs or device CSVs).
set -eu
for d in "$@"; do
  find "$d" -name __pycache__ -prune -o -type f \
    \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.py' -o -name '*.sh' -o -name '*.diff' -o -name '*.patch' \
       -o -name '*.md' -o -name '*.json' -o -name '*.jsonl' -o -name '*.txt' -o -name '*.graduated' \) \
    -size -300k -print0 | xargs -0 -r git add
done
