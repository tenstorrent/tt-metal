#!/usr/bin/env bash
# Correctness of a variant: verify.sh <variant> <logdir> [suite ...]; suites default to the golden suite and
# the op's unit tests. The op's files are not touched: issue_path_plugin.py points the descriptor at
# kernels_<variant>/ for the session and reports how many descriptors it saw and whether any used another folder.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
set +u; source python_env/bin/activate; set -u
export TT_METAL_HOME=$PWD
E=ttnn/ttnn/operations/g6_clone/perf_experiments/issue_path
R=generated/meta/g6_clone/lanes/perf1_issue_path; mkdir -p $R; cp $E/issue_path_plugin.py $R/
export ISSUE_KERNEL_ROOT=$PWD/$E ISSUE_VARIANT=$1 PYTHONPATH=$PWD/$R:${PYTHONPATH:-}
L=$2; shift 2; mkdir -p $L
suites=("$@"); [ ${#suites[@]} -eq 0 ] && suites=(eval/golden_tests/g6_clone/ tests/ttnn/unit_tests/operations/g6_clone/)
for s in "${suites[@]}"; do
  tag=$(echo "$s" | tr '/' '_' | sed 's/_$//')
  scripts/run_safe_pytest.sh --run-all "$s" -p issue_path_plugin > $L/$ISSUE_VARIANT.$tag.log 2>&1
  echo "$ISSUE_VARIANT $s exit=$? :: $(grep -aE '^=+ .*(passed|failed|error).* =+$' $L/$ISSUE_VARIANT.$tag.log | tail -1) :: $(grep -a '^ISSUE_PLUGIN variant' $L/$ISSUE_VARIANT.$tag.log | tail -1)"
done
