#!/usr/bin/env bash
# P2 - one clean end-to-end device sweep at the final commit.
#
# 37 cases across 9 groups, one pytest process at a time, never piped.
# Case 8 of the prefetcher suite (attention_decode_with_active_prefetch) is
# deselected: it is terminal FAILED by design (L3) and its TT_FATAL abort leaves
# the mesh un-drainable, so including it would cost a reset and poison the sweep.

set -uo pipefail
REPO=/proj_sw/user_dev/ctr-apbernal/tt-metal
LOGS="$REPO/tttv2_milestone_a_final_evidence/logs"
cd "$REPO" || exit 1

run() {   # run <nn_name> <expected> <pytest args...>
    local name="$1" expected="$2"; shift 2
    local log="$LOGS/${name}.log"
    echo "=== $(date -u +%H:%M:%SZ) $name (expect: $expected)"
    timeout --signal=TERM --kill-after=180 2700 \
        python -m pytest -v -rA --color=no -p no:cacheprovider "$@" > "$log" 2>&1
    local rc=$?
    echo "exit=$rc" >> "$log"
    local summary
    summary="$(grep -oE '[0-9]+ (passed|failed)[^=]*' "$log" | tail -1)"
    echo "    -> rc=$rc  ${summary:-NO SUMMARY LINE}"
    if [[ -n "$(pgrep -af 'python.*pytest' | grep -v grep || true)" ]]; then
        echo "    !! a pytest process is still alive after $name"
    fi
}

M=models/common/tests/modules
echo "### device matrix start $(date -u +%FT%TZ)  commit $(git rev-parse --short HEAD)"

run 20_embedding           "2 passed"  "$M/embedding/test_embedding_2d_wh_galaxy.py"
run 21_rope                "2 passed"  "$M/rope/test_rope_2d_wh_galaxy.py"
run 22_rmsnorm             "8 passed"  "$M/rmsnorm/test_rmsnorm_2d_wh_galaxy.py"
run 23_mlp                 "4 passed"  "$M/mlp/test_mlp_2d_wh_galaxy.py"
run 24_lm_head             "2 passed"  "$M/lm_head/test_lm_head_2d_wh_galaxy.py"
run 25_sampling_greedy     "1 passed"  "$M/sampling/test_sampling_2d_wh_galaxy.py"
run 26_sampling_stochastic "9 passed"  "$M/sampling/test_sampling_2d_wh_galaxy_stochastic.py"
run 27_attention           "2 passed"  "$M/attention/test_attention_2d_wh_galaxy.py"
run 28_prefetcher          "7 passed"  "$M/prefetcher/test_prefetcher_2d_wh_galaxy.py" \
                                       -k "not attention_decode_with_active_prefetch"

echo "### device matrix end $(date -u +%FT%TZ)"
ls /dev/tenstorrent | wc -l | sed 's/^/device nodes: /'
