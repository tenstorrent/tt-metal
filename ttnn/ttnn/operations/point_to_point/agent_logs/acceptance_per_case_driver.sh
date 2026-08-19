#!/bin/bash
# One pytest process (+ forced device reset) per acceptance case.
# On this board a multi-packet / multi-hop fabric transfer leaves the ethernet
# wedged, so the PROCESS often aborts (SIGABRT, exit 134) during device teardown
# AFTER the test itself has already passed. Classify from pytest's own verdict,
# not from the process exit code.
IDS="$1"; START="$2"; COUNT="$3"; OUT="$4"
cd /localdev/wransom/2026_08_19/0157_wransom_ccl_help_allreduce_eval/clones/point_to_point_run1/tt-metal
i=0
while IFS= read -r nid; do
  i=$((i+1))
  [ "$i" -le "$START" ] && continue
  [ "$i" -gt $((START+COUNT)) ] && break
  touch /tmp/tt-device.dirty
  timeout 200 scripts/run_safe_pytest.sh --dev "$nid" > /tmp/p2p_case.log 2>&1
  if grep -qE "^PASSED tests/" /tmp/p2p_case.log; then
    if grep -qE "Timed out while waiting for active ethernet" /tmp/p2p_case.log; then
      echo "$i,PASS_TEARDOWN_WEDGE" >> "$OUT"
    else
      echo "$i,PASS" >> "$OUT"
    fi
  elif grep -qE "^FAILED tests/" /tmp/p2p_case.log; then
    r=$(grep -oE "Max diff[^,]*|assert_with_pcc[^\"]{0,60}|AssertionError[^\"]{0,60}|ValueError[^\"]{0,60}|RuntimeError[^\"]{0,60}|invalid address alignment in NOC transaction" /tmp/p2p_case.log | head -1 | tr ',' ';')
    echo "$i,FAIL_TEST:${r:-unknown}" >> "$OUT"; cp /tmp/p2p_case.log "/tmp/p2p_fail_$i.log"
  elif grep -qE "ERROR at setup of" /tmp/p2p_case.log; then
    echo "$i,ERROR_SETUP" >> "$OUT"; cp /tmp/p2p_case.log "/tmp/p2p_fail_$i.log"
  else
    echo "$i,NORESULT" >> "$OUT"; cp /tmp/p2p_case.log "/tmp/p2p_fail_$i.log"
  fi
done < "$IDS"
