#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# TT-RDMA BH regression harness (Phase 0.2 of tt-rdma-production-plan.md). Turns the bench claims into
# automated PASS/FAIL tests so nothing rests on "it worked once". Run on the host after bringup.sh:
#
#   TT_METAL_HOME=<repo> ./regression.sh
#
# Exit 0 iff all tests PASS. Each test asserts an observable invariant (byte-exactness, egress, loss).
# Uses small (256B) frames from the host userspace sender so the core correctness tests do not depend
# on jumbo / DPU-MTU state (throughput/jumbo/ceiling are separate perf tests, not correctness gates).
set -uo pipefail

: "${TT_METAL_HOME:?set TT_METAL_HOME to the tt-metal-external-eth repo root}"
cd "$TT_METAL_HOME"
BIN=./build_Release/tests/tt_metal/tt_metal/tt_rdma_bh0
ALLOW=/home/alex/tenstorrent/tt-metal-external-eth/tt_metal/tt_rdma/bh0/tt_rdma_bf3_send
P0=enp193s0f0np0                       # host tt-rail for BH ext idx2
DMAC=02:00:00:00:00:02                 # unicast -> BH RXQ2
RKEY=0x00CAFE42
D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT
PASS=0; FAIL=0
pass(){ echo "  PASS: $*"; PASS=$((PASS+1)); }
fail(){ echo "  FAIL: $*"; FAIL=$((FAIL+1)); }
rxb(){ ethtool -S "$1" 2>/dev/null | awk -F: '/rx_bytes_phy/{gsub(/ /,"",$2);print $2;exit}'; }
# wait for the RX-dispatch kernel to come up in a bg log
wait_up(){ timeout 80 sh -c "tail -f '$1' | grep -q -m1 'dispatch kernel up'" >/dev/null 2>&1; }

echo "== T1: golden wire-header self-test (no HW) =="
if "$BIN/bh1_send_probe" --selftest 2>&1 | grep -q "golden self-test PASS"; then pass "golden vectors"; else fail "golden vectors"; fi

echo "== T2: TX egress (ring drainer -> frames on the wire) =="
a=$(rxb $P0)
"$BIN/bh1_tx_ring" 1 ext a0:88:c2:11:dd:74 32 256 3 288 2 500 >/dev/null 2>&1
b=$(rxb $P0)
[ "$((b-a))" -gt 0 ] && pass "TX egress ($((b-a)) wire bytes)" || fail "TX egress (0 wire bytes)"

echo "== T3: RX inbound WRITE byte-exact (dispatch + MR + noc_async_write to Tensix) =="
OUT=$D/rx.txt
timeout 115 "$BIN/bh1_rx_dispatch" 1 ext 8 1 1 2 >"$OUT" 2>&1 &
DPID=$!
if wait_up "$OUT"; then
  sudo -n "$ALLOW" $P0 40 $DMAC 0x1af6 0x10 256 $RKEY 0 >/dev/null 2>&1
  wait $DPID
  land=$(grep "WRITE landing" "$OUT" | tail -1)
  wok=$(grep "total=" "$OUT" | grep -v info | tail -1 | grep -oE 'write_ok=[0-9]+' | cut -d= -f2)
  echo "$land" | grep -q "= 52575454 " && pass "RX WRITE landing byte-exact (TTWR)" || fail "RX WRITE landing ($land)"
  [ "${wok:-0}" -ge 30 ] && pass "RX WRITE dispatched (write_ok=$wok)" || fail "RX WRITE dispatched (write_ok=${wok:-0})"
else fail "RX kernel did not come up (T3)"; kill $DPID 2>/dev/null; fi

echo "== T4: RX streaming lossless at a safe rate (BUF_WRAP, bad==0) =="
OUT=$D/rx2.txt
timeout 115 "$BIN/bh1_rx_dispatch" 1 ext 8 1 1 0 >"$OUT" 2>&1 &
DPID=$!
if wait_up "$OUT"; then
  sudo -n "$ALLOW" $P0 20000 $DMAC 0x1af6 0x01 256 0 0 >/dev/null 2>&1   # SEND, small, single-thread (safe rate)
  wait $DPID
  bad=$(grep "total=" "$OUT" | grep -v info | tail -1 | grep -oE 'bad=[0-9]+' | cut -d= -f2)
  tot=$(grep "total=" "$OUT" | grep -v info | tail -1 | grep -oE 'total=[0-9]+' | cut -d= -f2)
  [ "${bad:-1}" = 0 ] && [ "${tot:-0}" -ge 1000 ] && pass "RX streaming lossless (total=$tot, bad=0)" || fail "RX streaming (total=${tot:-0}, bad=${bad:-?})"
else fail "RX kernel did not come up (T4)"; kill $DPID 2>/dev/null; fi

echo "======================================================"
echo "REGRESSION: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ] && { echo "== ALL GREEN =="; exit 0; } || { echo "== FAILURES =="; exit 1; }
