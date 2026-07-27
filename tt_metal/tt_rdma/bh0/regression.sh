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

echo "== T5: RX CRC-32 integrity -- corrupt-header frames dropped, not dispatched =="
OUT=$D/rx3.txt
timeout 115 "$BIN/bh1_rx_dispatch" 1 ext 8 1 1 2 1 >"$OUT" 2>&1 &   # crc_check=1 (arg7)
DPID=$!
if wait_up "$OUT"; then
  sudo -n "$ALLOW" $P0 40 $DMAC 0x1af6 0x10 256 $RKEY 0 1 1 1 >/dev/null 2>&1   # badcrc=1 (arg11)
  wait $DPID
  line=$(grep "total=" "$OUT" | grep -v info | tail -1)
  cerr=$(echo "$line" | grep -oE 'crc_err=[0-9]+' | cut -d= -f2)
  wok=$(echo "$line" | grep -oE 'write_ok=[0-9]+' | cut -d= -f2)
  [ "${cerr:-0}" -ge 30 ] && [ "${wok:-1}" = 0 ] && pass "RX CRC drop (crc_err=$cerr, write_ok=$wok)" || fail "RX CRC drop (crc_err=${cerr:-0}, write_ok=${wok:-?})"
else fail "RX kernel did not come up (T5)"; kill $DPID 2>/dev/null; fi

echo "== T6: RX SEND -> RxWqeRing publish (byte-exact slots + producer index) =="
OUT=$D/rx4.txt
timeout 115 "$BIN/bh1_rx_dispatch" 1 ext 8 1 1 3 1 >"$OUT" 2>&1 &   # noc_target=3 (SEND-ring)
DPID=$!
if wait_up "$OUT"; then
  sudo -n "$ALLOW" $P0 20 $DMAC 0x1af6 0x01 256 0 0 >/dev/null 2>&1   # 20 SEND frames, distinct seqs
  wait $DPID
  prod=$(grep "prod_idx =" "$OUT" | tail -1 | grep -oE '[0-9]+' | head -1)
  bx=$(grep "ring: " "$OUT" | tail -1)   # "  ring: N/M shown slots valid"
  n=$(echo "$bx" | grep -oE '[0-9]+/[0-9]+' | cut -d/ -f1); m=$(echo "$bx" | grep -oE '[0-9]+/[0-9]+' | cut -d/ -f2)
  [ "${prod:-0}" -ge 20 ] && [ "${m:-0}" -gt 0 ] && [ "${n:-0}" = "${m:-1}" ] \
    && pass "RX SEND-ring (prod_idx=$prod, $n/$m slots byte-exact)" \
    || fail "RX SEND-ring (prod_idx=${prod:-0}, ${n:-0}/${m:-0} byte-exact)"
else fail "RX kernel did not come up (T6)"; kill $DPID 2>/dev/null; fi

echo "== T7: RX READ_REQ -> READ_RESP round-trip (MR read + TX egress, byte-exact on the wire) =="
OUT=$D/rx5.txt; PCAP=$D/read.pcap
sudo -n tcpdump -i $P0 -w "$PCAP" -c 40 'ether proto 0x1af6' >/dev/null 2>&1 &
TCP=$!
sleep 1
timeout 115 "$BIN/bh1_rx_dispatch" 1 ext 8 1 1 4 1 >"$OUT" 2>&1 &   # noc_target=4 (READ-target)
DPID=$!
if wait_up "$OUT"; then
  sudo -n "$ALLOW" $P0 5 $DMAC 0x1af6 0x20 256 $RKEY 0 >/dev/null 2>&1   # 5 READ_REQ, request 256B
  sleep 2
  wait $DPID
  sudo -n pkill -x tcpdump 2>/dev/null; wait $TCP 2>/dev/null
  rr=$(grep "read_resp=" "$OUT" | grep -v info | tail -1 | grep -oE 'read_resp=[0-9]+' | cut -d= -f2)
  # a READ_RESP (op 0x21) frame carrying the 'READ' pattern (52 45 41 44) must be on the wire
  resp=$(sudo -n tcpdump -r "$PCAP" -nn -X 2>/dev/null | grep -c "5245 4144")
  [ "${rr:-0}" -ge 4 ] && [ "${resp:-0}" -ge 1 ] \
    && pass "RX READ round-trip (read_resp=$rr, $resp READ_RESP frames byte-exact)" \
    || fail "RX READ round-trip (read_resp=${rr:-0}, resp_frames=${resp:-0})"
else fail "RX kernel did not come up (T7)"; kill $DPID 2>/dev/null; sudo -n pkill -x tcpdump 2>/dev/null; fi

echo "== T8: RX ACK reception + cumulative-ACK accounting (monotonic watermark, stale ignored) =="
OUT=$D/rx6.txt
timeout 115 "$BIN/bh1_rx_dispatch" 1 ext 8 1 1 0 1 >"$OUT" 2>&1 &
DPID=$!
if wait_up "$OUT"; then
  sudo -n "$ALLOW" $P0 40 $DMAC 0x1af6 0x40 0 0 0 >/dev/null 2>&1   # ACK seq 1..40 -> watermark 40
  sleep 1
  sudo -n "$ALLOW" $P0 20 $DMAC 0x1af6 0x40 0 0 0 >/dev/null 2>&1   # ACK seq 1..20 STALE (<=40)
  wait $DPID
  line=$(grep "ack=" "$OUT" | grep -v info | tail -1)
  ack=$(echo "$line" | grep -oE 'ack=[0-9]+' | head -1 | cut -d= -f2)
  aseq=$(echo "$line" | grep -oE 'ack_seq=[0-9]+' | cut -d= -f2)
  [ "${aseq:-0}" = 40 ] && [ "${ack:-0}" -ge 50 ] \
    && pass "RX ACK cumulative (ack=$ack, watermark=$aseq held vs stale)" \
    || fail "RX ACK cumulative (ack=${ack:-0}, ack_seq=${aseq:-0}, want watermark=40)"
else fail "RX kernel did not come up (T8)"; kill $DPID 2>/dev/null; fi

echo "== T9: RX WRITE_IMM -- payload lands at MR AND raises an imm completion slot =="
OUT=$D/rx7.txt
timeout 115 "$BIN/bh1_rx_dispatch" 1 ext 8 1 1 5 1 >"$OUT" 2>&1 &   # noc_target=5 (write-imm)
DPID=$!
if wait_up "$OUT"; then
  sudo -n "$ALLOW" $P0 20 $DMAC 0x1af6 0x11 256 $RKEY 0 >/dev/null 2>&1   # 20 WRITE_IMM frames
  wait $DPID
  land=$(grep "WRITE_IMM payload landing" "$OUT" | tail -1)
  wimm=$(grep "wimm=" "$OUT" | grep -v info | tail -1 | grep -oE 'wimm=[0-9]+' | cut -d= -f2)
  ringok=$(grep "ring: " "$OUT" | tail -1 | grep -oE '[0-9]+/[0-9]+' | head -1)
  n=$(echo "$ringok" | cut -d/ -f1); m=$(echo "$ringok" | cut -d/ -f2)
  echo "$land" | grep -q "= 52575454 " && pass "WRITE_IMM payload byte-exact (TTWR)" || fail "WRITE_IMM landing ($land)"
  [ "${wimm:-0}" -ge 15 ] && [ "${m:-0}" -gt 0 ] && [ "${n:-0}" = "${m:-1}" ] \
    && pass "WRITE_IMM imm completion (wimm=$wimm, $n/$m slots imm-valid)" \
    || fail "WRITE_IMM completion (wimm=${wimm:-0}, ${n:-0}/${m:-0})"
else fail "RX kernel did not come up (T9)"; kill $DPID 2>/dev/null; fi

echo "== T10: RX MR access-control -- rkey_miss / rkey_access / rkey_bounds each dropped + counted =="
OUT=$D/rx8.txt
timeout 115 "$BIN/bh1_rx_dispatch" 1 ext 12 1 1 2 1 >"$OUT" 2>&1 &   # WRITE -> tensix (would land if authorized)
DPID=$!
if wait_up "$OUT"; then
  sudo -n "$ALLOW" $P0 15 $DMAC 0x1af6 0x10 256 0x00CAFE99 0 >/dev/null 2>&1; sleep 1    # miss: gen mismatch
  sudo -n "$ALLOW" $P0 15 $DMAC 0x1af6 0x10 256 0x01ABCD01 0 >/dev/null 2>&1; sleep 1    # access: READ-only MR[1]
  sudo -n "$ALLOW" $P0 15 $DMAC 0x1af6 0x10 256 0x00CAFE42 8192 >/dev/null 2>&1; sleep 1 # bounds: roff > mr_len
  wait $DPID
  line=$(grep "rkey_miss=" "$OUT" | grep -v info | tail -1)
  gv(){ echo "$line" | grep -oE "$1=[0-9]+" | cut -d= -f2; }
  miss=$(gv rkey_miss); acc=$(gv rkey_access); bnd=$(gv rkey_bounds); wok=$(gv write_ok)
  [ "${wok:-1}" = 0 ] && [ "${miss:-0}" -ge 10 ] && [ "${acc:-0}" -ge 10 ] && [ "${bnd:-0}" -ge 10 ] \
    && pass "MR access-control (miss=$miss access=$acc bounds=$bnd, write_ok=0)" \
    || fail "MR access-control (miss=${miss:-0} access=${acc:-0} bounds=${bnd:-0} write_ok=${wok:-?})"
else fail "RX kernel did not come up (T10)"; kill $DPID 2>/dev/null; fi

echo "======================================================"
echo "REGRESSION: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ] && { echo "== ALL GREEN =="; exit 0; } || { echo "== FAILURES =="; exit 1; }
