#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# TT-RDMA BH performance gate (companion to regression.sh). Measures the two axes that matter per phase:
#   - LATENCY: READ round-trip (READ_REQ -> READ_RESP), host-side, p50/p99 (us). Host-only, always runs.
#   - BANDWIDTH: RX WRITE processing rate (Gbps) under a saturating DPU sender (eSwitch-bypassed) so the
#     BH RISC data plane is the bottleneck. Skipped if the DPU is unreachable.
# Records a baseline on first run; thereafter FLAGS a regression (latency up > LAT_TOL%, bw down > BW_TOL%).
# Run after regression.sh on every phase change:  TT_METAL_HOME=<repo> ./perf.sh   (add `rebaseline` to reset)
#
# NB latency here is userspace<->userspace over a raw socket (syscall + promisc dominated), NOT the wire
# floor -- absolute value is high; what matters is the DELTA vs baseline when the on-core kernel changes.
set -uo pipefail
: "${TT_METAL_HOME:?set TT_METAL_HOME to the tt-metal-external-eth repo root}"
cd "$TT_METAL_HOME"
BIN=./build_Release/tests/tt_metal/tt_metal/tt_rdma_bh0
ALLOW=/home/alex/tenstorrent/tt-metal-external-eth/tt_metal/tt_rdma/bh0/tt_rdma_bf3_send
P0=enp193s0f0np0
DMAC=02:00:00:00:00:02
RKEY=0x00CAFE42
BASE=tt_metal/tt_rdma/bh0/perf_baseline.txt
LAT_TOL=25   # allow p50 to grow this % over baseline before flagging
BW_TOL=15    # allow bw to drop this % under baseline before flagging
JUMBO=4080; LANDED=$((32 + JUMBO))
D=$(mktemp -d); trap 'rm -rf "$D"' EXIT
SP="$(cd "$(dirname "$0")" && pwd)"
wait_up(){ timeout 80 sh -c "tail -f '$1' | grep -q -m1 'dispatch kernel up'" >/dev/null 2>&1; }

# ---- LATENCY: READ round-trip p50/p99 (host-only) ----
echo "== perf: READ round-trip latency =="
OUT=$D/lat.txt
# NB: rx_dispatch must ALWAYS clean-shutdown (stop flag + Finish) -- SIGTERM/kill leaves the eth core
# wedged ("core did not become active again" -> board reset). So size hold_s to cover the readlat run and
# `wait` for the host's own clean exit; never kill it.
timeout 40 "$BIN/bh1_rx_dispatch" 1 ext 8 1 1 4 1 >"$OUT" 2>&1 &   # read-target mode
DP=$!
p50=; p99=
if wait_up "$OUT"; then
  latline=$(sudo -n "$ALLOW" $P0 500 $DMAC 0x1af6 0x20 64 $RKEY 0 1 1 0 500 2>&1 | grep -m1 readlat)
  wait $DP 2>/dev/null   # clean shutdown (do NOT kill)
  echo "  $latline"
  p50=$(echo "$latline" | grep -oE 'p50=[0-9.]+' | cut -d= -f2)
  p99=$(echo "$latline" | grep -oE 'p99=[0-9.]+' | cut -d= -f2)
else echo "  FAIL: read-target kernel did not come up"; kill $DP 2>/dev/null; fi

# ---- BANDWIDTH: RX WRITE processing rate under a saturating DPU sender ----
echo "== perf: RX WRITE bandwidth (DPU-saturated) =="
bw=NA
export SSH_ASKPASS="$SP/askpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=:0
[ -f "$SP/askpass.sh" ] || { printf '#!/bin/bash\necho ubuntu\n' > "$SP/askpass.sh"; chmod +x "$SP/askpass.sh"; }
SSH="setsid -w ssh -o StrictHostKeyChecking=no -o ConnectTimeout=6 ubuntu@192.168.100.2"
if ping -c1 -W2 192.168.100.2 >/dev/null 2>&1 && $SSH 'test -x /tmp/tt_send' >/dev/null 2>&1; then
  OUT=$D/bw.txt
  timeout 26 "$BIN/bh1_rx_dispatch" 1 ext 14 1 1 2 1 >"$OUT" 2>&1 &   # WRITE -> tensix, jumbo, crc on
  DP=$!
  if wait_up "$OUT"; then
    # Rate-matched sender (single-sendto, ~near the crc-on ceiling) so the RX stays near-lossless and we
    # measure the sustained PROCESSING rate, not the overload-collapse salvage rate (8x64 blast collapses).
    $SSH "echo ubuntu | sudo -S /tmp/tt_send p0 300000000 $DMAC 0x1af6 0x10 $JUMBO $RKEY 0 1 1" >/dev/null 2>&1 &
    SPID=$!
    wait $DP 2>/dev/null
    $SSH "echo ubuntu | sudo -S pkill -x tt_send" >/dev/null 2>&1; kill $SPID 2>/dev/null
    # peak per-second delta of total * landed bytes -> Gbps; also grab the final bad (lapping) count.
    bw=$(awk -v lb=$LANDED '/total=/ && !/info/ { for(i=1;i<=NF;i++) if($i~/^total=/){split($i,a,"=");t=a[2]}
           if(p!=""){d=t-p; if(d>m)m=d} p=t } END{ printf "%.1f", m*lb*8/1e9 }' "$OUT")
    badc=$(grep "total=" "$OUT" | grep -v info | tail -1 | grep -oE 'bad=[0-9]+' | cut -d= -f2)
    echo "  sustained RX WRITE processing = ${bw} Gbps (jumbo ${JUMBO}B, crc on, bad=${badc:-?})"
  else echo "  FAIL: WRITE kernel did not come up"; kill $DP 2>/dev/null; fi
else
  echo "  SKIP: DPU unreachable or /tmp/tt_send missing (run deploy: gcc tt_rdma_bf3_send.c on the DPU)"
fi

# ---- baseline / regression gate ----
echo "======================================================"
if [ "${1:-}" = "rebaseline" ] || [ ! -f "$BASE" ]; then
  { echo "# TT-RDMA perf baseline (us round-trip, Gbps)"; echo "p50=${p50:-NA}"; echo "p99=${p99:-NA}"; echo "bw=${bw}"; } > "$BASE"
  echo "PERF BASELINE written: p50=${p50:-NA}us p99=${p99:-NA}us bw=${bw}Gbps -> $BASE"
  exit 0
fi
b_p50=$(grep '^p50=' "$BASE" | cut -d= -f2); b_bw=$(grep '^bw=' "$BASE" | cut -d= -f2)
echo "PERF: p50=${p50:-NA}us p99=${p99:-NA}us bw=${bw}Gbps   (baseline p50=${b_p50}us bw=${b_bw}Gbps)"
fail=0
if [ -n "${p50:-}" ] && [ "$b_p50" != NA ]; then
  awk -v a="$p50" -v b="$b_p50" -v t="$LAT_TOL" 'BEGIN{exit !(a > b*(1+t/100))}' && { echo "  REGRESSION: read p50 ${p50}us > baseline ${b_p50}us +${LAT_TOL}%"; fail=1; }
fi
if [ "$bw" != NA ] && [ "$b_bw" != NA ]; then
  awk -v a="$bw" -v b="$b_bw" -v t="$BW_TOL" 'BEGIN{exit !(a < b*(1-t/100))}' && { echo "  REGRESSION: RX WRITE bw ${bw}Gbps < baseline ${b_bw}Gbps -${BW_TOL}%"; fail=1; }
fi
[ "$fail" = 0 ] && echo "== PERF OK (within baseline tolerance) ==" || { echo "== PERF REGRESSION =="; exit 1; }
