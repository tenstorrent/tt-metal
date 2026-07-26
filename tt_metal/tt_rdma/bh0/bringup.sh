#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# TT-RDMA BH<->BF3 rig cold-boot bring-up (Phase 0.1 of tt-rdma-production-plan.md).
# Idempotent: takes the rig from a fresh host/DPU reboot to "ready" with ZERO manual steps, and
# re-run is a no-op. Everything here is lost on reboot (hardware/OS state), so this is the single
# entry point to restore it. Run after any host or DPU reboot:
#
#   sudo -v ; ./bringup.sh            # bring up + verify
#   ./bringup.sh --verify-only        # just check current state (no changes)
#
# Env overrides:
#   RAILS="enp193s0f0np0 enp193s0f1np1"   host tt-rail netdevs (BF3 pciconf1 ports)
#   SPEED=200000  MTU=9000                 forced link speed / jumbo MTU (match the BH forced side)
#   DPU=ubuntu@192.168.100.2  DPU_PASS=ubuntu  HOST_TMFIFO=tmfifo_net0
#   DPU_PORTS="p0 p1 pf0hpf pf1hpf"        DPU-side ports needing MTU (uplinks + host-PF representors)
# Prereq assumed: the BH side is already flashed with the forced-200G topology-config FW.
set -uo pipefail

RAILS="${RAILS:-enp193s0f0np0 enp193s0f1np1}"
SPEED="${SPEED:-200000}"
MTU="${MTU:-9000}"
DPU="${DPU:-ubuntu@192.168.100.2}"
DPU_HOST="${DPU#*@}"
DPU_PASS="${DPU_PASS:-ubuntu}"
HOST_TMFIFO="${HOST_TMFIFO:-tmfifo_net0}"
DPU_PORTS="${DPU_PORTS:-p0 p1 pf0hpf pf1hpf}"
VERIFY_ONLY=0
[ "${1:-}" = "--verify-only" ] && VERIFY_ONLY=1

ok(){ echo "  [ok] $*"; }
warn(){ echo "  [!!] $*" >&2; }
step(){ echo "== $* =="; }

ASK="$(mktemp)"; printf '#!/bin/sh\necho %s\n' "$DPU_PASS" > "$ASK"; chmod +x "$ASK"; trap 'rm -f "$ASK"' EXIT
sshdpu(){ SSH_ASKPASS="$ASK" SSH_ASKPASS_REQUIRE=force setsid -w \
            ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8 "$DPU" "$@"; }

if [ "$VERIFY_ONLY" = 0 ]; then
  step "mst"
  ls /dev/mst/ 2>/dev/null | grep -q pciconf1 && ok "mst up" || { sudo mst start >/dev/null 2>&1 && ok "mst started" || warn "mst start failed"; }

  step "force ${SPEED} / AN off on host tt-rails (match BH forced side)"
  for r in $RAILS; do sudo ethtool -s "$r" autoneg off speed "$SPEED" 2>/dev/null && ok "$r forced $SPEED" || warn "$r ethtool failed"; done

  step "host<->DPU tmfifo IP"
  ip addr show "$HOST_TMFIFO" 2>/dev/null | grep -q 192.168.100.1 && ok "tmfifo IP set" || \
    { sudo ip addr add 192.168.100.1/30 dev "$HOST_TMFIFO" 2>/dev/null && ok "tmfifo IP added"; sudo ip link set "$HOST_TMFIFO" up 2>/dev/null; }

  step "DPU-side MTU ${MTU} (${DPU_PORTS})"
  if ping -c1 -W2 "$DPU_HOST" >/dev/null 2>&1; then
    sshdpu "for i in $DPU_PORTS; do echo $DPU_PASS | sudo -S ip link set dev \$i mtu $MTU 2>/dev/null; done" >/dev/null 2>&1 && ok "DPU ports -> $MTU" || warn "DPU MTU set failed"
  else warn "DPU unreachable over $HOST_TMFIFO (skipping DPU MTU)"; fi

  step "host PF MTU ${MTU} (re-apply AFTER DPU uplink change -- it flaps host to 1500)"
  for r in $RAILS; do sudo ip link set dev "$r" mtu "$MTU" 2>/dev/null && ok "$r mtu $MTU" || warn "$r mtu failed"; done
fi

# ---- Verify (the Phase 0.1 gate) ----
# Gate = MTU correct + DPU reachable + BH sees BOTH external rails (the authoritative link-up proof:
# base FW only tags a rail EXTERNAL/PORT_UP when its BF3 link is trained). mlxlink State is
# informational only -- it reads empty for a beat right after `mst start`, so it must not gate.
step "VERIFY"
rc=0
for d in /dev/mst/mt41692_pciconf1 /dev/mst/mt41692_pciconf1.1; do
  st=$(sudo mlxlink -d "$d" 2>/dev/null | grep -aE '^State' | tr -d '\033' | sed 's/\[[0-9;]*m//g' | awk -F: '{gsub(/ /,"",$2);print $2}')
  [ "$st" = "Active" ] && ok "$(basename $d) Active" || echo "  [info] $(basename $d) State=${st:-unknown} (informational)"
done
for r in $RAILS; do m=$(cat /sys/class/net/$r/mtu 2>/dev/null); [ "$m" = "$MTU" ] && ok "$r mtu=$MTU" || { warn "$r mtu=$m (want $MTU)"; rc=1; }; done
ping -c1 -W2 "$DPU_HOST" >/dev/null 2>&1 && ok "DPU reachable" || { warn "DPU unreachable"; rc=1; }
nrails=$(echo $RAILS | wc -w)
if [ -n "${TT_METAL_HOME:-}" ] && [ -x "$TT_METAL_HOME/build_Release/tests/tt_metal/tt_metal/tt_rdma_bh0/bh1_send_probe" ]; then
  n=$(timeout 90 "$TT_METAL_HOME/build_Release/tests/tt_metal/tt_metal/tt_rdma_bh0/bh1_send_probe" --list 1 2>/dev/null | grep -c EXTERNAL)
  if [ "${n:-0}" -ge "$nrails" ]; then ok "BH external rails present: $n/$nrails (links trained)";
  elif [ "${n:-0}" -ge 1 ]; then warn "BH sees $n/$nrails external rails (a link is down -- retrain)"; rc=1;
  else warn "BH sees 0 external rails (links down -- retrain)"; rc=1; fi
else warn "set TT_METAL_HOME to verify BH external rails (the authoritative link gate)"; rc=1; fi

[ "$rc" = 0 ] && echo "== BRING-UP OK ==" || echo "== BRING-UP INCOMPLETE (see [!!]) =="
exit $rc
