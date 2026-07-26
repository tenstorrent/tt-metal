#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Deploy + build (+ optionally run) the TT-RDMA DOCA HW-TX gateway sender on a BlueField-3.
# The DPU's /tmp is wiped on reboot, so the source lives here in the repo; this script scp's
# ttblast_sample.c to the DPU and builds it there against the DPU's DOCA sample sources
# (only the modified sample.c is ours; main.c/eth_common.c/eth_flow_common.c/common.c are stock).
#
#   ./deploy_doca_sender.sh            # scp + build -> /tmp/doca_ttblast on the DPU
#   ./deploy_doca_sender.sh --run      # ... then run it (mlx5_0 -> BH ext idx2, dst 02:...:02)
#
# Env overrides:
#   DPU=ubuntu@192.168.100.2  DPU_PASS=ubuntu  DEV=mlx5_0  DMAC=02:00:00:00:00:02
#   HOST_TMFIFO=tmfifo_net0   (host iface to the DPU; script ensures 192.168.100.1/30 on it)
# Prereqs (redo after a host/DPU reboot): BF3<->BH link trained (forced 200G) and MTU 9000 on the
# host PF + DPU-side p0/p1/pf0hpf/pf1hpf. See docs/tt-rdma-v1/tt-rdma-gateway-sender.md.
set -euo pipefail

DPU="${DPU:-ubuntu@192.168.100.2}"
DPU_HOST="${DPU#*@}"
DPU_PASS="${DPU_PASS:-ubuntu}"
DEV="${DEV:-mlx5_0}"
DMAC="${DMAC:-02:00:00:00:00:02}"
HOST_TMFIFO="${HOST_TMFIFO:-tmfifo_net0}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=0
[ "${1:-}" = "--run" ] && RUN=1

# Non-interactive ssh/scp password via SSH_ASKPASS (bench default ubuntu/ubuntu).
ASK="$(mktemp)"; printf '#!/bin/sh\necho %s\n' "$DPU_PASS" > "$ASK"; chmod +x "$ASK"
trap 'rm -f "$ASK"' EXIT
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8"
sshdpu() { SSH_ASKPASS="$ASK" SSH_ASKPASS_REQUIRE=force setsid -w $SSH "$DPU" "$@"; }
scpdpu() { SSH_ASKPASS="$ASK" SSH_ASKPASS_REQUIRE=force setsid -w \
             scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$1" "$DPU:$2"; }

echo "== ensure host can reach the DPU over $HOST_TMFIFO =="
ip addr show "$HOST_TMFIFO" 2>/dev/null | grep -q "192.168.100.1" || \
  sudo ip addr add 192.168.100.1/30 dev "$HOST_TMFIFO" 2>/dev/null || true
ping -c1 -W2 "$DPU_HOST" >/dev/null 2>&1 || { echo "  ERROR: DPU $DPU_HOST unreachable (train link / set tmfifo IP)"; exit 1; }
echo "  DPU reachable"

echo "== scp ttblast_sample.c -> DPU:/tmp/gw =="
sshdpu 'mkdir -p /tmp/gw'
scpdpu "$SELF_DIR/ttblast_sample.c" /tmp/gw/ttblast_sample.c

echo "== build /tmp/doca_ttblast on the DPU (gcc + pkg-config, no meson) =="
sshdpu '
  set -e
  export PKG_CONFIG_PATH=/opt/mellanox/doca/lib/aarch64-linux-gnu/pkgconfig
  S=/opt/mellanox/doca/samples/doca_eth/eth_txq_batch_send_ethernet_frames
  CM=/opt/mellanox/doca/samples/doca_eth
  gcc -O2 -w -I$S -I$CM -I/opt/mellanox/doca/samples $(pkg-config --cflags doca-eth doca-common doca-argp doca-flow) \
    /tmp/gw/ttblast_sample.c $S/eth_txq_batch_send_ethernet_frames_main.c \
    $CM/eth_common.c $CM/eth_flow_common.c /opt/mellanox/doca/samples/common.c \
    $(pkg-config --libs doca-eth doca-common doca-argp doca-flow) -lpthread -o /tmp/doca_ttblast
  ls -la /tmp/doca_ttblast && echo "  BUILT"
'

if [ "$RUN" = 1 ]; then
  echo "== run: sudo /tmp/doca_ttblast -d $DEV -m $DMAC =="
  sshdpu "echo $DPU_PASS | sudo -S /tmp/doca_ttblast -d $DEV -m $DMAC"
fi
