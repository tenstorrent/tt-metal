#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Deploy + build the TT-RDMA gateway BRIDGE (Arch-B, Phase B1) on a BlueField-3. The DPU /tmp is wiped on
# reboot, so the modified sample lives here in the repo; this scp's rdma_bridge_sample.c to the DPU and
# builds it against the DPU's stock DOCA rdma sample sources (only rdma_bridge_sample.c is ours; the stock
# rdma_write_immediate_responder_main.c + rdma_common.c are NVIDIA's).
#
#   ./deploy_rdma_bridge.sh          # scp + build -> /tmp/doca_ttbridge on the DPU
#
# Env overrides: DPU=ubuntu@192.168.100.2  DPU_PASS=ubuntu
set -euo pipefail

DPU="${DPU:-ubuntu@192.168.100.2}"
DPU_HOST="${DPU#*@}"
DPU_PASS="${DPU_PASS:-ubuntu}"
HOST_TMFIFO="${HOST_TMFIFO:-tmfifo_net0}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ASK="$(mktemp)"; printf '#!/bin/sh\necho %s\n' "$DPU_PASS" > "$ASK"; chmod +x "$ASK"
trap 'rm -f "$ASK"' EXIT
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8"
sshdpu() { SSH_ASKPASS="$ASK" SSH_ASKPASS_REQUIRE=force setsid -w $SSH "$DPU" "$@"; }
scpdpu() { SSH_ASKPASS="$ASK" SSH_ASKPASS_REQUIRE=force setsid -w \
             scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$1" "$DPU:$2"; }

ip addr show "$HOST_TMFIFO" 2>/dev/null | grep -q "192.168.100.1" || \
  sudo ip addr add 192.168.100.1/30 dev "$HOST_TMFIFO" 2>/dev/null || true
ping -c1 -W2 "$DPU_HOST" >/dev/null 2>&1 || { echo "ERROR: DPU $DPU_HOST unreachable"; exit 1; }

echo "== scp rdma_bridge_sample.c -> DPU:/tmp/gw =="
sshdpu 'mkdir -p /tmp/gw'
scpdpu "$SELF_DIR/rdma_bridge_sample.c" /tmp/gw/rdma_bridge_sample.c

echo "== build /tmp/doca_ttbridge on the DPU =="
sshdpu '
  set -e
  export PKG_CONFIG_PATH=/opt/mellanox/doca/lib/aarch64-linux-gnu/pkgconfig
  S=/opt/mellanox/doca/samples/doca_rdma
  R=$S/rdma_write_immediate_responder
  gcc -O2 -w -I$S -I/opt/mellanox/doca/samples $(pkg-config --cflags doca-rdma doca-common doca-argp) \
    /tmp/gw/rdma_bridge_sample.c $R/rdma_write_immediate_responder_main.c $S/rdma_common.c \
    /opt/mellanox/doca/samples/common.c \
    $(pkg-config --libs doca-rdma doca-common doca-argp) -lpthread -o /tmp/doca_ttbridge
  ls -la /tmp/doca_ttbridge && echo "  BRIDGE BUILT"
'
