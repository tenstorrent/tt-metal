#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Deploy + build the TT-RDMA gateway DPA egress prototype (Arch-B, Phase A1) on a BlueField-3.
#
# This is the "doca_ttblast-on-DPA" experiment: the RoCE->TT-RDMA re-head EGRESS runs entirely on the BF3 DPA
# (FlexIO), so the Arm does ZERO per-frame work. It is a small patch (dpa_ttblast/dpa_ttblast.patch) over the
# stock FlexIO `packet_processor` sample: the DPA-side kernel gains a `tt_blast()` that builds one TT-RDMA-v1
# WRITE frame (ethertype 0x1AF6 + 32B hdr + CRC) and posts `count` sends on its ETH SQ (SQ-CQ paced), invoked
# host->DPA synchronously via flexio_process_call (a `__dpa_rpc__` entry). Measured ~178 Gbps jumbo on the p0
# uplink -> Blackhole (see [[tt-rdma-dpa-rehead-plan]]).
#
#   ./deploy_dpa_ttblast.sh            # provision toolchain + build -> ~/flexio_samples/build/... on the DPU
#   ./deploy_dpa_ttblast.sh --run      # ... then blast (500k x 4080B jumbo) on mlx5_0
#
# Env overrides: DPU=ubuntu@192.168.100.2  DPU_PASS=ubuntu  DEV=mlx5_0
#   TTDPA_COUNT/TTDPA_PLEN/TTDPA_DMAC/TTDPA_RKEY tune the blast (see the patch).
#
# NOTE (steering): the stock packet_processor forwards SQ output to the wire by matching dst MAC == SMAC, so
# the A1 blast sets the frame dst MAC = SMAC (02:42:7e:7f:eb:02). Phase C changes the TX rule to keep dst=BH.
set -euo pipefail

DPU="${DPU:-ubuntu@192.168.100.2}"
DPU_HOST="${DPU#*@}"
DPU_PASS="${DPU_PASS:-ubuntu}"
DEV="${DEV:-mlx5_0}"
HOST_TMFIFO="${HOST_TMFIFO:-tmfifo_net0}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=0
[ "${1:-}" = "--run" ] && RUN=1

ASK="$(mktemp)"; printf '#!/bin/sh\necho %s\n' "$DPU_PASS" > "$ASK"; chmod +x "$ASK"
trap 'rm -f "$ASK"' EXIT
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=8"
sshdpu() { SSH_ASKPASS="$ASK" SSH_ASKPASS_REQUIRE=force setsid -w $SSH "$DPU" "$@"; }
scpdpu() { SSH_ASKPASS="$ASK" SSH_ASKPASS_REQUIRE=force setsid -w \
             scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "$1" "$DPU:$2"; }

ip addr show "$HOST_TMFIFO" 2>/dev/null | grep -q "192.168.100.1" || \
  sudo ip addr add 192.168.100.1/30 dev "$HOST_TMFIFO" 2>/dev/null || true
ping -c1 -W2 "$DPU_HOST" >/dev/null 2>&1 || { echo "ERROR: DPU $DPU_HOST unreachable"; exit 1; }

# 1. Ensure meson+ninja on the (offline) DPU: stage aarch64 wheels from this host if missing.
if ! sshdpu 'export PATH=$HOME/.local/bin:$PATH; command -v meson >/dev/null && command -v ninja >/dev/null'; then
  echo "== staging meson+ninja wheels to the DPU =="
  WHL="$(mktemp -d)"
  python3 -m pip download --no-deps -d "$WHL" meson >/dev/null
  python3 -m pip download --no-deps --only-binary=:all: --platform manylinux2014_aarch64 -d "$WHL" ninja >/dev/null
  sshdpu 'mkdir -p /tmp/wheels'
  for w in "$WHL"/*.whl; do scpdpu "$w" /tmp/wheels/; done
  sshdpu 'python3 -m pip install --user --no-index --break-system-packages --find-links /tmp/wheels meson ninja'
  rm -rf "$WHL"
fi

echo "== stage stock FlexIO samples + apply the TT-RDMA DPA patch =="
scpdpu "$SELF_DIR/dpa_ttblast/dpa_ttblast.patch" /tmp/dpa_ttblast.patch
sshdpu '
  set -e
  rm -rf ~/flexio_samples
  cp -r /opt/mellanox/flexio/samples ~/flexio_samples
  cd ~/flexio_samples
  patch -p1 < /tmp/dpa_ttblast.patch
  echo "  patch applied"
'

echo "== build on the DPU (dpacc + meson + ninja) =="
sshdpu '
  set -e
  export PATH=$HOME/.local/bin:$PATH
  cd ~/flexio_samples
  ./build.sh 2>&1 | tail -3
  ls -la build/packet_processor/host/flexio_packet_processor && echo "  DPA TT-RDMA BLASTER BUILT"
'

if [ "$RUN" = 1 ]; then
  echo "== run: DPA blast 500k x 4080B jumbo on $DEV (dst=SMAC per stock steering) =="
  sshdpu "echo $DPU_PASS | sudo -S env TTDPA_COUNT=500000 TTDPA_PLEN=4080 TTDPA_DMAC=02:42:7e:7f:eb:02 \
            ~/flexio_samples/build/packet_processor/host/flexio_packet_processor $DEV 2>&1 | grep -iE 'returned|blast done'"
fi
