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
# A2/A4 add a per-frame re-head: A4 ZERO-COPY GATHER (TTDPA_ZC=1) writes only the 46B header per frame and
# posts a 2-seg WQE whose 2nd data seg HW-reads the payload from a source mkey (no CPU payload copy). A single
# DPA thread caps ~49-57 Gbps (gather, CRC skipped). MULTI-THREAD FAN-OUT (TTDPA_THREADS=N) spawns N host
# pthreads, each a blocking flexio_process_call driving its OWN SQ on its own DPA EU -> the RPCs run in
# PARALLEL. Aggregate scales 1.49->2.90->3.70->4.11->4.43 Mpps for N=1,2,3,4,6, PEAKING ~146 Gbps at 6 threads
# (N=8 flat, N=12 oversubscribed). 146G is the DPA 2-seg-gather send ceiling (per-thread source mkeys don't
# raise it) and is in the doca_ttblast line-rate class (143-198G), fully Arm-free.
#
#   ./deploy_dpa_ttblast.sh            # provision toolchain + build -> ~/flexio_samples/build/... on the DPU
#   ./deploy_dpa_ttblast.sh --run      # ... then single-thread blast (500k x 4080B jumbo) on mlx5_0
#   ./deploy_dpa_ttblast.sh --mt       # ... then 6-thread gather fan-out (~146G peak) on mlx5_0
#
# Env overrides: DPU=ubuntu@192.168.100.2  DPU_PASS=ubuntu  DEV=mlx5_0
#   TTDPA_COUNT   total frames (split evenly across threads)   TTDPA_PLEN payload bytes/frame
#   TTDPA_THREADS N host pthreads / DPA SQs (fan-out; 1=single) TTDPA_ZC=1 zero-copy gather (A4 re-head)
#   TTDPA_NOCRC=1 skip header CRC (pool doesn't check it; needed for line rate)
#   TTDPA_PERSRC=1 per-thread source buffer+mkey (diagnostic; no measurable gain)
#   TTDPA_HOSTSRC=1 A3.1: gather the payload from HOST memory (ibv_mr on the process PD) instead of DPA heap
#                  -- proves the PF DPA can egress the RoCE-landed buffer (the A3 re-head memory seam)
#   TTDPA_DMAC/TTDPA_RKEY tune the frame dst/rkey (see the patch).
#
# NOTE (steering): the TX->wire rule matches on dst MAC. Phase C points it at the BH dst MAC (TT_BH_DMAC
# 0x020000000002), so frames use the real dst=02:00:00:00:00:02 (BH RXQ2) and still egress.
set -euo pipefail

DPU="${DPU:-ubuntu@192.168.100.2}"
DPU_HOST="${DPU#*@}"
DPU_PASS="${DPU_PASS:-ubuntu}"
DEV="${DEV:-mlx5_0}"
HOST_TMFIFO="${HOST_TMFIFO:-tmfifo_net0}"
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=0
MT=0
[ "${1:-}" = "--run" ] && RUN=1
[ "${1:-}" = "--mt" ] && MT=1

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
  echo "== run: DPA blast 500k x 4080B jumbo on $DEV (dst=BH RXQ2) =="
  sshdpu "echo $DPU_PASS | sudo -S env TTDPA_COUNT=500000 TTDPA_PLEN=4080 TTDPA_DMAC=02:00:00:00:00:02 \
            ~/flexio_samples/build/packet_processor/host/flexio_packet_processor $DEV 2>&1 | grep -iE 'returned|blast done'"
fi

if [ "$MT" = 1 ]; then
  echo "== run: 6-thread DPA zero-copy-gather fan-out (3M x 4080B jumbo, ~146G peak) on $DEV =="
  sshdpu "echo $DPU_PASS | sudo -S env TTDPA_COUNT=3000000 TTDPA_PLEN=4080 TTDPA_THREADS=6 TTDPA_ZC=1 \
            TTDPA_NOCRC=1 TTDPA_DMAC=02:00:00:00:00:02 \
            ~/flexio_samples/build/packet_processor/host/flexio_packet_processor $DEV 2>&1 \
            | grep -iE 'MULTI-THREAD|Gbps|blast done'"
fi
