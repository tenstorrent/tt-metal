#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# A3.3b-3 gateway launcher (runs ON the DPU Arm). Launches the merged DPA re-head gateway DETACHED (setsid +
# line-buffered log) so an ssh/tmfifo bounce can't SIGHUP it. The gateway runs the RDMA-CM RoCE responder on
# the SF (mlx5_2) + the DPA drain on the PF (mlx5_0); it prints "RDMA CM listening on :18515" then waits ~120s
# for a connection (re-run this just before the requester). Poll /tmp/gw.log; /tmp/gw.done marks exit.
#
#   env: TTDPA_COUNT (default 100000)  TTDPA_PLEN (default 256)
# Prereqs after a DPU reboot (runtime IP state is wiped; OVS ovsbr1/2 persists):
#   DPU:  sudo ip addr add 10.99.0.1/24 dev enp3s0f0s0 && sudo ip link set enp3s0f0s0 up
#   host: sudo -n ip addr add 10.99.0.10/24 dev enp193s0f0np0   (then ping 10.99.0.1 -> ARP REACHABLE)
# Then on the host: ./tt_roce_client 10.99.0.1 18515 <count> <plen>
set -u
BIN=/home/ubuntu/flexio_samples/build/packet_processor/host/flexio_packet_processor
COUNT="${TTDPA_COUNT:-100000}"
PLEN="${TTDPA_PLEN:-256}"
DTHREADS="${TTDPA_DRAIN_THREADS:-1}"   # A5 fan-out: N parallel DPA drain threads (interleaved stripe)

sudo pkill -f flexio_packet_processor 2>/dev/null
sleep 0.5
sudo rm -f /tmp/gw.log /tmp/gw.done
sudo setsid bash -c "stdbuf -oL -eL env TTDPA_DOORBELL=1 TTDPA_ROCE=1 TTDPA_HOSTSRC=1 \
  TTDPA_COUNT=$COUNT TTDPA_PLEN=$PLEN TTDPA_NOCRC=1 TTDPA_DRAIN_THREADS=$DTHREADS $BIN mlx5_0 >/tmp/gw.log 2>&1; \
  echo GW_EXIT=\$? >>/tmp/gw.log; touch /tmp/gw.done" </dev/null >/dev/null 2>&1 &
echo "gateway launched detached -> /tmp/gw.log (count=$COUNT plen=$PLEN drain_threads=$DTHREADS)"
