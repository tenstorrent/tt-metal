#!/bin/bash
# Set jumbo MTU on the BF3 "tt"-side ports (the TT-RDMA external rails).
#
# The BH Rianta MAC is already configured for 4096B frames (eth_init.cpp:500/505,
# max_pkt_len=4204). The BlueField NIC, however, defaults to MTU 1500 and silently
# drops larger frames at PHY ingress. Raise it so single-frame jumbo (MAX_PKT~4080)
# works end-to-end. NOT persistent across reboot -> run this at boot (or from a
# systemd unit / rc.local) after `mst start`.
#
# Rail map (bh-bf3, dev1): enp193s0f0np0 = TT ext idx2 (pciconf1);
#                          enp193s0f1np1 = TT ext idx5 (pciconf1.1).
MTU="${1:-9000}"
for IF in enp193s0f0np0 enp193s0f1np1; do
  if sudo ip link set dev "$IF" mtu "$MTU"; then
    echo "$IF -> mtu $(cat /sys/class/net/$IF/mtu)"
  else
    echo "$IF -> FAILED to set mtu $MTU" >&2
  fi
done
