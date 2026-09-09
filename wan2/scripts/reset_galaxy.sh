#!/usr/bin/env bash
# Reset all four Blackhole Galaxy hosts.
# This kills any job in flight on the whole cluster. Check it is idle first.

set -u

for ip in 10.81.14.13 10.81.14.14 10.81.14.15 10.81.14.16; do
  ssh -n -o BatchMode=yes ttuser@"$ip" 'tt-smi -glx_reset_auto' \
    >"/tmp/reset_$ip.log" 2>&1 &
done
wait

for ip in 10.81.14.13 10.81.14.14 10.81.14.15 10.81.14.16; do
  echo "--- $ip ---"
  tail -n 3 "/tmp/reset_$ip.log"
done

# Give the fabric time to retrain its inter-chip links before launching.
# 15s is sometimes not enough on the quad; 30s is safer. If a run still fails
# at mesh mapping ("no valid multi-mesh mapping" / "Controller likely failed"),
# run this reset once more before launching.
sleep 30
