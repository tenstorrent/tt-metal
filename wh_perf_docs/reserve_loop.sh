#!/bin/bash
# try to reserve a WH bgd card every 15 min for up to 10 h; on success print the machine and exit
for i in $(seq 1 40); do
  o=$(ssh -o BatchMode=yes -o ConnectTimeout=30 yyz-ird 'ird reserve --cluster tt_bgd --team bgd --docker-image llk --timeout 12:00:00 --no-shell wormhole_b0 --model x1 --num-pcie-chips 1 2>&1 | grep -v INFO | tail -4; echo ===; ird list 2>&1 | grep -v "INFO\|WARNING"' </dev/null 2>&1)
  if echo "$o" | sed -n '/===/,$p' | grep -q "bgd-lab"; then echo "RESERVED $(date -u)"; echo "$o" | sed -n '/===/,$p'; exit 0; fi
  echo "try $i failed $(date -u +%H:%M) $(echo "$o" | grep -o 'failed on node [a-z0-9-]*\|ERROR.*' | head -1)" >&2
  sleep 900
done
echo "GAVE UP $(date -u)"
