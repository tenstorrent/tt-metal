#!/bin/bash
# cmp.sh <base_label> <cand_label>
cd /home/malimpic/localdev/2026_10_30/1730_malimpic_nuke-ccl-ops/clones/distributed_rms_norm_run1/tt-metal
source python_env/bin/activate
for f in /tmp/esw/res/$1/*.json; do b=$(basename $f); c=/tmp/esw/res/$2/$b; [ -f "$c" ] || continue
  r=$(python -m eval.ccl_perf compare $f $c 2>&1 | tail -3 | awk '/fastest/{fa=$2" -> "$4" ("$7")"} /slowest/{sl=$2" -> "$4" ("$7")"} /verdict/{v=$0} END{print fa" | "sl" | "v}')
  printf "%-22s %s\n" "$(echo $b | sed -E 's/-alignment.*//;s/sweep-//')" "$r"; done
