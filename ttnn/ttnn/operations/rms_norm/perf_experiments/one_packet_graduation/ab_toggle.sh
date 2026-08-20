#!/usr/bin/env bash
# A/B toggle for the one_packet graduation, so the two arms are measured in ONE
# tree (a sibling agent is concurrently editing the host + compute kernel, so
# measuring "before my edit" and "after my edit" at different wall-clock times
# would attribute their change to mine).
#   ab_toggle.sh anylen     -> restore the any-length page-id form  (baseline)
#   ab_toggle.sh onepacket  -> restore the compile-time-bounded form (candidate)
set -euo pipefail
d="$(cd "$(dirname "$0")" && pwd)"; k="$d/../../kernels"
case "${1:?anylen|onepacket}" in
  anylen|onepacket) a="$1";;
  *) echo "usage: $0 anylen|onepacket" >&2; exit 2;;
esac
cp "$d/rms_norm_reader.$a.cpp" "$k/rms_norm_reader.cpp"
cp "$d/rms_norm_writer.$a.cpp" "$k/rms_norm_writer.cpp"
echo "kernels := $a"
