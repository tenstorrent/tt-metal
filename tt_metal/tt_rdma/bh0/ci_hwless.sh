#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# HW-less CI entry point (Phase 0.4 of tt-rdma-production-plan.md). Runs on any PR without a bench:
# builds + runs the TT-RDMA-v1 wire-header golden-vector oracle (catches wire-format / CRC / struct
# drift before it ships to the chip kernel or the DOCA gateway codec). No hardware, tt-metal, or DOCA.
#
#   ./ci_hwless.sh          # from anywhere; finds the repo root relative to this script
#
# Wire this into CI as the fast PR gate. The full on-silicon suite (regression.sh) runs separately on a
# labeled bench runner after bringup.sh -- see tt-rdma-production-plan.md 0.4.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # -> repo root (tt-metal-external-eth)
OUT="$(mktemp)"; trap 'rm -f "$OUT"' EXIT

echo "== build HW-less golden test =="
g++ -std=c++17 -Wall -Werror -I"$ROOT" "$ROOT/tt_metal/tt_rdma/bh0/ci_golden_test.cpp" -o "$OUT"
echo "== run =="
"$OUT"
