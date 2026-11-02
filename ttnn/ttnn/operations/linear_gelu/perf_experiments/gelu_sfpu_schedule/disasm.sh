#!/bin/bash
# Usage: disasm.sh <variant_num> [kernel_name]  -> prints the SFPU instruction stream of the newest trisc2 build
V=$1; K=${2:-bench_compute}
OD=/opt/tenstorrent/sfpi/compiler/bin/riscv-tt-elf-objdump
for d in $(ls -td /home/kgrujcic/.cache/tt-metal-cache/*/kernels/$K/*/); do
  if grep -rqs "GST_VARIANT $V\b\|GST_VARIANT=$V\b" $d 2>/dev/null && [ -f $d/trisc2/trisc2.elf ]; then
    echo "# $d"; $OD -d $d/trisc2/trisc2.elf | grep -E "\s(sfp|ttreplay|ttincrwc)" | awk '{$1="";$2=""; print}'; exit 0
  fi
done
