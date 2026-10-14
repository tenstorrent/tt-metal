#!/bin/bash
# disassemble the newest build of THIS dir's reader to stdout
cd "$(dirname "$0")/../../../../../.."
e=$(ls -t $(grep -l "reorder_cost" built/*/kernels/tilize_reader/*/kernel_includes.hpp | sed 's|kernel_includes.hpp|ncrisc/ncrisc.elf|') 2>/dev/null | head -1)
echo "# $e" >&2
runtime/sfpi/compiler/bin/riscv-tt-elf-objdump -d $e
