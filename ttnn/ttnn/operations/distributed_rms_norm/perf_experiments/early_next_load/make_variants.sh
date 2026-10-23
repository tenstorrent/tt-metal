#!/bin/bash
# Regenerate the per-variant kernel dirs from src/ (full copies: no cross-dir includes, so the JIT cache sees edits).
# v_base = the op's current reader (EARLY_MODE 0); v_hoist = 1; v_roothoist = 2; v_async = 3.
cd "$(dirname "$0")"
for pair in base:0 hoist:1 roothoist:2 async:3; do
  v=${pair%%:*}; m=${pair##*:}; d=v_$v
  rm -rf $d; mkdir -p $d
  cp src/distributed_rms_norm_compute.cpp src/distributed_rms_norm_port.cpp src/distributed_rms_norm_writer.cpp src/distributed_rms_norm_combine.hpp $d/
  { echo "#define EARLY_MODE $m"; cat src/reader_impl.hpp; } > $d/distributed_rms_norm_reader.cpp
done
