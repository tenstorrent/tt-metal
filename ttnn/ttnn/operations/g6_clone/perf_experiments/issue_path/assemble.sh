#!/usr/bin/env bash
# Builds the kernels_<variant>/ folders the lane loads from src/ (reader/writer variants) and kernels_base/
# (a verbatim copy of the op's kernels at the experiment's start). Re-run after editing src/.
set -euo pipefail
cd "$(dirname "$0")"
mk() {  # mk <variant> <reader src|base> <writer src|base>
  local d=kernels_$1; rm -rf $d; mkdir -p $d
  cp kernels_base/g6_clone_page_pieces.hpp kernels_base/g6_clone_zones.hpp src/g6_clone_issue.hpp $d/
  if [ "$2" = base ]; then cp kernels_base/g6_clone_reader.cpp $d/; else cp src/$2 $d/g6_clone_reader.cpp; fi
  if [ "$3" = base ]; then cp kernels_base/g6_clone_writer.cpp $d/; else cp src/$3 $d/g6_clone_writer.cpp; fi
}
mk rdlean reader_lean.cpp base
mk wrarm base writer_armed.cpp
mk both reader_lean.cpp writer_armed.cpp
mk both2 reader_lean2.cpp writer_armed.cpp
