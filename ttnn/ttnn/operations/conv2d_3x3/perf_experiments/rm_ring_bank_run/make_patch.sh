#!/bin/bash
# integration.patch: op_base -> op_copy (zone-hash pad lines stripped), paths relative to the repo root.
cd "$(dirname "$0")"
tmp=$(mktemp -d)
mkdir -p $tmp/a $tmp/b
for d in a:op_base b:op_copy; do
  dst=${d%%:*}; src=${d#*:}
  cp $src/conv2d_3x3.py $src/conv2d_3x3_program_descriptor.py $src/__init__.py $tmp/$dst/
  mkdir -p $tmp/$dst/kernels && cp $src/kernels/* $tmp/$dst/kernels/
  sed -i '/^\/\/ zone-hash pad$/d' $tmp/$dst/kernels/*
done
(cd $tmp && diff -ruN a b) | sed -e 's#^--- a/#--- a/ttnn/ttnn/operations/conv2d_3x3/#' -e 's#^+++ b/#+++ b/ttnn/ttnn/operations/conv2d_3x3/#' > integration.patch
rm -rf $tmp
grep -c "^@@" integration.patch; diffstat integration.patch 2>/dev/null || grep "^+++" integration.patch
