#!/bin/bash
# usage: mk_unz.sh <dir>...: (re)build <dir>_unz = <dir> with every MaybeDeviceZoneScope line stripped (as
# ablation_r11/ab_unzoned.sh does: zones cost ~9% on ~10 us kernels).
cd "$(dirname "$0")"
for d in "$@"; do
    rm -rf ${d}_unz
    cp -r $d ${d}_unz
    rm -rf ${d}_unz/__pycache__
    sed -i '/MaybeDeviceZoneScope(/d' ${d}_unz/kernels/*.cpp ${d}_unz/kernels/*.hpp
done
