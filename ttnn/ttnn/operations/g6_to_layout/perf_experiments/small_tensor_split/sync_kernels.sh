#!/bin/bash
# Regenerate kernels_m<mode>/ (one kernel directory per compile-time split mode) from kernels/, and
# kernels_m<mode>_nz/ with the per-block profiler zones compiled out (the firmware's kernel zones,
# which DEVICE KERNEL DURATION is measured from, stay).
set -e
D="$(cd "$(dirname "$0")" && pwd)"
for m in 0 1 2 3; do
    for nz in "" _nz; do
        rm -rf "$D/kernels_m$m$nz"
        cp -r "$D/kernels" "$D/kernels_m$m$nz"
        sed -i "s/constexpr uint32_t G6_SPLIT_MODE = 0;/constexpr uint32_t G6_SPLIT_MODE = $m;/" "$D/kernels_m$m$nz/g6_split_mode.hpp"
        if [[ -n "$nz" ]]; then
            cat >> "$D/kernels_m$m$nz/g6_split_mode.hpp" <<'EOT'
// [small_tensor_split] no per-block zones in this directory (instrumentation-cost probe).
#undef DeviceZoneScopedN
#define DeviceZoneScopedN(name)
#undef DeviceZoneScopedSumN1
#define DeviceZoneScopedSumN1(name)
#undef DeviceZoneScopedSumN2
#define DeviceZoneScopedSumN2(name)
EOT
        fi
    done
done
