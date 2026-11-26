// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MEASUREMENT ONLY: wait-time probes of the flat expert's kernels: on with the named compile-time arg zz_flat_probes
// (the host's FLAT_ZZ_PROBES=1) and the device profiler; otherwise compiled out. A probe sums the cycles a statement
// waits (ZZ_WAIT) into a counter; ZZ_EMIT writes a counter as profiler data at the kernel's end.
#pragma once

#include <cstdint>

#include "api/compile_time_args.h"
#include "api/debug/kernel_profiler.hpp"

namespace zz {
inline constexpr bool on = get_named_compile_time_arg_val("zz_flat_probes") != 0;
// isolation (host FLAT_ISO; the data movement skipped, every count and semaphore kept - the results are garbage): 1
// gate/up weight reads, 2 down weight reads, 4 x DRAM reads, 8 x helper -> primary landing writes, 16 x primary ->
// snake head writes, 32 x snake forwarding, 64 h gate/up -> down chain heads, 128 h along the down chains, 256 y writes
inline constexpr uint32_t iso = get_named_compile_time_arg_val("zz_iso");
// preload emulation (host FLAT_PRE_X / FLAT_PRE_W; timing only, the data is garbage): the first pre_x x sub-blocks
// count as already in every gate/up core's x ring (not sent by the primaries, not forwarded), the first pre_w loads'
// gate/up and down weights as already landed (not read)
inline constexpr uint32_t pre_x = get_named_compile_time_arg_val("zz_pre_x");
inline constexpr uint32_t pre_w = get_named_compile_time_arg_val("zz_pre_w");
}  // namespace zz

#define ZZ_COUNTER(name) [[maybe_unused]] uint32_t name = 0
#define ZZ_WAIT(counter, ...)                            \
    do {                                                 \
        if constexpr (zz::on) {                          \
            const uint32_t zz_t0_ = get_timestamp_32b(); \
            __VA_ARGS__;                                 \
            (counter) += get_timestamp_32b() - zz_t0_;   \
        } else {                                         \
            __VA_ARGS__;                                 \
        }                                                \
    } while (0)
#define ZZ_EMIT(name, value)                    \
    ([&]() {                                    \
        if constexpr (zz::on) {                 \
            DeviceTimestampedData(name, value); \
        }                                       \
    }())
#define ZZ_NOW() (zz::on ? get_timestamp_32b() : 0u)
