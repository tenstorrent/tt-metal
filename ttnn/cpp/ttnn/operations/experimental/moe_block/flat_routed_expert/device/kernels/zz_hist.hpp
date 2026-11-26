// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// MEASUREMENT ONLY (zz_probes.hpp's switch): bytes landed by a reader over time, in 2^14-cycle bins (~12 us) from the
// kernel's start (start(): a ZZ-HT0 marker at that time, data = the tag), as 256-byte units, emitted at the end as
// ZZ-HB markers of 4 x 16-bit bins (bin 4 i in the low bits)
#pragma once
#include "zz_probes.hpp"

namespace zz {
template <uint32_t K>
struct Hist {
    static constexpr uint32_t bins = 128, shift = 14;
    uint16_t b[bins];
    uint32_t t0;
    void start(uint32_t tag) {
        if constexpr (on) {
            for (uint32_t i = 0; i < bins; ++i) {
                b[i] = 0;
            }
            if constexpr (K == 0) {
                DeviceTimestampedData("ZZ-HT0", tag);
            } else {
                DeviceTimestampedData("ZZ-YT0", tag);
            }
            t0 = get_timestamp_32b();
        }
    }
    void add(uint32_t bytes) {
        if constexpr (on) {
            uint32_t i = (get_timestamp_32b() - t0) >> shift;
            b[i < bins ? i : bins - 1] += bytes >> 8;
        }
    }
    void emit() {
        if constexpr (on) {
            for (uint32_t i = 0; i < bins; i += 4) {
                const uint64_t v =
                    uint64_t(b[i]) | uint64_t(b[i + 1]) << 16 | uint64_t(b[i + 2]) << 32 | uint64_t(b[i + 3]) << 48;
                if constexpr (K == 0) {
                    DeviceTimestampedData("ZZ-HB", v);
                } else {
                    DeviceTimestampedData("ZZ-YB", v);
                }
            }
        }
    }
};
inline Hist<0> hist;    // DRAM reads (weights, x)
inline Hist<1> hist_y;  // y writes
}  // namespace zz
