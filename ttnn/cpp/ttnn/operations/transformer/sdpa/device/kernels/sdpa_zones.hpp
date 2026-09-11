// SPDX-License-Identifier: Apache-2.0
// Multi-slot accumulating device zones for the SDPA decomposition campaign (scratch branch only).
//
// Mechanism: SDPA_ZACC(idx) is an RAII scope that reads the 64-bit wall clock at entry and exit
// (RISCV_DEBUG_REG_WALL_CLOCK_L high/low, the same register kernel_profiler uses) and adds the
// difference to sums[idx], incrementing cnts[idx]. Nothing is written to L1 until SDPA_ZFLUSH(idx,
// "NAME") at kernel end, which emits two TS_DATA profiler records ("NAME" = summed cycles, "NAME_N"
// = occurrence count) into the per-RISC profiler buffer (2 marker slots each). Each RISC (BRISC,
// NCRISC, TRISC0/1/2) runs its own copy of the kernel, so every zone yields one record per RISC:
// the UNPACK (TRISC0) record measures unpack-side waits, PACK (TRISC2) the packer side, MATH
// (TRISC1) the math thread. SDPA_ZRAW(name) is a plain DeviceZoneScopedN (start/end pair in L1).
// The per-boundary cost (zone tax) is measured by analysis/zone_tax.py (T0.2).
#pragma once
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sdpa_zone_config.hpp"

#if defined(PROFILE_KERNEL) && (SDPA_ZONES)
#include "tools/profiler/kernel_profiler.hpp"

namespace sdpa_zones {
constexpr uint32_t N = 24;
inline uint64_t sums[N] = {};
inline uint32_t cnts[N] = {};

inline __attribute__((always_inline)) uint64_t now() {
    volatile tt_reg_ptr uint32_t* p = reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    return (static_cast<uint64_t>(p[1]) << 32) | p[0];
}

struct Acc {
    uint32_t idx;
    uint64_t t0;
    inline __attribute__((always_inline)) explicit Acc(uint32_t i) : idx(i), t0(now()) {}
    inline __attribute__((always_inline)) ~Acc() {
        sums[idx] += now() - t0;
        cnts[idx] += 1;
    }
};
}  // namespace sdpa_zones

#define SDPA_ZCAT2(a, b) a##b
#define SDPA_ZCAT(a, b) SDPA_ZCAT2(a, b)
#define SDPA_ZACC(idx) sdpa_zones::Acc SDPA_ZCAT(_sdpa_zacc_, __LINE__)((idx))
#define SDPA_ZRAW(name) DeviceZoneScopedN(name)
#define SDPA_ZFLUSH(idx, name)                                          \
    do {                                                                \
        DeviceTimestampedData(name, sdpa_zones::sums[idx]);             \
        DeviceTimestampedData(name "_N", (uint64_t)sdpa_zones::cnts[idx]); \
    } while (0)
#else
#define SDPA_ZACC(idx)
#define SDPA_ZRAW(name)
#define SDPA_ZFLUSH(idx, name)
#endif
