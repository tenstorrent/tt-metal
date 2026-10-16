// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tools/profiler/kernel_profiler.hpp"

// Profiler-only Tensix hardware counters around the compute kernel's gate/up and down phases,
// MATH thread only. Compiled in only under the device profiler (PROFILE_KERNEL); otherwise every
// macro is empty and nothing here exists.
//
// Each expert's gate/up window (RE-GU) and down window (RE-DN) starts the FPU, TDMA_UNPACK and
// INSTRN_THREAD counter banks, stops them at the window end and adds the selected counters into a
// per-phase total. These are the registers tools/profiler/perf_counters.hpp drives when
// TT_METAL_PROFILE_PERF_COUNTERS is set; without it the firmware never touches them, so this
// kernel owns them for the launch. The FW matmul-throttle word (MEM_L1_ARC_FW_SCRATCH, odd =
// matmul_block runs its MM_THROTTLE_MAX MOP) is sampled once per K-block. At kernel end MATH emits
// the totals as 16 "RE-HW" DeviceTimestampedData records (layout at emit()).
// Opt-in (define RE_HWPROF as well): the counters cost MATH-binary space in every profiled build, and
// every RISC binary of a core shares the kernel config buffer, which the h7168 programs fill.
#if defined(PROFILE_KERNEL) && defined(TRISC_MATH) && defined(RE_HWPROF)
namespace re_hw {

constexpr uint32_t kPhaseGu = 0;
constexpr uint32_t kPhaseDn = 1;

// Counter slots per phase.
enum Slot : uint32_t {
    kRef = 0,     // FPU bank reference cycles (window length)
    kFpu,         // FPU busy
    kSfpu,        // SFPU busy
    kUnp0,        // unpacker 0 (SrcA) busy, thread 0
    kUnp1,        // unpacker 1 (SrcB) busy, thread 0
    kSrcAWr,      // SrcA write requests
    kSrcBWr,      // SrcB write requests
    kSrcReady,    // math source data ready
    kStall0,      // thread 0 (UNPACK) stall cycles
    kStall1,      // thread 1 (MATH) stall cycles
    kStall2,      // thread 2 (PACK) stall cycles
    kWaitAValid,  // waiting for SrcA valid
    kWaitBValid,  // waiting for SrcB valid
    kThrOn,       // K-blocks that sampled the throttle word odd
    kThrN,        // K-blocks sampled
    kMults,       // tile-mults issued
    kNumSlots
};

inline uint32_t acc[2][kNumSlots];
inline uint32_t cur_phase = 0;

#define RE_HW_PERF_CNT_CONTINUOUS_MODE 0
#define RE_HW_PERF_CNT_BANK_SELECT_SHIFT 8

inline __attribute__((always_inline)) volatile tt_reg_ptr uint32_t* reg(uint32_t addr) {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(addr);
}

inline __attribute__((noinline)) void start_bank(uint32_t cntl) {
    volatile tt_reg_ptr uint32_t* c = reg(cntl);
    c[0] = 0xFFFFFFFF;
    c[1] = RE_HW_PERF_CNT_CONTINUOUS_MODE;
    c[2] = 0;
    c[2] = 1;  // start (clears)
}

inline __attribute__((noinline)) void stop_bank(uint32_t cntl) {
    volatile tt_reg_ptr uint32_t* c = reg(cntl);
    c[2] = 0;
    c[2] = 2;  // stop (latches)
}

// Counter `sel` of a stopped bank; `ref` receives the bank's reference count.
inline __attribute__((noinline)) uint32_t read_counter(uint32_t cntl, uint32_t out, uint32_t sel, uint32_t* ref) {
    volatile tt_reg_ptr uint32_t* c = reg(cntl);
    volatile tt_reg_ptr uint32_t* o = reg(out);
    const uint32_t mode = (sel << RE_HW_PERF_CNT_BANK_SELECT_SHIFT) | RE_HW_PERF_CNT_CONTINUOUS_MODE;
    c[1] = mode;
    while (c[1] != mode) {
    }
    if (ref != nullptr) {
        *ref = o[0];
    }
    return o[1];
}

inline __attribute__((noinline)) void init() {
    // Kernel globals are not re-zeroed between launches of a cached binary.
    for (uint32_t p = 0; p < 2; ++p) {
        for (uint32_t i = 0; i < kNumSlots; ++i) {
            acc[p][i] = 0;
        }
    }
    cur_phase = 0;
}

inline __attribute__((noinline, noclone)) void begin(uint32_t phase) {
    cur_phase = phase;
    start_bank(RISCV_DEBUG_REG_PERF_CNT_FPU0);
    start_bank(RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0);
    start_bank(RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0);
}

inline __attribute__((noinline, noclone)) void end(uint32_t phase, uint32_t mults) {
    stop_bank(RISCV_DEBUG_REG_PERF_CNT_FPU0);
    stop_bank(RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0);
    stop_bank(RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0);
    uint32_t* a = acc[phase];
    uint32_t ref = 0;
    constexpr uint32_t F = RISCV_DEBUG_REG_PERF_CNT_FPU0;
    constexpr uint32_t FO = RISCV_DEBUG_REG_PERF_CNT_OUT_L_FPU;
    constexpr uint32_t U = RISCV_DEBUG_REG_PERF_CNT_TDMA_UNPACK0;
    constexpr uint32_t UO = RISCV_DEBUG_REG_PERF_CNT_OUT_L_TDMA_UNPACK;
    constexpr uint32_t I = RISCV_DEBUG_REG_PERF_CNT_INSTRN_THREAD0;
    constexpr uint32_t IO = RISCV_DEBUG_REG_PERF_CNT_OUT_L_INSTRN_THREAD;
    a[kFpu] += read_counter(F, FO, 0, &ref);
    a[kRef] += ref;
    a[kSfpu] += read_counter(F, FO, 1, nullptr);
    a[kSrcReady] += read_counter(U, UO, 0, nullptr);
    a[kSrcBWr] += read_counter(U, UO, 5, nullptr);
    a[kSrcAWr] += read_counter(U, UO, 6, nullptr);
    a[kUnp0] += read_counter(U, UO, 7, nullptr);
    a[kUnp1] += read_counter(U, UO, 8, nullptr);
    a[kStall0] += read_counter(I, IO, 24, nullptr);
    a[kStall1] += read_counter(I, IO, 25, nullptr);
    a[kStall2] += read_counter(I, IO, 26, nullptr);
    a[kWaitAValid] += read_counter(I, IO, 29, nullptr);
    a[kWaitBValid] += read_counter(I, IO, 30, nullptr);
    a[kMults] += mults;
}

inline __attribute__((always_inline)) void sample_throttle() {
    const uint32_t w = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(MEM_L1_ARC_FW_SCRATCH);
    acc[cur_phase][kThrOn] += (w & 1u);
    acc[cur_phase][kThrN] += 1u;
}

// One record per (phase, slot pair): bits [63:56] = phase * 8 + pair index, [55:28] = first slot,
// [27:0] = second slot (28 bits each: a launch is well under 2^28 cycles per core). One call site,
// so the profiler write is inlined once.
inline __attribute__((noinline)) void emit_one(uint64_t data) { DeviceTimestampedData("RE-HW", data); }

inline __attribute__((noinline)) void emit() {
    constexpr uint8_t kPairs[8][2] = {
        {kRef, kFpu},
        {kSfpu, kMults},
        {kUnp0, kUnp1},
        {kSrcAWr, kSrcBWr},
        {kWaitAValid, kWaitBValid},
        {kStall0, kStall1},
        {kStall2, kSrcReady},
        {kThrOn, kThrN}};
    constexpr uint64_t kMask = (1ull << 28) - 1;
#pragma GCC unroll 1
    for (uint32_t p = 0; p < 2; ++p) {
#pragma GCC unroll 1
        for (uint32_t i = 0; i < 8; ++i) {
            const uint64_t data = (static_cast<uint64_t>(p * 8 + i) << 56) |
                                  ((static_cast<uint64_t>(acc[p][kPairs[i][0]]) & kMask) << 28) |
                                  (static_cast<uint64_t>(acc[p][kPairs[i][1]]) & kMask);
            emit_one(data);
        }
    }
}

}  // namespace re_hw

#define RE_HW_INIT() re_hw::init()
#define RE_HW_BEGIN(phase) re_hw::begin(phase)
#define RE_HW_END(phase, mults) re_hw::end(phase, mults)
#define RE_HW_SAMPLE_THROTTLE() re_hw::sample_throttle()
#define RE_HW_EMIT() re_hw::emit()
#else
#define RE_HW_INIT()
#define RE_HW_BEGIN(phase)
#define RE_HW_END(phase, mults)
#define RE_HW_SAMPLE_THROTTLE()
#define RE_HW_EMIT()
#endif
