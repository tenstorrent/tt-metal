// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DEBUG (dev branch only): windowed device-profiler zones for the sparse ring SDPA kernels.
//
// The per-RISC profiler buffer holds ~125 zones and silently drops the rest, so at 10 s / 15 s a whole op cannot be
// recorded. These zones only open inside a window: pass SPARSE_PROF_PASS, ring iteration SPARSE_PROF_ITER (-1 = every
// iteration), steps [SPARSE_PROF_SKIP, SPARSE_PROF_SKIP + SPARSE_PROF_STEPS) of that iteration (compute; the dataflow
// kernels window by pass / iteration only), and only while the buffer has room. Rerun with other windows to cover the
// op. SPARSE_PROF_SUM adds a sum zone (TT_METAL_PROFILER_SUM=1): one total per RISC for the whole op, no window; its
// total is written after the zones, so run sum builds with SPARSE_PROF_STEPS=0 to keep the buffer free for it.
#pragma once

#include <cstdint>

#ifndef SPARSE_PROF_PASS
#define SPARSE_PROF_PASS 0
#endif
#ifndef SPARSE_PROF_ITER
#define SPARSE_PROF_ITER 1
#endif
#ifndef SPARSE_PROF_SKIP
#define SPARSE_PROF_SKIP 0
#endif
#ifndef SPARSE_PROF_STEPS
#define SPARSE_PROF_STEPS 100
#endif

namespace sparse_prof {

inline bool pass_on = false;  // the current pass is the window's pass
inline bool iter_on = false;  // ... and the current ring iteration is the window's iteration
inline uint32_t step = 0;     // steps taken in the current iteration
inline bool step_on = false;  // the current step is inside the window (its stage zones open)

inline void begin_iter(uint32_t q_pass, uint32_t ring_iter) {
    pass_on = SPARSE_PROF_PASS < 0 || q_pass == static_cast<uint32_t>(SPARSE_PROF_PASS);  // -1 = every pass
    iter_on = pass_on && (SPARSE_PROF_ITER < 0 || ring_iter == static_cast<uint32_t>(SPARSE_PROF_ITER));
    step = 0;
    step_on = false;
}

// One compute step: true when it falls inside the window.
inline bool take_step() {
    const bool on = iter_on && step >= SPARSE_PROF_SKIP && step < SPARSE_PROF_SKIP + SPARSE_PROF_STEPS;
    ++step;
    step_on = on;
    return on;
}

}  // namespace sparse_prof

// Whole-op totals (SPARSE_PROF_TOTALS, from TT_SPARSE_PROF_TOTALS=1): every zone above instead adds its cycles to a
// per-RISC total for the whole op (no window, no zone records). SPARSE_TOTAL times any other block, SPARSE_COUNT adds
// a count, SPARSE_TOTALS_INIT zeroes the totals at kernel start, and SPARSE_TOTALS_FLUSH writes every nonzero total
// once at kernel end as a SPARSE_TOTALS TS_DATA record (index << 32 | value; dev/profile_summary.py decodes them with
// the same table). The names are the table below; a name not in it fails the build.
namespace sparse_prof {
constexpr const char* kTotalNames[] = {
    "KERNEL",      "RING_STEP_SUM", "RING_STEP",     "ST_S1_QK",      "ST_WAIT_QK",    "ST_S2_MAX",     "ST_S2B_TEST",
    "ST_S3_ALPHA", "ST_S4_RESCALE", "ST_S56_EXP_PV", "ST_S6B_ROWSUM", "ST_END_SYNC",   "C_K_WAIT",      "C_POP_KV",
    "C_PLAN_WAIT", "C_V_WAIT",      "C_ITER",        "RES_NORM",      "N_STEPS",       "N_SCORE_TILES", "N_RESCALE",
    "N_MEMBERS",   "RD_SYNC",       "RD_RES_Q",      "RD_RES_BITS",   "RD_RES_PLAN",   "RD_RES_PIECES", "RD_GROUP_RX",
    "RD_GROUP_TX", "RD_FETCH_K",    "RD_CHAIN_RX",   "RD_CHAIN_TX",   "RD_RES_POOLED", "WR_RES_OUT",
};
constexpr uint32_t kTotals = sizeof(kTotalNames) / sizeof(kTotalNames[0]);
constexpr bool total_name_eq(const char* a, const char* b) {
    while (*a != 0 && *a == *b) {
        ++a;
        ++b;
    }
    return *a == *b;
}
constexpr uint32_t total_index(const char* name) {
    for (uint32_t i = 0; i < kTotals; ++i) {
        if (total_name_eq(kTotalNames[i], name)) {
            return i;
        }
    }
    return kTotals;
}
}  // namespace sparse_prof

#if defined(PROFILE_KERNEL) && defined(SPARSE_STEP_PROFILE) && defined(SPARSE_PROF_TOTALS) && \
    !defined(PROFILE_STREAMING)

namespace sparse_prof {

inline uint32_t totals[kTotals];

inline __attribute__((always_inline)) uint32_t now() {
    return reinterpret_cast<volatile tt_reg_ptr uint32_t*>(
        RISCV_DEBUG_REG_WALL_CLOCK_L)[kernel_profiler::WALL_CLOCK_LOW_INDEX];
}

template <uint32_t idx>
struct Total {
    static_assert(idx < kTotals, "add the name to sparse_prof::kTotalNames");
    uint32_t t0;
    bool open = true;
    inline __attribute__((always_inline)) Total() : t0(now()) {}
    inline __attribute__((always_inline)) void end() {
        if (open) {
            totals[idx] += now() - t0;
            open = false;
        }
    }
    inline __attribute__((always_inline)) ~Total() { end(); }
};

}  // namespace sparse_prof

#define SPARSE_PROF_ZONE(name, on) \
    (void)(on);                    \
    sparse_prof::Total<sparse_prof::total_index(name)> sparse_prof_zone;
#define SPARSE_PROF_SUM(name) sparse_prof::Total<sparse_prof::total_index(name)> sparse_sum_zone;
#define SPARSE_STAGE(var, name) sparse_prof::Total<sparse_prof::total_index(name)> var;
#define SPARSE_STAGE_END(var) var.end();
#define SPARSE_TOTAL(var, name) sparse_prof::Total<sparse_prof::total_index(name)> var;
#define SPARSE_TOTAL_END(var) var.end();
#define SPARSE_COUNT(name, n)                                                                            \
    static_assert(sparse_prof::total_index(name) < sparse_prof::kTotals, "add the name to kTotalNames"); \
    sparse_prof::totals[sparse_prof::total_index(name)] += (n);
#define SPARSE_TOTALS_INIT()                                                \
    for (uint32_t total_i = 0; total_i < sparse_prof::kTotals; ++total_i) { \
        sparse_prof::totals[total_i] = 0;                                   \
    }
// One record type for every total (a single inlined timeStampedData, in a loop: one call site per RISC keeps the
// kernel config buffer within budget): data = index << 32 | value, the index into kTotalNames.
#define SPARSE_TOTALS_FLUSH()                                                                                      \
    {                                                                                                              \
        DO_PRAGMA(message(PROFILER_MSG_NAME("SPARSE_TOTALS")));                                                    \
        auto constexpr totals_hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME("SPARSE_TOTALS"));               \
        for (uint32_t total_i = 0; total_i < sparse_prof::kTotals; ++total_i) {                                    \
            if (sparse_prof::totals[total_i] != 0) {                                                               \
                kernel_profiler::timeStampedData<totals_hash>(                                                     \
                    (static_cast<uint64_t>(total_i) << 32) | static_cast<uint64_t>(sparse_prof::totals[total_i])); \
            }                                                                                                      \
        }                                                                                                          \
    }

#elif defined(PROFILE_KERNEL) && defined(SPARSE_STEP_PROFILE) && !defined(PROFILE_STREAMING)

namespace sparse_prof {

// kernel_profiler::profileScope that only opens when `on` (and the buffer has room).
template <uint32_t timer_id>
struct Scope {
    bool started = false;
    inline __attribute__((always_inline)) explicit Scope(bool on) {
        if (on && kernel_profiler::bufferHasRoom()) {
            kernel_profiler::stackSize += kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE;
            started = true;
            kernel_profiler::mark_time_at_index_inlined(kernel_profiler::wIndex, timer_id);
            kernel_profiler::wIndex += kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE;
        }
    }
    // Close the zone before the end of the scope (sequential stages in one scope); the destructor then does nothing.
    inline __attribute__((always_inline)) void end() {
        if (started) {
            kernel_profiler::mark_time_at_index_inlined(
                kernel_profiler::wIndex, kernel_profiler::get_const_id(timer_id, kernel_profiler::ZONE_END));
            kernel_profiler::wIndex += kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE;
            kernel_profiler::stackSize -= kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE;
            started = false;
        }
    }
    inline __attribute__((always_inline)) ~Scope() { end(); }
};

}  // namespace sparse_prof

#define SPARSE_PROF_ZONE(name, on)                                                         \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                                           \
    auto constexpr sparse_prof_hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    sparse_prof::Scope<sparse_prof_hash> sparse_prof_zone(on);

#define SPARSE_PROF_SUM(name)                                                             \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                                          \
    auto constexpr sparse_sum_hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    kernel_profiler::profileScopeAccumulate<sparse_sum_hash, 0> sparse_sum_zone;

// A stage of one compute step: opens when the step is inside the window (sparse_prof::step_on), closed explicitly
// by SPARSE_STAGE_END (or at the end of the scope), so consecutive stages need no extra braces.
#define SPARSE_STAGE(var, name)                                                      \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                                     \
    auto constexpr var##_hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    sparse_prof::Scope<var##_hash> var(sparse_prof::step_on);
#define SPARSE_STAGE_END(var) var.end();
#define SPARSE_TOTAL(var, name)
#define SPARSE_TOTAL_END(var)
#define SPARSE_COUNT(name, n)
#define SPARSE_TOTALS_INIT()
#define SPARSE_TOTALS_FLUSH()

#else

#define SPARSE_PROF_ZONE(name, on) (void)(on)
#define SPARSE_PROF_SUM(name)
#define SPARSE_STAGE(var, name)
#define SPARSE_STAGE_END(var)
#define SPARSE_TOTAL(var, name)
#define SPARSE_TOTAL_END(var)
#define SPARSE_COUNT(name, n)
#define SPARSE_TOTALS_INIT()
#define SPARSE_TOTALS_FLUSH()

#endif
