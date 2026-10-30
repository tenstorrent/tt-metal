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

#if defined(PROFILE_KERNEL) && defined(SPARSE_STEP_PROFILE) && !defined(PROFILE_STREAMING)

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

#else

#define SPARSE_PROF_ZONE(name, on) (void)(on)
#define SPARSE_PROF_SUM(name)
#define SPARSE_STAGE(var, name)
#define SPARSE_STAGE_END(var)

#endif
