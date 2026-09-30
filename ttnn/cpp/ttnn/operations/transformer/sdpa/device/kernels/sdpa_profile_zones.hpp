// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Device-profiler zones for the streaming SDPA compute and the ring-joint reader/writer kernels.

#pragma once

#include <cstdint>

#include "tools/profiler/kernel_profiler.hpp"

// Kernel profiler zones (device-profiler builds); the ring-joint factory sets it from TT_SDPA_PROFILE_ZONES.
// The joint and exp-ring factories do not emit it, so their zones stay off.
#ifndef SDPA_PROFILE_ZONES
#define SDPA_PROFILE_ZONES 0
#endif
// Profiling window (TT_SDPA_PROFILE_WINDOW="iter,q,klo,khi"): ring iteration, Q chunk index within the core's
// range and K chunks [klo, khi). -1 leaves that bound open; all -1 is no window.
#ifndef SDPA_PROFILE_ITER
#define SDPA_PROFILE_ITER -1
#endif
#ifndef SDPA_PROFILE_QCHUNK
#define SDPA_PROFILE_QCHUNK -1
#endif
#ifndef SDPA_PROFILE_KCHUNK_LO
#define SDPA_PROFILE_KCHUNK_LO -1
#endif
#ifndef SDPA_PROFILE_KCHUNK_HI
#define SDPA_PROFILE_KCHUNK_HI -1
#endif
#if SDPA_PROFILE_ITER >= 0 || SDPA_PROFILE_QCHUNK >= 0 || SDPA_PROFILE_KCHUNK_LO >= 0 || SDPA_PROFILE_KCHUNK_HI >= 0
#define SDPA_PROFILE_HAS_WINDOW 1
#else
#define SDPA_PROFILE_HAS_WINDOW 0
#endif
// TT_SDPA_PROFILE_FINE (with a window): exp and pack zones in every fixed-offset QK^T subblock, PACK thread only.
// 32 extra zone pairs per K chunk, so the window should span one chunk.
#ifndef SDPA_PROFILE_FINE
#define SDPA_PROFILE_FINE 0
#endif
#if SDPA_PROFILE_FINE == 1 && SDPA_PROFILE_HAS_WINDOW && defined(TRISC_PACK) && defined(PROFILE_KERNEL) && \
    !defined(PROFILE_STREAMING)
#define SDPA_PROFILE_FINE_ZONES 1
#include "ckernel.h"
#else
#define SDPA_PROFILE_FINE_ZONES 0
#endif

namespace sdpa_profile {
constexpr int32_t iter = SDPA_PROFILE_ITER;
constexpr int32_t q_chunk = SDPA_PROFILE_QCHUNK;
constexpr int32_t k_chunk_lo = SDPA_PROFILE_KCHUNK_LO;
constexpr int32_t k_chunk_hi = SDPA_PROFILE_KCHUNK_HI;
constexpr bool zones = SDPA_PROFILE_ZONES == 1;

inline __attribute__((always_inline)) bool iter_hit(uint32_t ring_iter) {
    return iter < 0 || ring_iter == static_cast<uint32_t>(iter);
}
inline __attribute__((always_inline)) bool iter_q_hit(uint32_t ring_iter, uint32_t q_index) {
    return iter_hit(ring_iter) && (q_chunk < 0 || q_index == static_cast<uint32_t>(q_chunk));
}
inline __attribute__((always_inline)) bool window_hit(uint32_t ring_iter, uint32_t q_index, uint32_t k_chunk) {
    return iter_q_hit(ring_iter, q_index) && (k_chunk_lo < 0 || k_chunk >= static_cast<uint32_t>(k_chunk_lo)) &&
           (k_chunk_hi < 0 || k_chunk < static_cast<uint32_t>(k_chunk_hi));
}

// Zones inside the sub_exp / normalize helpers: off with a window, whose budget goes to the per-phase zones
// (and whose code must fit the kernel config buffer).
template <bool profiling_enabled>
constexpr bool helper_zones = profiling_enabled && SDPA_PROFILE_HAS_WINDOW == 0;

// Fine zones start and end on an idle Tensix pipe, so they time execution rather than instruction issue.
inline __attribute__((always_inline)) void fine_sync([[maybe_unused]] bool on) {
#if SDPA_PROFILE_FINE_ZONES
    if (on) {
        ckernel::tensix_sync();
    }
#endif
}
}  // namespace sdpa_profile

// Template-driven profiling: MaybeDeviceZoneScopedN(ENABLED, name)
// When ENABLED=true: RAII profileScope writes timestamps (same as DeviceZoneScopedN)
// When ENABLED=false: empty struct, zero overhead (compiler eliminates entirely)
// MaybeDeviceZoneScopedNIf(ENABLED, name, on): also needs `on` when a window is set, else MaybeDeviceZoneScopedN.
// MaybeDeviceZoneScopedNWindow(ENABLED, name, on): a gated zone that exists only while a window is set.
#if defined(PROFILE_STREAMING)
// Not supported by the streaming profiler: these template-gated zones use the DRAM profiler's hash ids.
#define MaybeDeviceZoneScopedN(ENABLED, name)
#define MaybeDeviceZoneScopedNIf(ENABLED, name, on) (void(sizeof(on)))
#elif defined(PROFILE_KERNEL)
template <bool Enabled, uint32_t timer_id>
struct MaybeProfileScope {
    inline __attribute__((always_inline)) MaybeProfileScope() {}
    inline __attribute__((always_inline)) ~MaybeProfileScope() {}
};
template <uint32_t timer_id>
struct MaybeProfileScope<true, timer_id> : kernel_profiler::profileScope<timer_id> {};

#define MaybeDeviceZoneScopedN(ENABLED, name)                                  \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    MaybeProfileScope<ENABLED, hash> zone;

#if SDPA_PROFILE_HAS_WINDOW
template <bool Enabled, uint32_t timer_id>
struct MaybeProfileScopeIf {
    inline __attribute__((always_inline)) explicit MaybeProfileScopeIf(bool) {}
    inline __attribute__((always_inline)) ~MaybeProfileScopeIf() {}
};
// kernel_profiler::profileScope with a runtime gate; same buffer bookkeeping (drops zones once the buffer is full).
template <uint32_t timer_id>
struct MaybeProfileScopeIf<true, timer_id> {
    bool start_marked = false;
    inline __attribute__((always_inline)) explicit MaybeProfileScopeIf(bool on) {
#if defined(ARCH_QUASAR)
        if (on && kernel_profiler::bufferHasRoom(2 * kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE - 1)) {
#else
        if (on && kernel_profiler::bufferHasRoom()) {
#endif
            kernel_profiler::stackSize += kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE;
            start_marked = true;
            kernel_profiler::mark_time_at_index_inlined(kernel_profiler::wIndex, timer_id);
            kernel_profiler::wIndex += kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE;
        }
    }
    inline __attribute__((always_inline)) ~MaybeProfileScopeIf() {
        if (start_marked) {
            kernel_profiler::mark_time_at_index_inlined(
                kernel_profiler::wIndex, kernel_profiler::get_const_id(timer_id, kernel_profiler::ZONE_END));
            kernel_profiler::wIndex += kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE;
            kernel_profiler::stackSize -= kernel_profiler::PROFILER_L1_MARKER_UINT32_SIZE;
        }
    }
};

#define MaybeDeviceZoneScopedNIf(ENABLED, name, on)                            \
    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));                               \
    auto constexpr hash = kernel_profiler::Hash16_CT(PROFILER_MSG_NAME(name)); \
    MaybeProfileScopeIf<ENABLED, hash> zone(on);
#else
#define MaybeDeviceZoneScopedNIf(ENABLED, name, on) MaybeDeviceZoneScopedN(ENABLED, name)
#endif
#else
#define MaybeDeviceZoneScopedN(ENABLED, name)
#define MaybeDeviceZoneScopedNIf(ENABLED, name, on) (void(sizeof(on)))
#endif
// Not even an empty scope object without a window: those perturb register allocation in profiler builds.
#if SDPA_PROFILE_HAS_WINDOW
#define MaybeDeviceZoneScopedNWindow(ENABLED, name, on) MaybeDeviceZoneScopedNIf(ENABLED, name, on)
#else
#define MaybeDeviceZoneScopedNWindow(ENABLED, name, on) (void(sizeof(on)))
#endif
#if SDPA_PROFILE_FINE_ZONES
#define MaybeDeviceZoneScopedNFine(ENABLED, name, on) MaybeDeviceZoneScopedNIf(ENABLED, name, on)
#else
#define MaybeDeviceZoneScopedNFine(ENABLED, name, on) (void(sizeof(on)))
#endif
