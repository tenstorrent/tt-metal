// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>

// Platform-specific headers for pause/yield intrinsics.
//
// x86_64:  immintrin.h provides _mm_pause() on GCC, Clang, and MSVC.
//
// AArch64: Clang provides __yield() via arm_acle.h (ACLE Section 7.4).
//          MSVC provides __yield() via intrin.h.
//          GCC does NOT provide __yield() in its aarch64 arm_acle.h --
//          hint intrinsics were never implemented for the aarch64 backend.
//          See: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105416
//          For GCC aarch64, we fall back to inline assembly below.
//
// RISC-V:  __builtin_riscv_pause() is a compiler builtin in GCC (12+)
//          and Clang for RISC-V targets -- no header required. For older
//          toolchains we emit the raw instruction encoding (0x0100000F).
//          PAUSE is a HINT (Zihintpause extension); it executes as a
//          NOP on cores that don't implement the extension.
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__clang__)
#include <arm_acle.h>
#endif
#endif

namespace tt::stl {

/**
 * @brief Issues a pause/yield hint to the processor.
 *
 * On x86_64, this calls _mm_pause() which hints to the processor that the
 * code is in a spin-wait loop. On ARM64, this issues the YIELD instruction
 * (via __yield() on Clang/MSVC, or inline asm on GCC). On RISC-V, this
 * issues the PAUSE hint instruction (Zihintpause extension).
 *
 * This helps reduce power consumption and improve performance of other
 * threads sharing the same core during busy-wait loops.
 */
__attribute__((always_inline)) inline void TT_PAUSE() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
#if defined(__clang__) || defined(_MSC_VER)
    __yield();
#else
    // GCC aarch64: __yield() is not provided in GCC's arm_acle.h.
    // See: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105416
    __asm__ volatile("yield");
#endif
#elif defined(__riscv) || defined(__riscv_xlen)
#if defined(__has_builtin) && __has_builtin(__builtin_riscv_pause)
    __builtin_riscv_pause();
#else
    // PAUSE encoded as FENCE W,0 (Zihintpause). Executes as NOP without the extension.
    __asm__ volatile(".4byte 0x0100000F");
#endif
#else
    // Fallback for other architectures - do nothing
#endif
}

/**
 * @brief Spins with TT_PAUSE() until the predicate evaluates to true.
 *
 * This function implements a "nice" spin loop that uses exponential backoff
 * to reduce impact on system resources. Every N_SPINS iterations, it sleeps
 * for an increasing duration (starting at 1us, doubling each time up to
 * MAX_WAIT_US).
 *
 * @tparam N_SPINS Number of pause iterations between sleeps (default: 100)
 * @tparam MAX_WAIT_US Maximum sleep duration in microseconds (default: 16)
 * @param predicate A callable that returns true when the wait should end
 */
template <uint32_t N_SPINS = 100, uint32_t MAX_WAIT_US = 16, typename... Ts>
__attribute__((flatten)) inline void TT_NICE_SPIN_UNTIL(auto predicate, Ts&&... args) {
    uint32_t counter = 0;
    uint32_t sleep_us = 1;
    while (!predicate(std::forward<Ts>(args)...)) {
        ++counter;
        if (counter < N_SPINS) {
            TT_PAUSE();
        } else {
            counter = 0;
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_us));
            sleep_us = std::min(sleep_us * 2, MAX_WAIT_US);
        }
    }
}

}  // namespace tt::stl
