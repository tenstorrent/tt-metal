// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <thread>
#include <utility>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#include <arm_acle.h>
#endif

namespace tt::stl {

/**
 * @brief Issues a pause/yield hint to the processor.
 *
 * On x86_64, this calls _mm_pause() which hints to the processor that the
 * code is in a spin-wait loop. On ARM64, this calls __yield() which provides
 * similar functionality. On RISC-V, this calls __builtin_riscv_pause() which
 * provides similar functionality.
 *
 * This helps reduce power consumption and improve performance of other
 * threads sharing the same core during busy-wait loops.
 */
__attribute__((always_inline)) inline void TT_PAUSE() {
#if defined(__x86_64__) || defined(_M_X64)
    _mm_pause();
#elif defined(__aarch64__) || defined(_M_ARM64)
    __yield();
#elif defined(__riscv) || defined(__riscv_xlen)
    __builtin_riscv_pause();
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
