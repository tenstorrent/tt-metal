// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared by the TailA2A port kernels: compile-time args 0-5 and the run walk.

#pragma once

#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CB = get_compile_time_arg_val(0), PB = get_compile_time_arg_val(1), RP = get_compile_time_arg_val(2);
// (fused_tail_program_factory's A2ACb: the z-ready words' CB; the workers' zero-gate semaphore)
constexpr uint32_t CB_ZR = 5, SZR = 0;
constexpr uint32_t NB = get_compile_time_arg_val(3), GROUP = get_compile_time_arg_val(4),
                   INC = get_compile_time_arg_val(5);

// a port's runs of a destination block's row tile: link l of L takes the bank offsets t0 = l, l + L, .. (< min(B,
// NTB)); part 1 / 2 the first / second half of those; a run = up to RP pages t0 + (m + i) B
template <typename F>
FORCE_INLINE void for_each_run(uint32_t NTB, uint32_t l, uint32_t L, uint32_t part, F&& f) {
    const uint32_t nt = NTB < NB ? NTB : NB;
    const uint32_t nl = nt > l ? (nt - l + L - 1) / L : 0;  // (my bank offsets)
    const uint32_t lo = part == 2 ? nl / 2 : 0, hi = part == 1 ? nl / 2 : nl;
    for (uint32_t i = lo; i < hi; ++i) {
        const uint32_t t0 = l + i * L;
        for (uint32_t m = t0; m < NTB; m += RP * NB) {
            const uint32_t left = (NTB - m + NB - 1) / NB;
            f(m, left < RP ? left : RP);
        }
    }
}
