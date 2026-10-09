// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The streamed-expert weight schedule shared by se_reader.cpp and se_forward.cpp. A reader's region is a sequence of
// chunks in consumption order, gu(0), gu(1), d(0), gu(2), d(1), ..., d(E - 1), where gu(e) is NK_GU gate/up chunks
// and d(e) is P passes x NK_D down chunks (pass-major). A chunk holds one block per receiver: gate/up blocks are
// [KBLK x 2] tiles, down blocks of pass p are [KBLK x W(j, p)] tiles (the receiver's output columns of that pass).
#pragma once
#include <stdint.h>

// Calls f(is_down, pass) for every chunk in stream order. PIPELINED = 0 gives the plain order gu(0), d(0), gu(1), ...
// (the resident-weight big-M variant, where the ring holds a whole expert).
template <uint32_t nk_gu, uint32_t nk_d, uint32_t passes, uint32_t num_experts, bool pipelined = true, typename F>
inline void for_each_chunk(F&& f) {
    if constexpr (!pipelined) {
        for (uint32_t e = 0; e < num_experts; ++e) {
            for (uint32_t c = 0; c < nk_gu; ++c) {
                f(false, 0u);
            }
            for (uint32_t p = 0; p < passes; ++p) {
                for (uint32_t c = 0; c < nk_d; ++c) {
                    f(true, p);
                }
            }
        }
        return;
    }
    for (uint32_t step = 0; step <= num_experts; ++step) {
        if (step < num_experts) {
            for (uint32_t c = 0; c < nk_gu; ++c) {
                f(false, 0u);
            }
        }
        if (step > 0) {
            for (uint32_t p = 0; p < passes; ++p) {
                for (uint32_t c = 0; c < nk_d; ++c) {
                    f(true, p);
                }
            }
        }
    }
}
