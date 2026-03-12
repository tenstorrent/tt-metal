// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Systematic reproducer — all behavior via -D defines from test harness.

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"

#ifndef USE_FAST_TILIZE
#define USE_FAST_TILIZE 1
#endif
#ifndef SRCA_INDEX
#define SRCA_INDEX 1
#endif
#ifndef SRCB_INDEX
#define SRCB_INDEX 2
#endif

constexpr uint32_t c_in = 0;
constexpr uint32_t c_til = 1;
constexpr uint32_t c_out = 17;
constexpr uint32_t Wt = get_compile_time_arg_val(1);

void kernel_main() {
    uint32_t Ht = get_arg_val<uint32_t>(0);
    if (Ht == 0) {
        return;
    }

    compute_kernel_hw_startup(SRCA_INDEX, SRCB_INDEX, c_out);

    for (uint32_t row = 0; row < Ht; ++row) {
        cb_wait_front(c_in, Wt);
        cb_reserve_back(c_til, Wt);
#if USE_FAST_TILIZE
        fast_tilize_init(c_in, Wt, c_til);
        fast_tilize_block(c_in, Wt, c_til);
        fast_tilize_uninit(c_in, c_til);
#else
        tilize_init(c_in, Wt, c_til);
        tilize_block(c_in, Wt, c_til);
        tilize_uninit(c_in, c_til);
#endif
        cb_push_back(c_til, Wt);
        cb_pop_front(c_in, Wt);

        pack_untilize_init<Wt>(c_til, c_out);
        cb_wait_front(c_til, Wt);
        cb_reserve_back(c_out, Wt);
        pack_untilize_block<Wt>(c_til, 1, c_out, 0);
        cb_pop_front(c_til, Wt);
        cb_push_back(c_out, Wt);
        pack_untilize_uninit(c_out);
    }
}
