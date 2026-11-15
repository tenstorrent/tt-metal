// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gated_delta_net compute, R-PREP state role (GDN_ROLE_STATE): the chunk step's state part on prepped tile rows.

#include "gated_delta_net_compute_common.hpp"
#include "gated_delta_net_compute_chunk.hpp"

static_assert(STATE, "R-PREP state-role kernel");

// R-PREP state core: the chunk step's state part per tile row on its inbox slot; o goes to the row's prep core.
ALWI void state_main(Buffers& bufs, uint32_t first_unit, uint32_t T) {
    const uint32_t seq_lo = (first_unit / PIECES / Hv) * T;
    const uint32_t seq_hi = seq_lo + T;
    const uint32_t first_tr = seq_lo / TILE_ROWS;
    const uint32_t last_tr = (seq_hi - 1) / TILE_ROWS;
    uint32_t token = 0;
    for (uint32_t tr = first_tr; tr <= last_tr; ++tr) {
        const uint32_t base_row = tr * TILE_ROWS;
        const uint32_t lo = seq_lo > base_row ? seq_lo : base_row;
        const uint32_t hi = seq_hi < base_row + TILE_ROWS ? seq_hi : base_row + TILE_ROWS;
        cb_reserve_back(cb_o_acc, Vb);
        cb_wait_front(cb_in_vec, GDN_IN_VEC_PAGES);
        cb_wait_front(cb_in_knt, Kt);
        const uint32_t decay_bits = read_tile_value(cb_in_vec, 0, elem_index(hi - 1 - base_row, 0));  // exp(c_last)
        {
            GDN_ZONE("gdn_step");
            chunk_state_prepped(bufs, token == 0, token + (hi - lo) == T, decay_bits);
        }
        token += hi - lo;
        cb_push_back(cb_o_acc, Vb);
        cb_pop_front(cb_in_kq, 2 * Kt);
        cb_pop_front(cb_in_knt, Kt);
        cb_pop_front(cb_in_mat, GDN_IN_MAT_PAGES);
        cb_pop_front(cb_in_vec, GDN_IN_VEC_PAGES);
        cb_pop_front(cb_in_v, Vb);  // last: the reader grants a row's slots on a free cb_in_v slot
    }
}

void kernel_main() {
    const uint32_t first_unit = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(2);

    // R-PREP allocates each role only its own buffers: start up on buffers the role has.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in_kq, cb_state_mm, cb_reads);
    Buffers bufs;
    state_main(bufs, first_unit, T);
}
