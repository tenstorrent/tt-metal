// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gated_delta_net compute, R-PREP state role (GDN_ROLE_STATE): the chunk step's state part on prepped tile rows.

#include "gated_delta_net_compute_common.hpp"
#include "gated_delta_net_compute_chunk.hpp"

static_assert(STATE, "R-PREP state-role kernel");

// R-PREP state core: the chunk step's state part per tile row on its inbox slot; o goes to the row's prep core.
ALWI void state_main(Buffers& bufs, uint32_t first_unit, uint32_t T, uint32_t v_eps_bits) {
    // SEQ_SPLIT: sequences SEGS.. are the P halves of segments 1..; a segment 1.. holds its rows' o for seg_finish.
    const uint32_t vseq = first_unit / PIECES / Hv;
    const bool p_half = SPLIT && vseq >= SEGS;
    const uint32_t seq = p_half ? vseq - (SEGS - 1) : vseq;
    const bool held = SPLIT && seq > 0;
    const uint32_t seq_lo = seq * T;
    const uint32_t seq_hi = seq_lo + T;
    const uint32_t first_tr = seq_lo / TILE_ROWS;
    const uint32_t last_tr = (seq_hi - 1) / TILE_ROWS;
    uint32_t token = 0;
    for (uint32_t tr = first_tr; tr <= last_tr; ++tr) {
        const uint32_t base_row = tr * TILE_ROWS;
        const uint32_t lo = seq_lo > base_row ? seq_lo : base_row;
        const uint32_t hi = seq_hi < base_row + TILE_ROWS ? seq_hi : base_row + TILE_ROWS;
        // Host STATE_NORM_TAIL: the last rows' o stay here for this core's RMS norm after the chain.
        const uint32_t o_cb = held || tr + GDN_NORM_TAIL > last_tr ? cb_reads : cb_o_acc;
        cb_reserve_back(o_cb, Vb);
        cb_wait_front(cb_in_vec, GDN_IN_VEC_PAGES);
        cb_wait_front(cb_in_knt, Kt);
        const uint32_t decay_bits = read_tile_value(cb_in_vec, 0, elem_index(hi - 1 - base_row, 0));  // exp(c_last)
        {
            GDN_ZONE("gdn_step");
            // A segment 1..'s H stays in cb_state after the last step: the combine's operand.
            const bool last = token + (hi - lo) == T && !(held && !p_half);
            chunk_state_prepped(bufs, token == 0, last, decay_bits, p_half, o_cb);
        }
        token += hi - lo;
        cb_push_back(o_cb, Vb);
        cb_pop_front(cb_in_kq, 2 * Kt);
        cb_pop_front(cb_in_knt, Kt);
        cb_pop_front(cb_in_mat, GDN_IN_MAT_PAGES);
        cb_pop_front(cb_in_vec, GDN_IN_VEC_PAGES);
        cb_pop_front(cb_in_v, Vb);  // last: the reader grants a row's slots on a free cb_in_v slot
    }
    if constexpr (SPLIT) {
        if (held && !p_half) {
            seg_finish(last_tr - first_tr + 1);
        }
    }
    // Two empty windows first: STATE_HOLD leaves its DEST half as the chain used it, and their releases clear both.
    for (uint32_t w = 0; w < 2 && GDN_NORM_TAIL != 0; ++w) {
        tile_regs_acquire();
        tile_regs_commit();
        tile_regs_wait();
        tile_regs_release();
    }
#pragma GCC unroll 1
    for (uint32_t r = 0; r < GDN_NORM_TAIL; ++r) {
        GDN_ZONE("gdn_rmsnorm");
        rmsnorm_gate_block<cb_reads, 1>(bufs, v_eps_bits, RowScope{RowScope::FULL_TILE});
    }
}

void kernel_main() {
    const uint32_t first_unit = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(2);

    // R-PREP allocates each role only its own buffers: start up on buffers the role has.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_in_kq, cb_state_mm, cb_reads);
    Buffers bufs;
    state_main(bufs, first_unit, T, get_arg_val<uint32_t>(3));
}
