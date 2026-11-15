// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gated_delta_net compute, q / k roles (GDN_ROLE_QK: R-QKCORE, GDN_ROLE_PREP_QK: R-PREP): conv and L2 norm of one
// key head's q or k columns, handed to the writer through cb_qk_stage.

#include "gated_delta_net_compute_common.hpp"

static_assert(QK_CORE || PREP_QK, "q / k-role kernel");

void kernel_main() {
    const uint32_t first_unit = get_arg_val<uint32_t>(0);
    const uint32_t num_units = get_arg_val<uint32_t>(1);
    const uint32_t T = get_arg_val<uint32_t>(2);
    const uint32_t qk_role = get_arg_val<uint32_t>(4);
    const uint32_t tr_offset = get_arg_val<uint32_t>(5);  // R-PREP: tile rows offset, + stride, ...
    const uint32_t tr_stride = get_arg_val<uint32_t>(6);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_y_qk, cb_sq, cb_qk_stage);
    Buffers bufs;
    // R-QKCORE: num_units = the sequences B (token rows 0..B-1), one tile row. R-PREP q / k core: whole tiles of
    // tile rows offset, offset + stride, ... of its sequence. qn or kn of the Kt columns -> cb_qk_stage (the writer
    // multicasts / writes it to the prep cores).
    const bool is_q = qk_role == GDN_QK_Q_INJECTOR;
    const UnitCols cols{is_q ? 0u : Kt, Kt};
    const RowScope scope = PREP_QK ? RowScope{RowScope::FULL_TILE} : RowScope::of(0, num_units);
    uint32_t rows = 1;
    if constexpr (PREP_QK) {
        const uint32_t seq_lo = (first_unit / PIECES / Hv) * T;
        const uint32_t first_tr = seq_lo / TILE_ROWS;
        const uint32_t last_tr = (seq_lo + T - 1) / TILE_ROWS;
        rows = first_tr + tr_offset <= last_tr ? (last_tr - first_tr - tr_offset) / tr_stride + 1 : 0;
    }
    cb_wait_front(cb_conv_w, (CONV_ON_FPU ? 1 : GDN_CONV_TAPS) * Kt);
#pragma GCC unroll 1
    for (uint32_t row = 0; row < rows; ++row) {
        {
            GDN_ZONE("gdn_conv");
            if constexpr (CONV_ON_FPU) {
                conv_block_fpu(scope, cols, 0, Kt);
            } else {
                conv_block(scope, cols, 0, Kt);
            }
        }
        GDN_MARK("m_conv");
        GDN_ZONE("gdn_qk_norm");
        if constexpr (QK_RAW) {
            // y already went to the writer with the conv; the inverse norm follows it.
            l2_inverse_norm(bufs, 0, is_q ? Q_SCALE_BITS : ONE_BITS, scope);
            cb_pop_front(cb_y_qk, Kt);
        } else {
            qk_part_block(bufs, is_q, scope);
        }
        GDN_MARK("m_qk_norm");
    }
    cb_pop_front(cb_conv_w, (CONV_ON_FPU ? 1 : GDN_CONV_TAPS) * Kt);
}
