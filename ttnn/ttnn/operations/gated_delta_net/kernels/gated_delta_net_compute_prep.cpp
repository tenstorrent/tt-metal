// SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// gated_delta_net compute, R-PREP prep role (GDN_ROLE_PREP): gates, conv, L2 norm and head blocks of a (sequence,
// key head)'s tile rows for its state cores, then the RMS norm and gate of the o they return.

#include "gated_delta_net_compute_common.hpp"
#include "gated_delta_net_compute_chunk.hpp"

static_assert(PREP, "R-PREP prep-role kernel");

// R-PREP prep core: zw = norm_weight (*)row silu(z) of a tile row's Vb columns -> cb_zw (fp32), while the state cores
// still run the row, so the RMS norm after o lands is two products per tile.
template <uint32_t N>
ALWI void zw_block() {
    constexpr uint32_t W = PREP ? Vb / RATIO : 1;  // columns of one head (norm_weight holds W tiles)
    GDN_ZONE("gdn_zw");
    GDN_WAIT_FRONT("w_z", cb_z, N);
    cb_reserve_back(cb_zw, N);
    if constexpr (GDN_ZW_FPU != 0) {
        // Host PREP_ZW_FPU: silu(z) of the unit's tiles in one window; norm_weight enters in the norm's FPU product.
        static_assert(N <= ckl::DEST_AUTO_LIMIT);
        tile_regs_acquire();
        begin_loads(cb_z);
        for (uint32_t vt = 0; vt < N; ++vt) {
            load(cb_z, vt, vt);
        }
        silu_tile_init();
        for (uint32_t vt = 0; vt < N; ++vt) {
            if constexpr ((GDN_SILU_BF16 & 1) != 0) {
                silu_tile_bf16(vt);
            } else {
                silu_tile(vt);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_zw, 0);
        for (uint32_t vt = 1; vt < N; ++vt) {
            pack_at(vt, cb_zw, vt);
        }
        tile_regs_release();
    }
    for (uint32_t vt = 0; GDN_ZW_FPU == 0 && vt < N; ++vt) {
        tile_regs_acquire();
        begin_loads(cb_z);
        load(cb_z, vt, 0);
        load(cb_norm_w, vt % W, 1);
        silu_tile_init();
        if constexpr ((GDN_SILU_BF16 & 1) != 0) {
            silu_tile_bf16(0);
        } else {
            silu_tile(0);
        }
        sfpu_bcast_row_init();
        sfpu_mul_bcast_row(0, 1);
        tile_regs_commit();
        tile_regs_wait();
        pack_first(0, cb_zw, vt);
        tile_regs_release();
    }
    cb_push_back(cb_zw, N);
    cb_pop_front(cb_z, N);
}

// R-PREP prep core: tile rows offset, offset + stride, ... of its (sequence, key head). The RMS norm of a row runs
// after the next row's prep, by when the state cores have usually returned the row's o; under NORM_SPLIT after all of
// its prep rows, on the norm units the host gives it.
ALWI void prep_main(
    Buffers& bufs,
    uint32_t first_unit,
    uint32_t T,
    uint32_t tr_offset,
    uint32_t tr_stride,
    uint32_t v_eps_bits,
    uint32_t norm_rows,
    uint32_t norm_mid) {
    const UnitCols cols{0, PREP_SHARED_QK ? 0 : 2 * Kt};
    const uint32_t conv_w_pages = (CONV_ON_FPU ? 1 : GDN_CONV_TAPS) * cols.count();
    const uint32_t seq_lo = (first_unit / PIECES / Hv) * T;
    const uint32_t seq_hi = seq_lo + T;
    const uint32_t first_tr = seq_lo / TILE_ROWS;
    const uint32_t last_tr = (seq_hi - 1) / TILE_ROWS;
    const RowScope full{RowScope::FULL_TILE};
    cb_wait_front(cb_conv_w, conv_w_pages);
    cb_wait_front(cb_gate_w, 2);
    cb_wait_front(cb_norm_w, Vb / RATIO);
    // Pass k preps row k and normalizes row k - 1 (one call site each: the math thread's text budget is tight).
    constexpr bool NORM_SPLIT = GDN_PREP_NORM_SPLIT != 0;
    constexpr uint32_t NORM_HEADS = NORM_SPLIT ? 1 : RATIO;
    constexpr uint32_t NORM_TILES = NORM_HEADS * (Vb / RATIO);
    const uint32_t rows = first_tr + tr_offset <= last_tr ? (last_tr - first_tr - tr_offset) / tr_stride + 1 : 0;
    const uint32_t passes = NORM_SPLIT || rows != 0 ? rows + 1 : 0;
    // NORM_SPLIT: row k's conv, then norm units (norm_mid of them before a later row's heads, where a row that waits
    // for its grant leaves the time; the rest after the last row), then row k's heads. One call site per block.
    uint32_t done = 0;
#pragma GCC unroll 1
    for (uint32_t k = 0; k < passes; ++k) {
        const uint32_t base_row = (first_tr + tr_offset + k * tr_stride) * TILE_ROWS;
        const uint32_t hi = seq_hi < base_row + TILE_ROWS ? seq_hi : base_row + TILE_ROWS;
        if (k < rows) {
            GDN_ZONE("gdn_prep");  // gates, conv
            gates_block(full);
            if constexpr (CONV_ON_FPU) {
                conv_block_fpu(full, cols, 0, cols.count());
            } else {
                conv_block(full, cols, 0, cols.count());
            }
        }
        if constexpr (NORM_SPLIT) {
            uint32_t n = k == rows ? norm_rows : (k != 0 ? done + norm_mid : 0);
            n = n < norm_rows ? n : norm_rows;
#pragma GCC unroll 1
            for (; done < n; ++done) {
                zw_block<NORM_TILES>();
                GDN_ZONE("gdn_rmsnorm");
                rmsnorm_gate_block<cb_o_in, NORM_HEADS, false, NORM_TILES>(bufs, v_eps_bits, full);
            }
        }
        if (k < rows) {
            cb_wait_front(cb_tri_mask, 2);  // writer-built once (mask, identity), held for the kernel
            cb_wait_front(cb_v_prep, Vb);
            {
                GDN_ZONE("gdn_heads");  // L2 norm, gram, head blocks
#pragma GCC unroll 1
                for (uint32_t head = 0; head < RATIO; ++head) {
                    // The first head's chunk vectors need only the gates: under PREP_SHARE_QK they run while qn / kn
                    // are still on their way from the q / k cores.
                    head_prep_vectors(hi - 1 - base_row);
                    if (head == 0) {
                        if constexpr (PREP_SHARED_QK) {
                            kn_transpose_block(bufs);  // on kn / qn as landed by the reader
                        } else {
                            l2norm_block(bufs, full);
                        }
                        gram_block(bufs);
                    }
                    head_prep_matrices();
                    head_block(head);
                }
            }
            cb_pop_front(cb_v_prep, Vb);
            cb_pop_front(cb_gram, 2);
            cb_pop_front(cb_kq_rows, 2 * Kt);
            cb_pop_front(cb_knT, Kt);
        }
        if constexpr (!NORM_SPLIT) {
            if (k < rows) {
                zw_block<NORM_TILES>();
            }
            if (k > 0) {
                GDN_ZONE("gdn_rmsnorm");
                rmsnorm_gate_block<cb_o_in, NORM_HEADS, false, NORM_TILES>(bufs, v_eps_bits, full);
            }
        }
    }
    cb_pop_front(cb_norm_w, Vb / RATIO);
}

void kernel_main() {
    const uint32_t first_unit = get_arg_val<uint32_t>(0);
    const uint32_t T = get_arg_val<uint32_t>(2);
    const uint32_t v_eps_bits = get_arg_val<uint32_t>(3);  // V * epsilon
    const uint32_t tr_offset = get_arg_val<uint32_t>(5);   // R-PREP: tile rows offset, + stride, ...
    const uint32_t tr_stride = get_arg_val<uint32_t>(6);
    const uint32_t norm_rows = get_arg_val<uint32_t>(8);  // NORM_SPLIT: norm units this core normalizes
    const uint32_t norm_mid = get_arg_val<uint32_t>(9);   // of them, before the heads of each row after its first

    // R-PREP allocates each role only its own buffers: start up on buffers the role has.
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_kq_rows, cb_knT, cb_gram);
    Buffers bufs;
    prep_main(bufs, first_unit, T, tr_offset, tr_stride, v_eps_bits, norm_rows, norm_mid);
}
