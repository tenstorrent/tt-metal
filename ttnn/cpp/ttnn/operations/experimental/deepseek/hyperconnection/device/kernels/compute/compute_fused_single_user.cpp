// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "hc_single_user_sfpu.h"

namespace {

constexpr uint32_t kNoCb = 0xFFFFFFFF;

// Copies the reader's parameter tile into Dst 0 for one of the SFPU passes in hc_single_user_sfpu.h.
void load_params(uint32_t cb_params) {
    copy_init(cb_params);
    MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
    cb_wait_front(cb_params, 1);
    tile_regs_acquire();
    copy_tile(cb_params, 0, 0);
}

// Packs Dst dst to cb_out and, when cb_out2 is a CB, Dst dst2 to cb_out2.
void pack_params_outputs(
    uint32_t cb_params, uint32_t dst, uint32_t cb_out, uint32_t dst2 = 0, uint32_t cb_out2 = kNoCb) {
    tile_regs_commit();
    cb_pop_front(cb_params, 1);

    cb_reserve_back(cb_out, 1);
    if (cb_out2 != kNoCb) {
        cb_reserve_back(cb_out2, 1);
    }
    tile_regs_wait();
    pack_reconfig_data_format(cb_out);
    pack_tile(dst, cb_out);
    if (cb_out2 != kNoCb) {
        pack_reconfig_data_format(cb_out2);
        pack_tile(dst2, cb_out2);
    }
    tile_regs_release();
    cb_push_back(cb_out, 1);
    if (cb_out2 != kNoCb) {
        cb_push_back(cb_out2, 1);
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t role = get_compile_time_arg_val(0);
    constexpr uint32_t cb_hidden = get_compile_time_arg_val(1);
    constexpr uint32_t cb_params = get_compile_time_arg_val(2);
    constexpr uint32_t cb_pre = get_compile_time_arg_val(3);
    constexpr uint32_t cb_collapsed = get_compile_time_arg_val(4);
    constexpr uint32_t cb_post_out = get_compile_time_arg_val(5);
    constexpr uint32_t cb_comb_out = get_compile_time_arg_val(6);
    constexpr uint32_t cb_pre_out = get_compile_time_arg_val(7);
    [[maybe_unused]] constexpr uint32_t pre_scale_bits = get_compile_time_arg_val(8);
    [[maybe_unused]] constexpr uint32_t post_scale_bits = get_compile_time_arg_val(9);
    [[maybe_unused]] constexpr uint32_t comb_scale_bits = get_compile_time_arg_val(10);
    [[maybe_unused]] constexpr uint32_t eps_bits = get_compile_time_arg_val(11);
    constexpr uint32_t num_streams = get_compile_time_arg_val(12);
    [[maybe_unused]] constexpr uint32_t sinkhorn_iters = get_compile_time_arg_val(13);
    constexpr bool use_pre_mix = get_compile_time_arg_val(14) != 0;
    constexpr bool hidden_row_major = get_compile_time_arg_val(15) != 0;
    // Collapse output tiles: h x 32 chunks of the collapsed row (row-major) or 32x32 tiles (tiled).
    constexpr uint32_t collapse_tiles = get_compile_time_arg_val(16);

    if constexpr (role == 0) {
        // Every tile the row-major collapse handles (parameters, pre, hidden, collapsed row) is h x 32.
        compute_kernel_hw_startup(cb_hidden, cb_collapsed);
        if constexpr (!use_pre_mix) {
            load_params(cb_params);
            if constexpr (hidden_row_major) {
                MATH((ckernel::sfpu::hc_pre_interleaved<pre_scale_bits, eps_bits>()));
                pack_params_outputs(cb_params, 0, cb_pre);
            } else {
                MATH((
                    ckernel::sfpu::hc_post_pre<num_streams, post_scale_bits, pre_scale_bits, eps_bits, false, true>()));
                pack_params_outputs(cb_params, 2, cb_pre);
            }
        }
        cb_wait_front(cb_pre, 1);
        pack_reconfig_data_format(cb_collapsed);

        if constexpr (hidden_row_major) {
            // Row k of the [H, D/8] shard is tiles [k * collapse_tiles, (k + 1) * collapse_tiles) of cb_hidden;
            // the SFPU forms sum_k pre[k] * row_k one h x 32 chunk at a time.
            [[maybe_unused]] const volatile tt_l1_ptr uint16_t* pre =
                reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(get_tile_address(cb_pre, 0));
            // The math thread reads pre straight from L1; its data cache can hold an earlier program's lines.
            invalidate_l1_cache();
            copy_init(cb_hidden);
            MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
            cb_wait_front(cb_hidden, num_streams * collapse_tiles);
            for (uint32_t j = 0; j < collapse_tiles; ++j) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < num_streams; ++k) {
                    copy_tile(cb_hidden, k * collapse_tiles + j, k);
                }
                // pre_mix holds pre[k] at column k; the interleaved own-pre pass leaves it at column 2k.
                MATH((ckernel::sfpu::hc_collapse_streams<num_streams, use_pre_mix ? 1 : 2>(pre)));
                tile_regs_commit();

                cb_reserve_back(cb_collapsed, 1);
                tile_regs_wait();
                pack_tile(0, cb_collapsed);
                tile_regs_release();
                cb_push_back(cb_collapsed, 1);
            }
            cb_pop_front(cb_hidden, num_streams * collapse_tiles);
        } else {
            // pre (row 0 of a zero-padded tile) @ the [H, D/8] streams, one 32x32 tile at a time.
            matmul_init(cb_pre, cb_hidden);
            cb_wait_front(cb_hidden, collapse_tiles);
            for (uint32_t n = 0; n < collapse_tiles; ++n) {
                tile_regs_acquire();
                matmul_tiles(cb_pre, cb_hidden, 0, n, 0);
                tile_regs_commit();

                cb_reserve_back(cb_collapsed, 1);
                tile_regs_wait();
                pack_tile(0, cb_collapsed);
                tile_regs_release();
                cb_push_back(cb_collapsed, 1);
            }
            cb_pop_front(cb_hidden, collapse_tiles);
        }
        cb_pop_front(cb_pre, 1);
    } else if constexpr (role == 1) {
        compute_kernel_hw_startup(cb_params, cb_post_out);
        load_params(cb_params);
        MATH((ckernel::sfpu::hc_post_pre<num_streams, post_scale_bits, pre_scale_bits, eps_bits, true, use_pre_mix>()));
        if constexpr (use_pre_mix) {
            pack_params_outputs(cb_params, 1, cb_post_out, 2, cb_pre_out);
        } else {
            pack_params_outputs(cb_params, 1, cb_post_out);
        }
    } else {
        compute_kernel_hw_startup(cb_params, cb_comb_out);
        load_params(cb_params);
        MATH((ckernel::sfpu::hc_comb_sinkhorn<num_streams, sinkhorn_iters, eps_bits, comb_scale_bits>()));
        pack_params_outputs(cb_params, 0, cb_comb_out);
    }
}
