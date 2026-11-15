// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/experimental/add_rsqrt.h"
#include "api/compute/experimental/mul_reduce_scalar.h"
#include "api/compute/experimental/pack_block.h"
#include "api/compute/experimental/rmsnorm.h"
#include "api/compute/reconfig_data_format.h"
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

// The fused collapsed RMSNorm's last step (as compute_sharded_rmsnorm.cpp's): src[src_start .. + num_tiles) times
// the scalar scale[0][0] and gamma[src_start .. + num_tiles), packed into out_cb. num_tiles fp32 tiles fit in Dst.
template <uint32_t num_tiles>
ALWI void scale_norm_chunk(uint32_t src_cb, uint32_t src_start, uint32_t scale_cb, uint32_t gamma_cb, uint32_t out_cb) {
    reconfig_data_format(scale_cb, scale_cb);
    tile_regs_acquire();
    copy_init(scale_cb);
    copy_tile(scale_cb, 0, 0);
    reconfig_data_format_srca(scale_cb, src_cb);
    rmsnorm_bcast_scalar_reuse_tiles_init_fidelity<EltwiseBinaryType::ELWMUL, num_tiles, MathFidelity::HiFi4>(src_cb);
    rmsnorm_bcast_scalar_reuse_tiles_fidelity<EltwiseBinaryType::ELWMUL, num_tiles, MathFidelity::HiFi4, true>(
        src_cb, src_start, 0, 0);
    reconfig_data_format(gamma_cb, gamma_cb);
    mul_reuse_dest_init_fidelity<EltwiseBinaryReuseDestType::DEST_TO_SRCA, MathFidelity::HiFi4>(gamma_cb);
    for (uint32_t k = 0; k < num_tiles; ++k) {
        mul_reuse_dest_tiles_fidelity<EltwiseBinaryReuseDestType::DEST_TO_SRCA, MathFidelity::HiFi4>(
            gamma_cb, src_start + k, k);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_block_contiguous(0, out_cb, num_tiles);
    tile_regs_release();
}

// RMSNorm(collapsed row) * gamma over the collapse cores (see FusedSingleUserInputs::collapsed_norm_weight), on
// this core's num_chunks h x 32 tiles of the row in cb_x: sum(x^2) into cb_stat, which the norm writer swaps with
// the other collapse cores into cb_gather; then rsqrt(sum / D + eps) into cb_scale, and x * scale * gamma into
// cb_out. Each step reconfigures the formats it uses (the row is BFLOAT16, the statistics Float32).
template <uint32_t num_chunks, uint32_t num_cores, uint32_t inv_d_bits, uint32_t eps_bits>
ALWI void collapsed_rmsnorm(
    uint32_t cb_x, uint32_t cb_gamma, uint32_t cb_stat, uint32_t cb_gather, uint32_t cb_scale, uint32_t cb_out) {
    // 1. sum(x^2) over the core's row into cb_stat[0][0].
    cb_wait_front(cb_x, num_chunks);
    cb_reserve_back(cb_stat, 1);
    reconfig_data_format(cb_x, cb_x);
    pack_reconfig_data_format(cb_stat);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.0F);
    mul_reduce_scalar_init_fidelity<MathFidelity::HiFi4>(cb_x, cb_x);
    mul_reduce_scalar_tile_fidelity<MathFidelity::HiFi4, MathFidelity::HiFi4, true>(
        cb_x, cb_x, cb_stat, num_chunks, 1.0F, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_stat);
    mul_reduce_scalar_uninit();
    tile_regs_release();
    cb_push_back(cb_stat, 1);

    // 2. Every collapse core's partial, summed in fp32, then rsqrt(sum * (1/D) + eps).
    cb_wait_front(cb_gather, num_cores);
    cb_reserve_back(cb_scale, 1);
    reconfig_data_format(cb_gather, cb_gather);
    pack_reconfig_data_format(cb_scale);
    tile_regs_acquire();
    copy_init(cb_gather);
    copy_tile(cb_gather, 0, 0);
    for (uint32_t c = 1; c < num_cores; ++c) {
        copy_init(cb_gather);
        copy_tile(cb_gather, c, 1);
        add_binary_tile_init();
        add_binary_tile(0, 1, 0);
    }
    add_rsqrt_tile_init();
    add_rsqrt_tile<false, VectorMode::RC_custom, 1, false, inv_d_bits>(0, eps_bits);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_scale);
    tile_regs_release();
    cb_push_back(cb_scale, 1);
    cb_pop_front(cb_gather, num_cores);

    // 3. x * scale * gamma, at most four fp32 tiles at a time.
    cb_wait_front(cb_scale, 1);
    cb_wait_front(cb_gamma, num_chunks);
    cb_reserve_back(cb_out, num_chunks);
    pack_reconfig_data_format(cb_out);
    pack_block_contiguous_init(cb_out);
    for (uint32_t t = 0; t < num_chunks;) {
        const uint32_t rem = num_chunks - t;
        if (rem >= 4) {
            scale_norm_chunk<4>(cb_x, t, cb_scale, cb_gamma, cb_out);
            t += 4;
        } else if (rem == 3) {
            scale_norm_chunk<3>(cb_x, t, cb_scale, cb_gamma, cb_out);
            t += 3;
        } else if (rem == 2) {
            scale_norm_chunk<2>(cb_x, t, cb_scale, cb_gamma, cb_out);
            t += 2;
        } else {
            scale_norm_chunk<1>(cb_x, t, cb_scale, cb_gamma, cb_out);
            t += 1;
        }
    }
    cb_push_back(cb_out, num_chunks);
    cb_pop_front(cb_scale, 1);
    cb_pop_front(cb_gamma, num_chunks);
    cb_pop_front(cb_x, num_chunks);
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
    // The folded mix step (role 0, ROW_MAJOR streams, pre_mix): its weights (see hc_mix_collapse_streams), the
    // sublayer output's h x 32 chunks and the mixed streams' shard, laid out as cb_hidden.
    constexpr bool use_mix = get_compile_time_arg_val(17) != 0;
    [[maybe_unused]] constexpr uint32_t cb_mix_w = get_compile_time_arg_val(18);
    [[maybe_unused]] constexpr uint32_t cb_mix_sub = get_compile_time_arg_val(19);
    [[maybe_unused]] constexpr uint32_t cb_mixed = get_compile_time_arg_val(20);
    // The fused collapsed RMSNorm (role 0, ROW_MAJOR streams): see collapsed_rmsnorm.
    constexpr bool fuse_norm = get_compile_time_arg_val(21) != 0;
    [[maybe_unused]] constexpr uint32_t cb_norm_gamma = get_compile_time_arg_val(22);
    [[maybe_unused]] constexpr uint32_t cb_norm_stat = get_compile_time_arg_val(23);
    [[maybe_unused]] constexpr uint32_t cb_norm_gather = get_compile_time_arg_val(24);
    [[maybe_unused]] constexpr uint32_t cb_norm_scale = get_compile_time_arg_val(25);
    [[maybe_unused]] constexpr uint32_t cb_norm_out = get_compile_time_arg_val(26);
    [[maybe_unused]] constexpr uint32_t norm_inv_d_bits = get_compile_time_arg_val(27);
    [[maybe_unused]] constexpr uint32_t norm_eps_bits = get_compile_time_arg_val(28);
    [[maybe_unused]] constexpr uint32_t norm_num_cores = get_compile_time_arg_val(29);

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

        if constexpr (use_mix) {
            static_assert(
                !use_mix || (hidden_row_major && use_pre_mix),
                "the folded mix step takes ROW_MAJOR streams and pre_mix");
            // Chunk j: stream k's chunk j into Dst k, the sublayer output's into Dst H; the mixed streams come back
            // in Dst 0..H-1 and the collapse in Dst H (needs a full-sync Dst: H + 1 fp32 tiles).
            [[maybe_unused]] const volatile tt_l1_ptr uint16_t* pre =
                reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(get_tile_address(cb_pre, 0));
            cb_wait_front(cb_mix_w, 1);
            [[maybe_unused]] const volatile tt_l1_ptr uint16_t* mix_w =
                reinterpret_cast<const volatile tt_l1_ptr uint16_t*>(get_tile_address(cb_mix_w, 0));
            // The math thread reads the weights straight from L1; its data cache can hold an earlier program's lines.
            invalidate_l1_cache();
            cb_wait_front(cb_hidden, num_streams * collapse_tiles);
            cb_wait_front(cb_mix_sub, collapse_tiles);
            cb_reserve_back(cb_mixed, num_streams * collapse_tiles);
            for (uint32_t j = 0; j < collapse_tiles; ++j) {
                tile_regs_acquire();
                copy_init(cb_hidden);
                for (uint32_t k = 0; k < num_streams; ++k) {
                    copy_tile(cb_hidden, k * collapse_tiles + j, k);
                }
                copy_init(cb_mix_sub);
                copy_tile(cb_mix_sub, j, num_streams);
                MATH((llk_math_eltwise_unary_sfpu_init<SfpuType::unused, DST_ACCUM_MODE>()));
                MATH((ckernel::sfpu::hc_mix_collapse_streams<num_streams>(mix_w, pre)));
                tile_regs_commit();

                cb_reserve_back(cb_collapsed, 1);
                tile_regs_wait();
                for (uint32_t k = 0; k < num_streams; ++k) {
                    pack_tile<true>(k, cb_mixed, k * collapse_tiles + j);
                }
                pack_tile(num_streams, cb_collapsed);
                tile_regs_release();
                cb_push_back(cb_collapsed, 1);
            }
            // The reader raises the fn dot cores' semaphore on this; the program ends with it unpopped.
            cb_push_back(cb_mixed, num_streams * collapse_tiles);
            cb_pop_front(cb_hidden, num_streams * collapse_tiles);
            cb_pop_front(cb_mix_sub, collapse_tiles);
            cb_pop_front(cb_mix_w, 1);
        } else if constexpr (hidden_row_major) {
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
        if constexpr (fuse_norm) {
            static_assert(!fuse_norm || hidden_row_major, "the fused collapsed RMSNorm takes ROW_MAJOR streams");
            collapsed_rmsnorm<collapse_tiles, norm_num_cores, norm_inv_d_bits, norm_eps_bits>(
                cb_collapsed, cb_norm_gamma, cb_norm_stat, cb_norm_gather, cb_norm_scale, cb_norm_out);
        }
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
