// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/experimental/add_rsqrt.h"
#include "api/compute/experimental/mul_reduce_scalar.h"
#include "api/compute/experimental/rmsnorm.h"
#include "api/compute/experimental/rope_sfpu.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

using namespace ckernel;

void kernel_main() {
    constexpr uint32_t do_pool = get_compile_time_arg_val(0);
    if constexpr (do_pool == 0) {
        return;
    }

    constexpr uint32_t has_window = get_compile_time_arg_val(1);
    constexpr uint32_t has_rope = get_compile_time_arg_val(2);
    constexpr uint32_t num_cores = get_compile_time_arg_val(3);
    constexpr uint32_t inv_n_bits = get_compile_time_arg_val(4);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(5);

    constexpr uint32_t cb_kv = get_compile_time_arg_val(6);
    constexpr uint32_t cb_k0 = get_compile_time_arg_val(7);
    constexpr uint32_t cb_gate = get_compile_time_arg_val(8);
    constexpr uint32_t cb_g0 = get_compile_time_arg_val(9);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(10);
    constexpr uint32_t cb_cs = get_compile_time_arg_val(11);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(12);
    constexpr uint32_t cb_pooled = get_compile_time_arg_val(13);
    constexpr uint32_t cb_local_stat = get_compile_time_arg_val(14);
    constexpr uint32_t cb_gather_stats = get_compile_time_arg_val(15);
    constexpr uint32_t cb_scale_hub = get_compile_time_arg_val(16);
    constexpr uint32_t cb_scale = get_compile_time_arg_val(17);
    constexpr uint32_t cb_out_latent = get_compile_time_arg_val(18);
    constexpr uint32_t cb_out_rope = get_compile_time_arg_val(19);

    const bool is_hub = get_arg_val<uint32_t>(0) != 0;
    const bool do_rope = get_arg_val<uint32_t>(1) != 0;

    CircularBuffer kv_cb(cb_kv);
    CircularBuffer gamma_cb(cb_gamma);
    CircularBuffer local_stat_cb(cb_local_stat);
    CircularBuffer scale_cb(cb_scale);
    CircularBuffer out_latent_cb(cb_out_latent);

    compute_kernel_hw_startup(cb_kv, cb_kv, cb_out_latent);

    kv_cb.wait_front(1);

    uint32_t src_cb_id = cb_kv;
    if constexpr (has_window != 0) {
        CircularBuffer k0_cb(cb_k0);
        CircularBuffer gate_cb(cb_gate);
        CircularBuffer g0_cb(cb_g0);
        CircularBuffer tmp_cb(cb_tmp);
        CircularBuffer pooled_cb(cb_pooled);

        // 1. w0 = sigmoid(g0 - g1) -> cb_tmp
        g0_cb.wait_front(1);
        gate_cb.wait_front(1);
        tmp_cb.reserve_back(1);
        reconfig_data_format(cb_g0, cb_gate);
        pack_reconfig_data_format(cb_tmp);
        sub_init(cb_g0, cb_gate);
        tile_regs_acquire();
        sub_tiles(cb_g0, cb_gate, 0, 0, 0);
        sigmoid_tile_init<false>();
        sigmoid_tile<VectorMode::R, false, DST_ACCUM_MODE, 2>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_tmp);
        tile_regs_release();
        tmp_cb.push_back(1);
        g0_cb.pop_front(1);
        gate_cb.pop_front(1);

        // 2. pooled = k1 + w0 * (k0 - k1) -> cb_pooled
        k0_cb.wait_front(1);
        tmp_cb.wait_front(1);
        pooled_cb.reserve_back(1);
        reconfig_data_format(cb_k0, cb_kv);
        pack_reconfig_data_format(cb_pooled);
        sub_init(cb_k0, cb_kv);
        tile_regs_acquire();
        sub_tiles(cb_k0, cb_kv, 0, 0, 0);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_tmp);
        mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_tmp, 0, 0);
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_kv);
        add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_kv, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_pooled);
        tile_regs_release();
        pooled_cb.push_back(1);
        k0_cb.pop_front(1);
        tmp_cb.pop_front(1);
        kv_cb.pop_front(1);

        pooled_cb.wait_front(1);
        src_cb_id = cb_pooled;
    }

    // 3. Local sum of squares -> cb_local_stat[0][0]
    reconfig_data_format(src_cb_id, src_cb_id);
    pack_reconfig_data_format(cb_local_stat);
    local_stat_cb.reserve_back(1);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.0F);
    mul_reduce_scalar_init_fidelity<MathFidelity::HiFi4>(src_cb_id, src_cb_id);
    mul_reduce_scalar_tile_fidelity<MathFidelity::HiFi4, MathFidelity::HiFi4, true>(
        src_cb_id, src_cb_id, cb_local_stat, 1, 1.0F, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_local_stat);
    mul_reduce_scalar_uninit();
    tile_regs_release();
    local_stat_cb.push_back(1);

    // 4. Hub core sums per-core stats, scales by 1/Dh, and computes rsqrt(mean + eps)
    if (is_hub) {
        CircularBuffer gather_cb(cb_gather_stats);
        CircularBuffer scale_hub_cb(cb_scale_hub);
        gather_cb.wait_front(num_cores);
        scale_hub_cb.reserve_back(1);
        reconfig_data_format(cb_gather_stats, cb_gather_stats);
        pack_reconfig_data_format(cb_scale_hub);
        tile_regs_acquire();
        if constexpr (num_cores == 1) {
            copy_init(cb_gather_stats);
            copy_tile(cb_gather_stats, 0, 0);
        } else {
            add_init(cb_gather_stats, cb_gather_stats);
            add_tiles(cb_gather_stats, cb_gather_stats, 0, 1, 0);
            if constexpr (num_cores > 2) {
                add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gather_stats);
                for (uint32_t i = 2; i < num_cores; ++i) {
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gather_stats, i, 0);
                }
            }
        }
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_n_bits);
        add_rsqrt_tile_init();
        add_rsqrt_tile<false, VectorMode::RC_custom, 1>(0, eps_bits);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_scale_hub);
        tile_regs_release();
        gather_cb.pop_front(num_cores);
        scale_hub_cb.push_back(1);
    }

    // 5. Scale by rsqrt * norm_weight -> cb_out_latent (and cb_tmp if do_rope)
    gamma_cb.wait_front(1);
    scale_cb.wait_front(1);
    out_latent_cb.reserve_back(1);
    CircularBuffer tmp_cb(cb_tmp);
    if constexpr (has_rope != 0) {
        if (do_rope) {
            tmp_cb.reserve_back(1);
        }
    }

    reconfig_data_format(cb_scale, cb_scale);
    pack_reconfig_data_format(cb_out_latent);
    tile_regs_acquire();
    copy_init(cb_scale);
    copy_tile(cb_scale, 0, 0);
    reconfig_data_format_srca(cb_scale, src_cb_id);
    rmsnorm_bcast_scalar_reuse_tiles_init_fidelity<EltwiseBinaryType::ELWMUL, 1, MathFidelity::HiFi4>(src_cb_id);
    rmsnorm_bcast_scalar_reuse_tiles_fidelity<EltwiseBinaryType::ELWMUL, 1, MathFidelity::HiFi4, true>(
        src_cb_id, 0, 0, 0);
    reconfig_data_format(cb_gamma, cb_gamma);
    mul_reuse_dest_init_fidelity<EltwiseBinaryReuseDestType::DEST_TO_SRCA, MathFidelity::HiFi4>(cb_gamma);
    mul_reuse_dest_tiles_fidelity<EltwiseBinaryReuseDestType::DEST_TO_SRCA, MathFidelity::HiFi4>(cb_gamma, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_out_latent);
    if constexpr (has_rope != 0) {
        if (do_rope) {
            pack_reconfig_data_format(cb_tmp);
            pack_tile(0, cb_tmp);
        }
    }
    tile_regs_release();
    out_latent_cb.push_back(1);
    if constexpr (has_rope != 0) {
        if (do_rope) {
            tmp_cb.push_back(1);
        }
    }
    gamma_cb.pop_front(1);
    scale_cb.pop_front(1);
    CircularBuffer(src_cb_id).pop_front(1);

    // 6. Optional RoPE on trailing rope_tiles cores -> cb_out_rope
    if constexpr (has_rope != 0) {
        if (do_rope) {
            tmp_cb.wait_front(1);
            CircularBuffer cs_cb(cb_cs);
            CircularBuffer out_rope_cb(cb_out_rope);
            cs_cb.wait_front(2);
            out_rope_cb.reserve_back(1);
            reconfig_data_format(cb_tmp, cb_tmp);
            pack_reconfig_data_format(cb_out_rope);
            pack_init(cb_out_rope);
            tile_regs_acquire();
            copy_init(cb_tmp);
            copy_tile(cb_tmp, 0, 0);
            copy_init(cb_cs);
            copy_tile(cb_cs, 0, 1);
            copy_tile(cb_cs, 1, 2);
            rope_sfpu_fused_init();
            rope_sfpu_init();
            rope_sfpu_inplace<1, 1>();
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, cb_out_rope);
            tile_regs_release();
            out_rope_cb.push_back(1);
            tmp_cb.pop_front(1);
            cs_cb.pop_front(2);
        }
    }
}
