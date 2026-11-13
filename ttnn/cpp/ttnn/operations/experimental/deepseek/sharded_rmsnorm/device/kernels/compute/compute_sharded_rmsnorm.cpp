// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/experimental/add_rsqrt.h"
#include "api/compute/experimental/mul_reduce_scalar.h"
#include "api/compute/experimental/pack_block.h"
#include "api/compute/experimental/rmsnorm.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

namespace {

using namespace ckernel;

// Scales src[src_start .. src_start + num_tiles) by scalar scale[0][0] and gamma[gamma_start .. + num_tiles),
// packing into out_cb_id. Caller holds SyncHalf with fp32_dest_acc_en = true (num_tiles <= 4).
template <uint32_t num_tiles>
ALWI void scale_rmsnorm_chunk(
    uint32_t src_cb_id,
    uint32_t src_start,
    uint32_t scale_cb_id,
    uint32_t gamma_cb_id,
    uint32_t gamma_start,
    uint32_t out_cb_id) {
    reconfig_data_format(scale_cb_id, scale_cb_id);
    tile_regs_acquire();
    copy_init(scale_cb_id);
    copy_tile(scale_cb_id, 0, 0);
    reconfig_data_format_srca(scale_cb_id, src_cb_id);
    rmsnorm_bcast_scalar_reuse_tiles_init_fidelity<EltwiseBinaryType::ELWMUL, num_tiles, MathFidelity::HiFi4>(
        src_cb_id);
    rmsnorm_bcast_scalar_reuse_tiles_fidelity<EltwiseBinaryType::ELWMUL, num_tiles, MathFidelity::HiFi4, true>(
        src_cb_id, src_start, 0, 0);
    reconfig_data_format(gamma_cb_id, gamma_cb_id);
    mul_reuse_dest_init_fidelity<EltwiseBinaryReuseDestType::DEST_TO_SRCA, MathFidelity::HiFi4>(gamma_cb_id);
    for (uint32_t k = 0; k < num_tiles; ++k) {
        mul_reuse_dest_tiles_fidelity<EltwiseBinaryReuseDestType::DEST_TO_SRCA, MathFidelity::HiFi4>(
            gamma_cb_id, gamma_start + k, k);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_block_contiguous(0, out_cb_id, num_tiles);
    tile_regs_release();
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_local_stat_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_gather_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_scale_src_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_scale_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(6);
    constexpr uint32_t w_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t num_cores = get_compile_time_arg_val(8);
    constexpr uint32_t inv_d_bits = get_compile_time_arg_val(9);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(10);

    const bool is_hub = get_arg_val<uint32_t>(0) != 0;

    CircularBuffer cb_x(cb_x_id);
    CircularBuffer cb_gamma(cb_gamma_id);
    CircularBuffer cb_local_stat(cb_local_stat_id);
    CircularBuffer cb_gather(cb_gather_id);
    CircularBuffer cb_scale_src(cb_scale_src_id);
    CircularBuffer cb_scale(cb_scale_id);
    CircularBuffer cb_out(cb_out_id);

    compute_kernel_hw_startup(cb_x_id, cb_x_id, cb_local_stat_id);

    // 1. Reduce sum(x^2) over this core's w_tiles 1x32 tiles into cb_local_stat[0][0] (Float32).
    cb_x.wait_front(w_tiles);
    cb_local_stat.reserve_back(1);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.0F);
    mul_reduce_scalar_init_fidelity<MathFidelity::HiFi4>(cb_x_id, cb_x_id);
    mul_reduce_scalar_tile_fidelity<MathFidelity::HiFi4, MathFidelity::HiFi4, true>(
        cb_x_id, cb_x_id, cb_local_stat_id, w_tiles, 1.0F, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_local_stat_id);
    mul_reduce_scalar_uninit();
    tile_regs_release();
    cb_local_stat.push_back(1);

    // 2. Hub sums all N cores' partial sum(x^2) in FP32 and computes rsqrt(sum * (1/D) + eps).
    if (is_hub) {
        cb_gather.wait_front(num_cores);
        cb_scale_src.reserve_back(1);
        reconfig_data_format(cb_gather_id, cb_gather_id);
        pack_reconfig_data_format(cb_scale_src_id);
        tile_regs_acquire();
        copy_init(cb_gather_id);
        copy_tile(cb_gather_id, 0, 0);
        for (uint32_t c = 1; c < num_cores; ++c) {
            copy_init(cb_gather_id);
            copy_tile(cb_gather_id, c, 1);
            add_binary_tile_init();
            add_binary_tile(0, 1, 0);
        }
        add_rsqrt_tile_init();
        add_rsqrt_tile<false, VectorMode::RC_custom, 1, false, inv_d_bits>(0, eps_bits);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_scale_src_id);
        tile_regs_release();
        cb_scale_src.push_back(1);
        cb_gather.pop_front(num_cores);
    }

    // 3. Scale x * rsqrt_rms * gamma in chunks of <= 4 tiles (32-bit SyncHalf DST capacity).
    cb_scale.wait_front(1);
    cb_gamma.wait_front(w_tiles);
    cb_out.reserve_back(w_tiles);
    pack_reconfig_data_format(cb_out_id);
    pack_block_contiguous_init(cb_out_id);

    for (uint32_t t = 0; t < w_tiles;) {
        const uint32_t rem = w_tiles - t;
        if (rem >= 4) {
            scale_rmsnorm_chunk<4>(cb_x_id, t, cb_scale_id, cb_gamma_id, t, cb_out_id);
            t += 4;
        } else if (rem == 3) {
            scale_rmsnorm_chunk<3>(cb_x_id, t, cb_scale_id, cb_gamma_id, t, cb_out_id);
            t += 3;
        } else if (rem == 2) {
            scale_rmsnorm_chunk<2>(cb_x_id, t, cb_scale_id, cb_gamma_id, t, cb_out_id);
            t += 2;
        } else {
            scale_rmsnorm_chunk<1>(cb_x_id, t, cb_scale_id, cb_gamma_id, t, cb_out_id);
            t += 1;
        }
    }

    cb_out.push_back(w_tiles);
    cb_scale.pop_front(1);
    cb_gamma.pop_front(w_tiles);
    cb_x.pop_front(w_tiles);
}
