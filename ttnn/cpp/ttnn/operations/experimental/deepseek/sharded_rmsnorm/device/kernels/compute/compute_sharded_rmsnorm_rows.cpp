// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Several-row RMSNorm on (h x 32) tiles: one row per tile row, so every reduction is per row.
//   1. E = sum_t x_t * x_t elementwise (ELWMUL accumulates into DEST): row j of E holds row j's partial
//      sums of x^2, one per column.
//   2. Hub: sum_c E_c @ ones[32, 32] sums the cores' tiles and each row's 32 columns at once, so every
//      column of row j holds row j's sum(x^2); rsqrt(sum / D + eps) on that tile is the per-row scale tile.
//   3. y_t = (x_t * scale) * gamma_t elementwise.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/experimental/add_rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t cb_x_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_gamma_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_local_stat_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_gather_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_ones_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_scale_src_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scale_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(7);
    constexpr uint32_t w_tiles = get_compile_time_arg_val(8);
    constexpr uint32_t num_cores = get_compile_time_arg_val(9);
    constexpr uint32_t inv_d_bits = get_compile_time_arg_val(10);
    constexpr uint32_t eps_bits = get_compile_time_arg_val(11);
    // 32-bit DEST with SyncHalf holds 4 tiles.
    constexpr uint32_t kChunk = 4;

    const bool is_hub = get_arg_val<uint32_t>(0) != 0;

    CircularBuffer cb_x(cb_x_id);
    CircularBuffer cb_gamma(cb_gamma_id);
    CircularBuffer cb_local_stat(cb_local_stat_id);
    CircularBuffer cb_gather(cb_gather_id);
    CircularBuffer cb_ones(cb_ones_id);
    CircularBuffer cb_scale_src(cb_scale_src_id);
    CircularBuffer cb_scale(cb_scale_id);
    CircularBuffer cb_out(cb_out_id);

    compute_kernel_hw_startup(cb_x_id, cb_x_id, cb_local_stat_id);

    // 1. E = sum_t x_t^2 (elementwise, fp32 DEST).
    cb_x.wait_front(w_tiles);
    cb_local_stat.reserve_back(1);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 0.0F);
    mul_init(cb_x_id, cb_x_id);
    for (uint32_t t = 0; t < w_tiles; ++t) {
        mul_tiles(cb_x_id, cb_x_id, t, t, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_local_stat_id);
    tile_regs_release();
    cb_local_stat.push_back(1);

    // 2. Hub: the per-row scale tile.
    if (is_hub) {
        cb_gather.wait_front(num_cores);
        cb_ones.wait_front(1);
        cb_scale_src.reserve_back(1);
        reconfig_data_format(cb_ones_id, cb_gather_id);
        pack_reconfig_data_format(cb_scale_src_id);
        tile_regs_acquire();
        fill_tile_init();
        fill_tile(0, 0.0F);
        matmul_init(cb_gather_id, cb_ones_id);
        for (uint32_t c = 0; c < num_cores; ++c) {
            matmul_tiles(cb_gather_id, cb_ones_id, c, 0, 0);
        }
        add_rsqrt_tile_init();
        add_rsqrt_tile<false, VectorMode::R, 8, false, inv_d_bits>(0, eps_bits);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_scale_src_id);
        tile_regs_release();
        cb_scale_src.push_back(1);
        cb_gather.pop_front(num_cores);
        cb_ones.pop_front(1);
    }

    // 3. y = x * scale * gamma, in chunks of <= 4 tiles.
    cb_scale.wait_front(1);
    cb_gamma.wait_front(w_tiles);
    cb_out.reserve_back(w_tiles);
    pack_reconfig_data_format(cb_out_id);
    for (uint32_t t = 0; t < w_tiles; t += kChunk) {
        const uint32_t n = w_tiles - t < kChunk ? w_tiles - t : kChunk;
        tile_regs_acquire();
        reconfig_data_format(cb_x_id, cb_scale_id);
        mul_init(cb_x_id, cb_scale_id);
        for (uint32_t k = 0; k < n; ++k) {
            mul_tiles(cb_x_id, cb_scale_id, t + k, 0, k);
        }
        reconfig_data_format(cb_gamma_id, cb_gamma_id);
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gamma_id);
        for (uint32_t k = 0; k < n; ++k) {
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_gamma_id, t + k, k);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t k = 0; k < n; ++k) {
            pack_tile<true>(k, cb_out_id, t + k);
        }
        tile_regs_release();
    }
    cb_out.push_back(w_tiles);
    cb_scale.pop_front(1);
    cb_gamma.pop_front(w_tiles);
    cb_x.pop_front(w_tiles);
}
