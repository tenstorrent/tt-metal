// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/eltwise_unary/rsqrt.h"
#include "experimental/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace {

ALWI void mul_bcast_scalar_to_cb(
    uint32_t icb0,
    uint32_t scalar_cb,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t scalar_itile = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 0) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(scalar_cb, scalar_itile + 1);

    tile_regs_acquire();
    mul_tiles_bcast_scalar_init_short_with_dt(icb0, scalar_cb);
    mul_tiles_bcast_scalar(icb0, scalar_cb, itile0, scalar_itile, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop0) {
        cb_pop_front(icb0, pop0);
    }
    if (pop1) {
        cb_pop_front(scalar_cb, pop1);
    }
    cb_push_back(ocb, onetile);
}

ALWI void add_bcast_scalar_to_cb(
    uint32_t icb0,
    uint32_t scalar_cb,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t scalar_itile = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 0) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(scalar_cb, scalar_itile + 1);

    tile_regs_acquire();
    add_bcast_scalar_init_short_with_dt(icb0, scalar_cb);
    add_tiles_bcast_scalar(icb0, scalar_cb, itile0, scalar_itile, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop0) {
        cb_pop_front(icb0, pop0);
    }
    if (pop1) {
        cb_pop_front(scalar_cb, pop1);
    }
    cb_push_back(ocb, onetile);
}

ALWI void rsqrt_tile_to_cb(uint32_t icb, uint32_t ocb, uint32_t itile = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb, itile + 1);

    tile_regs_acquire();
    copy_tile_init_with_dt(icb);
    copy_tile(icb, itile, dst0);
    rsqrt_tile_init();
    rsqrt_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, ocb);
    tile_regs_release();

    if (pop) {
        cb_pop_front(icb, pop);
    }
    cb_push_back(ocb, onetile);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_input = get_compile_time_arg_val(0);
    constexpr uint32_t cb_output = get_compile_time_arg_val(1);
    constexpr uint32_t cb_sq = get_compile_time_arg_val(2);
    constexpr uint32_t cb_partial = get_compile_time_arg_val(3);
    constexpr uint32_t cb_remote_partials = get_compile_time_arg_val(4);
    constexpr uint32_t cb_unit_scaler = get_compile_time_arg_val(5);
    constexpr uint32_t cb_inv_rms = get_compile_time_arg_val(6);
    constexpr uint32_t cb_mean_scaler = get_compile_time_arg_val(7);
    constexpr uint32_t cb_eps = get_compile_time_arg_val(8);
    constexpr uint32_t cb_tmp = get_compile_time_arg_val(9);
    constexpr uint32_t shard_h_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t shard_w_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t num_cols = get_compile_time_arg_val(12);

    const uint32_t total_shard_tiles = shard_h_tiles * shard_w_tiles;

    compute_kernel_hw_startup(cb_input, cb_unit_scaler, cb_output);

    experimental::CircularBuffer input_cb(cb_input);
    experimental::CircularBuffer remote_partials_cb(cb_remote_partials);
    experimental::CircularBuffer unit_scaler_cb(cb_unit_scaler);
    experimental::CircularBuffer mean_scaler_cb(cb_mean_scaler);
    experimental::CircularBuffer eps_cb(cb_eps);
    experimental::CircularBuffer inv_rms_cb(cb_inv_rms);

    input_cb.wait_front(total_shard_tiles);
    unit_scaler_cb.wait_front(1);
    mean_scaler_cb.wait_front(1);
    eps_cb.wait_front(1);

    for (uint32_t row = 0; row < shard_h_tiles; ++row) {
        const uint32_t row_offset = row * shard_w_tiles;

        for (uint32_t col = 0; col < shard_w_tiles; ++col) {
            mul_tiles_to_cb(cb_input, cb_input, cb_sq, row_offset + col, row_offset + col, 0, 0);
        }

        compute_kernel_lib::reduce<SUM, REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
            cb_sq, cb_unit_scaler, cb_partial, compute_kernel_lib::ReduceInputBlockShape::row(shard_w_tiles));

        remote_partials_cb.wait_front(num_cols);
        compute_kernel_lib::reduce<SUM, REDUCE_ROW, compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop>(
            cb_remote_partials, cb_unit_scaler, cb_tmp, compute_kernel_lib::ReduceInputBlockShape::row(num_cols));

        mul_bcast_scalar_to_cb(cb_tmp, cb_mean_scaler, cb_inv_rms, 0, 0, 1, 0);
        add_bcast_scalar_to_cb(cb_inv_rms, cb_eps, cb_tmp, 0, 0, 1, 0);
        rsqrt_tile_to_cb(cb_tmp, cb_inv_rms, 0, 1);

        for (uint32_t col = 0; col < shard_w_tiles; ++col) {
            mul_tiles_bcast_cols_to_cb(cb_input, cb_inv_rms, cb_output, row_offset + col, 0, 0, 0);
        }

        inv_rms_cb.pop_front(1);
    }

    input_cb.pop_front(total_shard_tiles);
    unit_scaler_cb.pop_front(1);
    mean_scaler_cb.pop_front(1);
    eps_cb.pop_front(1);
}
