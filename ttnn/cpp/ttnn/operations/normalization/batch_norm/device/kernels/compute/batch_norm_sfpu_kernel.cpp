// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "debug/dprint.h"

#include <cstdint>

namespace NAMESPACE {

ALWI void batchnorm_bcast_tiles(
    uint32_t cb_bcast,
    uint32_t cb_other,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t cb_batch_var,
    uint32_t cb_eps,
    uint32_t cb_den,
    uint32_t cb_num,
    uint32_t cb_weight,
    uint32_t cb_bias,
    uint32_t cb_tmp_1,
    uint32_t cb_output_0,
    uint32_t weight_has,
    uint32_t bias_has) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;
    uint32_t weight_has_value = weight_has;
    uint32_t bias_has_value = bias_has;
    auto cb_affine_or_out = (weight_has_value || bias_has_value) ? cb_tmp_1 : cb_output_0;
    auto cb_scaled_output = (bias_has_value) ? cb_tmp_1 : cb_output_0;

    // input - batch_mean
    cb_wait_front(cb_bcast, onetile);
    // DPRINT_UNPACK(DPRINT << "this is the unpack cb_bcast" << ENDL());

    // // DPRINT_MATH(DPRINT << "this is the math kernel - 1" << ENDL());
    // DPRINT_PACK(DPRINT << "this is the pack kernel - 1" << ENDL());
    // DPRINT_UNPACK(DPRINT << "this is the unpack kernel - 1" << ENDL());

    // DPRINT << "cb_bcast : " << ENDL();
    // DPRINT_UNPACK(DPRINT << "this is the unpack kernel" << ENDL());
    // DPRINT << TSLICE(cb_bcast, 0, SliceRange::h0_w0_32()) << ENDL();
    for (uint32_t j = tile_start; j < freq; ++j) {
        cb_wait_front(cb_other, onetile);
        // DPRINT_UNPACK(DPRINT << "   this is the unpack cb_other" << ENDL());
        // DPRINT << "cb_other : " << ENDL();
        // DPRINT << TSLICE(cb_other, 0, SliceRange::h0_w0_32()) << ENDL();

        cb_reserve_back(cb_num, onetile);
        // DPRINT_PACK(DPRINT << "     this is the pack cb_num" << ENDL());

        sub_binary_tile_init();
        tile_regs_acquire();
        tile_regs_wait();
        copy_tile_to_dst_init_short_with_dt(cb_bcast, cb_other);
        for (uint32_t i = 0; i < onetile; ++i) {
            copy_tile(cb_other, i, i * 2);
        }
        copy_tile_to_dst_init_short_with_dt(cb_other, cb_bcast);
        for (uint32_t i = 0; i < onetile; ++i) {
            copy_tile(cb_bcast, i, i * 2 + 1);
            sub_binary_tile(i * 2, i * 2 + 1);
            tile_regs_commit();
            pack_tile(i * 2, cb_num);
        }
        tile_regs_release();
        cb_push_back(cb_num, onetile);
        cb_pop_front(cb_other, onetile);
    }
    // DPRINT << "out of cb_other loop : " << ENDL();
    cb_pop_front(cb_bcast, onetile);
    DPRINT << "doneeeeee : " << ENDL();

    // DPRINT_MATH(DPRINT << "this is the math kernel - 2" << ENDL());
    // DPRINT_PACK(DPRINT << "this is the pack kernel - 2" << ENDL());
    // DPRINT_UNPACK(DPRINT << "this is the unpack kernel - 2" << ENDL());

    // 1/(sqrt(batch_var + eps))
    cb_reserve_back(cb_den, onetile);
    DPRINT_PACK(DPRINT << "this is the pack cb_den" << ENDL());
    cb_wait_front(cb_batch_var, onetile);
    DPRINT_UNPACK(DPRINT << "this is the unpack cb_batch_var" << ENDL());
    // DPRINT << "cb_batch_var : " << ENDL();
    // DPRINT << TSLICE(cb_batch_var, 0, SliceRange::h0_w0_32()) << ENDL();

    // DPRINT << "cb_eps : " << ENDL();
    // DPRINT << TSLICE(cb_eps, 0, SliceRange::h0_w0_32()) << ENDL();

    tile_regs_acquire();
    // tile_regs_wait();

    DPRINT << "wait 1 : " << ENDL();
    copy_tile_to_dst_init_short_with_dt(cb_eps, cb_batch_var);
    for (uint32_t i = 0; i < onetile; ++i) {
        // DPRINT << "wait 1.3 : " << ENDL();
        copy_tile(cb_batch_var, i, i * 2);
        DPRINT << "wait 1.5 : " << ENDL();
    }
    DPRINT << "wait 2 : " << ENDL();
    copy_tile_to_dst_init_short_with_dt(cb_batch_var, cb_eps);
    for (uint32_t i = 0; i < onetile; ++i) {
        copy_tile(cb_eps, i, i * 2 + 1);

        DPRINT << "wait 3 : " << ENDL();
        add_binary_tile_init();
        add_binary_tile(i * 2, i * 2 + 1);
        rsqrt_tile_init();
        rsqrt_tile(i * 2);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(i * 2, cb_den);
    }
    tile_regs_release();

    cb_push_back(cb_den, onetile);
    cb_pop_front(cb_batch_var, onetile);
    // cb_pop_front(cb_eps, onetile);

    // DPRINT_MATH(DPRINT << "this is the math kernel - 3" << ENDL());
    // DPRINT_PACK(DPRINT << "this is the pack kernel - 3" << ENDL());
    // DPRINT_UNPACK(DPRINT << "this is the unpack kernel - 3" << ENDL());

    // (input - batch_mean)/(sqrt(batch_var + eps)) = result
    cb_wait_front(cb_den, onetile);
    DPRINT_UNPACK(DPRINT << "this is the unpack cb_den" << ENDL());
    // DPRINT << "cb_den : " << ENDL();
    // DPRINT << TSLICE(cb_den, 0, SliceRange::h0_w0_32()) << ENDL();
    for (uint32_t j = tile_start; j < freq; ++j) {
        cb_wait_front(cb_num, onetile);
        DPRINT_UNPACK(DPRINT << "this is the unpack cb_num" << ENDL());
        // DPRINT << "cb_num : " << ENDL();
        // DPRINT << TSLICE(cb_num, 0, SliceRange::h0_w0_32()) << ENDL();

        cb_reserve_back(cb_affine_or_out, onetile);
        DPRINT_PACK(DPRINT << "this is the pack cb_affine_or_out" << ENDL());

        mul_binary_tile_init();
        tile_regs_acquire();
        tile_regs_wait();
        copy_tile_to_dst_init_short_with_dt(cb_den, cb_num);
        for (uint32_t i = 0; i < onetile; ++i) {
            copy_tile(cb_num, i, i * 2);
        }
        copy_tile_to_dst_init_short_with_dt(cb_num, cb_den);
        for (uint32_t i = 0; i < onetile; ++i) {
            copy_tile(cb_den, i, i * 2 + 1);
            mul_binary_tile(i * 2, i * 2 + 1);
            tile_regs_commit();
            pack_tile(i * 2, cb_affine_or_out);
        }
        tile_regs_release();
        cb_push_back(cb_affine_or_out, onetile);
        cb_pop_front(cb_num, onetile);
    }
    cb_pop_front(cb_den, onetile);

    // DPRINT_MATH(DPRINT << "this is the math kernel - 4" << ENDL());
    // DPRINT_PACK(DPRINT << "this is the pack kernel - 4" << ENDL());
    // DPRINT_UNPACK(DPRINT << "this is the unpack kernel - 4" << ENDL());

    if (weight_has_value) {  // result = result * weight
        cb_wait_front(cb_weight, onetile);
        // DPRINT << "cb_weight : " << ENDL();
        // DPRINT << TSLICE(cb_weight, 0, SliceRange::h0_w0_32()) << ENDL();
        for (uint32_t j = tile_start; j < freq; ++j) {
            cb_wait_front(cb_affine_or_out, onetile);
            // DPRINT << "cb_affine_or_out : " << ENDL();
            // DPRINT << TSLICE(cb_affine_or_out, 0, SliceRange::h0_w0_32()) << ENDL();

            cb_reserve_back(cb_scaled_output, onetile);

            mul_binary_tile_init();
            tile_regs_acquire();
            tile_regs_wait();
            copy_tile_to_dst_init_short_with_dt(cb_weight, cb_affine_or_out);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_affine_or_out, i, i * 2);
            }
            copy_tile_to_dst_init_short_with_dt(cb_affine_or_out, cb_weight);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_weight, i, i * 2 + 1);
                mul_binary_tile(i * 2, i * 2 + 1);
                tile_regs_commit();
                pack_tile(i * 2, cb_scaled_output);
            }
            tile_regs_release();
            cb_push_back(cb_scaled_output, onetile);
            cb_pop_front(cb_affine_or_out, onetile);
        }
        cb_pop_front(cb_weight, onetile);
    }

    // DPRINT_MATH(DPRINT << "this is the math kernel - 5" << ENDL());
    // DPRINT_PACK(DPRINT << "this is the pack kernel - 5" << ENDL());
    // DPRINT_UNPACK(DPRINT << "this is the unpack kernel - 5" << ENDL());

    if (bias_has_value) {  // result = result + bias
        cb_wait_front(cb_bias, onetile);
        // DPRINT << "cb_bias : " << ENDL();
        // DPRINT << TSLICE(cb_bias, 0, SliceRange::h0_w0_32()) << ENDL();
        for (uint32_t j = tile_start; j < freq; ++j) {
            cb_wait_front(cb_tmp_1, onetile);
            // DPRINT << "cb_tmp_1 : " << ENDL();
            // DPRINT << TSLICE(cb_tmp_1, 0, SliceRange::h0_w0_32()) << ENDL();

            cb_reserve_back(cb_output_0, onetile);

            add_binary_tile_init();
            tile_regs_acquire();
            tile_regs_wait();
            copy_tile_to_dst_init_short_with_dt(cb_bias, cb_tmp_1);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_tmp_1, i, i * 2);
            }
            copy_tile_to_dst_init_short_with_dt(cb_tmp_1, cb_bias);
            for (uint32_t i = 0; i < onetile; ++i) {
                copy_tile(cb_bias, i, i * 2 + 1);
                add_binary_tile(i * 2, i * 2 + 1);
                tile_regs_commit();
                pack_tile(i * 2, cb_output_0);
            }
            tile_regs_release();
            cb_push_back(cb_output_0, onetile);
            cb_pop_front(cb_tmp_1, onetile);
        }
        cb_pop_front(cb_bias, onetile);
    }

    // DPRINT_MATH(DPRINT << "this is the math kernel - 6" << ENDL());
    // DPRINT_PACK(DPRINT << "this is the pack kernel - 6" << ENDL());
    // DPRINT_UNPACK(DPRINT << "this is the unpack kernel - 6" << ENDL());
}

void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    constexpr uint32_t weight_has_value = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t bias_has_value = get_compile_time_arg_val(1) == 1;
    // DPRINT << "compute kernel - 1" << ENDL();
    // DPRINT << "     num_tiles : " << num_tiles << ENDL();
    // DPRINT << "     tile_freq : " << tile_freq << ENDL();
    // DPRINT << "     tile_start : " << tile_start << ENDL();

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_input = get_compile_time_arg_val(2);       // input
    constexpr auto cb_batch_mean = get_compile_time_arg_val(3);  // batch_mean
    constexpr auto cb_output_0 =
        get_compile_time_arg_val(4);  // output -- > [(input - batch_mean)/(sqrt(batch_var + eps))] * weight
    constexpr auto cb_batch_var = get_compile_time_arg_val(5);  // batch_var
    constexpr auto cb_eps = get_compile_time_arg_val(6);        // eps
    constexpr auto cb_den = get_compile_time_arg_val(7);        // 1/(sqrt(batch_var + eps))
    constexpr auto cb_num = get_compile_time_arg_val(8);        // input - batch_mean
    constexpr auto cb_weight = get_compile_time_arg_val(9);     // weight tensor
    constexpr auto cb_tmp_1 = get_compile_time_arg_val(10);     // (input - batch_mean)/(sqrt(batch_var + eps))
    constexpr auto cb_bias = get_compile_time_arg_val(11);      // bias tensor

    auto cb_bcast = cb_batch_mean;
    auto cb_other = cb_input;

    unary_op_init_common(cb_other, cb_output_0);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;
    // DPRINT << "complete_iterations: " << complete_iterations << ENDL();
    // DPRINT << "remaining_iterations: " << remaining_iterations << ENDL();

    cb_wait_front(cb_eps, 1);
    DPRINT_UNPACK(DPRINT << "this is the unpack cb_eps" << ENDL());

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        DPRINT << "     iteration first: " << i << ENDL();
        batchnorm_bcast_tiles(
            cb_bcast,
            cb_other,
            tile_freq,
            tile_start,
            cb_batch_var,
            cb_eps,
            cb_den,
            cb_num,
            cb_weight,
            cb_bias,
            cb_tmp_1,
            cb_output_0,
            weight_has_value,
            bias_has_value);
    }
    if (remaining_iterations > 0) {
        // DPRINT << "     iteration second: " << ENDL();
        batchnorm_bcast_tiles(
            cb_bcast,
            cb_other,
            remaining_iterations,
            tile_start,
            cb_batch_var,
            cb_eps,
            cb_den,
            cb_num,
            cb_weight,
            cb_bias,
            cb_tmp_1,
            cb_output_0,
            weight_has_value,
            bias_has_value);
    }

    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;
    // DPRINT << "compute kernel - 2" << ENDL();
}
}  // namespace NAMESPACE
