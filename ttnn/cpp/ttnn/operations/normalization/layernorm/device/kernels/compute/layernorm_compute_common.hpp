#pragma once

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

ALWI void layernorm_copy_stage_kernel_main() {
    int i = 0;
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(i++);
    const uint32_t Wt = get_arg_val<uint32_t>(i++);
    const uint32_t logical_W = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_mask_w = tt::CBIndex::c_8;
    constexpr uint32_t cb_output = tt::CBIndex::c_16;
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t tile_width = 32;

    unary_op_init_common(cb_input, cb_output);

    const bool do_mask_w = (logical_W % tile_width) != 0;
    if (do_mask_w) {
        cb_wait_front(cb_mask_w, one_tile);
    }

    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
            cb_wait_front(cb_input, one_tile);

            tile_regs_acquire();
            copy_tile_init_with_dt(cb_input);
            copy_tile(cb_input, 0, dst0);

            if (do_mask_w && (col_idx == Wt - 1)) {
                copy_tile_init_with_dt(cb_mask_w);
                copy_tile(cb_mask_w, 0, dst1);
                mask_tile_init();
                mask_tile(dst0, dst1);
            }
            tile_regs_commit();

            cb_pop_front(cb_input, one_tile);
            cb_reserve_back(cb_output, one_tile);
            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_output);
            tile_regs_release();

            cb_push_back(cb_output, one_tile);
        }
    }

    if (do_mask_w) {
        cb_pop_front(cb_mask_w, one_tile);
    }
}

ALWI void layernorm_mean_reduce_stage_kernel_main() {
    int i = 0;
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(i++);
    const uint32_t Wt = get_arg_val<uint32_t>(i++);
    const uint32_t logical_W = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_6;
    constexpr uint32_t cb_mask_w = tt::CBIndex::c_8;
    constexpr uint32_t cb_output = tt::CBIndex::c_16;
    constexpr uint32_t cb_accum = tt::CBIndex::c_26;
    constexpr uint32_t cb_masked_input = tt::CBIndex::c_29;
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t tile_width = 32;

    binary_op_init_common(cb_input, cb_input, cb_output);

    const bool do_mask_w = (logical_W % tile_width) != 0;
    cb_wait_front(cb_reduce_scaler, one_tile);
    if (do_mask_w) {
        cb_wait_front(cb_mask_w, one_tile);
    }

    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        const bool is_w_single_tile = (Wt == 1);
        uint32_t reduce_input_cb = cb_input;

        if (!is_w_single_tile) {
            tile_regs_acquire();
            for (uint32_t wt = 0; wt < Wt - 1; ++wt) {
                cb_wait_front(cb_input, one_tile);
#if defined FP32_DEST_ACC_EN
                reconfig_data_format(cb_input, cb_reduce_scaler);
#endif
                mm_init_short(cb_input, cb_reduce_scaler, false);
                matmul_tiles(cb_input, cb_reduce_scaler, 0, 0, dst0);
                cb_pop_front(cb_input, one_tile);
            }
            tile_regs_commit();

            cb_reserve_back(cb_accum, one_tile);
            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_accum);
            tile_regs_release();
            cb_push_back(cb_accum, one_tile);
        }

        if (do_mask_w) {
            tile_regs_acquire();
            cb_wait_front(cb_input, one_tile);
            copy_tile_init_with_dt(cb_input);
            copy_tile(cb_input, 0, dst0);

            copy_tile_init_with_dt(cb_mask_w);
            copy_tile(cb_mask_w, 0, dst1);
            mask_tile_init();
            mask_tile(dst0, dst1);
            tile_regs_commit();

            cb_reserve_back(cb_masked_input, one_tile);
            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_masked_input);
            tile_regs_release();
            cb_push_back(cb_masked_input, one_tile);

            cb_pop_front(cb_input, one_tile);
            reduce_input_cb = cb_masked_input;
        }

        tile_regs_acquire();
        cb_wait_front(reduce_input_cb, one_tile);
        if (!is_w_single_tile) {
            cb_wait_front(cb_accum, one_tile);
            copy_tile_init_with_dt(cb_accum);
            copy_tile(cb_accum, 0, dst0);
        }
#if defined FP32_DEST_ACC_EN
        reconfig_data_format(reduce_input_cb, cb_reduce_scaler);
#endif
        mm_init_short(reduce_input_cb, cb_reduce_scaler, false);
        matmul_tiles(reduce_input_cb, cb_reduce_scaler, 0, 0, dst0);
        tile_regs_commit();

        cb_reserve_back(cb_output, one_tile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_output);
        tile_regs_release();
        cb_push_back(cb_output, one_tile);

        cb_pop_front(reduce_input_cb, one_tile);
        if (!is_w_single_tile) {
            cb_pop_front(cb_accum, one_tile);
        }
    }

    if (do_mask_w) {
        cb_pop_front(cb_mask_w, one_tile);
    }
    cb_pop_front(cb_reduce_scaler, one_tile);
}

ALWI void accumulate_row_mean_tile(
    uint32_t src_cb, uint32_t scaler_cb, uint32_t accum_cb, uint32_t dst_idx, bool has_prior_accum) {
    tile_regs_acquire();
    cb_wait_front(src_cb, 1);
    if (has_prior_accum) {
        cb_wait_front(accum_cb, 1);
        copy_tile_init_with_dt(accum_cb);
        copy_tile(accum_cb, 0, dst_idx);
        cb_pop_front(accum_cb, 1);
    }
#if defined FP32_DEST_ACC_EN
    reconfig_data_format(src_cb, scaler_cb);
#endif
    mm_init_short(src_cb, scaler_cb, false);
    matmul_tiles(src_cb, scaler_cb, 0, 0, dst_idx);
    tile_regs_commit();

    cb_reserve_back(accum_cb, 1);
    tile_regs_wait();
    pack_tile_with_dt(dst_idx, accum_cb);
    tile_regs_release();
    cb_push_back(accum_cb, 1);
}

ALWI void layernorm_invstd_reduce_stage_kernel_main() {
    int i = 0;
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(i++);
    const uint32_t Wt = get_arg_val<uint32_t>(i++);
    const uint32_t logical_W = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_6;
    constexpr uint32_t cb_epsilon = tt::CBIndex::c_7;
    constexpr uint32_t cb_mask_w = tt::CBIndex::c_8;
    constexpr uint32_t cb_output = tt::CBIndex::c_16;
    constexpr uint32_t cb_square_tile = tt::CBIndex::c_25;
    constexpr uint32_t cb_mean_accum = tt::CBIndex::c_26;
    constexpr uint32_t cb_sqmean_accum = tt::CBIndex::c_27;
    constexpr uint32_t cb_invstd = tt::CBIndex::c_28;
    constexpr uint32_t cb_masked_input = tt::CBIndex::c_29;
    constexpr uint32_t cb_variance = tt::CBIndex::c_30;
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t tile_width = 32;

    binary_op_init_common(cb_input, cb_input, cb_output);

    const bool do_mask_w = (logical_W % tile_width) != 0;
    cb_wait_front(cb_reduce_scaler, one_tile);
    cb_wait_front(cb_epsilon, one_tile);
    if (do_mask_w) {
        cb_wait_front(cb_mask_w, one_tile);
    }

    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
            const bool is_last_tile = col_idx == (Wt - 1);
            const bool has_prior_accum = col_idx > 0;
            uint32_t reduce_input_cb = cb_input;

            if (do_mask_w && is_last_tile) {
                tile_regs_acquire();
                cb_wait_front(cb_input, one_tile);
                copy_tile_init_with_dt(cb_input);
                copy_tile(cb_input, 0, dst0);

                copy_tile_init_with_dt(cb_mask_w);
                copy_tile(cb_mask_w, 0, dst1);
                mask_tile_init();
                mask_tile(dst0, dst1);
                tile_regs_commit();

                cb_reserve_back(cb_masked_input, one_tile);
                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_masked_input);
                tile_regs_release();
                cb_push_back(cb_masked_input, one_tile);

                cb_pop_front(cb_input, one_tile);
                reduce_input_cb = cb_masked_input;
            }

            accumulate_row_mean_tile(reduce_input_cb, cb_reduce_scaler, cb_mean_accum, dst0, has_prior_accum);

            tile_regs_acquire();
            cb_wait_front(reduce_input_cb, one_tile);
            copy_tile_init_with_dt(reduce_input_cb);
            copy_tile(reduce_input_cb, 0, dst1);
            square_tile_init();
            square_tile(dst1);
            tile_regs_commit();

            cb_reserve_back(cb_square_tile, one_tile);
            tile_regs_wait();
            pack_tile_with_dt(dst1, cb_square_tile);
            tile_regs_release();
            cb_push_back(cb_square_tile, one_tile);

            cb_pop_front(reduce_input_cb, one_tile);

            accumulate_row_mean_tile(cb_square_tile, cb_reduce_scaler, cb_sqmean_accum, dst0, has_prior_accum);
            cb_pop_front(cb_square_tile, one_tile);
        }

        tile_regs_acquire();
        cb_wait_front(cb_mean_accum, one_tile);
        copy_tile_init_with_dt(cb_mean_accum);
        copy_tile(cb_mean_accum, 0, dst0);
        square_tile_init();
        square_tile(dst0);
        tile_regs_commit();

        cb_reserve_back(cb_masked_input, one_tile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_masked_input);
        tile_regs_release();
        cb_push_back(cb_masked_input, one_tile);

        tile_regs_acquire();
        cb_wait_front(cb_sqmean_accum, one_tile);
        cb_wait_front(cb_masked_input, one_tile);
        sub_tiles_init_with_dt(cb_sqmean_accum, cb_masked_input);
        sub_tiles(cb_sqmean_accum, cb_masked_input, 0, 0, dst0);
        tile_regs_commit();

        cb_reserve_back(cb_variance, one_tile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_variance);
        tile_regs_release();
        cb_push_back(cb_variance, one_tile);

        cb_pop_front(cb_masked_input, one_tile);

        tile_regs_acquire();
        cb_wait_front(cb_variance, one_tile);
        cb_wait_front(cb_epsilon, one_tile);
        add_tiles_init_with_dt(cb_variance, cb_epsilon);
        add_tiles(cb_variance, cb_epsilon, 0, 0, dst0);
        rsqrt_tile_init();
        rsqrt_tile(dst0);
        tile_regs_commit();

        cb_reserve_back(cb_invstd, one_tile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_invstd);
        tile_regs_release();
        cb_push_back(cb_invstd, one_tile);

        cb_pop_front(cb_variance, one_tile);
        cb_pop_front(cb_mean_accum, one_tile);
        cb_pop_front(cb_sqmean_accum, one_tile);

        tile_regs_acquire();
        cb_wait_front(cb_invstd, one_tile);
        copy_tile_init_with_dt(cb_invstd);
        copy_tile(cb_invstd, 0, dst0);
        tile_regs_commit();

        cb_reserve_back(cb_output, one_tile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_output);
        tile_regs_release();
        cb_push_back(cb_output, one_tile);

        cb_pop_front(cb_invstd, one_tile);
    }

    if (do_mask_w) {
        cb_pop_front(cb_mask_w, one_tile);
    }
    cb_pop_front(cb_epsilon, one_tile);
    cb_pop_front(cb_reduce_scaler, one_tile);
}

ALWI void layernorm_normalize_stage_kernel_main() {
    int i = 0;
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(i++);
    const uint32_t Wt = get_arg_val<uint32_t>(i++);
    const uint32_t logical_W = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_input_pass0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_input_pass1 = tt::CBIndex::c_2;
    constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_6;
    constexpr uint32_t cb_epsilon = tt::CBIndex::c_7;
    constexpr uint32_t cb_mask_w = tt::CBIndex::c_8;
    constexpr uint32_t cb_output = tt::CBIndex::c_16;
    constexpr uint32_t cb_square_tile = tt::CBIndex::c_25;
    constexpr uint32_t cb_mean_accum = tt::CBIndex::c_26;
    constexpr uint32_t cb_sqmean_accum = tt::CBIndex::c_27;
    constexpr uint32_t cb_invstd = tt::CBIndex::c_28;
    constexpr uint32_t cb_masked_input = tt::CBIndex::c_29;
    constexpr uint32_t cb_variance = tt::CBIndex::c_30;
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t tile_width = 32;

    binary_op_init_common(cb_input_pass0, cb_input_pass0, cb_output);

    const bool do_mask_w = (logical_W % tile_width) != 0;
    cb_wait_front(cb_reduce_scaler, one_tile);
    cb_wait_front(cb_epsilon, one_tile);
    if (do_mask_w) {
        cb_wait_front(cb_mask_w, one_tile);
    }

    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
            const bool is_last_tile = col_idx == (Wt - 1);
            const bool has_prior_accum = col_idx > 0;
            uint32_t reduce_input_cb = cb_input_pass0;

            if (do_mask_w && is_last_tile) {
                mask_tile_to_cb(cb_input_pass0, cb_mask_w, cb_masked_input, 0, 0, 1, 0);
                reduce_input_cb = cb_masked_input;
            }

            accumulate_row_mean_tile(reduce_input_cb, cb_reduce_scaler, cb_mean_accum, dst0, has_prior_accum);

            tile_regs_acquire();
            cb_wait_front(reduce_input_cb, one_tile);
            copy_tile_init_with_dt(reduce_input_cb);
            copy_tile(reduce_input_cb, 0, dst1);
            square_tile_init();
            square_tile(dst1);
            tile_regs_commit();

            cb_reserve_back(cb_square_tile, one_tile);
            tile_regs_wait();
            pack_tile_with_dt(dst1, cb_square_tile);
            tile_regs_release();
            cb_push_back(cb_square_tile, one_tile);

            cb_pop_front(reduce_input_cb, one_tile);

            accumulate_row_mean_tile(cb_square_tile, cb_reduce_scaler, cb_sqmean_accum, dst0, has_prior_accum);
            cb_pop_front(cb_square_tile, one_tile);
        }

        tile_regs_acquire();
        cb_wait_front(cb_mean_accum, one_tile);
        copy_tile_init_with_dt(cb_mean_accum);
        copy_tile(cb_mean_accum, 0, dst0);
        square_tile_init();
        square_tile(dst0);
        tile_regs_commit();

        cb_reserve_back(cb_masked_input, one_tile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_masked_input);
        tile_regs_release();
        cb_push_back(cb_masked_input, one_tile);

        tile_regs_acquire();
        cb_wait_front(cb_sqmean_accum, one_tile);
        cb_wait_front(cb_masked_input, one_tile);
        sub_tiles_init_with_dt(cb_sqmean_accum, cb_masked_input);
        sub_tiles(cb_sqmean_accum, cb_masked_input, 0, 0, dst0);
        tile_regs_commit();

        cb_reserve_back(cb_variance, one_tile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_variance);
        tile_regs_release();
        cb_push_back(cb_variance, one_tile);

        cb_pop_front(cb_masked_input, one_tile);

        tile_regs_acquire();
        cb_wait_front(cb_variance, one_tile);
        cb_wait_front(cb_epsilon, one_tile);
        add_tiles_init_with_dt(cb_variance, cb_epsilon);
        add_tiles(cb_variance, cb_epsilon, 0, 0, dst0);
        rsqrt_tile_init();
        rsqrt_tile(dst0);
        tile_regs_commit();

        cb_reserve_back(cb_invstd, one_tile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_invstd);
        tile_regs_release();
        cb_push_back(cb_invstd, one_tile);

        cb_pop_front(cb_variance, one_tile);
        cb_pop_front(cb_sqmean_accum, one_tile);

        for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
            const bool is_last_tile = col_idx == (Wt - 1);
            uint32_t centered_input_cb = cb_input_pass1;

            if (do_mask_w && is_last_tile) {
                mask_tile_to_cb(cb_input_pass1, cb_mask_w, cb_masked_input, 0, 0, 1, 0);
                centered_input_cb = cb_masked_input;
            }

            sub_tiles_bcast_cols_to_cb(centered_input_cb, cb_mean_accum, cb_variance, 0, 0, 1, 0);
            mul_tiles_bcast_cols_to_cb(cb_variance, cb_invstd, cb_masked_input, 0, 0, 1, 0);

            if (do_mask_w && is_last_tile) {
                mask_tile_to_cb(cb_masked_input, cb_mask_w, cb_output, 0, 0, 1, 0);
            } else {
                copy_tile_to_cb(cb_masked_input, cb_output);
            }
        }

        cb_pop_front(cb_mean_accum, one_tile);
        cb_pop_front(cb_invstd, one_tile);
    }

    if (do_mask_w) {
        cb_pop_front(cb_mask_w, one_tile);
    }
    cb_pop_front(cb_epsilon, one_tile);
    cb_pop_front(cb_reduce_scaler, one_tile);
}

ALWI void layernorm_residual_affine_stage_kernel_main() {
    constexpr bool has_residual = get_compile_time_arg_val(0) == 1;
    constexpr bool has_weight = get_compile_time_arg_val(1) == 1;
    constexpr bool has_bias = get_compile_time_arg_val(2) == 1;

    int i = 0;
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(i++);
    const uint32_t Wt = get_arg_val<uint32_t>(i++);
    const uint32_t logical_W = get_arg_val<uint32_t>(i++);

    constexpr uint32_t cb_input_pass0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_residual_pass0 = tt::CBIndex::c_1;
    constexpr uint32_t cb_input_pass1 = tt::CBIndex::c_2;
    constexpr uint32_t cb_residual_pass1 = tt::CBIndex::c_3;
    constexpr uint32_t cb_weight = tt::CBIndex::c_4;
    constexpr uint32_t cb_bias = tt::CBIndex::c_5;
    constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_6;
    constexpr uint32_t cb_epsilon = tt::CBIndex::c_7;
    constexpr uint32_t cb_mask_w = tt::CBIndex::c_8;
    constexpr uint32_t cb_output = tt::CBIndex::c_16;
    constexpr uint32_t cb_effective = tt::CBIndex::c_24;
    constexpr uint32_t cb_square_tile = tt::CBIndex::c_25;
    constexpr uint32_t cb_mean_accum = tt::CBIndex::c_26;
    constexpr uint32_t cb_sqmean_accum = tt::CBIndex::c_27;
    constexpr uint32_t cb_invstd = tt::CBIndex::c_28;
    constexpr uint32_t cb_scratch0 = tt::CBIndex::c_29;
    constexpr uint32_t cb_scratch1 = tt::CBIndex::c_30;
    constexpr uint32_t one_tile = 1;
    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t tile_width = 32;

    binary_op_init_common(cb_input_pass0, cb_input_pass0, cb_output);

    const bool do_mask_w = (logical_W % tile_width) != 0;
    cb_wait_front(cb_reduce_scaler, one_tile);
    cb_wait_front(cb_epsilon, one_tile);
    if (do_mask_w) {
        cb_wait_front(cb_mask_w, one_tile);
    }

    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
            const bool is_last_tile = col_idx == (Wt - 1);
            const bool has_prior_accum = col_idx > 0;
            uint32_t reduce_input_cb = cb_input_pass0;

            if constexpr (has_residual) {
                add_tiles_to_cb(cb_input_pass0, cb_residual_pass0, cb_effective);
                reduce_input_cb = cb_effective;
            }

            if (do_mask_w && is_last_tile) {
                mask_tile_to_cb(reduce_input_cb, cb_mask_w, cb_scratch0, 0, 0, 1, 0);
                reduce_input_cb = cb_scratch0;
            }

            accumulate_row_mean_tile(reduce_input_cb, cb_reduce_scaler, cb_mean_accum, dst0, has_prior_accum);

            tile_regs_acquire();
            cb_wait_front(reduce_input_cb, one_tile);
            copy_tile_init_with_dt(reduce_input_cb);
            copy_tile(reduce_input_cb, 0, dst1);
            square_tile_init();
            square_tile(dst1);
            tile_regs_commit();

            cb_reserve_back(cb_square_tile, one_tile);
            tile_regs_wait();
            pack_tile_with_dt(dst1, cb_square_tile);
            tile_regs_release();
            cb_push_back(cb_square_tile, one_tile);

            cb_pop_front(reduce_input_cb, one_tile);

            accumulate_row_mean_tile(cb_square_tile, cb_reduce_scaler, cb_sqmean_accum, dst0, has_prior_accum);
            cb_pop_front(cb_square_tile, one_tile);
        }

        tile_regs_acquire();
        cb_wait_front(cb_mean_accum, one_tile);
        copy_tile_init_with_dt(cb_mean_accum);
        copy_tile(cb_mean_accum, 0, dst0);
        square_tile_init();
        square_tile(dst0);
        tile_regs_commit();

        cb_reserve_back(cb_scratch0, one_tile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_scratch0);
        tile_regs_release();
        cb_push_back(cb_scratch0, one_tile);

        tile_regs_acquire();
        cb_wait_front(cb_sqmean_accum, one_tile);
        cb_wait_front(cb_scratch0, one_tile);
        sub_tiles_init_with_dt(cb_sqmean_accum, cb_scratch0);
        sub_tiles(cb_sqmean_accum, cb_scratch0, 0, 0, dst0);
        tile_regs_commit();

        cb_reserve_back(cb_scratch1, one_tile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_scratch1);
        tile_regs_release();
        cb_push_back(cb_scratch1, one_tile);

        cb_pop_front(cb_scratch0, one_tile);

        tile_regs_acquire();
        cb_wait_front(cb_scratch1, one_tile);
        cb_wait_front(cb_epsilon, one_tile);
        add_tiles_init_with_dt(cb_scratch1, cb_epsilon);
        add_tiles(cb_scratch1, cb_epsilon, 0, 0, dst0);
        rsqrt_tile_init();
        rsqrt_tile(dst0);
        tile_regs_commit();

        cb_reserve_back(cb_invstd, one_tile);
        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_invstd);
        tile_regs_release();
        cb_push_back(cb_invstd, one_tile);

        cb_pop_front(cb_scratch1, one_tile);
        cb_pop_front(cb_sqmean_accum, one_tile);

        for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
            const bool is_last_tile = col_idx == (Wt - 1);
            uint32_t centered_input_cb = cb_input_pass1;
            uint32_t post_affine_cb = cb_scratch0;

            if constexpr (has_residual) {
                add_tiles_to_cb(cb_input_pass1, cb_residual_pass1, cb_effective);
                centered_input_cb = cb_effective;
            }

            if (do_mask_w && is_last_tile) {
                mask_tile_to_cb(centered_input_cb, cb_mask_w, cb_scratch0, 0, 0, 1, 0);
                centered_input_cb = cb_scratch0;
            }

            sub_tiles_bcast_cols_to_cb(centered_input_cb, cb_mean_accum, cb_scratch1, 0, 0, 1, 0);
            mul_tiles_bcast_cols_to_cb(cb_scratch1, cb_invstd, cb_scratch0, 0, 0, 1, 0);

            if constexpr (has_weight) {
                mul_tiles_bcast_rows_to_cb(cb_scratch0, cb_weight, cb_effective, 0, 0, 1, 1);
                post_affine_cb = cb_effective;
            }

            if constexpr (has_bias) {
                tile_regs_acquire();
                cb_wait_front(post_affine_cb, one_tile);
                cb_wait_front(cb_bias, one_tile);
#if defined FP32_DEST_ACC_EN
                reconfig_data_format(post_affine_cb, cb_bias);
#endif
                add_bcast_rows_init_short(post_affine_cb, cb_bias);
                add_tiles_bcast_rows(post_affine_cb, cb_bias, 0, 0, dst0);
                tile_regs_commit();

                cb_reserve_back(cb_scratch1, one_tile);
                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_scratch1);
                tile_regs_release();
                cb_push_back(cb_scratch1, one_tile);

                cb_pop_front(post_affine_cb, one_tile);
                cb_pop_front(cb_bias, one_tile);
                post_affine_cb = cb_scratch1;
            }

            if (do_mask_w && is_last_tile) {
                mask_tile_to_cb(post_affine_cb, cb_mask_w, cb_output, 0, 0, 1, 0);
            } else {
                copy_tile_to_cb(post_affine_cb, cb_output);
            }
        }

        cb_pop_front(cb_mean_accum, one_tile);
        cb_pop_front(cb_invstd, one_tile);
    }

    if (do_mask_w) {
        cb_pop_front(cb_mask_w, one_tile);
    }
    cb_pop_front(cb_epsilon, one_tile);
    cb_pop_front(cb_reduce_scaler, one_tile);
}

ALWI void layernorm_dispatch_stage_kernel_main() {
    constexpr uint32_t stage_mode = get_compile_time_arg_val(4);

    if constexpr (stage_mode == 0) {
        layernorm_copy_stage_kernel_main();
    } else if constexpr (stage_mode == 1) {
        layernorm_mean_reduce_stage_kernel_main();
    } else if constexpr (stage_mode == 2) {
        layernorm_invstd_reduce_stage_kernel_main();
    } else if constexpr (stage_mode == 3) {
        layernorm_normalize_stage_kernel_main();
    } else if constexpr (stage_mode == 4) {
        layernorm_residual_affine_stage_kernel_main();
    } else if constexpr (stage_mode == 5) {
        layernorm_residual_affine_stage_kernel_main();
    } else {
        layernorm_copy_stage_kernel_main();
    }
}
