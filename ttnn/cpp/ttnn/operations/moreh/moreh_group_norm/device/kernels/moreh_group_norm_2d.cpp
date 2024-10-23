// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    // runtime args
    int i{0};
    const auto C = get_arg_val<uint32_t>(i++);
    const auto num_groups = get_arg_val<uint32_t>(i++);
    const auto unit_offset = get_arg_val<uint32_t>(i++);

    // compile-time args
    constexpr uint32_t num_units_per_core = get_compile_time_arg_val(0);
    constexpr bool gamma_has_value = get_compile_time_arg_val(1) == 1;
    constexpr bool beta_has_value = get_compile_time_arg_val(2) == 1;
    constexpr bool mean_has_value = get_compile_time_arg_val(3) == 1;
    constexpr bool rstd_has_value = get_compile_time_arg_val(4) == 1;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0);

    constexpr auto cb_x = tt::CB::c_in0;       // input
    constexpr auto cb_scaler = tt::CB::c_in1;  // scaler
    constexpr auto cb_eps = tt::CB::c_in2;     // epsilon
    constexpr auto cb_gamma = tt::CB::c_in3;   // gamma
    constexpr auto cb_beta = tt::CB::c_in4;    // beta
    constexpr auto cb_mask = tt::CB::c_in5;    // mask
    constexpr auto cb_zeros = tt::CB::c_in6;   // zeros

    constexpr auto cb_output = tt::CB::c_out0;  // output
    constexpr auto cb_mean = tt::CB::c_out1;    // mean
    constexpr auto cb_rstd = tt::CB::c_out2;    // rstd

    constexpr auto cb_ex = tt::CB::c_intermed0;          // E[x]
    constexpr auto cb_xmm = tt::CB::c_intermed1;         // x - E[x]
    constexpr auto cb_xmm2 = tt::CB::c_intermed2;        // (x - E[x])^2
    constexpr auto cb_xmm2sum = tt::CB::c_intermed3;     // Sum[(x - E[x])^2]
    constexpr auto cb_var = tt::CB::c_intermed4;         // E[(x - E[x])^2] = Var[x]
    constexpr auto cb_recip_std = tt::CB::c_intermed5;   // 1.0/(sqrt(Var[x] + eps))
    constexpr auto cb_gamma_beta = tt::CB::c_intermed6;  // p * gamm + beta
    constexpr auto cb_xsum = tt::CB::c_intermed7;        // Sum[x]

    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;
    constexpr int dst1 = 1;

    cb_wait_front(cb_scaler, onetile);  // comes from the reader
    cb_wait_front(cb_eps, onetile);     // comes from the reader
    cb_wait_front(cb_zeros, onetile);   // comes from the reader

    for (uint32_t outer_idx = 0; outer_idx < num_units_per_core; outer_idx++) {
        // input[N, C]
        // reshaped_input[N, num_groups, C / num_groups]
        auto unit_idx = unit_offset + outer_idx;
        auto group_idx = unit_idx % num_groups;

        uint32_t c_size_per_unit = C / num_groups;
        uint32_t c_start = c_size_per_unit * group_idx;
        uint32_t c_end = c_size_per_unit * (group_idx + 1);

        {
            uint32_t read_start = c_start;
            uint32_t read_end = c_start;
            bool first_iter = true;
            for (;;) {
                // compute read start
                read_start = read_end;
                // compute read end
                read_end = std::min(round_up(read_start + 1, TILE_WIDTH), c_end);

                if (read_start >= c_end) {
                    break;
                }

                bool is_tile_end = (read_end % TILE_WIDTH == 0);
                bool is_last = (read_end == c_end);
                if (is_tile_end || is_last) {
                    /*
                     * Sum[x]
                     * cb_xsum
                     */
                    if (first_iter) {
                        cb_reserve_back(cb_xsum, onetile);
                        cb_wait_front(cb_x, onetile);

                        tile_regs_acquire();
                        copy_tile_init_with_dt(cb_x);
                        copy_tile(cb_x, 0, dst0);

                        bool mask_required = (read_start % TILE_WIDTH != 0) || (read_end % TILE_WIDTH != 0);
                        if (mask_required) {
                            cb_wait_front(cb_mask, onetile);

                            copy_tile_init_with_dt(cb_mask);
                            copy_tile(cb_mask, 0, dst1);

                            mask_tile_init();
                            mask_tile(dst0, dst1);

                            cb_pop_front(cb_mask, onetile);
                        }
                        tile_regs_commit();

                        tile_regs_wait();
                        pack_tile_with_dt(dst0, cb_xsum);
                        tile_regs_release();

                        cb_pop_front(cb_x, onetile);
                        cb_push_back(cb_xsum, onetile);
                    } else {
                        // I use cb_ex temporarily.
                        constexpr auto cb_tmp = cb_ex;
                        cb_reserve_back(cb_tmp, onetile);
                        cb_wait_front(cb_x, onetile);

                        copy_tile_init_with_dt(cb_x);
                        copy_tile(cb_x, 0, dst0);  // input

                        bool mask_required = (read_start % TILE_WIDTH != 0) || (read_end % TILE_WIDTH != 0);
                        if (mask_required) {
                            cb_wait_front(cb_mask, onetile);

                            copy_tile_init_with_dt(cb_mask);
                            copy_tile(cb_mask, 0, dst1);

                            mask_tile_init();
                            mask_tile(dst0, dst1);
                        }
                        tile_regs_commit();

                        tile_regs_wait();
                        pack_tile_with_dt(dst0, cb_tmp);
                        tile_regs_release();

                        cb_pop_front(cb_x, onetile);
                        cb_push_back(cb_tmp, onetile);

                        // add_tiles
                        cb_wait_front(cb_xsum, onetile);
                        cb_wait_front(cb_tmp, onetile);

                        tile_regs_acquire();
                        add_tiles_init_with_dt(cb_xsum, cb_tmp);
                        add_tiles(cb_xsum, cb_tmp, 0, 0, dst0);
                        tile_regs_commit();

                        cb_pop_front(cb_xsum, onetile);
                        cb_pop_front(cb_tmp, onetile);

                        cb_reserve_back(cb_xsum, onetile);

                        tile_regs_wait();
                        pack_tile_with_dt(dst0, cb_xsum);
                        tile_regs_release();

                        cb_push_back(cb_xsum, onetile);
                    }
                }  // if (is_tile_end || is_last)

                first_iter = false;
            }
        }

        /*
         * E[x]
         * cb_ex
         */
        cb_wait_front(cb_xsum, onetile);
        cb_reserve_back(cb_ex, onetile);
        if (mean_has_value) {
            cb_reserve_back(cb_mean, onetile);
        }

        tile_regs_acquire();
        reduce_init_delta_with_dt<false>(cb_ex, cb_xsum, cb_scaler);
        reduce_tile(cb_xsum, cb_scaler, 0, 0, dst0);
        reduce_revert_delta(cb_ex);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_ex);
        if (mean_has_value) {
            pack_tile_with_dt(dst0, cb_mean);
        }
        tile_regs_release();

        cb_pop_front(cb_xsum, onetile);
        cb_push_back(cb_ex, onetile);
        if (mean_has_value) {
            cb_push_back(cb_mean, onetile);
        }

        // mean
        {
            uint32_t read_start = c_start;
            uint32_t read_end = c_start;
            bool first_iter = true;
            for (;;) {
                // compute read start
                read_start = read_end;
                // compute read end
                read_end = std::min(round_up(read_start + 1, TILE_WIDTH), c_end);

                if (read_start >= c_end) {
                    break;
                }

                bool is_tile_end = (read_end % TILE_WIDTH == 0);
                bool is_last = (read_end == c_end);
                if (is_tile_end || is_last) {
                    /*
                     * x - E[x]
                     * xmm
                     */
                    cb_reserve_back(cb_xmm, onetile);
                    cb_wait_front(cb_x, onetile);
                    cb_wait_front(cb_ex, onetile);

                    tile_regs_acquire();
                    sub_bcast_cols_init_short_with_dt(cb_x, cb_ex);
                    sub_tiles_bcast_cols(cb_x, cb_ex, 0, 0, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, cb_xmm);
                    tile_regs_release();

                    cb_pop_front(cb_x, onetile);
                    cb_push_back(cb_xmm, onetile);

                    /*
                     * (x - E[x])^2
                     * cb_xmm2
                     */
                    cb_wait_front(cb_xmm, onetile);
                    cb_reserve_back(cb_xmm2, onetile);

                    tile_regs_acquire();
                    mul_tiles_init_with_dt(cb_xmm, cb_xmm);
                    mul_tiles(cb_xmm, cb_xmm, 0, 0, dst0);
                    bool mask_required = (read_start % TILE_WIDTH != 0) || (read_end % TILE_WIDTH != 0);
                    if (mask_required) {
                        cb_wait_front(cb_mask, onetile);

                        copy_tile_init_with_dt(cb_mask);
                        copy_tile(cb_mask, 0, dst1);

                        mask_tile_init();
                        mask_tile(dst0, dst1);

                        cb_pop_front(cb_mask, onetile);
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, cb_xmm2);
                    tile_regs_release();

                    cb_pop_front(cb_xmm, onetile);
                    cb_push_back(cb_xmm2, onetile);

                    /*
                     * Sum[(x-E[x])^2]
                     * cb_xmm2sum
                     */
                    if (first_iter) {
                        cb_reserve_back(cb_xmm2sum, onetile);
                        cb_wait_front(cb_xmm2, onetile);
                        cb_wait_front(cb_zeros, onetile);

                        tile_regs_acquire();
                        add_tiles_init_with_dt(cb_xmm2, cb_zeros);
                        add_tiles(cb_xmm2, cb_zeros, 0, 0, dst0);
                        tile_regs_commit();

                        tile_regs_wait();
                        pack_tile_with_dt(dst0, cb_xmm2sum);
                        tile_regs_release();

                        cb_push_back(cb_xmm2sum, onetile);
                        cb_pop_front(cb_xmm2, onetile);
                    } else {
                        cb_wait_front(cb_xmm2sum, onetile);
                        cb_wait_front(cb_xmm2, onetile);

                        tile_regs_acquire();
                        add_tiles_init_with_dt(cb_xmm2sum, cb_xmm2);
                        add_tiles(cb_xmm2sum, cb_xmm2, 0, 0, dst0);
                        tile_regs_commit();

                        cb_pop_front(cb_xmm2sum, onetile);

                        cb_reserve_back(cb_xmm2sum, onetile);

                        tile_regs_wait();
                        pack_tile_with_dt(dst0, cb_xmm2sum);
                        tile_regs_release();

                        cb_push_back(cb_xmm2sum, onetile);
                        cb_pop_front(cb_xmm2, onetile);
                    }
                    first_iter = false;
                }  // if (is_tile_end || is_last)
            }
        }

        /*
         * E[(x-E[x])^2 = Var[x]
         * cb_var
         */
        cb_wait_front(cb_xmm2sum, onetile);
        cb_reserve_back(cb_var, onetile);

        tile_regs_acquire();
        reduce_init_delta_with_dt<false>(cb_var, cb_xmm2sum, cb_scaler);
        reduce_tile(cb_xmm2sum, cb_scaler, 0, 0, dst0);
        reduce_revert_delta(cb_var);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_var);
        tile_regs_release();

        cb_pop_front(cb_xmm2sum, onetile);
        cb_push_back(cb_var, onetile);

        /*
         * 1.0/(sqrt(E[(x-E[x])^2] + eps))
         * cb_recip_std
         */
        cb_wait_front(cb_var, onetile);
        cb_reserve_back(cb_recip_std, onetile);
        if (rstd_has_value) {
            cb_reserve_back(cb_rstd, onetile);
        }

        tile_regs_acquire();
        add_tiles_init_with_dt(cb_var, cb_eps);
        add_tiles(cb_var, cb_eps, 0, 0, dst0);

        rsqrt_tile_init();
        rsqrt_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_recip_std);
        if (rstd_has_value) {
            pack_tile_with_dt(dst0, cb_rstd);
        }
        tile_regs_release();

        cb_pop_front(cb_var, onetile);
        cb_push_back(cb_recip_std, onetile);
        if (rstd_has_value) {
            cb_push_back(cb_rstd, onetile);
        }

        /*
         * (x - E[x]) * (1.0/(sqrt(E[(x-E[x])^2] + eps)))
         * (x - E[x]) * (1.0/(sqrt(E[(x-E[x])^2] + eps))) * gamma + beta
         * cb_output
         */
        {
            uint32_t read_start = c_start;
            uint32_t read_end = c_start;
            for (;;) {
                // compute read start
                read_start = read_end;
                // compute read end
                read_end = std::min(round_up(read_start + 1, TILE_WIDTH), c_end);

                if (read_start >= c_end) {
                    break;
                }

                bool is_tile_end = (read_end % TILE_WIDTH == 0);
                bool is_last = (read_end == c_end);
                if (is_tile_end || is_last) {
                    /*
                     * x - E[x]
                     * cb_reuse(==cb_xmm)
                     */
                    constexpr auto cb_reuse = cb_xmm;

                    cb_wait_front(cb_x, onetile);
                    cb_wait_front(cb_ex, onetile);
                    cb_reserve_back(cb_reuse, onetile);

                    tile_regs_acquire();
                    sub_bcast_cols_init_short_with_dt(cb_x, cb_ex);
                    sub_tiles_bcast_cols(cb_x, cb_ex, 0, 0, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, cb_reuse);
                    tile_regs_release();

                    cb_pop_front(cb_x, onetile);
                    cb_push_back(cb_reuse, onetile);

                    /*
                     * (x - E[x]) * 1.0/sqrt(Var[x] + eps)
                     * cb_gamma_beta_or_output
                     */
                    constexpr auto cb_gamma_beta_or_output =
                        (gamma_has_value || beta_has_value) ? cb_gamma_beta : cb_output;
                    cb_wait_front(cb_reuse, onetile);
                    cb_wait_front(cb_recip_std, onetile);
                    cb_reserve_back(cb_gamma_beta_or_output, onetile);

                    tile_regs_acquire();
                    mul_bcast_cols_init_short_with_dt(cb_reuse, cb_recip_std);
                    mul_tiles_bcast_cols(cb_reuse, cb_recip_std, 0, 0, dst0);
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(dst0, cb_gamma_beta_or_output);
                    tile_regs_release();

                    cb_pop_front(cb_reuse, onetile);
                    cb_push_back(cb_gamma_beta_or_output, onetile);

                    // * gamma
                    if (gamma_has_value) {
                        constexpr auto cb_gamma_output = beta_has_value ? cb_gamma_beta : cb_output;
                        cb_wait_front(cb_gamma_beta_or_output, onetile);
                        cb_wait_front(cb_gamma, onetile);
                        cb_reserve_back(cb_gamma_output, onetile);

                        tile_regs_acquire();
                        mul_bcast_rows_init_short_with_dt(cb_gamma_beta_or_output, cb_gamma);
                        mul_tiles_bcast_rows(cb_gamma_beta_or_output, cb_gamma, 0, 0, dst0);
                        tile_regs_commit();

                        tile_regs_wait();
                        pack_tile_with_dt(dst0, cb_gamma_output);
                        tile_regs_release();

                        cb_pop_front(cb_gamma_beta_or_output, onetile);
                        cb_pop_front(cb_gamma, onetile);
                        cb_push_back(cb_gamma_output, onetile);
                    }  // if (gamma_has_value)

                    // + beta
                    if (beta_has_value) {
                        cb_wait_front(cb_gamma_beta, onetile);
                        cb_wait_front(cb_beta, onetile);
                        cb_reserve_back(cb_output, onetile);

                        tile_regs_acquire();
                        add_bcast_rows_init_short_with_dt(cb_gamma_beta, cb_beta);
                        add_tiles_bcast_rows(cb_gamma_beta, cb_beta, 0, 0, dst0);
                        tile_regs_commit();

                        tile_regs_wait();
                        pack_tile_with_dt(dst0, cb_output);
                        tile_regs_release();

                        cb_pop_front(cb_gamma_beta, onetile);
                        cb_pop_front(cb_beta, onetile);
                        cb_push_back(cb_output, onetile);
                    }  // if (beta_has_value)
                }  // if (is_tile_end || is_last)
            }  // for(;;)
        }  // cb_out

        cb_pop_front(cb_recip_std, onetile);
        cb_pop_front(cb_ex, onetile);
    }

}  // void MAIN
}  // namespace NAMESPACE
