// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// post_stats_chain bench compute: per iteration (R tile-rows, resident z + mean) the whole-row chain
// center -> row_stats_block -> normalize -> affine_scale -> affine_shift, as variant CHAIN_VARIANT of
// chain_variants.hpp. Variant 0 is the op's code at 9daa4c30ecf verbatim.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "../chain_variants.hpp"

void kernel_main() {
    constexpr uint32_t cb_z = get_compile_time_arg_val(0);
    constexpr uint32_t cb_mean = get_compile_time_arg_val(1);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(2);
    constexpr uint32_t cb_beta = get_compile_time_arg_val(3);
    constexpr uint32_t cb_work_a = get_compile_time_arg_val(4);
    constexpr uint32_t cb_work_b = get_compile_time_arg_val(5);
    constexpr uint32_t cb_inv_std = get_compile_time_arg_val(6);
    constexpr uint32_t cb_out = get_compile_time_arg_val(7);
    constexpr uint32_t rows = get_compile_time_arg_val(8);
    constexpr uint32_t Wt = get_compile_time_arg_val(9);
    constexpr uint32_t iterations = get_compile_time_arg_val(10);
    constexpr uint32_t inv_w_bits = get_compile_time_arg_val(11);
    constexpr uint32_t epsilon_bits = get_compile_time_arg_val(12);
    constexpr uint32_t dest_lanes = get_compile_time_arg_val(13);
    constexpr uint32_t cb_aux = get_compile_time_arg_val(14);

    compute_kernel_hw_startup(cb_z, cb_gamma, cb_out);
    DataflowBuffer gamma(cb_gamma), beta(cb_beta);
    for (uint32_t it = 0; it < iterations; ++it) {
        if (it == 0) {
            gamma.wait_front(Wt);
            beta.wait_front(Wt);
        }
        post_stats::chain<
            cb_z,
            cb_mean,
            cb_gamma,
            cb_beta,
            cb_work_a,
            cb_work_b,
            cb_inv_std,
            cb_out,
            cb_aux,
            Wt,
            inv_w_bits,
            epsilon_bits,
            dest_lanes>(rows);
    }
    gamma.pop_front(Wt);
    beta.pop_front(Wt);
}
