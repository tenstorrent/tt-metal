// gelu_backward_kernel.cpp
#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_size = get_compile_time_arg_val(1);
    bool fast_approx = get_compile_time_arg_val(2);

    // Circular buffer indices
    constexpr auto cb_grad_out = tt::CBIndex::c_0;  // Upstream gradient
    constexpr auto cb_input_x = tt::CBIndex::c_1;   // Input x
    constexpr auto cb_output = tt::CBIndex::c_16;   // Output gradient

    // Constants
    vFloat sqrt_2_over_pi = s2vFloat16a(0.7978845608);  // sqrt(2/pi)
    vFloat const_0_044715 = s2vFloat16a(0.044715);
    vFloat const_3 = s2vFloat16a(3.0);
    vFloat const_0_5 = s2vFloat16a(0.5);
    vFloat const_1 = s2vFloat16a(1.0);
    vFloat const_0_134145 = s2vFloat16a(0.134145);  // 3 * 0.044715

    // Initialize binary operation
    binary_op_init_common(cb_grad_out, cb_input_x, cb_output);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        cb_wait_front(cb_grad_out, per_core_block_size);
        cb_wait_front(cb_input_x, per_core_block_size);
        cb_reserve_back(cb_output, per_core_block_size);

        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            tile_regs_acquire();

            // Load the gradient output and input x
            copy_tile(cb_grad_out, i, 1);  // Load grad_output to DST[1]
            copy_tile(cb_input_x, i, 0);   // Load input x to DST[0]

            // Calculate GELU backward
            // First compute tanh argument: sqrt(2/π) * (x + 0.044715 * x^3)
            vFloat x = dst_reg[0];
            vFloat x_squared;
            vFloat x_cubed;
            vFloat tanh_arg;
            vFloat tanh_val;
            vFloat tanh_squared;
            vFloat cdf_term;
            vFloat pdf_term;
            vFloat result;

            // Compute x^2 and x^3
            square_tile_init();
            square_tile(0);
            x_squared = dst_reg[2];

            mul_tiles_init(tt::CBIndex::c_in0, tt::CBIndex::c_in1);
            mul_tiles(x, x_squared, 0, 2, 3);
            x_cubed = dst_reg[3];

            // Compute tanh argument: sqrt(2/π) * (x + 0.044715 * x^3)
            mul_tiles(const_0_044715, x_cubed, 0, 3, 4);
            add_tiles_init(tt::CBIndex::c_in0, tt::CBIndex::c_in1);
            add_tiles(x, dst_reg[4], 0, 4, 5);
            mul_tiles(sqrt_2_over_pi, dst_reg[5], 0, 5, 6);
            tanh_arg = dst_reg[6];

            // Compute tanh
            tanh_tile_init();
            tanh_tile(6);
            tanh_val = dst_reg[6];

            // Compute tanh squared
            square_tile(6);
            tanh_squared = dst_reg[7];

            // CDF term: 0.5 * (1 + tanh(arg))
            add_tiles(const_1, tanh_val, 0, 6, 8);
            mul_tiles(const_0_5, dst_reg[8], 0, 8, 9);
            cdf_term = dst_reg[9];

            // PDF term: 0.5 * sqrt(2/π) * (1 + 0.134145 * x^2) * (1 - tanh^2)
            // 1 - tanh^2
            sub_tiles_init(tt::CBIndex::c_in0, tt::CBIndex::c_in1);
            sub_tiles(const_1, tanh_squared, 0, 7, 10);

            // 1 + 0.134145 * x^2
            mul_tiles(const_0_134145, x_squared, 0, 2, 11);
            add_tiles(const_1, dst_reg[11], 0, 11, 12);

            // 0.5 * sqrt(2/π) * (1 + 0.134145 * x^2) * (1 - tanh^2)
            mul_tiles(dst_reg[12], dst_reg[10], 12, 10, 13);
            mul_tiles(sqrt_2_over_pi, dst_reg[13], 0, 13, 14);
            mul_tiles(const_0_5, dst_reg[14], 0, 14, 15);
            pdf_term = dst_reg[15];

            // x * pdf_term
            mul_tiles(x, pdf_term, 0, 15, 10);

            // cdf_term + x * pdf_term
            add_tiles(cdf_term, dst_reg[10], 9, 10, 11);

            // grad_output * (cdf_term + x * pdf_term)
            mul_tiles(dst_reg[1], dst_reg[11], 1, 11, 0);

            tile_regs_commit();
            tile_regs_wait();

            // Pack the result
            pack_tile(0, cb_output);

            tile_regs_release();
        }

        cb_pop_front(cb_grad_out, per_core_block_size);
        cb_pop_front(cb_input_x, per_core_block_size);
        cb_push_back(cb_output, per_core_block_size);
    }
}
}  // namespace NAMESPACE
