#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel/dataflow/generate_mm_scaler.hpp"
#include "ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp"

void kernel_main() {
    constexpr bool has_residual = get_compile_time_arg_val(0) == 1;
    constexpr bool has_weight = get_compile_time_arg_val(1) == 1;
    constexpr bool has_bias = get_compile_time_arg_val(2) == 1;

    constexpr auto input_args = TensorAccessorArgs<3>();
    constexpr auto residual_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto weight_args = TensorAccessorArgs<residual_args.next_compile_time_args_offset()>();
    constexpr auto bias_args = TensorAccessorArgs<weight_args.next_compile_time_args_offset()>();

    int i = 0;
    const uint32_t input_addr = get_arg_val<uint32_t>(i++);
    const uint32_t residual_addr = get_arg_val<uint32_t>(i++);
    const uint32_t weight_addr = get_arg_val<uint32_t>(i++);
    const uint32_t bias_addr = get_arg_val<uint32_t>(i++);
    const uint32_t num_rows_per_core = get_arg_val<uint32_t>(i++);
    const uint32_t Wt = get_arg_val<uint32_t>(i++);
    const uint32_t tile_offset = get_arg_val<uint32_t>(i++);
    const uint32_t logical_W = get_arg_val<uint32_t>(i++);
    const uint32_t inv_width_bf16_packed = get_arg_val<uint32_t>(i++);
    const uint32_t epsilon_bits = get_arg_val<uint32_t>(i++);
    const uint32_t stage_mode = get_arg_val<uint32_t>(i++);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_residual = tt::CBIndex::c_1;
    constexpr auto cb_input_pass1 = tt::CBIndex::c_2;
    constexpr auto cb_residual_pass1 = tt::CBIndex::c_3;
    constexpr auto cb_weight = tt::CBIndex::c_4;
    constexpr auto cb_bias = tt::CBIndex::c_5;
    constexpr auto cb_reduce_scaler = tt::CBIndex::c_6;
    constexpr auto cb_epsilon = tt::CBIndex::c_7;
    constexpr auto cb_mask_w = tt::CBIndex::c_8;
    constexpr uint32_t tile_width = 32;

    const uint32_t input_tile_bytes = get_tile_size(cb_input);
    const auto input = TensorAccessor(input_args, input_addr, input_tile_bytes);
    const auto residual = TensorAccessor(residual_args, residual_addr, input_tile_bytes);
    const auto weight = TensorAccessor(weight_args, weight_addr, input_tile_bytes);
    const auto bias = TensorAccessor(bias_args, bias_addr, input_tile_bytes);

    generate_mm_scaler(cb_reduce_scaler, inv_width_bf16_packed);
    fill_cb_with_value(cb_epsilon, epsilon_bits);

    const bool do_mask_w = (logical_W % tile_width) != 0;
    if (do_mask_w) {
        generate_mask_w(cb_mask_w, logical_W % tile_width);
    }

    for (uint32_t row_idx = 0; row_idx < num_rows_per_core; ++row_idx) {
        const uint32_t row_tile_base = tile_offset + row_idx * Wt;

        for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
            noc_async_read_tile_helper(cb_input, 1, row_tile_base + col_idx, input);
            if (stage_mode >= 4) {
                if constexpr (has_residual) {
                    noc_async_read_tile_helper(cb_residual, 1, row_tile_base + col_idx, residual);
                }
            }
        }

        if (stage_mode >= 3) {
            for (uint32_t col_idx = 0; col_idx < Wt; ++col_idx) {
                noc_async_read_tile_helper(cb_input_pass1, 1, row_tile_base + col_idx, input);
                if (stage_mode >= 4) {
                    if constexpr (has_residual) {
                        noc_async_read_tile_helper(cb_residual_pass1, 1, row_tile_base + col_idx, residual);
                    }
                    if constexpr (has_weight) {
                        noc_async_read_tile_helper(cb_weight, 1, col_idx, weight);
                    }
                    if constexpr (has_bias) {
                        noc_async_read_tile_helper(cb_bias, 1, col_idx, bias);
                    }
                }
            }
        }
    }
}
