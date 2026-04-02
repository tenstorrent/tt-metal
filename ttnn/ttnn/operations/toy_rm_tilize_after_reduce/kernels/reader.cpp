// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t row_bytes = get_compile_time_arg_val(0);
    constexpr auto input_ta_args = TensorAccessorArgs<1>();

    const uint32_t input_addr = get_arg_val<uint32_t>(0);
    const uint32_t scaler_bits = get_arg_val<uint32_t>(1);
    const uint32_t total_num_rows = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_rm_in = 0;
    constexpr uint32_t cb_scaler = 8;

    float scaler_f = __builtin_bit_cast(float, scaler_bits);

    dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(scaler_f);

    const auto input_accessor = TensorAccessor(input_ta_args, input_addr);
    dataflow_kernel_lib::read_sticks_for_tilize<cb_rm_in, dataflow_kernel_lib::TilizeGranularity::TILE>(
        input_accessor, total_num_rows, row_bytes);
}
