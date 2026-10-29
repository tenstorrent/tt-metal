// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

#include <cstdint>

// RM gather writer: column-distributed multi-core. Parks FULL input row in input_dfb (reader needs any w).
// Drains the reader's output slice to memory at [w_start, w_start+w_per_core) via noc_async_*_sharded.
void kernel_main() {
    // Runtime args
    const uint32_t w_per_core = get_arg(args::w_per_core);
    const uint32_t w_start = get_arg(args::w_start);
    const uint32_t core_id = get_arg(args::core_id);

    // Compile time args
    constexpr uint32_t H = get_arg(args::H);
    constexpr uint32_t W_input = get_arg(args::W_input);
    constexpr uint32_t W_index = get_arg(args::W_index);
    constexpr uint32_t input_tensor_data_format_size = get_arg(args::input_tensor_data_format_size);
    constexpr uint32_t output_tensor_data_format_size = get_arg(args::output_tensor_data_format_size);

    if (w_per_core == 0) {
        return;
    }

    constexpr uint32_t one_stick = 1;
    constexpr uint32_t input_stick_size_bytes = W_input * input_tensor_data_format_size;
    const uint32_t output_byte_offset = w_start * output_tensor_data_format_size;
    const uint32_t output_slice_bytes = w_per_core * output_tensor_data_format_size;

    const auto input_accessor = TensorAccessor(tensor::input_tensor);
    const auto output_accessor = TensorAccessor(tensor::output_tensor);

    Noc noc;
    DataflowBuffer input_dfb(dfb::input_tensor);
    DataflowBuffer output_dfb(dfb::output_tensor);

    for (uint32_t h = 0; h < H; h++) {
        input_dfb.reserve_back(one_stick);
        tt::data_movement::common::noc_async_read_sharded(
            noc,
            input_dfb.get_write_ptr(),
            input_accessor,
            /*src_id=*/h,
            /*offset=*/0,
            /*size=*/input_stick_size_bytes);
        noc.async_read_barrier();
        input_dfb.push_back(one_stick);

        output_dfb.wait_front(one_stick);
        tt::data_movement::common::noc_async_write_sharded(
            noc,
            output_dfb.get_read_ptr(),
            output_accessor,
            /*dest_id=*/h,
            /*offset=*/output_byte_offset,
            /*size=*/output_slice_bytes);
        noc.async_write_barrier();
        output_dfb.pop_front(one_stick);
    }
}
