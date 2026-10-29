// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "gather_common.hpp"

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

#include <cstdint>

// RM gather reader: row-distributed single-core. Reads index stick, gathers from input_dfb, writes to output_dfb.
// I/O via noc_async_*_sharded with per-shard page size; same kernel covers interleaved + H/B/W-sharded.
void kernel_main() {
    // Runtime args
    const uint32_t core_loop_count = get_arg(args::core_loop_count);
    const uint32_t core_id = get_arg(args::core_id);

    // Compile time args
    constexpr uint32_t H = get_arg(args::H);
    constexpr uint32_t W_input = get_arg(args::W_input);
    constexpr uint32_t W_index = get_arg(args::W_index);
    constexpr uint32_t total_number_of_cores = get_arg(args::total_number_of_cores);
    constexpr uint32_t input_tensor_data_format_size = get_arg(args::input_tensor_data_format_size);
    constexpr uint32_t input_index_tensor_data_format_size = get_arg(args::input_index_tensor_data_format_size);
    constexpr uint32_t output_tensor_data_format_size = get_arg(args::output_tensor_data_format_size);

    constexpr uint32_t one_stick = 1;
    constexpr uint32_t input_index_stick_size_bytes = W_index * input_index_tensor_data_format_size;

    // The accessor's page size, which the tensor binding supplies, drives the multi-shard split:
    // shard_W * elem_size for B/W, full row otherwise.
    const auto input_index_accessor = TensorAccessor(tensor::input_index_tensor);

    Noc noc;
    DataflowBuffer input_dfb(dfb::input_tensor);
    DataflowBuffer input_index_dfb(dfb::input_index_tensor);
    DataflowBuffer output_dfb(dfb::output_tensor);

    for (uint32_t core_loop = 0; core_loop < core_loop_count; core_loop++) {
        const uint32_t h = core_loop * total_number_of_cores + core_id;

        input_index_dfb.reserve_back(one_stick);
        tt::data_movement::common::noc_async_read_sharded(
            noc,
            input_index_dfb.get_write_ptr(),
            input_index_accessor,
            /*src_id=*/h,
            /*offset=*/0,
            /*size=*/input_index_stick_size_bytes);
        noc.async_read_barrier();
        input_index_dfb.push_back(one_stick);

        input_dfb.wait_front(one_stick);
        output_dfb.reserve_back(one_stick);

        const uint32_t input_l1_read_addr = input_dfb.get_read_ptr();
        const uint32_t input_index_l1_read_addr = input_index_dfb.get_read_ptr();
        const uint32_t output_l1_write_addr = output_dfb.get_write_ptr();

        // Linear gather: output[w] = input[index[w]]. No tile-face math — RM data is
        // contiguous and `get_value_from_stick` indexes by element offset directly.
        for (uint32_t w = 0; w < W_index; w++) {
            const uint32_t global_index =
                get_value_from_stick(input_index_l1_read_addr, w, input_index_tensor_data_format_size);
            const uint32_t value =
                get_value_from_stick(input_l1_read_addr, global_index, input_tensor_data_format_size);
            write_value_to_stick(output_l1_write_addr, w, output_tensor_data_format_size, value);
        }

        output_dfb.push_back(one_stick);
        input_index_dfb.pop_front(one_stick);
        input_dfb.pop_front(one_stick);
    }
}
