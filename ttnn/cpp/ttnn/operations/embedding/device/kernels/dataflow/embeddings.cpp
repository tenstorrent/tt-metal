// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings_common.hpp"

void kernel_main() {
    const std::uint32_t input_buffer_src_addr = get_arg_val<uint32_t>(0);
    const std::uint32_t weight_buffer_src_addr = get_arg_val<uint32_t>(1);
    const std::uint32_t batch_offset = get_arg_val<uint32_t>(2);
    const std::uint32_t weights_offset = get_arg_val<uint32_t>(3);
    const std::uint32_t num_rows = get_arg_val<uint32_t>(4);

    const std::uint32_t index_idx = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_in2 = get_compile_time_arg_val(2);

    constexpr bool input_in_dram = get_compile_time_arg_val(3) == 1;
    constexpr uint32_t input_page_size = get_compile_time_arg_val(4);
    const auto input = get_interleaved_addr_gen<input_in_dram, input_page_size>(input_buffer_src_addr);

    constexpr bool weight_in_dram = get_compile_time_arg_val(5) == 1;
    constexpr uint32_t weight_stick_size = get_compile_time_arg_val(6);
    const auto weights = get_interleaved_addr_gen<weight_in_dram, weight_stick_size>(weight_buffer_src_addr);

    constexpr uint32_t rows_per_block = get_compile_time_arg_val(7);
    constexpr uint32_t input_block_size_bytes = get_compile_time_arg_val(8);

    prepare_local_cache(cb_id_in2, weights, weight_stick_size, /*pad_token_arg_idx=*/6);

    cb_reserve_back(cb_id_in1, 1);
    uint32_t input_l1_addr = get_write_ptr(cb_id_in1);
    volatile tt_l1_ptr input_token_t* input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr input_token_t*>(input_l1_addr);

    uint32_t curr_row = batch_offset;
    uint32_t offset = weights_offset;
    uint32_t index = index_idx;

    bool read_indices = true;
    for (uint32_t i = 0; i < num_rows; ++i) {
        if (read_indices) {
            uint64_t noc_input_src_addr = get_noc_addr(curr_row, input) + offset;
            noc_async_read(noc_input_src_addr, input_l1_addr, input_block_size_bytes);
            noc_async_read_barrier();
            read_indices = false;
        }
        cb_reserve_back(cb_id_in0, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        input_token_t token = input_l1_ptr[index];
        uint64_t src_noc_addr = get_token_noc_addr(token, weights);
        noc_async_read<weight_stick_size>(src_noc_addr, l1_write_addr, weight_stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);

        index++;
        if (index == rows_per_block) {
            index = 0;
            read_indices = true;
            offset += input_block_size_bytes;
            if (offset == input_page_size) {
                offset = 0;
                curr_row++;
            }
        }
    }
}
