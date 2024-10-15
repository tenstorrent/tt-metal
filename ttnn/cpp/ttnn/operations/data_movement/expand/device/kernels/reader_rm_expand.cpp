// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "risc_attribs.h"
#include "tensix.h"

void kernel_main() {
    std::uint32_t mem_buffer_src_addr = get_arg_val<uint32_t>(0);

    std::uint32_t num_rows = get_arg_val<uint32_t>(1);
    std::uint32_t element_per_row = get_arg_val<uint32_t>(2);
    std::uint32_t horz_expand_count = get_arg_val<uint32_t>(3);

    std::uint32_t dram_page_size = get_arg_val<uint32_t>(4);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t scratch_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t io_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t datasize_bytes = get_compile_time_arg_val(3);

    InterleavedAddrGen<src_is_dram> src_generator = {
        .bank_base_address = mem_buffer_src_addr,
        .page_size = dram_page_size,
    };

    cb_reserve_back(scratch_cb_id, 1);
    auto tmp_buf = get_write_ptr(scratch_cb_id);

    for (uint32_t i = 0; i < num_rows; i++) {
        cb_reserve_back(io_cb_id, 1);

        auto l1_addr = get_write_ptr(io_cb_id);
        auto noc_addr = get_noc_addr(i, src_generator);

        // Read the entire row into scratch buffer
        noc_async_read(noc_addr, tmp_buf, dram_page_size);
        noc_async_read_barrier();

        auto l1_ptr = reinterpret_cast<char *>(l1_addr);
        auto tmp_buf_ptr = reinterpret_cast<char *>(tmp_buf);

        for (uint32_t k = 0; k < horz_expand_count; k++) {
#pragma unroll
            for (uint32_t j = 0; j < element_per_row; j++) {
#pragma unroll
                for (uint32_t i = 0; i < datasize_bytes; i++) {
                    l1_ptr[k * element_per_row * datasize_bytes + j * datasize_bytes + i] =
                        tmp_buf_ptr[j * datasize_bytes + i];
                }
            }
        }

        cb_push_back(io_cb_id, 1);
    }
}
