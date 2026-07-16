// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/tensor_accessor.h"
#include "api/tensor/noc_traits.h"
///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t input_page_size = get_compile_time_arg_val(1);
constexpr uint32_t socket_page_size = get_compile_time_arg_val(2);
constexpr uint32_t num_pages_per_packet = get_compile_time_arg_val(3);
constexpr uint32_t num_whole_packets_per_page = get_compile_time_arg_val(4);
constexpr uint32_t partial_packet_size = get_compile_time_arg_val(5);
constexpr uint32_t whole_packet_size = get_compile_time_arg_val(6);
constexpr uint32_t input_args_cta_idx = 7;
constexpr uint32_t input_args_crta_idx = 0;

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t input_base_addr = get_arg_val<uint32_t>(0);
    uint32_t num_pages = get_arg_val<uint32_t>(1);            // pages for this core
    uint32_t page_start_offset = get_arg_val<uint32_t>(2);    // starting page offset
    uint32_t num_whole_packets = get_arg_val<uint32_t>(3);    // whole packets
    uint32_t num_pages_remainder = get_arg_val<uint32_t>(4);  // remainder pages

    auto input_addr_gen_args = TensorAccessorArgs<input_args_cta_idx, input_args_crta_idx>();
    auto input_addr_gen = TensorAccessor(input_addr_gen_args, input_base_addr);

    Noc noc_obj;
    DataflowBuffer cb0(cb0_id);

    // TODO #24995: Instead of page by page transfers, we can transfer bank by bank

    // Small pages. We pack multiple pages into a single packet.
    uint32_t page_index = page_start_offset;
    if constexpr (num_pages_per_packet > 0) {
        for (uint32_t i = 0; i < num_whole_packets; ++i) {
            cb0.reserve_back(1);
            auto l1_write_addr = cb0.get_write_ptr();
            for (uint32_t j = 0; j < num_pages_per_packet; ++j) {
                noc_obj.async_read(
                    input_addr_gen, CoreLocalMem<uint8_t>(l1_write_addr), input_page_size, {.page_id = page_index}, {});
                page_index++;
                l1_write_addr += socket_page_size;
            }
            noc_obj.async_read_barrier();
            cb0.push_back(1);
        }

        if (num_pages_remainder > 0) {
            cb0.reserve_back(1);
            auto l1_write_addr = cb0.get_write_ptr();
            for (uint32_t j = 0; j < num_pages_remainder; ++j) {
                noc_obj.async_read(
                    input_addr_gen, CoreLocalMem<uint8_t>(l1_write_addr), input_page_size, {.page_id = page_index}, {});
                page_index++;
                l1_write_addr += socket_page_size;
            }
            noc_obj.async_read_barrier();
            cb0.push_back(1);
        }

    }
    // Large pages. We pack page chunks into a single packet.
    else {
        // TODO #24995: Could read whole page into scratch, then copy locally
        for (uint32_t i = 0; i < num_pages; ++i) {
            auto noc_read_addr = input_addr_gen.get_noc_addr(page_index);
            for (uint32_t j = 0; j < num_whole_packets_per_page; ++j) {
                cb0.reserve_back(1);
                auto l1_write_addr = cb0.get_write_ptr();
                noc_async_read<whole_packet_size>(noc_read_addr, l1_write_addr, whole_packet_size);
                noc_read_addr += whole_packet_size;
                noc_obj.async_read_barrier();
                cb0.push_back(1);
            }
            if constexpr (partial_packet_size > 0) {
                cb0.reserve_back(1);
                auto l1_write_addr = cb0.get_write_ptr();
                noc_async_read<partial_packet_size>(noc_read_addr, l1_write_addr, partial_packet_size);
                noc_obj.async_read_barrier();
                cb0.push_back(1);
            }
            page_index++;
        }
    }
}
