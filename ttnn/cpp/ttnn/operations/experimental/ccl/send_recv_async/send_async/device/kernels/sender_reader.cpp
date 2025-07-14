// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "dataflow_api.h"
#include "tt_metal/hw/inc/accessor/tensor_accessor.h"

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t num_pages = get_compile_time_arg_val(1);
constexpr uint32_t input_page_size = get_compile_time_arg_val(2);
constexpr uint32_t socket_page_size = get_compile_time_arg_val(3);
constexpr uint32_t num_pages_per_packet = get_compile_time_arg_val(4);
// Used when there are multiple pages per packet
constexpr uint32_t num_whole_packets = get_compile_time_arg_val(5);
constexpr uint32_t num_pages_remainder = get_compile_time_arg_val(6);
// Used when there are multiple packets per page
constexpr uint32_t num_whole_packets_per_page = get_compile_time_arg_val(7);
constexpr uint32_t partial_packet_size = get_compile_time_arg_val(8);
constexpr uint32_t whole_packet_size = get_compile_time_arg_val(9);
constexpr uint32_t input_args_cta_idx = 10;
constexpr uint32_t input_args_crta_idx = 0;

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t input_base_addr = get_arg_val<uint32_t>(0);

    auto input_addr_gen_args = make_tensor_accessor_args<input_args_cta_idx, input_args_crta_idx>();
    auto input_addr_gen = make_tensor_accessor_from_args(input_addr_gen_args, input_base_addr, input_page_size);

    // TODO #24995: Instead of page by page transfers, we can transfer bank by bank

    // Small pages. We pack multiple pages into a single packet.
    uint32_t page_index = 0;
    if constexpr (num_pages_per_packet > 0) {
        for (uint32_t i = 0; i < num_whole_packets; ++i) {
            cb_reserve_back(cb0_id, 1);
            auto l1_write_addr = get_write_ptr(cb0_id);
            for (uint32_t j = 0; j < num_pages_per_packet; ++j) {
                auto noc_read_addr = input_addr_gen.get_noc_addr(page_index);
                noc_async_read<input_page_size>(noc_read_addr, l1_write_addr, input_page_size);
                page_index++;
                l1_write_addr += socket_page_size;
            }
            noc_async_read_barrier();
            cb_push_back(cb0_id, 1);
        }

        if constexpr (num_pages_remainder > 0) {
            cb_reserve_back(cb0_id, 1);
            auto l1_write_addr = get_write_ptr(cb0_id);
            for (uint32_t j = 0; j < num_pages_remainder; ++j) {
                auto noc_read_addr = input_addr_gen.get_noc_addr(page_index);
                noc_async_read<input_page_size>(noc_read_addr, l1_write_addr, input_page_size);
                page_index++;
                l1_write_addr += socket_page_size;
            }
            noc_async_read_barrier();
            cb_push_back(cb0_id, 1);
        }

    }
    // Large pages. We pack page chunks into a single packet.
    else {
        // TODO #24995: Could read whole page into scratch, then copy locally
        for (uint32_t i = 0; i < num_pages; ++i) {
            auto noc_read_addr = input_addr_gen.get_noc_addr(page_index);
            for (uint32_t j = 0; j < num_whole_packets_per_page; ++j) {
                cb_reserve_back(cb0_id, 1);
                auto l1_write_addr = get_write_ptr(cb0_id);
                noc_async_read<whole_packet_size>(noc_read_addr, l1_write_addr, whole_packet_size);
                noc_read_addr += whole_packet_size;
                noc_async_read_barrier();
                cb_push_back(cb0_id, 1);
            }
            if constexpr (partial_packet_size > 0) {
                cb_reserve_back(cb0_id, 1);
                auto l1_write_addr = get_write_ptr(cb0_id);
                noc_async_read<partial_packet_size>(noc_read_addr, l1_write_addr, partial_packet_size);
                noc_async_read_barrier();
                cb_push_back(cb0_id, 1);
            }
            page_index++;
        }
    }
}
