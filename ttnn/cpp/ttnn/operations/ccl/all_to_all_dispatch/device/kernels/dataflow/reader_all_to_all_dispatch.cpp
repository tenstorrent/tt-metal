// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    DPRINT << "Kernel started" << ENDL();
    constexpr bool input_is_dram = (bool)get_compile_time_arg_val(0);
    constexpr bool indices_is_dram = (bool)get_compile_time_arg_val(1);
    constexpr bool mapping_is_dram = (bool)get_compile_time_arg_val(2);
    constexpr bool output_is_dram = (bool)get_compile_time_arg_val(3);
    constexpr bool metadata_is_dram = (bool)get_compile_time_arg_val(4);

    constexpr uint32_t input_tensor_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t indices_tensor_cb_id = get_compile_time_arg_val(6);
    constexpr uint32_t mapping_tensor_cb_id = get_compile_time_arg_val(7);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(8);
    constexpr uint32_t send_preparation_buffer_cb_id = get_compile_time_arg_val(9);

    constexpr uint32_t input_pages = get_compile_time_arg_val(10);
    constexpr uint32_t indices_pages = get_compile_time_arg_val(11);
    constexpr uint32_t mapping_pages = get_compile_time_arg_val(12);
    constexpr uint32_t output_pages = get_compile_time_arg_val(13);
    constexpr uint32_t metadata_pages = get_compile_time_arg_val(14);

    constexpr uint32_t input_page_size = get_compile_time_arg_val(15);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(16);
    constexpr uint32_t mapping_page_size = get_compile_time_arg_val(17);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(18);
    constexpr uint32_t metadata_page_size = get_compile_time_arg_val(19);

    constexpr uint32_t num_devices = get_compile_time_arg_val(20);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(21);
    constexpr uint32_t batch_size = get_compile_time_arg_val(22);
    constexpr uint32_t selected_experts_k = get_compile_time_arg_val(23);
    constexpr uint32_t experts = get_compile_time_arg_val(24);
    constexpr uint32_t tokens_per_device = get_compile_time_arg_val(25);

    constexpr uint32_t num_links = get_compile_time_arg_val(26);
    constexpr bool is_ring_topology = (bool)get_compile_time_arg_val(27);

    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(28);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(29);

    constexpr uint32_t mesh_rows = get_compile_time_arg_val(30);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(31);  // ew_dim
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(32);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(33);
    constexpr uint32_t aligned_mapping_page_size = get_compile_time_arg_val(34);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(35);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(36);

    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(37);

#ifdef AXIS
    constexpr int axis = AXIS;
    constexpr uint32_t dispatch_devices = axis == 0 ? mesh_rows : mesh_cols;
    constexpr uint32_t dispatch_index = axis == 0 ? src_chip_id % mesh_rows : src_chip_id % mesh_cols;
#else
    constexpr int axis = -1;
    constexpr uint32_t dispatch_devices = num_devices;
    constexpr uint32_t dispatch_index = src_chip_id;
#endif

    uint32_t input_tensor_address = get_arg_val<uint32_t>(0);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(1);
    uint32_t mapping_tensor_address = get_arg_val<uint32_t>(2);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(3);
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(4);

    uint32_t global_semaphore_address = get_arg_val<uint32_t>(5);

    const auto input_addr_gen = get_interleaved_addr_gen<input_is_dram, input_page_size>(input_tensor_address);
    const auto indices_addr_gen = get_interleaved_addr_gen<indices_is_dram, indices_page_size>(indices_tensor_address);
    const auto mapping_addr_gen = get_interleaved_addr_gen<mapping_is_dram, mapping_page_size>(mapping_tensor_address);

    // read in expert indices

    for (uint32_t i = 0; i < indices_pages; i++) {
        cb_reserve_back(indices_tensor_cb_id, 1);
        uint32_t l1_write_addr = get_write_ptr(indices_tensor_cb_id);
        noc_async_read_page(i, indices_addr_gen, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(indices_tensor_cb_id, 1);
    }

    for (uint32_t i = 0; i < mapping_pages; i++) {
        cb_reserve_back(mapping_tensor_cb_id, 1);
        uint32_t l1_write_addr = get_write_ptr(mapping_tensor_cb_id);
        noc_async_read_page(i, mapping_addr_gen, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(mapping_tensor_cb_id, 1);
    }

    for (uint32_t i = 0; i < input_pages; i++) {
        cb_reserve_back(input_tensor_cb_id, 1);
        uint32_t l1_write_addr = get_write_ptr(input_tensor_cb_id);
        noc_async_read_page(i, input_addr_gen, l1_write_addr);
        noc_async_read_barrier();
        cb_push_back(input_tensor_cb_id, 1);
    }

    noc_semaphore_wait((uint32_t*)global_semaphore_address, tokens_per_device * dispatch_devices);
    noc_semaphore_set((uint32_t*)global_semaphore_address, 0);
    // wait(30000000000);

    DPRINT << "Kernel finished" << ENDL();
}
