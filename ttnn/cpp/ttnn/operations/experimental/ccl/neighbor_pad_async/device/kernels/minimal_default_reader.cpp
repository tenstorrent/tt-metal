// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;
using ttnn::ccl::Topology;

constexpr bool is_first_chip = get_compile_time_arg_val(0);
constexpr bool is_last_chip = get_compile_time_arg_val(1);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(2);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    // Load the input tensor spec
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t stick_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stick_start_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stick_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_per_core_read = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);

    constexpr auto src_args = TensorAccessorArgs<3>();
    uint32_t read_size = stick_size;
    const auto src_accessor = TensorAccessor(src_args, input_tensor_address, stick_size);

    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto dst_accessor = TensorAccessor(dst_args, output_tensor_address, stick_size);

    if (is_first_chip) {
        // Replicate a slice of 1 from input to output
        uint32_t src_stick_id = stick_start_id;
        for (uint32_t iter = 0; iter < stick_offset; ++iter) {
            cb_reserve_back(cb_output_id, 1);
            uint32_t src_buffer_l1_addr = get_write_ptr(cb_output_id);

            uint64_t src_noc_addr = get_noc_addr(src_stick_id, src_accessor);
            noc_async_read(src_noc_addr, src_buffer_l1_addr, read_size);

            src_stick_id++;

            noc_async_read_barrier();
            cb_push_back(cb_output_id, 1);
        }
    }

    if (!is_last_chip) {
        // Read the "end" of each slice into the CB to write to the neighbor
        uint32_t src_stick_id = num_sticks_per_core_read - (stick_offset * padding);
        for (uint32_t iter = 0; iter < stick_offset * padding; ++iter) {
            cb_reserve_back(cb_output_id, 1);
            uint32_t src_buffer_l1_addr = get_write_ptr(cb_output_id);

            uint64_t src_noc_addr = get_noc_addr(src_stick_id, src_accessor);
            noc_async_read(src_noc_addr, src_buffer_l1_addr, read_size);

            src_stick_id++;

            noc_async_read_barrier();
            cb_push_back(cb_output_id, 1);
        }
    }

    // Copy the entire input
    uint32_t src_stick_id = stick_start_id;
    for (uint32_t iter = 0; iter < num_sticks_per_core_read; ++iter) {
        cb_reserve_back(cb_output_id, 1);
        uint32_t src_buffer_l1_addr = get_write_ptr(cb_output_id);

        uint64_t src_noc_addr = get_noc_addr(src_stick_id, src_accessor);
        noc_async_read(src_noc_addr, src_buffer_l1_addr, read_size);

        src_stick_id++;

        noc_async_read_barrier();
        cb_push_back(cb_output_id, 1);
    }

    // Check that the semaphore is received
    if (!is_first_chip) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 1);
    }
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
}
