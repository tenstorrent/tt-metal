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
constexpr bool direction = get_compile_time_arg_val(3);
const uint32_t stick_size = get_compile_time_arg_val(4);

template <uint32_t stick_size_bytes>
inline void zeroPad(uint32_t cb_output_id) {
    //  Zero-fill from MEM_ZEROS
    constexpr uint32_t num_full_reads = stick_size_bytes / MEM_ZEROS_SIZE;
    constexpr uint32_t partial_read_size = stick_size_bytes % MEM_ZEROS_SIZE;
    const uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t cb_write_addr = get_write_ptr(cb_output_id);

    for (uint32_t i = 0; i < num_full_reads; ++i) {
        noc_async_read(zeros_noc_addr, cb_write_addr, MEM_ZEROS_SIZE);
        cb_write_addr += MEM_ZEROS_SIZE;
    }
    if (partial_read_size > 0) {
        noc_async_read(zeros_noc_addr, cb_write_addr, partial_read_size);
    }
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    uint32_t arg_idx = 0;
    // Load the input tensor spec
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t stick_start_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_outer_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t outer_dims_to_forward = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t outer_dims_from_forward = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t outer_dims_to_keep_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t outer_dims_to_keep_end = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_per_outer_dim = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);

    constexpr auto src_args = TensorAccessorArgs<5>();
    uint32_t read_size = stick_size;
    const auto src_accessor = TensorAccessor(src_args, input_tensor_address, stick_size);

    if (!is_last_chip) {
        // Read the "end" of each slice into the CB to write to the neighbor
        for (uint32_t outer_dim_id = outer_dims_to_forward; outer_dim_id > 0; outer_dim_id--) {
            uint32_t src_stick_id = 0;
            if (direction) {
                src_stick_id = (input_outer_dim_size - outer_dim_id) * num_sticks_per_outer_dim + stick_start_id;
            } else {
                src_stick_id = (outer_dims_to_forward - outer_dim_id) * num_sticks_per_outer_dim + stick_start_id;
            }
            for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                cb_reserve_back(cb_output_id, 1);
                uint32_t src_buffer_l1_addr = get_write_ptr(cb_output_id);

                uint64_t src_noc_addr = get_noc_addr(src_stick_id, src_accessor);
                noc_async_read(src_noc_addr, src_buffer_l1_addr, read_size);

                src_stick_id++;

                noc_async_read_barrier();
                cb_push_back(cb_output_id, 1);
            }
        }
    } else {
        // If we need extend beyond the original input tensor, pad
        if (direction) {
            if (outer_dims_from_forward) {
                cb_reserve_back(cb_output_id, 1);
                zeroPad<stick_size>(cb_output_id);
                noc_async_read_barrier();
                cb_push_back(cb_output_id, 1);
            }
        }
    }

    if (direction) {
        for (uint32_t outer_dim_id = outer_dims_to_keep_start; outer_dim_id <= outer_dims_to_keep_end; outer_dim_id++) {
            uint32_t src_stick_id = outer_dim_id * num_sticks_per_outer_dim + stick_start_id;
            for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                cb_reserve_back(cb_output_id, 1);
                uint32_t src_buffer_l1_addr = get_write_ptr(cb_output_id);

                uint64_t src_noc_addr = get_noc_addr(src_stick_id, src_accessor);
                noc_async_read(src_noc_addr, src_buffer_l1_addr, read_size);

                src_stick_id++;

                noc_async_read_barrier();
                cb_push_back(cb_output_id, 1);
            }
        }
    }

    // Check that the semaphore is received
    if (!is_first_chip) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 1);
    }

    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
}
