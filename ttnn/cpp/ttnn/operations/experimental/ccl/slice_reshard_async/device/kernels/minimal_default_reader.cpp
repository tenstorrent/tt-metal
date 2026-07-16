// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
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
inline void zeroPad(const Noc& noc, uint32_t cb_output_id) {
    // Zero-fill the CB's current write entry via the device-side zero API. Self-contained:
    // it waits for the zero (write_zeros_l1_barrier), so the caller needs no separate barrier.
    DataflowBuffer cb(cb_output_id);
    noc.async_write_zeros(cb, stick_size_bytes);
    noc.write_zeros_l1_barrier();
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
    const auto src_accessor = TensorAccessor(src_args, input_tensor_address);

    Noc noc_obj;
    DataflowBuffer cb_output(cb_output_id);

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
                cb_output.reserve_back(1);
                noc_obj.async_read(src_accessor, cb_output, read_size, {.page_id = src_stick_id}, {});

                src_stick_id++;

                noc_obj.async_read_barrier();
                cb_output.push_back(1);
            }
        }
    } else {
        // If we need extend beyond the original input tensor, pad
        if (direction) {
            if (outer_dims_from_forward) {
                cb_output.reserve_back(1);
                zeroPad<stick_size>(noc_obj, cb_output_id);
                cb_output.push_back(1);
            }
        }
    }

    if (direction) {
        for (uint32_t outer_dim_id = outer_dims_to_keep_start; outer_dim_id <= outer_dims_to_keep_end; outer_dim_id++) {
            uint32_t src_stick_id = outer_dim_id * num_sticks_per_outer_dim + stick_start_id;
            for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                cb_output.reserve_back(1);
                noc_obj.async_read(src_accessor, cb_output, read_size, {.page_id = src_stick_id}, {});

                src_stick_id++;

                noc_obj.async_read_barrier();
                cb_output.push_back(1);
            }
        }
    }

    // Check that the semaphore is received
    // Device 2.0 migration: legacy primitive retained, out_ready_sem is the address of a GlobalSemaphore.
    if (!is_first_chip) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 1);
    }

    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
}
