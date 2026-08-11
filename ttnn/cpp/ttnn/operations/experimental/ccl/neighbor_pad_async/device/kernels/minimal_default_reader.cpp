// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include <cstdint>
#include "api/tensor/noc_traits.h"

using address_t = uint32_t;

// Compile-time args (uniform across all H fabric reader cores)
constexpr uint32_t cb_output_id = get_compile_time_arg_val(0);
constexpr bool is_padding_zeros = get_compile_time_arg_val(1);
constexpr uint32_t stick_size = get_compile_time_arg_val(2);
// Input TensorAccessorArgs at index 3 (variable length)
constexpr auto src_ct_args = TensorAccessorArgs<3>();
constexpr uint32_t ct_after_src = src_ct_args.next_compile_time_args_offset();
// L1 intermediate config
constexpr bool use_l1_intermediate = get_compile_time_arg_val(ct_after_src);
constexpr uint32_t recv_cb_id = get_compile_time_arg_val(ct_after_src + 1);

template <uint32_t stick_size_bytes>
inline void zeroPad(Noc& noc, CircularBuffer& cb_output) {
    noc.async_write_zeros(cb_output, stick_size_bytes);
    noc.write_zeros_l1_barrier();
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    // Common runtime args (uniform across all cores, updated between dispatches)
    const address_t input_tensor_address = get_common_arg_val<address_t>(0);
    const address_t output_tensor_address = get_common_arg_val<address_t>(1);
    const size_t h_neighbor_sem = get_common_arg_val<uint32_t>(2);

    // Per-core runtime args
    uint32_t arg_idx = 0;
    const uint32_t outer_dim_offset_start_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stick_start_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_halo_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t outer_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_per_halo_dim = get_arg_val<uint32_t>(arg_idx++);
    // Number of corner sticks per pad row in L1 recv buffer (= pad2_left + pad2_right for 2D corners-only)
    const uint32_t num_l1_recv_sticks_per_row = get_arg_val<uint32_t>(arg_idx++);
    // Per-core direction args (moved from compile-time for kernel consolidation)
    const bool is_first_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool is_last_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);

    uint32_t read_size = stick_size;
    const auto src_accessor = TensorAccessor(src_ct_args, input_tensor_address);

    Noc noc_obj;
    CircularBuffer cb_output(cb_output_id);
    CircularBuffer cb_recv(recv_cb_id);

    uint32_t outer_dim_offset = outer_dim_offset_start_id;
    for (uint32_t outer_dim = 0; outer_dim < outer_dim_size; outer_dim++) {
        if (is_first_chip) {
            if (!is_padding_zeros) {
                // Replicate a slice of 1 from input to output
                uint32_t src_stick_id = 0;
                if (direction) {
                    src_stick_id = num_sticks_per_halo_dim * (input_halo_dim_size - 1) + stick_start_id;
                } else {
                    src_stick_id = stick_start_id;
                }
                src_stick_id += outer_dim_offset;
                for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                    cb_output.reserve_back(1);

                    noc_obj.async_read(src_accessor, cb_output, read_size, {.page_id = src_stick_id}, {});

                    src_stick_id++;

                    noc_obj.async_read_barrier();
                    cb_output.push_back(1);
                }
            } else {
                cb_output.reserve_back(1);
                zeroPad<stick_size>(noc_obj, cb_output);
                noc_obj.async_read_barrier();
                cb_output.push_back(1);
            }
        }

        if (!is_last_chip) {
            // Read the "end" of each slice into the CB to write to the neighbor
            for (uint32_t pad_id = padding; pad_id > 0; pad_id--) {
                uint32_t src_stick_id = 0;
                if (direction) {
                    src_stick_id = (padding - pad_id) * num_sticks_per_halo_dim + stick_start_id;
                } else {
                    src_stick_id = (input_halo_dim_size - pad_id) * num_sticks_per_halo_dim + stick_start_id;
                }
                src_stick_id += outer_dim_offset;
                for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                    cb_output.reserve_back(1);

                    noc_obj.async_read(src_accessor, cb_output, read_size, {.page_id = src_stick_id}, {});

                    src_stick_id++;

                    noc_obj.async_read_barrier();
                    cb_output.push_back(1);
                }
            }
        }

        // No local interior copy in this kernel. Dedicated local-copy kernels handle that work.

        outer_dim_offset += (num_sticks_per_halo_dim * input_halo_dim_size);
    }

    // Incoming H halo data from neighbor
    if (!is_first_chip) {
        if constexpr (use_l1_intermediate) {
            // L1 intermediate: fabric delivered H halo data to our L1 recv buffer.
            // Push it into CB for the paired writer to write to output DRAM.
            uint32_t recv_buf_addr = cb_recv.get_write_ptr();
            uint32_t buf_offset = 0;  // Accumulates across all outer_dims (no L1 reuse)

            for (uint32_t od = 0; od < outer_dim_size; od++) {
                // Wait for this outer_dim's data using cumulative count.
                // Using cumulative waits (od+1) instead of resetting to 0 each iteration
                // avoids a race where the writer sends multiple sem incs before the reader
                // processes them — resetting to 0 would discard pending incs and cause a hang.
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h_neighbor_sem), od + 1);

                for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                    // Use num_l1_recv_sticks_per_row (corners-only count) instead of
                    // num_sticks_to_read (full row width) for the L1 recv path.
                    for (uint32_t iter = 0; iter < num_l1_recv_sticks_per_row; iter++) {
                        cb_output.reserve_back(1);
                        uint32_t dst_l1_addr = cb_output.get_write_ptr();
                        noc_async_read(get_noc_addr(recv_buf_addr + buf_offset), dst_l1_addr, stick_size);
                        noc_obj.async_read_barrier();
                        cb_output.push_back(1);
                        buf_offset += stick_size;
                    }
                }
            }
            // Reset after all waits are complete (safe: no more fabric increments expected)
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h_neighbor_sem), 0);
        } else {
            // 1D case: fabric wrote directly to DRAM; just wait for all outer_dims
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h_neighbor_sem), outer_dim_size);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h_neighbor_sem), 0);
        }
    }
}
