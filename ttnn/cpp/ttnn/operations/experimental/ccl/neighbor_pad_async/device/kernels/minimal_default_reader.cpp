// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

using address_t = uint32_t;

constexpr bool is_first_chip = get_compile_time_arg_val(0);
constexpr bool is_last_chip = get_compile_time_arg_val(1);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(2);
constexpr bool direction = get_compile_time_arg_val(3);
constexpr bool is_padding_zeros = get_compile_time_arg_val(4);
constexpr uint32_t stick_size = get_compile_time_arg_val(5);
// Input TensorAccessorArgs at index 6 (variable length)
constexpr auto src_ct_args = TensorAccessorArgs<6>();
constexpr uint32_t ct_after_src = src_ct_args.next_compile_time_args_offset();
// L1 intermediate config
constexpr bool use_l1_intermediate = get_compile_time_arg_val(ct_after_src);
constexpr uint32_t recv_cb_id = get_compile_time_arg_val(ct_after_src + 1);

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
    const address_t output_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t outer_dim_offset_start_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t stick_start_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_halo_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t outer_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_to_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_sticks_per_halo_dim = get_arg_val<uint32_t>(arg_idx++);
    size_t h_neighbor_sem = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pad2_left_sticks = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pad2_right_sticks = get_arg_val<uint32_t>(arg_idx++);
    // W reader signaling args (for 2D corner forwarding)
    const uint32_t w_reader_signal_sem_addr = get_arg_val<uint32_t>(arg_idx++);  // [12]
    const uint32_t num_w_readers_to_signal = get_arg_val<uint32_t>(arg_idx++);   // [13]
    // Up to MAX_W_SIGNAL_TARGETS W reader core NOC coordinates
    constexpr uint32_t MAX_W_SIGNAL_TARGETS = 8;
    uint8_t w_reader_noc_x[MAX_W_SIGNAL_TARGETS];
    uint8_t w_reader_noc_y[MAX_W_SIGNAL_TARGETS];
    for (uint32_t w = 0; w < MAX_W_SIGNAL_TARGETS; w++) {
        w_reader_noc_x[w] = get_arg_val<uint32_t>(arg_idx++);
        w_reader_noc_y[w] = get_arg_val<uint32_t>(arg_idx++);
    }

    uint32_t read_size = stick_size;
    const auto src_accessor = TensorAccessor(src_ct_args, input_tensor_address, stick_size);

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
                    cb_reserve_back(cb_output_id, 1);
                    uint32_t src_buffer_l1_addr = get_write_ptr(cb_output_id);

                    uint64_t src_noc_addr = get_noc_addr(src_stick_id, src_accessor);
                    noc_async_read(src_noc_addr, src_buffer_l1_addr, read_size);

                    src_stick_id++;

                    noc_async_read_barrier();
                    cb_push_back(cb_output_id, 1);
                }
            } else {
                cb_reserve_back(cb_output_id, 1);
                zeroPad<stick_size>(cb_output_id);
                noc_async_read_barrier();
                cb_push_back(cb_output_id, 1);
            }
        }

        if (!is_last_chip) {
            if constexpr (use_l1_intermediate) {
                // 2D: corners-first ordering. Push all corner sticks (left+right) for all
                // pad_ids first, then all non-corner sticks. This lets the paired writer
                // send corners to neighbor L1 and signal sem_inc before processing non-corners.
                uint32_t non_corner_start = pad2_left_sticks;
                uint32_t non_corner_count = num_sticks_to_read - pad2_left_sticks - pad2_right_sticks;
                uint32_t right_start = num_sticks_to_read - pad2_right_sticks;

                // Phase A: corners for all pad_ids
                for (uint32_t pad_id = padding; pad_id > 0; pad_id--) {
                    uint32_t src_base = 0;
                    if (direction) {
                        src_base = (padding - pad_id) * num_sticks_per_halo_dim + stick_start_id;
                    } else {
                        src_base = (input_halo_dim_size - pad_id) * num_sticks_per_halo_dim + stick_start_id;
                    }
                    src_base += outer_dim_offset;
                    // Left corners
                    for (uint32_t c = 0; c < pad2_left_sticks; c++) {
                        cb_reserve_back(cb_output_id, 1);
                        uint32_t src_buffer_l1_addr = get_write_ptr(cb_output_id);
                        noc_async_read(get_noc_addr(src_base + c, src_accessor), src_buffer_l1_addr, read_size);
                        noc_async_read_barrier();
                        cb_push_back(cb_output_id, 1);
                    }
                    // Right corners
                    for (uint32_t c = 0; c < pad2_right_sticks; c++) {
                        cb_reserve_back(cb_output_id, 1);
                        uint32_t src_buffer_l1_addr = get_write_ptr(cb_output_id);
                        noc_async_read(
                            get_noc_addr(src_base + right_start + c, src_accessor), src_buffer_l1_addr, read_size);
                        noc_async_read_barrier();
                        cb_push_back(cb_output_id, 1);
                    }
                }

                // Phase B: non-corners for all pad_ids
                for (uint32_t pad_id = padding; pad_id > 0; pad_id--) {
                    uint32_t src_base = 0;
                    if (direction) {
                        src_base = (padding - pad_id) * num_sticks_per_halo_dim + stick_start_id;
                    } else {
                        src_base = (input_halo_dim_size - pad_id) * num_sticks_per_halo_dim + stick_start_id;
                    }
                    src_base += outer_dim_offset;
                    for (uint32_t c = 0; c < non_corner_count; c++) {
                        cb_reserve_back(cb_output_id, 1);
                        uint32_t src_buffer_l1_addr = get_write_ptr(cb_output_id);
                        noc_async_read(
                            get_noc_addr(src_base + non_corner_start + c, src_accessor), src_buffer_l1_addr, read_size);
                        noc_async_read_barrier();
                        cb_push_back(cb_output_id, 1);
                    }
                }
            } else {
                // 1D: sequential iteration (unchanged)
                for (uint32_t pad_id = padding; pad_id > 0; pad_id--) {
                    uint32_t src_stick_id = 0;
                    if (direction) {
                        src_stick_id = (padding - pad_id) * num_sticks_per_halo_dim + stick_start_id;
                    } else {
                        src_stick_id = (input_halo_dim_size - pad_id) * num_sticks_per_halo_dim + stick_start_id;
                    }
                    src_stick_id += outer_dim_offset;
                    for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                        cb_reserve_back(cb_output_id, 1);
                        uint32_t src_buffer_l1_addr = get_write_ptr(cb_output_id);
                        noc_async_read(get_noc_addr(src_stick_id, src_accessor), src_buffer_l1_addr, read_size);
                        src_stick_id++;
                        noc_async_read_barrier();
                        cb_push_back(cb_output_id, 1);
                    }
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
            uint32_t recv_buf_addr = get_write_ptr(recv_cb_id);
            uint32_t buf_offset = 0;  // Accumulates across all outer_dims (no L1 reuse)

            for (uint32_t od = 0; od < outer_dim_size; od++) {
                // Wait for this outer_dim's data using cumulative count.
                // Using cumulative waits (od+1) instead of resetting to 0 each iteration
                // avoids a race where the writer sends multiple sem incs before the reader
                // processes them — resetting to 0 would discard pending incs and cause a hang.
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h_neighbor_sem), od + 1);

                // Signal W reader cores that corner data for this batch is in L1.
                // W readers can start reading from our L1 recv buffer via NOC immediately.
                // Concurrent reads (us pushing to CB + W reader NOC reading) are safe.
                for (uint32_t w = 0; w < num_w_readers_to_signal; w++) {
                    uint64_t w_sem_noc_addr =
                        get_noc_addr(w_reader_noc_x[w], w_reader_noc_y[w], w_reader_signal_sem_addr);
                    noc_semaphore_inc(w_sem_noc_addr, 1);
                }

                for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                    // Only corner sticks arrive in L1 (non-corners went directly to DRAM).
                    for (uint32_t iter = 0; iter < pad2_left_sticks + pad2_right_sticks; iter++) {
                        cb_reserve_back(cb_output_id, 1);
                        uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                        noc_async_read(get_noc_addr(recv_buf_addr + buf_offset), dst_l1_addr, stick_size);
                        noc_async_read_barrier();
                        cb_push_back(cb_output_id, 1);
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
