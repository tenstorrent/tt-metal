// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Phase 2 W fabric reader for fused 2D neighbor pad.
// Reads W boundary sticks directly from the INPUT tensor for H-interior rows.
// For H-pad rows on H-interior devices, reads corner data from the H reader's L1
// recv buffer via NOC (diagonal corner forwarding: H-axis then W-axis).
// For H-pad rows on H-edge devices with zero-pad mode, outputs zeros directly.
//
// DRAM writes are handled by the paired writer (minimal_default_writer).

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

using address_t = uint32_t;

constexpr bool is_first_chip = get_compile_time_arg_val(0);
constexpr bool is_last_chip = get_compile_time_arg_val(1);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(2);
constexpr bool direction = get_compile_time_arg_val(3);
constexpr bool is_padding_zeros = get_compile_time_arg_val(4);
constexpr uint32_t stick_size = get_compile_time_arg_val(5);
// Input TensorAccessorArgs start at index 6 (variable length)
constexpr auto src_args = TensorAccessorArgs<6>();
constexpr uint32_t ct_after_src = src_args.next_compile_time_args_offset();
constexpr uint32_t recv_cb_id = get_compile_time_arg_val(ct_after_src);

template <uint32_t stick_size_bytes>
inline void zeroPad(uint32_t cb_id) {
    constexpr uint32_t num_full_reads = stick_size_bytes / MEM_ZEROS_SIZE;
    constexpr uint32_t partial_read_size = stick_size_bytes % MEM_ZEROS_SIZE;
    const uint64_t zeros_noc_addr = get_noc_addr(MEM_ZEROS_BASE);
    uint32_t cb_write_addr = get_write_ptr(cb_id);

    for (uint32_t i = 0; i < num_full_reads; ++i) {
        noc_async_read(zeros_noc_addr, cb_write_addr, MEM_ZEROS_SIZE);
        cb_write_addr += MEM_ZEROS_SIZE;
    }
    if (partial_read_size > 0) {
        noc_async_read(zeros_noc_addr, cb_write_addr, partial_read_size);
    }
}

void kernel_main() {
    uint32_t arg_idx = 0;
    const uint32_t outer_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t outer_dim_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t h_corner_dir0_sem_addr = get_arg_val<uint32_t>(arg_idx++);  // [3]
    const uint32_t h_corner_dir1_sem_addr = get_arg_val<uint32_t>(arg_idx++);  // [4]
    const uint32_t w_neighbor_sem_addr = get_arg_val<uint32_t>(arg_idx++);
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t input_row_width = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_halo_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t h_pad_left = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t input_halo_dim_size = get_arg_val<uint32_t>(arg_idx++);
    // H-corner forwarding args
    const uint8_t h_reader_dir0_noc_x = get_arg_val<uint32_t>(arg_idx++);          // [11]
    const uint8_t h_reader_dir0_noc_y = get_arg_val<uint32_t>(arg_idx++);          // [12]
    const uint8_t h_reader_dir1_noc_x = get_arg_val<uint32_t>(arg_idx++);          // [13]
    const uint8_t h_reader_dir1_noc_y = get_arg_val<uint32_t>(arg_idx++);          // [14]
    const bool has_h_dir0_neighbor = get_arg_val<uint32_t>(arg_idx++);             // [15]
    const bool has_h_dir1_neighbor = get_arg_val<uint32_t>(arg_idx++);             // [16]
    const uint32_t h_padding_left = get_arg_val<uint32_t>(arg_idx++);              // [17] H-pad rows per batch (top)
    const uint32_t h_padding_right = get_arg_val<uint32_t>(arg_idx++);             // [18] H-pad rows per batch (bottom)
    const uint32_t h_corner_sticks_per_padrow = get_arg_val<uint32_t>(arg_idx++);  // [19] = pad2_left + pad2_right
    const uint32_t h_corner_left_count = get_arg_val<uint32_t>(arg_idx++);         // [20] = pad2_left

    const auto src_accessor = TensorAccessor(src_args, input_tensor_address, stick_size);

    // H reader L1 recv buffer base address — same L1 offset as our own recv_cb_id
    // because both H and W cores have identical c_in0 configs (same size).
    const uint32_t h_recv_buf_base = get_write_ptr(recv_cb_id);

    // Semaphore pointers for H-corner readiness (on this W reader core's L1)
    volatile tt_l1_ptr uint32_t* h_corner_dir0_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h_corner_dir0_sem_addr);
    volatile tt_l1_ptr uint32_t* h_corner_dir1_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h_corner_dir1_sem_addr);

    // Corner offset within each pad_id's corner group:
    // H reader L1 layout per pad_id: [left_0..left_{P-1}, right_0..right_{P-1}]
    // direction=0 (forward, sends rightmost cols): needs right corners at offset pad2_left
    // direction=1 (backward, sends leftmost cols): needs left corners at offset 0
    const uint32_t corner_group_offset = direction ? 0 : (h_corner_left_count * stick_size);

    // Main loop: read W boundary sticks → CB for the paired writer.
    for (uint32_t outer_dim = 0; outer_dim < outer_dim_size; outer_dim++) {
        // Map output row to input row.
        // Output rows cycle through: [h_pad_left H-pad rows] [input_halo_dim_size interior rows] [h_pad_right rows]
        // within each outer block (of size output_halo_dim_size).
        uint32_t w_row = outer_dim_start + outer_dim;
        uint32_t h_pos = w_row % output_halo_dim_size;
        uint32_t outer_block = w_row / output_halo_dim_size;

        bool is_top_h_pad = (h_pos < h_pad_left);
        bool is_bottom_h_pad = (h_pos >= h_pad_left + input_halo_dim_size);
        bool is_h_pad_row = is_top_h_pad || is_bottom_h_pad;

        // For H-interior rows, compute input row for reading from INPUT tensor
        uint32_t input_h;
        if (is_top_h_pad) {
            input_h = 0;
        } else if (is_bottom_h_pad) {
            input_h = input_halo_dim_size - 1;
        } else {
            input_h = h_pos - h_pad_left;
        }
        uint32_t input_row = outer_block * input_halo_dim_size + input_h;
        uint32_t input_row_base = input_row * input_row_width;

        // Determine if this H-pad row has neighbor data (vs mesh-edge zero-pad)
        bool h_pad_has_neighbor = false;
        if (is_h_pad_row) {
            h_pad_has_neighbor = is_top_h_pad ? has_h_dir0_neighbor : has_h_dir1_neighbor;
        }

        // Self-pad: write W-pad data to local output (W-edge device only)
        if (is_first_chip) {
            if (!is_padding_zeros) {
                // Replicate mode: read one boundary stick from input.
                // For H-pad rows with H-neighbor, the replicate value is the H-neighbor's
                // boundary data — but for now, we read from local input (known limitation
                // for replicate mode on H-interior devices; VAE uses zero-pad only).
                uint32_t col;
                if (direction) {
                    col = input_row_width - 1;
                } else {
                    col = 0;
                }
                cb_reserve_back(cb_output_id, 1);
                uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                noc_async_read(get_noc_addr(input_row_base + col, src_accessor), dst_l1_addr, stick_size);
                noc_async_read_barrier();
                cb_push_back(cb_output_id, 1);
            } else {
                // Zero-pad at W-mesh-edge: always zeros (correct for all H rows)
                cb_reserve_back(cb_output_id, 1);
                zeroPad<stick_size>(cb_output_id);
                noc_async_read_barrier();
                cb_push_back(cb_output_id, 1);
            }
        }

        // Send W boundary sticks to W-neighbor
        if (!is_last_chip) {
            if (is_h_pad_row && !h_pad_has_neighbor && is_padding_zeros) {
                // H-edge device with zero-pad: corners at mesh boundary are zeros
                for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                    cb_reserve_back(cb_output_id, 1);
                    zeroPad<stick_size>(cb_output_id);
                    noc_async_read_barrier();
                    cb_push_back(cb_output_id, 1);
                }
            } else if (is_h_pad_row && h_pad_has_neighbor) {
                // H-interior device (or H-edge with neighbor on this side):
                // Read corner data from H reader's L1 recv buffer via NOC.
                // This is the diagonal corner forwarding path: data traveled H-axis first,
                // now we forward it along the W-axis.

                // Wait for H reader to confirm corners for this batch are in L1
                if (is_top_h_pad) {
                    noc_semaphore_wait_min(h_corner_dir0_sem_ptr, outer_block + 1);
                } else {
                    noc_semaphore_wait_min(h_corner_dir1_sem_ptr, outer_block + 1);
                }

                // Which H reader core has the data?
                uint8_t h_noc_x = is_top_h_pad ? h_reader_dir0_noc_x : h_reader_dir1_noc_x;
                uint8_t h_noc_y = is_top_h_pad ? h_reader_dir0_noc_y : h_reader_dir1_noc_y;

                // Compute pad_id within this direction's H-pad rows
                uint32_t h_pad_id = is_top_h_pad ? h_pos : (h_pos - h_pad_left - input_halo_dim_size);

                // H padding count for this direction
                uint32_t h_padding = is_top_h_pad ? h_padding_left : h_padding_right;

                // L1 offset for this batch + pad_id in the H reader's recv buffer
                // Layout: for each outer_block, for each pad_id:
                //   [left_corner_0..left_{pad2_left-1}, right_corner_0..right_{pad2_right-1}]
                uint32_t h_l1_base =
                    (outer_block * h_padding * h_corner_sticks_per_padrow + h_pad_id * h_corner_sticks_per_padrow) *
                    stick_size;

                // Read `padding` corner sticks from H reader L1
                for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                    cb_reserve_back(cb_output_id, 1);
                    uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                    uint64_t src_noc_addr = get_noc_addr(
                        h_noc_x, h_noc_y, h_recv_buf_base + h_l1_base + corner_group_offset + pad_id * stick_size);
                    noc_async_read(src_noc_addr, dst_l1_addr, stick_size);
                    noc_async_read_barrier();
                    cb_push_back(cb_output_id, 1);
                }
            } else {
                // H-interior row or H-pad row with replicate mode: read from INPUT
                for (uint32_t pad_id = padding; pad_id > 0; pad_id--) {
                    uint32_t col;
                    if (direction) {
                        // Send leftmost boundary columns
                        col = (padding - pad_id);
                    } else {
                        // Send rightmost boundary columns
                        col = input_row_width - pad_id;
                    }
                    cb_reserve_back(cb_output_id, 1);
                    uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                    noc_async_read(get_noc_addr(input_row_base + col, src_accessor), dst_l1_addr, stick_size);
                    noc_async_read_barrier();
                    cb_push_back(cb_output_id, 1);
                }
            }
        }
    }

    // Reset H-corner semaphores after all processing is complete
    // (safe: no more H reader signals expected)
    if (has_h_dir0_neighbor) {
        noc_semaphore_set(h_corner_dir0_sem_ptr, 0);
    }
    if (has_h_dir1_neighbor) {
        noc_semaphore_set(h_corner_dir1_sem_ptr, 0);
    }

    // Incoming W padding from neighbor: wait for fabric data in L1 recv buffer, push to CB.
    // The paired writer will pop from CB and write to output DRAM.
    if (!is_first_chip) {
        volatile tt_l1_ptr uint32_t* w_neighbor_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_neighbor_sem_addr);

        uint32_t recv_buf_addr = get_write_ptr(recv_cb_id);
        uint32_t buf_offset = 0;
        for (uint32_t od = 0; od < outer_dim_size; od++) {
            // Wait for this outer_dim's data using cumulative count
            noc_semaphore_wait_min(w_neighbor_sem_ptr, od + 1);

            for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                cb_reserve_back(cb_output_id, 1);
                uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                noc_async_read(get_noc_addr(recv_buf_addr + buf_offset), dst_l1_addr, stick_size);
                noc_async_read_barrier();
                cb_push_back(cb_output_id, 1);
                buf_offset += stick_size;
            }
        }
        // Reset after all waits complete (safe: no more fabric increments expected)
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_neighbor_sem_addr), 0);
    }
}
