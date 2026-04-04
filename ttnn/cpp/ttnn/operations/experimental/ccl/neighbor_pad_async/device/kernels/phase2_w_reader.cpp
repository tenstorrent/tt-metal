// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 2 W fabric reader for fused 2D neighbor pad.
// Reads W boundary sticks directly from the output DRAM tensor (written by Phase 1)
// instead of from an L1 boundary buffer. Phase 1 cores call noc_async_write_barrier()
// before signaling the Phase 2 barrier semaphore, guaranteeing DRAM writes are committed.
//
// DRAM writes are handled by the paired writer (minimal_default_writer).
//
// fabric_only mode: in this mode the output tensor is the compact halo buffer. The W reader
// must gather W-boundary sticks from three sources per T-slice:
//   - H-top halo rows  (h_ext < padding_h):     read from compact halo buffer H-top section
//   - Interior rows    (h_ext in [ph, ph+H_dev)): read from the input tensor directly
//   - H-bot halo rows  (h_ext >= ph + H_dev):    read from compact halo buffer H-bot section
// The gathered sticks feed the W writer, which sends them to the neighbor and writes the
// extended W halo section of the compact buffer.

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include <cstdint>

using address_t = uint32_t;

// Compile-time args (uniform across all W reader cores)
constexpr uint32_t cb_output_id = get_compile_time_arg_val(0);
constexpr bool is_padding_zeros = get_compile_time_arg_val(1);
constexpr uint32_t stick_size = get_compile_time_arg_val(2);
// Output (compact halo) buffer TensorAccessorArgs start at index 3 (variable length)
constexpr auto dst_args = TensorAccessorArgs<3>();
constexpr uint32_t ct_after_dst = dst_args.next_compile_time_args_offset();
// Input tensor TensorAccessorArgs start right after dst_args (used in fabric_only mode)
constexpr auto src_args = TensorAccessorArgs<ct_after_dst>();

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
    // Common runtime args (uniform across all cores, updated between dispatches)
    const address_t output_tensor_address = get_common_arg_val<address_t>(0);
    const uint32_t barrier_sem_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_neighbor_sem_addr = get_common_arg_val<uint32_t>(2);

    // Per-core runtime args
    uint32_t arg_idx = 0;
    const uint32_t outer_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t outer_dim_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t barrier_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_row_width = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pad2_left = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_interior_sticks = get_arg_val<uint32_t>(arg_idx++);
    // Per-core direction args (moved from compile-time for kernel consolidation)
    const bool is_first_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool is_last_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    // fabric_only extension args: non-zero only in fabric_only 2D mode.
    // input_buffer_addr: address of the unpadded input tensor (for interior row reads).
    // h_dev: number of interior H rows in the input tensor (= input_halo_dim_size).
    // h_padding: H padding (ph); top halo rows = [0, h_padding), bot = [h_padding+h_dev, H_total).
    // h_halo_hbot_base: first page of H-bot section in the compact halo buffer
    //                   = outer_dim_size * h_padding * num_interior_sticks.
    const address_t input_buffer_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t h_dev = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t h_padding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t h_halo_hbot_base = get_arg_val<uint32_t>(arg_idx++);

    const bool is_fabric_only = (input_buffer_addr != 0);

    const auto dst_accessor = TensorAccessor(dst_args, output_tensor_address, stick_size);
    const auto src_accessor = TensorAccessor(src_args, input_buffer_addr, stick_size);

    // Wait for Phase 1 to complete.
    if (barrier_count > 0) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr), barrier_count);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr), 0);
    }

    // H_total = h_dev + 2 * h_padding (extended rows per T-slice in fabric_only mode)
    const uint32_t h_total = h_dev + 2u * h_padding;

    // Main loop: read boundary sticks from source buffer(s) → CB for the paired writer.
    for (uint32_t outer_dim = 0; outer_dim < outer_dim_size; outer_dim++) {
        if (is_first_chip) {
            if (!is_padding_zeros) {
                cb_reserve_back(cb_output_id, 1);
                uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                if (is_fabric_only) {
                    // fabric_only: compute source from h_ext and t
                    uint32_t global_row = outer_dim_start + outer_dim;
                    uint32_t t = global_row / h_total;
                    uint32_t h_ext = global_row % h_total;
                    // W-boundary column within a row (no W padding in compact buffer / input tensor)
                    uint32_t w_col = direction ? (num_interior_sticks - 1u) : 0u;
                    uint32_t page;
                    if (h_ext < h_padding) {
                        // H-top halo row: read from compact halo buffer H-top section
                        page = t * h_padding * num_interior_sticks + h_ext * num_interior_sticks + w_col;
                        noc_async_read(get_noc_addr(page, dst_accessor), dst_l1_addr, stick_size);
                    } else if (h_ext < h_padding + h_dev) {
                        // Interior row: read from unpadded input tensor
                        uint32_t h_local = h_ext - h_padding;
                        page = t * h_dev * num_interior_sticks + h_local * num_interior_sticks + w_col;
                        noc_async_read(get_noc_addr(page, src_accessor), dst_l1_addr, stick_size);
                    } else {
                        // H-bot halo row: read from compact halo buffer H-bot section
                        uint32_t h_bot = h_ext - h_padding - h_dev;
                        page = h_halo_hbot_base + t * h_padding * num_interior_sticks + h_bot * num_interior_sticks +
                               w_col;
                        noc_async_read(get_noc_addr(page, dst_accessor), dst_l1_addr, stick_size);
                    }
                } else {
                    // Non-fabric_only: read one boundary stick from output DRAM (standard path).
                    // direction=0: leftmost interior, direction=1: rightmost interior
                    uint32_t row_base = (outer_dim_start + outer_dim) * output_row_width;
                    uint32_t col;
                    if (direction) {
                        col = pad2_left + num_interior_sticks - 1;
                    } else {
                        col = pad2_left;
                    }
                    noc_async_read(get_noc_addr(row_base + col, dst_accessor), dst_l1_addr, stick_size);
                }
                noc_async_read_barrier();
                cb_push_back(cb_output_id, 1);
            } else {
                cb_reserve_back(cb_output_id, 1);
                zeroPad<stick_size>(cb_output_id);
                noc_async_read_barrier();
                cb_push_back(cb_output_id, 1);
            }
        }

        if (!is_last_chip) {
            // Read boundary sticks from source to send to neighbor.
            for (uint32_t pad_id = padding; pad_id > 0; pad_id--) {
                cb_reserve_back(cb_output_id, 1);
                uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                if (is_fabric_only) {
                    uint32_t global_row = outer_dim_start + outer_dim;
                    uint32_t t = global_row / h_total;
                    uint32_t h_ext = global_row % h_total;
                    // For direction=0: send leftmost interior cols (padding - pad_id)th from left
                    // For direction=1: send rightmost interior cols
                    uint32_t w_col;
                    if (direction) {
                        w_col = padding - pad_id;  // leftmost interior (to send left)
                    } else {
                        w_col = num_interior_sticks - pad_id;  // rightmost interior (to send right)
                    }
                    uint32_t page;
                    if (h_ext < h_padding) {
                        page = t * h_padding * num_interior_sticks + h_ext * num_interior_sticks + w_col;
                        noc_async_read(get_noc_addr(page, dst_accessor), dst_l1_addr, stick_size);
                    } else if (h_ext < h_padding + h_dev) {
                        uint32_t h_local = h_ext - h_padding;
                        page = t * h_dev * num_interior_sticks + h_local * num_interior_sticks + w_col;
                        noc_async_read(get_noc_addr(page, src_accessor), dst_l1_addr, stick_size);
                    } else {
                        uint32_t h_bot = h_ext - h_padding - h_dev;
                        page = h_halo_hbot_base + t * h_padding * num_interior_sticks + h_bot * num_interior_sticks +
                               w_col;
                        noc_async_read(get_noc_addr(page, dst_accessor), dst_l1_addr, stick_size);
                    }
                } else {
                    uint32_t row_base = (outer_dim_start + outer_dim) * output_row_width;
                    uint32_t col;
                    if (direction) {
                        // Send leftmost boundary: interior column (padding - pad_id)
                        col = pad2_left + (padding - pad_id);
                    } else {
                        // Send rightmost boundary: interior column (W - pad_id)
                        col = pad2_left + num_interior_sticks - pad_id;
                    }
                    noc_async_read(get_noc_addr(row_base + col, dst_accessor), dst_l1_addr, stick_size);
                }
                noc_async_read_barrier();
                cb_push_back(cb_output_id, 1);
            }
        }
    }

    // Incoming W padding from neighbor: the neighbor's W writer sent padding sticks
    // directly to our output DRAM via fabric. Wait for all sem_incs confirming each
    // outer_dim's data has been sent. The startup barrier in the next dispatch ensures
    // DRAM writes are committed before any device proceeds (barrier goes through the
    // same fabric link as the data, so FIFO ordering guarantees completion).
    if (!is_first_chip) {
        volatile tt_l1_ptr uint32_t* w_neighbor_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_neighbor_sem_addr);
        noc_semaphore_wait_min(w_neighbor_sem_ptr, outer_dim_size);
        // Reset after all waits complete (safe: no more fabric increments expected)
        noc_semaphore_set(w_neighbor_sem_ptr, 0);
    }
}
