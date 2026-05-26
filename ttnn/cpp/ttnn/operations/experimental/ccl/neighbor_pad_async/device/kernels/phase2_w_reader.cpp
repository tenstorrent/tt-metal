// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// 3-Phase W fabric reader for fused 2D neighbor pad.
//
// Phase 1 (no wait): reads W boundary sticks for interior H rows from INPUT DRAM and
//   pushes them to CB for the W writer to send to the W neighbor immediately.
//
// Phase 2 (after H barrier): reads W boundary sticks for H-pad rows (corners) from
//   OUTPUT DRAM and pushes them to CB. The barrier_sem is signaled by H writers after
//   they write the H halo to output DRAM.
//
// Phase 3: waits for incoming W padding from the W neighbor via w_neighbor_sem.
//
// This overlaps ~(h_in/h_out) of the W exchange with the H exchange.

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include <cstdint>

using address_t = uint32_t;

// Compile-time args
constexpr uint32_t cb_output_id = get_compile_time_arg_val(0);
constexpr bool is_padding_zeros = get_compile_time_arg_val(1);
constexpr uint32_t stick_size = get_compile_time_arg_val(2);
// Output TensorAccessorArgs start at index 3 (variable length)
constexpr auto dst_args = TensorAccessorArgs<3>();
constexpr uint32_t ct_after_dst = dst_args.next_compile_time_args_offset();
// Input TensorAccessorArgs (for Phase 1 interior rows)
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
    // Common runtime args (uniform across all W cores)
    const address_t output_tensor_address = get_common_arg_val<address_t>(0);
    const uint32_t barrier_sem_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t w_neighbor_sem_addr = get_common_arg_val<uint32_t>(2);
    const address_t input_tensor_address = get_common_arg_val<address_t>(3);

    // Per-core runtime args
    uint32_t arg_idx = 0;
    const uint32_t t_count = get_arg_val<uint32_t>(arg_idx++);       // T batches for this link
    const uint32_t t_start = get_arg_val<uint32_t>(arg_idx++);       // starting T batch (incl. t_front_pad)
    const uint32_t padding = get_arg_val<uint32_t>(arg_idx++);       // W pad amount (left or right)
    const uint32_t barrier_count = get_arg_val<uint32_t>(arg_idx++); // num_h_fabric_cores only
    const uint32_t output_row_width = get_arg_val<uint32_t>(arg_idx++);  // W' = pad2_left + W_in + pad2_right
    const uint32_t pad2_left = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_interior_sticks = get_arg_val<uint32_t>(arg_idx++);  // W_in
    const bool is_first_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool is_last_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t h_pad_top = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t h_in = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t h_pad_bot = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t t_front_pad = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t logical_h = get_arg_val<uint32_t>(arg_idx++);        // 0 = no masking
    const uint32_t device_h_offset = get_arg_val<uint32_t>(arg_idx++);  // device_index * h_in

    const uint32_t h_out = h_pad_top + h_in + h_pad_bot;
    const bool do_h_masking = (logical_h > 0);

    const auto dst_accessor = TensorAccessor(dst_args, output_tensor_address);
    const auto src_accessor = TensorAccessor(src_args, input_tensor_address);

    // =========================================================================
    // Phase 1: interior rows from INPUT DRAM (no barrier wait)
    // =========================================================================
    for (uint32_t t = 0; t < t_count; ++t) {
        uint32_t t_abs = t_start + t;
        bool is_t_front = (t_abs < t_front_pad);
        // t_input: index into input tensor (valid only when !is_t_front)
        uint32_t t_input = is_t_front ? 0u : (t_abs - t_front_pad);

        for (uint32_t h = 0; h < h_in; ++h) {
            // Row is masked (beyond logical_h) if H masking is active and the global H
            // index falls at or past logical_h. This mirrors the fused masking in the
            // local_copy_writer: unmasked input rows must produce zeros in the W exchange
            // just as they do in the local output copy.
            const bool h_masked = do_h_masking && (device_h_offset + h >= logical_h);

            // Base stick index in input tensor for this (t_input, h) row
            uint32_t input_row_base = (t_input * h_in + h) * num_interior_sticks;

            if (is_first_chip) {
                cb_reserve_back(cb_output_id, 1);
                if (is_t_front || h_masked || is_padding_zeros) {
                    zeroPad<stick_size>(cb_output_id);
                    noc_async_read_barrier();
                } else {
                    // direction=0: replicate leftmost input col; direction=1: rightmost
                    uint32_t input_col = direction ? (num_interior_sticks - 1) : 0;
                    uint32_t src_stick = input_row_base + input_col;
                    uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                    noc_async_read(src_accessor.get_noc_addr(src_stick), dst_l1_addr, stick_size);
                    noc_async_read_barrier();
                }
                cb_push_back(cb_output_id, 1);
            }

            if (!is_last_chip) {
                // Inter-device W exchange: always ship real boundary data regardless of
                // is_padding_zeros. is_padding_zeros only gates the tensor-global edge pad
                // (is_first_chip branch above), never inter-device boundaries.
                for (uint32_t pad_id = padding; pad_id > 0; pad_id--) {
                    cb_reserve_back(cb_output_id, 1);
                    if (is_t_front || h_masked) {
                        zeroPad<stick_size>(cb_output_id);
                        noc_async_read_barrier();
                    } else {
                        // direction=0: send rightmost boundary cols (W_in - pad_id)
                        // direction=1: send leftmost boundary cols (padding - pad_id)
                        uint32_t input_col = direction ? (padding - pad_id) : (num_interior_sticks - pad_id);
                        uint32_t src_stick = input_row_base + input_col;
                        uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                        noc_async_read(src_accessor.get_noc_addr(src_stick), dst_l1_addr, stick_size);
                        noc_async_read_barrier();
                    }
                    cb_push_back(cb_output_id, 1);
                }
            }
        }
    }

    // =========================================================================
    // Phase 2: H-pad rows (corners) from OUTPUT DRAM (after H halo received)
    // =========================================================================
    if (barrier_count > 0) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr), barrier_count);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr), 0);
    }

    for (uint32_t t = 0; t < t_count; ++t) {
        uint32_t t_abs = t_start + t;
        bool is_t_front = (t_abs < t_front_pad);

        // Helper lambda-like macro: push one W-boundary stick from output DRAM at output_row
        // In C++ kernel context, use an inline block with a goto-free structure.

        // Process top H-pad rows (h_corner = 0 .. h_pad_top-1)
        for (uint32_t h_corner = 0; h_corner < h_pad_top; ++h_corner) {
            uint32_t output_row = t_abs * h_out + h_corner;
            uint32_t row_base = output_row * output_row_width;

            if (is_first_chip) {
                cb_reserve_back(cb_output_id, 1);
                if (is_t_front || is_padding_zeros) {
                    zeroPad<stick_size>(cb_output_id);
                    noc_async_read_barrier();
                } else {
                    // direction=0: leftmost interior col; direction=1: rightmost
                    uint32_t col = direction ? (pad2_left + num_interior_sticks - 1) : pad2_left;
                    uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                    noc_async_read(dst_accessor.get_noc_addr(row_base + col), dst_l1_addr, stick_size);
                    noc_async_read_barrier();
                }
                cb_push_back(cb_output_id, 1);
            }

            if (!is_last_chip) {
                // Inter-device W exchange: ship real boundary data regardless of is_padding_zeros.
                for (uint32_t pad_id = padding; pad_id > 0; pad_id--) {
                    cb_reserve_back(cb_output_id, 1);
                    if (is_t_front) {
                        zeroPad<stick_size>(cb_output_id);
                        noc_async_read_barrier();
                    } else {
                        uint32_t col = direction ? (pad2_left + (padding - pad_id))
                                                 : (pad2_left + num_interior_sticks - pad_id);
                        uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                        noc_async_read(dst_accessor.get_noc_addr(row_base + col), dst_l1_addr, stick_size);
                        noc_async_read_barrier();
                    }
                    cb_push_back(cb_output_id, 1);
                }
            }
        }

        // Process bottom H-pad rows (h_corner = h_pad_top + h_in .. h_out-1)
        for (uint32_t h_corner = 0; h_corner < h_pad_bot; ++h_corner) {
            uint32_t output_row = t_abs * h_out + h_pad_top + h_in + h_corner;
            uint32_t row_base = output_row * output_row_width;

            if (is_first_chip) {
                cb_reserve_back(cb_output_id, 1);
                if (is_t_front || is_padding_zeros) {
                    zeroPad<stick_size>(cb_output_id);
                    noc_async_read_barrier();
                } else {
                    uint32_t col = direction ? (pad2_left + num_interior_sticks - 1) : pad2_left;
                    uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                    noc_async_read(dst_accessor.get_noc_addr(row_base + col), dst_l1_addr, stick_size);
                    noc_async_read_barrier();
                }
                cb_push_back(cb_output_id, 1);
            }

            if (!is_last_chip) {
                // Inter-device W exchange: ship real boundary data regardless of is_padding_zeros.
                for (uint32_t pad_id = padding; pad_id > 0; pad_id--) {
                    cb_reserve_back(cb_output_id, 1);
                    if (is_t_front) {
                        zeroPad<stick_size>(cb_output_id);
                        noc_async_read_barrier();
                    } else {
                        uint32_t col = direction ? (pad2_left + (padding - pad_id))
                                                 : (pad2_left + num_interior_sticks - pad_id);
                        uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                        noc_async_read(dst_accessor.get_noc_addr(row_base + col), dst_l1_addr, stick_size);
                        noc_async_read_barrier();
                    }
                    cb_push_back(cb_output_id, 1);
                }
            }
        }
    }

    // =========================================================================
    // Phase 3: wait for all incoming W sticks from neighbor
    // =========================================================================
    if (!is_first_chip) {
        volatile tt_l1_ptr uint32_t* w_neighbor_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_neighbor_sem_addr);
        // Wait for t_count * h_out increments (one per row sent by our W neighbor)
        noc_semaphore_wait_min(w_neighbor_sem_ptr, t_count * h_out);
        noc_semaphore_set(w_neighbor_sem_ptr, 0);
    }
}
