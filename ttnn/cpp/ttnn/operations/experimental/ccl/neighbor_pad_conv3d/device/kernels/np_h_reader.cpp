// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <cstdint>

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
// Dedicated send CB for the batched H-halo send (separate ring from cb_output_id so the
// per-stick recv/is_first pushes don't desync the row reserves).
constexpr uint32_t send_cb_id = get_compile_time_arg_val(ct_after_src + 2);

template <uint32_t stick_size_bytes>
inline void zeroPadCb(uint32_t cb_output_id) {
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

    uint32_t outer_dim_offset = outer_dim_offset_start_id;
    uint32_t recv_buf_addr = 0;
    uint32_t buf_offset = 0;  // recv: accumulates across all outer_dims (no L1 reuse)
    if constexpr (use_l1_intermediate) {
        if (!is_first_chip) {
            recv_buf_addr = get_write_ptr(recv_cb_id);
        }
    }
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
                zeroPadCb<stick_size>(cb_output_id);
                noc_async_read_barrier();
                cb_push_back(cb_output_id, 1);
            }
        }

        if (!is_last_chip) {
            // Read the "end" of each slice into the dedicated send CB. Batch the whole row: one
            // cb_reserve + one barrier for all num_sticks_to_read sticks instead of per stick. The
            // per-stick path issued ~18k read+barrier pairs and was latency-bound; coalescing the
            // row makes it bandwidth-bound. send_cb_id holds exactly 2 rows so a row reserve never
            // wraps mid-batch, and the paired writer drains it one stick at a time.
            for (uint32_t pad_id = padding; pad_id > 0; pad_id--) {
                uint32_t src_stick_id = 0;
                if (direction) {
                    src_stick_id = (padding - pad_id) * num_sticks_per_halo_dim + stick_start_id;
                } else {
                    src_stick_id = (input_halo_dim_size - pad_id) * num_sticks_per_halo_dim + stick_start_id;
                }
                src_stick_id += outer_dim_offset;
                cb_reserve_back(send_cb_id, num_sticks_to_read);
                uint32_t row_base_l1_addr = get_write_ptr(send_cb_id);
                for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                    uint64_t src_noc_addr = get_noc_addr(src_stick_id + iter, src_accessor);
                    noc_async_read(src_noc_addr, row_base_l1_addr + iter * stick_size, read_size);
                }
                noc_async_read_barrier();
                cb_push_back(send_cb_id, num_sticks_to_read);
            }
        }

        // No local interior copy in this kernel. Dedicated local-copy kernels handle that work.

        outer_dim_offset += (num_sticks_per_halo_dim * input_halo_dim_size);

        // Per-batch H-commit: pull this od's incoming H-halo from the L1 recv buffer into the CB as
        // soon as its sender link delivers it (per-core cumulative count, 1 inc/outer_dim), so the
        // paired writer commits + signals HT/HB this batch rather than after the whole send pass.
        if constexpr (use_l1_intermediate) {
            if (!is_first_chip) {
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h_neighbor_sem), outer_dim + 1);
                for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
                    // num_l1_recv_sticks_per_row (corners-only) not num_sticks_to_read (full row).
                    for (uint32_t iter = 0; iter < num_l1_recv_sticks_per_row; iter++) {
                        cb_reserve_back(cb_output_id, 1);
                        uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                        noc_async_read(get_noc_addr(recv_buf_addr + buf_offset), dst_l1_addr, stick_size);
                        noc_async_read_barrier();
                        cb_push_back(cb_output_id, 1);
                        buf_offset += stick_size;
                    }
                }
            }
        }
    }

    // Drain the per-core fabric-arrival sem. 2D recvs were interleaved above; 1D wrote straight to
    // DRAM so just wait for all outer_dims here, then reset.
    if (!is_first_chip) {
        if constexpr (use_l1_intermediate) {
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h_neighbor_sem), 0);
        } else {
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h_neighbor_sem), outer_dim_size);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(h_neighbor_sem), 0);
        }
    }
}
