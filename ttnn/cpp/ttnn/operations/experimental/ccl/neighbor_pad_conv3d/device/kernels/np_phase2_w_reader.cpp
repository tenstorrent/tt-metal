// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 2 W fabric reader for the fused neighbor_pad + conv3d op.
//
// Reads W-boundary sticks from the input tensor (interior rows) and the compact halo
// buffer (H-padded rows) into a CB that the paired writer ships over the W fabric.
// Per-T-batch, the receiver-side reader increments a progress semaphore on every
// conv3d reader core so conv3d compute can start processing early T-blocks while
// later ones are still in flight.

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include <cstdint>

using address_t = uint32_t;

// Compile-time args (uniform across all W fabric reader cores)
constexpr uint32_t cb_output_id = get_compile_time_arg_val(0);
constexpr bool is_padding_zeros = get_compile_time_arg_val(1);
constexpr uint32_t stick_size = get_compile_time_arg_val(2);
// Halo buffer TensorAccessorArgs start at index 3 (variable length)
constexpr auto dst_args = TensorAccessorArgs<3>();
constexpr uint32_t ct_after_dst = dst_args.next_compile_time_args_offset();
// Input tensor TensorAccessorArgs follow the halo buffer args
constexpr auto src_args = TensorAccessorArgs<ct_after_dst>();
constexpr uint32_t ct_after_src = src_args.next_compile_time_args_offset();
// Granularity (in T-input frames) for per-batch progress-sem signals. Must match the
// conv3d reader's progress_t_batch_size compile-time arg.
constexpr uint32_t progress_t_batch_size = get_compile_time_arg_val(ct_after_src);

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
    // [3] progress_sem_addr: GlobalSemaphore address on conv3d reader cores.
    //     W-reader signals this after w_nbr_sem wait (guarantees W-halo from neighbor
    //     is in compact buffer DRAM).
    // [4] num_reader_cores: number of conv3d reader cores to signal.
    // [5+]: NOC coords of conv3d reader cores.
    const uint32_t progress_sem_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t num_reader_cores = get_common_arg_val<uint32_t>(4);

    // Per-core runtime args
    uint32_t arg_idx = 0;
    const uint32_t outer_dim_size = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t outer_dim_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t barrier_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t output_row_width = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pad2_left = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_interior_sticks = get_arg_val<uint32_t>(arg_idx++);
    const bool is_first_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool is_last_chip = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);
    const address_t input_tensor_address = get_arg_val<address_t>(arg_idx++);
    const uint32_t input_H_dev = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t padding_h = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t h_halo_hbot_base = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t h_total = input_H_dev + 2 * padding_h;

    const auto input_accessor = TensorAccessor(src_args, input_tensor_address, stick_size);
    const auto dst_accessor = TensorAccessor(dst_args, output_tensor_address, stick_size);

    // output_row_width and pad2_left are unused in the fused (fabric-only) path; keep
    // the RTA layout stable so the factory can share code with the standalone NP op.
    (void)output_row_width;
    (void)pad2_left;

    // Wait for Phase 1 to complete.
    if (barrier_count > 0) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr), barrier_count);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr), 0);
    }

    volatile tt_l1_ptr uint32_t* w_neighbor_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_neighbor_sem_addr);

    // outer_dim runs over T_in * h_total; one progress-sem batch = progress_t_batch_size * h_total sticks.
    const uint32_t sticks_per_batch = progress_t_batch_size * h_total;

    // Main loop: read W-boundary sticks → CB for the paired writer.
    for (uint32_t outer_dim = 0; outer_dim < outer_dim_size; outer_dim++) {
        // outer_dim maps to (t, h_padded) where h_padded ∈ [0, h_total).
        // Interior: padding_h ≤ h_padded < padding_h + H_dev.
        const uint32_t global_idx = outer_dim_start + outer_dim;
        const uint32_t t_idx = global_idx / h_total;
        const uint32_t h_padded = global_idx % h_total;
        const bool h_interior = (h_padded >= padding_h && h_padded < padding_h + input_H_dev);

        if (is_first_chip) {
            if (!is_padding_zeros) {
                cb_reserve_back(cb_output_id, 1);
                uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                uint32_t w_col = direction ? (num_interior_sticks - 1) : 0;
                if (h_interior) {
                    uint32_t h_in = h_padded - padding_h;
                    uint32_t page = t_idx * input_H_dev * num_interior_sticks + h_in * num_interior_sticks + w_col;
                    noc_async_read(get_noc_addr(page, input_accessor), dst_l1_addr, stick_size);
                } else {
                    uint32_t halo_page;
                    if (h_padded < padding_h) {
                        uint32_t pad_row = h_padded;
                        halo_page = t_idx * padding_h * num_interior_sticks + pad_row * num_interior_sticks + w_col;
                    } else {
                        uint32_t pad_row = h_padded - padding_h - input_H_dev;
                        halo_page = h_halo_hbot_base + t_idx * padding_h * num_interior_sticks +
                                    pad_row * num_interior_sticks + w_col;
                    }
                    noc_async_read(get_noc_addr(halo_page, dst_accessor), dst_l1_addr, stick_size);
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
            for (uint32_t pad_id = padding; pad_id > 0; pad_id--) {
                cb_reserve_back(cb_output_id, 1);
                uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                uint32_t w_col;
                if (direction) {
                    w_col = padding - pad_id;
                } else {
                    w_col = num_interior_sticks - pad_id;
                }
                if (h_interior) {
                    uint32_t h_in = h_padded - padding_h;
                    uint32_t page = t_idx * input_H_dev * num_interior_sticks + h_in * num_interior_sticks + w_col;
                    noc_async_read(get_noc_addr(page, input_accessor), dst_l1_addr, stick_size);
                } else {
                    uint32_t halo_page;
                    if (h_padded < padding_h) {
                        uint32_t pad_row = h_padded;
                        halo_page = t_idx * padding_h * num_interior_sticks + pad_row * num_interior_sticks + w_col;
                    } else {
                        uint32_t pad_row = h_padded - padding_h - input_H_dev;
                        halo_page = h_halo_hbot_base + t_idx * padding_h * num_interior_sticks +
                                    pad_row * num_interior_sticks + w_col;
                    }
                    noc_async_read(get_noc_addr(halo_page, dst_accessor), dst_l1_addr, stick_size);
                }
                noc_async_read_barrier();
                cb_push_back(cb_output_id, 1);
            }
        }

        // Per-batch signalling — only the receiver-side W-reader signals conv3d.
        // Sender side has no incoming-data wait, so its signal would fire before the
        // remote fabric write for this batch has landed. Receiver signals only after
        // wait_min(w_neighbor_sem, outer_dim+1) — fabric in-order delivery guarantees
        // batch data is in DRAM at that point.
        if (!is_first_chip && (outer_dim + 1) % sticks_per_batch == 0) {
            noc_semaphore_wait_min(w_neighbor_sem_ptr, outer_dim + 1);
            noc_async_write_barrier();
            for (uint32_t i = 0; i < num_reader_cores; i++) {
                const uint32_t rx = get_common_arg_val<uint32_t>(5 + i * 2);
                const uint32_t ry = get_common_arg_val<uint32_t>(5 + i * 2 + 1);
                noc_semaphore_inc(get_noc_addr(rx, ry, progress_sem_addr), 1);
            }
            noc_async_atomic_barrier();
        }
    }

    // Tail for partial last batch (receiver side only).
    if (!is_first_chip && outer_dim_size % sticks_per_batch != 0) {
        noc_semaphore_wait_min(w_neighbor_sem_ptr, outer_dim_size);
        noc_async_write_barrier();
        for (uint32_t i = 0; i < num_reader_cores; i++) {
            const uint32_t rx = get_common_arg_val<uint32_t>(5 + i * 2);
            const uint32_t ry = get_common_arg_val<uint32_t>(5 + i * 2 + 1);
            noc_semaphore_inc(get_noc_addr(rx, ry, progress_sem_addr), 1);
        }
        noc_async_atomic_barrier();
    }
    // Single end-of-kernel reset; sem was monotonically increasing across the loop.
    if (!is_first_chip) {
        noc_semaphore_set(w_neighbor_sem_ptr, 0);
    }
}
