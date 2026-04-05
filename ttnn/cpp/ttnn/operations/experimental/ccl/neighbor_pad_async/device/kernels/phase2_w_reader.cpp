// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 2 W fabric reader for fused 2D neighbor pad.
// Reads W boundary sticks directly from the output DRAM tensor (written by Phase 1)
// instead of from an L1 boundary buffer. Phase 1 cores call noc_async_write_barrier()
// before signaling the Phase 2 barrier semaphore, guaranteeing DRAM writes are committed.
//
// DRAM writes are handled by the paired writer (minimal_default_writer).

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include <cstdint>

using address_t = uint32_t;

// Compile-time args (uniform across all W fabric reader cores)
constexpr uint32_t cb_output_id = get_compile_time_arg_val(0);
constexpr bool is_padding_zeros = get_compile_time_arg_val(1);
constexpr uint32_t stick_size = get_compile_time_arg_val(2);
// Output TensorAccessorArgs start at index 3 (variable length)
constexpr auto dst_args = TensorAccessorArgs<3>();
constexpr uint32_t ct_after_dst = dst_args.next_compile_time_args_offset();

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
#if defined(NP_PROGRESS_SEM)
    // [3] progress_sem_addr: GlobalSemaphore address on conv3d reader cores.
    //     W-reader signals this after w_nbr_sem wait (guarantees W-halo from neighbor
    //     is in compact buffer DRAM). All NP_NUM_W_WRITERS W-readers signal once each.
    // [4] num_reader_cores: number of conv3d reader cores to signal.
    // [5+]: NOC coords of conv3d reader cores.
    const uint32_t progress_sem_addr = get_common_arg_val<uint32_t>(3);
    const uint32_t num_reader_cores = get_common_arg_val<uint32_t>(4);
#endif

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

    const auto dst_accessor = TensorAccessor(dst_args, output_tensor_address, stick_size);

    // Wait for Phase 1 to complete.
    if (barrier_count > 0) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr), barrier_count);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr), 0);
    }

    // Main loop: read boundary sticks from output DRAM → CB for the paired writer.
    for (uint32_t outer_dim = 0; outer_dim < outer_dim_size; outer_dim++) {
        uint32_t row_base = (outer_dim_start + outer_dim) * output_row_width;

        if (is_first_chip) {
            if (!is_padding_zeros) {
                // Read one boundary stick from output DRAM; writer replicates it to all padding columns.
                // direction=0: leftmost interior, direction=1: rightmost interior
                uint32_t col;
                if (direction) {
                    col = pad2_left + num_interior_sticks - 1;
                } else {
                    col = pad2_left;
                }
                cb_reserve_back(cb_output_id, 1);
                uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                noc_async_read(get_noc_addr(row_base + col, dst_accessor), dst_l1_addr, stick_size);
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
            // Read boundary sticks from output DRAM to send to neighbor
            for (uint32_t pad_id = padding; pad_id > 0; pad_id--) {
                uint32_t col;
                if (direction) {
                    // Send leftmost boundary: interior column (padding - pad_id)
                    col = pad2_left + (padding - pad_id);
                } else {
                    // Send rightmost boundary: interior column (W - pad_id)
                    col = pad2_left + num_interior_sticks - pad_id;
                }
                cb_reserve_back(cb_output_id, 1);
                uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                noc_async_read(get_noc_addr(row_base + col, dst_accessor), dst_l1_addr, stick_size);
                noc_async_read_barrier();
                cb_push_back(cb_output_id, 1);
            }
        }
    }

    // Incoming W padding from neighbor: the neighbor's W writer sent padding sticks
    // directly to our output DRAM via fabric. Wait for all sem_incs confirming each
    // outer_dim's data has been sent.
    if (!is_first_chip) {
        volatile tt_l1_ptr uint32_t* w_neighbor_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(w_neighbor_sem_addr);
        noc_semaphore_wait_min(w_neighbor_sem_ptr, outer_dim_size);
        // Reset after all waits complete (safe: no more fabric increments expected)
        noc_semaphore_set(w_neighbor_sem_ptr, 0);
    }

#if defined(NP_PROGRESS_SEM)
    // Signal conv3d readers that all compact buffer data (H-halo + W-halo) is committed.
    // H-halo ordering: all sticks go through L1→CB→local DRAM (noc_async_write_barrier()
    // in handle_incoming_writes guarantees commit before Phase 2 signal fires). The
    // fabric_only mode routes ALL sticks through L1 (not just corners) because BH does
    // not provide DRAM ordering between fabric writes and NOC reads.
    noc_async_write_barrier();
    for (uint32_t i = 0; i < num_reader_cores; i++) {
        const uint32_t rx = get_common_arg_val<uint32_t>(5 + i * 2);
        const uint32_t ry = get_common_arg_val<uint32_t>(5 + i * 2 + 1);
        noc_semaphore_inc(get_noc_addr(rx, ry, progress_sem_addr), 1);
    }
    noc_async_atomic_barrier();
#endif
}
