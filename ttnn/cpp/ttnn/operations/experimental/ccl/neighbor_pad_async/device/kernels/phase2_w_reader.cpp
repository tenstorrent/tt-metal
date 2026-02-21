// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Phase 2 W fabric reader for fused 2D neighbor pad.
// Reads W boundary sticks from a local L1 buffer (populated by Phase 1 cores via NOC
// writes) instead of from DRAM. This avoids cross-NOC DRAM visibility issues between
// Phase 1 (BRISC/NOC0 writes) and Phase 2 (NCRISC/NOC1 reads).
//
// Boundary buffer layout per row: [left_0..left_{p-1}, right_0..right_{p-1}]
// where left = leftmost p interior sticks, right = rightmost p interior sticks.
//
// This kernel ONLY reads. All DRAM writes are handled by the paired writer
// (minimal_default_writer on the same core's BRISC).

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include <tt-metalium/buffer_types.hpp>
#include <cstdint>

using address_t = uint32_t;

constexpr bool is_first_chip = get_compile_time_arg_val(0);
constexpr bool is_last_chip = get_compile_time_arg_val(1);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(2);
constexpr bool direction = get_compile_time_arg_val(3);
constexpr bool is_padding_zeros = get_compile_time_arg_val(4);
constexpr uint32_t stick_size = get_compile_time_arg_val(5);
constexpr uint32_t boundary_sticks_per_row = get_compile_time_arg_val(6);
constexpr uint32_t recv_cb_id = get_compile_time_arg_val(7);

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
    const uint32_t padding = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t barrier_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t barrier_count = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t final_sem_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    const uint32_t boundary_buf_addr = get_arg_val<uint32_t>(arg_idx++);

    const uint32_t row_stride = boundary_sticks_per_row * stick_size;

    DPRINT << "W Reader: is_first=" << (uint32_t)is_first_chip << " is_last=" << (uint32_t)is_last_chip
           << " dir=" << (uint32_t)direction << " od=" << outer_dim_size << " pad=" << padding
           << " bc=" << barrier_count << ENDL();
    DPRINT << "W Reader: barrier_sem addr=0x" << HEX() << barrier_sem_addr << DEC()
           << " val=" << *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr) << ENDL();
    if constexpr (!is_first_chip) {
        DPRINT << "W Reader: final_sem addr=0x" << HEX() << final_sem_addr << DEC()
               << " val=" << *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(final_sem_addr) << ENDL();
    }
    DPRINT << "W Reader: boundary_buf=0x" << HEX() << boundary_buf_addr << DEC() << " row_stride=" << row_stride
           << ENDL();

    // barrier_sem is a CreateSemaphore (initialized to 0 at program dispatch). No kernel-side init needed.
    // Wait for Phase 1 to complete.
    if (barrier_count > 0) {
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr), barrier_count);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem_addr), 0);
    }
    DPRINT << "W Reader: barrier passed" << ENDL();

    // Main loop: read boundary sticks from local L1 → CB for the paired writer.
    for (uint32_t outer_dim = 0; outer_dim < outer_dim_size; outer_dim++) {
        uint32_t boundary_row_base = boundary_buf_addr + outer_dim * row_stride;

        if (is_first_chip) {
            if (!is_padding_zeros) {
                // Read one boundary stick from L1; writer replicates it to all padding columns.
                // direction=0: left[0] (leftmost interior), direction=1: right[p-1] (rightmost interior)
                uint32_t stick_offset;
                if (direction) {
                    stick_offset = (2 * padding - 1) * stick_size;  // right[p-1]
                } else {
                    stick_offset = 0;  // left[0]
                }
                cb_reserve_back(cb_output_id, 1);
                uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                noc_async_read(get_noc_addr(boundary_row_base + stick_offset), dst_l1_addr, stick_size);
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
            // Read boundary sticks from L1 to send to neighbor
            for (uint32_t pad_id = padding; pad_id > 0; pad_id--) {
                uint32_t stick_offset;
                if (direction) {
                    // Send leftmost boundary: left[padding - pad_id]
                    stick_offset = (padding - pad_id) * stick_size;
                } else {
                    // Send rightmost boundary: right[padding - pad_id]
                    stick_offset = (padding + (padding - pad_id)) * stick_size;
                }
                cb_reserve_back(cb_output_id, 1);
                uint32_t dst_l1_addr = get_write_ptr(cb_output_id);
                noc_async_read(get_noc_addr(boundary_row_base + stick_offset), dst_l1_addr, stick_size);
                noc_async_read_barrier();
                cb_push_back(cb_output_id, 1);
            }
        }
    }

    // Incoming W padding from neighbor: wait for fabric data in L1 recv buffer, push to CB.
    // The paired writer will pop from CB and write to output DRAM.
    if (!is_first_chip) {
        // final_sem is a CreateSemaphore (initialized to 0 at program dispatch). No kernel-side init needed.
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(final_sem_addr), outer_dim_size);
        noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(final_sem_addr), 0);

        uint32_t recv_buf_addr = get_write_ptr(recv_cb_id);
        uint32_t buf_offset = 0;
        for (uint32_t od = 0; od < outer_dim_size; od++) {
            for (uint32_t pad_id = 0; pad_id < padding; pad_id++) {
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
