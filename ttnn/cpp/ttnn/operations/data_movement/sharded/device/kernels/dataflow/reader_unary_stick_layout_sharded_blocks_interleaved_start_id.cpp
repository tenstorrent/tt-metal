// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {

    const uint32_t src_addr                 = get_arg_val<uint32_t>(0);
    const uint32_t block_height             = get_arg_val<uint32_t>(2);
    const uint32_t block_width_bytes        = get_arg_val<uint32_t>(3);
    const uint32_t padded_block_width_bytes = get_arg_val<uint32_t>(4);
    const bool aligned                      = static_cast<bool>(get_arg_val<uint32_t>(5));
    const uint32_t aligned_input_width_offset_bytes = get_arg_val<uint32_t>(6);
    const uint32_t aligned_block_width_bytes = get_arg_val<uint32_t>(7);
    const uint32_t aligned_offset           = get_arg_val<uint32_t>(8);
    const uint32_t start_id                 = get_arg_val<uint32_t>(9);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(1);
    constexpr uint32_t stick_size = get_compile_time_arg_val(2);
    constexpr uint32_t num_trids = get_compile_time_arg_val(3);
    constexpr auto src_args = TensorAccessorArgs<4>();

    const auto s0 = TensorAccessor(src_args, src_addr + aligned_input_width_offset_bytes, stick_size);
    uint32_t stick_id = start_id;
    cb_reserve_back(cb_id_in0, block_height);
    uint32_t dest_write_addr = get_write_ptr(cb_id_in0);
    if (aligned) {
        for (uint32_t h = 0; h < block_height; ++h) {
            uint64_t src_noc_addr = get_noc_addr(stick_id, s0);
            noc_async_read(src_noc_addr, dest_write_addr, block_width_bytes);
            stick_id++;
            dest_write_addr += padded_block_width_bytes;
        }
        noc_async_read_barrier();
    } else {
        enum SlotState : uint8_t {
            IDLE = 0,
            SRC_PENDING = 1,
            SCRATCH_READY = 2,
            SCRATCH_PENDING = 3
        };

        constexpr uint32_t trid_base = 1;

        cb_reserve_back(cb_id_in1, num_trids);
        uint32_t scratch_write_addr_base = get_write_ptr(cb_id_in1);
        uint32_t scratch_cb_page_size = get_local_cb_interface(cb_id_in1).fifo_page_size;
        SlotState slot_states[num_trids];
        uint32_t dest_write_addrs[num_trids];
        uint32_t scratch_write_addrs[num_trids];

        // Initialize slots
        for (uint32_t i = 0; i < num_trids; i++) {
            slot_states[i] = SlotState::IDLE;
            scratch_write_addrs[i] = scratch_write_addr_base + i * scratch_cb_page_size;
        }

        uint32_t rows_issued = 0;      // Number of src->scratch transfers started
        uint32_t rows_completed = 0;   // Number of scratch->dest transfers completed

        while (rows_completed < block_height) {
            for (uint32_t slot = 0; slot < num_trids; slot++) {
                uint32_t active_trid = trid_base + slot;

                if (slot_states[slot] == SlotState::IDLE && rows_issued < block_height) {
                    // Start new src->scratch transfer
                    noc_async_read_set_trid(active_trid);
                    uint64_t src_noc_addr = get_noc_addr(stick_id, s0);
                    noc_async_read(src_noc_addr, scratch_write_addrs[slot], aligned_block_width_bytes);
                    dest_write_addrs[slot] = dest_write_addr;
                    slot_states[slot] = SlotState::SRC_PENDING;

                    stick_id++;
                    dest_write_addr += padded_block_width_bytes;
                    rows_issued++;
                }
                if (slot_states[slot] == SlotState::SRC_PENDING) {
                    // Check if src->scratch is complete
                    if (ncrisc_noc_read_with_transaction_id_flushed(noc_index, active_trid) == 1) {
                        slot_states[slot] = SlotState::SCRATCH_READY;
                    }
                }
                if (slot_states[slot] == SlotState::SCRATCH_READY) {
                    // Start scratch->dest transfer
                    noc_async_read_set_trid(active_trid);

                    uint64_t scratch_noc_read_addr = get_noc_addr(scratch_write_addrs[slot] + aligned_offset);
                    noc_async_read(scratch_noc_read_addr, dest_write_addrs[slot], block_width_bytes);

                    slot_states[slot] = SlotState::SCRATCH_PENDING;
                }
                if (slot_states[slot] == SlotState::SCRATCH_PENDING) {
                    // Check if scratch->dest is complete
                    if (ncrisc_noc_read_with_transaction_id_flushed(noc_index, active_trid) == 1) {
                        slot_states[slot] = SlotState::IDLE;
                        rows_completed++;
                    }
                }
            }
        }

    }
    noc_async_read_set_trid(0);
    cb_push_back(cb_id_in0, block_height);
}
