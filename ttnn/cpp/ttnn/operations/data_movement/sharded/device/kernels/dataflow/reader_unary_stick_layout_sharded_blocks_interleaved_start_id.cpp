// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

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
    constexpr auto src_args = TensorAccessorArgs<3>();

    const auto s0 = TensorAccessor(src_args, src_addr + aligned_input_width_offset_bytes, stick_size);
    uint32_t stick_id = start_id;
    cb_reserve_back(cb_id_in0, block_height);
    uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
    if (aligned) {
        for (uint32_t h = 0; h < block_height; ++h) {
            uint64_t src_noc_addr = get_noc_addr(stick_id, s0);
            noc_async_read(src_noc_addr, l1_write_addr, block_width_bytes);
            stick_id++;
            l1_write_addr += padded_block_width_bytes;
        }
        noc_async_read_barrier();
    } else {
        constexpr uint32_t num_trids = 2;
        constexpr uint32_t dram_trid_base = 0;
        constexpr uint32_t scratch_trid_base = dram_trid_base + num_trids;

        cb_reserve_back(cb_id_in1, num_trids);
        uint32_t scratch_l1_write_addr_base = get_write_ptr(cb_id_in1);
        uint32_t curr_slot = 0;
        uint32_t slot_to_wait = 0;
        uint32_t num_free_slots = num_trids;

        // Track L1 write addresses for each slot
        uint32_t l1_write_addrs[num_trids];
        uint32_t scratch_l1_write_addrs[num_trids];
        for (uint32_t i = 0; i < num_trids; i++) {
            scratch_l1_write_addrs[i] = scratch_l1_write_addr_base + i * aligned_block_width_bytes;
        }
        for (uint32_t h = 0; h < block_height; ++h) {
            uint32_t scratch_l1_write_addr = scratch_l1_write_addrs[curr_slot];

            l1_write_addrs[curr_slot] = l1_write_addr;

            uint32_t curr_dram_trid = dram_trid_base + curr_slot;

            if (num_free_slots > 0) {
                // First read
                noc_async_read_tile_dram_sharded_set_trid(curr_dram_trid);
                uint64_t src_noc_addr = get_noc_addr(stick_id, s0);
                noc_async_read(src_noc_addr, scratch_l1_write_addr, aligned_block_width_bytes);
                num_free_slots--;
            } else {
                uint32_t wait_dram_trid = dram_trid_base + slot_to_wait;
                uint32_t wait_scratch_trid = scratch_trid_base + slot_to_wait;

                // Wait for oldest DRAM->scratch to complete
                noc_async_read_barrier_with_trid(wait_dram_trid);

                // Issue scratch->L1 for the completed row (with separate trid)
                noc_async_read_tile_dram_sharded_set_trid(wait_scratch_trid);
                uint32_t completed_scratch_addr =
                    scratch_l1_write_addr_base + slot_to_wait * aligned_block_width_bytes;
                uint64_t scratch_l1_noc_read_addr = get_noc_addr(completed_scratch_addr + aligned_offset);
                noc_async_read(scratch_l1_noc_read_addr, l1_write_addrs[slot_to_wait], block_width_bytes);

                // Before reusing this scratch slot, wait for old scratch->L1 to complete
                if (h >= num_trids) {
                    uint32_t old_scratch_trid = scratch_trid_base + curr_slot;
                    noc_async_read_barrier_with_trid(old_scratch_trid);
                }

                // Issue DRAM -> scratch
                noc_async_read_tile_dram_sharded_set_trid(curr_dram_trid);
                uint64_t src_noc_addr = get_noc_addr(stick_id, s0);
                noc_async_read(src_noc_addr, scratch_l1_write_addr, aligned_block_width_bytes);

                slot_to_wait = slot_to_wait == num_trids - 1 ? 0 : (slot_to_wait + 1);
            }

            stick_id++;
            l1_write_addr += padded_block_width_bytes;
            curr_slot = curr_slot == num_trids - 1 ? 0 : (curr_slot + 1);
        }

        // Drain pipeline
        for (uint32_t i = 0; i < num_trids - 1; ++i) {
            uint32_t wait_dram_trid = dram_trid_base + slot_to_wait;
            uint32_t curr_scratch_trid = scratch_trid_base + slot_to_wait;

            noc_async_read_barrier_with_trid(wait_dram_trid);

            noc_async_read_tile_dram_sharded_set_trid(curr_scratch_trid);
            uint32_t completed_scratch_addr = scratch_l1_write_addr_base + slot_to_wait * aligned_block_width_bytes;
            uint64_t scratch_l1_noc_read_addr = get_noc_addr(completed_scratch_addr + aligned_offset);
            noc_async_read(scratch_l1_noc_read_addr, l1_write_addrs[slot_to_wait], block_width_bytes);
            noc_async_read_barrier_with_trid(curr_scratch_trid);

            slot_to_wait = slot_to_wait == num_trids - 1 ? 0 : (slot_to_wait + 1);
        }
    }
    cb_push_back(cb_id_in0, block_height);
}
