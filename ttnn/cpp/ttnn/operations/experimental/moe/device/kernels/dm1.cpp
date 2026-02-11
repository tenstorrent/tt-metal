// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/tt-metalium/constants.hpp"
#include "moe_ring_common.h"

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");

    // This is a define that is passed in from the program factory
    constexpr std::array<uint32_t, 2 * moe_ring::NUM_COMBINE_CORES> combine_core_map = OUTPUT_SHARD_CORE_MAP;

    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto w0_w1_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto w2_args = TensorAccessorArgs<w0_w1_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto in_addr = get_arg_val<uint32_t>(argidx++);
    const auto w0_w1_addr = get_arg_val<uint32_t>(argidx++);
    const auto w2_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);
    const auto ring_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_core_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_y = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_r2c_w0_w1 = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_rdy = tt::CBIndex::c_3;
    constexpr auto cb_s2c_in2 = tt::CBIndex::c_4;
    constexpr auto cb_c2s_out = tt::CBIndex::c_5;

    // CB Aliases
    constexpr auto cb_r2c_w2 = tt::CBIndex::c_0;

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w0_w1_tile_size = get_tile_size(cb_r2c_w0_w1);
    constexpr uint32_t w2_tile_size = get_tile_size(cb_r2c_w2);
    constexpr uint32_t in2_tile_size = get_tile_size(cb_s2c_in2);

    // Constants for MoE
    constexpr uint32_t num_w0_w1_tiles_h = moe_ring::NUM_W0_W1_TILES_H;
    constexpr uint32_t num_w2_tiles_h = moe_ring::NUM_W2_TILES_H;

    const uint32_t num_w0_w1_tiles_w = moe_ring::W0_W1_TILES_PER_CORE_PER_STEP_B[ring_core_id][0];
    const uint32_t num_w2_tiles_w = moe_ring::W2_TILES_PER_CORE_B[ring_core_id];

    const uint32_t num_elt_tiles = num_w0_w1_tiles_w;
    const uint32_t num_in2_tiles = num_w2_tiles_w;
    const uint32_t num_mm2_tiles = num_w2_tiles_w;

    //-------------------------------------------------------------------------
    // Ring setup
    //-------------------------------------------------------------------------
    // The number of times to repeat the all2all
    constexpr uint32_t num_a2a_iters = moe_ring::NUM_A2A_ITERS_B;

    // The number of steps to take in the all2all is the number of cores
    constexpr uint32_t num_a2a_steps_per_iter = moe_ring::NUM_CORES;

    // The number of tiles to send in each step
    // We send 6 tiles in each step, even though some cores in some steps may have only 5 valid ones
    constexpr uint32_t tiles_per_step = moe_ring::IN2_TILES_PER_STEP_B;  // max(num_w0_w1_tiles_w)

    //-------------------------------------------------------------------------
    // Ring NoC setup
    //-------------------------------------------------------------------------
    uint32_t semaphore_addr = get_semaphore(ring_semaphore_id);
    volatile tt_l1_ptr uint32_t* my_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);
    const uint64_t neighbor_semaphore_noc_addr =
        get_noc_addr(ring_neighbor_physical_x, ring_neighbor_physical_y, semaphore_addr);

    // Size of each transfer in bytes
    constexpr uint32_t a2a_xfer_bytes_per_step = tiles_per_step * in2_tile_size;

    // Split into 2 packets
    constexpr uint32_t a2a_packet_size = a2a_xfer_bytes_per_step / 2;

    // Source and destination addresses for the all2all
    const uint32_t local_base_addr = get_write_ptr(cb_s2c_in2);
    const uint64_t neighbor_base_addr =
        get_noc_addr(ring_neighbor_physical_x, ring_neighbor_physical_y, local_base_addr);

    // Precompute buffer offsets
    uint32_t LOCAL_BUFFER_OFFSET[num_a2a_steps_per_iter];
    for (uint32_t i = 0; i < num_a2a_steps_per_iter; ++i) {
        LOCAL_BUFFER_OFFSET[i] = local_base_addr + i * a2a_xfer_bytes_per_step;
    }
    uint32_t semaphore_value = 0;
    *my_semaphore_ptr = 0;

    //-------------------------------------------------------------------------
    // Write out constants
    //-------------------------------------------------------------------------
    // Each row (token) written to destination is num_w2_tiles_w * 32 * 2 bytes (18 or 19 tiles wide)
    const uint32_t write_out_packet_size = num_w2_tiles_w * tt::constants::TILE_WIDTH * sizeof(uint16_t);  // bf16

    // Source stride: each token in src is 20 tiles wide (20*32*2 bytes), we skip the extra tiles
    constexpr uint32_t write_out_src_stride = 20 * tt::constants::TILE_WIDTH * sizeof(uint16_t);

    // Destination stride: each row at the destination is NUM_COMBINE_TILES_PER_CORE_W tiles wide (full row width).
    // We write num_w2_tiles_w tiles into a column slice of that row, so successive tokens are one full row apart.
    constexpr uint32_t write_out_dst_stride =
        moe_ring::NUM_COMBINE_TILES_PER_CORE_W * tt::constants::TILE_WIDTH * sizeof(uint16_t);

    // W offset at destination for this source core's data
    const uint32_t write_out_dst_offset_w =
        moe_ring::COMBINE_W_OFFSET_PER_CORE_B[ring_core_id] * tt::constants::TILE_WIDTH * sizeof(uint16_t);

    // Base local address at each destination core (before adding w offset)
    const uint32_t write_out_dst_base_addr = get_write_ptr(cb_s2c_in);

    // Expert 1 data goes 32 rows after expert 0 (32 tokens * full row stride)
    constexpr uint32_t write_out_expert_offset = 128 * write_out_dst_stride;

    // 12 ring cores are grouped into 4 groups of 3 (matching COMBINE_W_OFFSET_PER_CORE_B reset pattern).
    // Each group of 3 ring cores writes to the same x-column of combine cores.
    constexpr uint32_t RING_CORES_PER_COMBINE_COL = moe_ring::NUM_CORES / moe_ring::NUM_COMBINE_CORES_W;  // 12/4 = 3
    const uint32_t combine_core_x = ring_core_id / RING_CORES_PER_COMBINE_COL;

    //-------------------------------------------------------------------------
    // Per expert counts
    //-------------------------------------------------------------------------

    // TODO get height_blocks from token_counts;
    // noc_semaphore_wait(reinterpret_cast < volatile tt_l1_ptr uint32_t*(metadata_ready_semaphore_addr), 1);
    // uint32_t* per_expert_counts_ptr = reinterpret_cast<uint32_t*>(get_read_ptr(per_expert_total_tokens_cb_id));

    uint32_t per_expert_counts_ptr[num_experts];
    per_expert_counts_ptr[0] = per_expert_counts_ptr[1] = 32;

    //-------------------------------------------------------------------------
    // Expert loop
    //-------------------------------------------------------------------------
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        // Assume data is already in the CB, but this will get plumbed to tilize in the future.
        cb_reserve_back(cb_s2c_in, num_w0_w1_tiles_h);
        cb_push_back(cb_s2c_in, num_w0_w1_tiles_h);

        // Wait for compute core to tell us that all mm01 data is ready
        cb_wait_front(cb_c2w_rdy, 1);
        cb_pop_front(cb_c2w_rdy, 1);

        // Set state for the data writes
        noc_async_write_one_packet_set_state</*posted=*/true>(neighbor_base_addr, a2a_packet_size, /*noc=*/1, vchannel);

        // Set state for the semaphore write
        noc_inline_dw_write_set_state</*posted=*/true, /*set_val=*/false>(
            neighbor_semaphore_noc_addr, /*val=*/0, /*be=*/0xF, /*cmd_buf=*/write_at_cmd_buf, /*noc=*/1, vchannel);

        // Take the data in cb_s2c_in2 and send it to the next core in the ring
        // Ring synchronization: all cores participate regardless of whether they had CB work
        // With 12 cores in a ring, we perform 12 steps so the signal propagates around the entire ring
        for (uint32_t i = 0; i < num_a2a_iters; ++i) {
            for (uint32_t step = 0; step < num_a2a_steps_per_iter; ++step) {
                // Wait for current data to be ready in cb_s2c_in2
                while ((*my_semaphore_ptr) < semaphore_value) {
                };

                // Signal to compute core that data is ready
                cb_reserve_back(cb_w2c_rdy, 1);
                cb_push_back(cb_w2c_rdy, 1);

                // Write 6 tiles from local cb_s2c_in2 to neighbor's cb_s2c_in2
                // Double buffer offset: alternate between buffer 0 and buffer 1 based on step
                const uint32_t local_src_addr = LOCAL_BUFFER_OFFSET[step];
                const uint64_t neighbor_dst_addr = LOCAL_BUFFER_OFFSET[(step == 11) ? 0 : (step + 1)];

                noc_async_write_one_packet_with_state</*posted=*/true>(local_src_addr, neighbor_dst_addr);
                noc_async_write_one_packet_with_state</*posted=*/true>(
                    local_src_addr + a2a_packet_size, neighbor_dst_addr + a2a_packet_size);

                // Signal neighbor that data is ready (increment their semaphore value)
                noc_inline_dw_write_with_state<
                    /*update_addr_lo=*/false,
                    /*update_counter=*/true,
                    /*posted=*/true,
                    /*update_addr_hi=*/false,
                    /*update_val=*/true>(++semaphore_value);

                // Ensure writes have left the core before continuing
                noc_async_posted_writes_flushed();
            }
        }

        //-------------------------------------------------------------------------
        // Write out to combine cores
        //-------------------------------------------------------------------------
        cb_wait_front(cb_c2s_out, num_w0_w1_tiles_h);

        uint32_t src_addr = get_read_ptr(cb_c2s_out);

        const uint32_t num_tokens = per_expert_counts_ptr[expert_id];
        const uint32_t num_full_blocks = num_tokens / moe_ring::NUM_TOKENS_PER_CORE;
        const uint32_t remainder_tokens = num_tokens % moe_ring::NUM_TOKENS_PER_CORE;
        // Total height blocks: full blocks + 1 partial block if there are remainder tokens
        const uint32_t num_height_blocks = num_full_blocks + (remainder_tokens > 0 ? 1 : 0);

        // Destination address for this expert at each core:
        //   base + w_offset + expert_id * expert_offset
        uint32_t dst_local_addr_base =
            write_out_dst_base_addr + write_out_dst_offset_w + expert_id * write_out_expert_offset;

        // y-core index: we start at 0 and advance by 1 every 8 tokens, wrapping at NUM_COMBINE_CORES_H (4)
        uint32_t combine_core_y_idx = 0;

        // Track how many groups of 8 tokens have been written to each y-core so far,
        // so that wrap-around writes continue after previously placed tokens.
        uint32_t y_core_visit_count[moe_ring::NUM_COMBINE_CORES_H] = {0};

        for (uint32_t height_block = 0; height_block < num_height_blocks; ++height_block) {
            // Compute the combine core map index: x + y * NUM_COMBINE_CORES_W
            uint32_t combine_core_idx = combine_core_x + combine_core_y_idx * moe_ring::NUM_COMBINE_CORES_W;

            // Pick the dst core for this height block
            const uint32_t dest_noc_x = combine_core_map[2 * combine_core_idx];
            const uint32_t dest_noc_y = combine_core_map[2 * combine_core_idx + 1];

            // Destination address: offset by how many groups of 8 we've already written to this y-core
            uint32_t dst_addr = dst_local_addr_base + y_core_visit_count[combine_core_y_idx] *
                                                          moe_ring::NUM_TOKENS_PER_CORE * write_out_dst_stride;

            // Set state for the data writes to this core
            uint64_t combine_dst_noc_addr = get_noc_addr(dest_noc_x, dest_noc_y, dst_addr);
            noc_async_write_one_packet_set_state</*posted=*/true>(
                combine_dst_noc_addr, write_out_packet_size, /*noc=*/1, vchannel);

            // Last block may be partial if num_tokens is not a multiple of 8
            const uint32_t tokens_this_block =
                (height_block == num_full_blocks) ? remainder_tokens : moe_ring::NUM_TOKENS_PER_CORE;

            for (uint32_t h = 0; h < tokens_this_block; ++h) {
                noc_async_write_one_packet_with_state</*posted=*/true>(src_addr, dst_addr);

                // Advance src by full token width in source (20 tiles)
                src_addr += write_out_src_stride;
                // Advance dst by one full row at the destination
                dst_addr += write_out_dst_stride;
            }

            // Record that we've written one group of 8 tokens to this y-core
            y_core_visit_count[combine_core_y_idx]++;

            // Move to the next y-core, wrapping around after NUM_COMBINE_CORES_H (4)
            combine_core_y_idx = (combine_core_y_idx + 1) % moe_ring::NUM_COMBINE_CORES_H;
        }
        cb_pop_front(cb_c2s_out, num_w0_w1_tiles_h);
    }  // end expert loop

    // Ensure all data writes have landed before signaling
    noc_async_posted_writes_flushed();

    // Atomically increment the semaphore at ALL destination combine cores in our x-column,
    // regardless of whether we wrote data to them, so they can wait for completion from all sources.
    for (uint32_t y = 0; y < moe_ring::NUM_COMBINE_CORES_H; ++y) {
        uint32_t idx = combine_core_x + y * moe_ring::NUM_COMBINE_CORES_W;
        uint64_t dest_sem_noc_addr =
            get_noc_addr(combine_core_map[2 * idx], combine_core_map[2 * idx + 1], semaphore_addr);
        noc_semaphore_inc(dest_sem_noc_addr, 1, /*noc_id=*/1, vchannel);
    }

    // TODO: Barrier to make sure the increments left the core before exiting.
}
