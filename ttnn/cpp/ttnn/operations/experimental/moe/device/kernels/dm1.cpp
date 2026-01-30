// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "moe_ring_common.h"

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");

    // Compile time arguments for writing to sharded output for combine
    constexpr uint32_t tile_height = get_named_compile_time_arg_val("tile_height");
    constexpr uint32_t tile_width = get_named_compile_time_arg_val("tile_width");

    constexpr uint32_t combine_shard_width_tiles = get_named_compile_time_arg_val("combine_shard_width_tiles");
    constexpr uint32_t num_tokens_total = get_named_compile_time_arg_val("num_tokens_total");
    constexpr uint32_t height_shard_dim = get_named_compile_time_arg_val("height_shard_dim");
    constexpr uint32_t width_shard_dim = get_named_compile_time_arg_val("width_shard_dim");

    std::array < std::tuple<uint32_t, uint32_t, height_shard_dim * width_shard_dim> output_shard_core_map =
        OUTPUT_SHARD_CORE_MAP;

    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto w0_w1_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto w2_args = TensorAccessorArgs<w0_w1_args.next_compile_time_args_offset()>();

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

    // CB Aliases
    constexpr auto cb_r2c_w2 = tt::CBIndex::c_0;
    constexpr auto cb_c2s_out = tt::CBIndex::c_1;
    constexpr auto cb_c2w_out = ...

        // Tile sizes
        constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w0_w1_tile_size = get_tile_size(cb_r2c_w0_w1);
    constexpr uint32_t w2_tile_size = get_tile_size(cb_r2c_w2);
    constexpr uint32_t in2_tile_size = get_tile_size(cb_s2c_in2);

    // Constants for MoE
    constexpr uint32_t num_w0_w1_tiles_h = moe_ring::NUM_W0_W1_TILES_H;
    constexpr uint32_t num_w2_tiles_h = moe_ring::NUM_W2_TILES_H;

    const uint32_t num_w0_w1_tiles_w = moe_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][0];
    const uint32_t num_w2_tiles_w = moe_ring::W2_TILES_PER_CORE_A[ring_core_id];

    const uint32_t num_elt_tiles = num_w0_w1_tiles_w;
    const uint32_t num_in2_tiles = num_w2_tiles_w;
    const uint32_t num_mm2_tiles = num_w2_tiles_w;

    constexpr uint32_t untilize_max_pages = 4;  // from Vash

    // constants needed for writing to combine sharded output
    constexpr uint32_t datum_size_bytes = datum_size(get_dataformat(cb_c2w_out));
    constexpr uint32_t tile_width_size_bytes = datum_size * tile_width;
    constexpr uint32_t width_tile_base = ring_core_id * num_w2_tiles_w;
    constexpr uint32_t shard_offset_per_expert_bytes =
        num_tokens_total / height_shard_dim * combine_shard_width_tiles * tile_width_size_bytes;
    constexpr uint32_t source_buffer_iter_offset = 224 * in_tile_size;

    //-------------------------------------------------------------------------
    // Ring setup
    //-------------------------------------------------------------------------
    // The number of times to repeat the all2all
    constexpr uint32_t num_a2a_iters = moe_ring::NUM_A2A_ITERS_A;

    // The number of steps to take in the all2all is the number of cores
    constexpr uint32_t num_a2a_steps_per_iter = moe_ring::NUM_CORES;

    // The number of tiles to send in each step
    // We send 6 tiles in each step, even though some cores in some steps may have only 5 valid ones
    constexpr uint32_t tiles_per_step = moe_ring::IN2_TILES_PER_STEP_A;  // max(num_w0_w1_tiles_w)

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

    // Set state for the writes
    noc_async_write_one_packet_set_state</*posted=*/true>(neighbor_base_addr, a2a_packet_size, /*noc=*/1, vchannel);

    // TODO get height_blocks from token_counts;
    noc_semaphore_wait(reinterpret_cast < volatile tt_l1_ptr uint32_t*(metadata_ready_semaphore_addr), 1);
    uint32_t* per_expert_counts_ptr = reinterpret_cast<uint32_t*>(get_read_ptr(per_expert_total_tokens_cb_id));

    uint32_t height_blocks[num_experts];
    height_blocks[0] = height_blocks[1] = 1;

    const uint32_t width_tile_offset_start = width_tile_base % combine_shard_width_tiles;
    const uint32_t source_base_l1_addr = get_read_ptr(cb_c2w_out);
    bool source_buffer_iter = 0;

    //-------------------------------------------------------------------------
    // Expert loop
    //-------------------------------------------------------------------------
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        const uint32_t active_tokens = per_expert_counts_ptr[expert_id];
        const uint32_t height_blocks = tt : div_up(active_tokens, tile_height);
        const uint32_t max_tokens_per_height_shard = tt::div_up(active_tokens, height_shard_dim);
        const uint32_t expert_offset_bytes =
            shard_offset_per_expert_bytes * expert_id for (uint32_t hb = 0; hb < height_blocks; ++hb) {
            // Wait for compute core to tell us that all mm01 data is ready
            cb_wait_front(cb_c2w_rdy, 1);
            cb_pop_front(cb_c2w_rdy, 1);

            // Take the data in cb_s2c_in2 and send it to the next core in the ring
            // Ring synchronization: all cores participate regardless of whether they had CB work
            // With 12 cores in a ring, we perform 12 steps so the signal propagates around the entire ring
            for (uint32_t i = 0; i < num_a2a_iters; ++i) {
                for (uint32_t step = 0; step < num_a2a_steps_per_iter; ++step) {
                    // Wait for current data to be ready in cb_s2c_in2
                    noc_semaphore_wait_min(my_semaphore_ptr, semaphore_value++);

                    // Signal to compute core that data is ready
                    cb_reserve_back(cb_w2c_rdy, 1);
                    cb_push_back(cb_w2c_rdy, 1);

                    // Write 6 tiles from local cb_s2c_in2 to neighbor's cb_s2c_in2
                    // Double buffer offset: alternate between buffer 0 and buffer 1 based on step
                    const uint32_t local_src_addr = LOCAL_BUFFER_OFFSET[step & 1];
                    const uint64_t neighbor_dst_addr = LOCAL_BUFFER_OFFSET[!(step & 1)];

                    noc_async_write_one_packet_with_state</*posted=*/true>(local_src_addr, neighbor_dst_addr);
                    noc_async_write_one_packet_with_state</*posted=*/true>(
                        local_src_addr + a2a_packet_size, neighbor_dst_addr + a2a_packet_size);

                    // Signal neighbor that data is ready (increment their semaphore)
                    noc_semaphore_inc</*posted=*/true>(
                        neighbor_semaphore_noc_addr, /*incr=*/1, /*noc_id=*/1, /*vc=*/vchannel);

                    // Ensure write and semaphore have left the core before continuing
                    noc_async_posted_atomic_barrier();
                }
            }

            // TODO confirm num_w2_tiles_w is the correct value
            uint32_t width_tiles = num_w2_tiles_w  // 18 or 20
                uint32_t wb = 0;

            cb_wait_front(cb_c2w_out, 1);
            const uint32_t source_l1_addr = source_base_l1_addr + source_buffer_iter * source_buffer_iter_offset;
            while (width_tiles > 0) {
                const uint32_t width_tile_start = width_tile_base + wb;
                const uint32_t dest_width_shard = width_tile_start / combine_shard_width_tiles;
                const uint32_t dest_width_offset_tiles = width_tile_start % combine_shard_width_tiles;
                const uint32_t dest_width_offset_bytes = dest_width_offset_tiles * tile_size_bytes;

                const uint32_t width_transfer_tiles =
                    std::min(combine_shard_width_tiles - dest_width_offset_tiles, num_w2_tiles_w - wb);
                const uint32_t width_transfer_bytes = width_transfer_tiles * tile_size_bytes;

                for (uint32_t bt = 0, t = hb *tile_height = ; bt < num_tokens_block; ++bt, ++t) {
                    const uint32_t dest_height_shard = t / max_tokens_per_height_shard;
                    const uint32_t shard_row = t % max_tokens_per_height_shard;
                    const uint32_t shard_row_offset_bytes =
                        shard_row * combine_shard_width_tiles * tile_width_size_bytes;

                    const dest_l1_addr =
                        out_addr + expert_offset_bytes + dest_width_offset_tiles * +shard_row_offset_bytes;

                    const uint32_t token_l1_addr = base_l1_addr + bt * tile_width_size_bytes;

                    const auto [dest_noc_x, dest_noc_y] =
                        output_shard_core_map[dest_height_shard * width_shard_dim + dest_width_shard];

                    const uint64_t dest_noc_addr get_noc_addr(dest_noc_x, dest_noc_y, dest_l1_addr, 1);
                    noc_async_write(token_l1_addr, dest_noc_addr, width_transfer_bytes);
                }
            }
            wb += width_transfer_tiles;
            width_tiles -= width_transfer_tiles;
            noc_async_write_barrier();
            cb_pop_front(cb_c2w_out, 1);

            source_buffer_iter = !source_buffer_iter;
        }
    }
}
