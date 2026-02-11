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

    // For synchronization with tilize cores
    constexpr uint32_t metadata_ready_semaphore_id = get_named_compile_time_arg_val("metadata_ready_semaphore_id");
    constexpr uint32_t matmul_chunk_ready_semaphore_id =
        get_named_compile_time_arg_val("matmul_chunk_ready_semaphore_id");
    constexpr uint32_t matmul_chunk_available_semaphore_id =
        get_named_compile_time_arg_val("matmul_chunk_available_semaphore_id");
    constexpr uint32_t per_expert_total_tokens_cb_id = get_named_compile_time_arg_val("per_expert_total_tokens_cb_id");
    constexpr uint32_t tokens_per_chunk = get_named_compile_time_arg_val("tokens_per_chunk");
    constexpr uint32_t tilize_drain_core_noc_x = get_named_compile_time_arg_val("tilize_drain_core_noc_x");
    constexpr uint32_t tilize_drain_core_noc_y = get_named_compile_time_arg_val("tilize_drain_core_noc_y");
    
    // Compile time arguments for writing to sharded output for combine
    constexpr uint32_t tile_height = get_named_compile_time_arg_val("tile_height");
    constexpr uint32_t tile_width = get_named_compile_time_arg_val("tile_width");
    constexpr uint32_t tile_width_size_bytes = get_named_compile_time_arg_val("tile_width_size_bytes");

    constexpr uint32_t combine_shard_width_tiles = get_named_compile_time_arg_val("combine_shard_width_tiles");
    constexpr uint32_t num_tokens_total = get_named_compile_time_arg_val("num_tokens_total");
    constexpr uint32_t height_shard_dim = get_named_compile_time_arg_val("height_shard_dim");
    constexpr uint32_t width_shard_dim = get_named_compile_time_arg_val("width_shard_dim");

    std::array<uint32_t, 2 * height_shard_dim * width_shard_dim> output_shard_core_map = OUTPUT_SHARD_CORE_MAP;

    constexpr auto w0_w1_args = TensorAccessorArgs<0>();
    constexpr auto w2_args = TensorAccessorArgs<w0_w1_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();

    // Run-time arguments
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto w0_w1_addr = get_arg_val<uint32_t>(argidx++);
    const auto w2_addr = get_arg_val<uint32_t>(argidx++);
    const auto out_addr = get_arg_val<uint32_t>(argidx++);
    const auto ring_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_core_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_y = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_s2c_in = tt::CBIndex::c_0;
    constexpr auto cb_r2c_w0_w1 = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_rdy = tt::CBIndex::c_3;
    constexpr auto cb_s2c_in2 = tt::CBIndex::c_4;
    constexpr auto cb_w2c_md = tt::CBIndex::c_5;

    // CB Aliases
    constexpr auto cb_c2s_out = tt::CBIndex::c_0;
    constexpr auto cb_r2c_w2 = tt::CBIndex::c_1;

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
    
    // constants needed for writing to combine sharded output
    constexpr uint32_t shard_offset_per_expert_bytes =
        num_tokens_total / height_shard_dim * combine_shard_width_tiles * tile_width_size_bytes;
    const uint32_t output_base_l1_addr = get_write_ptr(cb_s2c_in);
    constexpr uint32_t source_width_tiles = 20;  // token segments/core are all padded up to 20
    const uint32_t output_width_tiles_core = moe_ring::W2_TILES_PER_CORE_A[ring_core_id];

    const uint32_t width_tile_base = detail::accumulate(moe_ring::W2_TILES_PER_CORE_A, ring_core_id);

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

    // Set state for the data writes
    noc_async_write_one_packet_set_state</*posted=*/true>(neighbor_base_addr, a2a_packet_size, /*noc=*/1, vchannel);

    //-------------------------------------------------------------------------
    // Init synchronization with tilize cores
    //-------------------------------------------------------------------------

    // Receive number of tokens per expert from the tilize cores
    uint32_t metadata_ready_semaphore_addr = get_semaphore(metadata_ready_semaphore_id);
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata_ready_semaphore_addr), 1);

    // Signal to the compute core that num_tokens_per_expert has arrived.
    // We also use this CB to transfer (from the writer to compute) 2 semaphore addresses:
    // - 0: address of semaphore used to send metadata (number of tokens per expert)
    // - 1: address of semaphore used to notify matmuls cores that tilized chunks have arrived
    volatile tt_l1_ptr uint32_t* cb_w2c_md_write_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_w2c_md));
    cb_w2c_md_write_ptr[0] = get_semaphore(metadata_ready_semaphore_id);
    cb_w2c_md_write_ptr[1] = get_semaphore(matmul_chunk_ready_semaphore_id);
    cb_reserve_back(cb_w2c_md, 2);
    cb_push_back(cb_w2c_md, 2);

    // Precompute NUM_CHUNKS_PER_EXPERT
    volatile tt_l1_ptr uint32_t* metadata_ready_semaphore_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(metadata_ready_semaphore_id));
    uint32_t encoded_metadata_value = *metadata_ready_semaphore_ptr;

    constexpr uint32_t BITS_PER_EXPERT = 10;
    constexpr uint32_t EXPERT_MASK = 0x3FFu;
    uint32_t NUM_TOKENS_PER_EXPERT[num_experts];
    uint32_t NUM_CHUNKS_PER_EXPERT[num_experts];
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        uint32_t num_tokens = (encoded_metadata_value >> (1 + BITS_PER_EXPERT * expert_id)) & EXPERT_MASK;
        NUM_TOKENS_PER_EXPERT[expert_id] = num_tokens;
        NUM_CHUNKS_PER_EXPERT[expert_id] = (num_tokens + tokens_per_chunk - 1) / tokens_per_chunk;
    }

    // Tilize core we signal to that tilize cores can send another chunk of tiles
    uint64_t matmul_chunk_available_semaphore_noc_addr = get_noc_addr(
        tilize_drain_core_noc_x, tilize_drain_core_noc_y, get_semaphore(matmul_chunk_available_semaphore_id));

    //-------------------------------------------------------------------------
    // Expert loop
    //-------------------------------------------------------------------------
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        uint32_t num_expert_chunks = NUM_CHUNKS_PER_EXPERT[expert_id];
        
        const uint32_t active_tokens = per_expert_counts_ptr[expert_id];
        const uint32_t height_blocks = detail::div_up(active_tokens, tile_height);
        const uint32_t max_tokens_per_height_shard = detail::div_up(active_tokens, height_shard_dim);
        const uint32_t expert_offset_bytes = shard_offset_per_expert_bytes * expert_id;
        
        for (uint32_t chunk = 0; chunk < num_expert_chunks; ++chunk) {
            // Wait for compute core to tell us that all mm01 data is ready
            cb_wait_front(cb_c2w_rdy, 1);
            cb_pop_front(cb_c2w_rdy, 1);

            // Signal to tilize cores that they can send another chunk of tiles
            noc_semaphore_inc</*posted=*/true>(
                matmul_chunk_available_semaphore_noc_addr, /*incr=*/1, /*noc_id=*/1, /*vc=*/vchannel);

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
                    const uint32_t local_src_addr = LOCAL_BUFFER_OFFSET[step & 1];
                    const uint64_t neighbor_dst_addr = LOCAL_BUFFER_OFFSET[!(step & 1)];

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
            
            const uint32_t dest_height_shard_start = (hb * tile_height) / max_tokens_per_height_shard;
            const uint32_t shard_row_start = (hb * tile_height) % max_tokens_per_height_shard;
            
            uint32_t width_tiles_to_send = output_width_tiles_core;  // 18 or 19
            uint32_t width_tiles_sent = 0;

            const uint32_t num_tokens_block = std::min(tile_height, active_tokens - chunk * tile_height);

            cb_wait_front(cb_c2s_out, num_w0_w1_tiles_h);
            const uint32_t source_base_l1_addr = get_read_ptr(cb_c2s_out);

            while (width_tiles_to_send > 0) {
                const uint32_t width_tile_start = width_tile_base + width_tiles_sent;
                const uint32_t dest_width_shard = width_tile_start / combine_shard_width_tiles;
                const uint32_t dest_width_offset_tiles = width_tile_start % combine_shard_width_tiles;

                const uint32_t dest_width_offset_bytes = dest_width_offset_tiles * tile_width_size_bytes;

                const uint32_t width_transfer_tiles = std::min(
                    combine_shard_width_tiles - dest_width_offset_tiles, output_width_tiles_core - width_tiles_sent);
                const uint32_t width_transfer_bytes = width_transfer_tiles * tile_width_size_bytes;

                uint32_t dest_height_shard = dest_height_shard_start;
                uint32_t shard_row = shard_row_start;
                for (uint32_t bt = 0; bt < num_tokens_block; ++bt) {
                    const uint32_t shard_row_offset_bytes =
                        shard_row * combine_shard_width_tiles * tile_width_size_bytes;

                    const auto dest_noc_x =
                        output_shard_core_map[2 * (dest_height_shard * width_shard_dim + dest_width_shard)];
                    const auto dest_noc_y =
                        output_shard_core_map[2 * (dest_height_shard * width_shard_dim + dest_width_shard) + 1];

                    const uint64_t dest_noc_addr_base = get_noc_addr(dest_noc_x, dest_noc_y, output_base_l1_addr);
                    noc_async_write_one_packet_set_state</*posted=*/true>(
                        dest_noc_addr_base, width_transfer_bytes, /*noc=*/1, vchannel);

                    const uint32_t dest_l1_addr =
                        output_base_l1_addr + expert_offset_bytes + dest_width_offset_bytes + shard_row_offset_bytes;

                    const uint32_t source_l1_addr =
                        source_base_l1_addr + (bt * source_width_tiles + width_tiles_sent) * tile_width_size_bytes;

                    noc_async_write_one_packet_with_state</*posted=*/true>(source_l1_addr, dest_l1_addr);

                    noc_async_posted_writes_flushed(1);

                    noc_async_posted_atomic_barrier(1);

                    if (++shard_row == max_tokens_per_height_shard) {
                        ++dest_height_shard;
                        shard_row = 0;
                    }
                }
                width_tiles_sent += width_transfer_tiles;
                width_tiles_to_send -= width_transfer_tiles;
            }
            noc_async_posted_writes_flushed(1);
            noc_async_posted_atomic_barrier(1);
            cb_pop_front(cb_c2s_out, num_w0_w1_tiles_h);
        }
    }
}
