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
    uint32_t NUM_CHUNKS_PER_EXPERT[num_experts];
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        uint32_t num_tokens = (encoded_metadata_value >> (1 + BITS_PER_EXPERT * expert_id)) & EXPERT_MASK;
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
        }
    }
}
