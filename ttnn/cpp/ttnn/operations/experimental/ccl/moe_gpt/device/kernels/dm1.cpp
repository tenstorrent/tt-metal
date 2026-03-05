// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "moe_gpt_ring_common.h"

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t layer_id = get_named_compile_time_arg_val("layer_id");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t enable_dram_output = get_named_compile_time_arg_val("enable_dram_output");

#ifdef TILIZE_FUSED
    constexpr uint32_t metadata_ready_semaphore_id = get_named_compile_time_arg_val("metadata_ready_semaphore_id");
    constexpr uint32_t matmul_chunk_ready_semaphore_id =
        get_named_compile_time_arg_val("matmul_chunk_ready_semaphore_id");
    constexpr uint32_t matmul_chunk_available_semaphore_id =
        get_named_compile_time_arg_val("matmul_chunk_available_semaphore_id");
    constexpr uint32_t tokens_per_chunk = get_named_compile_time_arg_val("tokens_per_chunk");
    constexpr uint32_t tilize_drain_core_noc_x = get_named_compile_time_arg_val("tilize_drain_core_noc_x");
    constexpr uint32_t tilize_drain_core_noc_y = get_named_compile_time_arg_val("tilize_drain_core_noc_y");
#endif

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
#ifdef TILIZE_FUSED
    constexpr auto cb_w2c_md = tt::CBIndex::c_5;
#endif

    // CB Aliases
    constexpr auto cb_r2c_w2 = tt::CBIndex::c_0;
#ifndef TILIZE_FUSED
    constexpr auto cb_c2s_out = tt::CBIndex::c_1;
#endif

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t w0_w1_tile_size = get_tile_size(cb_r2c_w0_w1);
    constexpr uint32_t w2_tile_size = get_tile_size(cb_r2c_w2);
    constexpr uint32_t in2_tile_size = get_tile_size(cb_s2c_in2);

    // Constants for MoEGPT
    constexpr uint32_t num_w0_w1_tiles_h = moe_gpt_ring::NUM_W0_W1_TILES_H;
    constexpr uint32_t num_w2_tiles_h = moe_gpt_ring::NUM_W2_TILES_H;

    const uint32_t num_w0_w1_tiles_w = moe_gpt_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][0];  // 7 or 8
    const uint32_t num_w2_tiles_w = moe_gpt_ring::W2_TILES_PER_CORE_A[ring_core_id];                    // 7 or 8

    const uint32_t num_elt_tiles = num_w0_w1_tiles_w;
    const uint32_t num_in2_tiles = num_w2_tiles_w;
    const uint32_t num_mm2_tiles = num_w2_tiles_w;

#ifdef TILIZE_FUSED
    //=========================================================================
    // FUSED MODE: tilize → W0/W1 → SwiGLU → A2A ring → W2 → combine
    //=========================================================================

    // Combine output constants
    constexpr uint32_t height_shard_dim = get_named_compile_time_arg_val("height_shard_dim");
    constexpr uint32_t width_shard_dim = get_named_compile_time_arg_val("width_shard_dim");
    constexpr uint32_t combine_shard_width_tiles = get_named_compile_time_arg_val("combine_shard_width_tiles");
    constexpr uint32_t tile_width_size_bytes = get_named_compile_time_arg_val("tile_width_size_bytes");

    std::array<uint32_t, 2 * height_shard_dim * width_shard_dim> output_shard_core_map = OUTPUT_SHARD_CORE_MAP;

    // The number of tiles to send in each step (max of 7/8 = 8 for GPT-OSS)
    constexpr uint32_t tiles_per_step = moe_gpt_ring::IN2_TILES_PER_STEP_A;  // 8

    // CB Aliases
    constexpr auto cb_c2s_out = tt::CBIndex::c_14;  // untilized ROW_MAJOR output

    // Additional runtime args for fused combine mode
    const auto combine_semaphore_id = get_arg_val<uint32_t>(10);
    const auto k_start_tile = get_arg_val<uint32_t>(11);
    const auto output_base_l1_addr = get_arg_val<uint32_t>(12);

    const uint32_t tiles_per_core = moe_gpt_ring::W2_TILES_PER_CORE_A[ring_core_id];

    //-------------------------------------------------------------------------
    // Init synchronization with tilize cores
    //-------------------------------------------------------------------------

    // Wait for tilize drain to deliver metadata (token counts per expert)
    uint32_t metadata_ready_semaphore_addr = get_semaphore(metadata_ready_semaphore_id);
    noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata_ready_semaphore_addr), 1);

    // Transfer semaphore addresses to compute via cb_w2c_md:
    //   [0] = metadata_ready_semaphore address (for compute to decode token counts)
    //   [1] = matmul_chunk_ready_semaphore address (for compute to wait on tilized chunks)
    volatile tt_l1_ptr uint32_t* cb_w2c_md_write_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_w2c_md));
    cb_w2c_md_write_ptr[0] = metadata_ready_semaphore_addr;
    cb_w2c_md_write_ptr[1] = get_semaphore(matmul_chunk_ready_semaphore_id);
    cb_reserve_back(cb_w2c_md, 2);
    cb_push_back(cb_w2c_md, 2);

    //-------------------------------------------------------------------------
    // Decode metadata: per-expert token counts and chunk counts
    //-------------------------------------------------------------------------
    volatile tt_l1_ptr uint32_t* metadata_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata_ready_semaphore_addr);
    uint32_t encoded_metadata_value = *metadata_sem_ptr;

    constexpr uint32_t BITS_PER_EXPERT = 7;
    constexpr uint32_t EXPERT_MASK = 0x7Fu;
    uint32_t NUM_CHUNKS_PER_EXPERT[num_experts];
    for (uint32_t e = 0; e < num_experts; ++e) {
        uint32_t num_tokens = (encoded_metadata_value >> (1 + BITS_PER_EXPERT * e)) & EXPERT_MASK;
        NUM_CHUNKS_PER_EXPERT[e] = (num_tokens + tokens_per_chunk - 1) / tokens_per_chunk;
    }

    // NOC address to signal tilize drain that matmul has consumed a chunk
    uint32_t local_chunk_available_sem_addr = get_semaphore(matmul_chunk_available_semaphore_id);
    uint64_t matmul_chunk_available_noc_addr =
        get_noc_addr(tilize_drain_core_noc_x, tilize_drain_core_noc_y, local_chunk_available_sem_addr);

    //-------------------------------------------------------------------------
    // Combine core output constants
    //-------------------------------------------------------------------------
    constexpr uint32_t source_width_tiles = moe_gpt_ring::SOURCE_WIDTH_TILES;  // 8
    constexpr uint32_t tokens_per_chunk_combine = moe_gpt_ring::TOKENS_PER_CHUNK;
    constexpr uint32_t RING_CORES_PER_COMBINE_COL = moe_gpt_ring::RING_CORES_PER_COMBINE_COL;

    const uint32_t output_width_tiles_core = tiles_per_core;
    const uint32_t width_tile_base = moe_gpt_ring::COMBINE_W_OFFSET_PER_CORE_A[ring_core_id];
    const uint32_t combine_core_x = ring_core_id / RING_CORES_PER_COMBINE_COL;

    //-------------------------------------------------------------------------
    // Ring A2A setup
    //-------------------------------------------------------------------------
    constexpr uint32_t num_a2a_iters = moe_gpt_ring::NUM_A2A_ITERS_A;     // 2
    constexpr uint32_t num_a2a_steps_per_iter = moe_gpt_ring::NUM_CORES;  // 12

    uint32_t semaphore_addr = get_semaphore(ring_semaphore_id);
    volatile tt_l1_ptr uint32_t* my_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);

    const uint64_t neighbor_semaphore_noc_addr =
        get_noc_addr(ring_neighbor_physical_x, ring_neighbor_physical_y, semaphore_addr);

    constexpr uint32_t a2a_xfer_bytes_per_step = tiles_per_step * in2_tile_size;
    constexpr uint32_t a2a_tiles_per_packet = 4;
    constexpr uint32_t a2a_packet_size = a2a_tiles_per_packet * in2_tile_size;
    constexpr uint32_t a2a_num_packets = tiles_per_step / a2a_tiles_per_packet;

    const uint32_t local_base_addr = get_write_ptr(cb_s2c_in2);
    const uint64_t neighbor_base_addr =
        get_noc_addr(ring_neighbor_physical_x, ring_neighbor_physical_y, local_base_addr);

    constexpr uint32_t NUM_A2A_BUFFERS = 6;
    uint32_t LOCAL_BUFFER_OFFSET[NUM_A2A_BUFFERS];
    for (uint32_t i = 0; i < NUM_A2A_BUFFERS; ++i) {
        LOCAL_BUFFER_OFFSET[i] = local_base_addr + i * a2a_xfer_bytes_per_step;
    }

    noc_semaphore_set(my_semaphore_ptr, 0);
    uint32_t semaphore_value = 0;

    //-------------------------------------------------------------------------
    // Per-chunk processing: SwiGLU → A2A ring → combine write
    // Unified chunk loop following the deepseek moe_compute pattern.
    // The full pipeline completes per chunk, so:
    // - No cross-expert barrier needed (A2A step 0's semaphore wait implicitly
    //   ensures predecessor's writes have landed)
    // - No cb_w2c_rdy sync before SwiGLU (pipeline completion guarantees
    //   cb_s2c_in2 is free)
    //-------------------------------------------------------------------------
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        const uint32_t num_expert_chunks = NUM_CHUNKS_PER_EXPERT[expert_id];

        for (uint32_t chunk = 0; chunk < num_expert_chunks; ++chunk) {
            // Set NOC 1 write state at top of chunk, before waiting for compute.
            // This overlaps setup with compute's SwiGLU work.
            noc_async_write_one_packet_set_state<true>(neighbor_base_addr, a2a_packet_size, 1, vchannel);
            noc_inline_dw_write_set_state<true, false>(
                neighbor_semaphore_noc_addr, 0, 0xF, write_at_cmd_buf, 1, vchannel);

            // Wait for compute's SwiGLU output
            cb_wait_front(cb_c2w_rdy, 1);
            cb_pop_front(cb_c2w_rdy, 1);

            // A2A ring rotation for this chunk's SwiGLU output
            for (uint32_t i = 0; i < num_a2a_iters; ++i) {
                for (uint32_t step = 0; step < num_a2a_steps_per_iter; ++step) {
                    // Wait for data from predecessor
                    while ((*my_semaphore_ptr) < semaphore_value) {
                    };

                    // Signal compute that A2A data is ready for W2 matmul
                    cb_reserve_back(cb_w2c_rdy, 1);
                    cb_push_back(cb_w2c_rdy, 1);

                    // Send to ring neighbor
                    const uint32_t src_buf = step % NUM_A2A_BUFFERS;
                    const uint32_t dst_buf = (step + 1) % NUM_A2A_BUFFERS;
                    const uint32_t local_src_addr = LOCAL_BUFFER_OFFSET[src_buf];
                    const uint64_t neighbor_dst_addr = LOCAL_BUFFER_OFFSET[dst_buf];

                    for (uint32_t pkt = 0; pkt < a2a_num_packets; ++pkt) {
                        uint32_t pkt_offset = pkt * a2a_packet_size;
                        noc_async_write_one_packet_with_state<true>(
                            local_src_addr + pkt_offset, neighbor_dst_addr + pkt_offset);
                    }

                    // Signal neighbor that data is ready. No flush needed between data and
                    // semaphore: same destination, same NOC, NOC ordering guarantees order.
                    noc_inline_dw_write_with_state<false, true, true, false, true>(++semaphore_value);

                    noc_async_posted_writes_flushed(1);
                }
            }

            //-----------------------------------------------------------------
            // Combine core output write for this chunk
            // Untilized ROW_MAJOR data from c_14 → combine core L1
            //-----------------------------------------------------------------
            const uint32_t dest_height_shard = expert_id;

            cb_wait_front(cb_c2s_out, tokens_per_chunk_combine);
            const uint32_t source_base_l1_addr = get_read_ptr(cb_c2s_out);

            uint32_t width_tiles_to_send = output_width_tiles_core;
            uint32_t width_tiles_sent = 0;

            while (width_tiles_to_send > 0) {
                const uint32_t width_tile_start = width_tile_base + width_tiles_sent;
                const uint32_t dest_width_shard = width_tile_start / combine_shard_width_tiles;
                const uint32_t dest_width_offset_tiles = width_tile_start % combine_shard_width_tiles;
                const uint32_t dest_width_offset_bytes = dest_width_offset_tiles * tile_width_size_bytes;

                const uint32_t width_transfer_tiles = std::min(
                    combine_shard_width_tiles - dest_width_offset_tiles, output_width_tiles_core - width_tiles_sent);
                const uint32_t width_transfer_bytes = width_transfer_tiles * tile_width_size_bytes;

                const auto dest_noc_x =
                    output_shard_core_map[2 * (dest_height_shard * width_shard_dim + dest_width_shard)];
                const auto dest_noc_y =
                    output_shard_core_map[2 * (dest_height_shard * width_shard_dim + dest_width_shard) + 1];

                const uint64_t dest_noc_addr_base = get_noc_addr(dest_noc_x, dest_noc_y, output_base_l1_addr, 1);
                noc_async_write_one_packet_set_state<true>(dest_noc_addr_base, width_transfer_bytes, 1, vchannel);

                for (uint32_t bt = 0; bt < tokens_per_chunk_combine; ++bt) {
                    const uint32_t shard_row_offset_bytes = bt * combine_shard_width_tiles * tile_width_size_bytes;

                    const uint32_t dest_l1_addr =
                        output_base_l1_addr + dest_width_offset_bytes + shard_row_offset_bytes;

                    const uint32_t source_l1_addr =
                        source_base_l1_addr + (bt * source_width_tiles + width_tiles_sent) * tile_width_size_bytes;

                    noc_async_write_one_packet_with_state<true>(source_l1_addr, dest_l1_addr);
                }
                width_tiles_sent += width_transfer_tiles;
                width_tiles_to_send -= width_transfer_tiles;
            }

            noc_async_posted_writes_flushed(1);
            cb_pop_front(cb_c2s_out, tokens_per_chunk_combine);

            // Signal tilize drain that this chunk has been consumed
            noc_semaphore_inc<true>(matmul_chunk_available_noc_addr, 1, 1, vchannel);
        }
    }

    // Signal combine cores that all expert data is written
    uint32_t combine_semaphore_addr = get_semaphore(combine_semaphore_id);
    for (uint32_t y = 0; y < height_shard_dim; ++y) {
        uint32_t idx = combine_core_x + y * width_shard_dim;
        uint64_t dest_sem_noc_addr =
            get_noc_addr(output_shard_core_map[2 * idx], output_shard_core_map[2 * idx + 1], combine_semaphore_addr);
        noc_semaphore_inc(dest_sem_noc_addr, 1, 1, vchannel);
    }
    noc_async_atomic_barrier(1);

#else
    //=========================================================================
    // NON-FUSED MODE: Ring A2A + optional DRAM output
    //=========================================================================

    //-------------------------------------------------------------------------
    // Ring setup
    //-------------------------------------------------------------------------
    // The number of times to repeat the all2all
    constexpr uint32_t num_a2a_iters = moe_gpt_ring::NUM_A2A_ITERS_A;  // 2

    // The number of steps to take in the all2all is the number of cores
    constexpr uint32_t num_a2a_steps_per_iter = moe_gpt_ring::NUM_CORES;  // 12

    // The number of tiles to send in each step (max of 7/8 = 8 for GPT-OSS)
    constexpr uint32_t tiles_per_step = moe_gpt_ring::IN2_TILES_PER_STEP_A;  // 8

    //-------------------------------------------------------------------------
    // Ring NoC setup
    //-------------------------------------------------------------------------
    uint32_t semaphore_addr = get_semaphore(ring_semaphore_id);
    volatile tt_l1_ptr uint32_t* my_semaphore_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);

    const uint64_t neighbor_semaphore_noc_addr =
        get_noc_addr(ring_neighbor_physical_x, ring_neighbor_physical_y, semaphore_addr);

    // Size of each transfer in bytes: 8 tiles * 2048 = 16384 bytes
    constexpr uint32_t a2a_xfer_bytes_per_step = tiles_per_step * in2_tile_size;

    // NOC_MAX_BURST_SIZE = 8192 bytes on Wormhole. Split into packets of 4 tiles each
    // (4 * 2048 = 8192 bytes = NOC_MAX_BURST_SIZE). 8 tiles / 4 tiles per packet = 2 packets.
    constexpr uint32_t a2a_tiles_per_packet = 4;
    constexpr uint32_t a2a_packet_size = a2a_tiles_per_packet * in2_tile_size;   // 8192
    constexpr uint32_t a2a_num_packets = tiles_per_step / a2a_tiles_per_packet;  // 2

    // Source and destination addresses for the all2all
    const uint32_t local_base_addr = get_write_ptr(cb_s2c_in2);
    const uint64_t neighbor_base_addr =
        get_noc_addr(ring_neighbor_physical_x, ring_neighbor_physical_y, local_base_addr);

    // 6 buffers for A2A intermediate data. 6 divides 12 (steps per iter),
    // so buffer indices cycle back to 0 after a full ring rotation.
    // 5 steps of slack between write and overwrite is sufficient for compute.
    constexpr uint32_t NUM_A2A_BUFFERS = 6;
    uint32_t LOCAL_BUFFER_OFFSET[NUM_A2A_BUFFERS];
    for (uint32_t i = 0; i < NUM_A2A_BUFFERS; ++i) {
        LOCAL_BUFFER_OFFSET[i] = local_base_addr + i * a2a_xfer_bytes_per_step;
    }
    // Reset semaphore to 0 for clean state on each invocation.
    // When the program is cached and reused, the semaphore retains its value
    // from the previous run. Resetting here is safe because:
    // 1. The command queue guarantees all NOC activity from the previous
    //    invocation has completed before this kernel starts.
    // 2. All cores are waiting for cb_c2w_rdy (compute's SwiGLU) before
    //    any ring activity begins, so no predecessor can write to our
    //    semaphore before we reset it.
    noc_semaphore_set(my_semaphore_ptr, 0);
    uint32_t semaphore_value = 0;

    // Set state for the data writes
    noc_async_write_one_packet_set_state</*posted=*/true>(neighbor_base_addr, a2a_packet_size, /*noc=*/1, vchannel);

    // Set state for the semaphore write
    noc_inline_dw_write_set_state</*posted=*/true, /*set_val=*/false>(
        neighbor_semaphore_noc_addr, /*val=*/0, /*be=*/0xF, /*cmd_buf=*/write_at_cmd_buf, /*noc=*/1, vchannel);

    //-------------------------------------------------------------------------
    // Expert loop
    //-------------------------------------------------------------------------
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        // Wait for compute core to tell us that all mm01 data is ready
        cb_wait_front(cb_c2w_rdy, 1);
        cb_pop_front(cb_c2w_rdy, 1);

        // Take the data in cb_s2c_in2 and send it to the next core in the ring
        // GPT-OSS: 2 A2A iterations x 12 steps = 24 rotations per expert
        for (uint32_t i = 0; i < num_a2a_iters; ++i) {
            for (uint32_t step = 0; step < num_a2a_steps_per_iter; ++step) {
                // Wait for current data to be ready in cb_s2c_in2
                while ((*my_semaphore_ptr) < semaphore_value) {
                };

                // Signal to compute core that data is ready
                cb_reserve_back(cb_w2c_rdy, 1);
                cb_push_back(cb_w2c_rdy, 1);

                // Write 8 tiles from local cb_s2c_in2 to neighbor's cb_s2c_in2
                // Buffer index cycles with modulo, resetting at each iteration.
                const uint32_t src_buf = step % NUM_A2A_BUFFERS;
                const uint32_t dst_buf = (step + 1) % NUM_A2A_BUFFERS;
                const uint32_t local_src_addr = LOCAL_BUFFER_OFFSET[src_buf];
                const uint64_t neighbor_dst_addr = LOCAL_BUFFER_OFFSET[dst_buf];

                // Send as 2 packets of 4 tiles each (8192 bytes per packet)
                for (uint32_t pkt = 0; pkt < a2a_num_packets; ++pkt) {
                    uint32_t pkt_offset = pkt * a2a_packet_size;
                    noc_async_write_one_packet_with_state</*posted=*/true>(
                        local_src_addr + pkt_offset, neighbor_dst_addr + pkt_offset);
                }

                // Signal neighbor that data is ready (increment their semaphore value).
                // No flush needed between data and semaphore: both are posted writes to the
                // same destination on the same NOC, so NOC ordering guarantees data arrives first.
                noc_inline_dw_write_with_state<
                    /*update_addr_lo=*/false,
                    /*update_counter=*/true,
                    /*posted=*/true,
                    /*update_addr_hi=*/false,
                    /*update_val=*/true>(++semaphore_value);

                // Ensure writes have left the core before continuing
                noc_async_posted_writes_flushed(1);
            }
        }

        // Cross-expert boundary barrier: after the ring loop, wait for the
        // predecessor's FINAL write to our buf 0 before letting compute
        // overwrite buf 0 with the next expert's SwiGLU.
        //
        // The predecessor's last A2A step (step 11, iter 1) writes to our
        // dst_buf = (11+1)%12 = 0. The ring loop only waited for semaphore
        // value (semaphore_value - 1) which confirms step 10's data, not
        // step 11's. Here we wait for semaphore_value (set by ++semaphore_value
        // on the last iteration), confirming the predecessor's step 11 write
        // has landed at our L1.
        if (expert_id < num_experts - 1) {
            while ((*my_semaphore_ptr) < semaphore_value) {
            };
            // Signal compute that buf 0 is safe to overwrite
            cb_reserve_back(cb_w2c_rdy, 1);
            cb_push_back(cb_w2c_rdy, 1);
        }
    }

    //-------------------------------------------------------------------------
    // DRAM output write phase
    //-------------------------------------------------------------------------
    // After all experts are done, copy output tiles from L1 sharded buffer to
    // a contiguous interleaved DRAM tensor with shape (E, 1, M, K) in TILE_LAYOUT.
    // Each core owns tiles_per_core tiles (7 or 8) of the K dimension per expert.
    if constexpr (enable_dram_output) {
        // Read DRAM output runtime args
        const auto dram_output_addr = get_arg_val<uint32_t>(argidx++);
        const auto k_start_tile = get_arg_val<uint32_t>(argidx++);

        // K dimension in tiles (K=2880, 2880/32=90)
        constexpr uint32_t K_tiles = moe_gpt_ring::NUM_W0_W1_TILES_H;  // 90

        // tiles_per_core for this core (7 or 8)
        const uint32_t tiles_per_core = moe_gpt_ring::W2_TILES_PER_CORE_A[ring_core_id];

        // Create address generator for the interleaved DRAM output
        // Uses InterleavedAddrGen which computes bank-interleaved NOC addresses
        const InterleavedAddrGen<true> dram_output_addrgen = {
            .bank_base_address = dram_output_addr,
            .page_size = in_tile_size,
        };

        // L1 source: cb_c2s_out base address (same buffer as input, output written in-place)
        const uint32_t cb_base = get_write_ptr(cb_c2s_out);

        // For each expert, write this core's tiles to the DRAM output tensor
        for (uint32_t e = 0; e < num_experts; ++e) {
            for (uint32_t t = 0; t < tiles_per_core; ++t) {
                // L1 source: expert e's section starts at tile (e * K_tiles),
                // and this core's valid output is at tiles 0..tiles_per_core-1
                uint32_t l1_addr = cb_base + (e * K_tiles + t) * in_tile_size;

                // DRAM destination: linear tile index in (E, 1, M, K) tensor
                // Expert e occupies tiles [e*K_tiles .. (e+1)*K_tiles-1]
                uint32_t dram_tile_id = e * K_tiles + k_start_tile + t;

                noc_async_write_tile(dram_tile_id, dram_output_addrgen, l1_addr);
            }
        }
        noc_async_write_barrier();
    }
#endif  // TILIZE_FUSED
}
