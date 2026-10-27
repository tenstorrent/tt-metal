// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "moe_gpt_ring_common.h"

void kernel_main() {
    // Compile time arguments
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");
    constexpr uint32_t enable_dram_output = get_named_compile_time_arg_val("enable_dram_output");

    constexpr uint32_t metadata_ready_semaphore_id = get_named_compile_time_arg_val("metadata_ready_semaphore_id");
    constexpr uint32_t metadata_count_semaphore_base_id =
        get_named_compile_time_arg_val("metadata_count_semaphore_base_id");
    constexpr uint32_t matmul_chunk_ready_semaphore_id =
        get_named_compile_time_arg_val("matmul_chunk_ready_semaphore_id");
    constexpr uint32_t matmul_chunk_available_semaphore_id =
        get_named_compile_time_arg_val("matmul_chunk_available_semaphore_id");
    constexpr uint32_t tokens_per_chunk = get_named_compile_time_arg_val("tokens_per_chunk");
    constexpr uint32_t tilize_drain_core_noc_x = get_named_compile_time_arg_val("tilize_drain_core_noc_x");
    constexpr uint32_t tilize_drain_core_noc_y = get_named_compile_time_arg_val("tilize_drain_core_noc_y");

    constexpr auto in_args = TensorAccessorArgs<0>();
    constexpr auto w0_w1_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();
    constexpr auto w2_args = TensorAccessorArgs<w0_w1_args.next_compile_time_args_offset()>();
    [[maybe_unused]] constexpr auto out_args = TensorAccessorArgs<w2_args.next_compile_time_args_offset()>();

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

    // Noc typed wrapper (uses default noc_index)
    Noc noc_obj(noc_index);
    // Secondary noc used for A2A ring writes and combine writes (always NOC 1)
    Noc noc1_obj(1);

    // CBs
    constexpr auto cb_r2c_w0_w1_id = tt::CBIndex::c_0;
    constexpr auto cb_s2c_in_id = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy_id = tt::CBIndex::c_2;
    constexpr auto cb_w2c_rdy_id = tt::CBIndex::c_3;
    constexpr auto cb_s2c_in2_id = tt::CBIndex::c_4;
    constexpr auto cb_w2c_md_id = tt::CBIndex::c_5;

    // CB Aliases
    constexpr auto cb_r2c_w2_id = tt::CBIndex::c_0;

    CircularBuffer cb_c2w_rdy(cb_c2w_rdy_id);
    CircularBuffer cb_w2c_rdy(cb_w2c_rdy_id);
    CircularBuffer cb_s2c_in2(cb_s2c_in2_id);
    CircularBuffer cb_w2c_md(cb_w2c_md_id);

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in_id);
    constexpr uint32_t w0_w1_tile_size = get_tile_size(cb_r2c_w0_w1_id);
    constexpr uint32_t w2_tile_size = get_tile_size(cb_r2c_w2_id);
    constexpr uint32_t in2_tile_size = get_tile_size(cb_s2c_in2_id);

    // Constants for MoEGPT
    constexpr uint32_t num_w0_w1_tiles_h = moe_gpt_ring::NUM_W0_W1_TILES_H;
    constexpr uint32_t num_w2_tiles_h = moe_gpt_ring::NUM_W2_TILES_H;

    const uint32_t num_w0_w1_tiles_w = moe_gpt_ring::W0_W1_TILES_PER_CORE_PER_STEP_A[ring_core_id][0];  // 7 or 8
    const uint32_t num_w2_tiles_w = moe_gpt_ring::W2_TILES_PER_CORE_A[ring_core_id];                    // 7 or 8

    const uint32_t num_elt_tiles = num_w0_w1_tiles_w;
    const uint32_t num_in2_tiles = num_w2_tiles_w;
    const uint32_t num_mm2_tiles = num_w2_tiles_w;

    //=========================================================================
    // FUSED MODE: tilize → W0/W1 → SwiGLU → A2A ring → W2 → combine
    //=========================================================================

    // Combine output constants
    constexpr uint32_t height_shard_dim = get_named_compile_time_arg_val("height_shard_dim");
    constexpr uint32_t width_shard_dim = get_named_compile_time_arg_val("width_shard_dim");
    constexpr uint32_t combine_shard_width_tiles = get_named_compile_time_arg_val("combine_shard_width_tiles");
    constexpr uint32_t tile_width_size_bytes = get_named_compile_time_arg_val("tile_width_size_bytes");
    constexpr uint32_t num_tokens_total = get_named_compile_time_arg_val("num_tokens_total");
    constexpr uint32_t shard_offset_per_expert_bytes =
        num_tokens_total / height_shard_dim * combine_shard_width_tiles * tile_width_size_bytes;

    std::array<uint32_t, 2 * height_shard_dim * width_shard_dim> output_shard_core_map = OUTPUT_SHARD_CORE_MAP;

    // The number of tiles to send in each step (max of 7/8 = 8 for GPT-OSS)
    constexpr uint32_t tiles_per_step = moe_gpt_ring::IN2_TILES_PER_STEP_A;  // 8

    // CB Aliases
    constexpr auto cb_c2s_out_id = tt::CBIndex::c_14;  // untilized ROW_MAJOR output
    CircularBuffer cb_c2s_out(cb_c2s_out_id);

    // Additional runtime args for fused combine mode
    const auto combine_semaphore_id = get_arg_val<uint32_t>(10);
    const auto k_start_tile = get_arg_val<uint32_t>(11);
    const auto output_base_l1_addr = get_arg_val<uint32_t>(12);

    const uint32_t tiles_per_core = moe_gpt_ring::W2_TILES_PER_CORE_A[ring_core_id];

    //-------------------------------------------------------------------------
    // Init synchronization with tilize cores
    //-------------------------------------------------------------------------

    // Wait for tilize drain to deliver metadata (token counts per expert)
    Semaphore<> metadata_ready_sem(metadata_ready_semaphore_id);
    metadata_ready_sem.wait_min(1);

    // Read per-expert token counts from dedicated semaphores
    // Device 2.0 migration: legacy primitive retained: Semaphore<> has no accessor to read its
    // current value; use the legacy pointer-based read.
    uint32_t NUM_TOKENS_PER_EXPERT[num_experts];
    uint32_t NUM_CHUNKS_PER_EXPERT[num_experts];
    for (uint32_t e = 0; e < num_experts; ++e) {
        volatile tt_l1_ptr uint32_t* count_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(metadata_count_semaphore_base_id + e));
        uint32_t num_tokens = *count_sem_ptr;
        NUM_TOKENS_PER_EXPERT[e] = num_tokens;
        NUM_CHUNKS_PER_EXPERT[e] = (num_tokens + tokens_per_chunk - 1) / tokens_per_chunk;
    }

    // Transfer per-expert token counts + chunk_ready semaphore address to compute via cb_w2c_md:
    //   [0..num_experts-1] = raw token counts per expert
    //   [num_experts]      = matmul_chunk_ready_semaphore address
    cb_w2c_md.reserve_back(2);
    volatile tt_l1_ptr uint32_t* cb_w2c_md_write_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_w2c_md.get_write_ptr());
    for (uint32_t e = 0; e < num_experts; ++e) {
        volatile tt_l1_ptr uint32_t* count_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(metadata_count_semaphore_base_id + e));
        cb_w2c_md_write_ptr[e] = *count_sem_ptr;
    }
    cb_w2c_md_write_ptr[num_experts] = get_semaphore(matmul_chunk_ready_semaphore_id);
    cb_w2c_md.push_back(2);

    // NOC address to signal tilize drain that matmul has consumed a chunk
    uint32_t local_chunk_available_sem_addr = get_semaphore(matmul_chunk_available_semaphore_id);
    // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address
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

    Semaphore<> ring_sem(ring_semaphore_id);
    uint32_t semaphore_addr = get_semaphore(ring_semaphore_id);

    // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address
    const uint64_t neighbor_semaphore_noc_addr =
        get_noc_addr(ring_neighbor_physical_x, ring_neighbor_physical_y, semaphore_addr);

    constexpr uint32_t a2a_xfer_bytes_per_step = tiles_per_step * in2_tile_size;
    constexpr uint32_t a2a_tiles_per_packet = 4;
    constexpr uint32_t a2a_packet_size = a2a_tiles_per_packet * in2_tile_size;
    constexpr uint32_t a2a_num_packets = tiles_per_step / a2a_tiles_per_packet;

    const uint32_t local_base_addr = cb_s2c_in2.get_write_ptr();
    // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address
    const uint64_t neighbor_base_addr =
        get_noc_addr(ring_neighbor_physical_x, ring_neighbor_physical_y, local_base_addr);

    constexpr uint32_t NUM_A2A_BUFFERS = 6;
    uint32_t LOCAL_BUFFER_OFFSET[NUM_A2A_BUFFERS];
    for (uint32_t i = 0; i < NUM_A2A_BUFFERS; ++i) {
        LOCAL_BUFFER_OFFSET[i] = local_base_addr + i * a2a_xfer_bytes_per_step;
    }

    ring_sem.set(0);
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
        const uint32_t active_tokens = NUM_TOKENS_PER_EXPERT[expert_id];
        const uint32_t tokens_per_height_shard_chunk = active_tokens / height_shard_dim;
        const uint32_t tokens_per_height_shard_rem = active_tokens % height_shard_dim;
        const uint32_t expert_offset_bytes = shard_offset_per_expert_bytes * expert_id;

        uint32_t dest_height_shard_start = 0;
        uint32_t shard_row_start = 0;

        for (uint32_t chunk = 0; chunk < num_expert_chunks; ++chunk) {
            // Set NOC 1 write state at top of chunk, before waiting for compute.
            // This overlaps setup with compute's SwiGLU work.
            // Device 2.0 migration: legacy primitives retained: state-machine setup
            // (noc_async_write_one_packet_set_state, noc_inline_dw_write_set_state) has no
            // Device 2.0 wrappers
            noc_async_write_one_packet_set_state<true>(neighbor_base_addr, a2a_packet_size, 1, vchannel);
            noc_inline_dw_write_set_state<true, false>(
                neighbor_semaphore_noc_addr, 0, 0xF, write_at_cmd_buf, 1, vchannel);

            // Wait for compute's SwiGLU output
            cb_c2w_rdy.wait_front(1);
            cb_c2w_rdy.pop_front(1);

            // A2A ring rotation for this chunk's SwiGLU output
            for (uint32_t i = 0; i < num_a2a_iters; ++i) {
                for (uint32_t step = 0; step < num_a2a_steps_per_iter; ++step) {
                    // Wait for data from predecessor
                    ring_sem.wait_min(semaphore_value);

                    // Signal compute that A2A data is ready for W2 matmul
                    cb_w2c_rdy.reserve_back(1);
                    cb_w2c_rdy.push_back(1);

                    // Send to ring neighbor
                    const uint32_t src_buf = step % NUM_A2A_BUFFERS;
                    const uint32_t dst_buf = (step + 1) % NUM_A2A_BUFFERS;
                    const uint32_t local_src_addr = LOCAL_BUFFER_OFFSET[src_buf];
                    const uint64_t neighbor_dst_addr = LOCAL_BUFFER_OFFSET[dst_buf];

                    for (uint32_t pkt = 0; pkt < a2a_num_packets; ++pkt) {
                        uint32_t pkt_offset = pkt * a2a_packet_size;
                        // Device 2.0 migration: legacy primitive retained: paired with
                        // noc_async_write_one_packet_set_state above
                        noc_async_write_one_packet_with_state<true>(
                            local_src_addr + pkt_offset, neighbor_dst_addr + pkt_offset);
                    }

                    // Signal neighbor that data is ready. No flush needed between data and
                    // semaphore: same destination, same NOC, NOC ordering guarantees order.
                    // Device 2.0 migration: legacy primitive retained: paired with
                    // noc_inline_dw_write_set_state above
                    noc_inline_dw_write_with_state<false, true, true, false, true>(++semaphore_value);

                    noc1_obj.async_writes_flushed<NocOptions::POSTED>();
                }
            }

            //-----------------------------------------------------------------
            // Combine core output write for this chunk.
            // Matches moe_compute layout: tokens distributed across tp rows
            // using floor+remainder (matching selective_reduce_combine's
            // token_work_split_even). Shard h gets (chunk+1) tokens if
            // h < rem, else chunk tokens.
            // Tokens within a chunk can cross tp_row boundaries.
            //-----------------------------------------------------------------
            const uint32_t num_tokens_block =
                std::min(tokens_per_chunk_combine, active_tokens - chunk * tokens_per_chunk_combine);

            cb_c2s_out.wait_front(tokens_per_chunk_combine);
            const uint32_t source_base_l1_addr = cb_c2s_out.get_read_ptr();

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

                uint32_t dest_height_shard = dest_height_shard_start;
                uint32_t shard_row = shard_row_start;

                for (uint32_t bt = 0; bt < num_tokens_block; ++bt) {
                    const uint32_t shard_row_offset_bytes =
                        shard_row * combine_shard_width_tiles * tile_width_size_bytes;

                    const auto dest_noc_x =
                        output_shard_core_map[2 * (dest_height_shard * width_shard_dim + dest_width_shard)];
                    const auto dest_noc_y =
                        output_shard_core_map[2 * (dest_height_shard * width_shard_dim + dest_width_shard) + 1];

                    // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address
                    // used as state-machine base
                    const uint64_t dest_noc_addr_base = get_noc_addr(dest_noc_x, dest_noc_y, output_base_l1_addr, 1);
                    // Device 2.0 migration: legacy primitive retained: state-machine setup
                    // (noc_async_write_one_packet_set_state) has no Device 2.0 wrapper
                    noc_async_write_one_packet_set_state<true>(dest_noc_addr_base, width_transfer_bytes, 1, vchannel);

                    const uint32_t dest_l1_addr =
                        output_base_l1_addr + expert_offset_bytes + dest_width_offset_bytes + shard_row_offset_bytes;

                    const uint32_t source_l1_addr =
                        source_base_l1_addr + (bt * source_width_tiles + width_tiles_sent) * tile_width_size_bytes;

                    // Device 2.0 migration: legacy primitive retained: paired with
                    // noc_async_write_one_packet_set_state above
                    noc_async_write_one_packet_with_state<true>(source_l1_addr, dest_l1_addr);

                    const uint32_t shard_capacity = (dest_height_shard < tokens_per_height_shard_rem)
                                                        ? tokens_per_height_shard_chunk + 1
                                                        : tokens_per_height_shard_chunk;
                    if (++shard_row == shard_capacity) {
                        ++dest_height_shard;
                        shard_row = 0;
                    }
                }
                width_tiles_sent += width_transfer_tiles;
                width_tiles_to_send -= width_transfer_tiles;

                if (width_tiles_to_send == 0) {
                    dest_height_shard_start = dest_height_shard;
                    shard_row_start = shard_row;
                }
            }

            noc1_obj.async_writes_flushed<NocOptions::POSTED>();
            cb_c2s_out.pop_front(tokens_per_chunk_combine);

            // Signal tilize drain that this chunk has been consumed
            // Device 2.0 migration: legacy primitive retained: precomposed uint64_t NoC address
            // (matmul_chunk_available_noc_addr) cannot be wrapped by Semaphore<>::inc which
            // binds to a per-program id, not a resolved address
            noc_semaphore_inc<true>(matmul_chunk_available_noc_addr, 1, 1, vchannel);
        }
    }

    // Signal combine cores that all expert data is written
    Semaphore<> combine_sem(combine_semaphore_id);
    for (uint32_t y = 0; y < height_shard_dim; ++y) {
        uint32_t idx = combine_core_x + y * width_shard_dim;
        combine_sem.up(noc1_obj, output_shard_core_map[2 * idx], output_shard_core_map[2 * idx + 1], 1, vchannel);
    }
    // The combine_sem.up() calls above issue on NoC 1 (matching the original
    // noc_async_atomic_barrier(/*noc=*/1) call), so barrier on noc1_obj.
    noc1_obj.async_atomic_barrier();
}
