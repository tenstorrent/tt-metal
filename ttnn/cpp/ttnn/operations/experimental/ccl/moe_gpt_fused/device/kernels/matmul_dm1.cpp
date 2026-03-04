// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// DM1 (RISCV_0 / NOC_1) for moe_gpt_fused
// Ring A2A + combine core output write (DeepSeek moe_compute layout)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "moe_gpt_fused_ring_common.h"

namespace detail {
inline uint32_t div_up(const uint32_t a, const uint32_t b) { return (a + b - 1) / b; }
}  // namespace detail

void kernel_main() {
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t num_cores = get_named_compile_time_arg_val("num_cores");

    // Combine output constants
    constexpr uint32_t height_shard_dim = get_named_compile_time_arg_val("height_shard_dim");
    constexpr uint32_t width_shard_dim = get_named_compile_time_arg_val("width_shard_dim");
    constexpr uint32_t combine_shard_width_tiles = get_named_compile_time_arg_val("combine_shard_width_tiles");
    constexpr uint32_t tile_width = get_named_compile_time_arg_val("tile_width");
    constexpr uint32_t tile_width_size_bytes = get_named_compile_time_arg_val("tile_width_size_bytes");

    std::array<uint32_t, 2 * height_shard_dim * width_shard_dim> output_shard_core_map = OUTPUT_SHARD_CORE_MAP;

    // Runtime args
    uint32_t argidx = 0;
    const auto dram_bank_id = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);
    const auto w0_w1_addr = get_arg_val<uint32_t>(argidx++);
    const auto w2_addr = get_arg_val<uint32_t>(argidx++);
    const auto ring_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_core_id = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_x = get_arg_val<uint32_t>(argidx++);
    const auto ring_neighbor_physical_y = get_arg_val<uint32_t>(argidx++);
    const auto input_dram_addr = get_arg_val<uint32_t>(argidx++);
    const auto combine_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto k_start_tile = get_arg_val<uint32_t>(argidx++);
    const auto output_base_l1_addr = get_arg_val<uint32_t>(argidx++);

    // CBs
    constexpr auto cb_s2c_in = tt::CBIndex::c_1;
    constexpr auto cb_c2w_rdy = tt::CBIndex::c_2;
    constexpr auto cb_w2c_rdy = tt::CBIndex::c_3;
    constexpr auto cb_s2c_in2 = tt::CBIndex::c_4;

    // Aliases
    constexpr auto cb_c2s_out = tt::CBIndex::c_14;  // untilized ROW_MAJOR output

    // Tile sizes
    constexpr uint32_t in_tile_size = get_tile_size(cb_s2c_in);
    constexpr uint32_t in2_tile_size = get_tile_size(cb_s2c_in2);

    // Constants
    constexpr uint32_t num_w0_w1_tiles_h = moe_gpt_fused_ring::NUM_W0_W1_TILES_H;  // 90

    const uint32_t tiles_per_core = moe_gpt_fused_ring::W2_TILES_PER_CORE_A[ring_core_id];

    // Ring setup
    constexpr uint32_t num_a2a_iters = moe_gpt_fused_ring::NUM_A2A_ITERS_A;
    constexpr uint32_t num_a2a_steps_per_iter = moe_gpt_fused_ring::NUM_CORES;
    constexpr uint32_t tiles_per_step = moe_gpt_fused_ring::IN2_TILES_PER_STEP_A;

    //-------------------------------------------------------------------------
    // Ring NoC setup
    //-------------------------------------------------------------------------
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

    noc_async_write_one_packet_set_state<true>(neighbor_base_addr, a2a_packet_size, 1, vchannel);
    noc_inline_dw_write_set_state<true, false>(neighbor_semaphore_noc_addr, 0, 0xF, write_at_cmd_buf, 1, vchannel);

    //-------------------------------------------------------------------------
    // Expert loop - Ring A2A
    //-------------------------------------------------------------------------
    for (uint32_t expert_id = 0; expert_id < num_experts; ++expert_id) {
        // Wait for compute to signal SwiGLU output ready
        cb_wait_front(cb_c2w_rdy, 1);
        cb_pop_front(cb_c2w_rdy, 1);

        for (uint32_t i = 0; i < num_a2a_iters; ++i) {
            for (uint32_t step = 0; step < num_a2a_steps_per_iter; ++step) {
                // Wait for data from predecessor
                while ((*my_semaphore_ptr) < semaphore_value) {
                };

                // Signal compute that A2A data is ready
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

                noc_async_posted_writes_flushed();

                noc_inline_dw_write_with_state<false, true, true, false, true>(++semaphore_value);

                noc_async_posted_writes_flushed();
            }
        }

        // Cross-expert boundary barrier
        if (expert_id < num_experts - 1) {
            while ((*my_semaphore_ptr) < semaphore_value) {
            };
            cb_reserve_back(cb_w2c_rdy, 1);
            cb_push_back(cb_w2c_rdy, 1);
        }
    }

    //-------------------------------------------------------------------------
    // Combine core output write (DeepSeek moe_compute layout)
    // Untilized ROW_MAJOR data from c_14 → combine core L1
    //
    // Each shard contains E expert blocks of max_tokens_per_height_shard rows.
    // Tokens are distributed round-robin across height shards.
    // Layout per shard: [E0 rows][E1 rows][E2 rows][E3 rows]
    //-------------------------------------------------------------------------
    constexpr uint32_t source_width_tiles = moe_gpt_fused_ring::SOURCE_WIDTH_TILES;  // 8
    constexpr uint32_t tokens_per_chunk = moe_gpt_fused_ring::TOKENS_PER_CHUNK;
    constexpr uint32_t RING_CORES_PER_COMBINE_COL = moe_gpt_fused_ring::RING_CORES_PER_COMBINE_COL;
    constexpr uint32_t num_tokens_total = get_named_compile_time_arg_val("num_tokens_total");

    // Expert block offset within each shard (same as DeepSeek dm1.cpp line 113)
    constexpr uint32_t shard_offset_per_expert_bytes =
        num_tokens_total / height_shard_dim * combine_shard_width_tiles * tile_width_size_bytes;

    const uint32_t output_width_tiles_core = tiles_per_core;
    const uint32_t width_tile_base = moe_gpt_fused_ring::COMBINE_W_OFFSET_PER_CORE_A[ring_core_id];
    const uint32_t combine_core_x = ring_core_id / RING_CORES_PER_COMBINE_COL;

    // Semaphore for signaling combine cores
    uint32_t combine_semaphore_addr = get_semaphore(combine_semaphore_id);

    for (uint32_t e = 0; e < num_experts; ++e) {
        // TODO(T=128): Replace with per-expert active token count from gather metadata
        const uint32_t active_tokens = tokens_per_chunk;
        const uint32_t max_tokens_per_height_shard = detail::div_up(active_tokens, height_shard_dim);
        const uint32_t expert_offset_bytes = shard_offset_per_expert_bytes * e;

        // TODO(T=128): Add chunk loop here. For chunk c:
        //   dest_height_shard_start = (c * tile_height) / max_tokens_per_height_shard
        //   shard_row_start = (c * tile_height) % max_tokens_per_height_shard
        const uint32_t dest_height_shard_start = 0;
        const uint32_t shard_row_start = 0;

        cb_wait_front(cb_c2s_out, tokens_per_chunk);  // 32 pages of untilized data
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

            // Token-by-token: dest core changes when height shard changes
            // (follows DeepSeek dm1.cpp lines 282-308)
            uint32_t dest_height_shard = dest_height_shard_start;
            uint32_t shard_row = shard_row_start;

            for (uint32_t bt = 0; bt < tokens_per_chunk; ++bt) {
                const uint32_t shard_row_offset_bytes = shard_row * combine_shard_width_tiles * tile_width_size_bytes;

                const auto dest_noc_x =
                    output_shard_core_map[2 * (dest_height_shard * width_shard_dim + dest_width_shard)];
                const auto dest_noc_y =
                    output_shard_core_map[2 * (dest_height_shard * width_shard_dim + dest_width_shard) + 1];

                const uint64_t dest_noc_addr_base = get_noc_addr(dest_noc_x, dest_noc_y, output_base_l1_addr, 1);
                noc_async_write_one_packet_set_state</*posted=*/true>(
                    dest_noc_addr_base, width_transfer_bytes, /*noc=*/1, vchannel);

                const uint32_t dest_l1_addr =
                    output_base_l1_addr + expert_offset_bytes + dest_width_offset_bytes + shard_row_offset_bytes;

                const uint32_t source_l1_addr =
                    source_base_l1_addr + (bt * source_width_tiles + width_tiles_sent) * tile_width_size_bytes;

                noc_async_write_one_packet_with_state</*posted=*/true>(source_l1_addr, dest_l1_addr);

                if (++shard_row == max_tokens_per_height_shard) {
                    ++dest_height_shard;
                    shard_row = 0;
                }
            }
            width_tiles_sent += width_transfer_tiles;
            width_tiles_to_send -= width_transfer_tiles;
        }

        noc_async_posted_writes_flushed(1);
        cb_pop_front(cb_c2s_out, tokens_per_chunk);
    }

    // Signal combine cores that all expert data is written
    for (uint32_t y = 0; y < height_shard_dim; ++y) {
        uint32_t idx = combine_core_x + y * width_shard_dim;
        uint64_t dest_sem_noc_addr =
            get_noc_addr(output_shard_core_map[2 * idx], output_shard_core_map[2 * idx + 1], combine_semaphore_addr);
        noc_semaphore_inc(dest_sem_noc_addr, 1, /*noc_id=*/1, vchannel);
    }
    noc_async_atomic_barrier(/*noc_idx=*/1);
}
