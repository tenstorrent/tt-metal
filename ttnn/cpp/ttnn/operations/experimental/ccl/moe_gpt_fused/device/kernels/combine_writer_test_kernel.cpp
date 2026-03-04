// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Standalone test kernel for the combine-write data path.
//
// Uses the DeepSeek moe_compute layout: tokens distributed round-robin
// across height shards, with per-expert blocks within each shard.
//
// No ring all-to-all, no matmul, no tilize — just the data movement.
//

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "moe_gpt_fused_ring_common.h"

namespace detail {
inline uint32_t div_up(const uint32_t a, const uint32_t b) { return (a + b - 1) / b; }
}  // namespace detail

void kernel_main() {
    // Named compile-time args (same as matmul_dm1.cpp)
    constexpr uint32_t num_experts = get_named_compile_time_arg_val("num_experts");
    constexpr uint32_t height_shard_dim = get_named_compile_time_arg_val("height_shard_dim");
    constexpr uint32_t width_shard_dim = get_named_compile_time_arg_val("width_shard_dim");
    constexpr uint32_t combine_shard_width_tiles = get_named_compile_time_arg_val("combine_shard_width_tiles");
    constexpr uint32_t tile_width = get_named_compile_time_arg_val("tile_width");
    constexpr uint32_t tile_width_size_bytes = get_named_compile_time_arg_val("tile_width_size_bytes");
    constexpr uint32_t num_tokens_total = get_named_compile_time_arg_val("num_tokens_total");

    // Expert block offset within each shard (same as DeepSeek dm1.cpp line 113)
    constexpr uint32_t shard_offset_per_expert_bytes =
        num_tokens_total / height_shard_dim * combine_shard_width_tiles * tile_width_size_bytes;

    // OUTPUT_SHARD_CORE_MAP define injected by host (serialized physical coords of combine cores)
    std::array<uint32_t, 2 * height_shard_dim * width_shard_dim> output_shard_core_map = OUTPUT_SHARD_CORE_MAP;

    // Runtime args
    uint32_t argidx = 0;
    const auto ring_core_id = get_arg_val<uint32_t>(argidx++);
    const auto combine_semaphore_id = get_arg_val<uint32_t>(argidx++);
    const auto output_base_l1_addr = get_arg_val<uint32_t>(argidx++);
    const auto vchannel = get_arg_val<uint32_t>(argidx++);

    // CB alias (same as matmul_dm1.cpp)
    constexpr auto cb_c2s_out = tt::CBIndex::c_14;

    // Constants from moe_gpt_fused_ring_common.h
    constexpr uint32_t source_width_tiles = moe_gpt_fused_ring::SOURCE_WIDTH_TILES;                  // 8
    constexpr uint32_t tokens_per_chunk = moe_gpt_fused_ring::TOKENS_PER_CHUNK;                      // 32
    constexpr uint32_t RING_CORES_PER_COMBINE_COL = moe_gpt_fused_ring::RING_CORES_PER_COMBINE_COL;  // 4

    const uint32_t output_width_tiles_core = moe_gpt_fused_ring::W2_TILES_PER_CORE_A[ring_core_id];
    const uint32_t width_tile_base = moe_gpt_fused_ring::COMBINE_W_OFFSET_PER_CORE_A[ring_core_id];
    const uint32_t combine_core_x = ring_core_id / RING_CORES_PER_COMBINE_COL;

    // Semaphore for signaling combine cores
    uint32_t combine_semaphore_addr = get_semaphore(combine_semaphore_id);

    //-------------------------------------------------------------------------
    // Setup sharded buffer: make pre-loaded data available via CB.
    // cb_descriptor_from_sharded_tensor binds the CB to the tensor's L1 buffer,
    // but cb_wait_front requires pages to be "pushed" first.
    //-------------------------------------------------------------------------
    constexpr uint32_t total_pages = num_experts * tokens_per_chunk;  // 4 * 32 = 128
    cb_reserve_back(cb_c2s_out, total_pages);
    cb_push_back(cb_c2s_out, total_pages);

    //-------------------------------------------------------------------------
    // Combine-write loop (DeepSeek moe_compute layout)
    // Tokens distributed round-robin across height shards.
    // Each shard: [E0 block][E1 block][E2 block][E3 block]
    //-------------------------------------------------------------------------
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

    //-------------------------------------------------------------------------
    // Signal combine cores in this sender's width column
    //-------------------------------------------------------------------------
    for (uint32_t y = 0; y < height_shard_dim; ++y) {
        uint32_t idx = combine_core_x + y * width_shard_dim;
        uint64_t dest_sem_noc_addr =
            get_noc_addr(output_shard_core_map[2 * idx], output_shard_core_map[2 * idx + 1], combine_semaphore_addr);
        noc_semaphore_inc(dest_sem_noc_addr, 1, /*noc_id=*/1, vchannel);
    }
    noc_async_atomic_barrier(/*noc_idx=*/1);
}
