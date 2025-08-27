// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "dataflow_common.hpp"
#include "debug/dprint.h"

void kernel_main() {
    DPRINT << "RING_READER: Starting kernel_main()" << ENDL();
    // Standard SDPA compile-time arguments
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Skt = get_compile_time_arg_val(3);  // Global K sequence in tiles
    constexpr uint32_t DHt = get_compile_time_arg_val(4);
    constexpr uint32_t vDHt = get_compile_time_arg_val(5);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t local_q_num_chunks = get_compile_time_arg_val(7);  // Local Q chunks for this device
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(9);
    constexpr uint32_t valid_local_Sqt = get_compile_time_arg_val(10);  // Valid local Q tiles
    constexpr uint32_t valid_Skt = get_compile_time_arg_val(11);        // Valid K tiles
    constexpr uint32_t is_causal = get_compile_time_arg_val(12) == 1;

    // Ring-specific compile-time arguments
    constexpr uint32_t ring_size = get_compile_time_arg_val(13);
    constexpr uint32_t ring_id = get_compile_time_arg_val(14);
    constexpr uint32_t first_chunk_id = get_compile_time_arg_val(15);
    constexpr uint32_t second_chunk_id = get_compile_time_arg_val(16);

    DPRINT << "RING_READER: Parsed compile-time args - ring_size=" << (uint32_t)ring_size
           << " ring_id=" << (uint32_t)ring_id << ENDL();
    constexpr uint32_t global_chunk_size = get_compile_time_arg_val(17);  // Chunk size in tiles (fixed from positions)

    // Runtime arguments
    uint32_t argidx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(argidx++);
    const uint32_t mask_addr = get_arg_val<uint32_t>(argidx++);        // Unused - we generate masks internally
    const uint32_t page_table_addr = get_arg_val<uint32_t>(argidx++);  // Unused - no chunked support yet
    const uint32_t core_id = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t local_q_start = get_arg_val<uint32_t>(argidx++);  // Local chunk indices
    const uint32_t local_q_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t chunked_q_chunk_offset = get_arg_val<uint32_t>(argidx++);  // Unused

    const uint32_t local_q_chunks_per_core = local_q_end - local_q_start;

    // global_chunk_size is now passed in tile units from program factory
    const uint32_t global_chunk_size_t = global_chunk_size;
    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;

    constexpr bool is_dram = true;

    // Circular buffer indices
    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;

    // Tile parameters
    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);

    constexpr DataFormat q_data_format = get_dataformat(cb_q_in);
    constexpr DataFormat k_data_format = get_dataformat(cb_k_in);
    constexpr DataFormat v_data_format = get_dataformat(cb_v_in);

    constexpr uint32_t q_heads_per_kv = NQH / NKH;
    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, 1>();  // Single reader per core

    // Initialize memory readers
    const InterleavedAddrGenFast<is_dram> q_reader = {
        .bank_base_address = q_addr, .page_size = q_tile_bytes, .data_format = q_data_format};
    const InterleavedAddrGenFast<is_dram> k_reader = {
        .bank_base_address = k_addr, .page_size = k_tile_bytes, .data_format = k_data_format};
    const InterleavedAddrGenFast<is_dram> v_reader = {
        .bank_base_address = v_addr, .page_size = v_tile_bytes, .data_format = v_data_format};

    // Q tensor shape for tile ID calculation: [B, NQH, Sq_global, DHt]
    const auto q_tile_shape = TensorTileShape(B, NQH, Skt, DHt);  // Use global Skt for Q too
    const auto k_tile_shape = TensorTileShape(B, NKH, Skt, DHt);

    DPRINT << "RING_READER: Starting processing loops" << ENDL();

    // Process each batch
    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        DPRINT << "RING_READER: Processing batch " << (uint32_t)nb << ENDL();
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            DPRINT << "RING_READER: Processing head " << (uint32_t)nq << ENDL();
            for (uint32_t local_q_iter = 0; local_q_iter < local_q_chunks_per_core; ++local_q_iter) {
                DPRINT << "RING_READER: Processing local_q_iter " << (uint32_t)local_q_iter << ENDL();
                // Ring-distributed Q chunk selection
                // Map local iteration to one of the device's two assigned chunks
                uint32_t local_q_chunk_index = local_q_start + local_q_iter;
                uint32_t global_q_chunk;

                // Each device processes exactly 2 chunks from its allocation
                // local_q_num_chunks should be 2, so local chunks are distributed as:
                // local_q_chunk_index 0 -> first_chunk_id
                // local_q_chunk_index 1 -> second_chunk_id
                if ((local_q_chunk_index % 2) == 0) {
                    global_q_chunk = first_chunk_id;  // First chunk for this device
                } else {
                    global_q_chunk = second_chunk_id;  // Second chunk for this device
                }

                // Calculate global query positions for this chunk
                uint32_t global_q_start = global_q_chunk * global_chunk_size_t;
                uint32_t global_q_end = global_q_start + global_chunk_size_t;

                // Read Q chunk from global tensor
                // Use global sequence bounds for validation
                const uint32_t global_valid_Sqt = Skt;  // Global sequence length in tiles
                const uint32_t q_row_start_tile = std::min(global_q_start, global_valid_Sqt);
                const uint32_t q_row_end_tile = std::min(global_q_end, global_valid_Sqt);
                const uint32_t q_row_tile_count = q_row_end_tile - q_row_start_tile;
                const uint32_t q_tile_id = q_tile_shape.id_of(nb, nq, q_row_start_tile, 0);

                DPRINT << "RING_READER: About to read Q chunk - tile_id=" << (uint32_t)q_tile_id
                       << " row_count=" << (uint32_t)q_row_tile_count << ENDL();

                read_chunk_with_padding<is_dram, q_tile_bytes>(
                    q_reader, cb_q_in, q_tile_id, q_row_tile_count, DHt, Sq_chunk_t, DHt, barrier_threshold);

                DPRINT << "RING_READER: Finished reading Q chunk, pushed to cb_q_in" << ENDL();

                // Calculate causal bounds for this Q chunk
                uint32_t q_high_idx;
                if constexpr (is_causal) {
                    q_high_idx = global_q_end;  // This Q chunk can attend up to its highest position
                } else {
                    q_high_idx = Skt;  // Non-causal: attend to all keys
                }

                const uint32_t kv_head = nq / q_heads_per_kv;

                // Read K and V chunks (mask generation handled by writer kernel)
                DPRINT << "RING_READER: Starting K/V loop - q_high_idx=" << (uint32_t)q_high_idx
                       << " Sk_chunk_t=" << (uint32_t)Sk_chunk_t << ENDL();
                for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {
                    DPRINT << "RING_READER: Reading K/V chunk " << (uint32_t)k_chunk << ENDL();
                    const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
                    const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;

                    // Read K chunk
                    const uint32_t k_row_start_tile = std::min(k_chunk * Sk_chunk_t, valid_Skt);
                    const uint32_t k_row_end_tile = std::min(k_row_start_tile + Sk_chunk_t, valid_Skt);
                    const uint32_t k_row_tile_count = k_row_end_tile - k_row_start_tile;
                    DPRINT << "RING_READER: k_row_tile_count=" << (uint32_t)k_row_tile_count << " DHt=" << (uint32_t)DHt
                           << " total_k_tiles=" << (uint32_t)(k_row_tile_count * DHt) << ENDL();
                    const uint32_t k_start_tile_id = k_tile_shape.id_of(nb, kv_head, k_row_start_tile, 0);

                    read_chunk_with_padding<is_dram, k_tile_bytes>(
                        k_reader,
                        cb_k_in,
                        k_start_tile_id,
                        k_row_tile_count,
                        DHt,
                        Sk_chunk_t,
                        DHt,
                        barrier_threshold,
                        true);  // transpose = true for K

                    // Read V chunk
                    read_chunk_with_padding<is_dram, v_tile_bytes>(
                        v_reader,
                        cb_v_in,
                        k_start_tile_id,
                        k_row_tile_count,
                        vDHt,
                        Sk_chunk_t,
                        vDHt,
                        barrier_threshold);
                    DPRINT << "RING_READER: Finished reading K/V chunk " << (uint32_t)k_chunk << ENDL();
                }
            }
        }
    }
}
