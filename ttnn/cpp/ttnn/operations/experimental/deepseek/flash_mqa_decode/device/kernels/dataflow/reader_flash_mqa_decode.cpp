// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/experimental/deepseek/flash_mqa_decode/device/kernels/flash_mqa_decode_common.hpp"

void kernel_main() {
    constexpr uint32_t DHt = get_named_compile_time_arg_val("DHt");
    constexpr uint32_t Sk_chunk_t = get_named_compile_time_arg_val("Sk_chunk_t");
    constexpr uint32_t k_chunk_size = get_named_compile_time_arg_val("k_chunk_size");
    constexpr uint32_t num_s_blocks = get_named_compile_time_arg_val("num_s_blocks");
    constexpr uint32_t heads_per_core = get_named_compile_time_arg_val("heads_per_core");
    constexpr uint32_t q_row_bytes = get_named_compile_time_arg_val("q_row_bytes");
    constexpr uint32_t ring_capacity = get_named_compile_time_arg_val("ring_capacity");
    constexpr uint32_t has_ring_offset = get_named_compile_time_arg_val("has_ring_offset");
    constexpr uint32_t max_ring_first_tile_row = get_named_compile_time_arg_val("max_ring_first_tile_row");
    constexpr uint32_t selected_capacity = get_named_compile_time_arg_val("selected_capacity");
    constexpr uint32_t selected_first_tile_row = get_named_compile_time_arg_val("selected_first_tile_row");
    constexpr uint32_t cb_q_rm = get_named_compile_time_arg_val("cb_q_rm");
    constexpr uint32_t cb_k_in = get_named_compile_time_arg_val("cb_k_in");
    constexpr uint32_t cb_mask = get_named_compile_time_arg_val("cb_mask");
    constexpr uint32_t cb_lengths = get_named_compile_time_arg_val("cb_lengths");
    constexpr uint32_t cb_lengths_scratch = get_named_compile_time_arg_val("cb_lengths_scratch");

    constexpr auto q_args = TensorAccessorArgs<0>();
    constexpr auto kv_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto selected_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    constexpr auto lengths_args = TensorAccessorArgs<selected_args.next_compile_time_args_offset()>();

    uint32_t arg_idx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t kv_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t selected_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t lengths_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t s_block = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t head_group = get_arg_val<uint32_t>(arg_idx++);

    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;

    const auto lengths_reader = TensorAccessor(lengths_args, lengths_addr);
    const uint32_t lengths_l1 = get_write_ptr(cb_lengths_scratch);
    noc_async_read(lengths_reader.get_noc_addr(0), lengths_l1, (has_ring_offset ? 3 : 2) * sizeof(uint32_t));
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* lengths_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lengths_l1);
    const flash_mqa_decode::Lengths lengths =
        flash_mqa_decode::clamp_lengths(lengths_ptr[0], lengths_ptr[1], ring_capacity, selected_capacity);
    const uint32_t ring_first_tile_row =
        has_ring_offset ? flash_mqa_decode::min_u32(lengths_ptr[2] / 32, max_ring_first_tile_row) : 0;
    cb_reserve_back(cb_lengths, 1);
    volatile tt_l1_ptr uint32_t* compute_lengths =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_lengths));
    compute_lengths[0] = lengths.ring;
    compute_lengths[1] = lengths.selected;
    cb_push_back(cb_lengths, 1);

    const uint32_t total_chunks = flash_mqa_decode::num_chunks(lengths, k_chunk_size);
    const uint32_t block_chunks = flash_mqa_decode::num_block_chunks(s_block, total_chunks, num_s_blocks);
    if (block_chunks == 0) {
        return;
    }

    const auto q_reader = TensorAccessor(q_args, q_addr);
    cb_reserve_back(cb_q_rm, DHt);
    uint32_t q_l1 = get_write_ptr(cb_q_rm);
    for (uint32_t h = 0; h < heads_per_core; ++h) {
        noc_async_read(q_reader.get_noc_addr(head_group * heads_per_core + h), q_l1, q_row_bytes);
        q_l1 += q_row_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_q_rm, DHt);

    const auto kv_reader = TensorAccessor(kv_args, kv_addr);
    const auto selected_reader = TensorAccessor(selected_args, selected_addr);
    for (uint32_t j = 0; j < block_chunks; ++j) {
        const flash_mqa_decode::Chunk chunk =
            flash_mqa_decode::chunk_at(s_block + j * num_s_blocks, lengths, k_chunk_size);
        if (chunk.valid_rows < k_chunk_size) {
            // One bf16 per key, 0 for valid keys and -inf past them.
            cb_reserve_back(cb_mask, 1);
            volatile tt_l1_ptr uint32_t* mask_ptr =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_mask));
            uint32_t i = 0;
            for (; i < chunk.valid_rows / 2; ++i) {
                mask_ptr[i] = 0x00000000;
            }
            if (chunk.valid_rows % 2 == 1) {
                mask_ptr[i++] = 0xFF800000;
            }
            for (; i < k_chunk_size / 2; ++i) {
                mask_ptr[i] = 0xFF80FF80;
            }
            cb_push_back(cb_mask, 1);
        }

        cb_reserve_back(cb_k_in, k_chunk_tiles);
        uint32_t k_l1 = get_write_ptr(cb_k_in);
        const uint32_t k_tile_bytes = get_tile_size(cb_k_in);
        const uint32_t first_tile_row = chunk.first_row / 32;
        if (chunk.segment == 0) {
            const uint32_t first_page = (ring_first_tile_row + first_tile_row) * DHt;
            for (uint32_t t = 0; t < k_chunk_tiles; ++t) {
                noc_async_read_page(first_page + t, kv_reader, k_l1);
                k_l1 += k_tile_bytes;
            }
        } else {
            const uint32_t first_page = (selected_first_tile_row + first_tile_row) * DHt;
            for (uint32_t t = 0; t < k_chunk_tiles; ++t) {
                noc_async_read_page(first_page + t, selected_reader, k_l1);
                k_l1 += k_tile_bytes;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_k_in, k_chunk_tiles);
    }
}
