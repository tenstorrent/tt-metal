// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Masked reader for per_token_cast_back. input_e4m3 is a per-device dispatch buffer (T rows x H).
// Only this device's valid expert-region rows are decompressed: for each local expert e in
// [expert_start, expert_end), the valid rows are [start, start+count) where
//   start = expert_region_offsets[counter_offset + e], count = expert_token_counts[counter_offset + e]
// (clamped against T). Each valid dispatch row r uses scale row
// metadata[r][1] (token_idx). Within an expert region the rows are contiguous, so the e4m3 reads and
// the tile_h-block / broadcast-operand machinery are identical to the plain reader — only the scale
// page is gathered by token_idx, and num_blocks is dynamic (published to compute via cb_nblocks).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t input_e4m3_addr = get_arg_val<uint32_t>(0);
    uint32_t scale_addr = get_arg_val<uint32_t>(1);
    uint32_t counts_addr = get_arg_val<uint32_t>(2);
    uint32_t offsets_addr = get_arg_val<uint32_t>(3);
    uint32_t metadata_addr = get_arg_val<uint32_t>(4);
    uint32_t expert_start = get_arg_val<uint32_t>(5);
    uint32_t expert_end = get_arg_val<uint32_t>(6);
    uint32_t width = get_arg_val<uint32_t>(7);  // H

    constexpr uint32_t cb_input_e4m3 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scale_bcast = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scale_scratch = get_compile_time_arg_val(2);
    constexpr uint32_t cb_counts = get_compile_time_arg_val(3);
    constexpr uint32_t cb_offsets = get_compile_time_arg_val(4);
    constexpr uint32_t cb_metadata = get_compile_time_arg_val(5);
    constexpr uint32_t cb_nblocks = get_compile_time_arg_val(6);
    constexpr uint32_t input_e4m3_block_bytes = get_compile_time_arg_val(7);  // 128
    constexpr uint32_t block_ht = get_compile_time_arg_val(8);                // 1
    constexpr uint32_t scale_aligned_page_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t tile_h = get_compile_time_arg_val(10);
    constexpr uint32_t tile_w = get_compile_time_arg_val(11);
    constexpr uint32_t face_h = get_compile_time_arg_val(12);
    constexpr uint32_t face_w = get_compile_time_arg_val(13);
    constexpr uint32_t counts_aligned_page_bytes = get_compile_time_arg_val(14);
    constexpr uint32_t metadata_aligned_page_bytes = get_compile_time_arg_val(15);
    constexpr uint32_t counts_pages = get_compile_time_arg_val(16);
    constexpr uint32_t counter_offset = get_compile_time_arg_val(17);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(18);
    constexpr uint32_t max_dispatch_buffer_token_size = get_compile_time_arg_val(19);
    constexpr uint32_t num_routed_experts = get_compile_time_arg_val(20);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(21);
    constexpr uint32_t seq_len_per_chip = get_compile_time_arg_val(22);

    constexpr uint32_t block_w = 128;
    constexpr uint32_t tiles_per_block = block_w / tile_w;
    constexpr uint32_t face_elems = face_h * face_w;
    constexpr uint32_t faces_per_row = tile_w / face_w;
    constexpr uint32_t FACE_ROWS = tile_h / face_h;
    constexpr uint32_t FACE_ROW_STRIDE_BYTES = faces_per_row * face_elems * sizeof(float);
    constexpr uint32_t FACE_W_BYTES = face_w * sizeof(float);
    // metadata fields: [linearized_mesh_coord (source), token_idx (source-local), topk, expert, weight].
    constexpr uint32_t METADATA_SRC_COORD_FIELD = 0;
    constexpr uint32_t METADATA_TOKEN_IDX_FIELD = 1;

    (void)block_ht;
    (void)num_routed_experts;

    constexpr auto e4m3_args = TensorAccessorArgs<23>();
    constexpr auto scale_args = TensorAccessorArgs<e4m3_args.next_compile_time_args_offset()>();
    constexpr auto counts_args = TensorAccessorArgs<scale_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    const auto input_e4m3 = TensorAccessor(e4m3_args, input_e4m3_addr);
    const auto scale = TensorAccessor(scale_args, scale_addr);
    const auto counts_acc = TensorAccessor(counts_args, counts_addr);
    const auto offsets_acc = TensorAccessor(offsets_args, offsets_addr);
    const auto metadata = TensorAccessor(metadata_args, metadata_addr);

    Noc noc;
    CircularBuffer cb_input_e4m3_obj(cb_input_e4m3);
    CircularBuffer cb_scale_bcast_obj(cb_scale_bcast);
    CircularBuffer cb_scale_scratch_obj(cb_scale_scratch);
    CircularBuffer cb_counts_obj(cb_counts);
    CircularBuffer cb_offsets_obj(cb_offsets);
    CircularBuffer cb_metadata_obj(cb_metadata);
    CircularBuffer cb_nblocks_obj(cb_nblocks);

    const uint32_t blocks_per_row = width >> 7;  // H / 128

    // --- Load counts[] and offsets[] into L1 scratch; index flat by counter_offset. ---
    cb_counts_obj.reserve_back(counts_pages);
    cb_offsets_obj.reserve_back(counts_pages);
    const uint32_t counts_base = cb_counts_obj.get_write_ptr();
    const uint32_t offsets_base = cb_offsets_obj.get_write_ptr();
    for (uint32_t i = 0; i < counts_pages; ++i) {
        noc.async_read(
            counts_acc,
            cb_counts_obj,
            counts_aligned_page_bytes,
            {.page_id = i},
            {.offset_bytes = i * counts_aligned_page_bytes});
        noc.async_read(
            offsets_acc,
            cb_offsets_obj,
            counts_aligned_page_bytes,
            {.page_id = i},
            {.offset_bytes = i * counts_aligned_page_bytes});
    }
    noc.async_read_barrier();
    CoreLocalMem<volatile uint32_t> counts_l1(counts_base);
    CoreLocalMem<volatile uint32_t> offsets_l1(offsets_base);

    // Reader-private scratch buffers (reused every block).
    cb_scale_scratch_obj.reserve_back(1);
    cb_metadata_obj.reserve_back(1);
    const uint32_t scale_scratch = cb_scale_scratch_obj.get_write_ptr();
    const uint32_t metadata_scratch = cb_metadata_obj.get_write_ptr();

    // --- Pass 1: compute total block count for this core's experts; publish to compute. ---
    uint32_t total_blocks = 0;
    for (uint32_t e = expert_start; e < expert_end; ++e) {
        uint32_t start_page = offsets_l1[counter_offset + e];
        uint32_t count = counts_l1[counter_offset + e];
        if (start_page >= max_dispatch_buffer_token_size) {
            count = 0;
        } else if (start_page + count > max_dispatch_buffer_token_size) {
            count = max_dispatch_buffer_token_size - start_page;
        }
        const uint32_t total_scale_blocks = count * blocks_per_row;
        total_blocks += (total_scale_blocks + tile_h - 1) / tile_h;
    }
    cb_nblocks_obj.reserve_back(1);
    CoreLocalMem<volatile uint32_t> nblocks_mem(cb_nblocks_obj.get_write_ptr());
    nblocks_mem[0] = total_blocks;
    cb_nblocks_obj.push_back(1);

    // --- Pass 2: stream each expert region's valid rows as tile_h-scale-block compute blocks. ---
    for (uint32_t e = expert_start; e < expert_end; ++e) {
        uint32_t start_page = offsets_l1[counter_offset + e];
        uint32_t count = counts_l1[counter_offset + e];
        if (start_page >= max_dispatch_buffer_token_size) {
            count = 0;
        } else if (start_page + count > max_dispatch_buffer_token_size) {
            count = max_dispatch_buffer_token_size - start_page;
        }
        if (count == 0) {
            continue;
        }
        const uint32_t end_row = start_page + count;
        const uint32_t total_blocks_e = count * blocks_per_row;
        const uint32_t num_blocks_e = (total_blocks_e + tile_h - 1) / tile_h;

        // Persistent (row, block_idx) cursor over this expert's contiguous valid rows.
        uint32_t current_row = start_page;
        uint32_t block_idx_in_row = 0;

        for (uint32_t blk = 0; blk < num_blocks_e; ++blk) {
            const uint32_t base = blk * tile_h;
            const uint32_t remaining = total_blocks_e - base;
            const uint32_t real_in_block = remaining < tile_h ? remaining : tile_h;
            const uint32_t block_start_row = current_row;
            const uint32_t block_start_idx = block_idx_in_row;

            // --- input_e4m3: fill [tile_h scale-block rows x 128] via bank-contiguous runs. ---
            cb_input_e4m3_obj.reserve_back(tiles_per_block);
            uint32_t last_row = current_row;
            {
                uint32_t row = current_row;
                uint32_t block_idx = block_idx_in_row;
                uint32_t slot = 0;
                while (slot < real_in_block && row < end_row) {
                    uint32_t blocks_left_in_row = blocks_per_row - block_idx;
                    uint32_t slots_left = real_in_block - slot;
                    uint32_t run = blocks_left_in_row < slots_left ? blocks_left_in_row : slots_left;
                    noc.async_read(
                        input_e4m3,
                        cb_input_e4m3_obj,
                        run * input_e4m3_block_bytes,
                        {.page_id = row, .offset_bytes = block_idx * input_e4m3_block_bytes},
                        {.offset_bytes = slot * input_e4m3_block_bytes});
                    last_row = row;
                    slot += run;
                    block_idx += run;
                    if (block_idx >= blocks_per_row) {
                        block_idx = 0;
                        ++row;
                    }
                }
                current_row = row;
                block_idx_in_row = block_idx;
            }

            // --- metadata: read each spanned row's metadata page to recover its token_idx. ---
            for (uint32_t t = block_start_row; t <= last_row; ++t) {
                noc.async_read(
                    metadata,
                    use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_metadata_obj),
                    metadata_aligned_page_bytes,
                    {.page_id = t},
                    {.offset_bytes = (t - block_start_row) * metadata_aligned_page_bytes});
            }
            noc.async_read_barrier();  // e4m3 + metadata complete

            // --- scale: gather each spanned row's scale page by token_idx into the scratch slot. ---
            CoreLocalMem<volatile uint32_t> meta_mem(metadata_scratch);
            for (uint32_t t = block_start_row; t <= last_row; ++t) {
                const uint32_t slot = t - block_start_row;
                const uint32_t meta_base = (slot * metadata_aligned_page_bytes >> 2);
                const uint32_t src_coord = meta_mem[meta_base + METADATA_SRC_COORD_FIELD];
                const uint32_t token_idx = meta_mem[meta_base + METADATA_TOKEN_IDX_FIELD];
                // The scale tensor is the dispatch group's gathered per-token scale; the source's
                // slice begins at (source row in group) * seq_len_per_chip. token_idx is source-local.
                // On a 1x1 mesh src_coord==0, so this is just scale[token_idx].
                const uint32_t scale_page = (src_coord / mesh_cols) * seq_len_per_chip + token_idx;
                noc.async_read(
                    scale,
                    use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_scale_scratch_obj),
                    scale_aligned_page_bytes,
                    {.page_id = scale_page},
                    {.offset_bytes = slot * scale_aligned_page_bytes});
            }
            noc.async_read_barrier();

            // --- build the bcast operand: column 0 row r = scale[token_of_row_r][block_r]. ---
            // Identical face-aware walk to the plain reader: scratch slot k = scale row for the
            // dispatch row (block_start_row + k), so the (tok_off, block_idx_b) cursor maps each
            // scale-block to the correct scratch slot and block index.
            cb_scale_bcast_obj.reserve_back(1);
            CoreLocalMem<volatile uint32_t> scratch_mem(scale_scratch);
            CoreLocalMem<volatile uint32_t> page(cb_scale_bcast_obj.get_write_ptr());
            uint32_t tok_off = 0;
            uint32_t block_idx_b = block_start_idx;
            uint32_t s = 0;
            uint32_t face_base_off = 0;
            for (uint32_t fr = 0; fr < FACE_ROWS; ++fr) {
                uint32_t col0_off = face_base_off;
                for (uint32_t r = 0; r < face_h; ++r) {
                    if (s < real_in_block) {
                        uint32_t val = scratch_mem[(tok_off >> 2) + block_idx_b];
                        page[col0_off >> 2] = val;
                        ++block_idx_b;
                        if (block_idx_b >= blocks_per_row) {
                            block_idx_b = 0;
                            tok_off += scale_aligned_page_bytes;
                        }
                    }
                    col0_off += FACE_W_BYTES;
                    ++s;
                }
                face_base_off += FACE_ROW_STRIDE_BYTES;
            }
            cb_scale_bcast_obj.push_back(1);
            cb_input_e4m3_obj.push_back(tiles_per_block);
        }
    }
}
