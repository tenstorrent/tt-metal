// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for masked_per_token_cast_back. Same block-streaming logic as per_token_cast_back, but the
// row range is not passed by the host: this kernel reads the per-expert region offsets / token counts /
// global-expert-idx table from DRAM, computes the valid prefix length total_valid_rows, and derives its
// own balanced tile-row slice from (core_id, num_cores). It publishes the resulting num_blocks to the
// compute kernel via the cb_control mailbox. Per block:
//   - input_e4m3: read each bank-contiguous run of scale blocks into cb_input_e4m3 with a single
//     NoC async read (mirrors the writer);
//   - scale: read the (few) tokens spanned by the block as full, page-aligned scale rows into a
//     reader-private scratch, then build the single bcast operand tile whose column 0 row r = the
//     scale of block row r = scale[token_r][block_r] (face-aware).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "api/debug/assert.h"

void kernel_main() {
    uint32_t input_e4m3_addr = get_arg_val<uint32_t>(0);
    uint32_t scale_addr = get_arg_val<uint32_t>(1);
    uint32_t region_addr = get_arg_val<uint32_t>(2);
    uint32_t counts_addr = get_arg_val<uint32_t>(3);
    uint32_t table_addr = get_arg_val<uint32_t>(4);
    uint32_t core_id = get_arg_val<uint32_t>(5);
    uint32_t width = get_arg_val<uint32_t>(6);  // H (elements per row)

    constexpr uint32_t cb_input_e4m3 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scale_bcast = get_compile_time_arg_val(1);
    constexpr uint32_t cb_scale_scratch = get_compile_time_arg_val(2);
    constexpr uint32_t input_e4m3_block_bytes = get_compile_time_arg_val(3);    // 128 (1 byte/elem)
    constexpr uint32_t block_ht = get_compile_time_arg_val(4);                  // BlockHt = 1
    constexpr uint32_t scale_aligned_page_bytes = get_compile_time_arg_val(5);  // aligned row footprint
    // Tile / face dims from the tensor's tile spec.
    constexpr uint32_t tile_h = get_compile_time_arg_val(6);
    constexpr uint32_t tile_w = get_compile_time_arg_val(7);
    constexpr uint32_t face_h = get_compile_time_arg_val(8);
    constexpr uint32_t face_w = get_compile_time_arg_val(9);
    constexpr uint32_t cb_control = get_compile_time_arg_val(10);
    constexpr uint32_t cb_region_scratch = get_compile_time_arg_val(11);
    constexpr uint32_t cb_counts_scratch = get_compile_time_arg_val(12);
    constexpr uint32_t cb_table_scratch = get_compile_time_arg_val(13);
    constexpr uint32_t num_cores = get_compile_time_arg_val(14);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(15);
    // Column offset (in scale elements) of the scale tail within each scale-source row. 0 for a plain
    // (M, H/128) scale tensor; = metadata_len - H/128 (i.e. skip the routing header) for the metadata path.
    constexpr uint32_t scale_col_offset = get_compile_time_arg_val(16);
    // Scale element width (bytes): 4 for a FLOAT32 scale tensor (or the int32-stored metadata tail), 2 for
    // BFLOAT16. The bcast operand keeps the scale's own dtype (no conversion), so both the scratch read and
    // the operand strides use it.
    constexpr uint32_t scale_elem_bytes = get_compile_time_arg_val(17);
    // Bounds for validating untrusted device metadata (watcher-build asserts below).
    constexpr uint32_t num_routed_experts = get_compile_time_arg_val(18);  // length of region/counts vectors
    constexpr uint32_t input_num_rows = get_compile_time_arg_val(19);      // M: input buffer row capacity
    constexpr uint32_t scale_elem_shift = scale_elem_bytes == 4 ? 2 : 1;
    constexpr uint32_t block_w = 128;
    constexpr uint32_t tiles_per_block = block_w / tile_w;
    constexpr uint32_t face_elems = face_h * face_w;                                           // scale elems per face
    constexpr uint32_t faces_per_row = tile_w / face_w;                                        // face columns per tile
    constexpr uint32_t FACE_ROWS = tile_h / face_h;                                            // face rows per tile
    constexpr uint32_t FACE_ROW_STRIDE_BYTES = faces_per_row * face_elems * scale_elem_bytes;  // per face row
    constexpr uint32_t FACE_W_BYTES = face_w * scale_elem_bytes;                               // per in-face row

    (void)block_ht;  // kept as a compile-time layout arg for tensor accessor offset stability

    constexpr auto input_e4m3_accessor_args = TensorAccessorArgs<20>();
    constexpr auto scale_args = TensorAccessorArgs<input_e4m3_accessor_args.next_compile_time_args_offset()>();
    constexpr auto region_args = TensorAccessorArgs<scale_args.next_compile_time_args_offset()>();
    constexpr auto counts_args = TensorAccessorArgs<region_args.next_compile_time_args_offset()>();
    constexpr auto table_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
    const auto input_e4m3 = TensorAccessor(input_e4m3_accessor_args, input_e4m3_addr);
    const auto scale = TensorAccessor(scale_args, scale_addr);
    const auto region = TensorAccessor(region_args, region_addr);
    const auto counts = TensorAccessor(counts_args, counts_addr);
    const auto table = TensorAccessor(table_args, table_addr);
    Noc noc;
    CircularBuffer cb_input_e4m3_obj(cb_input_e4m3);
    CircularBuffer cb_scale_bcast_obj(cb_scale_bcast);
    CircularBuffer cb_scale_scratch_obj(cb_scale_scratch);
    CircularBuffer cb_control_obj(cb_control);
    CircularBuffer cb_region_scratch_obj(cb_region_scratch);
    CircularBuffer cb_counts_scratch_obj(cb_counts_scratch);
    CircularBuffer cb_table_scratch_obj(cb_table_scratch);

    // --- Read the three metadata vectors (small, 1 page each) into private L1 scratch. ---
    cb_region_scratch_obj.reserve_back(1);
    cb_counts_scratch_obj.reserve_back(1);
    cb_table_scratch_obj.reserve_back(1);
    noc.async_read(region, cb_region_scratch_obj, region.get_aligned_page_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(counts, cb_counts_scratch_obj, counts.get_aligned_page_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read(table, cb_table_scratch_obj, table.get_aligned_page_size(), {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();

    const volatile tt_l1_ptr uint32_t* region_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_region_scratch_obj.get_write_ptr());
    const volatile tt_l1_ptr uint32_t* counts_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_counts_scratch_obj.get_write_ptr());
    const volatile tt_l1_ptr uint32_t* table_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_table_scratch_obj.get_write_ptr());

    // Valid prefix length = max over local experts of (region_offset[g] + ceil_tile(counts[g])).
    // Device metadata is untrusted: the asserts (watcher build only) catch a stale/garbage table entry
    // before it indexes past the region/counts scratch pages or drives a NoC read past the input buffers.
    uint32_t total_valid_rows = 0;
    for (uint32_t local_slot = 0; local_slot < experts_per_chip; ++local_slot) {
        const uint32_t global_expert_id = table_ptr[local_slot];
        ASSERT(global_expert_id < num_routed_experts);
        const uint32_t token_count = counts_ptr[global_expert_id];
        const uint32_t token_count_ceil = ((token_count + tile_h - 1) / tile_h) * tile_h;
        const uint32_t region_end = region_ptr[global_expert_id] + token_count_ceil;
        if (region_end > total_valid_rows) {
            total_valid_rows = region_end;
        }
    }
    ASSERT(total_valid_rows <= input_num_rows);

    // Balanced split across the whole grid over the FLATTENED compute-block space (rows x col-blocks),
    // so all cores stay busy even when tile_rows < num_cores. Each compute-block is tile_h consecutive
    // scale-blocks in row-major (token, col_block) order and may cross a row boundary; the streaming
    // cursor below already handles an arbitrary (start_row, start_col) start.
    const uint32_t blocks_per_row = width >> 7;  // H / 128 (block_w = 128); one-time shift
    const uint32_t total_tile_rows = total_valid_rows / tile_h;
    const uint32_t total_compute_blocks = total_tile_rows * blocks_per_row;
    const uint32_t cb_start = (total_compute_blocks * core_id) / num_cores;
    const uint32_t cb_end = (total_compute_blocks * (core_id + 1)) / num_cores;
    const uint32_t num_blocks = cb_end - cb_start;
    // Flattened scale-block indices (tile_h scale-blocks per compute-block). One-time div/mod only.
    const uint32_t start_flat = cb_start * tile_h;
    const uint32_t end_flat = cb_end * tile_h;  // exclusive
    const uint32_t start_row = start_flat / blocks_per_row;
    const uint32_t start_col = start_flat % blocks_per_row;
    const uint32_t end_row = (end_flat == 0) ? 0 : ((end_flat - 1) / blocks_per_row) + 1;
    const uint32_t total_blocks = num_blocks * tile_h;  // per-core scale-blocks (exact multiple of tile_h)

    // Publish num_blocks to the compute kernel (read via read_tile_value on the TRISCs).
    cb_control_obj.reserve_back(1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_control_obj.get_write_ptr())[0] = num_blocks;
    cb_control_obj.push_back(1);

    // Reader-private scratch: up to tile_h tokens' full scale rows (one slot per token in a block).
    cb_scale_scratch_obj.reserve_back(1);
    const uint32_t scratch = cb_scale_scratch_obj.get_write_ptr();

    // Persistent (row, block_idx) cursor over the flat block stream: no per-block div/mod (expensive
    // on the Baby RISC-V); advance block_idx by the run and reset to the next row with a conditional.
    uint32_t current_row = start_row;
    uint32_t block_idx_in_row = start_col;

    for (uint32_t blk = 0; blk < num_blocks; ++blk) {
        const uint32_t base = blk * tile_h;
        const uint32_t remaining = total_blocks - base;
        const uint32_t real_in_block = remaining < tile_h ? remaining : tile_h;
        const uint32_t block_start_row = current_row;
        const uint32_t block_start_idx = block_idx_in_row;

        // --- input_e4m3: fill the block as [tile_h scale-block rows x 128] via bank-contiguous runs ---
        cb_input_e4m3_obj.reserve_back(tiles_per_block);
        uint32_t last_row = current_row;  // row of the last run = the block's last token
        {
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
                    if (block_idx >= blocks_per_row) {  // run never crosses a row boundary, so this is exact
                        block_idx = 0;
                        ++row;
                    }
                }
                current_row = row;  // advance the persistent cursor for the next block
                block_idx_in_row = block_idx;
            }

            // --- scale: read the tokens spanned by this block as full page-aligned rows ---
            for (uint32_t t = block_start_row; t <= last_row; ++t) {
                noc.async_read(
                    scale,
                    use<CircularBuffer::AddrSelector::WRITE_PTR>(cb_scale_scratch_obj),
                    scale_aligned_page_bytes,
                    {.page_id = t},
                    {.offset_bytes = (t - block_start_row) * scale_aligned_page_bytes});
            }
            noc.async_read_barrier();
        }

        // --- build the bcast operand: column 0 row r = scale[token_r][block_r] (face-aware walk) ---
        // (tok_off, block_idx_b) cursor walks the block rows in face order -> no per-row div/mod. The
        // operand keeps the scale's own dtype (raw copy, no conversion), so the multiply's SrcB matches
        // the bf16/fp32 input tile.
        cb_scale_bcast_obj.reserve_back(1);
        const uint32_t page_ptr = cb_scale_bcast_obj.get_write_ptr();
        auto build_bcast = [&](auto scratch_mem, auto page) {
            uint32_t tok_off = 0;  // (token - block_start_row) * scale_aligned_page_bytes
            uint32_t block_idx_b = block_start_idx;
            uint32_t s = 0;
            uint32_t face_base_off = 0;
            for (uint32_t fr = 0; fr < FACE_ROWS; ++fr) {
                uint32_t col0_off = face_base_off;
                for (uint32_t r = 0; r < face_h; ++r) {
                    if (s < real_in_block) {
                        page[col0_off >> scale_elem_shift] =
                            scratch_mem[(tok_off >> scale_elem_shift) + scale_col_offset + block_idx_b];
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
        };
        if constexpr (scale_elem_bytes == 4) {
            build_bcast(CoreLocalMem<volatile uint32_t>(scratch), CoreLocalMem<volatile uint32_t>(page_ptr));
        } else {
            build_bcast(CoreLocalMem<volatile uint16_t>(scratch), CoreLocalMem<volatile uint16_t>(page_ptr));
        }
        cb_scale_bcast_obj.push_back(1);
        cb_input_e4m3_obj.push_back(tiles_per_block);
    }
}
