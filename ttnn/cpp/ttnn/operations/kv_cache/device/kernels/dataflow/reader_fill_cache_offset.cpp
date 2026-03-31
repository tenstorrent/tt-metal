// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

inline uint32_t min(uint32_t a, uint32_t b) { return a < b ? a : b; }

// Copy rows between tiles in L1 using face-aware batched NOC reads.
// Rows within the same source and destination face pair are copied in a single NOC operation.
// Each tile has 4 faces of 16x16 elements:
//   Face 0: rows 0-15 cols 0-15,  Face 1: rows 0-15 cols 16-31
//   Face 2: rows 16-31 cols 0-15, Face 3: rows 16-31 cols 16-31
inline void copy_tile_rows(
    uint32_t dst_l1,
    uint32_t dst_row_start,
    uint32_t src_l1,
    uint32_t src_row_start,
    uint32_t num_rows,
    uint32_t face_row_bytes,
    uint32_t face_bytes) {
    uint32_t rows_done = 0;
    while (rows_done < num_rows) {
        uint32_t sr = src_row_start + rows_done;
        uint32_t dr = dst_row_start + rows_done;

        uint32_t src_remaining = (sr < 16) ? (16 - sr) : (32 - sr);
        uint32_t dst_remaining = (dr < 16) ? (16 - dr) : (32 - dr);
        uint32_t chunk = min(min(src_remaining, dst_remaining), num_rows - rows_done);

        uint32_t src_face0 = (sr < 16) ? 0 : 2;
        uint32_t src_rif = sr & 15;
        uint32_t dst_face0 = (dr < 16) ? 0 : 2;
        uint32_t dst_rif = dr & 15;

        uint32_t copy_bytes = chunk * face_row_bytes;

        uint64_t src_noc = get_noc_addr(src_l1 + src_face0 * face_bytes + src_rif * face_row_bytes);
        noc_async_read(src_noc, dst_l1 + dst_face0 * face_bytes + dst_rif * face_row_bytes, copy_bytes);

        src_noc = get_noc_addr(src_l1 + (src_face0 + 1) * face_bytes + src_rif * face_row_bytes);
        noc_async_read(src_noc, dst_l1 + (dst_face0 + 1) * face_bytes + dst_rif * face_row_bytes, copy_bytes);

        rows_done += chunk;
    }
}

void kernel_main() {
    const uint32_t cache_addr = get_arg_val<uint32_t>(0);
    const uint32_t input_addr = get_arg_val<uint32_t>(1);
    const uint32_t Wt = get_arg_val<uint32_t>(2);
    const uint32_t sub_offset = get_arg_val<uint32_t>(3);
    const uint32_t num_output_tiles_per_head = get_arg_val<uint32_t>(4);
    const uint32_t input_Ht = get_arg_val<uint32_t>(5);
    const uint32_t cache_batch_start = get_arg_val<uint32_t>(6);
    const uint32_t cache_HtWt = get_arg_val<uint32_t>(7);
    const uint32_t input_HtWt = get_arg_val<uint32_t>(8);
    const uint32_t num_blocks = get_arg_val<uint32_t>(9);
    const uint32_t block_start = get_arg_val<uint32_t>(10);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t face_row_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t face_bytes = get_compile_time_arg_val(3);
    constexpr auto cache_ta_args = TensorAccessorArgs<4>();
    constexpr auto input_ta_args = TensorAccessorArgs<cache_ta_args.next_compile_time_args_offset()>();

    const uint32_t tile_bytes = get_tile_size(cb_out);
    const auto cache_s = TensorAccessor(cache_ta_args, cache_addr, tile_bytes);
    const auto input_s = TensorAccessor(input_ta_args, input_addr, tile_bytes);

    // Reserve scratch space once (never pushed, reused each iteration)
    cb_reserve_back(cb_scratch, 2 * Wt);
    const uint32_t scratch_base = get_write_ptr(cb_scratch);

    for (uint32_t b = 0; b < num_blocks; b++) {
        const uint32_t global_block = block_start + b;
        const uint32_t head = global_block / num_output_tiles_per_head;
        const uint32_t t = global_block % num_output_tiles_per_head;
        const bool is_first = (t == 0);
        const bool is_last = (t == num_output_tiles_per_head - 1);

        // Compute cache tile start ID for this output tile
        const uint32_t cache_tile_id = cache_batch_start + head * cache_HtWt + t * Wt;

        // Read cache tile into output CB
        cb_reserve_back(cb_out, Wt);
        uint32_t out_addr = get_write_ptr(cb_out);
        for (uint32_t w = 0; w < Wt; w++) {
            noc_async_read_tile(cache_tile_id + w, cache_s, out_addr + w * tile_bytes);
        }
        noc_async_read_barrier();

        // Determine what to copy for this output tile
        uint32_t cache_row_dst;        // destination row in cache tile
        uint32_t num_rows_part_a;      // rows from first input tile
        uint32_t src_row_a;            // starting row in first input tile
        uint32_t input_tile_row_a;     // which input tile row (along seq_len)
        uint32_t num_rows_part_b = 0;  // rows from second input tile
        uint32_t input_tile_row_b = 0;

        if (is_first) {
            cache_row_dst = sub_offset;
            num_rows_part_a = 32 - sub_offset;
            src_row_a = 0;
            input_tile_row_a = 0;
        } else if (is_last) {
            cache_row_dst = 0;
            num_rows_part_a = sub_offset;
            src_row_a = 32 - sub_offset;
            input_tile_row_a = input_Ht - 1;
        } else {
            // Middle tile: rows from two input tiles
            cache_row_dst = 0;
            num_rows_part_a = sub_offset;
            src_row_a = 32 - sub_offset;
            input_tile_row_a = t - 1;
            num_rows_part_b = 32 - sub_offset;
            input_tile_row_b = t;
        }

        // Read input tile(s) into scratch
        const uint32_t input_base_a = head * input_HtWt + input_tile_row_a * Wt;
        for (uint32_t w = 0; w < Wt; w++) {
            noc_async_read_tile(input_base_a + w, input_s, scratch_base + w * tile_bytes);
        }
        if (num_rows_part_b > 0) {
            const uint32_t input_base_b = head * input_HtWt + input_tile_row_b * Wt;
            for (uint32_t w = 0; w < Wt; w++) {
                noc_async_read_tile(input_base_b + w, input_s, scratch_base + (Wt + w) * tile_bytes);
            }
        }
        noc_async_read_barrier();

        // Merge rows into output tile using face-aware copies
        for (uint32_t w = 0; w < Wt; w++) {
            const uint32_t out_tile = out_addr + w * tile_bytes;
            const uint32_t scratch_tile_a = scratch_base + w * tile_bytes;

            copy_tile_rows(
                out_tile, cache_row_dst, scratch_tile_a, src_row_a, num_rows_part_a, face_row_bytes, face_bytes);

            if (num_rows_part_b > 0) {
                const uint32_t scratch_tile_b = scratch_base + (Wt + w) * tile_bytes;
                copy_tile_rows(
                    out_tile,
                    cache_row_dst + num_rows_part_a,
                    scratch_tile_b,
                    0,
                    num_rows_part_b,
                    face_row_bytes,
                    face_bytes);
            }
        }
        noc_async_read_barrier();

        cb_push_back(cb_out, Wt);
    }
}
