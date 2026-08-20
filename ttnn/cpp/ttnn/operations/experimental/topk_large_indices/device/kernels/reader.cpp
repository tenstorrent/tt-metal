// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Chunked row reader, both parallelization modes (role picked by the factory):
//   * default (row-parallel): each core reads full rows.
//   * TOPK_TREE (column-parallel): each core reads only its contiguous slice
//     of every row, offset by the extra slice_offset_bytes runtime arg. An
//     empty slice (num_chunks == 0, valid_length cut) reads nothing.

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

#ifdef THRESHOLD_FILTER
#include "topk_large_indices_threshold_filter.hpp"
#ifndef TF_MIN_CHUNKS
#define TF_MIN_CHUNKS 8
#endif
#endif

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);
    const uint32_t num_chunks = get_arg_val<uint32_t>(3);
    const uint32_t tail_chunk_bytes = get_arg_val<uint32_t>(4);
    const uint32_t input_page_bytes = get_arg_val<uint32_t>(5);
#ifdef TOPK_TREE
    // Byte offset of this core's slice within each row.
    const uint32_t slice_offset_bytes = get_arg_val<uint32_t>(6);
#else
    constexpr uint32_t slice_offset_bytes = 0;
#endif

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_chunk = get_compile_time_arg_val(3);
    constexpr auto input_args = TensorAccessorArgs<4>();
#ifdef THRESHOLD_FILTER
    constexpr uint32_t cb_tf_ctrl_id = get_compile_time_arg_val(input_args.next_compile_time_args_offset());
    const uint32_t tf_ctrl_base = get_write_ptr(cb_tf_ctrl_id);
    const uint32_t tf_epoch = get_arg_val<uint32_t>(6);
#endif

    const auto input = TensorAccessor(input_args, src_addr, input_page_bytes);
    CircularBuffer input_cb(cb_in);
    Noc noc;

#ifdef THRESHOLD_FILTER
    // Threshold-filter launches gate per launch on the runtime chunk count;
    // narrow valid prefixes run the classic body everywhere (all RISCs derive
    // the same predicate from the same runtime arg).
    const bool tf_active = num_chunks >= TF_MIN_CHUNKS;
#endif

    for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
        const uint32_t row = start_row + local_row;

#ifdef THRESHOLD_FILTER
        if (tf_active) {
            // SAMPLE chunk: 64 x 64B strided blocks spanning the valid prefix.
            const uint64_t valid_bytes =
                static_cast<uint64_t>(num_chunks - 1) * chunk_bytes + tail_chunk_bytes;
            input_cb.reserve_back(tiles_per_chunk);
            for (uint32_t b = 0; b < 64; ++b) {
                const uint32_t src_off =
                    static_cast<uint32_t>((static_cast<uint64_t>(b) * (valid_bytes - 64) / 63)) & ~63u;
                noc.async_read(
                    input,
                    input_cb,
                    64,
                    {.page_id = row, .offset_bytes = src_off},
                    {.offset_bytes = b * 64});
            }
            noc.async_read_barrier();
            input_cb.push_back(tiles_per_chunk);
        }
#endif

        for (uint32_t pass = 0; pass < 2; ++pass) {
            for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
                const uint32_t active_chunk_bytes = (chunk + 1 == num_chunks) ? tail_chunk_bytes : chunk_bytes;
                input_cb.reserve_back(tiles_per_chunk);
                for (uint32_t tile = 0; tile < tiles_per_chunk; ++tile) {
                    const uint32_t tile_offset = tile * tile_bytes;
                    const uint32_t read_bytes =
                        tile_offset < active_chunk_bytes
                            ? (active_chunk_bytes - tile_offset < tile_bytes ? active_chunk_bytes - tile_offset
                                                                             : tile_bytes)
                            : 0;
                    if (read_bytes != 0) {
                        noc.async_read(
                            input,
                            input_cb,
                            read_bytes,
                            {.page_id = row, .offset_bytes = slice_offset_bytes + chunk * chunk_bytes + tile_offset},
                            {.offset_bytes = tile_offset});
                    }
                }
                noc.async_read_barrier();
                input_cb.push_back(tiles_per_chunk);
            }
#ifdef THRESHOLD_FILTER
            if (tf_active && pass == 0) {
                // Wait for the row decision from the parser; re-stream on RETRY.
                namespace tf = topk_large_indices_threshold_filter;
                volatile tt_l1_ptr uint32_t* ctrl =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tf_ctrl_base);
                const uint32_t want_ok = tf::decision_value(local_row, tf::kDecOk);
                uint32_t v;
                do {
                    v = tf::ctrl_value(tf_epoch, ctrl[0]);
                } while (v < want_ok);
                if (v == want_ok) {
                    break;  // decision OK: no second pass
                }
                // RETRY: fall through into pass 1 (classic re-stream)
            } else {
                break;
            }
#else
            break;
#endif
        }
    }
}
