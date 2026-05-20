// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "llk_defs.h"
#include "ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reduce_rm_dataflow_common.hpp"

//
// Dense RM reduce reader (handles both W reduce and H reduce; branched on REDUCE_DIM).
//
// Stages packed-row pages for the compute kernel to tilize+reduce. Loop structure mirrors the chunked
// hierarchy of chunks-over-chunks-over-tiles for each reduce dim — see the per-branch comments below.
//
//   for h_chunks:                          // chunks of Ht
//     for w_chunks:                        // chunks of Wt
//       for h_tiles_in_chunk:              // one CB page per slab (mirrors pool window-load: reserve, fill, read,
//                                          //                      barrier, push)
//         for w_tiles_in_chunk (implicit): // covered by a single per-row noc_async_read inside
//                                          //                      rm_read_slab_into_page (contiguous over W in RM)
//
// W reduce path: each core owns a contiguous range of (NC × H) logical rows; loop order is h_chunks outer
// to match compute's `chunk_idx` reset rule.
//
// H reduce path: each core owns a contiguous range of (NC × Wt) output tile-columns; loop order is
// w_chunks outer (over the core's output tile range, possibly crossing NCs), inner h_chunks re-reads
// the same H rows for that NC's full source range. `chunk_idx` in compute resets per W chunk in this mode.
//
void kernel_main() {
    //
    // Runtime args. Slots are shared between paths; semantics differ:
    //   W reduce: (src_addr, num_rows_for_this_core, start_row_id)
    //   H reduce: (src_addr, num_output_tile_cols_for_this_core, start_output_tile_col_id)
    //
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t rt_count = get_arg_val<uint32_t>(1);
    const uint32_t rt_start = get_arg_val<uint32_t>(2);

    //
    // Compile-time args
    //
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(0);
    constexpr uint32_t W_logical = get_compile_time_arg_val(1);
    constexpr uint32_t elem_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t padding_identity_bits = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t wt_tiles_per_chunk = get_compile_time_arg_val(5);
    constexpr uint32_t rm_rows_per_tile = get_compile_time_arg_val(6);
    constexpr uint32_t ht_tiles_per_chunk = get_compile_time_arg_val(7);
    // H_logical is only consumed by the H reduce path (rows per NC slab in source). The W factory passes 0.
    constexpr uint32_t H_logical = get_compile_time_arg_val(8);
    constexpr auto tensor_args = TensorAccessorArgs<9>();

    static_assert(ht_tiles_per_chunk <= RM_MAX_HT_TILES_PER_CHUNK, "ht_tiles_per_chunk exceeds reader slab cap");

    //
    // CBs (shared between both paths)
    //
    constexpr uint32_t cb_id_scaler = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_rm = tt::CBIndex::c_24;
    constexpr uint32_t cb_id_clear_value = tt::CBIndex::c_4;
    constexpr uint32_t onepage = 1;

    experimental::CB cb_rm(cb_id_rm);
    experimental::CB cb_clear_value(cb_id_clear_value);
    Noc noc;

    //
    // Scaler tile (reduce_helpers_dataflow): pushed once, used by compute for every reduce call.
    //
    const float scaler_f = __builtin_bit_cast(float, scaler_bits);
    const uint32_t scaler_valid_for_reduce = []() -> uint32_t {
        if constexpr (REDUCE_OP == ckernel::PoolType::SUM) {
            return tt::constants::TILE_WIDTH;
        }
        return (W_logical < tt::constants::TILE_WIDTH) ? W_logical : tt::constants::TILE_WIDTH;
    }();
    dataflow_kernel_lib::prepare_reduce_scaler<cb_id_scaler, REDUCE_OP, REDUCE_DIM>(scaler_f, scaler_valid_for_reduce);

    //
    // Identity template (reused for clearing every staged page before real RM data is overlaid).
    //
    const uint32_t clear_template_bytes = get_tile_size(cb_id_clear_value);
    rm_fill_buffer_with_identity_pattern(
        cb_clear_value.get_write_ptr(), clear_template_bytes, elem_bytes, padding_identity_bits);
    cb_clear_value.push_back(onepage);
    const auto clear_template_src = experimental::local_addr(cb_clear_value.get_read_ptr(), noc.get_noc_id());

    //
    // Source accessor + cached page byte count.
    //
    const auto tensor_accessor = TensorAccessor(tensor_args, src_addr);
    const uint32_t page_bytes = get_local_cb_interface(cb_id_rm).fifo_page_size;
    const uint32_t valid_row_bytes = W_logical * elem_bytes;

    if constexpr (REDUCE_DIM == ckernel::ReduceDim::REDUCE_ROW) {
        //
        // === W reduce path ===
        //
        // Staging loop. State carried across H chunks:
        //   packed_row_base: index of the next source RM page to consume
        //   rows_remaining:  logical rows left in this core's shard
        //
        const uint32_t num_pages = rt_count;
        const uint32_t Ht_reader = (num_pages + rm_rows_per_tile - 1) / rm_rows_per_tile;
        uint32_t packed_row_base = rt_start;
        uint32_t rows_remaining = num_pages;

        for (uint32_t h_chunk_base = 0; h_chunk_base < Ht_reader; h_chunk_base += ht_tiles_per_chunk) {
            const uint32_t ht_in_chunk =
                (h_chunk_base + ht_tiles_per_chunk < Ht_reader) ? ht_tiles_per_chunk : (Ht_reader - h_chunk_base);

            // Layout the slabs that belong to this H chunk (which source pages each slab pulls + how many rows).
            // Done once per H chunk so the W chunk inner loop just indexes into it.
            RmSlabInfo slabs[RM_MAX_HT_TILES_PER_CHUNK];
            rm_precompute_slabs_for_h_chunk(slabs, ht_in_chunk, rm_rows_per_tile, packed_row_base, rows_remaining);

            for (uint32_t w_chunk_base = 0; w_chunk_base < Wt; w_chunk_base += wt_tiles_per_chunk) {
                const uint32_t wt_in_chunk =
                    (w_chunk_base + wt_tiles_per_chunk < Wt) ? wt_tiles_per_chunk : (Wt - w_chunk_base);
                const RmWChunkBytes w_range =
                    rm_compute_w_chunk_bytes(w_chunk_base, wt_in_chunk, valid_row_bytes, elem_bytes);

                // One staged page per slab — same shape as pool's window-load pattern in reader_pool_2d.cpp:
                //   reserve → fill identity → read window → barrier → push.
                for (uint32_t hti = 0; hti < ht_in_chunk; ++hti) {
                    cb_rm.reserve_back(onepage);

                    rm_fill_page_with_clear_template(noc, cb_rm, page_bytes, clear_template_src, clear_template_bytes);
                    rm_read_slab_into_page(noc, cb_rm, tensor_accessor, slabs[hti], w_range);
                    noc.async_read_barrier();
                    cb_rm.push_back(onepage);
                }
            }
        }
    } else {
        //
        // === H reduce path ===
        //
        // Each core handles a contiguous range of output tile-columns indexed by (nc * Wt + wt). For each
        // (nc, w_chunk) work unit we re-read all H_logical rows of that NC's source pages into chunked slab
        // pages, in the same (h_chunk_base, hti) order the compute kernel consumes them.
        //
        // chunk_idx in the compute kernel resets per w_chunk and advances per h_chunk_base — so within this
        // path the outer loop is w_chunks (over the core's output tile range) and the inner loop is h_chunks.
        //
        const uint32_t Ht_total = (H_logical + rm_rows_per_tile - 1) / rm_rows_per_tile;
        uint32_t outputs_remaining = rt_count;
        uint32_t current_nc = rt_start / Wt;
        uint32_t wt_in_nc = rt_start % Wt;

        while (outputs_remaining > 0) {
            const uint32_t wt_remaining_in_nc = Wt - wt_in_nc;
            uint32_t wt_in_chunk = wt_tiles_per_chunk;
            if (wt_in_chunk > wt_remaining_in_nc) {
                wt_in_chunk = wt_remaining_in_nc;
            }
            if (wt_in_chunk > outputs_remaining) {
                wt_in_chunk = outputs_remaining;
            }
            const RmWChunkBytes w_range = rm_compute_w_chunk_bytes(wt_in_nc, wt_in_chunk, valid_row_bytes, elem_bytes);

            // Source pages for this NC: H_logical contiguous pages starting at current_nc * H_logical.
            uint32_t packed_row_base = current_nc * H_logical;
            uint32_t rows_remaining = H_logical;

            for (uint32_t h_chunk_base = 0; h_chunk_base < Ht_total; h_chunk_base += ht_tiles_per_chunk) {
                const uint32_t ht_in_chunk =
                    (h_chunk_base + ht_tiles_per_chunk < Ht_total) ? ht_tiles_per_chunk : (Ht_total - h_chunk_base);

                RmSlabInfo slabs[RM_MAX_HT_TILES_PER_CHUNK];
                rm_precompute_slabs_for_h_chunk(slabs, ht_in_chunk, rm_rows_per_tile, packed_row_base, rows_remaining);

                for (uint32_t hti = 0; hti < ht_in_chunk; ++hti) {
                    cb_rm.reserve_back(onepage);

                    rm_fill_page_with_clear_template(noc, cb_rm, page_bytes, clear_template_src, clear_template_bytes);
                    rm_read_slab_into_page(noc, cb_rm, tensor_accessor, slabs[hti], w_range);
                    noc.async_read_barrier();
                    cb_rm.push_back(onepage);
                }
            }

            wt_in_nc += wt_in_chunk;
            outputs_remaining -= wt_in_chunk;
            if (wt_in_nc == Wt) {
                wt_in_nc = 0;
                ++current_nc;
            }
        }
    }
}
