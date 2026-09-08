// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "llk_defs.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reduce_rm_dataflow_common.hpp"

//
// Dense RM reduce reader (handles both W reduce and H reduce; branched on REDUCE_DIM).
//
// Buffer layout: one dfb::rm entry = one chunk-wide RM row (chunk_row_bytes). Per (h_chunk, w_chunk,
// hti) tuple the reader pushes rm_rows_per_tile (= TILE_HEIGHT) entries — real rows overlaid on the
// identity-pad template, plus identity for the partial-last-h-tile case (H_logical %
// rm_rows_per_tile != 0). This matches compute_kernel_lib::tilize's asymmetric mode: each
// tile-block consumes TILE_HEIGHT input entries.
//
// Loop shape, both paths:
//   for h_chunks × for w_chunks × for hti_in_chunk:
//     reserve TILE_HEIGHT entries → identity-fill (slab × entry_bytes)
//                                → async_read real rows
//                                → barrier + push TILE_HEIGHT
//
// W reduce: outer iteration is h_chunks over this core's row range (rt_count rows starting at
// rt_start). w_chunks inner — chunk_idx in compute resets per h_chunk.
//
// H reduce: outer iteration is the core's output-tile range (rt_count outputs starting at
// rt_start, decomposed into (nc, wt_in_nc)). h_chunks inner. wt_tiles_per_chunk is mandated 1 by
// the H factory, so each output tile is one work unit.
//
template <ckernel::ReduceDim DIM>
void reduce_rm_reader() {
    // Runtime args (W: rt_count = num rows, rt_start = first row;
    //               H: rt_count = num output tiles, rt_start = first output tile).
    const uint32_t rt_count = get_arg(args::rt_count);
    const uint32_t rt_start = get_arg(args::rt_start);

    // Compile-time args. Both paths receive the whole set; each branch reads the names it needs.
    constexpr auto scaler_bits = get_arg(args::scaler_bits);
    constexpr auto W_logical = get_arg(args::W_logical);
    constexpr auto elem_bytes = get_arg(args::elem_bytes);
    constexpr auto padding_identity_bits = get_arg(args::padding_identity_bits);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto wt_tiles_per_chunk = get_arg(args::wt_tiles_per_chunk);
    constexpr auto rm_rows_per_tile = get_arg(args::rm_rows_per_tile);
    constexpr auto ht_tiles_per_chunk = get_arg(args::ht_tiles_per_chunk);

    constexpr uint32_t onepage = 1;

    // dfb::rm stages row-major rows for compute to tilize; dfb::clear_value is this kernel's private
    // identity template — it both fills it and re-reads it as a NoC source, so it is the buffer's
    // only toucher.
    DataflowBuffer dfb_rm(dfb::rm);
    DataflowBuffer dfb_clear_value(dfb::clear_value);
    Noc noc;

    // Scaler entry — pushed once, used by every compute reduce() call.
    const float scaler_f = __builtin_bit_cast(float, scaler_bits);
    const uint32_t scaler_valid_for_reduce = []() -> uint32_t {
        if constexpr (REDUCE_OP == ckernel::PoolType::SUM) {
            return tt::constants::TILE_WIDTH;
        }
        return (W_logical < tt::constants::TILE_WIDTH) ? W_logical : tt::constants::TILE_WIDTH;
    }();
    dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, REDUCE_OP, DIM>(scaler_f, scaler_valid_for_reduce);

    // Identity template — pre-built once, reused as the pad source for every staged slab.
    const uint32_t clear_template_bytes = dfb_clear_value.get_tile_size();
    rm_fill_buffer_with_identity_pattern(
        dfb_clear_value.get_write_ptr(), clear_template_bytes, elem_bytes, padding_identity_bits);
    dfb_clear_value.push_back(onepage);
    const auto clear_template_src = experimental::local_addr(dfb_clear_value.get_read_ptr(), noc.get_noc_id());

    const auto tensor_accessor = TensorAccessor(tensor::src);
    const uint32_t page_bytes = dfb_rm.get_entry_size();
    const uint32_t valid_row_bytes = W_logical * elem_bytes;

    // Stage one h-tile slab (TILE_HEIGHT entries) for the given W chunk:
    //   reserve → bulk identity-fill the whole slab → async_read real_rows real rows → barrier → push.
    // Padded rows past `real_rows` and padded columns past `valid_bytes` retain the identity (0 for
    // SUM) from the fill, contributing nothing to the running sum.
    auto stage_slab = [&](uint32_t slab_first_global_row, uint32_t real_rows, const RmWChunkBytes& w_range) {
        dfb_rm.reserve_back(rm_rows_per_tile);
        rm_fill_page_with_clear_template(
            noc, dfb_rm, rm_rows_per_tile * page_bytes, clear_template_src, clear_template_bytes);
        if (w_range.valid_bytes > 0) {
            for (uint32_t r = 0; r < real_rows; ++r) {
                noc.async_read(
                    tensor_accessor,
                    dfb_rm,
                    w_range.valid_bytes,
                    {.page_id = slab_first_global_row + r, .offset_bytes = w_range.chunk_start_bytes},
                    {.offset_bytes = r * page_bytes});
            }
        }
        noc.async_read_barrier();
        dfb_rm.push_back(rm_rows_per_tile);
    };

    if constexpr (DIM == ckernel::ReduceDim::REDUCE_ROW) {
        // === W reduce ===
        const uint32_t num_logical_rows = rt_count;
        const uint32_t Ht_reader = (num_logical_rows + rm_rows_per_tile - 1) / rm_rows_per_tile;

        for (uint32_t h_chunk_base = 0; h_chunk_base < Ht_reader; h_chunk_base += ht_tiles_per_chunk) {
            const uint32_t ht_in_chunk =
                (h_chunk_base + ht_tiles_per_chunk < Ht_reader) ? ht_tiles_per_chunk : (Ht_reader - h_chunk_base);

            for (uint32_t w_chunk_base = 0; w_chunk_base < Wt; w_chunk_base += wt_tiles_per_chunk) {
                const RmWChunkBytes w_range =
                    rm_compute_w_chunk_bytes(w_chunk_base, wt_tiles_per_chunk, valid_row_bytes, elem_bytes);

                for (uint32_t hti = 0; hti < ht_in_chunk; ++hti) {
                    const uint32_t slab_base_row_local = (h_chunk_base + hti) * rm_rows_per_tile;
                    const uint32_t slab_rows_avail = num_logical_rows - slab_base_row_local;
                    const uint32_t real_rows =
                        (slab_rows_avail < rm_rows_per_tile) ? slab_rows_avail : rm_rows_per_tile;
                    stage_slab(rt_start + slab_base_row_local, real_rows, w_range);
                }
            }
        }
    } else {
        // === H reduce ===
        // H_logical / split geometry are only meaningful on the H path (the W factory emits them
        // too, as 1 / Ht_rm, so the names resolve in either build).
        constexpr auto H_logical = get_arg(args::H_logical);
        constexpr auto num_h_slices = get_arg(args::num_h_slices);
        // Tiles per work unit == the compute kernel's Ht loop bound; == Ht_rm when num_h_slices == 1.
        constexpr auto slice_Ht = get_arg(args::slice_Ht);

        // Each owned output tile is one work unit (wt_tiles_per_chunk == 1). Decompose its global id
        // into (nc, slice, wt_in_nc) and read only this slice's contiguous H slice. Tiles that run
        // past H_logical (last slice's overhang) stage as all-identity (real_rows == 0) → contribute 0.
        for (uint32_t out_idx = 0; out_idx < rt_count; ++out_idx) {
            const uint32_t global_tile_id = rt_start + out_idx;
            const uint32_t wt_in_nc = global_tile_id % Wt;
            // Peeling off the W-column leaves the flattened (nc, slice) index == nc*num_h_slices + slice.
            const uint32_t nc_slice_idx = global_tile_id / Wt;
            const uint32_t h_slice_idx = nc_slice_idx % num_h_slices;
            const uint32_t nc = nc_slice_idx / num_h_slices;
            const RmWChunkBytes w_range =
                rm_compute_w_chunk_bytes(wt_in_nc, wt_tiles_per_chunk, valid_row_bytes, elem_bytes);
            const uint32_t nc_base_page = nc * H_logical;
            const uint32_t slice_first_tile = h_slice_idx * slice_Ht;

            for (uint32_t slice_tile_base = 0; slice_tile_base < slice_Ht; slice_tile_base += ht_tiles_per_chunk) {
                const uint32_t ht_in_chunk = (slice_tile_base + ht_tiles_per_chunk < slice_Ht)
                                                 ? ht_tiles_per_chunk
                                                 : (slice_Ht - slice_tile_base);

                for (uint32_t hti = 0; hti < ht_in_chunk; ++hti) {
                    const uint32_t slab_base_row_in_nc = (slice_first_tile + slice_tile_base + hti) * rm_rows_per_tile;
                    const uint32_t slab_rows_avail =
                        (slab_base_row_in_nc < H_logical) ? (H_logical - slab_base_row_in_nc) : 0;
                    const uint32_t real_rows =
                        (slab_rows_avail < rm_rows_per_tile) ? slab_rows_avail : rm_rows_per_tile;
                    stage_slab(nc_base_page + slab_base_row_in_nc, real_rows, w_range);
                }
            }
        }
    }
}

void kernel_main() { reduce_rm_reader<REDUCE_DIM>(); }
