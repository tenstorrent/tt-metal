// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Reads border tiles from the tensor (DRAM interleaved or sharded) into the
 * double-buffered dfb_data_in. The writer (BRISC) applies the padding-fill mask
 * in L1 and writes each tile back.
 *
 * Unified border-tile split. The host enumerates border tiles across all
 * slices into three contiguous blocks (right / bottom / corner) and gives
 * this core a per-phase (start, num) range inside each block:
 *
 *   Right phase  (num_right > 0 only when has_right_pad):
 *     for each i in [start_right, start_right + num_right):
 *       slice = i / right_slice_stride;  row = i % right_slice_stride
 *       tile_id = slice * H_tiles * W_tiles + row * W_tiles + (W_tiles - 1)
 *     where right_slice_stride = (H_tiles - 1) if has_bottom_pad else H_tiles.
 *
 *   Bottom phase (num_bottom > 0 only when has_bottom_pad):
 *     for each j in [start_bottom, start_bottom + num_bottom):
 *       slice = j / bottom_slice_stride;  col = j % bottom_slice_stride
 *       tile_id = slice * H_tiles * W_tiles + (H_tiles - 1) * W_tiles + col
 *     where bottom_slice_stride = (W_tiles - 1) if has_right_pad else W_tiles.
 *
 *   Corner phase (num_corner > 0 only when has_right_pad && has_bottom_pad):
 *     for each k in [start_corner, start_corner + num_corner):
 *       slice = k
 *       tile_id = slice * H_tiles * W_tiles + (H_tiles - 1) * W_tiles + (W_tiles - 1)
 *
 * Tile ordering across the three phases must match fill_pad_writer.cpp and
 * fill_pad_compute.cpp exactly (DFBs are FIFO).
 *
 * Metal 2.0 named resources:
 *   CTAs:  W_tiles, H_tiles, has_right_pad, has_bottom_pad, elem_size (elem_size unused).
 *   DFB:   dfb::data_in  (this reader is its PRODUCER).
 *   tensor: tensor::src (in-place tensor; base address auto-injected by the binding).
 *   RTAs:  start_right, num_right, start_bottom, num_bottom, start_corner, num_corner.
 */

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto W_tiles = get_arg(args::W_tiles);
    constexpr auto H_tiles = get_arg(args::H_tiles);
    constexpr auto has_right_pad = get_arg(args::has_right_pad);
    constexpr auto has_bottom_pad = get_arg(args::has_bottom_pad);
    [[maybe_unused]] constexpr auto elem_size = get_arg(args::elem_size);

    // Per-phase slice strides (meaningful only when the corresponding phase is active).
    // Clamped to >= 1 so the compiler does not see a constexpr divide-by-zero in
    // the dead-code branches (when H_tiles==1 or W_tiles==1 the host sets the
    // matching num_* to 0 and the loop below never executes).
    constexpr std::uint32_t right_slice_stride =
        has_right_pad ? (has_bottom_pad ? ((H_tiles > 1u) ? (H_tiles - 1u) : 1u) : H_tiles) : 1u;
    constexpr std::uint32_t bottom_slice_stride =
        has_bottom_pad ? (has_right_pad ? ((W_tiles > 1u) ? (W_tiles - 1u) : 1u) : W_tiles) : 1u;

    const auto start_right = get_arg(args::start_right);
    const auto num_right = get_arg(args::num_right);
    const auto start_bottom = get_arg(args::start_bottom);
    const auto num_bottom = get_arg(args::num_bottom);
    const auto start_corner = get_arg(args::start_corner);
    const auto num_corner = get_arg(args::num_corner);

    // Tensor base address and layout metadata are supplied by the tensor::src binding.
    const auto s = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer dfb_tile_in(dfb::data_in);
    const std::uint32_t tile_bytes = dfb_tile_in.get_entry_size();

    // ---- Right phase ----
    // Maintain (slice, row) incrementally instead of dividing every iteration
    // — RV32IM division is slow. Startup division runs at most once.
    if constexpr (has_right_pad) {
        std::uint32_t slice = num_right ? start_right / right_slice_stride : 0u;
        std::uint32_t row = num_right ? start_right - slice * right_slice_stride : 0u;
        for (std::uint32_t i = 0; i < num_right; ++i) {
            const std::uint32_t tile_id = slice * H_tiles * W_tiles + row * W_tiles + (W_tiles - 1u);
            dfb_tile_in.reserve_back(1);
            noc.async_read(s, dfb_tile_in, tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_tile_in.push_back(1);
            ++row;
            if (row == right_slice_stride) {
                row = 0;
                ++slice;
            }
        }
    }

    // ---- Bottom phase ----
    if constexpr (has_bottom_pad) {
        std::uint32_t slice = num_bottom ? start_bottom / bottom_slice_stride : 0u;
        std::uint32_t col = num_bottom ? start_bottom - slice * bottom_slice_stride : 0u;
        for (std::uint32_t j = 0; j < num_bottom; ++j) {
            const std::uint32_t tile_id = slice * H_tiles * W_tiles + (H_tiles - 1u) * W_tiles + col;
            dfb_tile_in.reserve_back(1);
            noc.async_read(s, dfb_tile_in, tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_tile_in.push_back(1);
            ++col;
            if (col == bottom_slice_stride) {
                col = 0;
                ++slice;
            }
        }
    }

    // ---- Corner phase ----
    if constexpr (has_right_pad && has_bottom_pad) {
        for (std::uint32_t k = 0; k < num_corner; ++k) {
            const std::uint32_t slice = start_corner + k;
            const std::uint32_t tile_id = slice * H_tiles * W_tiles + (H_tiles - 1u) * W_tiles + (W_tiles - 1u);
            dfb_tile_in.reserve_back(1);
            noc.async_read(s, dfb_tile_in, tile_bytes, {.page_id = tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_tile_in.push_back(1);
        }
    }
}
