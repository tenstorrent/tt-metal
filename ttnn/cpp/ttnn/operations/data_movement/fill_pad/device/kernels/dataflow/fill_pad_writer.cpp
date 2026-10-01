// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Phase 1 – Mask generation (before the main loop):
 *   Builds a "right mask" tile (if the right mask is bound) and a "bottom mask"
 *   tile (if the bottom mask is bound) in face layout and pushes them to their
 *   respective dataflow buffers. The compute kernel holds these tiles
 *   persistently (never pops them) and uses them with where_tile to apply the
 *   fill.
 *
 *   Mask encoding (same DataFormat as the input tensor):
 *     Float types  : 1.0 at padding positions, 0.0 elsewhere.
 *     Integer types: integer 1 at padding positions, 0 elsewhere.
 *
 * Phase 2 – Write-back loop:
 *   Reads masked tiles produced by the compute kernel from dfb::data_out and
 *   writes them back to DRAM (or sharded L1). No masking is done here.
 *
 *   Three phase loops mirror fill_pad_reader.cpp's right / bottom / corner
 *   phases, using the same per-phase (start, num) RT args so that reader,
 *   compute and writer process tiles in lock-step.
 *
 * Metal 2.0 named resources:
 *   CTAs:  W_tiles, H_tiles; W_mod32 (only when the right mask is bound),
 *          H_mod32 (only when the bottom mask is bound).
 *   Defines: HAS_RIGHT_PAD / HAS_BOTTOM_PAD gate the
 *            conditionally-bound right / bottom mask DFBs (promoted from the
 *            legacy has_right_pad / has_bottom_pad compile-time args).
 *   DFBs:  dfb::right_mask (PRODUCER, conditional), dfb::bot_mask (PRODUCER,
 *          conditional), dfb::data_out (this writer is its CONSUMER).
 *   tensor: tensor::dst (in-place tensor; base address auto-injected).
 *   RTAs:  start_right, num_right, start_bottom, num_bottom, start_corner, num_corner.
 */

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "fill_pad_dataflow_common.hpp"

void kernel_main() {
    constexpr auto W_tiles = get_arg(args::W_tiles);
    constexpr auto H_tiles = get_arg(args::H_tiles);

    // has_right_pad / has_bottom_pad are carried as preprocessor defines (not CTAs),
    // because they gate references to the conditionally-bound right / bottom mask DFBs.
#ifdef HAS_RIGHT_PAD
    constexpr std::uint32_t has_right_pad = 1;
#else
    constexpr std::uint32_t has_right_pad = 0;
#endif
#ifdef HAS_BOTTOM_PAD
    constexpr std::uint32_t has_bottom_pad = 1;
#else
    constexpr std::uint32_t has_bottom_pad = 0;
#endif

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

    // Tensor base address and layout metadata are supplied by the tensor::dst binding.
    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
#ifdef HAS_RIGHT_PAD
    DataflowBuffer dfb_right_mask(dfb::right_mask);
#endif
#ifdef HAS_BOTTOM_PAD
    DataflowBuffer dfb_bot_mask(dfb::bot_mask);
#endif
    DataflowBuffer dfb_data_out(dfb::data_out);
    const std::uint32_t tile_bytes = dfb_data_out.get_entry_size();

    // ---- Phase 1: generate and push mask tile(s) ----
#if defined(HAS_RIGHT_PAD) || defined(HAS_BOTTOM_PAD)
    using mask_t = MASK_ELEM_UINT;
    constexpr std::uint32_t TILE = 32;
#endif
#ifdef HAS_RIGHT_PAD
    constexpr auto W_mod32 = get_arg(args::W_mod32);
    push_right_mask_tile<mask_t, W_mod32, TILE>(dfb_right_mask, static_cast<mask_t>(MASK_VALUE));
#endif
#ifdef HAS_BOTTOM_PAD
    constexpr auto H_mod32 = get_arg(args::H_mod32);
    push_bottom_mask_tile<mask_t, H_mod32, TILE>(dfb_bot_mask, static_cast<mask_t>(MASK_VALUE));
#endif

    // ---- Phase 2: write-back loop ----
    // Tiles arrive in the same order as the reader pushes them (right, bottom, corner).

    // Right phase. Maintain (slice, row) incrementally instead of dividing every iteration
    // — RV32IM division is slow. Startup division runs at most once per kernel invocation.
    if constexpr (has_right_pad) {
        std::uint32_t slice = num_right ? start_right / right_slice_stride : 0u;
        std::uint32_t row = num_right ? start_right - slice * right_slice_stride : 0u;
        for (std::uint32_t i = 0; i < num_right; ++i) {
            const std::uint32_t tile_id = slice * H_tiles * W_tiles + row * W_tiles + (W_tiles - 1u);
            dfb_data_out.wait_front(1);
            noc.async_write(dfb_data_out, s, tile_bytes, {.offset_bytes = 0}, {.page_id = tile_id});
            noc.async_writes_flushed();
            dfb_data_out.pop_front(1);
            ++row;
            if (row == right_slice_stride) {
                row = 0;
                ++slice;
            }
        }
    }

    // Bottom phase. Same incremental pattern as the right phase.
    if constexpr (has_bottom_pad) {
        std::uint32_t slice = num_bottom ? start_bottom / bottom_slice_stride : 0u;
        std::uint32_t col = num_bottom ? start_bottom - slice * bottom_slice_stride : 0u;
        for (std::uint32_t j = 0; j < num_bottom; ++j) {
            const std::uint32_t tile_id = slice * H_tiles * W_tiles + (H_tiles - 1u) * W_tiles + col;
            dfb_data_out.wait_front(1);
            noc.async_write(dfb_data_out, s, tile_bytes, {.offset_bytes = 0}, {.page_id = tile_id});
            noc.async_writes_flushed();
            dfb_data_out.pop_front(1);
            ++col;
            if (col == bottom_slice_stride) {
                col = 0;
                ++slice;
            }
        }
    }

    // Corner phase
    if constexpr (has_right_pad && has_bottom_pad) {
        for (std::uint32_t k = 0; k < num_corner; ++k) {
            const std::uint32_t slice = start_corner + k;
            const std::uint32_t tile_id = slice * H_tiles * W_tiles + (H_tiles - 1u) * W_tiles + (W_tiles - 1u);
            dfb_data_out.wait_front(1);
            noc.async_write(dfb_data_out, s, tile_bytes, {.offset_bytes = 0}, {.page_id = tile_id});
            noc.async_writes_flushed();
            dfb_data_out.pop_front(1);
        }
    }

    noc.async_write_barrier();
}
