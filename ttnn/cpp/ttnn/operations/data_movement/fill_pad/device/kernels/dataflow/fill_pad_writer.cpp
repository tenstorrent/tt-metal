// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/**
 * Phase 1 – Mask generation (before the main loop):
 *   Builds a "right mask" tile (if HAS_RIGHT_PAD) and a "bottom mask" tile
 *   (if HAS_BOTTOM_PAD) in face layout and pushes them to their respective
 *   dataflow buffers. The compute kernel holds these tiles persistently
 *   (never pops them) and uses them with where_tile to apply the fill.
 *
 *   Mask encoding (same DataFormat as the input tensor):
 *     Float types  : 1.0 at padding positions, 0.0 elsewhere.
 *     Integer types: integer 1 at padding positions, 0 elsewhere.
 *
 * Phase 2 – Write-back loop:
 *   Reads masked tiles produced by the compute kernel from dfb::data_out and writes
 *   them back to DRAM (or sharded L1). No masking is done here.
 *
 *   Three phase loops mirror fill_pad_reader.cpp's right / bottom / corner
 *   phases, using the same per-phase (start, num) RT args so that reader,
 *   compute and writer process tiles in lock-step.
 *
 * The right / bottom mask pads are conditionally-bound DFBs: HAS_RIGHT_PAD /
 * HAS_BOTTOM_PAD are #defines (the host binds dfb::right_mask / dfb::bot_mask only
 * under their respective pad config), so every reference is #ifdef-gated.
 *
 * Named compile-time args: W_tiles, H_tiles, W_mod32, H_mod32 (N_slices also declared, unused).
 * Named runtime args: start_right, num_right, start_bottom, num_bottom, start_corner, num_corner.
 * Resource bindings: dfb::data_out (consumed), dfb::right_mask / dfb::bot_mask (produced,
 *   conditional), tensor::dst (the input tensor, written via TensorAccessor).
 */

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "fill_pad_dataflow_common.hpp"

void kernel_main() {
    constexpr auto W_tiles = get_arg(args::W_tiles);
    constexpr auto H_tiles = get_arg(args::H_tiles);
    // Mask thresholds are read only where their mask is bound (avoids an unused-arg warning
    // in single-pad builds; the arg is still declared on the host for both).
#ifdef HAS_RIGHT_PAD
    constexpr auto W_mod32 = get_arg(args::W_mod32);
#endif
#ifdef HAS_BOTTOM_PAD
    constexpr auto H_mod32 = get_arg(args::H_mod32);
#endif

    Noc noc;
    DataflowBuffer dfb_data_out(dfb::data_out);
#ifdef HAS_RIGHT_PAD
    DataflowBuffer dfb_right_mask(dfb::right_mask);
#endif
#ifdef HAS_BOTTOM_PAD
    DataflowBuffer dfb_bot_mask(dfb::bot_mask);
#endif

    const uint32_t tile_bytes = dfb_data_out.get_tile_size();

    // Base address and layout metadata are injected by the TensorBinding.
    const auto s = TensorAccessor(tensor::dst);

    // ---- Phase 1: generate and push mask tile(s) ----
    using mask_t = MASK_ELEM_UINT;
    constexpr uint32_t TILE = 32;
#ifdef HAS_RIGHT_PAD
    push_right_mask_tile<mask_t, W_mod32, TILE>(dfb_right_mask, static_cast<mask_t>(MASK_VALUE));
#endif
#ifdef HAS_BOTTOM_PAD
    push_bottom_mask_tile<mask_t, H_mod32, TILE>(dfb_bot_mask, static_cast<mask_t>(MASK_VALUE));
#endif

    // ---- Phase 2: write-back loop ----
    // Tiles arrive in the same order as the reader pushes them (right, bottom, corner).

    // Right phase. Maintain (slice, row) incrementally instead of dividing every iteration
    // — RV32IM division is slow. Startup division runs at most once per kernel invocation.
#ifdef HAS_RIGHT_PAD
    {
        const uint32_t start_right = get_arg(args::start_right);
        const uint32_t num_right = get_arg(args::num_right);
        // right_slice_stride = (H_tiles - 1) if HAS_BOTTOM_PAD else H_tiles.
        // Clamped to >= 1 so the compiler never sees a constexpr divide-by-zero.
#ifdef HAS_BOTTOM_PAD
        constexpr uint32_t right_slice_stride = (H_tiles > 1u) ? (H_tiles - 1u) : 1u;
#else
        constexpr uint32_t right_slice_stride = H_tiles;
#endif
        uint32_t slice = num_right ? start_right / right_slice_stride : 0u;
        uint32_t row = num_right ? start_right - slice * right_slice_stride : 0u;
        for (uint32_t i = 0; i < num_right; ++i) {
            const uint32_t tile_id = slice * H_tiles * W_tiles + row * W_tiles + (W_tiles - 1u);
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
#endif

    // Bottom phase. Same incremental pattern as the right phase.
#ifdef HAS_BOTTOM_PAD
    {
        const uint32_t start_bottom = get_arg(args::start_bottom);
        const uint32_t num_bottom = get_arg(args::num_bottom);
        // bottom_slice_stride = (W_tiles - 1) if HAS_RIGHT_PAD else W_tiles.
#ifdef HAS_RIGHT_PAD
        constexpr uint32_t bottom_slice_stride = (W_tiles > 1u) ? (W_tiles - 1u) : 1u;
#else
        constexpr uint32_t bottom_slice_stride = W_tiles;
#endif
        uint32_t slice = num_bottom ? start_bottom / bottom_slice_stride : 0u;
        uint32_t col = num_bottom ? start_bottom - slice * bottom_slice_stride : 0u;
        for (uint32_t j = 0; j < num_bottom; ++j) {
            const uint32_t tile_id = slice * H_tiles * W_tiles + (H_tiles - 1u) * W_tiles + col;
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
#endif

    // Corner phase
#if defined(HAS_RIGHT_PAD) && defined(HAS_BOTTOM_PAD)
    {
        const uint32_t start_corner = get_arg(args::start_corner);
        const uint32_t num_corner = get_arg(args::num_corner);
        for (uint32_t k = 0; k < num_corner; ++k) {
            const uint32_t slice = start_corner + k;
            const uint32_t tile_id = slice * H_tiles * W_tiles + (H_tiles - 1u) * W_tiles + (W_tiles - 1u);
            dfb_data_out.wait_front(1);
            noc.async_write(dfb_data_out, s, tile_bytes, {.offset_bytes = 0}, {.page_id = tile_id});
            noc.async_writes_flushed();
            dfb_data_out.pop_front(1);
        }
    }
#endif

    noc.async_write_barrier();
}
