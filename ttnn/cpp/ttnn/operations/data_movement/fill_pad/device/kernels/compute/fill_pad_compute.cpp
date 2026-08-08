// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

ALWI void process_masked_tile(
    DataflowBuffer& dfb_data_in, DataflowBuffer& dfb_mask, DataflowBuffer& dfb_data_out, std::uint32_t fill_bits) {
    constexpr std::uint32_t CB_DATA_IN = 0;
    constexpr std::uint32_t CB_DATA_PADDING = 1;
    constexpr std::uint32_t CB_MASK = 2;
    constexpr std::uint32_t CB_OUT = 2;  // reuse CB_MASK tile

    dfb_data_in.wait_front(1);
    dfb_data_out.reserve_back(1);
    tile_regs_acquire();

    copy_tile_to_dst_init_short(dfb_data_in.get_id());
    copy_tile(dfb_data_in.get_id(), 0, CB_DATA_IN);  // data → DST[0]

    copy_tile_to_dst_init_short(dfb_mask.get_id());
    copy_tile(dfb_mask.get_id(), 0, CB_MASK);  // mask → DST[2]

    fill_tile_init();
    FILL_PAD_FILL_FN(CB_DATA_PADDING, FILL_PAD_FILL_ARG);  // fill → DST

    where_tile_init();
    where_tile<FILL_PAD_DATA_FMT>(CB_MASK, CB_DATA_PADDING, CB_DATA_IN, CB_OUT);  // if mask then padidng else in -> out

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(CB_OUT, dfb_data_out.get_id());  // result is at DST[2]
    tile_regs_release();

    dfb_data_in.pop_front(1);
    dfb_data_out.push_back(1);
}

// Corner tile: two sequential where_tile calls give (right_mask OR bot_mask) → fill.
ALWI void process_corner_tile(
    DataflowBuffer& dfb_data_in,
    DataflowBuffer& dfb_right_mask,
    DataflowBuffer& dfb_bot_mask,
    DataflowBuffer& dfb_data_out,
    std::uint32_t fill_bits) {
    constexpr std::uint32_t CB_DATA_IN = 0;
    constexpr std::uint32_t CB_DATA_PADDING = 1;
    constexpr std::uint32_t CB_RIGHT_MASK = 2;
    constexpr std::uint32_t CB_BOTTOM_MASK = 3;
    constexpr std::uint32_t CB_OUT = 3;  // reuse CB_MASK tile

    dfb_data_in.wait_front(1);
    dfb_data_out.reserve_back(1);
    tile_regs_acquire();

    copy_tile_to_dst_init_short(dfb_data_in.get_id());
    copy_tile(dfb_data_in.get_id(), 0, CB_DATA_IN);  // data       → DST[0]

    copy_tile_to_dst_init_short(dfb_right_mask.get_id());
    copy_tile(dfb_right_mask.get_id(), 0, CB_RIGHT_MASK);  // right_mask → DST[2]

    copy_tile_to_dst_init_short(dfb_bot_mask.get_id());
    copy_tile(dfb_bot_mask.get_id(), 0, CB_BOTTOM_MASK);  // bot_mask   → DST[3]

    fill_tile_init();
    FILL_PAD_FILL_FN(CB_DATA_PADDING, FILL_PAD_FILL_ARG);  // fill → DST[1]

    where_tile_init();
    // Combine right and bottom mask into corner mask
    where_tile<FILL_PAD_DATA_FMT>(
        CB_RIGHT_MASK, CB_DATA_PADDING, CB_DATA_IN, CB_RIGHT_MASK);  // if mask then padidng else in -> out
    where_tile<FILL_PAD_DATA_FMT>(
        CB_BOTTOM_MASK, CB_DATA_PADDING, CB_RIGHT_MASK, CB_OUT);  // if mask then padidng else in -> out

    tile_regs_commit();
    tile_regs_wait();
    pack_tile(CB_OUT, dfb_data_out.get_id());  // final result is at DST[3]
    tile_regs_release();

    dfb_data_in.pop_front(1);
    dfb_data_out.push_back(1);
}

void kernel_main() {
    // W_tiles / H_tiles / elem_size are carried for parity with the reader/writer arg
    // layout but are unused by this kernel (the loops are driven by the per-phase counts).
    [[maybe_unused]] constexpr auto W_tiles = get_arg(args::W_tiles);
    [[maybe_unused]] constexpr auto H_tiles = get_arg(args::H_tiles);
    [[maybe_unused]] constexpr auto elem_size = get_arg(args::elem_size);
    constexpr auto fill_bits_ct = get_arg(args::fill_bits);

    // has_right_pad / has_bottom_pad are carried as preprocessor defines (not CTAs),
    // because they gate references to the conditionally-bound right / bottom mask DFBs.

    // Per-phase tile counts. Phases with num == 0 are skipped. When the
    // corresponding mask is not bound the host always sets num to 0, and the
    // #ifdef gating below removes the dead code path entirely.
    const auto num_right = get_arg(args::num_right);
    const auto num_bottom = get_arg(args::num_bottom);
    const auto num_corner = get_arg(args::num_corner);

    if (num_right + num_bottom + num_corner == 0) {
        return;
    }

    DataflowBuffer dfb_data_in(dfb::data_in);
#ifdef HAS_RIGHT_PAD
    DataflowBuffer dfb_right_mask(dfb::right_mask);
#endif
#ifdef HAS_BOTTOM_PAD
    DataflowBuffer dfb_bot_mask(dfb::bot_mask);
#endif
    DataflowBuffer dfb_data_out(dfb::data_out);

    // Standard init for unary-style SFPU compute with one primary input DFB.
    unary_op_init_common(dfb::data_in, dfb::data_out);

    // Wait for persistent mask tiles pushed once by the writer. They are popped
    // once at cleanup; during the main loop they are reused persistently.
#ifdef HAS_RIGHT_PAD
    dfb_right_mask.wait_front(1);
#endif
#ifdef HAS_BOTTOM_PAD
    dfb_bot_mask.wait_front(1);
#endif

    // ---- Main loop: same tile ordering as reader and writer (right/bottom/corner) ----

#ifdef HAS_RIGHT_PAD
    for (std::uint32_t i = 0; i < num_right; ++i) {
        process_masked_tile(dfb_data_in, dfb_right_mask, dfb_data_out, fill_bits_ct);
    }
#endif
#ifdef HAS_BOTTOM_PAD
    for (std::uint32_t j = 0; j < num_bottom; ++j) {
        process_masked_tile(dfb_data_in, dfb_bot_mask, dfb_data_out, fill_bits_ct);
    }
#endif
#if defined(HAS_RIGHT_PAD) && defined(HAS_BOTTOM_PAD)
    for (std::uint32_t k = 0; k < num_corner; ++k) {
        process_corner_tile(dfb_data_in, dfb_right_mask, dfb_bot_mask, dfb_data_out, fill_bits_ct);
    }
#endif

    // Clean-up
#ifdef HAS_RIGHT_PAD
    dfb_right_mask.pop_front(1);
#endif
#ifdef HAS_BOTTOM_PAD
    dfb_bot_mask.pop_front(1);
#endif
}
