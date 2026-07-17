// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 op-local compute kernel for transpose's row-major sharded WH factory, split out from the
// SHARDED branch of the row-major WH compute path. Resource bindings use the Metal 2.0 namespaces
// (dfb::/args::). Differences from the original combined kernel:
//   - the non-SHARDED branch is dropped (this fork is only compiled for the sharded caller);
//   - the output buffer is referenced through a single accessor (dfb::cb_out); the host factory maps
//     it to the borrowed output shard (ht<=8) or the intermediate staging DFB (ht>8), so the kernel
//     needs no Ht>8 ternary over two cb indices;
//   - the two dead compile-time args of the legacy SHARDED path (pack_num_pages, pack_num_pages_last_row)
//     are dropped — they were read into constexprs the kernel never used.
// The producer-side cb_out_buf.wait_front in transpose_with_pack_untilize is preserved verbatim: it
// reads the received-tiles counter the kernel just bumped (returns immediately) and is a producer-side
// barrier, not a consume — so the factory binds compute as PRODUCER-only of the output DFB.
// The interleaved (non-sharded) row-major path lives in transpose_wh_rm.cpp.

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/transpose.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
// DEBUG [#47797 WH-transpose compute hang]: localize the LLK stall via the watcher RING BUFFER, NOT
// DPRINT — DPRINT on Quasar craq-sim trips an unimplemented MMIO read (t_tile_mmio_rd32). The ring
// buffer is the safe localizer (push from MATH only; the watcher dump shows the last entries -> the
// furthest stage reached). Run with the watcher enabled and the ring buffer NOT disabled. Remove after.
#include "api/debug/ring_buffer.h"
// 0x57_48_<stage> markers ("WH"). Pushed only from MATH to avoid a 3-TRISC race on the ring pointer.
#define RBT(stage) MATH(WATCHER_RING_BUFFER_PUSH(0x57480000u | (uint32_t)(stage)))

template <
    uint32_t Wt,
    uint32_t Ht,
    uint32_t HtWt,
    bool use_narrow_row,
    uint32_t row_size,
    uint32_t pack_num_pages_last_col,
    uint32_t pack_num_pages_last_row_col,
    uint32_t cb_out>
ALWI void transpose_with_pack_untilize_narrow_row(uint32_t cb_tilize, DataflowBuffer& cb_out_buf) {
    uint32_t tile_idx = 0;

    transpose_init(cb_tilize);
    RBT(0x0F1);  // DEBUG narrow: post transpose_wh_init_short, pre pack_untilize_dest_init (remove after)
    pack_untilize_dest_init<Ht, Ht, use_narrow_row, row_size>(cb_out);
    RBT(0x100);  // DEBUG narrow: init done, pre w-loop (remove after)
    for (uint32_t w = 0; w < Wt; ++w) {
        RBT(0x110 | (w & 0xf));  // DEBUG narrow: pre transpose_wh_tile (remove after)
        tile_regs_acquire();
        for (uint32_t h = 0; h < Ht; ++h) {
            transpose_tile(cb_tilize, tile_idx, h);
            tile_idx += Wt;
        }

        tile_regs_commit();
        RBT(0x120 | (w & 0xf));  // DEBUG narrow: post transpose_wh_tile, pre pack (remove after)

        if (w == Wt - 1) {  // last row
            cb_out_buf.reserve_back(pack_num_pages_last_row_col);
            tile_regs_wait();
            pack_untilize_dest<Ht, Ht, false, use_narrow_row, row_size>(cb_out);
            tile_regs_release();
            cb_out_buf.push_back(pack_num_pages_last_row_col);
        } else {
            cb_out_buf.reserve_back(pack_num_pages_last_col);
            tile_regs_wait();
            pack_untilize_dest<Ht, Ht, false, use_narrow_row, row_size>(cb_out);
            tile_regs_release();
            cb_out_buf.push_back(pack_num_pages_last_col);
        }
        RBT(0x130 | (w & 0xf));  // DEBUG narrow: post pack_untilize (remove after)
        tile_idx = tile_idx - HtWt + 1;
    }
    pack_untilize_uninit(cb_out);
    RBT(0x140);  // DEBUG narrow: fn done (remove after)
}

// Helper constexpr function to compute num_blocks_per_col
constexpr uint32_t compute_num_blocks_per_col(uint32_t per_core_block_tile_cnt) {
    const uint32_t max_bct = DST_ACCUM_MODE ? 4 : 8;

    for (uint32_t bct = max_bct; bct >= 1; --bct) {
        if (per_core_block_tile_cnt % bct == 0) {
            return per_core_block_tile_cnt / bct;
        }
    }

    return 1;
}

template <uint32_t Wt, uint32_t Ht, uint32_t HtWt, uint32_t cb_out>
ALWI void transpose_with_pack_untilize(uint32_t cb_tilize, DataflowBuffer& cb_out_buf) {
    uint32_t tile_idx = 0;

    transpose_init(cb_tilize);
    RBT(0x0F2);  // DEBUG wide: post transpose_wh_init_short, pre pack_untilize_dest_init (remove after)
    constexpr uint32_t num_blocks_per_col = compute_num_blocks_per_col(Ht);
    constexpr uint32_t block_ct_dim = Ht / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = Ht;
    pack_untilize_dest_init<block_ct_dim, full_ct_dim>(cb_out);
    RBT(0x200);  // DEBUG wide: init done, pre w-loop (remove after)
    for (uint32_t w = 0; w < Wt; ++w) {
        RBT(0x210 | (w & 0xf));  // DEBUG wide: pre reserve cb_out (remove after)
        cb_out_buf.reserve_back(Ht);
        for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
            RBT(0x220 | (w & 0xf));  // DEBUG wide: pre transpose_wh_tile (remove after)
            tile_regs_acquire();
            for (uint32_t h = 0; h < block_ct_dim; ++h) {
                transpose_tile(cb_tilize, tile_idx, h);
                tile_idx += Wt;
            }
            tile_regs_commit();
            RBT(0x230 | (w & 0xf));  // DEBUG wide: post transpose_wh_tile, pre pack (remove after)

            tile_regs_wait();
            pack_untilize_dest<block_ct_dim, full_ct_dim>(cb_out, 1, b);
            tile_regs_release();
            RBT(0x240 | (w & 0xf));  // DEBUG wide: post pack_untilize (remove after)
        }
        cb_out_buf.push_back(Ht);

        // Producer-side barrier on the just-pushed column (reads the received-tiles counter this
        // kernel bumped; returns immediately). Not a consume — there is no pop_front.
        RBT(0x250 | (w & 0xf));  // DEBUG wide: post push, pre self wait_front (remove after)
        cb_out_buf.wait_front(Ht);
        RBT(0x260 | (w & 0xf));  // DEBUG wide: post self wait_front (remove after)
        tile_idx = tile_idx - HtWt + 1;
    }
    pack_untilize_uninit(cb_out);
    RBT(0x270);  // DEBUG wide: fn done (remove after)
}

void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t HtWt = get_arg(args::HtWt);
    constexpr uint32_t num_hw_blocks_per_core = get_arg(args::num_hw_blocks_per_core);
    constexpr uint32_t last_output_row_num_datums = get_arg(args::last_output_row_num_datums);
    constexpr uint32_t pack_num_pages_last_col = get_arg(args::pack_num_pages_last_col);
    constexpr uint32_t pack_num_pages_last_row_col = get_arg(args::pack_num_pages_last_row_col);

    // In order to support full_ct_dim > block_ct_dim, we would need to change use_narrow_row and row_size conditions to
    // be:
    //
    //     constexpr bool use_narrow_row = last_output_row_num_datums < FACE_WIDTH ? true : false;
    //     constexpr uint32_t row_size = last_output_row_num_datums < FACE_WIDTH ? last_output_row_num_datums :
    //     TILE_WIDTH;
    //
    // However, changing these conditions makes yolov4 and transpose tests fail with a PCC error. For the error to be
    // present in BH, it needs to be run in the full model sweep, but for WH_B0 it can be seen in isolation.
    constexpr bool use_narrow_row = last_output_row_num_datums < TILE_WIDTH ? true : false;
    constexpr uint32_t row_size = last_output_row_num_datums < TILE_WIDTH ? last_output_row_num_datums : TILE_WIDTH;

    DataflowBuffer cb_tilize_buf(dfb::cb_tilize);
    DataflowBuffer cb_out(dfb::cb_out);

    unary_op_init_common(dfb::cb_in, dfb::cb_out);

    // DEBUG [#47797 WH-transpose compute hang]: localize where the 4 compute TRISCs stall (watcher: all
    // at WFD, reader at RBW). Watcher RING BUFFER markers (safe on Quasar; DPRINT trips an unimplemented
    // sim MMIO read). The watcher dump's last ring entries = the furthest stage MATH reached. Stage map:
    //   0x001 start | 0x0E<nn> enter iter n | 0x002 pre-tilize | 0x003 post-tilize | 0x004 cb_tilize got
    //   0x005 post-transpose-fn | 0x006 iter done | 0x00F end
    //   0x0F1/0x0F2 = post transpose_wh_init_short, pre pack_untilize_dest_init (narrow/wide) — disambiguates
    //       the two init calls: last==0x004 -> stuck in transpose_wh_init_short; last==0x0F1/0x0F2 -> stuck in
    //       pack_untilize_dest_init
    //   narrow fn: 0x100 init, 0x11w pre-transpose_wh_tile, 0x12w post-transpose/pre-pack, 0x13w post-pack, 0x140 done
    //   wide   fn: 0x200 init, 0x21w reserve, 0x22w pre-transpose, 0x23w post-transpose/pre-pack, 0x24w post-pack,
    //              0x25w pre-self-wait, 0x26w post-self-wait, 0x270 done
    // (the shape — nblk/Ht/Wt/HtWt/narrow — is already in the host compile-args log). Remove after.
    RBT(0x001);

    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        RBT(0x0E00 | (n & 0xff));  // DEBUG: enter iter n (remove after)
        RBT(0x002);                // DEBUG: pre tilize (remove after)
        // Tilize input (Ht rows × Wt tiles). Fp32Mode::Lossless keeps the full
        // Float32 mantissa through tilization; the default Fast mode would
        // collapse it to tf32 precision before the transpose ever runs.
        compute_kernel_lib::tilize<
            Wt,
            dfb::cb_in,
            dfb::cb_tilize,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
            compute_kernel_lib::tilize_config::Fp32Mode::Lossless>(Ht);

        // transpose
        RBT(0x003);  // DEBUG: post tilize, pre cb_tilize wait_front (remove after)
        cb_tilize_buf.wait_front(HtWt);
        RBT(0x004);  // DEBUG: cb_tilize got, pre transpose fn (remove after)
        if constexpr (use_narrow_row) {
            transpose_with_pack_untilize_narrow_row<
                Wt,
                Ht,
                HtWt,
                use_narrow_row,
                row_size,
                pack_num_pages_last_col,
                pack_num_pages_last_row_col,
                dfb::cb_out>(dfb::cb_tilize, cb_out);
        } else {
            transpose_with_pack_untilize<Wt, Ht, HtWt, dfb::cb_out>(dfb::cb_tilize, cb_out);
        }
        RBT(0x005);  // DEBUG: post transpose fn, pre cb_tilize pop_front (remove after)

        cb_tilize_buf.pop_front(HtWt);
        RBT(0x006);  // DEBUG: iter done (remove after)
    }
    RBT(0x00F);  // DEBUG: end (remove after)
}
