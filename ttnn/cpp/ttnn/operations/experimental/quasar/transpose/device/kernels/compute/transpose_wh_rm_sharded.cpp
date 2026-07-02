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
#include "api/compute/transpose_wh.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

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

    transpose_wh_init_short(cb_tilize);
    pack_untilize_dest_init<Ht, Ht, use_narrow_row, row_size>(cb_out);
    for (uint32_t w = 0; w < Wt; ++w) {
        tile_regs_acquire();
        for (uint32_t h = 0; h < Ht; ++h) {
            transpose_wh_tile(cb_tilize, tile_idx, h);
            tile_idx += Wt;
        }

        tile_regs_commit();

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
        tile_idx = tile_idx - HtWt + 1;
    }
    pack_untilize_uninit(cb_out);
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

    transpose_wh_init_short(cb_tilize);
    constexpr uint32_t num_blocks_per_col = compute_num_blocks_per_col(Ht);
    constexpr uint32_t block_ct_dim = Ht / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = Ht;
    pack_untilize_dest_init<block_ct_dim, full_ct_dim>(cb_out);
    for (uint32_t w = 0; w < Wt; ++w) {
        cb_out_buf.reserve_back(Ht);
        for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
            tile_regs_acquire();
            for (uint32_t h = 0; h < block_ct_dim; ++h) {
                transpose_wh_tile(cb_tilize, tile_idx, h);
                tile_idx += Wt;
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_untilize_dest<block_ct_dim, full_ct_dim>(cb_out, 1, b);
            tile_regs_release();
        }
        cb_out_buf.push_back(Ht);

        // Producer-side barrier on the just-pushed column (reads the received-tiles counter this
        // kernel bumped; returns immediately). Not a consume — there is no pop_front.
        cb_out_buf.wait_front(Ht);
        tile_idx = tile_idx - HtWt + 1;
    }
    pack_untilize_uninit(cb_out);
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

    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
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
        cb_tilize_buf.wait_front(HtWt);
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

        cb_tilize_buf.pop_front(HtWt);
    }
}
