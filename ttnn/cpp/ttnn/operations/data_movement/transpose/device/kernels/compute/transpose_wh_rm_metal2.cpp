// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row-major WH transpose: each block of Ht x Wt tiles is tilized, transposed tile by tile, and
// pack-untilized back to rows.
//   - dfb::in0 / dfb::tilize / dfb::out: the tilize input, the tilized tiles the transpose reads,
//     and the pack destination. With OUT_STAGING (sharded path, Ht > 8) the pack destination is
//     dfb::out_staging instead.
//   - Ht / Wt / HtWt are compile-time args. num_hw_blocks_per_core is a runtime arg on the
//     non-sharded path; under SHARDED it is a compile-time arg, as are the output-row and
//     pack-page settings.
//
// TransposeWHProgramFactory binds this source without SHARDED; TransposeWHShardedRMProgramFactory
// binds it with SHARDED.

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/transpose.h"
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
    uint32_t dfb_out,
    typename DFB>
ALWI void transpose_with_pack_untilize_narrow_row(uint32_t dfb_tilize, DFB& dfb_out_buf) {
    uint32_t tile_idx = 0;

    transpose_init(dfb_tilize);
    pack_untilize_dest_init<Ht, Ht, use_narrow_row, row_size>(dfb_out);
    for (uint32_t w = 0; w < Wt; ++w) {
        tile_regs_acquire();
        for (uint32_t h = 0; h < Ht; ++h) {
            transpose_tile(dfb_tilize, tile_idx, h);
            tile_idx += Wt;
        }

        tile_regs_commit();

        if (w == Wt - 1) {  // last row
            dfb_out_buf.reserve_back(pack_num_pages_last_row_col);
            tile_regs_wait();
            pack_untilize_dest<Ht, Ht, false, use_narrow_row, row_size>(dfb_out);
            tile_regs_release();
            dfb_out_buf.push_back(pack_num_pages_last_row_col);
        } else {
            dfb_out_buf.reserve_back(pack_num_pages_last_col);
            tile_regs_wait();
            pack_untilize_dest<Ht, Ht, false, use_narrow_row, row_size>(dfb_out);
            tile_regs_release();
            dfb_out_buf.push_back(pack_num_pages_last_col);
        }
        tile_idx = tile_idx - HtWt + 1;
    }
    pack_untilize_uninit(dfb_out);
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

template <uint32_t Wt, uint32_t Ht, uint32_t HtWt, uint32_t dfb_out, typename DFB>
ALWI void transpose_with_pack_untilize(uint32_t dfb_tilize, DFB& dfb_out_buf) {
    uint32_t tile_idx = 0;

    transpose_init(dfb_tilize);
    constexpr uint32_t num_blocks_per_col = compute_num_blocks_per_col(Ht);
    constexpr uint32_t block_ct_dim = Ht / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = Ht;
    pack_untilize_dest_init<block_ct_dim, full_ct_dim>(dfb_out);
    for (uint32_t w = 0; w < Wt; ++w) {
        dfb_out_buf.reserve_back(Ht);
        for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
            tile_regs_acquire();
            for (uint32_t h = 0; h < block_ct_dim; ++h) {
                transpose_tile(dfb_tilize, tile_idx, h);
                tile_idx += Wt;
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_untilize_dest<block_ct_dim, full_ct_dim>(dfb_out, 1, b);
            tile_regs_release();
        }
        dfb_out_buf.push_back(Ht);

        tile_idx = tile_idx - HtWt + 1;
    }
    pack_untilize_uninit(dfb_out);
}

void kernel_main() {
    constexpr auto Ht = get_arg(args::Ht);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto HtWt = get_arg(args::HtWt);
#ifdef SHARDED
    constexpr auto num_hw_blocks_per_core = get_arg(args::num_hw_blocks_per_core);
    constexpr auto last_output_row_num_datums = get_arg(args::last_output_row_num_datums);
    constexpr auto pack_num_pages = get_arg(args::pack_num_pages);
    constexpr auto pack_num_pages_last_col = get_arg(args::pack_num_pages_last_col);
    constexpr auto pack_num_pages_last_row = get_arg(args::pack_num_pages_last_row);
    constexpr auto pack_num_pages_last_row_col = get_arg(args::pack_num_pages_last_row_col);
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

#else
    uint32_t num_hw_blocks_per_core = get_arg(args::num_hw_blocks_per_core);
#endif

    constexpr auto dfb_in = dfb::in0;
    constexpr auto dfb_tilize = dfb::tilize;
#ifdef OUT_STAGING
    // Sharded path only, when Ht > 8: pack into the staging buffer the writer drains into the
    // output shard. (temporary fix until pack_untilize is fully fixed)
    constexpr auto dfb_out_idx = dfb::out_staging;
#else
    constexpr auto dfb_out_idx = dfb::out;
#endif

    DataflowBuffer dfb_tilize_buf(dfb_tilize);
    DataflowBuffer dfb_out(dfb_out_idx);

#ifdef SHARDED
    compute_kernel_hw_startup(dfb_in, dfb_out_idx);
    copy_init(dfb_in);
#else
    unary_op_init_common(dfb_in, dfb_out_idx);
#endif

    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        // Tilize input (Ht rows × Wt tiles). Fp32Mode::Lossless keeps the full
        // Float32 mantissa through tilization; the default Fast mode would
        // collapse it to tf32 precision before the transpose ever runs.
        compute_kernel_lib::tilize<
            Wt,
            dfb_in,
            dfb_tilize,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
            compute_kernel_lib::tilize_config::Fp32Mode::Lossless>(Ht);

        // transpose
        dfb_tilize_buf.wait_front(HtWt);
#ifdef SHARDED
        if constexpr (use_narrow_row) {
            transpose_with_pack_untilize_narrow_row<
                Wt,
                Ht,
                HtWt,
                use_narrow_row,
                row_size,
                pack_num_pages_last_col,
                pack_num_pages_last_row_col,
                dfb_out_idx>(dfb_tilize, dfb_out);
        } else {
            transpose_with_pack_untilize<Wt, Ht, HtWt, dfb_out_idx>(dfb_tilize, dfb_out);
        }
#else
        transpose_with_pack_untilize<Wt, Ht, HtWt, dfb_out_idx>(dfb_tilize, dfb_out);
#endif

        dfb_tilize_buf.pop_front(HtWt);
    }
}
