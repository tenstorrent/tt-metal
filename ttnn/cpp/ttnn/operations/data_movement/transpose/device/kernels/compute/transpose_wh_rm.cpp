// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/transpose.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "api/dataflow/dataflow_buffer.h"

template <
    uint32_t Wt,
    uint32_t Ht,
    uint32_t HtWt,
    bool use_narrow_row,
    uint32_t row_size,
    uint32_t pack_num_pages_last_col,
    uint32_t pack_num_pages_last_row_col,
    uint32_t dfb_out>
ALWI void transpose_with_pack_untilize_narrow_row(uint32_t cb_tilize, DataflowBuffer& dfb_out_buf) {
    uint32_t tile_idx = 0;

    transpose_init(cb_tilize);
    pack_untilize_dest_init<Ht, Ht, use_narrow_row, row_size>(dfb_out);
    for (uint32_t w = 0; w < Wt; ++w) {
        tile_regs_acquire();
        for (uint32_t h = 0; h < Ht; ++h) {
            transpose_tile(cb_tilize, tile_idx, h);
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

template <uint32_t Wt, uint32_t Ht, uint32_t HtWt, uint32_t dfb_out>
ALWI void transpose_with_pack_untilize(uint32_t cb_tilize, DataflowBuffer& dfb_out_buf) {
    uint32_t tile_idx = 0;

    transpose_init(cb_tilize);
    constexpr uint32_t num_blocks_per_col = compute_num_blocks_per_col(Ht);
    constexpr uint32_t block_ct_dim = Ht / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = Ht;
    pack_untilize_dest_init<block_ct_dim, full_ct_dim>(dfb_out);
    for (uint32_t w = 0; w < Wt; ++w) {
        dfb_out_buf.reserve_back(Ht);
        for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
            tile_regs_acquire();
            for (uint32_t h = 0; h < block_ct_dim; ++h) {
                transpose_tile(cb_tilize, tile_idx, h);
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
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t HtWt = get_compile_time_arg_val(2);
#ifdef SHARDED
    constexpr uint32_t num_hw_blocks_per_core = get_compile_time_arg_val(3);
    constexpr uint32_t last_output_row_num_datums = get_compile_time_arg_val(4);
    constexpr uint32_t pack_num_pages = get_compile_time_arg_val(5);
    constexpr uint32_t pack_num_pages_last_col = get_compile_time_arg_val(6);
    constexpr uint32_t pack_num_pages_last_row = get_compile_time_arg_val(7);
    constexpr uint32_t pack_num_pages_last_row_col = get_compile_time_arg_val(8);
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
    uint32_t num_hw_blocks_per_core = get_arg_val<uint32_t>(0);
#endif

#ifdef SHARDED
    constexpr auto cb_in = tt::CBIndex::c_24;
    constexpr auto cb_tilize = tt::CBIndex::c_25;
    constexpr auto cb_out_idx =
        (Ht > 8) ? tt::CBIndex::c_27 : tt::CBIndex::c_16;  // temporary fix until pack_untilize is fully fixed
#else
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_tilize = tt::CBIndex::c_24;
    constexpr auto cb_out_idx = tt::CBIndex::c_16;
#endif

    DataflowBuffer dfb_tilize_buf(cb_tilize);
    DataflowBuffer dfb_out(cb_out_idx);

    unary_op_init_common(cb_in, cb_out_idx);

    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        // Tilize input (Ht rows × Wt tiles). Fp32Mode::Lossless keeps the full
        // Float32 mantissa through tilization; the default Fast mode would
        // collapse it to tf32 precision before the transpose ever runs.
        compute_kernel_lib::tilize<
            Wt,
            cb_in,
            cb_tilize,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
            compute_kernel_lib::tilize_config::Fp32Mode::Lossless>(Ht);

        // transpose
        dfb_tilize_buf.wait_front(HtWt);
        uint32_t tile_idx = 0;
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
                cb_out_idx>(cb_tilize, dfb_out);
        } else {
            transpose_with_pack_untilize<Wt, Ht, HtWt, cb_out_idx>(cb_tilize, dfb_out);
        }
#else
        transpose_with_pack_untilize<Wt, Ht, HtWt, cb_out_idx>(cb_tilize, dfb_out);
#endif

        dfb_tilize_buf.pop_front(HtWt);
    }
}
