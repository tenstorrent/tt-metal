// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of transpose_wh_rm.cpp (see the legacy copy alongside).
// Compute logic is unchanged; resource access uses Metal 2.0 named handles:
//   - c_0 / c_24 / c_16 (legacy magic CB indices) -> dfb::in0 / dfb::tilize / dfb::out
//   - Ht / Wt / HtWt (legacy CTAs 0-2)            -> named compile-time args
//   - num_hw_blocks_per_core (legacy RTA 0)       -> named RTA
//
// Forked (not modified in place) because the legacy copy is also bound by
// TransposeWHShardedRMProgramFactory, which is still on the legacy host API and compiles
// this source with SHARDED defined. Only the non-sharded path lives here: the fork is never
// compiled with SHARDED, so carrying that branch would be dead code referencing buffers this
// spec does not bind. The sharded variant — including its pack_untilize narrow-row path and
// the yolov4/PCC note explaining the use_narrow_row conditions — remains in the legacy copy.

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/transpose.h"
#include "api/compute/tilize.h"
#include "api/compute/pack_untilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

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
ALWI void transpose_with_pack_untilize(uint32_t dfb_tilize, DataflowBuffer& dfb_out_buf) {
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

    uint32_t num_hw_blocks_per_core = get_arg(args::num_hw_blocks_per_core);

    constexpr auto dfb_in = dfb::in0;
    constexpr auto dfb_tilize = dfb::tilize;
    constexpr auto dfb_out_idx = dfb::out;

    DataflowBuffer dfb_tilize_buf(dfb_tilize);
    DataflowBuffer dfb_out(dfb_out_idx);

    unary_op_init_common(dfb_in, dfb_out_idx);

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
        transpose_with_pack_untilize<Wt, Ht, HtWt, dfb_out_idx>(dfb_tilize, dfb_out);

        dfb_tilize_buf.pop_front(HtWt);
    }
}
