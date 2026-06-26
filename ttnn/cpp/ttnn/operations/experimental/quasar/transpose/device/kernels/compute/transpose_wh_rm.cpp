// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 op-local compute kernel for transpose's row-major WH (interleaved) factory. Resource
// bindings use the Metal 2.0 namespaces (dfb::/args::). The SHARDED row-major path lives in a
// separate kernel (transpose_wh_rm_sharded.cpp); this kernel is only ever compiled for the
// interleaved (non-sharded) caller.

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

template <uint32_t Wt, uint32_t Ht, uint32_t HtWt, uint32_t cb_out>
ALWI void transpose_with_pack_untilize(uint32_t cb_tilize, DataflowBuffer& cb_out_buf) {
    uint32_t tile_idx = 0;

    transpose_init(cb_tilize);
    constexpr uint32_t num_blocks_per_col = compute_num_blocks_per_col(Ht);
    constexpr uint32_t block_ct_dim = Ht / num_blocks_per_col;
    constexpr uint32_t full_ct_dim = Ht;
    pack_untilize_dest_init<block_ct_dim, full_ct_dim>(cb_out);
    for (uint32_t w = 0; w < Wt; ++w) {
        cb_out_buf.reserve_back(Ht);
        for (uint32_t b = 0; b < num_blocks_per_col; ++b) {
            tile_regs_acquire();
            for (uint32_t h = 0; h < block_ct_dim; ++h) {
                transpose_tile(cb_tilize, tile_idx, h);
                tile_idx += Wt;
            }
            tile_regs_commit();

            tile_regs_wait();
            pack_untilize_dest<block_ct_dim, full_ct_dim>(cb_out, 1, b);
            tile_regs_release();
        }
        cb_out_buf.push_back(Ht);
        tile_idx = tile_idx - HtWt + 1;
    }
    pack_untilize_uninit(cb_out);
}

void kernel_main() {
    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t HtWt = get_arg(args::HtWt);
    uint32_t num_hw_blocks_per_core = get_arg(args::num_hw_blocks);

    DataflowBuffer cb_tilize_buf(dfb::cb_tilize);
    DataflowBuffer cb_out(dfb::cb_out0);

    unary_op_init_common(dfb::cb_in0, dfb::cb_out0);

    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        // Tilize input (Ht rows × Wt tiles). Fp32Mode::Lossless keeps the full
        // Float32 mantissa through tilization; the default Fast mode would
        // collapse it to tf32 precision before the transpose ever runs.
        compute_kernel_lib::tilize<
            Wt,
            dfb::cb_in0,
            dfb::cb_tilize,
            compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
            compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
            compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure,
            compute_kernel_lib::tilize_config::Fp32Mode::Lossless>(Ht);

        // transpose
        cb_tilize_buf.wait_front(HtWt);
        transpose_with_pack_untilize<Wt, Ht, HtWt, dfb::cb_out0>(dfb::cb_tilize, cb_out);

        cb_tilize_buf.pop_front(HtWt);
    }
}
