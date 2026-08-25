// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of rotary_embedding_single_tile.cpp (which keeps serving the
// consumers still on the legacy API until the last of them migrates).
//
// Compute kernel for ttnn.experimental.rotary_embedding when head_dim == TILE_WIDTH
// (single tile along the W dimension). Uses an in-L1 transformation matrix to do
// HF-style rotate_half via matmul_tiles, since the inter-tile half-swap used for
// Wt >= 2 cannot express a sub-tile rotation.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/kernel_args.h"

template <uint32_t in0_dfb, uint32_t out_dfb>
ALWI void UNTILIZE_ONE_TILE() {
    compute_kernel_lib::untilize<
        1,
        in0_dfb,
        out_dfb,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitUpfront,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
}

template <uint32_t in0_dfb, uint32_t out_dfb>
ALWI void TILIZE_ONE_TILE(uint32_t sync_dfb) {
    DataflowBuffer dfb_sync(sync_dfb);
    dfb_sync.wait_front(1);
    compute_kernel_lib::tilize<
        1,
        in0_dfb,
        out_dfb,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    dfb_sync.pop_front(1);
}

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_dfb = dfb::in;
    constexpr uint32_t cos_dfb = dfb::cos;
    constexpr uint32_t sin_dfb = dfb::sin;
    constexpr uint32_t trans_mat_dfb = dfb::trans_mat;
    constexpr uint32_t rotated_in_interm_dfb = dfb::rotated_in_interm;
    constexpr uint32_t cos_interm_dfb = dfb::cos_interm;
    constexpr uint32_t sin_interm_dfb = dfb::sin_interm;
    constexpr uint32_t out_dfb = dfb::out;
    constexpr uint32_t num_rows = get_arg(args::num_rows);

    DataflowBuffer dfb_in(in_dfb);
    DataflowBuffer dfb_trans_mat(trans_mat_dfb);
    DataflowBuffer dfb_rotated_in_interm(rotated_in_interm_dfb);
    DataflowBuffer dfb_cos_interm(cos_interm_dfb);
    DataflowBuffer dfb_sin_interm(sin_interm_dfb);
    DataflowBuffer dfb_out(out_dfb);

    uint32_t updated_cos_dfb = cos_dfb;
    uint32_t updated_sin_dfb = sin_dfb;

#ifdef DECODE_MODE
    constexpr uint32_t untilized_cos_dfb = dfb::untilized_cos;
    constexpr uint32_t untilized_cos_sync_dfb = dfb::untilized_cos_sync;
    constexpr uint32_t untilized_sin_dfb = dfb::untilized_sin;
    constexpr uint32_t untilized_sin_sync_dfb = dfb::untilized_sin_sync;
    constexpr uint32_t retilized_cos_dfb = dfb::retilized_cos;
    constexpr uint32_t retilized_sin_dfb = dfb::retilized_sin;

    compute_kernel_hw_startup(sin_dfb, sin_dfb, untilized_sin_dfb);
    UNTILIZE_ONE_TILE<sin_dfb, untilized_sin_dfb>();
    UNTILIZE_ONE_TILE<cos_dfb, untilized_cos_dfb>();
    reconfig_data_format_srca(cos_dfb, untilized_sin_dfb);
    pack_reconfig_data_format(untilized_cos_dfb, retilized_sin_dfb);
    TILIZE_ONE_TILE<untilized_sin_dfb, retilized_sin_dfb>(untilized_sin_sync_dfb);
    TILIZE_ONE_TILE<untilized_cos_dfb, retilized_cos_dfb>(untilized_cos_sync_dfb);
    updated_cos_dfb = retilized_cos_dfb;
    updated_sin_dfb = retilized_sin_dfb;
#endif

    dfb_trans_mat.wait_front(onetile);
    compute_kernel_hw_startup(rotated_in_interm_dfb, updated_sin_dfb, sin_interm_dfb);

    for (uint32_t i = 0; i < num_rows; ++i) {
        // rotated = in @ trans_mat  (HF rotate_half on a single 32x32 tile)
        dfb_in.wait_front(onetile);
        reconfig_data_format(in_dfb, trans_mat_dfb);
        pack_reconfig_data_format(rotated_in_interm_dfb);
        matmul_init(in_dfb, trans_mat_dfb);

        tile_regs_acquire();
        matmul_tiles(in_dfb, trans_mat_dfb, 0, 0, 0);
        tile_regs_commit();

        dfb_rotated_in_interm.reserve_back(onetile);

        tile_regs_wait();
        pack_tile(0, rotated_in_interm_dfb);
        tile_regs_release();

        dfb_rotated_in_interm.push_back(onetile);

        // sin_interim = rotated * sin
        DataflowBuffer dfb_updated_sin(updated_sin_dfb);
        dfb_rotated_in_interm.wait_front(onetile);
        dfb_updated_sin.wait_front(onetile);
        reconfig_data_format(rotated_in_interm_dfb, updated_sin_dfb);
        pack_reconfig_data_format(sin_interm_dfb);

        tile_regs_acquire();
#ifdef DECODE_MODE
        mul_bcast_rows_init(rotated_in_interm_dfb, updated_sin_dfb);
        mul_tiles_bcast_rows(rotated_in_interm_dfb, updated_sin_dfb, 0, 0, 0);
#else
        mul_init(rotated_in_interm_dfb, updated_sin_dfb);
        mul_tiles(rotated_in_interm_dfb, updated_sin_dfb, 0, 0, 0);
#endif
        tile_regs_commit();

        dfb_rotated_in_interm.pop_front(onetile);
#ifndef DECODE_MODE
        dfb_updated_sin.pop_front(onetile);
#endif

        dfb_sin_interm.reserve_back(onetile);

        tile_regs_wait();
        pack_tile(0, sin_interm_dfb);
        tile_regs_release();

        dfb_sin_interm.push_back(onetile);

        // cos_interim = in * cos
        DataflowBuffer dfb_updated_cos(updated_cos_dfb);
        dfb_updated_cos.wait_front(onetile);
        reconfig_data_format(in_dfb, updated_cos_dfb);
        pack_reconfig_data_format(cos_interm_dfb);

        tile_regs_acquire();
#ifdef DECODE_MODE
        mul_bcast_rows_init(in_dfb, updated_cos_dfb);
        mul_tiles_bcast_rows(in_dfb, updated_cos_dfb, 0, 0, 0);
#else
        mul_init(in_dfb, updated_cos_dfb);
        mul_tiles(in_dfb, updated_cos_dfb, 0, 0, 0);
#endif
        tile_regs_commit();

        dfb_in.pop_front(onetile);
#ifndef DECODE_MODE
        dfb_updated_cos.pop_front(onetile);
#endif

        dfb_cos_interm.reserve_back(onetile);

        tile_regs_wait();
        pack_tile(0, cos_interm_dfb);
        tile_regs_release();

        dfb_cos_interm.push_back(onetile);

        // out = cos_interim + sin_interim
        dfb_cos_interm.wait_front(onetile);
        dfb_sin_interm.wait_front(onetile);
        reconfig_data_format(cos_interm_dfb, sin_interm_dfb);
        pack_reconfig_data_format(out_dfb);
        add_init(cos_interm_dfb, sin_interm_dfb);

        tile_regs_acquire();
        add_tiles(cos_interm_dfb, sin_interm_dfb, 0, 0, 0);
        tile_regs_commit();

        dfb_cos_interm.pop_front(onetile);
        dfb_sin_interm.pop_front(onetile);

        dfb_out.reserve_back(onetile);

        tile_regs_wait();
        pack_tile(0, out_dfb);
        tile_regs_release();

        dfb_out.push_back(onetile);
    }
}
