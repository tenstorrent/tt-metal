// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/kernel_args.h"

ALWI void MUL_TILES(uint32_t in0_dfb, uint32_t in1_dfb, uint32_t out_dfb, uint32_t num_tiles, uint32_t in1_idx) {
    DataflowBuffer dfb_in0(in0_dfb);
    DataflowBuffer dfb_in1(in1_dfb);
    DataflowBuffer dfb_out(out_dfb);
    // Multiply input by cos
    dfb_in0.wait_front(num_tiles);
    dfb_in1.wait_front(in1_idx + 1);

    tile_regs_acquire();
#ifdef DECODE_MODE
    mul_bcast_rows_init(in0_dfb, in1_dfb);
    mul_tiles_bcast_rows(in0_dfb, in1_dfb, 0, in1_idx, 0);
#else
    mul_init(in0_dfb, in1_dfb);
    mul_tiles(in0_dfb, in1_dfb, 0, 0, 0);
#endif
    tile_regs_commit();

    dfb_in0.pop_front(num_tiles);
#ifndef DECODE_MODE
    // We don't pop in1 in decode which is sin/cos since we don't stream
    dfb_in1.pop_front(num_tiles);
#endif

    dfb_out.reserve_back(num_tiles);

    tile_regs_wait();
    pack_tile(0, out_dfb);
    tile_regs_release();

    dfb_out.push_back(num_tiles);
}

template <uint32_t num_tiles, uint32_t in0_dfb, uint32_t out_dfb>
ALWI void UNTILIZE_TILES() {
    compute_kernel_lib::untilize<
        num_tiles,
        in0_dfb,
        out_dfb,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitUpfront,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
}

template <uint32_t num_tiles, uint32_t in0_dfb, uint32_t out_dfb>
ALWI void TILIZE_ROWS(uint32_t sync_dfb) {
    DataflowBuffer dfb_sync(sync_dfb);
    dfb_sync.wait_front(num_tiles);
    compute_kernel_lib::tilize<
        num_tiles,
        in0_dfb,
        out_dfb,
        compute_kernel_lib::tilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::tilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::tilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    dfb_sync.pop_front(num_tiles);
}

void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t in_dfb = dfb::in;
    constexpr uint32_t rotated_in_dfb = dfb::rotated_in;
    constexpr uint32_t cos_dfb = dfb::cos;
    constexpr uint32_t sin_dfb = dfb::sin;
    constexpr uint32_t scalar_dfb = dfb::scalar;
    constexpr uint32_t rotated_in_interm_dfb = dfb::rotated_in_interm;
    constexpr uint32_t cos_interm_dfb = dfb::cos_interm;
    constexpr uint32_t sin_interm_dfb = dfb::sin_interm;
    constexpr uint32_t out_dfb = dfb::out;
    constexpr uint32_t num_rows = get_arg(args::num_rows);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t half_Wt = get_arg(args::half_Wt);

    DataflowBuffer dfb_in(in_dfb);
    DataflowBuffer dfb_rotated_in(rotated_in_dfb);
    DataflowBuffer dfb_scalar(scalar_dfb);
    DataflowBuffer dfb_rotated_in_interm(rotated_in_interm_dfb);
    DataflowBuffer dfb_cos_interm(cos_interm_dfb);
    DataflowBuffer dfb_sin_interm(sin_interm_dfb);
    DataflowBuffer dfb_out(out_dfb);

    dfb_scalar.wait_front(onetile);

    uint32_t updated_cos_dfb = cos_dfb;
    uint32_t updated_sin_dfb = sin_dfb;

#ifdef DECODE_MODE
    constexpr uint32_t untilized_cos_dfb = dfb::untilized_cos;
    constexpr uint32_t untilized_cos_sync_dfb = dfb::untilized_cos_sync;
    constexpr uint32_t untilized_sin_dfb = dfb::untilized_sin;
    constexpr uint32_t untilized_sin_sync_dfb = dfb::untilized_sin_sync;
    constexpr uint32_t retilized_cos_dfb = dfb::retilized_cos;
    constexpr uint32_t retilized_sin_dfb = dfb::retilized_sin;
    compute_kernel_hw_startup(sin_dfb, scalar_dfb, untilized_sin_dfb);
    UNTILIZE_TILES<Wt, sin_dfb, untilized_sin_dfb>();
    UNTILIZE_TILES<Wt, cos_dfb, untilized_cos_dfb>();
    reconfig_data_format_srca(cos_dfb, untilized_sin_dfb);
    pack_reconfig_data_format(untilized_cos_dfb, retilized_sin_dfb);
    TILIZE_ROWS<Wt, untilized_sin_dfb, retilized_sin_dfb>(untilized_sin_sync_dfb);
    TILIZE_ROWS<Wt, untilized_cos_dfb, retilized_cos_dfb>(untilized_cos_sync_dfb);
    updated_cos_dfb = retilized_cos_dfb;
    updated_sin_dfb = retilized_sin_dfb;
#else
    compute_kernel_hw_startup(rotated_in_dfb, scalar_dfb, rotated_in_interm_dfb);
#endif
    uint32_t in1_idx = 0;
    for (uint32_t i = 0; i < num_rows; ++i) {
        for (uint32_t j = 0; j < Wt; ++j) {
#ifdef DECODE_MODE
            in1_idx = j;
#endif
            if (j < half_Wt) {
                // Multiply half of the rotated input by scalar (-1)
                reconfig_data_format(rotated_in_dfb, scalar_dfb);
                pack_reconfig_data_format(rotated_in_interm_dfb);
                dfb_rotated_in.wait_front(onetile);

                tile_regs_acquire();
                mul_bcast_scalar_init(rotated_in_dfb, scalar_dfb);
                mul_tiles_bcast_scalar(rotated_in_dfb, scalar_dfb, 0, 0, 0);
                tile_regs_commit();

                dfb_rotated_in.pop_front(onetile);

                dfb_rotated_in_interm.reserve_back(onetile);

                tile_regs_wait();
                pack_tile(0, rotated_in_interm_dfb);
                tile_regs_release();

                dfb_rotated_in_interm.push_back(onetile);
                reconfig_data_format_srcb(scalar_dfb, updated_sin_dfb);
                pack_reconfig_data_format(rotated_in_interm_dfb, sin_interm_dfb);
                // Multiply rotated input by sin
                MUL_TILES(rotated_in_interm_dfb, updated_sin_dfb, sin_interm_dfb, onetile, in1_idx);
            } else {
                reconfig_data_format(rotated_in_dfb, updated_sin_dfb);
                pack_reconfig_data_format(out_dfb, sin_interm_dfb);
                // Multiply rotated input by sin
                MUL_TILES(rotated_in_dfb, updated_sin_dfb, sin_interm_dfb, onetile, in1_idx);
            }

            // Multiply input by cos
            MUL_TILES(in_dfb, updated_cos_dfb, cos_interm_dfb, onetile, in1_idx);

            // Add applied sin/cos tensors
            dfb_cos_interm.wait_front(onetile);
            dfb_sin_interm.wait_front(onetile);

            reconfig_data_format_srca(rotated_in_dfb, cos_interm_dfb);
            pack_reconfig_data_format(cos_interm_dfb, out_dfb);

            tile_regs_acquire();
            add_init(cos_interm_dfb, sin_interm_dfb);
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
}
