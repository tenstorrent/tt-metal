// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t packed_scalar1 = get_arg_val<uint32_t>(1);
    const uint32_t packed_scalar2 = get_arg_val<uint32_t>(2);
    const auto true_value = reinterpret_cast<const float*>(&packed_scalar1);
    const auto false_value = reinterpret_cast<const float*>(&packed_scalar2);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    DataflowBuffer dfb_in(cb_input);
    DataflowBuffer dfb_out(cb_output);

    compute_kernel_hw_startup(cb_input, cb_output);
    copy_init(cb_input);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);
        tile_regs_acquire();
        copy_init(cb_input);
        copy_tile(cb_input, 0, 0);

        fill_tile_init();
#if defined(INP_INT32)
        fill_tile_int<DataFormat::Int32>(1, packed_scalar1);
        fill_tile_int<DataFormat::Int32>(2, packed_scalar2);
#endif
#if defined(INP_UINT32)
        fill_tile_int<DataFormat::UInt32>(1, packed_scalar1);
        fill_tile_int<DataFormat::UInt32>(2, packed_scalar2);
#endif
#if defined(INP_FLOAT) || defined(INP_FLOAT32)
        fill_tile(1, *true_value);
        fill_tile(2, *false_value);
#endif
#ifndef SFPU_OP_CHAIN_0
#error "where_tss_kernel requires SFPU_OP_CHAIN_0 to be defined via get_block_defines"
#endif
        SFPU_OP_CHAIN_0
        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_output);
        tile_regs_release();

        dfb_in.pop_front(1);
        dfb_out.push_back(1);
    }
}
