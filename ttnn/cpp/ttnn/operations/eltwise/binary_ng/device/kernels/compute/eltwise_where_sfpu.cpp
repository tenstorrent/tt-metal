// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "eltwise_utils_common.hpp"
#include "eltwise_utils_sfpu.hpp"
#include "api/dataflow/dataflow_buffer.h"

ALWI void process_tile(
    tt::CBIndex cb_in0_id,
    tt::CBIndex cb_in1_id,
    tt::CBIndex cb_out_id,
    const uint32_t scalar_value,
    const float* scalar_val,
    uint32_t freq,
    uint32_t tile_start,
    uint32_t num_tiles_per_cycle) {
    using namespace ckernel;

    DataflowBuffer dfb_in0(cb_in0_id);
    DataflowBuffer dfb_in1(cb_in1_id);
    DataflowBuffer dfb_out(cb_out_id);

#if BCAST_INPUT  // BCAST_INPUT == 1 : input B ( true or false tensor) is broadcasted
    DataflowBuffer& dfb_bcast = dfb_in1;
    DataflowBuffer& dfb_other = dfb_in0;
#else  // BCAST_INPUT == 0 : input A (condition tensor)  is broadcasted
    DataflowBuffer& dfb_bcast = dfb_in0;
    DataflowBuffer& dfb_other = dfb_in1;
#endif

    dfb_bcast.wait_front(num_tiles_per_cycle);

    for (uint32_t j = tile_start; j < freq; ++j) {
        dfb_other.wait_front(num_tiles_per_cycle);
        dfb_out.reserve_back(num_tiles_per_cycle);

        tile_regs_acquire();
        copy_tile_to_dst_init_short(dfb_in0.get_id());
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(dfb_in0.get_id(), i, i * 3);
        }
        copy_tile_to_dst_init_short(dfb_in1.get_id());
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            // TTS: tensor is true value, goes to dst_reg 1
#if WHERE_TTS
            copy_tile(dfb_in1.get_id(), i, i * 3 + 1);  // Copy true tensor to dst_reg 1
            fill_tile_init();
// TTS: scalar is false value, goes to dst_reg 2
#ifdef FILL_WITH_VALUE_FLOAT
            FILL_LLK(i * 3 + 2, *scalar_val);
#endif
#ifdef FILL_WITH_VALUE_INT
            FILL_LLK(i * 3 + 2, scalar_value);
#endif
#endif

// TST: tensor is false value, goes to dst_reg 2
#if WHERE_TST
            copy_tile(dfb_in1.get_id(), i, i * 3 + 2);  // Copy false tensor to dst_reg 2
            fill_tile_init();
// TST: scalar is true value, goes to dst_reg 1
#ifdef FILL_WITH_VALUE_FLOAT
            FILL_LLK(i * 3 + 1, *scalar_val);
#endif
#ifdef FILL_WITH_VALUE_INT
            FILL_LLK(i * 3 + 1, scalar_value);
#endif
#endif
            BINARY_SFPU_OP(i * 3, i * 3 + 1, i * 3 + 2, i * 3);
        }

        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            pack_tile(i * 3, dfb_out.get_id());
        }
        tile_regs_release();

        dfb_out.push_back(num_tiles_per_cycle);
        dfb_other.pop_front(num_tiles_per_cycle);
    }
    dfb_bcast.pop_front(num_tiles_per_cycle);
}

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);
    const float* scalar_value_ptr = reinterpret_cast<const float*>(&scalar_value);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_in0_id = tt::CBIndex::c_0;
    constexpr auto cb_in1_id = tt::CBIndex::c_1;
    constexpr auto cb_out_id = tt::CBIndex::c_2;

    unary_op_init_common(cb_in0_id, cb_out_id);
    BINARY_SFPU_INIT

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(
            cb_in0_id,
            cb_in1_id,
            cb_out_id,
            scalar_value,
            scalar_value_ptr,
            tile_freq,
            tile_start,
            num_tiles_per_cycle);
    }

    if (remaining_iterations > 0) {
        process_tile(
            cb_in0_id,
            cb_in1_id,
            cb_out_id,
            scalar_value,
            scalar_value_ptr,
            remaining_iterations,
            tile_start,
            num_tiles_per_cycle);
    }
}
