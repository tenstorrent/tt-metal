// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/where.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/bcast.h"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils_common.hpp"
#include "ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_utils.hpp"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    const uint32_t scalar_value = get_arg_val<uint32_t>(3);
    const auto scalar_val = reinterpret_cast<const float*>(&scalar_value);

    constexpr uint32_t num_tiles_per_cycle = get_compile_time_arg_val(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;
    DataflowBuffer exp_dfb_out(cb_out);

    // The broadcast operand reaches DEST without a pack: see the invariant in eltwise_binary_sfpu_row_bcast.cpp.
    // DEST slots per tile: [0] = condition, [1] = true value, [2] = false value, stride 3. The second tensor
    // goes to slot 1 (TTS) or 2 (TST); the scalar fills the remaining slot.
#if WHERE_TTS
    constexpr uint32_t tensor_dst_offset = 1;
#else
    constexpr uint32_t tensor_dst_offset = 2;
#endif
#if SRC_BCAST
    constexpr auto cb_bcast = cb_in0;
    constexpr auto cb_plain = cb_in1;
    constexpr uint32_t bcast_dst_offset = 0;
    constexpr uint32_t plain_dst_offset = tensor_dst_offset;
#endif
#if SRC_BCAST_B
    constexpr auto cb_bcast = cb_in1;
    constexpr auto cb_plain = cb_in0;
    constexpr uint32_t bcast_dst_offset = tensor_dst_offset;
    constexpr uint32_t plain_dst_offset = 0;
#endif

    DataflowBuffer exp_dfb_bcast(cb_bcast);
    DataflowBuffer exp_dfb_plain(cb_plain);

    compute_kernel_hw_startup(cb_in0, cb_out);
    copy_init(cb_in0);
    BINARY_SFPU_INIT

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        exp_dfb_bcast.wait_front(num_tiles_per_cycle);
        exp_dfb_plain.wait_front(num_tiles_per_cycle);
        exp_dfb_out.reserve_back(num_tiles_per_cycle);

        tile_regs_acquire();
        reconfig_data_format(cb_bcast, cb_bcast);
        unary_bcast_init<BroadcastType::ROW>(cb_bcast);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            unary_bcast<BroadcastType::ROW>(cb_bcast, i, i * 3 + bcast_dst_offset);
        }
        reconfig_data_format_srca(cb_bcast, cb_plain);
        copy_init(cb_plain);
        for (uint32_t i = 0; i < num_tiles_per_cycle; ++i) {
            copy_tile(cb_plain, i, i * 3 + plain_dst_offset);
            fill_tile_init();
// TTS: scalar is false value, goes to dst_reg 2
#if WHERE_TTS
#ifdef FILL_WITH_VALUE_FLOAT
            FILL_LLK(i * 3 + 2, *scalar_val);
#endif
#ifdef FILL_WITH_VALUE_INT
            FILL_LLK(i * 3 + 2, scalar_value);
#endif
#endif

// TST: scalar is true value, goes to dst_reg 1
#if WHERE_TST
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
            pack_tile(i * 3, cb_out);
        }
        tile_regs_release();

        exp_dfb_out.push_back(num_tiles_per_cycle);
        exp_dfb_bcast.pop_front(num_tiles_per_cycle);
        exp_dfb_plain.pop_front(num_tiles_per_cycle);
    }
}
