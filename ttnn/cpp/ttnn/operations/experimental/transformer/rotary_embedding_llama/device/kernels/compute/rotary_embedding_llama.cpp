// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "api/dataflow/circular_buffer.h"

namespace ckl = compute_kernel_lib;

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

void kernel_main() {
    uint32_t argrt = 0;
    uint32_t batch_start = get_arg_val<uint32_t>(argrt++);
    uint32_t batch_end = get_arg_val<uint32_t>(argrt++);
    uint32_t seq_t_start = get_arg_val<uint32_t>(argrt++);
    uint32_t seq_t_end = get_arg_val<uint32_t>(argrt++);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t in_cb = get_compile_time_arg_val(0);
    constexpr uint32_t cos_cb = get_compile_time_arg_val(1);
    constexpr uint32_t sin_cb = get_compile_time_arg_val(2);
    constexpr uint32_t trans_mat_cb = get_compile_time_arg_val(3);

    constexpr uint32_t rotated_in_interm_cb = get_compile_time_arg_val(4);
    constexpr uint32_t cos_interm_cb = get_compile_time_arg_val(5);
    constexpr uint32_t sin_interm_cb = get_compile_time_arg_val(6);
    constexpr uint32_t out_cb = get_compile_time_arg_val(7);
    constexpr uint32_t Wt = get_compile_time_arg_val(8);
    constexpr uint32_t n_heads = get_compile_time_arg_val(9);
    constexpr uint32_t rotary_Ht = get_compile_time_arg_val(10);

    CircularBuffer in_cb_obj(in_cb);
    CircularBuffer cos_cb_obj(cos_cb);
    CircularBuffer sin_cb_obj(sin_cb);
    CircularBuffer trans_mat_cb_obj(trans_mat_cb);
    CircularBuffer rotated_in_interm_cb_obj(rotated_in_interm_cb);
    CircularBuffer cos_interm_cb_obj(cos_interm_cb);
    CircularBuffer sin_interm_cb_obj(sin_interm_cb);
    CircularBuffer out_cb_obj(out_cb);

    const uint32_t rotary_seq_t_end = seq_t_end < rotary_Ht ? seq_t_end : rotary_Ht;
    const uint32_t my_rotary_seq_tiles = seq_t_start < rotary_seq_t_end ? rotary_seq_t_end - seq_t_start : 0;
    const uint32_t my_cos_sin_tiles = my_rotary_seq_tiles * Wt;

    compute_kernel_hw_startup<SrcOrder::Reverse>(in_cb, trans_mat_cb, out_cb);
    matmul_init(in_cb, trans_mat_cb);
    binary_op_init_common(rotated_in_interm_cb, cos_cb, out_cb);  // General Init for all binary ops

    // Get the trans_mat
    trans_mat_cb_obj.wait_front(onetile);

    uint32_t in0_index = 0;
    uint32_t in1_index = 0;
    uint32_t interm_index = 0;

    for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
#if RELOAD_IMPL == 0
        if (my_cos_sin_tiles > 0) {
            sin_cb_obj.wait_front(my_cos_sin_tiles);
            cos_cb_obj.wait_front(my_cos_sin_tiles);
        }
#endif
        for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
            uint32_t sin_cos_row_cnt = 0;
            for (uint32_t seq_tile = seq_t_start; seq_tile < rotary_seq_t_end; ++seq_tile) {
                // input cb wait and reserve
                in_cb_obj.wait_front(Wt);
#if RELOAD_IMPL == 1
                sin_cb_obj.wait_front(Wt);
                cos_cb_obj.wait_front(Wt);
#endif

                rotated_in_interm_cb_obj.reserve_back(Wt);
                sin_interm_cb_obj.reserve_back(Wt);
                cos_interm_cb_obj.reserve_back(Wt);
                out_cb_obj.reserve_back(Wt);

                // // rotated = x @ trans_mat
                matmul_init(in_cb, trans_mat_cb);
                ACQ();
                for (uint32_t j = 0; j < Wt; ++j) {
                    matmul_tiles(in_cb, trans_mat_cb, j, in1_index, j);
                    pack_tile(j, rotated_in_interm_cb, j);
                }
                REL();
                rotated_in_interm_cb_obj.push_back(Wt);
                rotated_in_interm_cb_obj.wait_front(Wt);

#if RELOAD_IMPL == 0
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(Wt, /*block_size=*/Wt),
                    ckl::BinaryFpu<
                        ckl::input(rotated_in_interm_cb, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                        ckl::input(
                            sin_cb,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::OperandKind::Block,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileOffset::Set),
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::None>{0u, sin_cos_row_cnt * Wt},
                    ckl::PackTile<ckl::output(
                        sin_interm_cb, ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled)>{});
#else
                ckl::mul<
                    ckl::input(rotated_in_interm_cb, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                    ckl::input(sin_cb, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                    ckl::output(sin_interm_cb, ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled),
                    ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(Wt, /*block_size=*/Wt));
#endif

#if RELOAD_IMPL == 0
                ckl::eltwise_chain(
                    ckl::EltwiseShape::tiles(Wt, /*block_size=*/Wt),
                    ckl::BinaryFpu<
                        ckl::input(in_cb, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                        ckl::input(
                            cos_cb,
                            ckl::InputLifecycle::CallerManaged,
                            ckl::OperandKind::Block,
                            ckl::DataFormatReconfig::Enabled,
                            ckl::TileOffset::Set),
                        ckl::BinaryFpuOp::Mul,
                        ckl::BroadcastDim::None>{0u, sin_cos_row_cnt * Wt},
                    ckl::PackTile<ckl::output(
                        cos_interm_cb, ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled)>{});
#else
                ckl::mul<
                    ckl::input(in_cb, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                    ckl::input(cos_cb, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                    ckl::output(cos_interm_cb, ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled),
                    ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(Wt, /*block_size=*/Wt));
#endif

                ckl::add<
                    ckl::input(cos_interm_cb, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                    ckl::input(sin_interm_cb, ckl::InputLifecycle::Bulk, ckl::OperandKind::Block),
                    ckl::output(out_cb, ckl::OutputLifecycle::Bulk, ckl::DataFormatReconfig::Disabled),
                    ckl::BroadcastDim::None>(ckl::EltwiseShape::tiles(Wt, /*block_size=*/Wt));

#if RELOAD_IMPL == 0
                // no-reload needs to increment this counter
                // Used a sin/cos row
                sin_cos_row_cnt++;
#endif
            }
        }

#if RELOAD_IMPL == 0
        if (my_cos_sin_tiles > 0) {
            sin_cb_obj.pop_front(my_cos_sin_tiles);
            cos_cb_obj.pop_front(my_cos_sin_tiles);
        }
#endif
    }

    // Done with the transformation matrix, so remove from CB
    trans_mat_cb_obj.pop_front(onetile);
}
