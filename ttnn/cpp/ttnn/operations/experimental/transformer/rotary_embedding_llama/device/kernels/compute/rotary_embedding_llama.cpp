// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"

ALWI void ACQ() { acquire_dst(); }
ALWI void REL() { release_dst(); }

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

    const uint32_t rotary_seq_t_end = seq_t_end < rotary_Ht ? seq_t_end : rotary_Ht;
    const uint32_t my_rotary_seq_tiles = seq_t_start < rotary_seq_t_end ? rotary_seq_t_end - seq_t_start : 0;
    const uint32_t my_cos_sin_tiles = my_rotary_seq_tiles * Wt;

    mm_init(in_cb, trans_mat_cb, out_cb);
    binary_op_init_common(rotated_in_interm_cb, cos_cb, out_cb);  // General Init for all binary ops

    // Get the trans_mat
    cb_wait_front(trans_mat_cb, onetile);

    uint32_t in0_index = 0;
    uint32_t in1_index = 0;
    uint32_t interm_index = 0;

    for (uint32_t batch_id = batch_start; batch_id < batch_end; ++batch_id) {
#if RELOAD_IMPL == 0
        if (my_cos_sin_tiles > 0) {
            cb_wait_front(sin_cb, my_cos_sin_tiles);
            cb_wait_front(cos_cb, my_cos_sin_tiles);
        }
#endif
        for (uint32_t head_num = 0; head_num < n_heads; ++head_num) {
            uint32_t sin_cos_row_cnt = 0;
            for (uint32_t seq_tile = seq_t_start; seq_tile < rotary_seq_t_end; ++seq_tile) {
                // input cb wait and reserve
                cb_wait_front(in_cb, Wt);
#if RELOAD_IMPL == 1
                cb_wait_front(sin_cb, Wt);
                cb_wait_front(cos_cb, Wt);
#endif

                cb_reserve_back(rotated_in_interm_cb, Wt);
                cb_reserve_back(sin_interm_cb, Wt);
                cb_reserve_back(cos_interm_cb, Wt);
                cb_reserve_back(out_cb, Wt);

                // // rotated = x @ trans_mat
                mm_init_short(in_cb, trans_mat_cb);
                ACQ();
                for (uint32_t j = 0; j < Wt; ++j) {
                    matmul_tiles(in_cb, trans_mat_cb, j, in1_index, j);
                    pack_tile(j, rotated_in_interm_cb, j);
                }
                REL();
                cb_push_back(rotated_in_interm_cb, Wt);
                cb_wait_front(rotated_in_interm_cb, Wt);

                // sin_interim = rotated * sin
                // A = rotated_in_interm_cb InputLifecycle::Bulk + Block (Wt tiles waited above, popped below).
                // B = sin_cb: index = j + (sin_cos_row_cnt * Wt). Block + compute_kernel_lib::TileOffset::Set.
                //   RELOAD_IMPL==0: sin held externally (waited my_cos_sin_tiles outside)
                //     -> InputLifecycle::CallerManaged; sin_cos_row_cnt increments per seq_tile so offset varies.
                //   RELOAD_IMPL==1: sin waited Wt per iter (line 63), popped Wt below
                //     -> InputLifecycle::Bulk; sin_cos_row_cnt always 0 so compute_kernel_lib::TileOffset::Set(0).
                // Output sin_interm_cb OutputLifecycle::Bulk + Block (Wt tiles).
                // Reconfig audit: mul_tiles_init reconfigs srca/srcb -> Input.
                //   No explicit pack_reconfig (relies on sin_interm_cb format == out_cb's
                //   from startup) -> PackTileReconfig::None.
#if RELOAD_IMPL == 0
                compute_kernel_lib::eltwise_chain(
                    compute_kernel_lib::EltwiseShape::tiles(Wt, /*block_size=*/Wt),
                    compute_kernel_lib::BinaryFpu<
                        rotated_in_interm_cb,
                        sin_cb,
                        compute_kernel_lib::BinaryFpuOp::Mul,
                        compute_kernel_lib::BroadcastDim::None,
                        compute_kernel_lib::InputLifecycle::Bulk,
                        compute_kernel_lib::InputLifecycle::CallerManaged,
                        compute_kernel_lib::BinaryDataFormatReconfig::Input,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OperandKind::Block,
                        compute_kernel_lib::OperandKind::Block,
                        compute_kernel_lib::TileOffset::Unset,
                        compute_kernel_lib::TileOffset::Set>{0u, sin_cos_row_cnt * Wt},
                    compute_kernel_lib::PackTile<
                        sin_interm_cb,
                        compute_kernel_lib::OutputLifecycle::Bulk,
                        compute_kernel_lib::PackTileReconfig::None>{});
#else
                compute_kernel_lib::mul<
                    rotated_in_interm_cb,
                    sin_cb,
                    sin_interm_cb,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::InputLifecycle::Bulk,
                    compute_kernel_lib::InputLifecycle::Bulk,
                    compute_kernel_lib::OutputLifecycle::Bulk,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::PackTileReconfig::None,
                    compute_kernel_lib::OperandKind::Block>(
                    compute_kernel_lib::EltwiseShape::tiles(Wt, /*block_size=*/Wt));
#endif

                // cos_interim = x * cos  — same pattern as sin_interim.
#if RELOAD_IMPL == 0
                compute_kernel_lib::eltwise_chain(
                    compute_kernel_lib::EltwiseShape::tiles(Wt, /*block_size=*/Wt),
                    compute_kernel_lib::BinaryFpu<
                        in_cb,
                        cos_cb,
                        compute_kernel_lib::BinaryFpuOp::Mul,
                        compute_kernel_lib::BroadcastDim::None,
                        compute_kernel_lib::InputLifecycle::Bulk,
                        compute_kernel_lib::InputLifecycle::CallerManaged,
                        compute_kernel_lib::BinaryDataFormatReconfig::Input,
                        compute_kernel_lib::Dst::D0,
                        compute_kernel_lib::OperandKind::Block,
                        compute_kernel_lib::OperandKind::Block,
                        compute_kernel_lib::TileOffset::Unset,
                        compute_kernel_lib::TileOffset::Set>{0u, sin_cos_row_cnt * Wt},
                    compute_kernel_lib::PackTile<
                        cos_interm_cb,
                        compute_kernel_lib::OutputLifecycle::Bulk,
                        compute_kernel_lib::PackTileReconfig::None>{});
#else
                compute_kernel_lib::mul<
                    in_cb,
                    cos_cb,
                    cos_interm_cb,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::InputLifecycle::Bulk,
                    compute_kernel_lib::InputLifecycle::Bulk,
                    compute_kernel_lib::OutputLifecycle::Bulk,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::PackTileReconfig::None,
                    compute_kernel_lib::OperandKind::Block>(
                    compute_kernel_lib::EltwiseShape::tiles(Wt, /*block_size=*/Wt));
#endif

                // out = cos_interim + sin_interim
                // Both operands InputLifecycle::Bulk + Block (Wt tiles, popped at end), out_cb OutputLifecycle::Bulk +
                // Block. Reconfig: add_tiles_init reconfigs srca/srcb -> Input. No pack_reconfig -> None.
                compute_kernel_lib::add<
                    cos_interm_cb,
                    sin_interm_cb,
                    out_cb,
                    compute_kernel_lib::BroadcastDim::None,
                    compute_kernel_lib::InputLifecycle::Bulk,
                    compute_kernel_lib::InputLifecycle::Bulk,
                    compute_kernel_lib::OutputLifecycle::Bulk,
                    compute_kernel_lib::BinaryDataFormatReconfig::Input,
                    compute_kernel_lib::PackTileReconfig::None,
                    compute_kernel_lib::OperandKind::Block>(
                    compute_kernel_lib::EltwiseShape::tiles(Wt, /*block_size=*/Wt));

#if RELOAD_IMPL == 0
                // no-reload needs to increment this counter
                // Used a sin/cos row
                sin_cos_row_cnt++;
#endif
            }
        }

#if RELOAD_IMPL == 0
        if (my_cos_sin_tiles > 0) {
            cb_pop_front(sin_cb, my_cos_sin_tiles);
            cb_pop_front(cos_cb, my_cos_sin_tiles);
        }
#endif
    }

    // Done with the transformation matrix, so remove from CB
    cb_pop_front(trans_mat_cb, onetile);
}
