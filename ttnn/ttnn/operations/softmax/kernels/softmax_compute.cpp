// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Softmax compute kernel.
//
// 4-phase numerically-stable softmax pipeline:
//   Phase 1: max reduce       → cb_max
//   Phase 2: sub + exp (fused) → cb_exp
//   Phase 3: sum reduce + recip→ cb_recip_sum
//   Phase 4: mul (broadcast)   → cb_output_tiles
//
// Layout dispatch (CT arg is_rm):
//   TILE path: math reads from cb_input_tiles (reader populated it directly)
//   RM path:   tilize cb_rm_in → cb_input_tiles at slab start,
//               math runs on cb_input_tiles as usual,
//               untilize cb_output_tiles → cb_rm_out at slab end.
//
// The 4-phase math is identical for both layouts — the layout decision lives
// at the data-access boundary (tilize/untilize wrap), not in the math.
//
// All compute phases use kernel-lib helpers. The cb_pop_front calls between
// phases are CB maintenance (freeing HeldBulk intermediates), not compute.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/recip.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace {
// CB indices — must match program descriptor
constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_scaler_max = 1;
constexpr uint32_t cb_scaler_sum = 2;
constexpr uint32_t cb_rm_in = 3;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_rm_out = 17;
constexpr uint32_t cb_max = 24;
constexpr uint32_t cb_exp = 25;
constexpr uint32_t cb_recip_sum = 26;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr int32_t dim = static_cast<int32_t>(get_compile_time_arg_val(2));
    constexpr uint32_t is_rm = get_compile_time_arg_val(3);
    constexpr uint32_t origin_W = get_compile_time_arg_val(4);
    constexpr uint32_t origin_H = get_compile_time_arg_val(5);

    // Partial scaler: needed when the REDUCTION axis is non-tile-aligned.
    // dim=-1 reduces along W → partial if W % 32 != 0
    // dim=-2 reduces along H → partial if H % 32 != 0
    constexpr uint32_t partial_W = origin_W % 32;
    constexpr uint32_t partial_H = origin_H % 32;
    constexpr bool has_partial = (dim == -1) ? (partial_W > 0) : (partial_H > 0);
    constexpr auto partial_scaler = has_partial ? compute_kernel_lib::ReducePartialScaler::last_tile_at(1)
                                                : compute_kernel_lib::ReducePartialScaler::none();

    uint32_t num_slabs = get_arg_val<uint32_t>(0);  // slabs assigned to this core

    // compute_kernel_hw_startup configures SrcA=input, SrcB=scaler, Pack=intermediate
    // For RM path: icb0=cb_rm_in is the tilize input; ocb=cb_input_tiles is the tilize output.
    //   The softmax math will later reconfigure SrcA to cb_input_tiles (handled by helpers).
    // For TILE path: icb0=cb_input_tiles is the math input; icb1=cb_scaler_max; ocb=cb_max.
    if constexpr (is_rm) {
        // RM path: hw_startup with tilize's input and output CBs
        compute_kernel_hw_startup(cb_rm_in, cb_input_tiles);
    } else {
        // TILE path: hw_startup with softmax math's input/scaler/output CBs
        compute_kernel_hw_startup(cb_input_tiles, cb_scaler_max, cb_max);
    }

    // PostReduceOp: reciprocal after sum reduce
    auto recip_op = [](uint32_t dst_idx) {
        recip_tile_init();
        recip_tile(dst_idx);
    };

    constexpr auto reduce_block_shape = ckl::ReduceInputBlockShape::of(Ht, Wt, 1);
    constexpr auto eltwise_shape = ckl::EltwiseShape::grid(Ht, Wt);

    for (uint32_t slab = 0; slab < num_slabs; ++slab) {
        // ===== RM path: tilize cb_rm_in → cb_input_tiles =====
        if constexpr (is_rm) {
            // Tilize Ht blocks of Wt tiles each.
            // InitAndUninit: full init on first call, uninit on last.
            // WaitBlock: wait for each block before processing.
            // NoReconfigure: same dtype on input and output (both are the input dtype).
            ckl::tilize<
                Wt,
                cb_rm_in,
                cb_input_tiles,
                ckl::tilize_config::InitUninitMode::InitAndUninit,
                ckl::tilize_config::WaitMode::WaitBlock,
                ckl::tilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(Ht);
        }

        if constexpr (dim == -1) {
            // ===== dim=-1 (W reduction): REDUCE_ROW, BroadcastDim::Col =====

            // Phase 1: Max reduce (WaitUpfrontNoPop — input retained for Phase 2)
            ckl::reduce<
                ckernel::PoolType::MAX,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_input_tiles,
                cb_scaler_max,
                cb_max,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ckl::NoAccumulation,
                ckl::NoOp>(
                reduce_block_shape,
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::NoAccumulation{},
                ckl::NoOp{},
                partial_scaler);

            // Phase 2: Sub + Exp (fused chain)
            ckl::eltwise_chain(
                eltwise_shape,
                ckl::BinaryFpu<
                    cb_input_tiles,
                    cb_max,
                    ckl::BinaryFpuOp::Sub,
                    ckl::BroadcastDim::Col,
                    ckl::InputLifecycle::Bulk,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::Dst::D0,
                    ckl::OperandKind::Block,
                    ckl::OperandKind::Col>{},
                ckl::Exp<>{},
                ckl::PackTile<cb_exp, ckl::OutputLifecycle::Streaming>{});

            // Pop cb_max — HeldBulk left it unpopped
            cb_pop_front(cb_max, Ht);

            // Phase 3: Sum + Recip (WaitUpfrontNoPop — exp retained for Phase 4)
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_ROW,
                cb_exp,
                cb_scaler_sum,
                cb_recip_sum,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ckl::NoAccumulation,
                decltype(recip_op)>(
                reduce_block_shape,
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::NoAccumulation{},
                recip_op,
                partial_scaler);

            // Phase 4: Mul (broadcast recip_sum across W columns)
            ckl::mul<
                cb_exp,
                cb_recip_sum,
                cb_output_tiles,
                ckl::BroadcastDim::Col,
                ckl::InputLifecycle::Bulk,
                ckl::InputLifecycle::HeldBulk,
                ckl::OutputLifecycle::Streaming,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::PackTileReconfig::Output,
                ckl::OperandKind::Block,
                ckl::OperandKind::Col>(eltwise_shape);

            // Pop cb_recip_sum — HeldBulk left it unpopped
            cb_pop_front(cb_recip_sum, Ht);

        } else {
            // ===== dim=-2 (H reduction): REDUCE_COL, BroadcastDim::Row =====

            // Phase 1: Max reduce (WaitUpfrontNoPop — input retained for Phase 2)
            ckl::reduce<
                ckernel::PoolType::MAX,
                ckernel::ReduceDim::REDUCE_COL,
                cb_input_tiles,
                cb_scaler_max,
                cb_max,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ckl::NoAccumulation,
                ckl::NoOp>(
                reduce_block_shape,
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::NoAccumulation{},
                ckl::NoOp{},
                partial_scaler);

            // Phase 2: Sub + Exp (fused chain)
            ckl::eltwise_chain(
                eltwise_shape,
                ckl::BinaryFpu<
                    cb_input_tiles,
                    cb_max,
                    ckl::BinaryFpuOp::Sub,
                    ckl::BroadcastDim::Row,
                    ckl::InputLifecycle::Bulk,
                    ckl::InputLifecycle::HeldBulk,
                    ckl::BinaryDataFormatReconfig::Input,
                    ckl::Dst::D0,
                    ckl::OperandKind::Block,
                    ckl::OperandKind::Row>{},
                ckl::Exp<>{},
                ckl::PackTile<cb_exp, ckl::OutputLifecycle::Streaming>{});

            // Pop cb_max — HeldBulk left it unpopped (Wt tiles for REDUCE_COL)
            cb_pop_front(cb_max, Wt);

            // Phase 3: Sum + Recip (WaitUpfrontNoPop — exp retained for Phase 4)
            ckl::reduce<
                ckernel::PoolType::SUM,
                ckernel::ReduceDim::REDUCE_COL,
                cb_exp,
                cb_scaler_sum,
                cb_recip_sum,
                ckl::ReduceInputPolicy::WaitUpfrontNoPop,
                ckl::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
                ckl::NoAccumulation,
                decltype(recip_op)>(
                reduce_block_shape,
                ckl::ReduceInputMemoryLayout::contiguous(),
                ckl::NoAccumulation{},
                recip_op,
                partial_scaler);

            // Phase 4: Mul (broadcast recip_sum across H rows)
            ckl::mul<
                cb_exp,
                cb_recip_sum,
                cb_output_tiles,
                ckl::BroadcastDim::Row,
                ckl::InputLifecycle::Bulk,
                ckl::InputLifecycle::HeldBulk,
                ckl::OutputLifecycle::Streaming,
                ckl::BinaryDataFormatReconfig::Input,
                ckl::PackTileReconfig::Output,
                ckl::OperandKind::Block,
                ckl::OperandKind::Row>(eltwise_shape);

            // Pop cb_recip_sum — HeldBulk left it unpopped (Wt tiles for REDUCE_COL)
            cb_pop_front(cb_recip_sum, Wt);
        }

        // ===== RM path: untilize cb_output_tiles → cb_rm_out =====
        if constexpr (is_rm) {
            // Untilize Ht blocks of Wt tiles each.
            // InitAndUninit: full init on first call, uninit on last.
            // WaitBlock: wait for each block before processing.
            // NoReconfigure: same dtype on input and output.
            ckl::untilize<
                Wt,
                cb_output_tiles,
                cb_rm_out,
                ckl::untilize_config::InitUninitMode::InitAndUninit,
                ckl::untilize_config::WaitMode::WaitBlock,
                ckl::untilize_config::ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(Ht);
        }
    }
}
