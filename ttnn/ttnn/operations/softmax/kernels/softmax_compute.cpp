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
// All compute phases use kernel-lib helpers. The two cb_pop_front calls
// are CB maintenance (freeing HeldBulk intermediates between phases),
// not compute phases.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/recip.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

namespace {
// CB indices — must match program descriptor
constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_scaler_max = 1;
constexpr uint32_t cb_scaler_sum = 2;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_max = 24;
constexpr uint32_t cb_exp = 25;
constexpr uint32_t cb_recip_sum = 26;
}  // namespace

namespace ckl = compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    // dim is passed as uint32_t (two's complement); cast to int32_t to recover -1 or -2
    constexpr int32_t dim = static_cast<int32_t>(get_compile_time_arg_val(2));

    uint32_t num_slabs = get_arg_val<uint32_t>(0);  // slabs assigned to this core

    // compute_kernel_hw_startup configures SrcA=input, SrcB=scaler, Pack=intermediate
    compute_kernel_hw_startup(cb_input_tiles, cb_scaler_max, cb_max);

    // PostReduceOp: reciprocal after sum reduce
    auto recip_op = [](uint32_t dst_idx) {
        recip_tile_init();
        recip_tile(dst_idx);
    };

    constexpr auto reduce_block_shape = ckl::ReduceInputBlockShape::of(Ht, Wt, 1);
    constexpr auto eltwise_shape = ckl::EltwiseShape::grid(Ht, Wt);

    for (uint32_t slab = 0; slab < num_slabs; ++slab) {
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
                ckl::NoOp>(reduce_block_shape);

            // Phase 2: Sub + Exp (fused chain)
            //   BinaryFpu(Sub, Col): cb_input_tiles(Bulk, pops at end) - cb_max(HeldBulk, no pop) → D0
            //   Exp: D0
            //   PackTile: cb_exp (Streaming)
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
                reduce_block_shape, ckl::ReduceInputMemoryLayout::contiguous(), ckl::NoAccumulation{}, recip_op);

            // Phase 4: Mul (broadcast recip_sum across W columns)
            //   cb_exp(Bulk, pops at end) * cb_recip_sum(HeldBulk, no pop) → cb_output_tiles(Streaming)
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
                ckl::NoOp>(reduce_block_shape);

            // Phase 2: Sub + Exp (fused chain)
            //   BinaryFpu(Sub, Row): cb_input_tiles(Bulk) - cb_max(HeldBulk, Row) → D0
            //   Exp → PackTile(cb_exp, Streaming)
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
                reduce_block_shape, ckl::ReduceInputMemoryLayout::contiguous(), ckl::NoAccumulation{}, recip_op);

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
    }
}
