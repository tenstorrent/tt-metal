// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for toy_binary_in_place.
//
// Single full init (compute_kernel_hw_startup) at kernel start.
// Phase transitions use reconfig (inside the helpers) instead of
// a second full init.
//
// Supports: add(0), sub(1), mul(2), square(3), sfpu_square(4)
// Supports: in_place(1) and normal(0) modes
//
// All compute is expressed through compute_kernel_lib's eltwise convenience layer:
// add/sub/mul/square (FPU) -> eltwise_chain, the SFPU-square branch (op_code==4) ->
// unary<Square>, and the input/output copy phases -> copy<>. The op_code/bcast_code/
// in_place dispatch behavior is preserved exactly.
//
// binary_op_helpers BinaryInputPolicy -> eltwise InputLifecycle mapping used here:
//   WaitAndPopPerTile   -> Streaming      (A operand, per-tile front-relative; OperandKind::Scalar)
//   WaitUpfrontNoPop    -> HeldBulk       (held B, popped never)
//   WaitUpfrontPopAtEnd -> Bulk           (held B, popped at end)
// Broadcast B index: NONE->Block/Scalar (per-tile), ROW->Row, COL->Col, SCALAR->Scalar.

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"  // Square (SFPU unary)

namespace ckl = compute_kernel_lib;
using namespace compute_kernel_lib;

// op_code: 0=add, 1=sub, 2=mul. Picks the eltwise convenience func at compile time.
template <
    uint32_t op_code,
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    ckl::BroadcastDim Bcast,
    ckl::OperandKind AIdx,
    ckl::InputLifecycle ALife,
    ckl::InputLifecycle BLife,
    ckl::OperandKind BIdx,
    ckl::OutputLifecycle OutLife>
ALWI void binary_dispatch(ckl::EltwiseShape shape) {
    if constexpr (op_code == 0) {
        ckl::add<
            CbA,
            CbB,
            CbOut,
            Bcast,
            ALife,
            BLife,
            OutLife,
            ckl::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::PackTileReconfig::Output,
            AIdx,
            BIdx>(shape);
    } else if constexpr (op_code == 1) {
        ckl::sub<
            CbA,
            CbB,
            CbOut,
            Bcast,
            ALife,
            BLife,
            OutLife,
            ckl::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::PackTileReconfig::Output,
            AIdx,
            BIdx>(shape);
    } else {
        ckl::mul<
            CbA,
            CbB,
            CbOut,
            Bcast,
            ALife,
            BLife,
            OutLife,
            ckl::BinaryDataFormatReconfig::Input,
            compute_kernel_lib::PackTileReconfig::Output,
            AIdx,
            BIdx>(shape);
    }
}

// In-place: cb_work = cb_work op cb_b. A (cb_work) is per-tile streamed (Streaming/Scalar),
// output writes back into cb_work (OutputLifecycle::Streaming, CbOut == CbA). The B lifecycle
// / index follow the original DISPATCH_IN_PLACE broadcast-specific policies.
template <uint32_t op_code, uint32_t bcast_code, uint32_t CbWork, uint32_t CbB>
ALWI void op_in_place(ckl::EltwiseShape shape) {
    if constexpr (bcast_code == 0) {
        // NONE: B WaitUpfrontPopAtEnd -> Bulk, B index Block.
        binary_dispatch<
            op_code,
            CbWork,
            CbB,
            CbWork,
            ckl::BroadcastDim::None,
            ckl::OperandKind::Scalar,
            ckl::InputLifecycle::Streaming,
            ckl::InputLifecycle::Bulk,
            ckl::OperandKind::Block,
            ckl::OutputLifecycle::Streaming>(shape);
    } else if constexpr (bcast_code == 1) {
        // ROW: B WaitUpfrontNoPop -> HeldBulk, B index Row.
        binary_dispatch<
            op_code,
            CbWork,
            CbB,
            CbWork,
            ckl::BroadcastDim::Row,
            ckl::OperandKind::Scalar,
            ckl::InputLifecycle::Streaming,
            ckl::InputLifecycle::HeldBulk,
            ckl::OperandKind::Row,
            ckl::OutputLifecycle::Streaming>(shape);
    } else if constexpr (bcast_code == 2) {
        // COL: B WaitUpfrontPopAtEnd -> Bulk, B index Col.
        binary_dispatch<
            op_code,
            CbWork,
            CbB,
            CbWork,
            ckl::BroadcastDim::Col,
            ckl::OperandKind::Scalar,
            ckl::InputLifecycle::Streaming,
            ckl::InputLifecycle::Bulk,
            ckl::OperandKind::Col,
            ckl::OutputLifecycle::Streaming>(shape);
    } else {
        // SCALAR: B WaitUpfrontNoPop -> HeldBulk, B index Scalar.
        binary_dispatch<
            op_code,
            CbWork,
            CbB,
            CbWork,
            ckl::BroadcastDim::Scalar,
            ckl::OperandKind::Scalar,
            ckl::InputLifecycle::Streaming,
            ckl::InputLifecycle::HeldBulk,
            ckl::OperandKind::Scalar,
            ckl::OutputLifecycle::Streaming>(shape);
    }
}

// Normal: cb_out = cb_input op cb_b. A (cb_input) always WaitAndPopPerTile -> Streaming/Scalar.
// B lifecycle / index follow the original DISPATCH_NORMAL broadcast-specific policies.
template <uint32_t op_code, uint32_t bcast_code, uint32_t CbIn, uint32_t CbB, uint32_t CbOut>
ALWI void op_normal(ckl::EltwiseShape shape) {
    if constexpr (bcast_code == 0) {
        // NONE: B WaitAndPopPerTile -> Streaming, B index Scalar (per-tile, tile_b=0).
        binary_dispatch<
            op_code,
            CbIn,
            CbB,
            CbOut,
            ckl::BroadcastDim::None,
            ckl::OperandKind::Scalar,
            ckl::InputLifecycle::Streaming,
            ckl::InputLifecycle::Streaming,
            ckl::OperandKind::Scalar,
            ckl::OutputLifecycle::Streaming>(shape);
    } else if constexpr (bcast_code == 1) {
        // ROW: B WaitUpfrontNoPop -> HeldBulk, B index Row.
        binary_dispatch<
            op_code,
            CbIn,
            CbB,
            CbOut,
            ckl::BroadcastDim::Row,
            ckl::OperandKind::Scalar,
            ckl::InputLifecycle::Streaming,
            ckl::InputLifecycle::HeldBulk,
            ckl::OperandKind::Row,
            ckl::OutputLifecycle::Streaming>(shape);
    } else if constexpr (bcast_code == 2) {
        // COL: B WaitAndPopPerTile (waits 1/tile, pops 1/row) -> consumes Ht tiles like
        //   WaitUpfrontPopAtEnd -> Bulk, B index Col. Same Ht tiles consumed, same numerics.
        binary_dispatch<
            op_code,
            CbIn,
            CbB,
            CbOut,
            ckl::BroadcastDim::Col,
            ckl::OperandKind::Scalar,
            ckl::InputLifecycle::Streaming,
            ckl::InputLifecycle::Bulk,
            ckl::OperandKind::Col,
            ckl::OutputLifecycle::Streaming>(shape);
    } else {
        // SCALAR: B WaitUpfrontNoPop -> HeldBulk, B index Scalar.
        binary_dispatch<
            op_code,
            CbIn,
            CbB,
            CbOut,
            ckl::BroadcastDim::Scalar,
            ckl::OperandKind::Scalar,
            ckl::InputLifecycle::Streaming,
            ckl::InputLifecycle::HeldBulk,
            ckl::OperandKind::Scalar,
            ckl::OutputLifecycle::Streaming>(shape);
    }
}

void kernel_main() {
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t bcast_code = get_compile_time_arg_val(2);
    constexpr uint32_t in_place_flag = get_compile_time_arg_val(3);
    constexpr uint32_t op_code = get_compile_time_arg_val(4);

    constexpr uint32_t cb_input = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;
    constexpr uint32_t cb_work = tt::CBIndex::c_2;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t total_a_tiles = Ht * Wt;
    constexpr auto shape = ckl::EltwiseShape::of(Ht, Wt);

    if constexpr (in_place_flag == 1) {
        // === IN-PLACE MODE ===
        compute_kernel_hw_startup(cb_input, cb_work);

        // Phase 1: Copy A tiles from cb_input → cb_work (per-tile streaming, no reconfig).
        ckl::copy<
            cb_input,
            cb_work,
            ckl::InputLifecycle::Streaming,
            ckl::OutputLifecycle::Streaming,
            ckl::CopyTileReconfig::None,
            ckl::PackTileReconfig::None>(total_a_tiles);

        // Phase 2: In-place op on cb_work (reconfig handles format transition)
        if constexpr (op_code == 4) {
            // SFPU SQUARE (in-place): copy to DEST, square_tile, pack back to cb_work.
            ckl::unary<
                ckl::Square<>,
                cb_work,
                cb_work,
                ckl::InputLifecycle::Streaming,
                ckl::OutputLifecycle::Streaming,
                ckl::CopyTileReconfig::None,
                ckl::PackTileReconfig::None>(shape);
        } else if constexpr (op_code == 3) {
            // FPU SQUARE: cb_work = cb_work * cb_work (binary MUL with same operand, in-place)
            ckl::square<cb_work, cb_work>(shape);
        } else {
            op_in_place<op_code, bcast_code, cb_work, cb_b>(shape);
        }

        // Phase 3: Copy modified tiles from cb_work → cb_out (per-tile streaming, output reconfig).
        ckl::copy<
            cb_work,
            cb_out,
            ckl::InputLifecycle::Streaming,
            ckl::OutputLifecycle::Streaming,
            ckl::CopyTileReconfig::None>(total_a_tiles);

    } else {
        // === NORMAL (NON-IN-PLACE) MODE ===
        compute_kernel_hw_startup(cb_input, cb_b, cb_out);

        if constexpr (op_code == 4) {
            // SFPU SQUARE (non-in-place): copy to DEST, square_tile, pack to cb_out.
            ckl::unary<
                ckl::Square<>,
                cb_input,
                cb_out,
                ckl::InputLifecycle::Streaming,
                ckl::OutputLifecycle::Streaming,
                ckl::CopyTileReconfig::None,
                ckl::PackTileReconfig::None>(total_a_tiles);
        } else if constexpr (op_code == 3) {
            // FPU SQUARE: cb_out = cb_input * cb_input
            ckl::square<cb_input, cb_out>(shape);
        } else {
            op_normal<op_code, bcast_code, cb_input, cb_b, cb_out>(shape);
        }
    }
}
