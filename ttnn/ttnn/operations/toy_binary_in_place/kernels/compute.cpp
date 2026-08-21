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
// CB synchronization is expressed as an independent (wait, pop) pair per input and a
// (reserve, push) pair per output:
//   A operand : wait/pop per tile, front-relative (OperandKind::Scalar)
//   held B    : wait Upfront, then either pop AtEnd or never pop (PopPolicy::None)
//   output    : reserve/push per tile
// Broadcast B index: NONE->Block, ROW->Row, COL->Col, SCALAR->Scalar.

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/unary/misc.hpp"  // Square (SFPU unary)

namespace ckl = compute_kernel_lib;
using namespace compute_kernel_lib;

// op_code: 0=add, 1=sub, 2=mul. Picks the eltwise convenience func at compile time.
template <uint32_t op_code, ckl::InputSpec AInput, auto BInput, ckl::OutputSpec Output, ckl::IterationShapeKind Kind>
ALWI void binary_dispatch(ckl::TypedIterationShape<Kind> shape) {
    if constexpr (op_code == 0) {
        ckl::add<AInput, BInput, Output>(shape);
    } else if constexpr (op_code == 1) {
        ckl::sub<AInput, BInput, Output>(shape);
    } else {
        ckl::mul<AInput, BInput, Output>(shape);
    }
}

// In-place: cb_work = cb_work op cb_b. A (cb_work) is per-tile streamed, and the output
// writes back into cb_work (same buffer, reserve/push per tile). The B policy / index follow
// the original DISPATCH_IN_PLACE broadcast-specific policies.
template <uint32_t op_code, uint32_t bcast_code, uint32_t CbWork, uint32_t CbB, ckl::IterationShapeKind Kind>
ALWI void op_in_place(ckl::TypedIterationShape<Kind> shape) {
    constexpr auto a_input =
        ckl::input(CbWork, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::OperandKind::Scalar);
    constexpr auto out = ckl::output(CbWork);

    if constexpr (bcast_code == 0) {
        // NONE: B waits upfront and pops at end, B index Block.
        binary_dispatch<
            op_code,
            a_input,
            ckl::input(
                CbB, ckl::BroadcastDim::None, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block),
            out>(shape);
    } else if constexpr (bcast_code == 1) {
        // ROW: B waits upfront and is never popped by the chain, B index Row.
        binary_dispatch<
            op_code,
            a_input,
            ckl::input(
                CbB, ckl::BroadcastDim::Row, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Row),
            out>(shape);
    } else if constexpr (bcast_code == 2) {
        // COL: B waits upfront and pops at end, B index Col.
        binary_dispatch<
            op_code,
            a_input,
            ckl::input(
                CbB, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Col),
            out>(shape);
    } else {
        // SCALAR: B waits upfront and is never popped by the chain, B index Scalar.
        binary_dispatch<
            op_code,
            a_input,
            ckl::input(
                CbB,
                ckl::BroadcastDim::Scalar,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::None,
                ckl::OperandKind::Scalar),
            out>(shape);
    }
}

// Normal: cb_out = cb_input op cb_b. A (cb_input) always waits and pops per tile.
// B policy / index follow the original DISPATCH_NORMAL broadcast-specific policies.
template <
    uint32_t op_code,
    uint32_t bcast_code,
    uint32_t CbIn,
    uint32_t CbB,
    uint32_t CbOut,
    ckl::IterationShapeKind Kind>
ALWI void op_normal(ckl::TypedIterationShape<Kind> shape) {
    constexpr auto a_input =
        ckl::input(CbIn, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::OperandKind::Scalar);
    constexpr auto out = ckl::output(CbOut);

    if constexpr (bcast_code == 0) {
        // NONE: B waits and pops per tile, B index Scalar (per-tile, tile_b=0).
        binary_dispatch<
            op_code,
            a_input,
            ckl::input(
                CbB,
                ckl::BroadcastDim::None,
                ckl::WaitPolicy::PerTile,
                ckl::PopPolicy::PerTile,
                ckl::OperandKind::Scalar),
            out>(shape);
    } else if constexpr (bcast_code == 1) {
        // ROW: B waits upfront and is never popped by the chain, B index Row.
        binary_dispatch<
            op_code,
            a_input,
            ckl::input(
                CbB, ckl::BroadcastDim::Row, ckl::WaitPolicy::Upfront, ckl::PopPolicy::None, ckl::OperandKind::Row),
            out>(shape);
    } else if constexpr (bcast_code == 2) {
        // COL: B waits upfront and pops at end, B index Col. Consumes the same Ht tiles with
        // the same numerics as the original per-tile-wait / per-row-pop policy.
        binary_dispatch<
            op_code,
            a_input,
            ckl::input(
                CbB, ckl::BroadcastDim::Col, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Col),
            out>(shape);
    } else {
        // SCALAR: B waits upfront and is never popped by the chain, B index Scalar.
        binary_dispatch<
            op_code,
            a_input,
            ckl::input(
                CbB,
                ckl::BroadcastDim::Scalar,
                ckl::WaitPolicy::Upfront,
                ckl::PopPolicy::None,
                ckl::OperandKind::Scalar),
            out>(shape);
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
    constexpr auto shape = ckl::IterationShape::of(Ht, Wt);

    if constexpr (in_place_flag == 1) {
        // === IN-PLACE MODE ===
        compute_kernel_hw_startup(cb_input, cb_work);

        // Phase 1: Copy A tiles from cb_input → cb_work (per-tile streaming, no reconfig).
        ckl::copy<
            ckl::input(cb_input, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            ckl::output(
                cb_work, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>(
            ckl::IterationShape::tiles(total_a_tiles));

        // Phase 2: In-place op on cb_work (reconfig handles format transition)
        if constexpr (op_code == 4) {
            // SFPU SQUARE (in-place): copy to DEST, square_tile, pack back to cb_work.
            ckl::unary<
                ckl::Square<>,
                ckl::input(
                    cb_work, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::output(
                    cb_work, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>(
                shape);
        } else if constexpr (op_code == 3) {
            // FPU SQUARE: cb_work = cb_work * cb_work (binary MUL with same operand, in-place)
            ckl::square<ckl::input(cb_work), ckl::output(cb_work)>(shape);
        } else {
            op_in_place<op_code, bcast_code, cb_work, cb_b>(shape);
        }

        // Phase 3: Copy modified tiles from cb_work → cb_out (per-tile streaming, output reconfig).
        ckl::copy<
            ckl::input(cb_work, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
            ckl::output(cb_out)>(ckl::IterationShape::tiles(total_a_tiles));

    } else {
        // === NORMAL (NON-IN-PLACE) MODE ===
        compute_kernel_hw_startup(cb_input, cb_b, cb_out);

        if constexpr (op_code == 4) {
            // SFPU SQUARE (non-in-place): copy to DEST, square_tile, pack to cb_out.
            ckl::unary<
                ckl::Square<>,
                ckl::input(
                    cb_input, ckl::WaitPolicy::PerTile, ckl::PopPolicy::PerTile, ckl::DataFormatReconfig::Disabled),
                ckl::output(
                    cb_out, ckl::ReservePolicy::PerTile, ckl::PushPolicy::PerTile, ckl::DataFormatReconfig::Disabled)>(
                ckl::IterationShape::tiles(total_a_tiles));
        } else if constexpr (op_code == 3) {
            // FPU SQUARE: cb_out = cb_input * cb_input
            ckl::square<ckl::input(cb_input), ckl::output(cb_out)>(shape);
        } else {
            op_normal<op_code, bcast_code, cb_input, cb_b, cb_out>(shape);
        }
    }
}
