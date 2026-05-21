// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Compute kernel for toy_binary_in_place — migrated to the eltwise helper
// surface (eltwise_chain + eltwise_convenience + eltwise_misc::Square).
//
// One compute_kernel_hw_startup at MAIN entry; each chain call emits its own
// per-element reconfigs via the prev-CB fold.
//
// Supports: add(0), sub(1), mul(2), square(3 — FPU mul same-CB), sfpu_square(4)
// Supports: in_place(1) and normal(0) modes, with NONE / ROW / COL / SCALAR broadcast.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_helpers.hpp"

namespace ckl = compute_kernel_lib;

namespace {

template <uint32_t Code>
struct BcastFor;
template <>
struct BcastFor<0> {
    static constexpr auto value = ckl::BroadcastDim::None;
};
template <>
struct BcastFor<1> {
    static constexpr auto value = ckl::BroadcastDim::Row;
};
template <>
struct BcastFor<2> {
    static constexpr auto value = ckl::BroadcastDim::Col;
};
template <>
struct BcastFor<3> {
    static constexpr auto value = ckl::BroadcastDim::Scalar;
};

// B-side operand kind + lifecycle per broadcast.
//   NONE   — B is per-tile streamed (cb_b sized Ht*Wt+1, reader pushes Ht*Wt).
//   ROW    — B is Wt tiles, read by `wt` each iter — wait upfront, pop at end.
//   COL    — B is Ht tiles, read by `ht` each iter — wait upfront, pop at end.
//   SCALAR — B is 1 tile, read at index 0 — wait upfront, pop at end.
template <ckl::BroadcastDim D>
struct BSide;
template <>
struct BSide<ckl::BroadcastDim::None> {
    static constexpr auto kind = ckl::OperandKind::Scalar;  // FirstTile (per-tile pop chase)
    static constexpr auto policy = ckl::Streaming;
};
template <>
struct BSide<ckl::BroadcastDim::Row> {
    static constexpr auto kind = ckl::OperandKind::Row;
    static constexpr auto policy = ckl::Bulk;
};
template <>
struct BSide<ckl::BroadcastDim::Col> {
    static constexpr auto kind = ckl::OperandKind::Col;
    static constexpr auto policy = ckl::Bulk;
};
template <>
struct BSide<ckl::BroadcastDim::Scalar> {
    static constexpr auto kind = ckl::OperandKind::Scalar;
    static constexpr auto policy = ckl::Bulk;
};

template <uint32_t Code>
struct FpuOpFor;
template <>
struct FpuOpFor<0> {
    static constexpr auto value = ckl::BinaryFpuOp::Add;
};
template <>
struct FpuOpFor<1> {
    static constexpr auto value = ckl::BinaryFpuOp::Sub;
};
template <>
struct FpuOpFor<2> {
    static constexpr auto value = ckl::BinaryFpuOp::Mul;
};

template <
    uint32_t CbA,
    uint32_t CbB,
    uint32_t CbOut,
    ckl::BinaryFpuOp Op,
    ckl::BroadcastDim Bcast,
    ckl::InputLifecycle BPolicy,
    ckl::OperandKind BIndex>
ALWI void run_binary(ckl::EltwiseShape shape) {
    ckl::eltwise_chain(
        shape,
        ckl::BinaryFpu<
            CbA,
            CbB,
            Op,
            Bcast,
            ckl::BinaryDataFormatReconfig::Input,
            ckl::Streaming,            // A: per-tile pop chase (in-place safe)
            BPolicy,                   // B: depends on broadcast
            ckl::OperandKind::Scalar,  // AIndex = FirstTile (Streaming requires it)
            ckl::Dst::D0,
            BIndex>{},
        ckl::PackTile<CbOut, ckl::Dst::D0, ckl::OutStreaming>{});
}

// FPU square: x*x via same-CB BinaryFpu (chain dedups B-side wait/pop when CbA == CbB).
template <uint32_t Cb, uint32_t CbOut>
ALWI void run_fpu_square(ckl::EltwiseShape shape) {
    ckl::eltwise_chain(
        shape,
        ckl::BinaryFpu<
            Cb,
            Cb,
            ckl::BinaryFpuOp::Mul,
            ckl::BroadcastDim::None,
            ckl::BinaryDataFormatReconfig::Input,
            ckl::Streaming,
            ckl::Streaming,
            ckl::OperandKind::Scalar,
            ckl::Dst::D0,
            ckl::OperandKind::Scalar>{},
        ckl::PackTile<CbOut, ckl::Dst::D0, ckl::OutStreaming>{});
}

}  // namespace

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
    constexpr auto Bcast = BcastFor<bcast_code>::value;
    constexpr auto BKind = BSide<Bcast>::kind;
    constexpr auto BPol = BSide<Bcast>::policy;

    if constexpr (in_place_flag == 1) {
        // === IN-PLACE MODE ===
        // Boot once covering Phase 1 (cb_input → cb_work) — Phase 2/3 reconfigs
        // are emitted by the chain's prev-CB fold at each chain call's entry.
        compute_kernel_hw_startup(cb_input, cb_b, cb_work);

        // Phase 1: stream A into the working CB.
        ckl::copy<cb_input, cb_work>(total_a_tiles);

        // Phase 2: in-place transform on cb_work.
        if constexpr (op_code == 4) {
            // SFPU SQUARE: copy → square → pack, all on cb_work (pop-chase in-place).
            ckl::unary<ckl::Square<>, cb_work, cb_work>(total_a_tiles);
        } else if constexpr (op_code == 3) {
            // FPU SQUARE: same-CB BinaryFpu mul, packs back to cb_work.
            run_fpu_square<cb_work, cb_work>(shape);
        } else {
            constexpr auto Op = FpuOpFor<op_code>::value;
            run_binary<cb_work, cb_b, cb_work, Op, Bcast, BPol, BKind>(shape);
        }

        // Phase 3: drain cb_work → cb_out.
        ckl::copy<cb_work, cb_out>(total_a_tiles);

    } else {
        // === NORMAL (NON-IN-PLACE) MODE ===
        compute_kernel_hw_startup(cb_input, cb_b, cb_out);

        if constexpr (op_code == 4) {
            ckl::unary<ckl::Square<>, cb_input, cb_out>(total_a_tiles);
        } else if constexpr (op_code == 3) {
            run_fpu_square<cb_input, cb_out>(shape);
        } else {
            constexpr auto Op = FpuOpFor<op_code>::value;
            run_binary<cb_input, cb_b, cb_out, Op, Bcast, BPol, BKind>(shape);
        }
    }
}
