// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// groupnorm_sc_N_1_HW_C — lane-form SFPU chain elements (Refinement 5).
//
// The per-group statistics live in LANE FORM: lane g of ROW 0 of a 32x32 tile (op_design.md -> lane form), rows
// 1..31 carry nothing the op ever reads (the finalize output is broadcast from row 0 by unary_bcast<Row>). The
// kernel_lib chain elements (MulUnary, AddUnary, Relu, Rsqrt, MulBinary, SubBinary) issue their SFPU kernel over the
// WHOLE tile — `VectorMode::RC` with the LLK's 8 iterations per face = 32 SFPU vector passes per op — and expose no
// vector-mode / iteration knob. On the latency floor that full-tile finalize measured 2.5-3 us of a 7 us op
// (Refinement 5 attribution, math-thread zone `c_finalize`), i.e. 16x the work the lane form needs.
//
// These elements are the same chain-element shape (`UnaryOp` / `BinaryOp` CRTP: static init + exec) and reuse the
// public `*_tile_init()`s, but issue the SFPU kernel through the LLK call macros with `VectorMode::R` (faces 0 and 1
// only) and LANE_ITERATIONS passes (the minimum that completes row 0, see below). Everything else the chain owns —
// DEST sync window, CB wait/pop/reserve/push, dtype reconfig — is untouched. This is a documented helper bypass: the
// helpers cannot express a row-only SFPU pass (no vector-mode parameter on the chain elements or the public
// `*_tile()` API); rows 1..31 of the output tiles are left as they were.
//
// Contract: only valid for tiles whose consumer reads row 0 exclusively (lane-form statistics).

#pragma once

#include <stdint.h>

#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/relu.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_binary_sfpu.h"

namespace groupnorm_lane_sfpu {

namespace ckl = compute_kernel_lib;

// SFPU passes per face under VectorMode::R (faces 0 and 1 only). Measured on Blackhole with a Float32 DEST
// (DEVICE_PRINT of the finalize output, Refinement 5): ONE pass covers rows 0..3 x the EVEN columns of a face (the
// 32-bit datums interleave in DEST), the second pass the odd columns — so 2 passes are the minimum that completes
// row 0 (all 32 lanes of the tile across the two faces). With a 16-bit DEST one pass covers rows 0..1 x all columns,
// so 2 passes complete row 0 there as well. 2 x 2 = 4 passes vs the full-tile 4 x 8 = 32.
constexpr int LANE_ITERATIONS = 2;

#define GROUPNORM_LANE_UNARY_CALL(FN, TEMPLATES, IDST, ...) \
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, FN, TEMPLATES, IDST, ckernel::VectorMode::R, ##__VA_ARGS__))
#define GROUPNORM_LANE_BINARY_CALL(FN, TEMPLATES, IN0, IN1, OUT) \
    MATH((SFPU_BINARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, FN, TEMPLATES, IN0, IN1, OUT, ckernel::VectorMode::R)))

// x *= scalar (row 0)
template <ckl::Dst Slot>
struct LaneMulUnary : ckl::UnaryOp<LaneMulUnary<Slot>, Slot> {
    uint32_t param0;
    constexpr explicit LaneMulUnary(uint32_t p) noexcept : param0(p) {}
    constexpr LaneMulUnary() noexcept : param0(0) {}
    static ALWI void init() { binop_with_scalar_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        GROUPNORM_LANE_UNARY_CALL(
            calculate_binop_with_scalar,
            (APPROX, ckernel::MUL_UNARY, LANE_ITERATIONS, DST_ACCUM_MODE),
            ckl::to_u32(Slot) + slot_offset,
            param0);
    }
};

// x += scalar (row 0)
template <ckl::Dst Slot>
struct LaneAddUnary : ckl::UnaryOp<LaneAddUnary<Slot>, Slot> {
    uint32_t param0;
    constexpr explicit LaneAddUnary(uint32_t p) noexcept : param0(p) {}
    constexpr LaneAddUnary() noexcept : param0(0) {}
    static ALWI void init() { binop_with_scalar_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        GROUPNORM_LANE_UNARY_CALL(
            calculate_binop_with_scalar,
            (APPROX, ckernel::ADD_UNARY, LANE_ITERATIONS, DST_ACCUM_MODE),
            ckl::to_u32(Slot) + slot_offset,
            param0);
    }
};

// x = max(x, 0) (row 0)
template <ckl::Dst Slot>
struct LaneRelu : ckl::UnaryOp<LaneRelu<Slot>, Slot> {
    static ALWI void init() { relu_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        GROUPNORM_LANE_UNARY_CALL(
            _relu_min_,
            (sfpi::vFloat, APPROX, LANE_ITERATIONS, uint32_t),
            ckl::to_u32(Slot) + slot_offset,
            0u /*threshold*/);
    }
};

// x = rsqrt(x), exact (non-approximate, non-legacy) (row 0)
template <ckl::Dst Slot>
struct LaneRsqrt : ckl::UnaryOp<LaneRsqrt<Slot>, Slot> {
    static ALWI void init() { rsqrt_tile_init<false>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        GROUPNORM_LANE_UNARY_CALL(
            calculate_rsqrt,
            (APPROX, LANE_ITERATIONS, DST_ACCUM_MODE, false /*FAST_APPROX*/, false /*legacy_compat*/),
            ckl::to_u32(Slot) + slot_offset);
    }
};

// out = in0 * in1 (row 0)
template <ckl::Dst In0, ckl::Dst In1, ckl::Dst Out>
struct LaneMulBinary : ckl::BinaryOp<LaneMulBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { mul_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        GROUPNORM_LANE_BINARY_CALL(
            calculate_sfpu_binary_mul,
            (APPROX, ckernel::BinaryOp::MUL, LANE_ITERATIONS, DST_ACCUM_MODE),
            ckl::to_u32(In0) + slot_offset,
            ckl::to_u32(In1) + slot_offset,
            ckl::to_u32(Out) + slot_offset);
    }
};

// out = in0 - in1 (row 0)
template <ckl::Dst In0, ckl::Dst In1, ckl::Dst Out>
struct LaneSubBinary : ckl::BinaryOp<LaneSubBinary<In0, In1, Out>, In0, In1, Out> {
    static ALWI void init() { sub_binary_tile_init(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        GROUPNORM_LANE_BINARY_CALL(
            calculate_sfpu_binary,
            (APPROX, ckernel::BinaryOp::SUB, LANE_ITERATIONS, DST_ACCUM_MODE, ckernel::DstRoundingMode::Default),
            ckl::to_u32(In0) + slot_offset,
            ckl::to_u32(In1) + slot_offset,
            ckl::to_u32(Out) + slot_offset);
    }
};

#undef GROUPNORM_LANE_UNARY_CALL
#undef GROUPNORM_LANE_BINARY_CALL

}  // namespace groupnorm_lane_sfpu
