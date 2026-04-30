// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/eltwise_binary_sfpu.h"

#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"

/**
 * @file eltwise_binary.hpp
 * @brief V2 binary op structs.
 *
 * Two layers:
 *   1. SFPU binary tile structs (in-DEST) — `MulBinary`, `AddBinary`, etc.
 *      They consume two DEST slots and produce a DEST slot output. Used inside
 *      eltwise_chain alongside CopyTile / SFPU unary structs.
 *   2. (TODO) FPU binary `binary_op<>` free function — wraps add_tiles /
 *      sub_tiles / mul_tiles with all policies + broadcast + activation. Lives
 *      here when implemented.
 *
 * NOTE: does NOT include legacy `binary_op_helpers.hpp`. Calls go directly into
 * `compute_kernel_api/eltwise_binary_sfpu.h`.
 */

namespace compute_kernel_lib {

// =============================================================================
// MulBinary — SFPU binary tile op. Reads idst0 * idst1 -> odst (in DEST).
// Init touches the SFPU binop dispatcher; safe with non-LUT ops.
// =============================================================================

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct MulBinary : BinaryOp<MulBinary<In0, In1, Out>, In0, In1, Out> {
    static constexpr bool clobbers_sfpu_lut = false;

    ALWI static void init() { ckernel::mul_binary_tile_init(); }
    ALWI static void call(uint32_t i0, uint32_t i1, uint32_t out_idx) { ckernel::mul_binary_tile(i0, i1, out_idx); }
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct AddBinary : BinaryOp<AddBinary<In0, In1, Out>, In0, In1, Out> {
    static constexpr bool clobbers_sfpu_lut = false;

    ALWI static void init() { ckernel::add_binary_tile_init(); }
    ALWI static void call(uint32_t i0, uint32_t i1, uint32_t out_idx) { ckernel::add_binary_tile(i0, i1, out_idx); }
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SubBinary : BinaryOp<SubBinary<In0, In1, Out>, In0, In1, Out> {
    static constexpr bool clobbers_sfpu_lut = false;

    ALWI static void init() { ckernel::sub_binary_tile_init(); }
    ALWI static void call(uint32_t i0, uint32_t i1, uint32_t out_idx) { ckernel::sub_binary_tile(i0, i1, out_idx); }
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct DivBinary : BinaryOp<DivBinary<In0, In1, Out>, In0, In1, Out> {
    static constexpr bool clobbers_sfpu_lut = false;

    ALWI static void init() { ckernel::div_binary_tile_init(); }
    ALWI static void call(uint32_t i0, uint32_t i1, uint32_t out_idx) { ckernel::div_binary_tile(i0, i1, out_idx); }
};

}  // namespace compute_kernel_lib
