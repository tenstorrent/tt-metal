// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_chain.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/mask.h"

namespace compute_kernel_lib {

// --- Binary SFPU Ops ---

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuAdd : BinaryOp<SfpuAdd<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuSub : BinaryOp<SfpuSub<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuMul : BinaryOp<SfpuMul<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuDiv : BinaryOp<SfpuDiv<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuRsub : BinaryOp<SfpuRsub<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuPow : BinaryOp<SfpuPow<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuEq : BinaryOp<SfpuEq<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuMax : BinaryOp<SfpuMax<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

template <Dst In0 = Dst::D0, Dst In1 = Dst::D1, Dst Out = Dst::D0>
struct SfpuMin : BinaryOp<SfpuMin<In0, In1, Out>, In0, In1, Out> {
    ALWI void init() const;
    ALWI void call(uint32_t a, uint32_t b, uint32_t c) const;
};

/**
 * @brief Mask-with-zero SFPU op: zeros data elements where mask is zero.
 *
 * Reads data from DataSlot, mask from DataSlot + 1, writes the masked result
 * back to DataSlot (in-place). The underlying LLK ignores any explicit mask-slot
 * argument and unconditionally uses DataSlot + 1 — this struct honours that
 * contract by binding In1 = DataSlot + 1 at compile time.
 *
 * Typical chain usage (softmax / moreh_*):
 *   sfpu_chain(
 *       Load<cb_data, Dst::D0>{},
 *       Load<cb_mask, Dst::D1>{},
 *       Mask<DataFormat::Float16_b>{});
 *
 * Supported DataFormat values: Float16, Float16_b, Int32 (per `mask.h`).
 *
 * @tparam DF        Data-tile format consumed by `mask_tile` (default Float16_b)
 * @tparam DataSlot  DEST slot holding the data; mask is implicitly at DataSlot+1
 */
template <DataFormat DF = DataFormat::Float16_b, Dst DataSlot = Dst::D0>
struct Mask : BinaryOp<Mask<DF, DataSlot>, DataSlot, static_cast<Dst>(static_cast<uint32_t>(DataSlot) + 1), DataSlot> {
    static_assert(static_cast<uint32_t>(DataSlot) < 7, "Mask requires DataSlot + 1 < 8; place data at D0..D6.");
    ALWI void init() const;
    ALWI void call(uint32_t data, uint32_t mask, uint32_t out) const;
};

/**
 * @brief Mask-with-+inf SFPU op: sets data elements to +inf where mask is zero.
 *
 * Same DEST-layout contract as Mask (data at DataSlot, mask at DataSlot + 1,
 * in-place output). The LLK does not take a DataFormat argument — its behaviour
 * is governed by the data tile's configured format, so no DF template param is
 * exposed.
 *
 * @tparam DataSlot  DEST slot holding the data; mask is implicitly at DataSlot+1
 */
template <Dst DataSlot = Dst::D0>
struct MaskPosInf
    : BinaryOp<MaskPosInf<DataSlot>, DataSlot, static_cast<Dst>(static_cast<uint32_t>(DataSlot) + 1), DataSlot> {
    static_assert(static_cast<uint32_t>(DataSlot) < 7, "MaskPosInf requires DataSlot + 1 < 8; place data at D0..D6.");
    ALWI void init() const;
    ALWI void call(uint32_t data, uint32_t mask, uint32_t out) const;
};

}  // namespace compute_kernel_lib

#include "sfpu_binary.inl"
