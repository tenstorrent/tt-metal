// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "ttnn/cpp/ttnn/kernel_lib/sfpu_chain.hpp"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"

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

}  // namespace compute_kernel_lib

#include "sfpu_binary.inl"
