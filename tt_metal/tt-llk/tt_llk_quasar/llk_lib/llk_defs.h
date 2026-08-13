// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ckernel
{

// Currently unused but kept for backwards compatibility
enum class VectorMode : std::uint8_t
{
    None      = 0,
    R         = 1,
    C         = 2,
    RC        = 4,
    RC_custom = 6,
    Invalid   = 0xFF,
};

enum class ReduceDim : std::uint8_t
{
    REDUCE_ROW,
    REDUCE_COL,
    REDUCE_SCALAR,
};

enum class PoolType : std::uint8_t
{
    SUM,
    AVG,
    MAX,
};

enum class DataCopyType : std::uint8_t
{
    A2D,
    B2D,
};

enum class EltwiseBinaryType : std::uint8_t
{
    ELWMUL,
    ELWADD,
    ELWSUB,
};

enum class EltwiseBinaryReuseDestType
{
    NONE         = 0,
    DEST_TO_SRCA = 1,
    DEST_TO_SRCB = 2,
};

// Logical source/destination layout selected by integer SFPI APIs. Quasar's
// kernels map these modes to sfpi::DataLayout rather than encoding them into
// SFPLOAD/SFPSTORE instruction fields as Blackhole does.
enum class InstrModLoadStore : std::uint8_t
{
    DEFAULT       = 0,
    FP16A         = 1,
    FP16B         = 2,
    FP32          = 3,
    INT32         = 4,
    INT8          = 5,
    LO16          = 6,
    HI16          = 7,
    INT32_2S_COMP = 12,
    INT8_2S_COMP  = 13,
    LO16_ONLY     = 14,
    HI16_ONLY     = 15,
};

// Broadcasts only occur on SrcB
enum class BroadcastType : std::uint8_t
{
    NONE,
    COL,
    ROW,
    SCALAR,
};

enum class Transpose : std::uint8_t
{
    None      = 0,
    IntraFace = 1,
    InterFace = 2,
    Both      = 3,
};

enum class SfpuType : std::uint32_t
{
    tanh,
    gelu,
    exponential,
    reciprocal,
    sqrt,
    rsqrt,
    relu,
    lrelu,
    relu_min,
    relu_max,
    stochround,
    typecast,
    add,
    square,
    sigmoid,
    silu,
    abs,
    clamp,
    negative,
    softplus,
    sine,
    cosine,
    acosh,
    asinh,
    atanh,
    fill,
    swiglu,
    where,
    unused,
    lt,
    gt,
    le,
    ge,
    lt_int,
    gt_int,
    le_int,
    ge_int,
    mul_int,
    topk_local_sort,
    topk_merge,
    topk_rebuild,
    equal_zero,
    not_equal_zero,
    less_than_zero,
    greater_than_zero,
    less_than_equal_zero,
    greater_than_equal_zero,
    // Portable-SFPI operations ported from Blackhole. Keep these append-only:
    // SfpuType values are consumed by generated kernels and test dispatch.
    hardtanh,
    cast_fp32_to_fp16a,
    sign,
    exp2,
    heaviside,
    expm1,
    erf,
    erfc,
    erfinv,
    i0,
    i1,
    logical_not_unary,
    add1,
    div_int32,
    div_int32_floor,
    unary_ne,
    unary_eq,
    unary_gt,
    unary_lt,
    unary_ge,
    unary_le,
    tiled_prod,
    bitwise_not,
    right_shift,
    left_shift,
    remainder,
    fmod,
    isclose,
    prelu,
    alt_complex_rotate90,
    softshrink,
    tanhshrink,
    hardshrink,
    hardsigmoid,
    softsign,
    celu,
    elu,
    selu,
    rpow,
    cbrt,
    hardmish,
    rdiv,
    addcmul,
    addcdiv,
    lerp,
    snake_beta,
    xielu,
    lgamma,
    polygamma,
    digamma,
    identity,
    power,
    int_sum_col,
    int_sum_row,
    unary_bitwise_and,
    unary_bitwise_or,
    unary_bitwise_xor,
    mask,
    int_mask,
};

enum class DstSync : std::uint8_t
{
    SyncHalf,
    SyncFull,
};

enum class MathFidelity : std::uint8_t
{
    LoFi  = 0,
    HiFi2 = 2,
    HiFi3 = 3,
    HiFi4 = 4
};

enum class StochRndType : std::uint8_t
{
    None = 0,
    Fpu  = 1,
    Pack = 2,
    All  = 3,
};

enum class PackMode : std::uint8_t
{
    Default  = 0,
    Untilize = 1,
    Tilize   = 2,
};

// Packer ReLU modes; encoding matches RELU_MODE (2 bits) in HW.
enum class ReluType : std::uint8_t
{
    NO_RELU = 0,
    ZERO_RELU,
    MIN_THRESHOLD_RELU,
    MAX_THRESHOLD_RELU,
};

/** Packer ReLU config: mode + 16-bit threshold (bits 16–31 in HW). */
struct ReluConfig
{
    static constexpr ReluConfig none()
    {
        return {ReluType::NO_RELU};
    }

    static constexpr ReluConfig zero()
    {
        return {ReluType::ZERO_RELU};
    }

    static constexpr ReluConfig min_threshold(std::uint32_t t)
    {
        return {ReluType::MIN_THRESHOLD_RELU, t};
    }

    static constexpr ReluConfig max_threshold(std::uint32_t t)
    {
        return {ReluType::MAX_THRESHOLD_RELU, t};
    }

    static constexpr ReluConfig from_packed(std::uint32_t packed)
    {
        return {static_cast<ReluType>(packed & 0x3), (packed >> 16) & 0xFFFF};
    }

    constexpr ReluType get_mode() const
    {
        return mode;
    }

    constexpr std::uint32_t get_threshold() const
    {
        return threshold;
    }

private:
    constexpr ReluConfig(ReluType m, std::uint32_t t = 0) : mode(m), threshold(t)
    {
    }

    ReluType mode           = ReluType::NO_RELU;
    std::uint32_t threshold = 0;
};

constexpr std::uint32_t SFPU_ITERATIONS = 8; // Number of iterations to unroll for SFPU loops

} // namespace ckernel

// Make SfpuType available in global namespace for compatibility with test infrastructure
using SfpuType = ckernel::SfpuType;
