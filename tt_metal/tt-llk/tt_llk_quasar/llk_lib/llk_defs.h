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

    // Keep the original Quasar values above stable.  The remaining selectors
    // complete the Blackhole SFPI operation surface used by the parity ports;
    // most are compile-time API tags (the Quasar init wrappers do not encode
    // their numeric value into hardware state).
    hardtanh,
    gelu_tanh,
    exp_with_base,
    power,
    tanh_derivative,
    log,
    log_with_base,
    log1p,
    gelu_derivative,
    dropout,
    abs_int32,
    sign,
    max,
    cosh,
    sinh,
    tan,
    cast_fp32_to_fp16a,
    sigmoid_appx,
    gelu_appx,
    elu,
    min,
    exp2,
    heaviside,
    expm1,
    signbit,
    asin,
    acos,
    atan,
    erf,
    erfc,
    isfinite,
    isinf,
    isposinf,
    isneginf,
    isnan,
    logical_not_unary,
    erfinv,
    i0,
    i1,
    mask,
    quant_int32,
    requant_int32,
    dequant_int32,
    add_int32,
    add_uint32,
    add_uint16,
    add1,
    sub_int32,
    sub_uint16,
    mul_uint16,
    mul_int32,
    div_int32,
    div_int32_floor,
    div_int32_trunc,
    remainder_int32,
    remainder_uint32,
    fmod_int32,
    eq_int,
    ne_int,
    eq,
    unary_ne,
    unary_eq,
    unary_gt,
    unary_lt,
    unary_ge,
    unary_le,
    tiled_prod,
    bitwise_xor,
    bitwise_not,
    bitwise_and,
    bitwise_or,
    right_shift,
    floor,
    trunc,
    frac,
    left_shift,
    remainder,
    fmod,
    ceil,
    isclose,
    reshuffle_rows,
    cumsum,
    prelu,
    alt_complex_rotate90,
    round,
    cpy_values,
    unary_max,
    unary_min,
    gcd,
    lcm,
    softshrink,
    tanhshrink,
    hardshrink,
    hardsigmoid,
    threshold,
    softsign,
    celu,
    max_pool_with_indices,
    selu,
    rpow,
    cbrt,
    reduce,
    add_top_row,
    rdiv,
    addcmul,
    max_int32,
    min_int32,
    max_uint32,
    min_uint32,
    unary_max_int32,
    unary_min_int32,
    unary_max_uint32,
    unary_min_uint32,
    addcdiv,
    lerp,
    snake_beta,
    xielu,
    lgamma,
    polygamma,
    mish,
    ne,
    digamma,
    identity,
    sqrt_custom,
    tanh_derivative_lut,
    rsqrt_compat,
    expm1_cw,
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
