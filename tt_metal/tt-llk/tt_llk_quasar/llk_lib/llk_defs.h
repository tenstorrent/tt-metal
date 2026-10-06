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
    MIN,
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
    floor,
    ceil,
    trunc,
    frac,
    round,
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
    cumsum,
    reduce,
    add1,
    hardsigmoid,
    celu,
    elu,
    hardmish,
    mish,
    hardshrink,
    hardtanh,
    heaviside,
    prelu,
    selu,
    sigmoid_appx,
    softshrink,
    softsign,
    tanhshrink,
    threshold,
    xielu,
    cbrt,
    exp2,
    expm1,
    rpow,
    sign,
    power,
    power_iterative,
    log,
    digamma,
    erf,
    erfc,
    erfinv,
    i0,
    i1,
    lgamma,
    polygamma,
    isinf,
    isposinf,
    isneginf,
    isnan,
    isfinite,
    logical_not_unary,
    unary_gt,
    unary_lt,
    unary_ge,
    unary_le,
    unary_eq,
    unary_ne,
    bitwise_not,
    left_shift,
    right_shift,
    bitwise_and,
    bitwise_or,
    bitwise_xor,
    rsub_scalar_int32,
    fmod,
    remainder,
    remainder_uint32,
    rdiv,
    sum_int_col,
    sum_int_row,
    tiled_prod,
    identity,
    cast_fp32_to_fp16a,
    alt_complex_rotate90,
    softcap,
    tanh_derivative,
    abs_int32,
    // Named by compute API init macros for the ported SFPI kernels; Quasar's init wrappers ignore
    // the SfpuType, so these only need to exist.
    acos,
    add_top_row,
    addcdiv,
    addcmul,
    asin,
    atan,
    cosh,
    div_int32,
    div_int32_floor,
    div_int32_trunc,
    fmod_int32,
    isclose,
    lerp,
    log_with_base,
    mac,
    mask,
    remainder_int32,
    sinh,
    situ_glu,
    snake_beta,
    tan,
    max_pool_with_indices,
};

// Load/store layout selectors shared with the WH/BH SFPI kernels: calculate_logical_not picks its
// sfpi::DataLayout from DEFAULT / LO16 / INT32. The numbers are the Blackhole SFPLOAD instr_mod0
// codes, kept so the names mean the same on every arch; Quasar's own SFPLOAD/SFPSTORE encodings are
// p_sfpu::sfpmem, so never pack these values into a Quasar instruction word.
enum class InstrModLoadStore
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
    HI16_ONLY     = 15
};

// The integer layouts the WH/BH SFPI integer kernels accept (e.g. calculate_rsub_int), as on Blackhole.
inline constexpr bool is_valid_instruction_mode(InstrModLoadStore mode)
{
    return mode == InstrModLoadStore::INT32_2S_COMP || mode == InstrModLoadStore::INT32 || mode == InstrModLoadStore::LO16;
}

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
