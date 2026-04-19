// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ckernel
{

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
    relumin,
    relumax,
    stochround,
    typecast,
    add,
    square,
    sigmoid,
    silu,
    fill
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

} // namespace ckernel

// Make SfpuType available in global namespace for compatibility with test infrastructure
using SfpuType = ckernel::SfpuType;
