// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "tensix_types.h"

namespace ckernel
{

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
    ELWDIV,
    ELWADD,
    ELWSUB,
    ELWLESS,
};

enum class EltwiseBinaryReuseDestType
{
    NONE         = 0,
    DEST_TO_SRCA = 1,
    DEST_TO_SRCB = 2,
};

enum class DstSync
{
    SyncHalf = 0,
    SyncFull = 1,
};

enum class BroadcastType : std::uint8_t
{
    NONE   = 0x0, // A - None || B - None
    COL    = 0x1, // A - None || B - Col Broadcast
    ROW    = 0x2, // A - None || B - Row Broadcast
    SCALAR = 0x3, // A - None || B - Scalar Broadcast
};

enum class Transpose : std::uint8_t
{
    None      = 0,
    IntraFace = 1,
    InterFace = 2,
    Both      = 3,
};

// Packer ReLU modes; encoding matches RELU_MODE (2 bits) in HW.
enum class ReluType : std::uint8_t
{
    NO_RELU = 0,
    ZERO_RELU,
    MIN_THRESHOLD_RELU,
    MAX_THRESHOLD_RELU,
};

/** Packer ReLU config: mode + 16-bit threshold (bits 16-31 in HW). */
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

enum class MathFidelity : std::uint8_t
{
    LoFi  = 0,
    HiFi2 = 2,
    HiFi3 = 3,
    HiFi4 = 4
};

constexpr bool UnpackToDestEn  = true;
constexpr bool UnpackToDestDis = false;

/*
Stochastic rounding modes:
    None: No stochastic rounding enabled, default rounding is round to nearest even.
    Fpu: Enables stochastic rounding for every accumulation in the fpu
    Pack: Enables stochastic rounding in both gasket and packer. Gasket rounding is in
    data format conversion stage from dest format to pack_src_format. Packer rounding
    is in data format conversion stage from pack_src_format to pack_dst_format.
    All: Enables fpu, pack and gasket rounding.
*/
enum class StochRndType : std::uint8_t
{
    None = 0,
    Fpu  = 1,
    Pack = 2,
    All  = 0xf,
};

// This is populated per Blackhole ISA for SFPLOAD/SFPSTORE instructions.
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

/**
 * @brief Left-shifts the numeric value of an InstrModLoadStore instruction mode.
 *
 * Provides an integer left-shift for the scoped InstrModLoadStore enum so its value can be packed
 * directly into SFPLOAD/SFPSTORE instruction words (e.g. the `(instr_mod0) << 16` field encoding).
 *
 * @param mod   The instruction mode whose underlying value is shifted.
 * @param shift The number of bit positions to shift left.
 * @return The underlying value of @p mod shifted left by @p shift, as a std::uint32_t.
 */
constexpr std::uint32_t operator<<(InstrModLoadStore mod, int shift)
{
    return static_cast<std::uint32_t>(mod) << shift;
}

/**
 * @brief Bitwise-ORs an integer with the numeric value of an InstrModLoadStore instruction mode.
 *
 * Provides bitwise-OR for the scoped InstrModLoadStore enum so its value can be packed directly
 * into the immediate field of SFPCONFIG-style instruction words (e.g. `0x310 | InstrModLoadStore::FP16B`).
 *
 * @param bits The integer bits to OR with.
 * @param mod  The instruction mode whose underlying value is OR-ed in.
 * @return @p bits bitwise-OR the underlying value of @p mod, as a std::uint32_t.
 */
constexpr std::uint32_t operator|(std::uint32_t bits, InstrModLoadStore mod)
{
    return bits | static_cast<std::uint32_t>(mod);
}

constexpr std::uint32_t operator|(InstrModLoadStore mod, std::uint32_t bits)
{
    return operator|(bits, mod);
}

template <DataFormat format, bool is_fp32_dest_acc_en = false>
constexpr InstrModLoadStore GetSfpLoadStoreInstrMod()
{
    switch (format)
    {
        case DataFormat::Float32:
            return InstrModLoadStore::FP32; // spec value 3: fp32 format
        case DataFormat::Float16:
            // With 32-bit (fp32) dest accumulation the datum is stored as full fp32 in the dest word, so
            // SFPLOAD/SFPSTORE must use FP32 access; otherwise the narrow fp16_a format applies.
            return is_fp32_dest_acc_en ? InstrModLoadStore::FP32 : InstrModLoadStore::FP16A; // spec value 1: fp16_a format
        case DataFormat::Bfp8:
            return InstrModLoadStore::DEFAULT; // spec value 0: default format
        case DataFormat::Bfp4:
            return InstrModLoadStore::DEFAULT; // spec value 0: default format
        case DataFormat::Bfp2:
            return InstrModLoadStore::DEFAULT; // spec value 0: default format
        case DataFormat::Float16_b:
            // With 32-bit (fp32) dest accumulation the datum is stored as full fp32 in the dest word, so
            // SFPLOAD/SFPSTORE must use FP32 access; otherwise the narrow fp16_b format applies.
            return is_fp32_dest_acc_en ? InstrModLoadStore::FP32 : InstrModLoadStore::FP16B; // spec value 2: bfloat/fp16_b
        case DataFormat::Bfp8_b:
            return InstrModLoadStore::DEFAULT; // spec value 0: default format
        case DataFormat::Bfp4_b:
            return InstrModLoadStore::DEFAULT; // spec value 0: default format
        case DataFormat::Bfp2_b:
            return InstrModLoadStore::DEFAULT; // spec value 0: default format
        case DataFormat::Lf8:
            return InstrModLoadStore::DEFAULT; // spec value 0: default format
        case DataFormat::Int8:
            return InstrModLoadStore::INT8; // spec value 5: int8 format
        case DataFormat::UInt8:
            return InstrModLoadStore::INT8; // spec value 5: int8 format
        case DataFormat::UInt16:
            return is_fp32_dest_acc_en ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16; // spec value 6: unsigned int16 format
        case DataFormat::Int32:
            return InstrModLoadStore::INT32; // spec value 4: int32 format
        case DataFormat::UInt32:
            return InstrModLoadStore::INT32; // spec value 4: int32 format
        case DataFormat::Tf32:
            return InstrModLoadStore::FP32; // spec value 3: fp32 format
        default:
            return InstrModLoadStore::DEFAULT; // spec value 0: default format
    }
}

// This is populated per Blackhole ISA for SFPCAST instruction.
enum class InstrModCast
{
    INT32_TO_FP32_NEAREST_EVEN     = 0,
    INT32_TO_FP32_STOCHASTIC       = 1,
    INT32_2S_COMP_TO_INT_SIGN_MAGN = 2,
    INT_SIGN_MAGN_TO_INT32_2S_COMP = 3
};

/**
 * @brief Left-shifts the numeric value of an InstrModCast instruction mode.
 *
 * Provides an integer left-shift for the scoped InstrModCast enum so its value can be packed
 * directly into the SFPCAST instruction word (e.g. the `(instr_mod1) << 0` field encoding).
 *
 * @param mod   The instruction mode whose underlying value is shifted.
 * @param shift The number of bit positions to shift left.
 * @return The underlying value of @p mod shifted left by @p shift, as a std::uint32_t.
 */
constexpr std::uint32_t operator<<(InstrModCast mod, int shift)
{
    return static_cast<std::uint32_t>(mod) << shift;
}

} // namespace ckernel
