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

enum class DstSync : std::uint8_t
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

enum ReluType
{
    NO_RELU,
    ZERO_RELU,
    MIN_THRESHOLD_RELU,
    MAX_THRESHOLD_RELU,
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

// This is populated per Wormhole ISA for SFPLOAD/SFPSTORE instructions.
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

template <DataFormat format>
constexpr InstrModLoadStore GetSfpLoadStoreInstrMod()
{
    switch (format)
    {
        case DataFormat::Float32:
            return InstrModLoadStore::FP32; // spec value 3: fp32 format
        case DataFormat::Float16:
            return InstrModLoadStore::FP16A; // spec value 1: fp16_a format
        case DataFormat::Bfp8:
            return InstrModLoadStore::DEFAULT; // spec value 0: default format
        case DataFormat::Bfp4:
            return InstrModLoadStore::DEFAULT; // spec value 0: default format
        case DataFormat::Bfp2:
            return InstrModLoadStore::DEFAULT; // spec value 0: default format
        case DataFormat::Float16_b:
            return InstrModLoadStore::FP16B; // spec value 2: bfloat/fp16_b
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
            return InstrModLoadStore::LO16; // spec value 6: unsigned int16 format
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

} // namespace ckernel
