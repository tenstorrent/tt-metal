// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "tensix.h"
#include "tensix_types.h"

// #include "ckernel_cmd.h"
// #include "ckernel_enum.h"
// #include "ckernel_ops.h"

namespace ckernel
{

// Helper function to convert to underlying type
// e.g. to_underlying(MathFidelity::HiFi4) -> 4 (underlying type of MathFidelity is std::uint8_t)
template <typename T>
constexpr auto to_underlying(T t) noexcept
{
    return static_cast<std::underlying_type_t<T>>(t);
}

enum Srcs
{
    SrcA = 0,
    SrcB = 1,
    SrcC = 2
};

enum register_space_e
{
    TDMA_REGS     = 0x0,
    LOCAL_REGS    = 0x1,
    ADDR_COUNTERS = 0x2
};

// Mailbox slot index per Quasar Neo-cluster TRISC role: 0=unpack, 1=math, 2=pack, 3=isolate-sfpu.
// The ground truth for this numbering is the mailbox aperture itself -- TENSIX_MAILBOX0..3_BASE in
// quasar/tt_t6_trisc_map.h, which backs mailbox_base[4] in ckernel.h. Note hal_2xx_common.cpp's
// COMPILE_FOR_TRISC=<processor_id> is the cluster-GLOBAL id (NEO_n_COMPUTE_m = n*4 + m, see
// QuasarComputeProcessor in impl/kernels/kernel.hpp); it only agrees with this enum modulo 4.
enum ThreadId
{
    UnpackThreadId      = 0,
    MathThreadId        = 1,
    PackThreadId        = 2,
    IsolateSfpuThreadId = 3
};

// Selects how a float32 SFPU result is narrowed when it is stored back into a
// bf16 DEST. Ignored when fp32 DEST accumulation is enabled, since no narrowing
// happens in that case.
enum class DstRoundingMode : std::uint8_t
{
    Default     = 0, // SFPSTORE truncates fp32->bf16 on all architectures; no software rounding
    NearestEven = 1, // IEEE 754 round-to-nearest-even, applied in software before the store
};

enum class BinaryOp : std::uint8_t
{
    ADD,
    SUB,
    MUL,
    DIV,
    GT,
    LT,
    LE,
    GE,
    MAX,
    MIN,
    QUANT,
    REQUANT,
    DEQUANT,
    ATAN2,
    LOGADDEXP,
    LOGADDEXP2,
};

// For instructions that address lower/upper 16 bits of a register
#define LO_16(REG) (2 * (REG))
#define HI_16(REG) (2 * (REG) + 1)

constexpr std::uint32_t FACE_HEIGHT = 16;
constexpr std::uint32_t FACE_WIDTH  = 16;
constexpr std::uint32_t TILE_HEIGHT = 32;
constexpr std::uint32_t TILE_WIDTH  = 32;

constexpr std::uint32_t FACE_R_DIM = FACE_HEIGHT;
constexpr std::uint32_t FACE_C_DIM = FACE_WIDTH;

constexpr std::uint32_t TILE_R_DIM = TILE_HEIGHT;
constexpr std::uint32_t TILE_C_DIM = TILE_WIDTH;

constexpr std::uint32_t TILE_NUM_FACES = ((TILE_R_DIM * TILE_C_DIM) / (FACE_R_DIM * FACE_C_DIM));

// Number of 32x32 tiles that fit in 16bit mode dest register for DstSync::Full and DstSync::Half
constexpr std::uint32_t DEST_NUM_TILES_FP16      = (DEST_REGISTER_FULL_SIZE * DEST_FACE_WIDTH) / (TILE_HEIGHT * TILE_HEIGHT);
constexpr std::uint32_t DEST_NUM_TILES_FP16_HALF = DEST_NUM_TILES_FP16 / 2;
static_assert((DEST_NUM_TILES_FP16 & (DEST_NUM_TILES_FP16 - 1)) == 0);

} // namespace ckernel
