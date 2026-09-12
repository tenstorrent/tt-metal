// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "cfg_defines.h"
#include "ckernel.h"
#include "ckernel_addrmod.h"
#include "ckernel_buf_desc.h"
#include "ckernel_instr_params.h"
#include "ckernel_proj_params.h"
#include "ckernel_template.h"
#include "llk_assert.h"
#include "llk_defs.h"
#include "llk_reinit_guard.h"
#include "llk_tdma_guard.h"
#include "tensix_types.h"
#include "tensor_shape.h"

namespace ckernel::trisc
{
// Fixed hardware thread ids, matching the DEST section register layout (SEC0..SEC3) and the
// -DCOMPILE_FOR_TRISC=<n> the build assigns per thread. Use these when a call must address a specific
// thread's slot rather than the compile-time thread it runs on -- e.g. the unpack-to-dest path, where
// the unpack thread programs the UNP_DEST producer slot (Unpack) regardless of what it is compiled as.
enum class TriscID : std::uint8_t
{
    Unpack = 0,
    Math   = 1,
    Pack   = 2,
    Sfpu   = 3, // isolate-SFPU
};

using ckernel::FACE_C_DIM;
using ckernel::FACE_R_DIM;
using ckernel::TILE_C_DIM;
using ckernel::TILE_R_DIM;

// Default number of faces
constexpr static std::uint32_t NUM_FACES = 4;

// Default number of tiles processed per one _llk_* execute call
static constexpr std::uint32_t NUM_TILES = 1;

// Number of rows that can fit into Dest
static constexpr std::uint32_t DEST_REGISTER_FULL_SIZE = 64 * FACE_R_DIM;
static constexpr std::uint32_t DEST_REGISTER_HALF_SIZE = DEST_REGISTER_FULL_SIZE >> 1;

// Number of bits used to represent data format in unpacker/packer config.
// The reason we keep only the bottom 4 bits is because the HW only has 4 bits to represent the dataformat.
// Essentially, only using the bottom 4 bits when programming the HW dataformats is not a bug, because
// the higher bits should be used to program other registers.
// Also, uint32 does not require special handling because both int32 and uint32 are stored the same way in DEST
// (just the highest bit is interpreted as a sign bit vs. a magnitude bit by the user). So when the packer
// needs to read DEST in order to pack the data out it reads the same data either way and just moves 32bits in both cases.
// Uint8 requires special handling because when int8 is put into DEST, the sign bit actually gets put
// to the MSB of the 32bit container, rather than to bit 8. So for int8 the packer will read the 7 LSBs + 1 MSB,
// but for uint8 the packer will read the 8 LSBs.
constexpr std::uint32_t DATA_FORMAT_BIT_COUNT = 5;
// Mask to extract data format bits
constexpr std::uint32_t DATA_FORMAT_CONFIG_MASK = (1 << DATA_FORMAT_BIT_COUNT) - 1;
constexpr std::uint32_t NUM_WORDS_TILE_CNT = 8;

typedef struct
{
    std::uint32_t reserved0                    : 32;
    std::uint32_t reset                        : 32;
    std::uint32_t posted                       : 32;
    std::uint32_t acked                        : 32;
    std::uint32_t buf_capacity                 : 32;
    std::uint32_t reserved1                    : 32;
    std::uint32_t tiles_avail_interrupt_thresh : 32;
    std::uint32_t space_avail_interrupt_thresh : 32;
} tile_counter_t;

static_assert(sizeof(tile_counter_t) == NUM_WORDS_TILE_CNT * sizeof(std::uint32_t), "tile_counter_t must be 96b!");

typedef union
{
    std::uint32_t words[NUM_WORDS_TILE_CNT];
    tile_counter_t f;
} tile_counter_u;

// Points to the tile counters
tile_counter_u volatile* const tile_counters = (tile_counter_u volatile* const)TILE_COUNTERS_BASE;

// Destination register offset, offset = 0 -> targets dest bank 0, offset = 512 for 16bit dest, 256 for 32bit dest -> targets dest bank 1
#ifdef ENV_LLK_INFRA
static std::uint32_t dest_register_offset = 0;
#else
extern thread_local std::uint32_t dest_register_offset;
#endif

/**
* @brief Check divisibility by power of 2
* @param value: input value to check divisibility
* @param power_of_two_divisor: divisor that must be a power of 2, will check if
    value % power_of_two_divisor == 0
*/
inline bool _divisible_by_pow_two_(const std::uint32_t value, const std::uint32_t power_of_two_divisor)
{
    return ((value & (power_of_two_divisor - 1)) == 0);
}

enum class DstTileShape : std::uint8_t
{
    Tile32x1  = 1,
    Tile32x2  = 2,
    Tile32x4  = 3,
    Tile32x8  = 4,
    Tile32x16 = 5,
    Tile32x32 = 6
};

constexpr std::uint32_t get_dest_tile_size_log2(const DstTileShape tile_shape)
{
    return ckernel::to_underlying(tile_shape) < ckernel::to_underlying(DstTileShape::Tile32x8) ? ckernel::to_underlying(DstTileShape::Tile32x8)
                                                                                               : ckernel::to_underlying(tile_shape);
}

/**
 * @brief Calculates the maximum number of tiles that fit in the math destination region.
 *
 * Destination addressing uses a minimum 16-row footprint for tile shapes smaller
 * than 32x16.
 *
 * @tparam SYNC_MODE: Destination synchronization mode, values = <SyncHalf/SyncFull>
 * @tparam ACCUM_MODE: Accumulation mode, true for 32-bit and false for 16-bit
 * @tparam TILE_SHAPE: Destination tile shape
 * @return Maximum number of destination tiles.
 */
template <ckernel::DstSync SYNC_MODE, bool ACCUM_MODE, DstTileShape TILE_SHAPE>
constexpr std::uint32_t get_dest_max_tiles()
{
    constexpr std::uint32_t DEST_REGISTER_SIZE = SYNC_MODE == ckernel::DstSync::SyncHalf
                                                     ? (ACCUM_MODE ? DEST_REGISTER_HALF_SIZE >> 1 : DEST_REGISTER_HALF_SIZE)
                                                     : (ACCUM_MODE ? DEST_REGISTER_FULL_SIZE >> 1 : DEST_REGISTER_FULL_SIZE);

    return DEST_REGISTER_SIZE >> get_dest_tile_size_log2(TILE_SHAPE);
}

/**
 * @brief Sets the destination register base address, each Trisc0/1/2/3 has separate
 * registers for setting dest base address.
 * @tparam TRISC_ID: Trisc core which is executing this function, values = [0, 1, 2, 3]
 * @param base_addr: Base address for destination register
 * Bank 0 -> base_addr = 0 - 511
 * Bank 1 -> base_addr = 512 - 1023
 */
template <std::uint8_t TRISC_ID>
inline void _set_dest_section_base_(const std::uint32_t base_addr)
{
    if constexpr (TRISC_ID == 0)
    {
        cfg[DEST_TARGET_REG_CFG_MATH_SEC0_Offset_ADDR32] = base_addr;
    }
    else if constexpr (TRISC_ID == 1)
    {
        cfg[DEST_TARGET_REG_CFG_MATH_SEC1_Offset_ADDR32] = base_addr;
    }
    else if constexpr (TRISC_ID == 2)
    {
        cfg[DEST_TARGET_REG_CFG_MATH_SEC2_Offset_ADDR32] = base_addr;
    }
    else
    {
        cfg[DEST_TARGET_REG_CFG_MATH_SEC3_Offset_ADDR32] = base_addr;
    }
}

/**
 * @brief Helper function to calculate log2 for FPU rows
 * since FPU rows are <=16, and are power of 2, can use
 * simplified higher perf method
 * @param val: Input value to log2 operation
 */
inline std::uint32_t rows_log2(const std::uint32_t math_rows)
{
    switch (math_rows)
    {
        case 16:
            return 4;
        case 8:
            return 3;
        case 4:
            return 2;
        case 2:
            return 1;
        default:
            return 0;
    }
}

/**
 * @brief Returns dest buffer base addr
 * If dest register is set to 16bit mode:
 *     Bank 0 -> addr = 0
 *     Bank 1 -> addr = 512
 * If dest register is set to 32bit mode:
 *     Bank 0 -> addr = 0
 *     Bank 1 -> addr = 256
 */
inline std::uint32_t _get_dest_buffer_base_()
{
    return dest_register_offset;
}

inline constexpr static std::uint32_t masked_data_format(std::uint32_t data_format)
{
    return data_format & DATA_FORMAT_CONFIG_MASK;
}

constexpr static std::uint32_t SCALE_DATUM_SIZE(std::uint32_t format, std::uint32_t datum_count)
{
    switch (masked_data_format(format))
    {
        case (to_underlying(DataFormat::Int32)):
        case (to_underlying(DataFormat::Float32)):
            return (datum_count << 2);
        case (to_underlying(DataFormat::Float16)):
        case (to_underlying(DataFormat::Float16_b)):
            return (datum_count << 1);
        default:
            return datum_count;
    };
}

/**
 * All the following functions are added to enable Math <-> Pack synchronization
 *
 * Another Issue:
 * Some of the following enums/functions are needed to increment to arbitrary addresses in dest, the functions can be removed
 * if a Tensix instruction is added to address the full dest
 *
 * The following functions should be removed once the dvalid scheme is completely used for all LLK operations
 */

/**
 * @brief Set destination register offset variable to 0
 */
inline void _reset_dest_register_offset_()
{
    dest_register_offset = 0;
}

/**
 * @brief Update destination register offset, offset can only toggle between 0 & 512 for 16bit dest, 0 & 256 for 32bit dest
 */
template <bool EN_32BIT_DEST>
inline void _update_dest_register_offset_()
{
    constexpr std::uint32_t dest_bank1_offset = EN_32BIT_DEST ? DEST_REGISTER_HALF_SIZE >> 1 : DEST_REGISTER_HALF_SIZE;
    dest_register_offset                      = (dest_register_offset == 0) ? dest_bank1_offset : 0;
}

// Semaphores mapping and trisc space -> tensix space conversion
struct semaphore
{
    // The math thread is the middleman, for regular unpack and for unpack_to_dest.
    // When unpacking to dest, math thread doesn't produce data, it just bridges UNPACK_MATH -> MATH_PACK.
    // Packer only listens on MATH_PACK, so something has to translate the unpack completion into a
    // pack-visible event. Math being the forwarder is also what makes future fused ops cheap:
    // SFPU/FPU work slots in between the UNPACK_MATH get and the MATH_PACK post.
    //
    // Keep pairwise naming with producer_consumer direction:
    // - MATH_PACK = math->pack
    // - UNPACK_MATH = unpack->math
    // - PACK_UNPACK = pack->unpack
    constexpr static std::uint32_t MATH_PACK   = 1; // math <-> pack sync on dest register
    constexpr static std::uint32_t UNPACK_MATH = 4; // unpack <-> math sync on dest register
    constexpr static std::uint32_t PACK_UNPACK = 7; // pack <-> unpack sync on L1 memory

    constexpr static std::uint16_t t6_sem(const std::uint8_t sem_index)
    {
        return (1u << sem_index);
    }
};

// Tensix thread semaphore post optionally stalled
// Can stall on up to 3 resources at a time
template <std::uint32_t WaitRes0 = p_stall::NOTHING, std::uint32_t WaitRes1 = p_stall::NOTHING, std::uint32_t WaitRes2 = p_stall::NOTHING>
inline void t6_semaphore_post(const std::uint8_t index)
{
    if constexpr (WaitRes0 != p_stall::NOTHING)
    {
        TTI_STALLWAIT(p_stall::STALL_SYNC, WaitRes2, WaitRes1, WaitRes0);
    }

    TT_SEMPOST(0, semaphore::t6_sem(index));
}

// Tensix thread semaphore get optionally stalled
// Can stall on up to 3 resources at a time
template <std::uint32_t WaitRes0 = p_stall::NOTHING, std::uint32_t WaitRes1 = p_stall::NOTHING, std::uint32_t WaitRes2 = p_stall::NOTHING>
inline void t6_semaphore_get(const std::uint8_t index)
{
    if constexpr (WaitRes0 != p_stall::NOTHING)
    {
        TTI_STALLWAIT(p_stall::STALL_SYNC, WaitRes2, WaitRes1, WaitRes0);
    }

    TT_SEMGET(0, semaphore::t6_sem(index));
}

/**
 * @brief Set packer's dest register offset to the current dest bank base.
 *
 * In SyncHalf mode, alternates between bank 0 (offset 0) and bank 1
 * (DEST_REGISTER_HALF_SIZE for 16-bit dest, DEST_REGISTER_HALF_SIZE/2 for 32-bit dest).
 * In SyncFull mode, always reads from offset 0.
 */
template <std::uint32_t PACK_SEL, ckernel::DstSync DST>
inline void _set_packer_dest_registers_()
{
    static_assert(DST == ckernel::DstSync::SyncHalf || DST == ckernel::DstSync::SyncFull);
    std::uint32_t dest_buffer_base_offset = (DST == ckernel::DstSync::SyncFull) ? 0 : _get_dest_buffer_base_();

    // Masked write of just SRC_ADDR_OFFSET. On PACKER1 this cfg word (ADDR32 65) also holds the
    // INSTRN_LOOP_COUNT/COUNT auto-loop bits programmed by _llk_pack_srcs_config_ (llk_srcs.h); a
    // full-word write would zero them (per-tile in SyncHalf, once the SrcS->Packer1 path is wired).
    // PACKER0's word has no such siblings today, but keep it masked for symmetry.
    if constexpr (PACK_SEL == p_pacr::PACK0)
    {
        cfg_rmw(THCON_PACKER0_REG0_SRC_ADDR_OFFSET_RMW, dest_buffer_base_offset);
    }
    else
    {
        cfg_rmw(THCON_PACKER1_REG0_SRC_ADDR_OFFSET_RMW, dest_buffer_base_offset);
    }
}

// SrcS register tile geometry (HW-defined for Quasar SrcS).
// A 32x32 tile is produced/consumed across SLICE_COUNT SrcS slices, where one slice
// holds XDIM * YDIM * ZDIM datums. YDIM halves in 32-bit element mode because the
// SrcS columns are 16-bit wide in HW.
struct srcs_dims
{
    static constexpr std::uint32_t XDIM      = 16; // datums per row of SrcS slice
    static constexpr std::uint32_t ZDIM      = 1;
    static constexpr std::uint32_t YDIM_BASE = 8; // rows per slice when SrcS is in 16-bit mode

    static constexpr std::uint32_t ydim(bool srcs_32bit_mode)
    { // TODO for metal bringup: make programmable based on tensor_shape for tiny tile support
        return srcs_32bit_mode ? (YDIM_BASE / 2) : YDIM_BASE;
    }

    static constexpr std::uint32_t slice_count(bool srcs_32bit_mode)
    {
        return (TILE_R_DIM * TILE_C_DIM) / (XDIM * ydim(srcs_32bit_mode) * ZDIM);
    }
};

// SrcS runs in 32-bit element mode when the UNP_S destination format is 32-bit wide.
// Unpack-to-SrcS cannot convert fp16 to TF32, so Tf32 is not a legal unpack_S_dst here.
inline constexpr bool _is_srcs_32bit_mode_(const DataFormat unpack_S_dst_format)
{
    return unpack_S_dst_format == DataFormat::Float32 || unpack_S_dst_format == DataFormat::Int32;
}

/**
 * @brief finds and returns the larger value between two inputs
 * @note if both values are equal returns input1
 *
 * @param input1/input2: the values to be compared
 */
inline std::uint32_t find_max(std::uint32_t input1, std::uint32_t input2)
{
    return (input1 >= input2) ? input1 : input2;
}

} // namespace ckernel::trisc
