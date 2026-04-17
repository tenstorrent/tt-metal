// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Hand-coded parameter encoding for various GPR mappings
#include <cstdint>

namespace ckernel
{

// Common GPR mapping across all threads
struct p_gpr
{
    constexpr static std::uint32_t ZERO     = 0; // Always stores 0
    constexpr static std::uint32_t RBASE    = 1; // Ring pointer base
    constexpr static std::uint32_t ROFFSET  = 2; // Ring pointer offset
    constexpr static std::uint32_t DBG_CKID = 3; // Ckernel ID
};

// Unpack GPR thread
struct p_gpr_unpack
{
    constexpr static std::uint32_t X_DIM_TOP     = 50; // top halo params
    constexpr static std::uint32_t X_DIM_BOT     = 51; // bottom halo params
    constexpr static std::uint32_t WEIGHT_OFFSET = 50; //
    constexpr static std::uint32_t WEIGHT_INCR   = 51; //
    constexpr static std::uint32_t TEMPORARY     = 52; //
    constexpr static std::uint32_t DATUM_CNT_TOP = 56; // top halo params
    constexpr static std::uint32_t DATUM_CNT_BOT = 57; // bottom halo params
};

// Math GPR thread
struct p_gpr_math
{
    constexpr static std::uint32_t TEMP0             = 4;  // dest rwc base (1st set)
    constexpr static std::uint32_t DEST_REGW_OFFSET  = 50; // dest rwc base (1st set)
    constexpr static std::uint32_t DEST_REGW_INCR    = 51; // dest rwc incr (1st set)
    constexpr static std::uint32_t DEST_REGW_OFFSET2 = 52; // dest rwc base (2nd set)
    constexpr static std::uint32_t DEST_REGW_INCR2   = 53; // dest rwc incr (2nd set)
};

// Pack GPR thread
struct p_gpr_pack
{
    constexpr static std::uint32_t TEMP0     = 4;
    constexpr static std::uint32_t TEMP1     = 5;
    constexpr static std::uint32_t TEMP2     = 6;
    constexpr static std::uint32_t SYNC_FLAG = 7;

    // Standard 1p/2p/4p/strips packers
    constexpr static std::uint32_t OUTPUT_ADDR    = 12; // output address that packer is writing to
    constexpr static std::uint32_t MSGINFO_ADDR   = 16; // msg info address read from overlay
    constexpr static std::uint32_t FORCE_MAX_XY   = 39; // 32b aligned read for force_max_xy reg
    constexpr static std::uint32_t TILE_HEADER    = 40; // tile header - ID + tile size
    constexpr static std::uint32_t TILE_METADATA  = 41; // tile meta data + format and compression flag
    constexpr static std::uint32_t TILE_ZERO_MASK = 42; // zero mask for the output tile
    constexpr static std::uint32_t TILE_RESERVED  = 43; // reserved
    constexpr static std::uint32_t DEST_OFFSET_HI = 56; // dest upper bank offsets
    constexpr static std::uint32_t DEST_OFFSET_LO = 60; // dest lower bank offsets
    constexpr static std::uint32_t FLUSH_COUNTERS = 56; // flush pack counters
    constexpr static std::uint32_t FLUSH_OFFSET   = 60; // flush dest read offsets

    // Special packer spilling to L1 scratch buffer
    constexpr static std::uint32_t OUTPUT_FORMAT        = 50; // output buffer format
    constexpr static std::uint32_t SCRATCH_FORMAT       = 51; // scratch buffer format
    constexpr static std::uint32_t TILE_STITCHER_HEADER = 52; // used to store tile header offset
    constexpr static std::uint32_t TILE_STITCHER_GAP    = 53; // used to store tile stitching gap
    constexpr static std::uint32_t TEMPORARY            = 54; //
    constexpr static std::uint32_t PACK_COUNTERS        = 55; // only used if counter reprogramming is required
};

struct p_params
{
    // Param array mappings
    constexpr static std::uint32_t CMD_BASE = 0;

    // Param load settings
    constexpr static std::uint32_t DST_COPY   = 0; // L1 copy into localmem then access (Requires repeated use to amortize L1 copy cost)
    constexpr static std::uint32_t SRC_DIRECT = 1; // L1 direct access of params (Fastest for single use or user managed local copies)
};

} // namespace ckernel
