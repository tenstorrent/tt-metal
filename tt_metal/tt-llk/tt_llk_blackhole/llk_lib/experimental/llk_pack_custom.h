// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_template.h"
#include "tensix.h"

/*************************************************************************
 * LLK PACK CUSTOM - Lightweight MOP outer-loop patching
 *************************************************************************/

// WARNING: Experimental API for SDPA optimizations only.

// Patch only mop_cfg[0] (outer loop count = num_faces * num_tiles).
// Precondition: a full _llk_pack_mop_config_ must have been called first
// to program all 9 MOP registers. Safe when all CBs share the same tile
// format (same num_faces, face_r_dim, tile_c_dim).
inline void _llk_pack_set_mop_outer_loop_(std::uint32_t num_faces, std::uint32_t num_tiles)
{
    volatile std::uint32_t* mop_cfg = reinterpret_cast<volatile std::uint32_t*>(TENSIX_MOP_CFG_BASE);
    ckernel::mop_sync();
    mop_cfg[0] = num_faces * num_tiles;
}
