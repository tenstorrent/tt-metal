// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/tile_move_copy.h"

#ifndef ARCH_QUASAR
// True when DFB `other` has the same unpack formats and tile geometry as DFB `ref`.  The JIT format
// arrays are indexed by each DFB's slot.  The kernels pass dfb:: tokens, whose conversion to the
// slot is constexpr.
template <uint32_t ref, uint32_t other>
constexpr bool copy_init_shared_with() {
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    return unpack_src_format[other] == unpack_src_format[ref] && unpack_dst_format[other] == unpack_dst_format[ref] &&
           unpack_tile_num_faces[other] == unpack_tile_num_faces[ref] &&
           unpack_tile_face_r_dim[other] == unpack_tile_face_r_dim[ref] &&
           unpack_partial_face[other] == unpack_partial_face[ref] &&
           unpack_narrow_tile[other] == unpack_narrow_tile[ref] && unpack_tile_r_dim[other] == unpack_tile_r_dim[ref] &&
           unpack_tile_c_dim[other] == unpack_tile_c_dim[ref];
#else
    return true;
#endif
}
#endif

// The call line goes through so the compute kernel sentinel names the kernel's line.
template <bool init_once>
ALWI void copy_init_unless(uint32_t dfb_id, uint32_t call_line = __builtin_LINE()) {
    if constexpr (!init_once) {
        ckernel::copy_init(dfb_id, 0, false, call_line);
    }
}
