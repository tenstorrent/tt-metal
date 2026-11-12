// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/tile_move_copy.h"

#ifndef ARCH_QUASAR
template <uint32_t cb>
constexpr bool copy_init_shared_with_c0() {
#if defined(TRISC_UNPACK) || defined(TRISC_MATH)
    constexpr uint32_t c0 = tt::CBIndex::c_0;
    return unpack_src_format[cb] == unpack_src_format[c0] && unpack_dst_format[cb] == unpack_dst_format[c0] &&
           unpack_tile_num_faces[cb] == unpack_tile_num_faces[c0] &&
           unpack_tile_face_r_dim[cb] == unpack_tile_face_r_dim[c0] &&
           unpack_partial_face[cb] == unpack_partial_face[c0] && unpack_narrow_tile[cb] == unpack_narrow_tile[c0] &&
           unpack_tile_r_dim[cb] == unpack_tile_r_dim[c0] && unpack_tile_c_dim[cb] == unpack_tile_c_dim[c0];
#else
    return true;
#endif
}
#endif

// The call line goes through so the compute kernel sentinel names the kernel's line.
template <bool init_once>
ALWI void copy_init_unless(uint32_t cb, uint32_t call_line = __builtin_LINE()) {
    if constexpr (!init_once) {
        ckernel::copy_init(cb, 0, false, call_line);
    }
}
