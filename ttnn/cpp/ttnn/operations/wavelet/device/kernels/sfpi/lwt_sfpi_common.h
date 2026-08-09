// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "cmath_common.h"
#include "sfpi.h"

#if defined(ARCH_BLACKHOLE) == defined(ARCH_WORMHOLE)
#error "TT-Metal JIT must define exactly one supported architecture"
#endif

using namespace sfpi;

namespace ckernel::sfpu {

inline uint32_t _lwt_dst_base(const std::uint32_t tile_index, const std::uint32_t face_index) {
    return (tile_index << 6) + (face_index << 4);
}

inline uint32_t _lwt_narrow_dst_base(const std::uint32_t tile_index, const std::uint32_t face_index) {
    return (tile_index << 5) + (face_index << 4);
}

inline void _lwt_clear_addr_mod_base_() {
#if defined(ARCH_WORMHOLE)
    ckernel::math::clear_addr_mod_base();
#elif defined(ARCH_BLACKHOLE)
    // Blackhole has no alternate address-modifier bank or base selector.
#else
#error "Unsupported Tensix architecture"
#endif
}

}  // namespace ckernel::sfpu
