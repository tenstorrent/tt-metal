// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace compute_kernel_lib {

template <std::uint32_t cb_id>
constexpr std::uint32_t cb_l1_format() {
#if defined(UCK_CHLKC_PACK)
    return pack_dst_format[cb_id];
#else
    return unpack_src_format[cb_id];
#endif
}

template <std::uint32_t cb_id>
constexpr bool cb_has_32x32_tiles() {
#if defined(UCK_CHLKC_PACK)
    constexpr std::uint32_t tile_r_dim = pack_tile_r_dim[cb_id];
    constexpr std::uint32_t tile_c_dim = pack_tile_c_dim[cb_id];
#else
    constexpr std::uint32_t tile_r_dim = unpack_tile_r_dim[cb_id];
    constexpr std::uint32_t tile_c_dim = unpack_tile_c_dim[cb_id];
#endif
    return tile_r_dim == 32 && tile_c_dim == 32;
}

}  // namespace compute_kernel_lib
