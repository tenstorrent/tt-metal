// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Implementation file for dfb_helpers_dataflow.hpp
// Do not include directly - include dfb_helpers_dataflow.hpp instead

namespace dataflow_kernel_lib {

template <uint32_t dfb_id>
FORCE_INLINE constexpr uint32_t get_tile_r_dim() {
    return unpack_tile_r_dim[dfb_id];
}

template <uint32_t dfb_id>
FORCE_INLINE constexpr uint32_t get_tile_c_dim() {
    return unpack_tile_c_dim[dfb_id];
}

}  // namespace dataflow_kernel_lib
