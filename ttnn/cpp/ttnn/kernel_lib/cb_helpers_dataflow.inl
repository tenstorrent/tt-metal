// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Implementation file for cb_helpers_dataflow.hpp
// Do not include directly - include cb_helpers_dataflow.hpp instead

namespace dataflow_kernel_lib {

template <uint32_t cb_id>
FORCE_INLINE constexpr uint32_t get_tile_r_dim() {
    return unpack_tile_r_dim[cb_id];
}

template <uint32_t cb_id>
FORCE_INLINE constexpr uint32_t get_tile_c_dim() {
    return unpack_tile_c_dim[cb_id];
}

}  // namespace dataflow_kernel_lib
