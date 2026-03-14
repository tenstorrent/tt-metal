// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file cb_helpers_dataflow.hpp
 * @brief Dataflow-kernel circular buffer tile dimension helpers
 */

namespace dataflow_kernel_lib {

template <uint32_t cb_id>
FORCE_INLINE constexpr uint32_t get_tile_r_dim();

template <uint32_t cb_id>
FORCE_INLINE constexpr uint32_t get_tile_c_dim();

}  // namespace dataflow_kernel_lib

#include "cb_helpers_dataflow.inl"
