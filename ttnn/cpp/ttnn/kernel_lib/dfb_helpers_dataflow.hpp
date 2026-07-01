// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file dfb_helpers_dataflow.hpp
 * @brief Dataflow-kernel DataflowBuffer tile dimension helpers
 */

namespace dataflow_kernel_lib {

template <uint32_t dfb_id>
FORCE_INLINE constexpr uint32_t get_tile_r_dim();

template <uint32_t dfb_id>
FORCE_INLINE constexpr uint32_t get_tile_c_dim();

}  // namespace dataflow_kernel_lib

#include "dfb_helpers_dataflow.inl"
