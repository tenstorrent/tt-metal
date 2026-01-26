// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp"

namespace dataflow_kernel_lib {

// Row size in uint32 (8 u32 = 16 bf16)
constexpr uint32_t ROW_SIZE_U32 = 8;

/**
 * @brief Fill row 0 of each face with a scaler value
 *
 * @tparam half_tile If true, fill faces 0-1 only. If false, fill all 4 faces.
 * @param ptr Pointer to the start of the tile in L1 memory
 * @param scaler Packed bf16 value to write (bf16 << 16 | bf16)
 */
template <bool half_tile>
FORCE_INLINE void fill_row0(volatile tt_l1_ptr uint32_t* ptr, uint32_t scaler);

/**
 * @brief Generate a reduce scaler tile
 *
 * Creates a tile in the specified circular buffer with the scaler value
 * placed in row 0 of each face. The scaler is typically 1.0 for SUM/MAX
 * reductions, and 1/N for AVG reductions.
 *
 * @tparam half_tile If true, only fill faces 0-1 (half tile mode)
 * @param cb_id Circular buffer ID to write the tile to
 * @param scaler Packed bf16 value (bf16 << 16 | bf16)
 */
template <bool half_tile = false>
FORCE_INLINE void generate_reduce_scaler(const uint32_t cb_id, const uint32_t scaler);

}  // namespace dataflow_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.inl"
