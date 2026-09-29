// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "select_tile_rows.hpp"

#include "device/select_tile_rows_device_operation.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor select_tile_rows(
    const ttnn::Tensor& input,
    const ttnn::Tensor& indices,
    uint32_t width,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    TT_FATAL(
        input.storage_type() == StorageType::DEVICE && input.buffer() != nullptr,
        "select_tile_rows: input must be an allocated device tensor");
    return ttnn::experimental::prim::select_tile_rows(
        input, indices, width, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG));
}

}  // namespace ttnn::experimental::kda
