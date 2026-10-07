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
        input, indices, width, memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG))[0];
}

std::vector<ttnn::Tensor> select_history_rows(
    const ttnn::Tensor& input,
    uint32_t record,
    const ttnn::Tensor& actual_start,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    std::optional<uint32_t> width,
    const std::optional<ttnn::Tensor>& actual_end,
    uint32_t rows_per_output,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    TT_FATAL(
        input.storage_type() == StorageType::DEVICE && input.buffer() != nullptr,
        "select_history_rows: input must be an allocated device tensor");
    return ttnn::experimental::prim::select_tile_rows(
        input,
        std::nullopt,
        width.value_or(static_cast<uint32_t>(input.logical_shape()[-1])),
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        record,
        actual_start,
        actual_end,
        sequence_parallel_axis,
        local_rows,
        rows_per_output);
}

}  // namespace ttnn::experimental::kda
