// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::kda {

ttnn::Tensor select_tile_rows(
    const ttnn::Tensor& input,
    const ttnn::Tensor& indices,
    uint32_t width,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

// Rows of a history selection record, derived on device from the chronology; split into outputs of rows_per_output
// rows (0: one output). The input may be TILE (any leading width columns) or ROW_MAJOR (whole rows).
std::vector<ttnn::Tensor> select_history_rows(
    const ttnn::Tensor& input,
    uint32_t record,
    const ttnn::Tensor& actual_start,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    std::optional<uint32_t> width = std::nullopt,
    const std::optional<ttnn::Tensor>& actual_end = std::nullopt,
    uint32_t rows_per_output = 0,
    const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental::kda
