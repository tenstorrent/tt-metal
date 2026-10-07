// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct SelectTileRowsParams {
    uint32_t width;
    tt::tt_metal::MemoryConfig output_mem_config;
    // A chronological history selection record: the rows are derived on device from actual_start (and actual_end)
    // instead of read from an index tensor.
    std::optional<uint32_t> record;
    uint32_t sequence_parallel_axis = 0;
    uint32_t local_rows = 0;
    // The selected rows are split into outputs of this many rows; 0 keeps them in one output.
    uint32_t rows_per_output = 0;
};

struct SelectTileRowsInputs {
    Tensor input;
    std::optional<Tensor> indices;
    std::optional<Tensor> actual_start;
    std::optional<Tensor> actual_end;
};

}  // namespace ttnn::experimental::prim
