// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct SelectFinalCarryParams {
    uint32_t sequence_parallel_axis;
    uint32_t local_rows;
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct SelectFinalCarryInputs {
    Tensor rank_finals;
    Tensor prefix_final;
    Tensor actual_start;
    std::optional<Tensor> actual_end;
};

}  // namespace ttnn::experimental::prim
