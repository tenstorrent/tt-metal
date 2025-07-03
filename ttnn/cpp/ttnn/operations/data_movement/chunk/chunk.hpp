// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ChunkOperation {
    static std::vector<ttnn::Tensor> invoke(const ttnn::Tensor& input_tensor, uint32_t num_chunks, int dim);
};

}  // namespace operations::data_movement

constexpr auto chunk = ttnn::register_operation<"ttnn::chunk", ttnn::operations::data_movement::ChunkOperation>();

}  // namespace ttnn
