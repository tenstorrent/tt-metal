// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ChunkOperation {
    static std::vector<ttnn::Tensor> invoke(const ttnn::Tensor& input_tensor, const int chunks, const int dim);
};

}  // namespace operations::data_movement

constexpr auto chunk = ttnn::register_operation<"ttnn::chunk", ttnn::operations::data_movement::ChunkOperation>();

}  // namespace ttnn
