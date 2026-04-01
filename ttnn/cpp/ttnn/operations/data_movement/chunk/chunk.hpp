// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <vector>
#include "ttnn/types.hpp"

namespace ttnn {

std::vector<ttnn::Tensor> chunk(const ttnn::Tensor& input_tensor, uint32_t num_chunks, int dim);

}  // namespace ttnn
