// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

bool supported_by_codegen(
    const Tensor& input, uint32_t repeats, int32_t dim, const std::optional<MemoryConfig>& output_mem_config);

}  // namespace ttnn::operations::data_movement
