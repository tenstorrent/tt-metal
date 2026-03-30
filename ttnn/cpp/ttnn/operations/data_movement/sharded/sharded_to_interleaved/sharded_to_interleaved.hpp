// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/types.hpp"

namespace ttnn {

ttnn::Tensor sharded_to_interleaved(
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& memory_config,
    const std::optional<DataType>& output_dtype = std::nullopt);

}  // namespace ttnn
