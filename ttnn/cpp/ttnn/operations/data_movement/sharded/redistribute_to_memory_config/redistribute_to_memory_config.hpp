// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn {

ttnn::Tensor redistribute_to_memory_config(
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& output_memory_config,
    const std::optional<DataType>& data_type_arg,
    const std::optional<Tensor>& preallocated_output = std::nullopt);

}  // namespace ttnn
