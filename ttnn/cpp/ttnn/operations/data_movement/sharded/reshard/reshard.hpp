// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn {

Tensor reshard(
    const Tensor& input_tensor,
    const tt::tt_metal::MemoryConfig& memory_config,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

}  // namespace ttnn
