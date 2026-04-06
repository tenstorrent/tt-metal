// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn {

ttnn::Tensor reshard(
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& memory_config,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

}  // namespace ttnn
