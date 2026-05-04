// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/types.hpp"

namespace ttnn {

Tensor copy_new(const Tensor& src_tensor, const Tensor& dst_tensor);

Tensor assign_new(
    const Tensor& input,
    const MemoryConfig& output_mem_config,
    std::optional<const DataType> output_dtype = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor assign_new(const Tensor& input_a, const Tensor& input_b);

}  // namespace ttnn
