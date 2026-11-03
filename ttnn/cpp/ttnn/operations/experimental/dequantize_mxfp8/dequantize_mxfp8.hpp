// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental {

// Decode MXFP8 (OCP MX, E4M3 elements, E8M0 scale per 32 horizontal values).
Tensor dequantize_mxfp8(
    const Tensor& packed,
    const Tensor& scales,
    DataType dtype = DataType::BFLOAT16,
    Layout output_layout = Layout::TILE,
    bool fp32_dest_acc_en = false,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::operations::experimental
