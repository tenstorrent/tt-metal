// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental {

enum class Nvfp4ScaleFormat { E4M3, E2M1 };

Tensor dequantize_nvfp4(
    const Tensor& packed,
    const Tensor& scales,
    float global_scale = 1.0f,
    Nvfp4ScaleFormat scale_format = Nvfp4ScaleFormat::E4M3,
    DataType dtype = DataType::BFLOAT16,
    Layout output_layout = Layout::TILE,
    bool fp32_dest_acc_en = false,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    bool experimental_fast = false,
    bool fpu_tensor_scale = true,
    bool transpose_output = false);

}  // namespace ttnn::operations::experimental
