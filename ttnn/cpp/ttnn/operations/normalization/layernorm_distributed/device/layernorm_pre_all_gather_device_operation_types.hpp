// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"
#include "ttnn/operations/normalization/layernorm_distributed/device/layernorm_distributed_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tuple>

namespace ttnn::prim {

struct LayerNormPreAllGatherParams {
    LayerNormDistributedType norm_type = LayerNormDistributedType::LAYERNORM;
    std::optional<tt::tt_metal::DataType> dtype = std::nullopt;
    DeviceComputeKernelConfig compute_kernel_config;
    LayerNormProgramConfig program_config;
    std::optional<bool> use_2d_core_grid;

    static constexpr auto attribute_names =
        std::forward_as_tuple("norm_type", "dtype", "compute_kernel_config", "program_config", "use_2d_core_grid");
    auto attribute_values() const {
        return std::forward_as_tuple(norm_type, dtype, compute_kernel_config, program_config, use_2d_core_grid);
    }
};

}  // namespace ttnn::prim
