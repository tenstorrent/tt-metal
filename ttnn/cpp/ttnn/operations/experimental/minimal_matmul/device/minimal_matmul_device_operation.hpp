// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::minimal_matmul {

struct MinimalMatmulConfig {
    MinimalMatmulConfig(CoreCoord compute_with_storage_grid_size_ = {1, 1}) :
        compute_with_storage_grid_size(compute_with_storage_grid_size_) {}

    CoreCoord compute_with_storage_grid_size;

    static constexpr auto attribute_names = std::make_tuple("compute_with_storage_grid_size");

    auto attribute_values() const { return std::forward_as_tuple(this->compute_with_storage_grid_size); }
};

struct MinimalMatmulOp {
    MinimalMatmulConfig config;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;

    void validate(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;

    std::vector<TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;

    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::experimental::minimal_matmul
