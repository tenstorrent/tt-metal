// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/core_coord.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tuple>

namespace ttnn::prim {

// Program config types
struct GroupNormMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    tt::tt_metal::DataType im_data_format{tt::tt_metal::DataType::INVALID};
    tt::tt_metal::DataType out_data_format{tt::tt_metal::DataType::INVALID};
    bool inplace{};
    tt::tt_metal::Layout output_layout{tt::tt_metal::Layout::INVALID};
    int num_out_blocks{};

    static constexpr auto attribute_names = std::forward_as_tuple(
        "compute_with_storage_grid_size",
        "im_data_format",
        "out_data_format",
        "inplace",
        "output_layout",
        "num_out_blocks");
    auto attribute_values() const {
        return std::forward_as_tuple(
            compute_with_storage_grid_size, im_data_format, out_data_format, inplace, output_layout, num_out_blocks);
    }
};

struct GroupNormShardedMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    tt::tt_metal::DataType im_data_format{tt::tt_metal::DataType::INVALID};
    tt::tt_metal::DataType out_data_format{tt::tt_metal::DataType::INVALID};
    bool inplace{};
    tt::tt_metal::Layout output_layout{tt::tt_metal::Layout::INVALID};

    static constexpr auto attribute_names = std::forward_as_tuple(
        "compute_with_storage_grid_size", "im_data_format", "out_data_format", "inplace", "output_layout");
    auto attribute_values() const {
        return std::forward_as_tuple(
            compute_with_storage_grid_size, im_data_format, out_data_format, inplace, output_layout);
    }
};

using GroupNormProgramConfig = std::variant<GroupNormMultiCoreProgramConfig, GroupNormShardedMultiCoreProgramConfig>;

// Device operation types
struct GroupNormParams {
    float eps = 0.0f;
    uint32_t num_groups = 0;
    tt::tt_metal::MemoryConfig output_mem_config;
    GroupNormProgramConfig program_config;
    DeviceComputeKernelConfig compute_kernel_config;
    bool use_welford = false;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "eps", "num_groups", "output_mem_config", "program_config", "compute_kernel_config", "use_welford");
    auto attribute_values() const {
        return std::forward_as_tuple(
            eps, num_groups, output_mem_config, program_config, compute_kernel_config, use_welford);
    }
};

struct GroupNormInputs {
    Tensor input;
    std::optional<Tensor> gamma;
    std::optional<Tensor> beta;
    std::optional<Tensor> input_mask;
    std::optional<Tensor> negative_mask;
    std::optional<Tensor> reciprocals;

    static constexpr auto attribute_names =
        std::forward_as_tuple("input", "gamma", "beta", "input_mask", "negative_mask", "reciprocals");
    auto attribute_values() const {
        return std::forward_as_tuple(input, gamma, beta, input_mask, negative_mask, reciprocals);
    }
};

}  // namespace ttnn::prim
