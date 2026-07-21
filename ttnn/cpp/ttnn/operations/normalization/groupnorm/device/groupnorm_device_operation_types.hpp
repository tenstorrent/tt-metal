// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/core_coord.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

// Program config types
struct GroupNormMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    tt::tt_metal::DataType im_data_format{tt::tt_metal::DataType::INVALID};
    tt::tt_metal::DataType out_data_format{tt::tt_metal::DataType::INVALID};
    bool inplace{};
    tt::tt_metal::Layout output_layout{tt::tt_metal::Layout::INVALID};
    // Number of chunks to split the per-core output height (block_h) into.
    // Smaller chunks reduce per-iteration SRAM use at the cost of perf.
    // Sentinel value -1 means "auto": the program factory picks a value
    // using its built-in heuristic (see groupnorm_{mcast,no_mcast}_program_factory.cpp).
    // Any non-negative value is taken as an explicit chunk count and must
    // satisfy 1 <= num_out_blocks <= block_h after grid sizing.
    int num_out_blocks{};
};

struct GroupNormShardedMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    tt::tt_metal::DataType im_data_format{tt::tt_metal::DataType::INVALID};
    tt::tt_metal::DataType out_data_format{tt::tt_metal::DataType::INVALID};
    bool inplace{};
    tt::tt_metal::Layout output_layout{tt::tt_metal::Layout::INVALID};
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
};

struct GroupNormInputs {
    Tensor input;
    std::optional<Tensor> gamma;
    std::optional<Tensor> beta;
    std::optional<Tensor> input_mask;
    std::optional<Tensor> negative_mask;
    std::optional<Tensor> reciprocals;
};

}  // namespace ttnn::prim
