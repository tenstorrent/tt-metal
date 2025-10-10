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
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::operations::experimental::minimal_matmul {

struct MinimalMatmulConfig {
    MinimalMatmulConfig(
        uint32_t M_block_size_ = 1,
        uint32_t K_block_size_ = 1,
        uint32_t N_block_size_ = 1,
        uint32_t subblock_h_ = 1,
        uint32_t subblock_w_ = 1,
        CoreCoord compute_with_storage_grid_size_ = {1, 1}) :
        M_block_size(M_block_size_),
        K_block_size(K_block_size_),
        N_block_size(N_block_size_),
        subblock_h(subblock_h_),
        subblock_w(subblock_w_),
        compute_with_storage_grid_size(compute_with_storage_grid_size_) {}

    uint32_t M_block_size;
    uint32_t K_block_size;
    uint32_t N_block_size;
    uint32_t subblock_h;
    uint32_t subblock_w;

    CoreCoord compute_with_storage_grid_size;

    static constexpr auto attribute_names = std::make_tuple(
        "M_block_size", "K_block_size", "N_block_size", "subblock_h", "subblock_w", "compute_with_storage_grid_size");

    auto attribute_values() const {
        return std::forward_as_tuple(
            this->M_block_size,
            this->K_block_size,
            this->N_block_size,
            this->subblock_h,
            this->subblock_w,
            this->compute_with_storage_grid_size);
    }
};

struct MinimalMatmulOp {
    MinimalMatmulConfig config;
    std::optional<unary::UnaryWithParam> fused_activation;
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
