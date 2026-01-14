// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::conv3d {

struct Conv3dConfig {
    Conv3dConfig(
        tt::tt_metal::DataType weights_dtype_ = tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::Layout output_layout_ = tt::tt_metal::Layout::ROW_MAJOR,
        uint32_t T_out_block_ = 1,
        uint32_t W_out_block_ = 1,
        uint32_t H_out_block_ = 1,
        uint32_t C_out_block_ = 0,
        uint32_t C_in_block_ = 0,
        CoreCoord compute_with_storage_grid_size_ = {1, 1}) :
        weights_dtype(weights_dtype_),
        output_layout(output_layout_),
        T_out_block(T_out_block_),
        W_out_block(W_out_block_),
        H_out_block(H_out_block_),
        C_out_block(C_out_block_),
        C_in_block(C_in_block_),
        compute_with_storage_grid_size(compute_with_storage_grid_size_) {}

    tt::tt_metal::DataType weights_dtype;
    tt::tt_metal::Layout output_layout;
    uint32_t T_out_block;
    uint32_t W_out_block;
    uint32_t H_out_block;
    uint32_t C_out_block;
    uint32_t C_in_block;
    CoreCoord compute_with_storage_grid_size;

    static constexpr auto attribute_names = std::make_tuple(
        "weights_dtype",
        "output_layout",
        "T_out_block",
        "W_out_block",
        "H_out_block",
        "C_out_block",
        "C_in_block",
        "compute_with_storage_grid_size");

    auto attribute_values() const {
        return std::forward_as_tuple(
            this->weights_dtype,
            this->output_layout,
            this->T_out_block,
            this->W_out_block,
            this->H_out_block,
            this->C_out_block,
            this->C_in_block,
            this->compute_with_storage_grid_size);
    }
};

struct operation_attributes_t {
    Conv3dConfig config;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    tt::tt_metal::DataType dtype;
    uint32_t output_channels;
    std::array<uint32_t, 3> kernel_size;
    std::array<uint32_t, 3> stride;
    std::array<uint32_t, 3> padding;
    std::array<uint32_t, 3> dilation;
    std::string padding_mode;
    uint32_t groups;
};

struct tensor_args_t {
    Tensor input_tensor;
    Tensor weight_tensor;
    std::optional<const Tensor> bias_tensor;
};

namespace detail {
std::tuple<uint32_t, uint32_t, uint32_t> compute_output_dims(
    uint32_t T_in,
    uint32_t H_in,
    uint32_t W_in,
    const std::array<uint32_t, 3>& padding,
    const std::array<uint32_t, 3>& stride,
    const std::array<uint32_t, 3>& kernel_size);
}  // namespace detail

}  // namespace ttnn::operations::experimental::conv3d
