// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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

namespace ttnn::operations::experimental::conv3d {

namespace detail {
std::tuple<uint32_t, uint32_t, uint32_t> compute_output_dims(
    uint32_t T_in,
    uint32_t H_in,
    uint32_t W_in,
    const std::array<uint32_t, 3>& padding,
    const std::array<uint32_t, 3>& kernel_size);
}  // namespace detail

struct Conv3dConfig {
    Conv3dConfig(
        tt::tt_metal::DataType dtype_ = tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::DataType weights_dtype_ = tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::Layout output_layout_ = tt::tt_metal::Layout::ROW_MAJOR,
        uint32_t T_out_block_ = 1,
        uint32_t W_out_block_ = 1,
        uint32_t H_out_block_ = 1,
        uint32_t C_out_block_ = 0,
        uint32_t C_in_block_ = 0,
        uint32_t output_channels_ = 0,
        const std::array<uint32_t, 3> kernel_size_ = {1, 1, 1},
        const std::array<uint32_t, 3> stride_ = {1, 1, 1},
        const std::array<uint32_t, 3> padding_ = {0, 0, 0},
        const std::string padding_mode_ = "zeros",
        uint32_t groups_ = 1,
        CoreCoord compute_with_storage_grid_size_ = {1, 1}) :
        dtype(dtype_),
        weights_dtype(weights_dtype_),
        output_layout(output_layout_),
        T_out_block(T_out_block_),
        W_out_block(W_out_block_),
        H_out_block(H_out_block_),
        C_out_block(C_out_block_),
        C_in_block(C_in_block_),
        output_channels(output_channels_),
        kernel_size(kernel_size_),
        stride(stride_),
        padding(padding_),
        padding_mode(padding_mode_),
        groups(groups_),
        compute_with_storage_grid_size(compute_with_storage_grid_size_) {}

    tt::tt_metal::DataType dtype;
    tt::tt_metal::DataType weights_dtype;
    tt::tt_metal::Layout output_layout;
    uint32_t T_out_block;
    uint32_t W_out_block;
    uint32_t H_out_block;
    uint32_t C_out_block;
    uint32_t C_in_block;
    uint32_t output_channels;
    std::array<uint32_t, 3> kernel_size;
    std::array<uint32_t, 3> stride;
    std::array<uint32_t, 3> padding;
    std::string padding_mode;
    uint32_t groups;
    CoreCoord compute_with_storage_grid_size;

    static constexpr auto attribute_names = std::make_tuple(
        "dtype",
        "weights_dtype",
        "output_layout",
        "T_out_block",
        "W_out_block",
        "H_out_block",
        "C_out_block",
        "C_in_block",
        "output_channels",
        "kernel_size",
        "stride",
        "padding",
        "padding_mode",
        "groups",
        "compute_with_storage_grid_size");

    auto attribute_values() const {
        return std::forward_as_tuple(
            this->dtype,
            this->weights_dtype,
            this->output_layout,
            this->T_out_block,
            this->W_out_block,
            this->H_out_block,
            this->C_out_block,
            this->C_in_block,
            this->output_channels,
            this->kernel_size,
            this->stride,
            this->padding,
            this->padding_mode,
            this->groups,
            this->compute_with_storage_grid_size);
    }
};

struct Conv3dOp {
    Conv3dConfig config;
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

}  // namespace ttnn::operations::experimental::conv3d
