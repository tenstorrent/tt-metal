// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <string>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct Conv3dConfig {
    Conv3dConfig(
        tt::tt_metal::DataType weights_dtype_ = tt::tt_metal::DataType::BFLOAT16,
        tt::tt_metal::Layout output_layout_ = tt::tt_metal::Layout::ROW_MAJOR,
        uint32_t T_out_block_ = 1,
        uint32_t W_out_block_ = 1,
        uint32_t H_out_block_ = 1,
        uint32_t C_out_block_ = 0,
        uint32_t C_in_block_ = 0,
        std::array<uint32_t, 3> dilation_ = {1, 1, 1},
        uint32_t alignment_ = 32,
        CoreCoord compute_with_storage_grid_size_ = {1, 1}) :
        weights_dtype(weights_dtype_),
        output_layout(output_layout_),
        T_out_block(T_out_block_),
        W_out_block(W_out_block_),
        H_out_block(H_out_block_),
        C_out_block(C_out_block_),
        C_in_block(C_in_block_),
        dilation(dilation_),
        alignment(alignment_),
        compute_with_storage_grid_size(compute_with_storage_grid_size_) {}

    tt::tt_metal::DataType weights_dtype;
    tt::tt_metal::Layout output_layout;
    uint32_t T_out_block;
    uint32_t W_out_block;
    uint32_t H_out_block;
    uint32_t C_out_block;
    uint32_t C_in_block;
    std::array<uint32_t, 3> dilation;
    uint32_t alignment;
    CoreCoord compute_with_storage_grid_size;

    static constexpr auto attribute_names = std::make_tuple(
        "weights_dtype",
        "output_layout",
        "T_out_block",
        "W_out_block",
        "H_out_block",
        "C_out_block",
        "C_in_block",
        "dilation",
        "alignment",
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
            this->dilation,
            this->alignment,
            this->compute_with_storage_grid_size);
    }
};

struct Conv3dParams {
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
    // Logical-pad masking (opt-in): zero interior sticks whose global spatial index is >= logical_*_mask,
    // so a persistent-padded plain conv masks its logical-pad positions in-kernel instead of a pre-mask mul.
    // 0 == disabled (byte-identical for every other conv3d caller).
    uint32_t logical_h_mask = 0;
    uint32_t logical_w_mask = 0;
    // Padded-output mode (opt-in): the writer places the [H_out,W_out] result into the interior of a
    // spatially padded [H_out+2*output_pad_h, W_out+2*output_pad_w] output buffer. 0 == compact output.
    uint32_t output_pad_h = 0;
    uint32_t output_pad_w = 0;
};

struct Conv3dInputs {
    Tensor input_tensor;
    Tensor weight_tensor;
    std::optional<const Tensor> bias_tensor;
    // Per-device [h_start, w_start] global spatial offset (uint32, one page per device), read by the
    // reader to evaluate the logical-pad mask when masking is enabled.
    std::optional<const Tensor> pad_offset_tensor;
};

namespace detail {
std::tuple<uint32_t, uint32_t, uint32_t> compute_output_dims(
    uint32_t T_in,
    uint32_t H_in,
    uint32_t W_in,
    const std::array<uint32_t, 3>& padding,
    const std::array<uint32_t, 3>& stride,
    const std::array<uint32_t, 3>& kernel_size,
    const std::array<uint32_t, 3>& dilation);
}  // namespace detail

}  // namespace ttnn::experimental::prim
