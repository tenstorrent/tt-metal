// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

// Internal slide-axis enum. Not exposed in the public Conv3dConfig.
// Resolved before launch and stored in Conv3dParams so program-cache hashing sees it.
enum class Conv3dSlideAxis : uint32_t {
    None = 0,
    W = 1,
    H = 2,
};

struct Conv3dExecutionPolicy {
    bool use_l1_prefetch = false;
    Conv3dSlideAxis slide_axis = Conv3dSlideAxis::None;

    // User-declared default ctor disables aggregate, which prevents the reflect
    // library from also matching this struct (otherwise both compile-time and
    // reflectable specializations of to_json_t / hash_object collide).
    Conv3dExecutionPolicy() = default;
    Conv3dExecutionPolicy(bool use_l1_prefetch_, Conv3dSlideAxis slide_axis_) :
        use_l1_prefetch(use_l1_prefetch_), slide_axis(slide_axis_) {}

    static constexpr auto attribute_names = std::make_tuple("use_l1_prefetch", "slide_axis");
    auto attribute_values() const {
        return std::make_tuple(this->use_l1_prefetch, static_cast<uint32_t>(this->slide_axis));
    }
};

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
    // Resolved execution policy. Set in ttnn::prim::conv3d before launch so it
    // participates in compute_program_hash, then consumed by the program factory.
    Conv3dExecutionPolicy execution_policy;
};

struct Conv3dInputs {
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
    const std::array<uint32_t, 3>& kernel_size,
    const std::array<uint32_t, 3>& dilation);
}  // namespace detail

}  // namespace ttnn::experimental::prim
