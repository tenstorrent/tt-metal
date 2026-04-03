// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <string>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/sub_device_types.hpp>

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

    // Progress semaphore pipelining with NeighborPadAsync.
    uint32_t input_progress_t_batch_size = 0;  // 0 = disabled; must match NeighborPad's t_batch_size
    uint32_t input_progress_sem_addr = 0;      // set each call; not part of program hash

    // Halo-buffer mode: conv3d reads H-boundary rows from a compact halo buffer
    // (populated by fabric-only NeighborPad on 4 cores) instead of from the padded tensor.
    // The interior is read from the ORIGINAL unpadded input tensor.
    // use_h_halo_buffer=true sets CONV3D_H_HALO define (part of program hash).
    // h_halo_* fields are updated each call (not part of hash).
    // Sub-device: when set, conv3d program targets only these cores
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;

    // Halo buffer mode: reads H and W boundary rows from compact halo buffer
    // (populated by fabric-only NeighborPad on 4 cores) instead of the padded tensor.
    // Buffer layout: [H_top | H_bot | W_left | W_right], each T×dim sticks.
    bool use_h_halo_buffer = false;      // compile-time: enables CONV3D_H_HALO path
    uint32_t h_halo_buffer_addr = 0;     // compact halo buffer DRAM address (set each call)
    uint32_t h_halo_outer_dim_size = 0;  // outer_dim (B*T per device)
    uint32_t h_halo_H = 0;               // H_dev per device (for W halo index)
    uint32_t h_halo_W = 0;               // W sticks per H halo row
    uint32_t h_halo_padding_h = 0;       // H padding per side (1 for k3)
    uint32_t h_halo_padding_w = 0;       // W padding per side (1 for k3, 0 if no W halo)
    // Derived offsets into the compact buffer (set each call alongside h_halo_buffer_addr):
    //   H-top base = 0
    //   H-bot base = outer_dim * padding_h * W
    //   W-left base = 2 * outer_dim * padding_h * W
    //   W-right base = 2*outer_dim*pH*W + outer_dim*padding_w*H

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
