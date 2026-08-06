// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/neighbor_pad_conv3d/device/neighbor_pad_conv3d_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn::experimental {

// Fused NeighborPad (fabric-only H-halo) + Conv3d in a single device program
ttnn::Tensor neighbor_pad_conv3d(
    const ttnn::Tensor& input,
    const ttnn::Tensor& weight,
    const std::optional<ttnn::Tensor>& bias,
    const ttnn::Tensor& halo_buffer,
    uint32_t np_padding_h,
    uint32_t np_padding_w,
    uint32_t np_cluster_axis,
    size_t np_num_links,
    ttnn::ccl::Topology np_topology,
    const GlobalSemaphore& h_neighbor_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const GlobalSemaphore& w_neighbor_semaphore,
    uint32_t np_pad_dim2,  // 0 = no 2D padding; >0 = secondary dim index
    uint32_t np_pad2_left,
    uint32_t np_pad2_right,
    uint32_t np_pad2_cluster_axis,  // ignored when np_pad_dim2==0
    size_t np_pad2_num_links,
    const ttnn::experimental::prim::NpConv3dConfig& conv_config,
    uint32_t output_channels,
    const std::array<uint32_t, 3>& kernel_size,
    const std::array<uint32_t, 3>& stride = {1u, 1u, 1u},
    const std::array<uint32_t, 3>& padding = {0u, 0u, 0u},
    const std::array<uint32_t, 3>& dilation = {1u, 1u, 1u},
    const std::string& padding_mode = "zeros",
    uint32_t groups = 1,
    tt::tt_metal::DataType dtype = tt::tt_metal::DataType::BFLOAT16,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
