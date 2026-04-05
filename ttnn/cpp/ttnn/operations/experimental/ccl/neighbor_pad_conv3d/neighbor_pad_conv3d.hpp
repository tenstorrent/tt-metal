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
#include "ttnn/operations/experimental/conv3d/device/conv3d_device_operation_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn::experimental {

// Fused NeighborPad (fabric-only H-halo) + Conv3d in a single device program.
//
// The operation:
//   1. Launches fabric NP writer/reader kernels on boundary cores to exchange halo rows
//      into `halo_buffer` (compact DRAM, pre-allocated by the caller).
//   2. Launches Conv3d compute/reader/writer kernels concurrently; boundary T-slices
//      synchronise via the halo buffer rather than waiting for a fully-padded tensor.
//
// Args:
//   input        : unpadded input tensor [B, T, H, W, C] row-major bfloat16/float32
//   weight       : prepared conv3d weights [kD*kH*kW*C_in, C_out] tiled
//   bias         : optional bias [1, C_out] tiled
//   halo_buffer  : pre-allocated compact DRAM halo buffer (see NP design docs)
//   np_padding_h : halo rows per side in the H dimension (typically 1 for k=3)
//   np_padding_w : halo columns per side in the W dimension (0 if not needed)
//   np_cluster_axis : mesh axis for H-parallel devices
//   np_num_links    : number of fabric links for NP
//   np_topology     : fabric topology (Linear or Ring)
//   h_neighbor_semaphore : GlobalSemaphore for H-neighbor handshake
//   barrier_semaphore    : GlobalSemaphore for NP barrier
//   w_neighbor_semaphore : GlobalSemaphore for W-neighbor handshake
//   np_pad_dim2          : optional secondary padding dimension index
//   np_pad2_left/right   : padding amounts for secondary dimension
//   np_pad2_cluster_axis : cluster axis for secondary dimension
//   np_pad2_num_links    : links for secondary dimension
//   conv_config          : Conv3dConfig (blocking, grid, dtype, halo flags)
//   output_channels      : number of output feature channels
//   kernel_size          : [kD, kH, kW]
//   stride               : [sD, sH, sW] (default {1,1,1})
//   padding              : [pD, pH, pW] excluding halo (default {0,0,0})
//   dilation             : [dD, dH, dW] (default {1,1,1})
//   padding_mode         : "zeros" or "replicate"
//   groups               : depthwise-group count (default 1)
//   dtype                : output dtype
//   compute_kernel_config: optional device compute config
//   memory_config        : optional output memory config
//
// Returns: output tensor [B, T_out, H_out, W_out, C_out]
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
    const ttnn::experimental::prim::Conv3dConfig& conv_config,
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
