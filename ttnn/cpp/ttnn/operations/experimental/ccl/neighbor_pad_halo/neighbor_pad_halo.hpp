// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/experimental/ccl/neighbor_pad_halo/device/neighbor_pad_halo_device_operation_types.hpp"
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn::experimental {

// Standalone halo-only neighbor-pad (no conv, no interior copy).
//
// Launches the fabric NP writer/reader kernels on boundary cores to exchange halo rows into
// `halo_buffer` (compact DRAM, pre-allocated by the caller), and returns that buffer. This is the
// fabric H+W halo exchange from neighbor_pad_conv3d with the conv3d stage stripped — pure transport,
// benchmarked toward DRAM read + fabric bandwidth.
//
// Args:
//   input        : unpadded input tensor [B, T, H, W, C] row-major bfloat16/float32
//   halo_buffer  : pre-allocated compact DRAM halo buffer [H-top | H-bot | W-left | W-right]
//   np_padding_h : halo rows per side in the H dimension (typically 1 for k=3)
//   np_padding_w : halo columns per side in the W dimension
//   np_cluster_axis : mesh axis for H-parallel devices
//   np_num_links    : number of fabric links for the H exchange
//   np_topology     : fabric topology (Linear or Ring)
//   h_neighbor_semaphore / barrier_semaphore / w_neighbor_semaphore : NP handshake semaphores
//   np_pad_dim2          : secondary (W-axis) padding dim index (must be > 0 — 2D required)
//   np_pad2_left/right   : padding amounts for the secondary dim
//   np_pad2_cluster_axis : cluster axis for the secondary dim
//   np_pad2_num_links    : links for the secondary dim
//   padding_mode         : "zeros" or "replicate"
//   memory_config        : optional output memory config (defaults to the halo_buffer's)
//
// Returns: the compact halo buffer (written in place).
ttnn::Tensor neighbor_pad_halo(
    const ttnn::Tensor& input,
    const ttnn::Tensor& halo_buffer,
    uint32_t np_padding_h,
    uint32_t np_padding_w,
    uint32_t np_cluster_axis,
    size_t np_num_links,
    ttnn::ccl::Topology np_topology,
    const GlobalSemaphore& h_neighbor_semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const GlobalSemaphore& w_neighbor_semaphore,
    uint32_t np_pad_dim2,
    uint32_t np_pad2_left,
    uint32_t np_pad2_right,
    uint32_t np_pad2_cluster_axis,
    size_t np_pad2_num_links,
    const std::string& padding_mode = "zeros",
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    uint32_t input_pad_h = 0,
    uint32_t input_pad_w = 0,
    const std::optional<ttnn::Tensor>& padded_output = std::nullopt,
    bool border_only = false);

}  // namespace ttnn::experimental
