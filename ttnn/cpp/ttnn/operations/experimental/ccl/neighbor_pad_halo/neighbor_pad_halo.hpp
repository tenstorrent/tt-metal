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

// Standalone halo-only neighbor-pad (no conv, no interior copy)
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
