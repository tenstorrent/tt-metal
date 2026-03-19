// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

namespace ttnn {
namespace operations::experimental::ccl {

// Dispatch algorithm for routing tokens to their destination devices
enum class DispatchAlgorithm : uint8_t {
    BROADCAST = 0,                   // Broadcast all tokens to ALL devices (bidirectional multicast)
    SPARSE_UNICAST = 1,              // Send to each target device individually (point-to-point)
    SPARSE_MCAST_LINEAR = 2,         // Sparse multicast in single direction
    SPARSE_MCAST_SHORTEST_PATH = 3,  // Sparse multicast with bidirectional shortest path routing (default)
    SPARSE_MCAST_SPLIT_BW = 4        // Sparse multicast, split token data 50/50 between directions
};

// Worker mode for controlling how workers are distributed across links
enum class WorkerMode : uint8_t {
    DIRECT = 0,            // Direct EDM, 1 worker per link (default)
    MUX_TOKEN_SPLIT = 1,   // Mux enabled, tokens distributed across multiple workers per link
    MUX_PAYLOAD_SPLIT = 2  // Workers on same link split token payload (not yet implemented)
};

}  // namespace operations::experimental::ccl

namespace experimental {

std::array<ttnn::Tensor, 3> all_to_all_dispatch_metadata(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_scores_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    std::optional<uint32_t> axis = std::nullopt,
    const std::optional<std::array<ttnn::Tensor, 3>>& optional_output_tensors = std::nullopt,
    std::optional<uint32_t> num_links = std::nullopt,
    const std::optional<tt::tt_metal::CoreCoord>& drain_sync_tilizer_core = std::nullopt,
    ttnn::operations::experimental::ccl::WorkerMode worker_mode =
        ttnn::operations::experimental::ccl::WorkerMode::DIRECT,
    ttnn::operations::experimental::ccl::DispatchAlgorithm dispatch_algorithm =
        ttnn::operations::experimental::ccl::DispatchAlgorithm::SPARSE_MCAST_SHORTEST_PATH,
    const std::optional<tt::tt_metal::CoreRangeSet>& worker_core_range_set = std::nullopt,
    const std::optional<tt::tt_metal::CoreRangeSet>& mux_core_range_set = std::nullopt,
    const std::optional<GlobalSemaphore>& cross_device_semaphore = std::nullopt);

}  // namespace experimental
}  // namespace ttnn
