// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#pragma once
#include <tt-metalium/mesh_coord.hpp>

#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn {
namespace operations::point_to_point {

struct ExecutePointToPoint {
    static ttnn::Tensor invoke(
        QueueId queue_id,
        const ttnn::Tensor& input_tensor,
        const MeshCoordinate& receiver_coord,
        const MeshCoordinate& sender_coord,
        ccl::Topology topology,
        const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt);
};
}  // namespace operations::point_to_point

constexpr auto point_to_point =
    ttnn::register_operation<"ttnn::point_to_point", ttnn::operations::point_to_point::ExecutePointToPoint>();

}  // namespace ttnn
