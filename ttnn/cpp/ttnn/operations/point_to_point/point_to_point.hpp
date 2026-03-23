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

ttnn::Tensor point_to_point(
    const ttnn::Tensor& input_tensor,
    const MeshCoordinate& receiver_coord,
    const MeshCoordinate& sender_coord,
    ccl::Topology topology = ccl::Topology::Linear,
    const std::optional<ttnn::Tensor>& optional_output_tensor = std::nullopt,
    const std::optional<ttnn::Tensor>& optional_intermediate_tensor = std::nullopt);

namespace operations::point_to_point {

ttnn::TensorSpec p2p_compute_intermediate_tensor_spec(
    const ttnn::Tensor& input_tensor,
    const MeshCoordinate& receiver_coord,
    const MeshCoordinate& sender_coord,
    ccl::Topology topology);

}  // namespace operations::point_to_point

}  // namespace ttnn
