// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <string_view>
#include <utility>
#include <vector>

#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/small_vector.hpp>
#include <ttnn/distributed/distributed_configs.hpp>

namespace tt::tt_metal {
class Tensor;
namespace distributed {
class MeshDevice;
class MeshWorkload;
}  // namespace distributed
}  // namespace tt::tt_metal

namespace ttnn::device_operation::detail {

/**
 * Non-template implementation of output placement and shape computation.
 *
 * This function computes the output tensor topology (placements and distribution shape)
 * from a pre-extracted list of input tensors, avoiding the need to template on the
 * operation's tensor_args_t type.
 *
 * Factored out of the template pipeline to reduce per-operation template instantiation cost.
 */
std::pair<
    tt::stl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>,
    tt::tt_metal::distributed::MeshShape>
compute_output_placements_and_shape(
    const std::vector<std::reference_wrapper<const tt::tt_metal::Tensor>>& tensors,
    const tt::tt_metal::Tensor& first_tensor);

/**
 * Non-template implementation of mesh workload annotation for the inspector.
 *
 * Formats tensor arguments and emits annotation metadata without requiring
 * the operation type template parameter.
 */
void emit_mesh_workload_annotation_impl(
    tt::tt_metal::distributed::MeshWorkload& workload,
    std::string_view operation_name,
    const std::vector<std::reference_wrapper<const tt::tt_metal::Tensor>>& tensors);

/**
 * Non-template implementation of tensor coordinate extraction.
 *
 * Extracts and validates mesh coordinates from a pre-extracted list of input tensors.
 */
std::vector<tt::tt_metal::distributed::MeshCoordinate> extract_tensor_coordinates_impl(
    const std::vector<std::reference_wrapper<const tt::tt_metal::Tensor>>& tensors,
    tt::tt_metal::distributed::MeshDevice* mesh_device);

}  // namespace ttnn::device_operation::detail
