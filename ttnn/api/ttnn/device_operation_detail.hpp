// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>
#include <string_view>
#include <utility>
#include <vector>

#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/small_vector.hpp>
#include <ttnn/distributed/distributed_configs.hpp>

namespace ttnn {
class Tensor;
}  // namespace ttnn

namespace tt::tt_metal {
namespace distributed {
class MeshDevice;
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
    ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>,
    tt::tt_metal::distributed::MeshShape>
compute_output_placements_and_shape(const std::vector<std::reference_wrapper<const ttnn::Tensor>>& tensors);

/**
 * Non-template implementation of tensor coordinate extraction.
 *
 * Extracts and validates mesh coordinates from a pre-extracted list of input tensors.
 */
std::vector<tt::tt_metal::distributed::MeshCoordinate> extract_tensor_coordinates_impl(
    const std::vector<std::reference_wrapper<const ttnn::Tensor>>& tensors,
    tt::tt_metal::distributed::MeshDevice* mesh_device);

/**
 * Fail if `tensor` is per-core allocated. Called by launch() for every tensor in tensor_args when
 * the operation does not satisfy SupportsPerCoreAllocation.
 *
 * `input_index` is the tensor's position in the visit order of tensor_args, reported so the
 * message names which one of an op's several tensors is at fault.
 *
 * tensor_args holds preallocated output tensors as well as inputs, so those are covered too --
 * an op that cannot resolve per-core addresses cannot write to a per-core output either. What is
 * *not* covered is the output MemoryConfig requested through operation_attributes: reaching it
 * would mean walking an attributes struct, and the non-matching overload of
 * ttsl::reflection::visit_object_of_type_t throws on any leaf that is neither the target type
 * nor reflectable. Fine for tensor_args, which holds nothing but tensors; not for attributes
 * structs full of scalars. An op that rebuilds its output MemoryConfig from named fields can
 * therefore still drop the per-core bit unnoticed; tracked in #51482.
 */
void validate_no_per_core_allocation(const ttnn::Tensor& tensor, std::string_view operation_name, size_t input_index);

}  // namespace ttnn::device_operation::detail
