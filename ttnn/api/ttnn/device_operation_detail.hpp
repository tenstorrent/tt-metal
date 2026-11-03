// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/small_vector.hpp>
#include <ttnn/distributed/distributed_configs.hpp>

namespace ttnn {
class Tensor;
}  // namespace ttnn

namespace tt::tt_metal {
class TensorTopology;
}  // namespace tt::tt_metal

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

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
 * The same union over TensorTopology labels instead of tensors. The tensor overload above is this function
 * applied to each tensor's tensor_topology(); it exists so a device operation that knows what one of its
 * operands WILL be labelled (a collective's result feeding a fused matmul) can take the union of that label
 * with the labels of its other operands, rather than re-implementing the rule. Semantics: among the labels of
 * the highest distribution rank, the first Shard seen on a mesh axis wins and a tensor dim already claimed by
 * another axis reads Replicate; the distribution shape is the per-axis maximum; lower-rank sharded labels are
 * dropped; fully-replicated labels only decide the rank when nothing is sharded.
 */
std::pair<
    ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>,
    tt::tt_metal::distributed::MeshShape>
compute_output_placements_and_shape(
    const std::vector<std::reference_wrapper<const tt::tt_metal::TensorTopology>>& topologies);

/**
 * Which mesh axes of `mesh_shape` a label says the data differs along. One placement per axis is read as is; a
 * collapsed 1-D label reads as Shard on every axis of size > 1 for a Shard and as Replicate everywhere for a
 * Replicate. Anything else is a label this rule cannot read: nullopt.
 */
std::optional<ttsl::SmallVector<bool>> sharded_per_mesh_axis(
    const tt::tt_metal::TensorTopology& topology, const tt::tt_metal::distributed::MeshShape& mesh_shape);

/**
 * The label of an output that is the caller's own tensor (an in-place result or a preallocated output): the caller's
 * label while it still describes the data, nullopt otherwise. It stops describing the data when an operand is
 * sharded along a mesh axis on which the caller's tensor is replicated (every device then holds a different result
 * there), when a label cannot be read per mesh axis, or when an operand is distributed over different mesh
 * coordinates; the output then takes the union of the operand labels. `operands` may hold nullptr (an absent
 * optional). The fallback is logged once per process per `op_name`.
 */
std::optional<tt::tt_metal::TensorTopology> caller_owned_output_topology(
    const ttnn::Tensor& caller_tensor, std::span<const ttnn::Tensor* const> operands, std::string_view op_name);

inline std::optional<tt::tt_metal::TensorTopology> caller_owned_output_topology(
    const ttnn::Tensor& caller_tensor, std::initializer_list<const ttnn::Tensor*> operands, std::string_view op_name) {
    return caller_owned_output_topology(
        caller_tensor, std::span<const ttnn::Tensor* const>(operands.begin(), operands.size()), op_name);
}

/**
 * Logs, once per process per `op_name`, that the operation declares a compute_output_topologies the framework does not
 * call (HasIgnoredComputeOutputTopologies in ttnn/operation_concepts.hpp), so the default labels its outputs.
 */
void warn_ignored_compute_output_topologies(std::string_view op_name);

/**
 * The labels an operation without its own compute_output_topologies gets. An output that shares its device buffer
 * with a tensor in tensor_args is the caller's tensor (in place or preallocated) and keeps that tensor's label while
 * the rule of caller_owned_output_topology accepts it against the other tensors; every other output, and a caller's
 * tensor the rule rejects, takes the union of all of them (compute_output_placements_and_shape) on its own mesh
 * coordinates.
 */
void apply_default_output_topologies(
    const std::vector<std::reference_wrapper<ttnn::Tensor>>& outputs,
    const std::vector<std::reference_wrapper<const ttnn::Tensor>>& inputs,
    std::string_view op_name);

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
