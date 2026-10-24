// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <optional>
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
 * Which mesh axes of `mesh_shape` a label says the data differs along. One placement per axis is read as is; a
 * collapsed 1-D label (what the default mappers produce) reads as Shard on every axis of size > 1 for a Shard and
 * as Replicate everywhere for a Replicate. Anything else is a label this rule cannot read: nullopt.
 */
std::optional<ttsl::SmallVector<bool>> sharded_per_mesh_axis(
    const tt::tt_metal::TensorTopology& topology, const tt::tt_metal::distributed::MeshShape& mesh_shape);

/**
 * Output-topology rule for a device operation that hands back a caller-owned tensor: an in-place variant, or a
 * preallocated output that the op overwrites (in whole or in part) from its operands. For use from an op's
 * compute_output_topologies.
 *
 * The caller's label stays while it still describes the data: no other value-affecting operand may be sharded
 * along a mesh axis on which the caller's tensor is replicated. If one is, every device now holds a different
 * result along that axis, and a label that still said Replicate would make the serialiser deduplicate shards
 * that differ and composers read one device's result as everyone's. The rule then returns nullopt, and the hook
 * returns {} so the framework's union of all operand labels (compute_output_placements_and_shape) is applied
 * instead -- data-preserving: the returned handle (and, where it aliases the storage, the caller's own handle)
 * reads as sharded, which is what the data is. A label this rule cannot read per axis, or an operand distributed
 * over a different set of mesh coordinates than the caller's tensor, also falls back to the union. `operands` may
 * contain nullptr entries (absent optionals). The fallback is logged once per process per `op_name`.
 */
std::optional<tt::tt_metal::TensorTopology> caller_owned_output_topology(
    const ttnn::Tensor& caller_tensor, std::initializer_list<const ttnn::Tensor*> operands, std::string_view op_name);

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
