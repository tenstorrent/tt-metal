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
 * The label of an output that is a function of several labelled operands -- a fused matmul of a collective's result
 * by a weight, an elementwise op on a reduce_scatter result -- as a TensorTopology over `mesh_coords`: the union
 * above, with `operands.front()` the label the operation computed itself (the collective's result, or the matmul
 * label a fused reduce_scatter consumes) and the rest the other operands' labels. This is what every fused
 * experimental CCL hook spells, so a rule for how such unions behave has one home.
 *
 * The one rule beyond the union's: the first operand's distribution rank is the result's whenever that operand is
 * sharded, and sharded operands of another rank are left out. The union keeps only the labels of the highest rank
 * among the sharded ones, so a collapsed {N},[Shard{d}] reduce_scatter result unioned with N-D [Replicate, Shard{d}]
 * elementwise operands would otherwise read [Replicate, Shard{d}]: Replicate on an axis whose devices hold different
 * pieces, data loss when serialised (uncollapsing it first, [Shard{d}, Shard{d}], reads [Shard{d}, Replicate]: the same
 * over-claim on the other axis). A collapsed label already says every device holds a distinct piece, which no other
 * operand can make less distinct; an N-D first operand drops exactly the operands the union would have dropped. A
 * fully replicated first operand decides nothing, as in the union.
 */
tt::tt_metal::TensorTopology union_topology(
    const std::vector<std::reference_wrapper<const tt::tt_metal::TensorTopology>>& operands,
    std::vector<tt::tt_metal::distributed::MeshCoordinate> mesh_coords);

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
