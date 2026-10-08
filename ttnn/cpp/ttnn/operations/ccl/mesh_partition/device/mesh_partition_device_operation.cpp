// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "ttnn/tensor/types.hpp"
#include "mesh_partition_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "cpp/ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>

namespace ttnn::operations::ccl {

namespace detail {
uint32_t get_cluster_axis_size(const ttnn::Tensor& input_tensor, const std::optional<uint32_t>& cluster_axis) {
    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();
    return cluster_axis.has_value() ? ((cluster_axis.value() == 0) ? mesh_view.num_rows() : mesh_view.num_cols())
                                    : mesh_view.num_devices();
}
}  // namespace detail

void MeshPartitionDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    uint32_t rank = input_tensor.logical_shape().rank();
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    if (tensor_args.optional_output_tensor.has_value()) {
        TT_FATAL(
            tensor_args.optional_output_tensor.value().tensor_spec() == output_spec,
            "Output tensor spec must match computed output spec");
    }
    const auto& input_shape = input_tensor.logical_shape();
    const auto& output_shape = output_spec.padded_shape();
    const auto& output_padded_shape = output_spec.padded_shape();

    TT_FATAL(
        !(operation_attributes.cluster_axis.has_value() && operation_attributes.cluster_axis.value() > 1),
        "Only support cluster axis of None, 0 or 1");

    TT_FATAL(operation_attributes.dim < rank, "dim must be less than the rank of the input tensor");

    const uint32_t cluster_axis_size = detail::get_cluster_axis_size(input_tensor, operation_attributes.cluster_axis);

    TT_FATAL(
        cluster_axis_size > 1,
        "Partition has only been tested with mesh axis size > 1, but has {} devices",
        cluster_axis_size);
    TT_FATAL(
        input_shape[operation_attributes.dim] % cluster_axis_size == 0,
        "input shape {} must be divisible by cluster axis size {}",
        input_tensor.logical_shape(),
        cluster_axis_size);

    if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
        TT_FATAL(
            output_shape[operation_attributes.dim] == output_padded_shape[operation_attributes.dim],
            "for tiled inputs, partitioning along dim {} should not create padding in output shape {}",
            operation_attributes.dim,
            output_shape);
    }
}

void MeshPartitionDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {}

MeshPartitionDeviceOperation::spec_return_value_t MeshPartitionDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto input_tensor = tensor_args.input_tensor;
    auto output_shape = input_tensor.logical_shape();

    const uint32_t cluster_axis_size = detail::get_cluster_axis_size(input_tensor, operation_attributes.cluster_axis);

    output_shape[operation_attributes.dim] = output_shape[operation_attributes.dim] / cluster_axis_size;
    return {tt::tt_metal::TensorSpec(
        Shape(output_shape),
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(),
            tt::tt_metal::PageConfig(input_tensor.layout()),
            operation_attributes.output_mem_config))};
}

MeshPartitionDeviceOperation::tensor_return_value_t MeshPartitionDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }

    auto output_spec = compute_output_specs(operation_attributes, tensor_args);

    auto tensor = create_device_tensor(output_spec, tensor_args.input_tensor.device());
    return tensor;
}

namespace detail {

namespace {

using tt::tt_metal::TensorTopology;
using tt::tt_metal::distributed::MeshCoordinate;
using tt::tt_metal::distributed::MeshShape;
using Placement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
using Replicate = tt::tt_metal::distributed::MeshMapperConfig::Replicate;

constexpr const char* kCoordsOffAxis =
    "the label's coordinates do not follow the cluster axis the op partitions by (not a mapper's block)";

MeshPartitionTopology fallback(const char* reason) { return MeshPartitionTopology{std::nullopt, reason}; }

MeshPartitionTopology label(TensorTopology topology) { return MeshPartitionTopology{std::move(topology), nullptr}; }

// Whether `placement` shards tensor dim `dim`; negative shard dims are normalised by `rank`, out-of-range ones never
// match.
bool shards_dim(const Placement& placement, uint32_t dim, uint32_t rank) {
    const auto* shard = std::get_if<Shard>(&placement);
    const int signed_rank = static_cast<int>(rank);
    if (shard == nullptr || shard->dim >= signed_rank || shard->dim < -signed_rank) {
        return false;
    }
    return (shard->dim < 0 ? shard->dim + signed_rank : shard->dim) == static_cast<int>(dim);
}

// The label's extent along `axis`; axes past its last dim count as size 1.
uint32_t axis_size(const MeshShape& distribution_shape, size_t axis) {
    return axis < distribution_shape.dims() ? distribution_shape[static_cast<int>(axis)] : 1;
}

// Whether every grid point's device sits, on mesh axis `mesh_axis`, at the point's own row-major index along
// `grid_axis`. The program factory picks each device's chunk from the device's own mesh coordinate, never from the
// label, so an emitted label is honest only if its coordinates agree (mapper-built labels always do).
bool coords_follow_axis(
    const std::vector<MeshCoordinate>& coords, const MeshShape& grid, size_t grid_axis, size_t mesh_axis) {
    if (grid_axis >= grid.dims() || coords.size() != grid.mesh_size()) {
        return false;
    }
    for (size_t flat = 0; flat < coords.size(); ++flat) {
        const auto grid_index = (flat / grid.get_stride(grid_axis)) % grid[static_cast<int>(grid_axis)];
        if (coords[flat].dims() <= mesh_axis || coords[flat][static_cast<int32_t>(mesh_axis)] != grid_index) {
            return false;
        }
    }
    return true;
}

// Whether the coordinates are the mesh's row-major enumeration, the order get_linearized_index partitions in.
bool coords_are_mesh_row_major(const std::vector<MeshCoordinate>& coords, const MeshShape& mesh_shape) {
    for (size_t axis = 0; axis < mesh_shape.dims(); ++axis) {
        if (!coords_follow_axis(coords, mesh_shape, axis, axis)) {
            return false;
        }
    }
    return true;
}

// {N},[Shard{dim}] over the input's coordinates in row-major order: the label ShardTensorToMesh(dim) produces, and the
// order the program factory partitions in (get_linearized_index = row * cols + col).
MeshPartitionTopology collapsed_shard_label(const TensorTopology& input_topology, uint32_t dim) {
    return label(TensorTopology(
        MeshShape(static_cast<uint32_t>(input_topology.distribution_shape().mesh_size())),
        {Shard{static_cast<int>(dim)}},
        input_topology.mesh_coords()));
}

// Whether an N-D label fits inside the mesh axis by axis (distribute_tensor's SUBMESH mode).
bool is_sub_block(const MeshShape& distribution_shape, const MeshShape& mesh_shape) {
    if (distribution_shape.dims() != mesh_shape.dims()) {
        return false;
    }
    for (size_t i = 0; i < mesh_shape.dims(); ++i) {
        if (distribution_shape[static_cast<int>(i)] > mesh_shape[static_cast<int>(i)]) {
            return false;
        }
    }
    return true;
}

// Rule 0: why the label does not describe the devices the partition acts on, or nullptr when it does.
const char* coverage_violation(
    size_t num_placements,
    const MeshShape& distribution_shape,
    const MeshShape& mesh_shape,
    std::optional<uint32_t> cluster_axis) {
    if (num_placements == 1) {
        // Collapsed label: no axes to reason with, so it must cover the whole mesh.
        return distribution_shape.mesh_size() == mesh_shape.mesh_size()
                   ? nullptr
                   : "the label covers fewer devices than the mesh the op partitions across";
    }
    if (distribution_shape == mesh_shape) {
        return nullptr;
    }
    const bool sub_block = is_sub_block(distribution_shape, mesh_shape);
    if (!cluster_axis.has_value()) {
        // Only a row-major reshape of the whole mesh (e.g. {1,8} over 2x4) passes: rule 2 is exact over its coords.
        return (sub_block || distribution_shape.mesh_size() != mesh_shape.mesh_size())
                   ? "a whole-mesh partition reaches devices outside the label"
                   : nullptr;
    }
    if (!sub_block) {
        return "the label's axes are a reshape of the mesh, not the axes the op partitions along";
    }
    if (*cluster_axis >= distribution_shape.dims() ||
        distribution_shape[static_cast<int>(*cluster_axis)] != mesh_shape[static_cast<int>(*cluster_axis)]) {
        return "the partition groups along the cluster axis reach devices outside the label";
    }
    return nullptr;  // sub-mesh block spanning complete partition groups: rule 1 applies within it
}

// Whether a non-trivial axis of the label shards a tensor dim other than `dim` (rule 2's restriction).
bool shards_another_dim_on_a_nontrivial_axis(
    const ttsl::SmallVector<Placement>& placements, const MeshShape& distribution_shape, uint32_t dim, uint32_t rank) {
    for (size_t axis = 0; axis < placements.size(); ++axis) {
        if (axis_size(distribution_shape, axis) > 1 && std::holds_alternative<Shard>(placements[axis]) &&
            !shards_dim(placements[axis], dim, rank)) {
            return true;
        }
    }
    return false;
}

// Rule 2 (cluster_axis=None): the collapsed {N},[Shard{dim}] label, unless another dim's Shard or the coords block it.
MeshPartitionTopology partition_whole_mesh(
    const TensorTopology& input_topology, const MeshShape& mesh_shape, uint32_t dim, uint32_t rank) {
    if (shards_another_dim_on_a_nontrivial_axis(
            input_topology.placements(), input_topology.distribution_shape(), dim, rank)) {
        return fallback("a whole-mesh partition of a tensor sharded on another dim is not expressible");
    }
    if (!coords_are_mesh_row_major(input_topology.mesh_coords(), mesh_shape)) {
        return fallback("the label's coordinates are not the mesh's row-major order the op partitions by");
    }
    return collapsed_shard_label(input_topology, dim);
}

// Rule 1: the first axis other than `partitioned_axis` with extent != 1 that also shards `dim`, if any.
std::optional<size_t> other_nontrivial_axis_sharding_dim(
    const ttsl::SmallVector<Placement>& placements,
    const MeshShape& distribution_shape,
    size_t partitioned_axis,
    uint32_t dim,
    uint32_t rank) {
    for (size_t axis = 0; axis < placements.size(); ++axis) {
        if (axis != partitioned_axis && shards_dim(placements[axis], dim, rank) &&
            axis_size(distribution_shape, axis) != 1) {
            return axis;
        }
    }
    return std::nullopt;
}

// Rule 1(ii): whether sharding `dim` on `same_dim_axis` and then on `partitioned_axis` is row-major hierarchical
// sharding, i.e. the collapsed {N},[Shard{dim}] label states it exactly.
bool same_dim_shards_collapse_row_major(
    const ttsl::SmallVector<Placement>& placements,
    const MeshShape& distribution_shape,
    size_t same_dim_axis,
    size_t partitioned_axis,
    uint32_t dim,
    uint32_t rank) {
    const Placement& partitioned_placement = placements[partitioned_axis];
    const bool partitioned_axis_composes =
        std::holds_alternative<Replicate>(partitioned_placement) || shards_dim(partitioned_placement, dim, rank);
    bool other_axes_trivial = true;
    for (size_t other = 0; other < placements.size(); ++other) {
        if (other != same_dim_axis && other != partitioned_axis && axis_size(distribution_shape, other) > 1) {
            other_axes_trivial = false;
        }
    }
    return same_dim_axis < partitioned_axis && partitioned_axis_composes && other_axes_trivial;
}

// Rule 1 (N-D label, cluster axis a): Shard{dim} on a, and 1(i)-(iii) for another axis that also shards dim.
MeshPartitionTopology partition_nd_label(
    const TensorTopology& input_topology, size_t partitioned_axis, uint32_t dim, uint32_t rank) {
    const auto& placements = input_topology.placements();
    const auto& distribution_shape = input_topology.distribution_shape();
    if (partitioned_axis >= placements.size()) {
        return fallback(nullptr);  // validation rejects this cluster_axis right after the hook
    }
    if (!coords_follow_axis(input_topology.mesh_coords(), distribution_shape, partitioned_axis, partitioned_axis)) {
        return fallback(kCoordsOffAxis);
    }
    if (const auto same_dim_axis =
            other_nontrivial_axis_sharding_dim(placements, distribution_shape, partitioned_axis, dim, rank)) {
        if (same_dim_shards_collapse_row_major(
                placements, distribution_shape, *same_dim_axis, partitioned_axis, dim, rank)) {
            return collapsed_shard_label(input_topology, dim);  // 1(ii)
        }
        return fallback("partitioning a dim that another mesh axis already shards is not expressible");  // 1(iii)
    }
    auto output_placements = placements;
    output_placements[partitioned_axis] = Shard{static_cast<int>(dim)};
    for (size_t axis = 0; axis < output_placements.size(); ++axis) {
        if (axis != partitioned_axis && shards_dim(placements[axis], dim, rank)) {
            output_placements[axis] = Replicate{};  // 1(i): only size-1 axes are left sharding dim here
        }
    }
    return label(TensorTopology(distribution_shape, std::move(output_placements), input_topology.mesh_coords()));
}

// Rule 3 (collapsed label along the cluster axis): {N},[Shard{dim}] whatever it held, if the coords follow that axis.
MeshPartitionTopology partition_collapsed_line(
    const TensorTopology& input_topology, size_t partitioned_axis, uint32_t dim) {
    if (!coords_follow_axis(input_topology.mesh_coords(), input_topology.distribution_shape(), 0, partitioned_axis)) {
        return fallback(kCoordsOffAxis);
    }
    return label(TensorTopology(
        input_topology.distribution_shape(), {Shard{static_cast<int>(dim)}}, input_topology.mesh_coords()));
}

// Rule 4 (collapsed Replicate on a multi-axis mesh): uncollapse to the mesh, Shard{dim} on the cluster axis.
MeshPartitionTopology uncollapse_replicate(
    const TensorTopology& input_topology, const MeshShape& mesh_shape, size_t partitioned_axis, uint32_t dim) {
    if (!coords_follow_axis(input_topology.mesh_coords(), mesh_shape, partitioned_axis, partitioned_axis)) {
        return fallback(kCoordsOffAxis);
    }
    ttsl::SmallVector<Placement> output_placements(mesh_shape.dims(), Replicate{});
    output_placements[partitioned_axis] = Shard{static_cast<int>(dim)};
    return label(TensorTopology(mesh_shape, std::move(output_placements), input_topology.mesh_coords()));
}

}  // namespace

MeshPartitionTopology compute_mesh_partition_topology(
    const tt::tt_metal::TensorTopology& input_topology,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    uint32_t dim,
    std::optional<uint32_t> cluster_axis,
    uint32_t rank) {
    const auto& placements = input_topology.placements();
    const auto& distribution_shape = input_topology.distribution_shape();
    if (placements.empty()) {
        return fallback(nullptr);
    }
    // Rule 0: the label must describe the devices the partition acts on.
    if (const char* reason = coverage_violation(placements.size(), distribution_shape, mesh_shape, cluster_axis)) {
        return fallback(reason);
    }
    // Rule 2: whole-mesh partition.
    if (!cluster_axis.has_value()) {
        return partition_whole_mesh(input_topology, mesh_shape, dim, rank);
    }
    const size_t partitioned_axis = *cluster_axis;
    // Rule 1: N-D label.
    if (placements.size() > 1) {
        return partition_nd_label(input_topology, partitioned_axis, dim, rank);
    }
    // Rules 3-5: collapsed {N},[p] label from the default mappers; rule 0 made N the mesh size.
    const uint32_t cluster_axis_size =
        partitioned_axis < mesh_shape.dims() ? mesh_shape[static_cast<int>(partitioned_axis)] : 0;
    if (distribution_shape.mesh_size() == cluster_axis_size) {
        return partition_collapsed_line(input_topology, partitioned_axis, dim);  // Rule 3
    }
    if (partitioned_axis < mesh_shape.dims() && std::holds_alternative<Replicate>(placements[0])) {
        return uncollapse_replicate(input_topology, mesh_shape, partitioned_axis, dim);  // Rule 4
    }
    return fallback("a collapsed Shard label on a multi-axis mesh cannot also hold the partition's Shard");  // Rule 5
}

}  // namespace detail

std::vector<tt::tt_metal::TensorTopology> MeshPartitionDeviceOperation::compute_output_topologies(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Runs before validation, which is where a host-resident input is rejected: do not reach for its device here.
    const auto& input_tensor = tensor_args.input_tensor;
    if (!ttnn::is_device_tensor(input_tensor)) {
        return {};
    }
    const auto result = detail::compute_mesh_partition_topology(
        input_tensor.tensor_topology(),
        input_tensor.device()->get_view().shape(),
        operation_attributes.dim,
        operation_attributes.cluster_axis,
        input_tensor.logical_shape().rank());
    if (result.topology.has_value()) {
        return {*result.topology};
    }
    if (result.fallback_reason != nullptr) {
        // Once per distinct (dim, cluster_axis, distribution shape, reason): the hook runs on every launch, program
        // cache hits included, and the framework silenced its own union-rule warnings for that spam (#25340).
        static std::mutex warned_mutex;
        static std::unordered_set<std::string> warned;
        std::string message = fmt::format(
            "mesh_partition(dim={}, cluster_axis={}) on an input distributed over {}: {}; the output keeps the input's "
            "TensorTopology, which does not describe the partitioned result",
            operation_attributes.dim,
            operation_attributes.cluster_axis.has_value() ? static_cast<int>(*operation_attributes.cluster_axis) : -1,
            input_tensor.tensor_topology().distribution_shape(),
            result.fallback_reason);
        bool first_occurrence = false;
        {
            std::lock_guard<std::mutex> lock(warned_mutex);
            first_occurrence = warned.insert(message).second;
        }
        if (first_occurrence) {
            log_warning(tt::LogOp, "{} (logged once per process for this case)", message);
        }
    }
    return {};  // union default = the input's label
}

}  // namespace ttnn::operations::ccl

namespace ttnn::prim {
ttnn::Tensor mesh_partition(
    const ttnn::Tensor& input_tensor,
    int32_t dim,
    std::optional<uint32_t> cluster_axis,
    const ttnn::MemoryConfig& memory_config,
    const std::optional<ttnn::Tensor>& optional_output_tensor) {
    using OperationType = ttnn::operations::ccl::MeshPartitionDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .dim = (dim < 0 ? uint32_t(input_tensor.logical_shape().rank() + dim) : (uint32_t)dim),
            .cluster_axis = cluster_axis,
            .output_mem_config = memory_config,
        },
        OperationType::tensor_args_t{.input_tensor = input_tensor, .optional_output_tensor = optional_output_tensor});
}
}  // namespace ttnn::prim
