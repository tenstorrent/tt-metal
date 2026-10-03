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

MeshPartitionTopology compute_mesh_partition_topology(
    const tt::tt_metal::TensorTopology& input_topology,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    uint32_t dim,
    std::optional<uint32_t> cluster_axis,
    uint32_t rank) {
    using Placement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
    using Replicate = tt::tt_metal::distributed::MeshMapperConfig::Replicate;

    const auto& input_placements = input_topology.placements();
    const auto& distribution_shape = input_topology.distribution_shape();
    const Shard shard_placement{static_cast<int>(dim)};
    const auto fallback = [](const char* reason) { return MeshPartitionTopology{std::nullopt, reason}; };
    const auto label = [](tt::tt_metal::TensorTopology topology) {
        return MeshPartitionTopology{std::move(topology), nullptr};
    };
    const auto shards_dim = [&](const Placement& placement) {
        const auto* shard = std::get_if<Shard>(&placement);
        const int signed_rank = static_cast<int>(rank);
        if (shard == nullptr || shard->dim >= signed_rank || shard->dim < -signed_rank) {
            return false;
        }
        return (shard->dim < 0 ? shard->dim + signed_rank : shard->dim) == static_cast<int>(dim);
    };
    const auto axis_size = [&](size_t axis) -> uint32_t {
        return axis < distribution_shape.dims() ? distribution_shape[static_cast<int>(axis)] : 1;
    };
    // {N},[Shard{dim}] over the coordinates the input records in row-major order: the label ShardTensorToMesh(dim)
    // produces, and the order the program factory partitions in (get_linearized_index = row * cols + col).
    const auto collapsed_label = [&]() {
        return label(tt::tt_metal::TensorTopology(
            tt::tt_metal::distributed::MeshShape(static_cast<uint32_t>(distribution_shape.mesh_size())),
            {shard_placement},
            input_topology.mesh_coords()));
    };
    // The program factory never reads the label: it picks each device's chunk from the device's own mesh coordinate
    // (coord[a] along a cluster axis, the row-major linearised index for the whole mesh). An emitted label is honest
    // only if its coordinates agree, which the mappers guarantee (SUBMESH mode writes an axis-aligned block, ROW_MAJOR
    // mode the mesh's row-major enumeration) and a label assembled by hand with update_tensor_topology need not.
    const auto& coords = input_topology.mesh_coords();
    // Every grid point's device sits, on mesh axis `mesh_axis`, at the point's own row-major index along `grid_axis`.
    const auto coords_follow_axis =
        [&](const tt::tt_metal::distributed::MeshShape& grid, size_t grid_axis, size_t mesh_axis) {
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
        };
    // The coordinates are the mesh's row-major enumeration, the order get_linearized_index partitions in.
    const auto coords_are_mesh_row_major = [&]() {
        for (size_t axis = 0; axis < mesh_shape.dims(); ++axis) {
            if (!coords_follow_axis(mesh_shape, axis, axis)) {
                return false;
            }
        }
        return true;
    };
    constexpr const char* kCoordsOffAxis =
        "the label's coordinates do not follow the cluster axis the op partitions by (not a mapper's block)";

    if (input_placements.empty()) {
        return fallback(nullptr);
    }
    // Rule 0: the label must describe the devices the partition acts on.
    if (input_placements.size() == 1) {
        // Collapsed label: no axes to reason with, so it must cover the whole mesh.
        if (distribution_shape.mesh_size() != mesh_shape.mesh_size()) {
            return fallback("the label covers fewer devices than the mesh the op partitions across");
        }
    } else if (distribution_shape != mesh_shape) {
        // Either a sub-mesh label (fits the mesh per axis: distribute_tensor's SUBMESH mode, coordinates are the
        // block's own) or a row-major reshape of the whole mesh (e.g. {1,8} over 2x4, coordinates row-major).
        bool sub_block = distribution_shape.dims() == mesh_shape.dims();
        for (size_t i = 0; sub_block && i < mesh_shape.dims(); ++i) {
            sub_block = distribution_shape[static_cast<int>(i)] <= mesh_shape[static_cast<int>(i)];
        }
        if (!cluster_axis.has_value()) {
            if (sub_block || distribution_shape.mesh_size() != mesh_shape.mesh_size()) {
                return fallback("a whole-mesh partition reaches devices outside the label");
            }
            // Row-major reshape of the whole mesh: rule 2 below is exact over its row-major coordinates.
        } else if (!sub_block) {
            return fallback("the label's axes are a reshape of the mesh, not the axes the op partitions along");
        } else if (
            *cluster_axis >= distribution_shape.dims() ||
            distribution_shape[static_cast<int>(*cluster_axis)] != mesh_shape[static_cast<int>(*cluster_axis)]) {
            return fallback("the partition groups along the cluster axis reach devices outside the label");
        }
        // Sub-mesh label whose block spans complete partition groups: rule 1 applies within the block.
    }

    // Rule 2.
    if (!cluster_axis.has_value()) {
        for (size_t axis = 0; axis < input_placements.size(); ++axis) {
            if (axis_size(axis) > 1 && std::holds_alternative<Shard>(input_placements[axis]) &&
                !shards_dim(input_placements[axis])) {
                return fallback("a whole-mesh partition of a tensor sharded on another dim is not expressible");
            }
        }
        if (!coords_are_mesh_row_major()) {
            return fallback("the label's coordinates are not the mesh's row-major order the op partitions by");
        }
        return collapsed_label();
    }
    const size_t partitioned_axis = *cluster_axis;

    // Rule 1.
    if (input_placements.size() > 1) {
        if (partitioned_axis >= input_placements.size()) {
            return fallback(nullptr);  // validation rejects this cluster_axis right after the hook
        }
        if (!coords_follow_axis(distribution_shape, partitioned_axis, partitioned_axis)) {
            return fallback(kCoordsOffAxis);
        }
        auto output_placements = input_placements;
        output_placements[partitioned_axis] = shard_placement;
        const Placement& partitioned_placement = input_placements[partitioned_axis];
        const bool partitioned_axis_composes =
            std::holds_alternative<Replicate>(partitioned_placement) || shards_dim(partitioned_placement);
        for (size_t axis = 0; axis < output_placements.size(); ++axis) {
            if (axis == partitioned_axis || !shards_dim(input_placements[axis])) {
                continue;
            }
            if (axis_size(axis) == 1) {
                output_placements[axis] = Replicate{};  // 1(i)
                continue;
            }
            bool other_axes_trivial = true;
            for (size_t other = 0; other < output_placements.size(); ++other) {
                if (other != axis && other != partitioned_axis && axis_size(other) > 1) {
                    other_axes_trivial = false;
                }
            }
            if (axis < partitioned_axis && partitioned_axis_composes && other_axes_trivial) {
                return collapsed_label();  // 1(ii)
            }
            return fallback("partitioning a dim that another mesh axis already shards is not expressible");  // 1(iii)
        }
        return label(tt::tt_metal::TensorTopology(
            distribution_shape, std::move(output_placements), input_topology.mesh_coords()));
    }

    // Collapsed {N},[p] label from the default mappers; rule 0 made N the mesh size.
    const uint32_t cluster_axis_size =
        partitioned_axis < mesh_shape.dims() ? mesh_shape[static_cast<int>(partitioned_axis)] : 0;
    // Rule 3.
    if (distribution_shape.mesh_size() == cluster_axis_size) {
        if (!coords_follow_axis(distribution_shape, 0, partitioned_axis)) {
            return fallback(kCoordsOffAxis);
        }
        return label(tt::tt_metal::TensorTopology(distribution_shape, {shard_placement}, input_topology.mesh_coords()));
    }
    // Rule 4.
    if (partitioned_axis < mesh_shape.dims() && std::holds_alternative<Replicate>(input_placements[0])) {
        if (!coords_follow_axis(mesh_shape, partitioned_axis, partitioned_axis)) {
            return fallback(kCoordsOffAxis);
        }
        ttsl::SmallVector<Placement> output_placements(mesh_shape.dims(), Replicate{});
        output_placements[partitioned_axis] = shard_placement;
        return label(
            tt::tt_metal::TensorTopology(mesh_shape, std::move(output_placements), input_topology.mesh_coords()));
    }
    // Rule 5.
    return fallback("a collapsed Shard label on a multi-axis mesh cannot also hold the partition's Shard");
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
