// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>

#include "ttnn/device_operation_detail.hpp"

#include <algorithm>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/per_core_allocation/buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/small_vector.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::device_operation::detail {

// Bring mesh coordinate types into scope for readability.
using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;
using MeshCoordinateRange = tt::tt_metal::distributed::MeshCoordinateRange;
using MeshCoordinateRangeSet = tt::tt_metal::distributed::MeshCoordinateRangeSet;

static bool is_fully_replicated(const ttnn::Tensor& tensor) {
    for (const auto& placement : tensor.tensor_topology().placements()) {
        if (std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(placement)) {
            return false;
        }
    }
    return true;
}

// Factored from the former template get_output_placements_and_shape<device_operation_t>. The only
// reason it was templated was to call visit_object_of_type<Tensor> on tensor_args; callers now
// extract the tensors first and pass them as a vector.
std::pair<
    ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>,
    tt::tt_metal::distributed::MeshShape>
compute_output_placements_and_shape(const std::vector<std::reference_wrapper<const ttnn::Tensor>>& tensors) {
    using Tensor = ttnn::Tensor;
    using Placement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
    using Replicate = tt::tt_metal::distributed::MeshMapperConfig::Replicate;

    TT_FATAL(!tensors.empty(), "Cannot compute output placements and shape with no tensors");

    std::vector<std::reference_wrapper<const Tensor>> sharded_tensors;
    sharded_tensors.reserve(tensors.size());
    for (const auto& tensor_ref : tensors) {
        if (!is_fully_replicated(tensor_ref.get())) {
            sharded_tensors.push_back(tensor_ref);
        }
    }

    // Compute max distribution rank: use only sharded tensors if they exist, otherwise use all tensors (fully
    // replicated)
    size_t max_distribution_rank = 0;
    if (!sharded_tensors.empty()) {
        for (const auto& tensor_ref : sharded_tensors) {
            max_distribution_rank =
                std::max(max_distribution_rank, tensor_ref.get().tensor_topology().distribution_shape().dims());
        }
    } else {
        const auto& first_tensor = tensors.front().get();
        max_distribution_rank = first_tensor.tensor_topology().distribution_shape().dims();
    }

    auto result_strides = ttsl::SmallVector<uint32_t>(max_distribution_rank, 1);
    auto result_placements = ttsl::SmallVector<Placement>(max_distribution_rank, Replicate{});
    std::unordered_map<int, int> shard_dim_to_distribution_dim;
    bool dim_mismatch = false;

    // TODO: #25340 - Add back logging / validation. Currently, this results in a lot of log spam.
    constexpr bool kEnableLogging = false;
    for (const auto& tensor_ref : tensors) {
        const Tensor& tensor = tensor_ref.get();
        // Augment output tensor distribution shape with the max strides of all input tensors with the max
        // distribution rank
        const auto& tensor_distribution_shape = tensor.tensor_topology().distribution_shape();
        if (tensor_distribution_shape.dims() == max_distribution_rank) {
            for (size_t i = 0; i < std::min(result_strides.size(), tensor_distribution_shape.dims()); i++) {
                result_strides[i] = std::max(result_strides[i], tensor_distribution_shape[i]);
            }

            const auto& tensor_placements = tensor.tensor_topology().placements();
            for (size_t i = 0; i < tensor_placements.size(); i++) {
                Placement output_placement = result_placements[i];
                if (std::holds_alternative<Shard>(tensor_placements[i])) {
                    auto new_shard_placement = std::get<Shard>(tensor_placements[i]);

                    // Only shard if the tensor dimension is not already sharded
                    if (!shard_dim_to_distribution_dim.contains(new_shard_placement.dim)) {
                        shard_dim_to_distribution_dim.insert({new_shard_placement.dim, static_cast<int>(i)});
                        if (std::holds_alternative<Shard>(output_placement)) {
                            auto existing_shard_placement = std::get<Shard>(output_placement);

                            // If a different tensor dim is sharded across this distribution dim, keep the
                            // earliest-seen shard dimension.
                            if (new_shard_placement.dim != existing_shard_placement.dim && kEnableLogging) {
                                log_warning(
                                    tt::LogOp,
                                    "Output tensor cannot shard different tensor dimensions across the same "
                                    "distribution "
                                    "dimension: tensor dims {} (kept) and {} (ignored) across distribution dim {}",
                                    existing_shard_placement.dim,
                                    new_shard_placement.dim,
                                    i);
                            }
                            continue;
                        }
                        output_placement = new_shard_placement;
                    } else if (
                        shard_dim_to_distribution_dim.at(new_shard_placement.dim) != static_cast<int>(i) &&
                        kEnableLogging) {
                        log_warning(
                            tt::LogOp,
                            "Duplicate tensor shard dimension {} across distribution dim {} replaced with "
                            "Replicate",
                            new_shard_placement.dim,
                            i);
                    }
                }
                result_placements[i] = output_placement;
            }
        } else if (!is_fully_replicated(tensor)) {
            dim_mismatch = true;
        }
    }
    if (dim_mismatch && kEnableLogging) {
        log_warning(
            tt::LogOp,
            "Input tensors have different distribution ranks, only imputing output tensor topology with tensors that "
            "have the max distribution rank");
    }
    return {std::move(result_placements), tt::tt_metal::distributed::MeshShape(std::move(result_strides))};
}

// Checks if the MeshCoordinateRangeSet containing all coordinates in b is a subset of a.
static bool is_subset_of(const std::vector<MeshCoordinate>& a, const std::vector<MeshCoordinate>& b) {
    MeshCoordinateRangeSet a_set;
    MeshCoordinateRangeSet b_set;

    for (const auto& coord : a) {
        a_set.merge(MeshCoordinateRange(coord));
    }
    for (const auto& coord : b) {
        b_set.merge(MeshCoordinateRange(coord));
    }

    bool is_subset = false;
    for (const auto& b_range : b_set.ranges()) {
        is_subset = false;
        for (const auto& a_range : a_set.ranges()) {
            if (a_range.contains(b_range)) {
                is_subset = true;
                break;
            }
        }
        if (not is_subset) {
            return is_subset;
        }
    }
    return is_subset;
}

std::vector<MeshCoordinate> extract_tensor_coordinates_impl(
    const std::vector<std::reference_wrapper<const ttnn::Tensor>>& tensors,
    tt::tt_metal::distributed::MeshDevice* mesh_device) {
    using Tensor = ttnn::Tensor;

    // If no tensor is found, return zero coordinate
    if (tensors.empty()) {
        if (mesh_device == nullptr) {
            TT_THROW("No tensors found in tensor_args and no mesh_device provided to extract_tensor_coordinates");
        }
        return {MeshCoordinate::zero_coordinate(mesh_device->shape().dims())};
    }

    const Tensor& first_tensor = tensors.front().get();
    std::vector<ttnn::MeshCoordinate> tensor_coordinates;
    tensor_coordinates.reserve(first_tensor.device_storage().get_coords().size());
    std::transform(
        first_tensor.device_storage().get_coords().begin(),
        first_tensor.device_storage().get_coords().end(),
        std::back_inserter(tensor_coordinates),
        [](const auto& coord) { return coord; });

    // Verification Step: Assert if the tensors are placed on different coordinate ranges
    // that do not overlap.
    for (const auto& tensor_ref : tensors) {
        const Tensor& tensor = tensor_ref.get();
        if (tensor.device_storage().get_coords().size() != tensor_coordinates.size()) {
            std::vector<ttnn::MeshCoordinate> tensor_mesh_coords;
            tensor_mesh_coords.reserve(tensor.device_storage().get_coords().size());
            std::transform(
                tensor.device_storage().get_coords().begin(),
                tensor.device_storage().get_coords().end(),
                std::back_inserter(tensor_mesh_coords),
                [](const auto& coord) { return coord; });
            if (tensor_mesh_coords.size() < tensor_coordinates.size()) {
                TT_ASSERT(
                    is_subset_of(tensor_coordinates, tensor_mesh_coords),
                    "Tensors are placed on different MeshCoordinate ranges that do not intersect.");
                tensor_coordinates = std::move(tensor_mesh_coords);
            } else {
                TT_ASSERT(
                    is_subset_of(tensor_mesh_coords, tensor_coordinates),
                    "Tensors are placed on different MeshCoordinate ranges that do not intersect.");
            }
        }
    }
    return tensor_coordinates;
}

std::optional<ttsl::SmallVector<bool>> sharded_per_mesh_axis(
    const tt::tt_metal::TensorTopology& topology, const tt::tt_metal::distributed::MeshShape& mesh_shape) {
    const auto is_shard = [](const auto& placement) {
        return std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(placement);
    };
    const auto& placements = topology.placements();
    const size_t dims = mesh_shape.dims();
    ttsl::SmallVector<bool> sharded(dims, false);
    if (placements.size() == dims) {
        for (size_t axis = 0; axis < dims; ++axis) {
            sharded[axis] = is_shard(placements[axis]);
        }
        return sharded;
    }
    if (placements.size() == 1) {
        const bool shard = is_shard(placements[0]);
        for (size_t axis = 0; axis < dims; ++axis) {
            sharded[axis] = shard && mesh_shape[static_cast<int>(axis)] > 1;
        }
        return sharded;
    }
    return std::nullopt;
}

std::optional<tt::tt_metal::TensorTopology> caller_owned_output_topology(
    const ttnn::Tensor& caller_tensor, std::initializer_list<const ttnn::Tensor*> operands, std::string_view op_name) {
    const auto& caller_topology = caller_tensor.tensor_topology();
    if (caller_tensor.device() == nullptr) {
        return caller_topology;  // not a mesh tensor: nothing to compare against
    }
    const auto& mesh_shape = caller_tensor.device()->shape();
    const auto caller_sharded = sharded_per_mesh_axis(caller_topology, mesh_shape);

    std::string_view reason;
    for (const ttnn::Tensor* operand : operands) {
        if (operand == nullptr || operand == &caller_tensor) {
            continue;
        }
        const auto& operand_topology = operand->tensor_topology();
        if (operand_topology.mesh_coords() != caller_topology.mesh_coords()) {
            reason = "an operand is distributed over a different set of mesh coordinates";
            break;
        }
        const auto operand_sharded = sharded_per_mesh_axis(operand_topology, mesh_shape);
        if (!caller_sharded.has_value() || !operand_sharded.has_value()) {
            reason = "a label does not have one placement per mesh axis and is not a collapsed whole-mesh label";
            break;
        }
        for (size_t axis = 0; axis < mesh_shape.dims(); ++axis) {
            if ((*operand_sharded)[axis] && !(*caller_sharded)[axis]) {
                reason = "an operand is sharded along a mesh axis on which the caller's tensor is replicated";
                break;
            }
        }
        if (!reason.empty()) {
            break;
        }
    }
    if (reason.empty()) {
        return caller_topology;
    }

    static std::mutex warned_mutex;
    static std::unordered_set<std::string> warned;
    bool first_time = false;
    {
        std::lock_guard<std::mutex> guard(warned_mutex);
        first_time = warned.insert(std::string(op_name)).second;
    }
    if (first_time) {
        log_warning(
            tt::LogOp,
            "{}: the result takes the union of the operand topologies instead of the caller-owned tensor's own label: "
            "{}. The per-device results differ, so the caller's handle now reads as sharded (logged once per process "
            "for this op)",
            op_name,
            reason);
    }
    return std::nullopt;
}

void validate_no_per_core_allocation(const ttnn::Tensor& tensor, std::string_view operation_name, size_t input_index) {
    // Ask device_local_config().sharding_args, not the MeshBuffer overload of
    // is_per_core_allocation. That overload resolves MeshBuffer::get_reference_buffer(), which
    // TT_THROWs "no local buffer found" when no shard is local -- and this runs before launch()'s
    // inactive-MeshDevice short-circuit, which exists precisely because most MeshDevice calls fail
    // there. A guard whose contract is to be inert must not be able to throw.
    // This is also the expression behind Tensor.is_per_core_allocated(), so the guard and the
    // accessor callers branch on cannot disagree about what per-core means.
    TT_FATAL(
        !tt::tt_metal::experimental::per_core_allocation::is_per_core_allocation(
            tensor.mesh_buffer().device_local_config().sharding_args),
        "{}: tensor {} is per-core allocated, but this operation has not opted in to per-core "
        "allocation. Ops address a buffer by a single L1 address, so a per-core buffer would be read as "
        "though every core shared the first core's allocation (#51354). If the operation resolves "
        "per-core addresses, declare `static constexpr bool supports_per_core_allocation = true` on it; "
        "otherwise build the tensor with a lockstep memory config. Memory config: {}",
        operation_name,
        input_index,
        tensor.memory_config());
}

}  // namespace ttnn::device_operation::detail
