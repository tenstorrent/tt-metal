// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device_operation_detail.hpp"

#include <algorithm>
#include <functional>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/inspector.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/small_vector.hpp>

#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::device_operation::detail {

// Bring mesh coordinate types into scope for readability.
using MeshCoordinate = tt::tt_metal::distributed::MeshCoordinate;
using MeshCoordinateRange = tt::tt_metal::distributed::MeshCoordinateRange;
using MeshCoordinateRangeSet = tt::tt_metal::distributed::MeshCoordinateRangeSet;

// ─────────────────────────────────────────────────────────────────────
// compute_output_placements_and_shape
// ─────────────────────────────────────────────────────────────────────
//
// Factored from the former template get_output_placements_and_shape<device_operation_t>.
// The only reason it was templated was to call visit_object_of_type<Tensor> on tensor_args.
// Callers now extract tensors first and pass them as a vector.

static bool is_fully_replicated(const tt::tt_metal::Tensor& tensor) {
    for (const auto& placement : tensor.tensor_topology().placements()) {
        if (std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(placement)) {
            return false;
        }
    }
    return true;
}

std::pair<
    tt::stl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>,
    tt::tt_metal::distributed::MeshShape>
compute_output_placements_and_shape(
    const std::vector<std::reference_wrapper<const tt::tt_metal::Tensor>>& tensors,
    const tt::tt_metal::Tensor& first_tensor) {
    using Tensor = tt::tt_metal::Tensor;
    using Placement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
    using Replicate = tt::tt_metal::distributed::MeshMapperConfig::Replicate;

    std::vector<std::reference_wrapper<const Tensor>> sharded_tensors;
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
        max_distribution_rank = first_tensor.tensor_topology().distribution_shape().dims();
    }

    auto result_strides = tt::stl::SmallVector<uint32_t>(max_distribution_rank, 1);
    auto result_placements = tt::stl::SmallVector<Placement>(max_distribution_rank, Replicate{});
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
    return {result_placements, tt::tt_metal::distributed::MeshShape(result_strides)};
}

// ─────────────────────────────────────────────────────────────────────
// emit_mesh_workload_annotation_impl
// ─────────────────────────────────────────────────────────────────────

void emit_mesh_workload_annotation_impl(
    tt::tt_metal::distributed::MeshWorkload& workload,
    std::string_view operation_name,
    const std::vector<std::reference_wrapper<const tt::tt_metal::Tensor>>& tensors) {
    if (tt::tt_metal::experimental::inspector::IsEnabled()) {
        constexpr size_t TENSOR_ARGS_BUFFER_SIZE = 4096;
        fmt::memory_buffer tensor_args_buffer;
        tensor_args_buffer.reserve(TENSOR_ARGS_BUFFER_SIZE);

        int index = 0;
        for (const auto& tensor_ref : tensors) {
            if (index > 0) {
                fmt::format_to(std::back_inserter(tensor_args_buffer), ", ");
            }
            fmt::format_to(std::back_inserter(tensor_args_buffer), "[{}]: {}", index, tensor_ref.get());
            index++;
        }

        tt::tt_metal::experimental::inspector::EmitMeshWorkloadAnnotation(
            workload, operation_name, std::string_view(tensor_args_buffer.data(), tensor_args_buffer.size()));
    }
}

// ─────────────────────────────────────────────────────────────────────
// extract_tensor_coordinates_impl
// ─────────────────────────────────────────────────────────────────────

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
    const std::vector<std::reference_wrapper<const tt::tt_metal::Tensor>>& tensors,
    tt::tt_metal::distributed::MeshDevice* mesh_device) {
    using Tensor = tt::tt_metal::Tensor;

    // If no tensor is found, return zero coordinate
    if (tensors.empty()) {
        if (mesh_device == nullptr) {
            TT_THROW("No tensors found in tensor_args and no mesh_device provided to extract_tensor_coordinates");
        }
        return {MeshCoordinate::zero_coordinate(mesh_device->shape().dims())};
    }

    const Tensor& first_tensor = tensors.front().get();
    std::vector<ttnn::MeshCoordinate> tensor_coordinates;
    std::transform(
        first_tensor.device_storage().coords.begin(),
        first_tensor.device_storage().coords.end(),
        std::back_inserter(tensor_coordinates),
        [](const auto& coord) { return coord; });

    // Verification Step: Assert if the tensors are placed on different coordinate ranges
    // that do not overlap.
    for (const auto& tensor_ref : tensors) {
        const Tensor& tensor = tensor_ref.get();
        if (tensor.device_storage().coords.size() != tensor_coordinates.size()) {
            std::vector<ttnn::MeshCoordinate> tensor_mesh_coords;
            std::transform(
                tensor.device_storage().coords.begin(),
                tensor.device_storage().coords.end(),
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

}  // namespace ttnn::device_operation::detail
