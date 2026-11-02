// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>

#include "ttnn/device_operation_detail.hpp"

#include <algorithm>
#include <functional>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>
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

static bool is_fully_replicated(const tt::tt_metal::TensorTopology& topology) {
    for (const auto& placement : topology.placements()) {
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
    TT_FATAL(!tensors.empty(), "Cannot compute output placements and shape with no tensors");

    std::vector<std::reference_wrapper<const tt::tt_metal::TensorTopology>> topologies;
    topologies.reserve(tensors.size());
    for (const auto& tensor_ref : tensors) {
        topologies.emplace_back(tensor_ref.get().tensor_topology());
    }
    return compute_output_placements_and_shape(topologies);
}

std::pair<
    ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>,
    tt::tt_metal::distributed::MeshShape>
compute_output_placements_and_shape(
    const std::vector<std::reference_wrapper<const tt::tt_metal::TensorTopology>>& topologies) {
    using TensorTopology = tt::tt_metal::TensorTopology;
    using Placement = tt::tt_metal::distributed::MeshMapperConfig::Placement;
    using Shard = tt::tt_metal::distributed::MeshMapperConfig::Shard;
    using Replicate = tt::tt_metal::distributed::MeshMapperConfig::Replicate;

    TT_FATAL(!topologies.empty(), "Cannot compute output placements and shape with no tensor topologies");

    std::vector<std::reference_wrapper<const TensorTopology>> sharded_topologies;
    sharded_topologies.reserve(topologies.size());
    for (const auto& topology_ref : topologies) {
        if (!is_fully_replicated(topology_ref.get())) {
            sharded_topologies.push_back(topology_ref);
        }
    }

    // Compute max distribution rank: use only sharded tensors if they exist, otherwise use all tensors (fully
    // replicated)
    size_t max_distribution_rank = 0;
    if (!sharded_topologies.empty()) {
        for (const auto& topology_ref : sharded_topologies) {
            max_distribution_rank = std::max(max_distribution_rank, topology_ref.get().distribution_shape().dims());
        }
    } else {
        const auto& first_topology = topologies.front().get();
        max_distribution_rank = first_topology.distribution_shape().dims();
    }

    auto result_strides = ttsl::SmallVector<uint32_t>(max_distribution_rank, 1);
    auto result_placements = ttsl::SmallVector<Placement>(max_distribution_rank, Replicate{});
    std::unordered_map<int, int> shard_dim_to_distribution_dim;
    bool dim_mismatch = false;

    // TODO: #25340 - Add back logging / validation. Currently, this results in a lot of log spam.
    constexpr bool kEnableLogging = false;
    for (const auto& topology_ref : topologies) {
        const TensorTopology& topology = topology_ref.get();
        // Augment output tensor distribution shape with the max strides of all input tensors with the max
        // distribution rank
        const auto& tensor_distribution_shape = topology.distribution_shape();
        if (tensor_distribution_shape.dims() == max_distribution_rank) {
            for (size_t i = 0; i < std::min(result_strides.size(), tensor_distribution_shape.dims()); i++) {
                result_strides[i] = std::max(result_strides[i], tensor_distribution_shape[i]);
            }

            const auto& tensor_placements = topology.placements();
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
        } else if (!is_fully_replicated(topology)) {
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

namespace {

// A tensor as the caller-owned rule reads it: its label, and that label per mesh axis (sharded_per_mesh_axis).
struct RuleOperand {
    const tt::tt_metal::TensorTopology* topology = nullptr;
    std::optional<ttsl::SmallVector<bool>> sharded;
};

// Why the caller's label stops describing the data next to the operands', or an empty string while it still does.
// With `same_mesh_coords` every label is already known to span the same mesh coordinates.
std::string rejection_reason(
    const RuleOperand& caller, std::span<const RuleOperand* const> operands, size_t mesh_dims, bool same_mesh_coords) {
    for (const RuleOperand* operand : operands) {
        if (!same_mesh_coords && operand->topology->mesh_coords() != caller.topology->mesh_coords()) {
            return "an operand is distributed over a different set of mesh coordinates";
        }
        if (!caller.sharded.has_value() || !operand->sharded.has_value()) {
            return "a label does not have one placement per mesh axis and is not a collapsed whole-mesh label";
        }
        for (size_t axis = 0; axis < mesh_dims; ++axis) {
            if ((*operand->sharded)[axis] && !(*caller.sharded)[axis]) {
                return fmt::format(
                    "an operand is sharded along mesh axis {}, on which the caller's tensor is replicated", axis);
            }
        }
    }
    return {};
}

void warn_once(std::string_view op_name, std::string_view reason) {
    static std::mutex warned_mutex;
    static std::set<std::string, std::less<>> warned;
    {
        std::lock_guard<std::mutex> guard(warned_mutex);
        if (warned.contains(op_name)) {
            return;
        }
        warned.emplace(op_name);
    }
    log_warning(
        tt::LogOp,
        "{}: the caller's tensor takes the union of the operand labels instead of its own: {} (logged once per "
        "process for this op)",
        op_name,
        reason);
}

// The device buffer `tensor` lives in, or nullptr for a host or deallocated tensor.
const void* device_buffer(const ttnn::Tensor& tensor) {
    if (tensor.storage_type() != ttnn::StorageType::DEVICE || !tensor.is_allocated()) {
        return nullptr;
    }
    return &tensor.mesh_buffer();
}

}  // namespace

std::optional<tt::tt_metal::TensorTopology> caller_owned_output_topology(
    const ttnn::Tensor& caller_tensor, std::span<const ttnn::Tensor* const> operands, std::string_view op_name) {
    const auto& caller_topology = caller_tensor.tensor_topology();
    if (caller_tensor.device() == nullptr) {
        return caller_topology;  // not a mesh tensor: nothing to compare against
    }
    const auto& mesh_shape = caller_tensor.device()->shape();
    ttsl::SmallVector<RuleOperand> read;
    for (const ttnn::Tensor* operand : operands) {
        if (operand != nullptr && operand != &caller_tensor) {
            const auto& topology = operand->tensor_topology();
            read.push_back({&topology, sharded_per_mesh_axis(topology, mesh_shape)});
        }
    }
    ttsl::SmallVector<const RuleOperand*> read_operands;
    for (const auto& operand : read) {
        read_operands.push_back(&operand);
    }
    const RuleOperand caller{&caller_topology, sharded_per_mesh_axis(caller_topology, mesh_shape)};
    const std::string reason = rejection_reason(caller, read_operands, mesh_shape.dims(), /*same_mesh_coords=*/false);
    if (reason.empty()) {
        return caller_topology;
    }
    warn_once(op_name, reason);
    return std::nullopt;
}

void apply_default_output_topologies(
    const std::vector<std::reference_wrapper<ttnn::Tensor>>& outputs,
    const std::vector<std::reference_wrapper<const ttnn::Tensor>>& inputs,
    std::string_view op_name) {
    // Each input is read once, and every label is decided from the labels the tensors were launched with before any
    // is applied: an in-place output is also an input the other outputs are checked against.
    ttsl::SmallVector<const void*> buffers;
    for (const auto& input : inputs) {
        buffers.push_back(device_buffer(input.get()));
    }
    ttsl::SmallVector<RuleOperand> read;  // the inputs as the rule reads them, filled at the first caller-owned output
    ttsl::SmallVector<const RuleOperand*> operands;
    size_t mesh_dims = 0;
    bool same_mesh_coords = true;
    ttsl::SmallVector<const ttnn::Tensor*> kept(outputs.size(), nullptr);  // the input whose label an output keeps
    bool needs_union = false;
    for (size_t o = 0; o < outputs.size(); ++o) {
        const void* buffer = device_buffer(outputs[o].get());
        const auto caller = buffer == nullptr ? buffers.end() : std::find(buffers.begin(), buffers.end(), buffer);
        if (caller == buffers.end()) {
            needs_union = true;
            continue;
        }
        const auto caller_index = static_cast<size_t>(caller - buffers.begin());
        if (read.empty()) {
            const auto& mesh_shape = inputs[caller_index].get().device()->shape();
            mesh_dims = mesh_shape.dims();
            for (const auto& input : inputs) {
                const auto& topology = input.get().tensor_topology();
                read.push_back({&topology, sharded_per_mesh_axis(topology, mesh_shape)});
            }
            for (size_t i = 1; i < read.size() && same_mesh_coords; ++i) {
                same_mesh_coords = read[i].topology->mesh_coords() == read[0].topology->mesh_coords();
            }
        }
        operands.clear();
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (buffers[i] != buffer) {
                operands.push_back(&read[i]);
            }
        }
        const std::string reason = rejection_reason(read[caller_index], operands, mesh_dims, same_mesh_coords);
        if (reason.empty()) {
            kept[o] = &inputs[caller_index].get();
        } else {
            warn_once(op_name, reason);
            needs_union = true;
        }
    }

    std::optional<std::pair<
        ttsl::SmallVector<tt::tt_metal::distributed::MeshMapperConfig::Placement>,
        tt::tt_metal::distributed::MeshShape>>
        union_label;
    if (needs_union) {
        union_label = compute_output_placements_and_shape(inputs);
    }
    for (size_t o = 0; o < outputs.size(); ++o) {
        ttnn::Tensor& output = outputs[o].get();
        if (kept[o] == nullptr) {
            output.update_tensor_topology(tt::tt_metal::TensorTopology(
                union_label->second, union_label->first, output.tensor_topology().mesh_coords()));
        } else if (&output.tensor_topology() != &kept[o]->tensor_topology()) {  // a handle that shares no label with it
            output.update_tensor_topology(kept[o]->tensor_topology());
        }
    }
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
