// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <initializer_list>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_set>
#include <variant>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>
#include <tt_stl/small_vector.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::core {

// Which mesh axes of `mesh_shape` a label says the data differs along. One placement per axis is read as is; a
// collapsed 1-D label (what the default mappers produce) reads as Shard on every axis of size > 1 for a Shard and
// as Replicate everywhere for a Replicate. Anything else is a label this rule cannot read: nullopt.
inline std::optional<ttsl::SmallVector<bool>> sharded_per_mesh_axis(
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

// Output-topology rule for a device operation that hands back a caller-owned tensor: an in-place variant, or a
// preallocated output that the op overwrites (in whole or in part) from its operands.
//
// The caller's label stays while it still describes the data: no other value-affecting operand may be sharded
// along a mesh axis on which the caller's tensor is replicated. If one is, every device now holds a different
// result along that axis, and a label that still said Replicate would make the serialiser deduplicate shards
// that differ and composers read one device's result as everyone's. The rule then returns nullopt, and the hook
// returns {} so the framework's union of all operand labels is applied instead -- data-preserving: the
// returned handle (and, where it aliases the storage, the caller's own handle) reads as sharded, which is what
// the data is. A label this rule cannot read per axis, or an operand distributed over a different set of mesh
// coordinates than the caller's tensor, also falls back to the union. `operands` may contain nullptr entries
// (absent optionals). The fallback is logged once per process per `op_name`.
inline std::optional<tt::tt_metal::TensorTopology> caller_owned_output_topology(
    const Tensor& caller_tensor, std::initializer_list<const Tensor*> operands, const char* op_name) {
    const auto& caller_topology = caller_tensor.tensor_topology();
    if (caller_tensor.device() == nullptr) {
        return caller_topology;  // not a mesh tensor: nothing to compare against
    }
    const auto& mesh_shape = caller_tensor.device()->shape();
    const auto caller_sharded = sharded_per_mesh_axis(caller_topology, mesh_shape);

    const char* reason = nullptr;
    for (const Tensor* operand : operands) {
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
        if (reason != nullptr) {
            break;
        }
    }
    if (reason == nullptr) {
        return caller_topology;
    }

    static std::mutex warned_mutex;
    static std::unordered_set<std::string> warned;
    bool first_time = false;
    {
        std::lock_guard<std::mutex> guard(warned_mutex);
        first_time = warned.insert(op_name).second;
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

}  // namespace ttnn::operations::core
