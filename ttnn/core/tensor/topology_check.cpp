// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/topology_check.hpp"

#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>
#include <tt-metalium/host_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/distributed/tensor_topology.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/storage.hpp"

#include <cstddef>
#include <cstring>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace ttnn {
namespace {

std::string describe(const tt::tt_metal::TensorTopology& topology) {
    std::ostringstream os;
    os << topology;
    return os.str();
}

}  // namespace

void verify_shards_match_topology(const Tensor& tensor, bool compare_replica_contents) {
    TT_FATAL(
        tensor.storage_type() == tt::tt_metal::StorageType::HOST,
        "verify_shards_match_topology checks host shards; move the tensor to the host with cpu() first");
    const auto& distributed_buffer = tensor.host_storage().buffer();
    const auto& topology = tensor.tensor_topology();
    const auto& placements = topology.placements();
    const auto& mesh_shape = topology.distribution_shape();
    const auto& mesh_coords = topology.mesh_coords();
    TT_FATAL(
        mesh_coords.size() == mesh_shape.mesh_size(),
        "Topology mesh coords size {} should match distribution shape size {}",
        mesh_coords.size(),
        mesh_shape.mesh_size());

    // Every populated local shard has to be reachable through the label, or whatever keeps one copy per replica group
    // leaves it out. Remote coordinates (multi-host) hold no data on this host and are exempt.
    const std::set<tt::tt_metal::distributed::MeshCoordinate> labelled_coords(mesh_coords.begin(), mesh_coords.end());
    size_t num_populated_local_shards = 0;
    for (const auto& shard_coord : distributed_buffer.shard_coords()) {
        if (!distributed_buffer.is_local(shard_coord) || !distributed_buffer.get_shard(shard_coord).has_value()) {
            continue;
        }
        ++num_populated_local_shards;
        TT_FATAL(
            labelled_coords.contains(shard_coord),
            "Host storage holds a shard at mesh coordinate {} that the tensor topology does not cover ({}; host "
            "storage shape {}). The topology label does not describe the data, so this shard would be left out. "
            "Relabel the tensor with update_tensor_topology() so the label matches how the shards were produced.",
            shard_coord,
            describe(topology),
            distributed_buffer.shape());
    }

    // Replica groups: coordinates that differ only along Replicate dimensions, keyed by the coordinates at the
    // sharded dimensions. The first member met stands for its group; every later member must match it.
    size_t unique_keys = 1;
    for (size_t dim = 0; dim < placements.size(); ++dim) {
        if (std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(placements[dim])) {
            unique_keys *= mesh_shape[dim];
        }
    }
    struct Representative {
        tt::tt_metal::distributed::MeshCoordinate coord;
        const std::byte* data = nullptr;
        size_t size = 0;
    };
    std::vector<std::optional<Representative>> groups(unique_keys);
    size_t dist_idx = 0;
    for (const auto& dist_coord : tt::tt_metal::distributed::MeshCoordinateRange(mesh_shape)) {
        const auto& coord = mesh_coords[dist_idx++];
        const auto buffer = distributed_buffer.get_shard(coord);
        if (!buffer.has_value()) {
            // A labelled coordinate without a shard is only legitimate when the shard lives on another host.
            TT_FATAL(
                !distributed_buffer.is_local(coord),
                "Tensor topology lists mesh coordinate {} but the host storage has no shard there ({}; host storage "
                "shape {} with {} populated shard(s)). The topology label describes data that does not exist, "
                "typically because the tensor is a single shard taken out of a distributed tensor "
                "(get_device_tensors(...)[i].cpu()), or this is a LOCAL-mode partial file re-dumped on a single host "
                "(re-dump from the multi-host job instead). Relabel the tensor with update_tensor_topology() so the "
                "label matches how the shards were produced.",
                coord,
                describe(topology),
                distributed_buffer.shape(),
                num_populated_local_shards);
            continue;
        }
        const auto bytes = buffer->view_bytes();

        size_t key = 0;
        for (size_t dim = 0; dim < placements.size(); ++dim) {
            if (std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Shard>(placements[dim])) {
                key = key * mesh_shape[dim] + dist_coord[dim];
            }
        }
        auto& group = groups[key];
        if (!group.has_value()) {
            group = Representative{.coord = coord, .data = bytes.data(), .size = bytes.size()};
            continue;
        }
        TT_FATAL(
            group->size == bytes.size(),
            "Tensor topology labels mesh coordinates {} and {} as replicas of each other ({}; host storage shape {}), "
            "but their shard sizes differ ({} vs {} bytes). The topology label does not describe the data. Relabel the "
            "tensor with update_tensor_topology() so the label matches how the shards were produced.",
            group->coord,
            coord,
            describe(topology),
            distributed_buffer.shape(),
            group->size,
            bytes.size());
        // Two views of one allocation hold the same bytes by construction (the fully replicated mapper path aliases
        // one buffer); only distinct buffers are compared.
        if (compare_replica_contents && group->data != bytes.data()) {
            TT_FATAL(
                std::memcmp(group->data, bytes.data(), bytes.size()) == 0,
                "Tensor topology labels mesh coordinates {} and {} as replicas of each other ({}; host storage shape "
                "{}), but their shard contents differ. The topology label does not describe the data, and keeping one "
                "copy for both would lose the shard at {}. Relabel the tensor with update_tensor_topology() so the "
                "label matches how the shards were produced.",
                group->coord,
                coord,
                describe(topology),
                distributed_buffer.shape(),
                coord);
        }
    }
}

}  // namespace ttnn
