// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/host_ccl.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <ttnn/tensor/storage.hpp>
#include <ttnn/tensor/tensor.hpp>

#include "tt_stl/span.hpp"
#include "ttnn/tensor/tensor_impl.hpp"

namespace ttnn::distributed::host_ccl {

using ::tt::tt_metal::DistributedHostBuffer;
using ::tt::tt_metal::HostBuffer;

Tensor all_gather(const Tensor& tensor) {
    TT_FATAL(tensor.storage_type() == tt::tt_metal::StorageType::HOST, "Tensor must be on host");
    const auto& ctx = tensor.host_storage().buffer().context();
    if (*ctx->size() == 1) {
        // Single-host deployment. Validate this host has all the data.
        for (const auto& coord : tensor.host_storage().buffer().shard_coords()) {
            auto shard = tensor.host_storage().buffer().get_shard(coord);
            TT_FATAL(shard.has_value(), "Shard at coordinate {} is missing", coord);
        }
        return tensor;
    }

    // Destination buffer for all-gather data is fully local on each host.
    auto all_gather_buffer = DistributedHostBuffer::create(tensor.host_storage().buffer().shape());

    // Prepare shard presence data: first element is rank, rest are 1/0 for shard presence
    const int this_rank = *ctx->rank();
    constexpr int kShardPresent = 1;
    constexpr int kShardAbsent = 0;

    // Note the use of `std::set` is required to ensure the ordering of the shard coordinates.
    const std::set<tt::tt_metal::distributed::MeshCoordinate>& shard_coords =
        tensor.host_storage().buffer().shard_coords();
    std::vector<int> local_shard_info;
    local_shard_info.reserve(1 + shard_coords.size());
    local_shard_info.push_back(this_rank);

    for (const auto& coord : shard_coords) {
        auto shard = tensor.host_storage().buffer().get_shard(coord);
        local_shard_info.push_back(shard.has_value() ? kShardPresent : kShardAbsent);
    }

    std::vector<int> global_shard_info(local_shard_info.size() * *ctx->size());
    ctx->all_gather(
        ttsl::as_writable_bytes(ttsl::make_span(local_shard_info)),
        ttsl::as_writable_bytes(ttsl::make_span(global_shard_info)));

    auto find_shard_distribution = [&](size_t shard_index) -> std::pair<std::optional<int>, bool> {
        std::optional<int> lowest_rank_with_shard;
        bool any_host_missing = false;

        for (int host = 0; host < *ctx->size(); ++host) {
            int base_idx = host * local_shard_info.size();
            int rank = global_shard_info[base_idx];
            int has_shard = global_shard_info[base_idx + 1 + shard_index];

            if (has_shard) {
                lowest_rank_with_shard = std::min(lowest_rank_with_shard.value_or(rank), rank);
            } else {
                any_host_missing = true;
            }
        }

        return {lowest_rank_with_shard, any_host_missing};
    };

    size_t shard_idx = 0;
    for (const auto& coord : shard_coords) {
        auto local_shard = tensor.host_storage().buffer().get_shard(coord);

        const auto [lowest_rank_with_shard, any_host_missing_shard] = find_shard_distribution(shard_idx++);
        TT_FATAL(lowest_rank_with_shard.has_value(), "No host has shard at coordinate {}", coord);

        // Only broadcast if at least one host is missing the shard.
        if (any_host_missing_shard) {
            if (local_shard.has_value() && *lowest_rank_with_shard == this_rank) {
                ctx->broadcast(
                    local_shard->view_bytes(), tt::tt_metal::distributed::multihost::Rank(*lowest_rank_with_shard));
                all_gather_buffer.emplace_shard(coord, [&local_shard]() { return *local_shard; });
            } else {
                HostBuffer buffer = tt::tt_metal::tensor_impl::allocate_host_buffer(tensor.tensor_spec());
                ctx->broadcast(
                    buffer.view_bytes(), tt::tt_metal::distributed::multihost::Rank(*lowest_rank_with_shard));
                all_gather_buffer.emplace_shard(coord, [&buffer]() { return std::move(buffer); });
            }
        } else {
            TT_FATAL(local_shard.has_value(), "Expected all hosts to have shard at coordinate {}", coord);
            all_gather_buffer.emplace_shard(coord, [&local_shard]() { return *local_shard; });
        }
    }

    return Tensor(
        tt::tt_metal::HostStorage{std::move(all_gather_buffer)}, tensor.tensor_spec(), tensor.tensor_topology());
}
}  // namespace ttnn::distributed::host_ccl
