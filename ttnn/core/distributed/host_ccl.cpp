// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/host_ccl.hpp"

#include <tt-metalium/assert.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <ttnn/tensor/storage.hpp>
#include <ttnn/tensor/tensor.hpp>

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

    for (const auto& coord : tensor.host_storage().buffer().shard_coords()) {
        auto shard = tensor.host_storage().buffer().get_shard(coord);

        // Run all-reduce to determine which rank has data for this shard.
        int has_data = shard.has_value() ? *ctx->rank() : -1;
        int has_data_rank = -1;

        ctx->all_reduce(
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&has_data), sizeof(int)),
            tt::stl::Span<std::byte>(reinterpret_cast<std::byte*>(&has_data_rank), sizeof(int)),
            tt::tt_metal::distributed::multihost::ReduceOp::MAX,
            tt::tt_metal::distributed::multihost::DType::INT32);

        TT_FATAL(has_data_rank != -1, "Failed to find shard for rank {}", *ctx->rank());

        if (shard.has_value()) {
            ctx->broadcast(shard->view_bytes(), tt::tt_metal::distributed::multihost::Rank(has_data_rank));
            all_gather_buffer.emplace_shard(coord, [&shard]() { return *shard; });
        } else {
            HostBuffer buffer = tt::tt_metal::tensor_impl::allocate_host_buffer(tensor.tensor_spec());
            ctx->broadcast(buffer.view_bytes(), tt::tt_metal::distributed::multihost::Rank(has_data_rank));
            all_gather_buffer.emplace_shard(coord, [&buffer]() { return std::move(buffer); });
        }
    }

    return Tensor(
        tt::tt_metal::HostStorage{std::move(all_gather_buffer)}, tensor.tensor_spec(), tensor.tensor_topology());
}
}  // namespace ttnn::distributed::host_ccl
