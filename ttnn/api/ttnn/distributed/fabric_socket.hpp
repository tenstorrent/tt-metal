// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/distributed/isocket.hpp"

namespace ttnn::distributed {

/**
 * @brief Fabric-based implementation of distributed tensor communication.
 *
 * Provides high-performance point-to-point tensor communication using Tenstorrent's
 * fabric interconnect technology. This implementation leverages the underlying fabric
 * hardware for direct chip-to-chip communication, offering lower latency and higher
 * bandwidth compared to traditional network protocols.
 *
 * The FabricSocket wraps a MeshSocket to handle tensor serialization and fabric
 * message routing. It supports efficient tensor transfer between specific sender
 * and receiver ranks in a fabric-connected topology, making it ideal for
 * high-throughput distributed training and inference workloads.
 */
class FabricSocket : public ISocket {
public:
    FabricSocket(const tt::tt_metal::distributed::MeshSocket& mesh_socket);
    ~FabricSocket() override = default;

    void send(const ttnn::Tensor& tensor) override;
    void recv(ttnn::Tensor& tensor) override;

    tt::tt_metal::distributed::multihost::Rank get_rank() const override;
    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> get_distributed_context() const override;

    static std::unique_ptr<FabricSocket> create(
        const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
        tt::tt_metal::distributed::multihost::Rank sender_rank,
        tt::tt_metal::distributed::multihost::Rank receiver_rank,
        tt::tt_metal::distributed::SocketConfig socket_config);

private:
    tt::tt_metal::distributed::MeshSocket mesh_socket_;
};

}  // namespace ttnn::distributed
