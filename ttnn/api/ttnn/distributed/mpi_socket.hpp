// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/distributed/isocket.hpp"

namespace ttnn::distributed {

class MPISocket : public ISocket {
public:
    MPISocket(const tt::tt_metal::distributed::MeshSocket& mesh_socket);
    ~MPISocket() override = default;

    void send(const ttnn::Tensor& tensor) override;
    void recv(ttnn::Tensor& tensor) override;

    tt::tt_metal::distributed::multihost::Rank get_rank() const override;
    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> get_distributed_context() const override;

    static std::unique_ptr<MPISocket> create(
        const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh_device,
        tt::tt_metal::distributed::multihost::Rank rank,
        tt::tt_metal::distributed::SocketConfig socket_config);

private:
    tt::tt_metal::distributed::MeshSocket mesh_socket_;
};

}  // namespace ttnn::distributed
