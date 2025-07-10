// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/distributed/isocket.hpp"

namespace ttnn::distributed {

class FabricSocket : public ISocket {
public:
    FabricSocket(const tt::tt_metal::distributed::MeshSocket& mesh_socket);
    ~FabricSocket() override = default;

    void send(const ttnn::Tensor& tensor) override;
    void recv(ttnn::Tensor& tensor) override;

    tt::tt_metal::distributed::multihost::Rank get_sender_rank() const override;
    tt::tt_metal::distributed::multihost::Rank get_receiver_rank() const override;
    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> get_distributed_context() const override;

private:
    tt::tt_metal::distributed::MeshSocket mesh_socket_;
};

}  // namespace ttnn::distributed
