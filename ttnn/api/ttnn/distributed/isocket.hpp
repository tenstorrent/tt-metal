// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/distributed_context.hpp>
#include <ttnn/tensor/tensor.hpp>
#include "tt-metalium/mesh_socket.hpp"

namespace ttnn::distributed {

class ISocket {
public:
    virtual ~ISocket() = default;

    virtual void send(const ttnn::Tensor& tensor) = 0;
    virtual void recv(ttnn::Tensor& tensor) = 0;

    virtual tt::tt_metal::distributed::multihost::Rank get_sender_rank() const = 0;
    virtual tt::tt_metal::distributed::multihost::Rank get_receiver_rank() const = 0;
    virtual std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> get_distributed_context()
        const = 0;
};

}  // namespace ttnn::distributed
