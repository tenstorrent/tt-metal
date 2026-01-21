// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/distributed_context.hpp>
#include <ttnn/tensor/tensor.hpp>
#include "tt-metalium/experimental/sockets/mesh_socket.hpp"

namespace ttnn::distributed {

/**
 * @brief Abstract interface for distributed tensor communication over sockets.
 *
 * Provides methods for sending and receiving tensors between distributed nodes,
 * along with access to rank information and distributed context.
 */
class ISocket {
public:
    virtual ~ISocket() = default;

    virtual void send(const ttnn::Tensor& tensor) = 0;
    virtual void recv(ttnn::Tensor& tensor) = 0;

    virtual tt::tt_metal::distributed::multihost::Rank get_rank() const = 0;
    virtual std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> get_distributed_context()
        const = 0;
};

}  // namespace ttnn::distributed
