// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/distributed/isocket.hpp"

namespace ttnn::distributed {

/**
 * @brief MPI-based implementation of distributed tensor communication.
 *
 * Provides point-to-point tensor communication between MPI ranks using the Message
 * Passing Interface. This implementation wraps a MeshSocket to handle the underlying
 * tensor serialization and MPI message passing. Supports both blocking send/recv
 * operations for reliable tensor exchange in distributed training and inference.
 *
 * The socket maintains connection to a specific remote rank and handles tensor
 * metadata (shape, dtype, layout) along with the tensor data during transmission.
 */
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
