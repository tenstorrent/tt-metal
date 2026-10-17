// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "ttnn/distributed/isocket.hpp"

namespace ttnn::distributed {

namespace detail {

struct MPITensorDescriptor {
    uint32_t dtype{};
    uint32_t layout{};
    std::vector<uint32_t> logical_shape;
    std::vector<uint32_t> padded_shape;
    std::vector<uint64_t> segment_sizes;

    bool operator==(const MPITensorDescriptor&) const = default;
};

std::vector<std::byte> serialize_mpi_tensor_descriptor(const MPITensorDescriptor& descriptor);
bool deserialize_mpi_tensor_descriptor(
    std::span<const std::byte> bytes, MPITensorDescriptor& descriptor, std::string& error);
std::string compare_mpi_tensor_descriptors(const MPITensorDescriptor& expected, const MPITensorDescriptor& received);
void send_mpi_tensor_descriptor(
    const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& context,
    tt::tt_metal::distributed::multihost::Rank peer,
    const MPITensorDescriptor& descriptor);
void receive_and_validate_mpi_tensor_descriptor(
    const std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext>& context,
    tt::tt_metal::distributed::multihost::Rank peer,
    const MPITensorDescriptor& expected);

}  // namespace detail

/**
 * @brief MPI-based implementation of distributed tensor communication.
 *
 * Provides point-to-point tensor communication between MPI ranks using the Message
 * Passing Interface. Supports blocking send/recv operations for reliable tensor
 * exchange in distributed training and inference.
 *
 * The socket maintains connection to a specific remote rank and handles tensor
 * metadata (shape, dtype, layout) along with the tensor data during transmission.
 */
class MPISocket : public ISocket {
public:
    MPISocket(
        std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> distributed_context,
        tt::tt_metal::distributed::multihost::Rank sender_rank,
        tt::tt_metal::distributed::multihost::Rank receiver_rank);
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
    std::shared_ptr<tt::tt_metal::distributed::multihost::DistributedContext> distributed_context_;
    tt::tt_metal::distributed::multihost::Rank sender_rank_;
    tt::tt_metal::distributed::multihost::Rank receiver_rank_;
    std::mutex send_mutex_;
    std::mutex recv_mutex_;
};

}  // namespace ttnn::distributed
