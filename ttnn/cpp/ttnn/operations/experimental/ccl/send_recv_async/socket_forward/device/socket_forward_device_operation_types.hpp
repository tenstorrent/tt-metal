// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt_stl/reflection.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::ccl::socket_forward {

struct operation_attributes_t {
    const tt::tt_metal::distributed::MeshSocket recv_socket;
    const tt::tt_metal::distributed::MeshSocket send_socket;
    const std::size_t num_bytes;
    operation_attributes_t(
        const tt::tt_metal::distributed::MeshSocket& recv_socket,
        const tt::tt_metal::distributed::MeshSocket& send_socket,
        const std::size_t num_bytes) :
        recv_socket(recv_socket), send_socket(send_socket), num_bytes(num_bytes) {}
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("recv_socket", recv_socket);
        attrs.emplace_back("send_socket", send_socket);
        attrs.emplace_back("num_bytes", num_bytes);
        return attrs;
    }
};

struct tensor_args_t {
    const Tensor tensor;
};

using spec_return_value_t = std::vector<ttnn::TensorSpec>;

using tensor_return_value_t = std::vector<Tensor>;

}  // namespace ttnn::operations::experimental::ccl::socket_forward
