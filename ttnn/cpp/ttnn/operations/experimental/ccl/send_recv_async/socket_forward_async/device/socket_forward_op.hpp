// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/mesh_socket.hpp>
#include <tt_stl/reflection.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn {

struct SocketForward {
    const tt::tt_metal::distributed::MeshSocket& recv_socket;
    const tt::tt_metal::distributed::MeshSocket& send_socket;
    std::size_t num_bytes;

    SocketForward(
        const tt::tt_metal::distributed::MeshSocket& recv_socket,
        const tt::tt_metal::distributed::MeshSocket& send_socket,
        std::size_t num_bytes) :
        recv_socket(recv_socket), send_socket(send_socket), num_bytes(num_bytes) {}

    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("sender_socket", send_socket);
        attrs.emplace_back("receiver_socket", recv_socket);
        return attrs;
    }

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor>& input_tensors) const;
    tt::tt_metal::operation::MeshWorkloadWithCallbacks create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::ProgramWithCallbacks create_program_at(
        const ttnn::MeshCoordinate& coord,
        const std::vector<Tensor>& input_tensors,
        std::vector<Tensor>& output_tensors) const;
    tt::tt_metal::operation::Hash compute_program_hash(const std::vector<Tensor>& input_tensors) const;
};

tt::tt_metal::operation::ProgramWithCallbacks socket_forward_minimal_single_core_multi_link(
    const tt::tt_metal::distributed::MeshSocket& send_socket,
    const tt::tt_metal::distributed::MeshSocket& recv_socket,
    std::size_t num_bytes);

namespace operations::experimental::ccl {

std::vector<Tensor> socket_forward(
    const tt::tt_metal::distributed::MeshSocket& recv_socket,
    const tt::tt_metal::distributed::MeshSocket& send_socket,
    std::size_t num_bytes);

}  // namespace operations::experimental::ccl

}  // namespace ttnn
