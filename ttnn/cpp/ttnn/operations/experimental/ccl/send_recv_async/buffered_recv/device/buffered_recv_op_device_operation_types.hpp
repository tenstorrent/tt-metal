// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <vector>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::experimental::prim {

struct BufferedRecvParams {
    const tt::tt_metal::distributed::MeshSocket mesh_socket;  // No default constructor
    const tt::tt_metal::GlobalSemaphore global_semaphore;
    BufferedRecvParams(
        const tt::tt_metal::distributed::MeshSocket& mesh_socket,
        const tt::tt_metal::GlobalSemaphore& global_semaphore) :
        mesh_socket(mesh_socket), global_semaphore(global_semaphore) {}
    // Add attributes method for reflection
    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("mesh_socket", mesh_socket);
        attrs.emplace_back("global_semaphore", global_semaphore);
        return attrs;
    }
};

}  // namespace ttnn::experimental::prim
