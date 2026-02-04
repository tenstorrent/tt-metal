// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt_stl/reflection.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct RecvAsyncParams {
    const tt::tt_metal::distributed::MeshSocket mesh_socket;  // No default constructor
    RecvAsyncParams(const tt::tt_metal::distributed::MeshSocket& mesh_socket) : mesh_socket(mesh_socket) {}
    // Add attributes method for reflection
    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("mesh_socket", mesh_socket);
        return attrs;
    }
};

}  // namespace ttnn::experimental::prim
