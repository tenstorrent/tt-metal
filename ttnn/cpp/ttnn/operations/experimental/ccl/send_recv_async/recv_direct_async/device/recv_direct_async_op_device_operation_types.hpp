// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <vector>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct RecvDirectAsyncParams {
    const tt::tt_metal::distributed::MeshSocket mesh_socket;  // No default constructor
    // When true, no address is written back to the sender: only wait for the completion page
    // (pair with send_direct_async(static_dst_address=...)).
    const bool wait_only;
    RecvDirectAsyncParams(const tt::tt_metal::distributed::MeshSocket& mesh_socket, bool wait_only = false) :
        mesh_socket(mesh_socket), wait_only(wait_only) {}
    // Add attributes method for reflection
    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("mesh_socket", mesh_socket);
        attrs.emplace_back("wait_only", wait_only);
        return attrs;
    }
};

}  // namespace ttnn::experimental::prim
