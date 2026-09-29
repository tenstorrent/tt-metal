// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <optional>
#include <vector>

#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct SendDirectAsyncParams {
    const tt::tt_metal::distributed::MeshSocket mesh_socket;  // No default constructor
    // When set, the sender skips the address handshake and writes straight to this receiver buffer
    // base address (fire-and-forget; pair with recv_direct_async(wait_only=True)).
    const std::optional<uint32_t> static_dst_address;
    // When set, only the first num_pages pages of the input buffer are sent (e.g. a leading block prefix of a
    // paged KV cache); otherwise the whole buffer.
    const std::optional<uint32_t> num_pages;
    SendDirectAsyncParams(
        const tt::tt_metal::distributed::MeshSocket& mesh_socket,
        std::optional<uint32_t> static_dst_address = std::nullopt,
        std::optional<uint32_t> num_pages = std::nullopt) :
        mesh_socket(mesh_socket), static_dst_address(static_dst_address), num_pages(num_pages) {}
    // Add attributes method for reflection
    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("mesh_socket", mesh_socket);
        attrs.emplace_back("static_dst_address", static_dst_address);
        attrs.emplace_back("num_pages", num_pages);
        return attrs;
    }
};

}  // namespace ttnn::experimental::prim
