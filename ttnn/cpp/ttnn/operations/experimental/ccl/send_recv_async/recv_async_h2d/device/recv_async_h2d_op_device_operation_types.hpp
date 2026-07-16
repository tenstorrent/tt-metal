// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// H2DSocket is non-copyable, so we hold a non-owning pointer. The caller (Python or C++)
// is responsible for keeping the H2DSocket alive for the duration of the op launch /
// device execution.
struct RecvAsyncH2DParams {
    const tt::tt_metal::distributed::H2DSocket* h2d_socket = nullptr;

    explicit RecvAsyncH2DParams(const tt::tt_metal::distributed::H2DSocket& h2d_socket) : h2d_socket(&h2d_socket) {}

    static constexpr auto attribute_names = std::forward_as_tuple("h2d_config_buffer_address", "h2d_mode");
    auto attribute_values() const {
        TT_FATAL(h2d_socket != nullptr, "recv_async_h2d: H2DSocket pointer is null");
        return std::forward_as_tuple(
            h2d_socket->get_config_buffer_address(), static_cast<uint8_t>(h2d_socket->get_h2d_mode()));
    }

    // Reflection: surface a small set of stable, hashable properties of the H2D socket.
    // These uniquely identify the socket from a program-cache perspective: same device,
    // same config buffer address and same mode produces the same program.
    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        TT_FATAL(h2d_socket != nullptr, "recv_async_h2d: H2DSocket pointer is null");
        attrs.emplace_back("h2d_config_buffer_address", h2d_socket->get_config_buffer_address());
        attrs.emplace_back("h2d_mode", static_cast<uint8_t>(h2d_socket->get_h2d_mode()));
        return attrs;
    }
};

}  // namespace ttnn::experimental::prim
