// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt_stl/reflection.hpp>

#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// D2HSocket is non-copyable, so we hold a non-owning pointer. The caller (Python or C++)
// is responsible for keeping the D2HSocket alive for the duration of the op launch /
// device execution.
struct SendAsyncD2HParams {
    const tt::tt_metal::distributed::D2HSocket* d2h_socket = nullptr;

    explicit SendAsyncD2HParams(const tt::tt_metal::distributed::D2HSocket& d2h_socket) : d2h_socket(&d2h_socket) {}

    // Reflection: surface a small set of stable, hashable properties of the D2H socket.
    // These uniquely identify the socket from a program-cache perspective: same device
    // and same config buffer address produces the same program.
    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("d2h_config_buffer_address", d2h_socket->get_config_buffer_address());
        return attrs;
    }
};

}  // namespace ttnn::experimental::prim
