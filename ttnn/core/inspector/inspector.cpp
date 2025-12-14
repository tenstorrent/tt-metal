// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/inspector/inspector.hpp"
#include "core/inspector/ttnn_rpc_channel_generated.hpp"
#include "tt-metalium/inspector.hpp"

namespace ttnn {
namespace inspector {

void register_inspector_rpc() {
    static rpc::TtnnInspectorRpcChannel ttnn_inspector_rpc_channel;

    tt::tt_metal::RegisterInspectorRpcChannel(
        "TtnnInspector",
        tt::tt_metal::inspector::rpc::InspectorChannel::Client(
            ::kj::Own<rpc::TtnnInspectorRpcChannel>(&ttnn_inspector_rpc_channel, ::kj::NullDisposer::instance)));
}

}  // namespace inspector
}  // namespace ttnn
