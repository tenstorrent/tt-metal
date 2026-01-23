// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/inspector/inspector.hpp"
#include "core/inspector/ttnn_rpc_channel_generated.hpp"
#include "tt-metalium/experimental/inspector.hpp"
#include <mutex>

namespace ttnn::inspector {

void register_inspector_rpc() {
    static rpc::TtnnInspectorRpcChannel ttnn_inspector_rpc_channel;
    static std::once_flag register_flag;

    std::call_once(register_flag, []() {
        tt::tt_metal::experimental::inspector::RegisterInspectorRpcChannel(
            "TtnnInspector",
            tt::tt_metal::inspector::rpc::InspectorChannel::Client(
                ::kj::Own<rpc::TtnnInspectorRpcChannel>(&ttnn_inspector_rpc_channel, ::kj::NullDisposer::instance)));
    });
}

}  // namespace ttnn::inspector
