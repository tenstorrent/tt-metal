// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "persistent_moe_request.hpp"

#include "device/persistent_moe_request_device_operation.hpp"

namespace ttnn::experimental::deepseek::moe {

Tensor persistent_moe_request(
    const Tensor& input_tensor,
    uint32_t layer,
    const tt::tt_metal::distributed::MeshSocket& request_socket,
    const Tensor& partials,
    const std::optional<Tensor>& streams,
    const std::optional<Tensor>& post,
    const std::optional<Tensor>& comb) {
    return ttnn::prim::persistent_moe_request(input_tensor, layer, request_socket, partials, streams, post, comb);
}

}  // namespace ttnn::experimental::deepseek::moe
