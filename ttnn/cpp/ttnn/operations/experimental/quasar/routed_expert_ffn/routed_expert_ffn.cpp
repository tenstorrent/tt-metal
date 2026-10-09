// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_expert_ffn.hpp"

#include "device/routed_expert_ffn_device_operation.hpp"

namespace ttnn::operations::experimental::quasar {

Tensor routed_expert_ffn(
    const Tensor& x,
    const Tensor& w_gate,
    const Tensor& w_up,
    const Tensor& w_down,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::qsr::routed_expert_ffn(x, w_gate, w_up, w_down, memory_config.value_or(x.memory_config()));
}

}  // namespace ttnn::operations::experimental::quasar
