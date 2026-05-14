// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/chunked_gated_delta_device_operation.hpp"
#include "chunked_gated_delta.hpp"

namespace ttnn::experimental {

Tensor chunked_gated_delta(const Tensor& g_exp, const Tensor& factor, const Tensor& bktv, const Tensor& state) {
    return ttnn::prim::chunked_gated_delta(g_exp, factor, bktv, state);
}

}  // namespace ttnn::experimental
