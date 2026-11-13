// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compress_pool_decode.hpp"

#include "device/compress_pool_decode_device_operation.hpp"

namespace ttnn::experimental::deepseek {

std::vector<Tensor> compress_pool_decode(
    const Tensor& kv,
    const Tensor& norm_weight,
    float epsilon,
    const std::optional<Tensor>& gate,
    const std::optional<Tensor>& win_kv,
    const std::optional<Tensor>& win_gate,
    const std::optional<Tensor>& win_slot,
    const std::optional<Tensor>& prev_slot,
    const std::optional<Tensor>& cos,
    const std::optional<Tensor>& sin,
    bool do_pool) {
    return ttnn::prim::compress_pool_decode(
        kv, norm_weight, epsilon, gate, win_kv, win_gate, win_slot, prev_slot, cos, sin, do_pool);
}

}  // namespace ttnn::experimental::deepseek
