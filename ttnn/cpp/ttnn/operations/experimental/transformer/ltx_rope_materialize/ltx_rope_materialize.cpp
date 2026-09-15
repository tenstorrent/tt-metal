// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ltx_rope_materialize.hpp"

#include "device/ltx_rope_materialize_device_operation.hpp"

namespace ttnn::experimental {

std::tuple<Tensor, Tensor, Tensor, Tensor> ltx_rope_materialize(
    const Tensor& compact_self_cos,
    const Tensor& compact_self_sin,
    const Tensor& compact_cross_cos,
    const Tensor& compact_cross_sin,
    const Tensor& metadata,
    const Tensor& self_cos_output,
    const Tensor& self_sin_output,
    const Tensor& cross_cos_output,
    const Tensor& cross_sin_output,
    uint32_t sp_axis,
    uint32_t tp_axis) {
    return ttnn::prim::ltx_rope_materialize(
        compact_self_cos,
        compact_self_sin,
        compact_cross_cos,
        compact_cross_sin,
        metadata,
        self_cos_output,
        self_sin_output,
        cross_cos_output,
        cross_sin_output,
        sp_axis,
        tp_axis);
}

}  // namespace ttnn::experimental
