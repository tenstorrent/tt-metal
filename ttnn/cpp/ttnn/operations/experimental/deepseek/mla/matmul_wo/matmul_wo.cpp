// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_wo.hpp"
#include "device/matmul_wo_device_operation.hpp"
#include "ttnn/graph/composite_trace.hpp"

namespace ttnn::experimental::deepseek::mla {

ttnn::Tensor matmul_wo(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& w_tensor,
    const ttnn::Tensor& output_tensor,
    uint32_t layer_id) {
    ttnn::graph::ScopedCompositeTrace _trace("ttnn::experimental::deepseek::mla::matmul_wo");
    return ttnn::prim::matmul_wo(input_tensor, w_tensor, output_tensor, layer_id);
}

}  // namespace ttnn::experimental::deepseek::mla
