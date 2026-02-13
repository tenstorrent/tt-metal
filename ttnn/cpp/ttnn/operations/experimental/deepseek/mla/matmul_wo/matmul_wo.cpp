// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_wo.hpp"
#include "device/matmul_wo_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::mla {

ttnn::Tensor ExecuteMatmulWO::invoke(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& w_tensor,
    const ttnn::Tensor& output_tensor,
    const uint32_t layer_id) {
    return ttnn::prim::matmul_wo(input_tensor, w_tensor, output_tensor, layer_id);
}

}  // namespace ttnn::operations::experimental::deepseek::mla
