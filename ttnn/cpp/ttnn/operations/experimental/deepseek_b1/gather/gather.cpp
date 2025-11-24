// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gather.hpp"
#include "device/gather_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_b1::gather {

ttnn::Tensor GatherOperation::invoke(
    const ttnn::Tensor& input_tensor, const ttnn::Tensor& output_tensor, std::optional<uint32_t> noc) {
    return ttnn::prim::gather(input_tensor, output_tensor, noc);
}

}  // namespace ttnn::operations::experimental::deepseek_b1::gather
