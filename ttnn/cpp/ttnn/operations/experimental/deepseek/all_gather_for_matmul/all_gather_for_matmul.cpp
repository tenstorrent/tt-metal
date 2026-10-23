// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_for_matmul.hpp"
#include "device/all_gather_for_matmul_device_operation.hpp"

namespace ttnn::experimental::deepseek {

ttnn::Tensor all_gather_for_matmul(
    const ttnn::Tensor& input_tensor,
    const CoreRangeSet& output_core_range_set,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::prim::all_gather_for_matmul(input_tensor, output_core_range_set, optional_output_tensor);
}

}  // namespace ttnn::experimental::deepseek
