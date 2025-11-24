// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "device/gather_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_b1::gather {

struct GatherOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor, const ttnn::Tensor& output_tensor, std::optional<uint32_t> noc);
};

}  // namespace ttnn::operations::experimental::deepseek_b1::gather

namespace ttnn::prim {
constexpr auto gather = ttnn::register_operation<
    "ttnn::prim::gather",
    ttnn::operations::experimental::deepseek_b1::gather::GatherDeviceOperation>();
}  // namespace ttnn::prim

namespace ttnn::experimental::deepseek_b1 {

constexpr auto gather = ttnn::register_operation<
    "ttnn::experimental::deepseek_b1::gather",
    ttnn::operations::experimental::deepseek_b1::gather::GatherOperation>();

}  // namespace ttnn::experimental::deepseek_b1
