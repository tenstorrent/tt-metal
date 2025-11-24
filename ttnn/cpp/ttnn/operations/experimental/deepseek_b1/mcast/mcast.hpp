// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "device/mcast_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_b1::mcast {

struct McastOperation {
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, const ttnn::Tensor& output_tensor, uint32_t noc);
};

}  // namespace ttnn::operations::experimental::deepseek_b1::mcast

namespace ttnn::prim {
constexpr auto mcast = ttnn::
    register_operation<"ttnn::prim::mcast", ttnn::operations::experimental::deepseek_b1::mcast::McastDeviceOperation>();
}  // namespace ttnn::prim

namespace ttnn::experimental::deepseek_b1 {

constexpr auto mcast = ttnn::register_operation<
    "ttnn::experimental::deepseek_b1::mcast",
    ttnn::operations::experimental::deepseek_b1::mcast::McastOperation>();

}  // namespace ttnn::experimental::deepseek_b1
