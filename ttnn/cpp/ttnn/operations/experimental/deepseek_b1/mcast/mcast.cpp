// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "mcast.hpp"
#include "device/mcast_device_operation.hpp"
#include <algorithm>

namespace ttnn::operations::experimental::deepseek_b1::mcast {

ttnn::Tensor McastOperation::invoke(const ttnn::Tensor& input_tensor, const ttnn::Tensor& output_tensor, uint32_t noc) {
    return ttnn::prim::mcast(input_tensor, output_tensor, noc);
}

}  // namespace ttnn::operations::experimental::deepseek_b1::mcast
